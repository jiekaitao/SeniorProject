#!/bin/bash
#SBATCH --job-name=best_cfg
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_best_configs_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

# Instruct + L12 on HellaSwag and ARC
echo "=== Instruct L12 on HellaSwag ==="
for SEED in 42 7; do
    $PYTHON -c "
import sys, os
os.environ['HF_HOME']='/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, 'solver')
import torch, random, time, json, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from eval_deliberation_creative import MidLayerDeliberation, RESULTS_DIR

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B-Instruct')
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained('models/full/Llama-3.1-8B-Instruct', dtype=torch.bfloat16).to(device)
for p in base_model.parameters(): p.requires_grad = False

ds = load_dataset('Rowan/hellaswag')
train_data = [s for s in ds['train'] if len(s['endings']) == 4]
val_data = [s for s in ds['validation'] if len(s['endings']) == 4]
random.seed(0)
random.shuffle(train_data)

def fmt(s):
    p = f\"{s['ctx']}\n\nWhich ending is most likely?\n\"
    for l, e in zip(['A','B','C','D'], s['endings']):
        p += f\"{l}. {e}\n\"
    return p

ids = []
for c in ['A','B','C','D']:
    toks = tokenizer.encode(f' {c}', add_special_tokens=False)
    ids.append(toks[0])
choice_ids_t = torch.tensor(ids, device=device)
lm_model = base_model.model

seed = $SEED
total_steps = 2000
tag = f'instructL12_hellaswag_seed{seed}'
random.seed(seed)
torch.manual_seed(seed)
print(f'\n=== {tag} ===', flush=True)

# Baseline
n_test = min(len(val_data), 500)
test_subset = val_data[:n_test]
correct_base = 0
for sample in test_subset:
    prompt = fmt(sample) + 'Answer:'
    label = int(sample['label'])
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = base_model(enc['input_ids'])
        logits = out.logits[:, -1, ids]
        if logits.argmax(dim=-1).item() == label:
            correct_base += 1
base_acc = correct_base / n_test
print(f'  Baseline: {base_acc:.4f}', flush=True)

controller = MidLayerDeliberation(
    frozen_llm=base_model, inject_layer=12, rank=64,
    d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
).to(device=device, dtype=torch.bfloat16)

optimizer = torch.optim.AdamW([p for p in controller.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
warmup = 200
def lr_sched(s):
    if s < warmup: return s / warmup
    return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

t0 = time.time()
losses = []
optimizer.zero_grad(set_to_none=True)
for step in range(total_steps):
    sample = train_data[step % len(train_data)]
    prompt_text = fmt(sample)
    label = int(sample['label'])
    prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=480).to(device)
    answer_enc = tokenizer('\nAnswer:', return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
    label_t = torch.tensor([label], device=device, dtype=torch.long)
    all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t, rounds=3)
    loss, lp = controller.compute_loss(all_cl, all_v, label_t)
    loss = loss / 8
    loss.backward()
    if (step+1) % 8 == 0:
        torch.nn.utils.clip_grad_norm_([p for p in controller.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    losses.append(lp['final_ce'])
    if (step+1) % 500 == 0:
        print(f'  step {step+1} | ce={sum(losses[-500:])/500:.4f} | {time.time()-t0:.0f}s', flush=True)

controller.eval()
correct = 0
for sample in test_subset:
    prompt_text = fmt(sample)
    label = int(sample['label'])
    prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=480).to(device)
    answer_enc = tokenizer('\nAnswer:', return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
        all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t, rounds=3)
        if all_cl[-1].argmax(dim=-1).item() == label: correct += 1
acc = correct / n_test
print(f'  {tag}: {acc:.4f} vs base={base_acc:.4f} delta={acc-base_acc:+.3f}', flush=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
    json.dump({'tag': tag, 'model': 'Instruct', 'task': 'hellaswag', 'inject_layer': 12, 'seed': seed, 'baseline': base_acc, 'accuracy': acc, 'delta': acc-base_acc}, f, indent=2)

del controller, optimizer
torch.cuda.empty_cache()
"
done
