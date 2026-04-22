#!/bin/bash
#SBATCH --job-name=inst_ml
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_instruct_midlayer_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Mid-layer sweep on Llama Instruct ==="
for L in 8 12 14 16; do
    for SEED in 42; do
        $PYTHON -c "
import sys, os
os.environ['HF_HOME']='/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, 'solver')
import torch, random, time, json, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from eval_deliberation_creative import MidLayerDeliberation, get_choice_token_ids, RESULTS_DIR, CHOICE_MAP

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B-Instruct')
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained('models/full/Llama-3.1-8B-Instruct', dtype=torch.bfloat16).to(device)
for p in base_model.parameters(): p.requires_grad = False

ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
data = [s for s in ds if s['id'].startswith('mazenav')]
random.seed(0)
indices = list(range(len(data)))
random.shuffle(indices)
train_idx, eval_idx = indices[:1000], indices[1000:]
choice_ids = get_choice_token_ids(tokenizer)
choice_ids_t = torch.tensor(choice_ids, device=device)
lm_model = base_model.model

L = $L
seed = $SEED
tag = f'instruct_midL{L}_seed{seed}'
random.seed(seed)
torch.manual_seed(seed)
print(f'\n=== {tag} ===', flush=True)

controller = MidLayerDeliberation(
    frozen_llm=base_model, inject_layer=L, rank=64,
    d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
).to(device=device, dtype=torch.bfloat16)

optimizer = torch.optim.AdamW([p for p in controller.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
warmup = 200
total_steps = 3000
def lr_sched(s):
    if s < warmup: return s / warmup
    return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

t0 = time.time()
losses = []
optimizer.zero_grad(set_to_none=True)

for step in range(total_steps):
    sample = data[train_idx[step % len(train_idx)]]
    text = sample['text'][:1500]
    oracle = sample['oracle_option'].strip().upper()
    answer_label = CHOICE_MAP.get(oracle[0], 0)
    prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
    answer_enc = tokenizer('\nAnswer:', return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
    label_t = torch.tensor([answer_label], device=device, dtype=torch.long)
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
for idx in eval_idx:
    sample = data[idx]
    text = sample['text'][:1500]
    oracle = sample['oracle_option'].strip().upper()
    al = CHOICE_MAP.get(oracle[0], 0)
    prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
    answer_enc = tokenizer('\nAnswer:', return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
        all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t, rounds=3)
        if all_cl[-1].argmax(dim=-1).item() == al: correct += 1
acc = correct / len(eval_idx)
print(f'  {tag}: {acc:.4f}', flush=True)

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
    json.dump({'tag': tag, 'model': 'Llama-3.1-8B-Instruct', 'inject_layer': L, 'accuracy': acc, 'seed': seed}, f, indent=2)
del controller, optimizer
torch.cuda.empty_cache()
"
    done
done
