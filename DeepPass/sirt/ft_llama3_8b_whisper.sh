#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_ll3_whisper_%j.log
#SBATCH --job-name=ll3_whi

# LLaMA 3 8B: Recursion FT with WHISPER FFN during K=2
# Combine our two best findings:
# 1. Recursion fine-tuning (K=2 > K=1)
# 2. Whisper FFN (β=0.2 preserves MMLU, helps MATH)
# During fine-tuning AND inference, scale FFN by 0.2 on second pass

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 8B Whisper FFN Recursion FT ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, torch, torch.nn as nn, random, numpy as np, functools
sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')
from recursion_finetune import LayerIdxWrapper, build_recursive_model, restore_model, evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
SAVE_DIR = 'sirt/recursion_ft/llama3_8b_whisper'
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda')

print('Loading LLaMA 3 8B...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)
inner = model.model; original_layers = list(inner.layers); N = len(original_layers)

# Baseline
print('\\n=== Baseline ===', flush=True)
baseline = evaluate_model(model, tokenizer, device, 1, 'baseline')

# ======================================================================
# Fine-tune with whisper FFN: during K=2, scale FFN output by 0.2
# ======================================================================
print('\\n=== Whisper FFN Recursion Fine-Tune ===', flush=True)
core_start, core_end = 10, 13
BETA = 0.2

for param in model.parameters(): param.requires_grad = False
for i in range(core_start, core_end):
    for param in original_layers[i].parameters(): param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Trainable: {trainable:,}', flush=True)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-7, weight_decay=0.01)
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
model.train()
step = 0; token_buffer = []

for example in ds:
    if step >= 500: break
    text = example.get('text', '')
    if not text or len(text) < 100: continue
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2048)
    if tokenizer.eos_token_id: tokens.append(tokenizer.eos_token_id)
    token_buffer.extend(tokens)

    while len(token_buffer) >= 1025 and step < 500:
        chunk = torch.tensor([token_buffer[:1025]], dtype=torch.long).to(device)
        token_buffer = token_buffer[1024:]

        K = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        orig, orig_N, new_N = build_recursive_model(model, core_start, core_end, K)

        # Add whisper FFN hooks for K>1 passes
        hooks = []
        if K > 1:
            for layer_idx in range(core_start, core_end):
                module = original_layers[layer_idx]
                counter = [0]
                def make_whisper(ctr, beta=BETA):
                    def hook(mod, inp, out):
                        ctr[0] += 1
                        # Apply whisper only on passes 2+ (counter > core_end - core_start)
                        n_core = core_end - core_start
                        if ctr[0] > n_core:  # second pass onwards
                            if isinstance(out, tuple):
                                return (beta * out[0],) + out[1:]
                            return beta * out
                        return out
                    return hook
                h = module.mlp.register_forward_hook(make_whisper(counter))
                hooks.append(h)

        try:
            outputs = model(chunk[:, :-1], use_cache=False)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                chunk[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            print(f'  step {step} error: {e}', flush=True)
            optimizer.zero_grad(set_to_none=True)

        for h in hooks: h.remove()
        restore_model(model, orig, orig_N)
        step += 1
        if step % 100 == 0: print(f'  step {step} | loss {loss.item():.4f} | K={K}', flush=True)

for param in model.parameters(): param.requires_grad = False
model.eval()
print(f'Fine-tune done: {step} steps', flush=True)

# ======================================================================
# Evaluate: K=1 (no hooks), K=2 with whisper FFN, K=2 without whisper
# ======================================================================
print('\\n=== Evaluation ===', flush=True)

# K=1 (should be preserved)
post_k1 = evaluate_model(model, tokenizer, device, 1, 'post-ft K=1')

# K=2 without whisper (standard)
orig, oN, nN = build_recursive_model(model, core_start, core_end, 2)
post_k2_full = evaluate_model(model, tokenizer, device, 2, 'post-ft K=2 full FFN')
restore_model(model, orig, oN)

# K=2 with whisper FFN
orig, oN, nN = build_recursive_model(model, core_start, core_end, 2)
hooks = []
for layer_idx in range(core_start, core_end):
    module = original_layers[layer_idx]
    counter = [0]
    def make_whisper_eval(ctr, beta=BETA, n_core=core_end-core_start):
        def hook(mod, inp, out):
            ctr[0] += 1
            if ctr[0] > n_core:
                if isinstance(out, tuple):
                    return (beta * out[0],) + out[1:]
                return beta * out
            return out
        return hook
    h = module.mlp.register_forward_hook(make_whisper_eval(counter))
    hooks.append(h)

post_k2_whisper = evaluate_model(model, tokenizer, device, 2, 'post-ft K=2 whisper FFN')
for h in hooks: h.remove()
restore_model(model, orig, oN)

# ======================================================================
# lm-eval on best config
# ======================================================================
print('\\n=== lm-eval on best K=2 config ===', flush=True)

best_config = 'whisper' if post_k2_whisper['combined'] > post_k2_full['combined'] else 'full'
print(f'  Best: K=2 {best_config} (whisper={post_k2_whisper[\"combined\"]:.2f}, full={post_k2_full[\"combined\"]:.2f})', flush=True)

# Apply K=2
orig, oN, nN = build_recursive_model(model, core_start, core_end, 2)
if best_config == 'whisper':
    hooks = []
    for layer_idx in range(core_start, core_end):
        module = original_layers[layer_idx]
        counter = [0]
        h = module.mlp.register_forward_hook(make_whisper_eval([0]))
        hooks.append(h)

model.config.use_cache = True
from lm_eval.models.huggingface import HFLM; from lm_eval import evaluator
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
t0 = time.time()
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)
elapsed = time.time() - t0

scores = {f'{t}/{m}': v for t, d in results['results'].items() for m, v in d.items() if isinstance(v, (int, float))}
print(f'\\n=== lm-eval RESULTS ({best_config} K=2) === [{elapsed:.0f}s]', flush=True)
metrics = ['leaderboard_bbh/acc_norm,none', 'leaderboard_math_hard/exact_match,none', 'leaderboard_mmlu_pro/acc,none', 'leaderboard_musr/acc_norm,none']
for m in metrics:
    v = scores.get(m, 0)
    print(f'  {m.split(\"/\")[0].replace(\"leaderboard_\",\"\")}: {v:.4f}', flush=True)

# Save
all_results = {
    'baseline': baseline, 'post_k1': post_k1,
    'post_k2_full': post_k2_full, 'post_k2_whisper': post_k2_whisper,
    'best_config': best_config,
    'lm_eval_scores': scores,
}
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\\n{\"=\" * 50}', flush=True)
print(f'SUMMARY', flush=True)
print(f'  Baseline K=1:        {baseline[\"combined\"]:.2f}', flush=True)
print(f'  Post-ft K=1:         {post_k1[\"combined\"]:.2f} ({post_k1[\"combined\"]-baseline[\"combined\"]:+.2f})', flush=True)
print(f'  Post-ft K=2 full:    {post_k2_full[\"combined\"]:.2f} ({post_k2_full[\"combined\"]-baseline[\"combined\"]:+.2f})', flush=True)
print(f'  Post-ft K=2 whisper: {post_k2_whisper[\"combined\"]:.2f} ({post_k2_whisper[\"combined\"]-baseline[\"combined\"]:+.2f})', flush=True)
print(f'COMPLETE', flush=True)
"

echo "=== Finished: $(date) ==="
