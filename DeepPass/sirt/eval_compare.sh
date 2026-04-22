#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_compare_%j.log
#SBATCH --job-name=sirt_cmp

# Compare SIRT vs Dense on dual probe + generation + adaptive behavior

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT vs Dense Comparison ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')

from model import SIRTConfig, SIRTLM, create_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoTokenizer

SAVE_DIR = 'sirt/eval_results'
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf')
eq_all = _load_questions()

def load_model(ckpt_path):
    cfg = SIRTConfig()
    model = create_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).to(torch.bfloat16).eval()
    return model

def gen_with_k(model, prompt, k, max_tokens=64):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _, _ = model(input_ids, fixed_recursions=k)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def gen_adaptive(model, prompt, max_tokens=64):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    steps = []
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _, aux = model(input_ids, train_halting=True)
        ek = aux['expected_steps'].item() if isinstance(aux['expected_steps'], torch.Tensor) else aux['expected_steps']
        steps.append(ek)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True), sum(steps)/len(steps) if steps else 0

results = {}

# ======================================================================
# SIRT evaluation at different K
# ======================================================================
print('=== SIRT (long training) ===', flush=True)
sirt = load_model('sirt/checkpoints_long/stage3_final.pt')

for k in [1, 2, 3]:
    gen_fn = lambda p, k=k: gen_with_k(sirt, p, k, 64)
    gen_long_fn = lambda p, k=k: gen_with_k(sirt, p, k, 128)
    t0 = time.time()
    math_r = run_math_probe(gen_fn, verbose=False)
    eq_r = run_eq_bench_probe(gen_long_fn, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  SIRT K={k}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    results[f'sirt_k{k}'] = {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# Adaptive
_, avg_k = gen_adaptive(sirt, 'What is 127 * 348?')
print(f'  SIRT adaptive E[K]: {avg_k:.2f}', flush=True)

# Easy vs hard
easy = ['The capital of France is', 'Water boils at', 'The sun rises in the', 'One plus one equals']
hard = ['What is 127 * 348?', 'The square root of 152399025 is', 'If f(x) = 3x^2 - 2x + 1, then f(5) equals', 'What is 2^48?']

easy_ks = [gen_adaptive(sirt, p, 16)[1] for p in easy]
hard_ks = [gen_adaptive(sirt, p, 16)[1] for p in hard]
import numpy as np
print(f'  Easy E[K]: {np.mean(easy_ks):.3f}', flush=True)
print(f'  Hard E[K]: {np.mean(hard_ks):.3f}', flush=True)
print(f'  Adaptive: {\"YES\" if np.mean(hard_ks) > np.mean(easy_ks) else \"NO\"}', flush=True)

del sirt; import gc; gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# Dense baseline evaluation (K=1 only, no recursion)
# ======================================================================
print('\\n=== Dense Baseline ===', flush=True)
dense = load_model('sirt/checkpoints_dense_long/stage1_final.pt')

gen_fn = lambda p: gen_with_k(dense, p, 1, 64)
gen_long_fn = lambda p: gen_with_k(dense, p, 1, 128)
t0 = time.time()
math_r = run_math_probe(gen_fn, verbose=False)
eq_r = run_eq_bench_probe(gen_long_fn, questions=eq_all, verbose=False)
combined = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'  Dense K=1: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
results['dense_k1'] = {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

del dense; gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# Generation samples
# ======================================================================
print('\\n=== Generation Samples ===', flush=True)
sirt = load_model('sirt/checkpoints_long/stage3_final.pt')
dense = load_model('sirt/checkpoints_dense_long/stage1_final.pt')

prompts = [
    'The meaning of life is',
    'To solve a math problem, first',
    'The most important discovery in science was',
]
for p in prompts:
    sirt_out = gen_with_k(sirt, p, 2, 40)
    dense_out = gen_with_k(dense, p, 1, 40)
    print(f'  Prompt: {p}', flush=True)
    print(f'  SIRT:  {sirt_out[:80]}', flush=True)
    print(f'  Dense: {dense_out[:80]}', flush=True)
    print(flush=True)

# ======================================================================
# Summary
# ======================================================================
print(f'\\n{\"=\" * 50}', flush=True)
print('SIRT vs DENSE COMPARISON', flush=True)
print(f'{\"=\" * 50}', flush=True)
for name, r in sorted(results.items()):
    print(f'  {name:>12s}: math={r[\"math\"]:.4f} eq={r[\"eq\"]:.1f} combined={r[\"combined\"]:.2f}', flush=True)
print(f'  Easy E[K]: {np.mean(easy_ks):.3f}', flush=True)
print(f'  Hard E[K]: {np.mean(hard_ks):.3f}', flush=True)
print('COMPLETE', flush=True)

with open(f'{SAVE_DIR}/sirt_vs_dense.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {SAVE_DIR}/sirt_vs_dense.json', flush=True)
"

echo "=== Finished: $(date) ==="
