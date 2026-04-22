#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_eval_%j.log
#SBATCH --job-name=sirt_ev

# Evaluate SIRT-172M: dual probe, adaptive behavior test, generation quality

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT-172M Evaluation ==="
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
tokenizer = AutoTokenizer.from_pretrained(
    '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
)
eq_all = _load_questions()

def load_sirt(checkpoint_path):
    cfg = SIRTConfig()
    model = create_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, cfg

def generate(model, prompt, max_tokens=64):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _, aux = model(input_ids, fixed_recursions=2)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def generate_adaptive(model, prompt, max_tokens=64):
    \"\"\"Generate with adaptive halting (Stage 3 model).\"\"\"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    input_ids = inputs['input_ids']
    total_steps = []
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _, aux = model(input_ids, train_halting=True)
        total_steps.append(aux['expected_steps'].item() if isinstance(aux['expected_steps'], torch.Tensor) else aux['expected_steps'])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    avg_k = sum(total_steps) / len(total_steps) if total_steps else 0
    return tokenizer.decode(input_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True), avg_k

# ======================================================================
# Test 1: Dual Probe on SIRT (fixed K=2)
# ======================================================================
print('=== Test 1: SIRT Dual Probe (K=2) ===', flush=True)

model, cfg = load_sirt('sirt/checkpoints_full/stage3_final.pt')
gen_fn = lambda p: generate(model, p, 64)
gen_long_fn = lambda p: generate(model, p, 128)

t0 = time.time()
math_r = run_math_probe(gen_fn, verbose=True)
eq_r = run_eq_bench_probe(gen_long_fn, questions=eq_all, verbose=False)
combined = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'\\nSIRT (K=2): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)

# ======================================================================
# Test 2: Different recursion depths
# ======================================================================
print('\\n=== Test 2: SIRT at Different K ===', flush=True)

k_results = []
for k in [1, 2, 3, 4]:
    gen_k = lambda p, k=k: generate_with_k(model, p, k)
    # Inline generate with specific K
    def gen_with_fixed_k(prompt, fixed_k=k):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        input_ids = inputs['input_ids']
        for _ in range(64):
            with torch.no_grad():
                logits, _, _ = model(input_ids, fixed_recursions=fixed_k)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id: break
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return tokenizer.decode(input_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen_with_fixed_k, verbose=False)
    eq_r = run_eq_bench_probe(lambda p: gen_with_fixed_k(p), verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  K={k}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    k_results.append({'K': k, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# ======================================================================
# Test 3: Adaptive behavior (easy vs hard prompts)
# ======================================================================
print('\\n=== Test 3: Adaptive Behavior ===', flush=True)

easy_prompts = [
    'The capital of France is',
    'Water boils at',
    'The color of the sky is',
    'One plus one equals',
    'The sun rises in the',
]

hard_prompts = [
    'What is 127 multiplied by 348? The answer is',
    'The square root of 152399025 is',
    'If f(x) = 3x^2 - 2x + 1, then f(5) equals',
    'What is 9999999 multiplied by 9999999?',
    'Calculate 2 raised to the power of 48.',
]

easy_ks = []
hard_ks = []

for prompt in easy_prompts:
    _, avg_k = generate_adaptive(model, prompt, max_tokens=16)
    easy_ks.append(avg_k)
    print(f'  EASY: E[K]={avg_k:.2f} | {prompt[:40]}', flush=True)

for prompt in hard_prompts:
    _, avg_k = generate_adaptive(model, prompt, max_tokens=16)
    hard_ks.append(avg_k)
    print(f'  HARD: E[K]={avg_k:.2f} | {prompt[:40]}', flush=True)

import numpy as np
easy_mean = np.mean(easy_ks)
hard_mean = np.mean(hard_ks)
print(f'\\n  Easy avg E[K]: {easy_mean:.3f}', flush=True)
print(f'  Hard avg E[K]: {hard_mean:.3f}', flush=True)
print(f'  Difference: {hard_mean - easy_mean:+.3f}', flush=True)
print(f'  Adaptive: {\"YES\" if hard_mean > easy_mean else \"NO\"} (hard uses more recursion)', flush=True)

# ======================================================================
# Test 4: Generation quality samples
# ======================================================================
print('\\n=== Test 4: Generation Samples ===', flush=True)

sample_prompts = [
    'The meaning of life is',
    'In a distant galaxy, there was',
    'The most important invention in history was',
    'To solve climate change, we need to',
]

for prompt in sample_prompts:
    output = generate(model, prompt, max_tokens=50)
    print(f'  Prompt: {prompt}', flush=True)
    print(f'  Output: {output[:100]}', flush=True)
    print(flush=True)

# ======================================================================
# Save results
# ======================================================================
results = {
    'model': 'SIRT-172M (stage3_final)',
    'k_comparison': k_results,
    'adaptive_behavior': {
        'easy_mean_K': float(easy_mean),
        'hard_mean_K': float(hard_mean),
        'difference': float(hard_mean - easy_mean),
        'is_adaptive': bool(hard_mean > easy_mean),
    },
}
with open(f'{SAVE_DIR}/sirt_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/sirt_evaluation.json', flush=True)

print(f'\\n{\"=\" * 50}', flush=True)
print('SIRT EVALUATION COMPLETE', flush=True)
print(f'{\"=\" * 50}', flush=True)
"

echo "=== Finished: $(date) ==="
