#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_variance_%j.log
#SBATCH --job-name=deeppass_var

# Variance test: run best configs 5 times each to confirm results aren't flukes
# The dual probe has inherent variance from generation randomness

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Variance Test: Best Configs ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('VARIANCE TEST: 5 runs each of best configs')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def find_seams(layer_order, blocks):
    seams = []
    for block in sorted(blocks):
        last_layer = block[1] - 1
        occurrences = [step for step, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) >= 2: seams.append((occurrences[0], occurrences[1]))
        else: seams.append(None)
    return seams

def generate_multi_alpha(prompt, blocks, alphas, max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    block_to_alpha = {b: a for b, a in zip(blocks, alphas)}
    layer_order = build_order(sorted_blocks, N)
    seams = find_seams(layer_order, sorted_blocks)
    sorted_alphas = [block_to_alpha[b] for b in sorted_blocks]
    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            saved_h1 = {}
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                for si, seam in enumerate(seams):
                    if seam is None: continue
                    if step_idx == seam[0]: saved_h1[si] = h.clone()
                    if step_idx == seam[1] and si in saved_h1:
                        h = saved_h1[si] + sorted_alphas[si] * (h - saved_h1[si])
                        del saved_h1[si]
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

CONFIGS = [
    ('baseline', [], []),
    ('pair @1.0', [(0,7), (45,52)], [1.0, 1.0]),
    ('pair @0.9/1.0', [(0,7), (45,52)], [0.9, 1.0]),
    ('triple (15,20)@0.1', [(0,7), (15,20), (45,52)], [1.0, 0.1, 1.0]),
    ('triple (20,27)@0.1 opt', [(0,7), (20,27), (45,52)], [0.9, 0.1, 1.0]),
]

N_RUNS = 5
all_results = {}

for config_name, blocks, alphas in CONFIGS:
    print(f'\\n--- {config_name} ({N_RUNS} runs) ---', flush=True)
    runs = []
    for run_idx in range(N_RUNS):
        if not blocks:
            gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
            gen_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
        else:
            gen = lambda p, b=blocks, a=alphas: generate_multi_alpha(p, b, a, max_new_tokens=64)
            gen_long = lambda p, b=blocks, a=alphas: generate_multi_alpha(p, b, a, max_new_tokens=128)
        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        runs.append({'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})
        print(f'  Run {run_idx+1}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}', flush=True)

    combineds = [r['combined'] for r in runs]
    mean_c = np.mean(combineds)
    std_c = np.std(combineds)
    print(f'  MEAN={mean_c:.2f} STD={std_c:.2f} MIN={min(combineds):.2f} MAX={max(combineds):.2f}', flush=True)
    all_results[config_name] = {'runs': runs, 'mean': mean_c, 'std': std_c}

# Summary with significance
print(f'\\n{\"=\" * 70}')
print('VARIANCE SUMMARY')
print(f'{\"=\" * 70}')
for name, data in all_results.items():
    print(f'  {name:35s}: {data[\"mean\"]:.2f} +/- {data[\"std\"]:.2f}', flush=True)

# Statistical test: does triple beat pair?
from scipy.stats import ttest_ind
if 'pair @1.0' in all_results and 'triple (15,20)@0.1' in all_results:
    pair_vals = [r['combined'] for r in all_results['pair @1.0']['runs']]
    triple_vals = [r['combined'] for r in all_results['triple (15,20)@0.1']['runs']]
    t, p = ttest_ind(triple_vals, pair_vals)
    print(f'\\n  t-test triple vs pair: t={t:.3f} p={p:.4f} {\"SIGNIFICANT\" if p < 0.05 else \"not significant\"}', flush=True)

os.makedirs('results/data/72b/variance_test', exist_ok=True)
with open('results/data/72b/variance_test/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'n_runs': N_RUNS, 'results': {k: v for k, v in all_results.items()}}, f, indent=2)
print(f'\\nSaved to results/data/72b/variance_test/results.json', flush=True)
"

echo "=== Done at $(date) ==="
