#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_pla_pair_%j.log
#SBATCH --job-name=deeppass_plap

# Per-LAYER alpha on BOTH blocks in the pair (0,7)+(45,52) on 72B.
# 14 total alphas: 7 for block (0,7) + 7 for block (45,52).
#
# Start with known optimals from single-block experiment:
#   (45,52): [L0=1.1, L1=1.0, L2=0.5, L3=1.3, L4=1.0, L5=0.9, L6=1.1]
#   (0,7):   all at 1.0 (no prior per-layer data)
#
# Steps:
# 1. Sensitivity analysis: disable each layer one at a time
# 2. Coordinate descent: sweep each layer's alpha while holding others fixed

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Per-Layer Alpha PAIR Optimization ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('PER-LAYER ALPHA PAIR OPTIMIZATION')
print('Blocks: (0,7) + (45,52) = 14 per-layer alphas')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

# =====================================================================
# Core functions
# =====================================================================

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def generate_per_layer_alpha(prompt, blocks, layer_alphas, max_new_tokens=64):
    \"\"\"
    layer_alphas: dict mapping (block_tuple, layer_offset) -> alpha
    For each step in the second pass of a block, applies:
      h = h_before + alpha * (h_after - h_before)
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)

    # Map each second-pass step to its per-layer alpha
    step_alphas = {}
    for block in sorted_blocks:
        i, j = block
        block_layers = list(range(i, j))
        count = {}
        offset = 0
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:  # second occurrence = second pass
                    key = (block, offset)
                    step_alphas[step] = layer_alphas.get(key, 1.0)
                    offset += 1

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                h_before = h.clone()
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h_after = out[0] if isinstance(out, tuple) else out

                if step_idx in step_alphas:
                    alpha = step_alphas[step_idx]
                    h = h_before + alpha * (h_after - h_before)
                else:
                    h = h_after

            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def evaluate(blocks, layer_alphas, name):
    gen = lambda p: generate_per_layer_alpha(p, blocks, layer_alphas, max_new_tokens=64)
    gen_long = lambda p: generate_per_layer_alpha(p, blocks, layer_alphas, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:70s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# =====================================================================
# Setup: blocks and initial alphas
# =====================================================================

pair_blocks = [(0, 7), (45, 52)]
block_07 = (0, 7)
block_4552 = (45, 52)
block_size = 7  # both blocks have 7 layers

# Initialize alphas: (0,7) at 1.0, (45,52) from known single-block optimal
best_la = {}
# Block (0,7): start at 1.0
for i in range(block_size):
    best_la[(block_07, i)] = 1.0
# Block (45,52): known optimal from single-block experiment
known_4552 = [1.1, 1.0, 0.5, 1.3, 1.0, 0.9, 1.1]
for i in range(block_size):
    best_la[(block_4552, i)] = known_4552[i]

all_results = []

# =====================================================================
# TEST 1: Baselines
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 1: Baselines with pair (0,7)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

# Baseline: all uniform alpha=1.0
la_uniform = {}
for i in range(block_size):
    la_uniform[(block_07, i)] = 1.0
    la_uniform[(block_4552, i)] = 1.0
r = evaluate(pair_blocks, la_uniform, 'uniform alpha=1.0 both blocks')
all_results.append(r)

# Known (45,52) per-layer optimal + (0,7) uniform 1.0
r = evaluate(pair_blocks, dict(best_la), '(0,7)@1.0 + (45,52) per-layer [1.1,1.0,0.5,1.3,1.0,0.9,1.1]')
all_results.append(r)
best_combined = r['combined']

# Known (45,52) per-layer optimal + (0,7) uniform 0.9
la_09 = dict(best_la)
for i in range(block_size):
    la_09[(block_07, i)] = 0.9
r = evaluate(pair_blocks, la_09, '(0,7)@0.9 + (45,52) per-layer optimal')
all_results.append(r)
if r['combined'] > best_combined:
    best_combined = r['combined']
    best_la = dict(la_09)
    print(f'    >>> New best from (0,7)@0.9: {best_combined:.2f}', flush=True)

# =====================================================================
# TEST 2: Sensitivity analysis — disable each of 14 layers one at a time
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 2: Sensitivity analysis — disable each layer (alpha=0)')
print('14 layers total: (0,7) L0-L6 + (45,52) L0-L6')
print(f'{\"=\" * 70}', flush=True)

sensitivity = {}

print(f'\n  --- Block (0,7): disabling layers one at a time ---', flush=True)
for layer_offset in range(block_size):
    trial_la = dict(best_la)
    trial_la[(block_07, layer_offset)] = 0.0
    r = evaluate(pair_blocks, trial_la, f'(0,7) L{layer_offset} (global {layer_offset}) disabled')
    sensitivity[(block_07, layer_offset)] = r['combined']
    all_results.append(r)

print(f'\n  --- Block (45,52): disabling layers one at a time ---', flush=True)
for layer_offset in range(block_size):
    trial_la = dict(best_la)
    trial_la[(block_4552, layer_offset)] = 0.0
    r = evaluate(pair_blocks, trial_la, f'(45,52) L{layer_offset} (global {45+layer_offset}) disabled')
    sensitivity[(block_4552, layer_offset)] = r['combined']
    all_results.append(r)

print(f'\n  Layer sensitivity (combined when disabled):', flush=True)
for key, c in sorted(sensitivity.items(), key=lambda x: x[1]):
    block, offset = key
    global_idx = block[0] + offset
    block_name = f'({block[0]},{block[1]})'
    label = 'CRITICAL' if c < 70 else 'important' if c < 75 else 'dispensable'
    print(f'    {block_name} L{offset} (global {global_idx}): {c:.2f} — {label}', flush=True)

# =====================================================================
# TEST 3: Coordinate descent on (0,7) layers (hold (45,52) at known optimal)
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 3: Coordinate descent on block (0,7) — 7 layers')
print('Holding (45,52) at known optimal: [1.1,1.0,0.5,1.3,1.0,0.9,1.1]')
print(f'{\"=\" * 70}', flush=True)

alpha_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

for iteration in range(2):
    improved = False
    print(f'\n  --- Iteration {iteration} (block (0,7)) ---', flush=True)
    for layer_offset in range(block_size):
        current_a = best_la[(block_07, layer_offset)]
        for a in alpha_grid:
            if a == current_a: continue
            trial_la = dict(best_la)
            trial_la[(block_07, layer_offset)] = a
            r = evaluate(pair_blocks, trial_la, f'iter{iteration} (0,7) L{layer_offset}={a}')
            all_results.append(r)
            if r['combined'] > best_combined:
                best_combined = r['combined']
                best_la = dict(trial_la)
                improved = True
                print(f'    >>> Improved! (0,7) L{layer_offset}={a} -> {best_combined:.2f}', flush=True)
    if not improved:
        print(f'  Block (0,7) converged at iteration {iteration}', flush=True)
        break

block07_str = ', '.join(f'L{i}={best_la[(block_07,i)]:.1f}' for i in range(block_size))
print(f'\n  Best (0,7) per-layer: [{block07_str}]', flush=True)
print(f'  Combined so far: {best_combined:.2f}', flush=True)

# =====================================================================
# TEST 4: Coordinate descent on (45,52) layers (with updated (0,7))
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 4: Coordinate descent on block (45,52) — refine with (0,7) context')
print(f'{\"=\" * 70}', flush=True)

for iteration in range(2):
    improved = False
    print(f'\n  --- Iteration {iteration} (block (45,52)) ---', flush=True)
    for layer_offset in range(block_size):
        current_a = best_la[(block_4552, layer_offset)]
        for a in alpha_grid:
            if a == current_a: continue
            trial_la = dict(best_la)
            trial_la[(block_4552, layer_offset)] = a
            r = evaluate(pair_blocks, trial_la, f'iter{iteration} (45,52) L{layer_offset}={a}')
            all_results.append(r)
            if r['combined'] > best_combined:
                best_combined = r['combined']
                best_la = dict(trial_la)
                improved = True
                print(f'    >>> Improved! (45,52) L{layer_offset}={a} -> {best_combined:.2f}', flush=True)
    if not improved:
        print(f'  Block (45,52) converged at iteration {iteration}', flush=True)
        break

block4552_str = ', '.join(f'L{i}={best_la[(block_4552,i)]:.1f}' for i in range(block_size))
print(f'\n  Best (45,52) per-layer: [{block4552_str}]', flush=True)
print(f'  Combined so far: {best_combined:.2f}', flush=True)

# =====================================================================
# TEST 5: Joint refinement — one more pass over ALL 14 alphas
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 5: Joint refinement pass over all 14 alphas')
print(f'{\"=\" * 70}', flush=True)

# Finer grid around current values
fine_grid = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.15, 1.2, 1.3, 1.5]

improved = False
for block in [block_07, block_4552]:
    block_name = f'({block[0]},{block[1]})'
    print(f'\n  --- Refining {block_name} ---', flush=True)
    for layer_offset in range(block_size):
        current_a = best_la[(block, layer_offset)]
        for a in fine_grid:
            if a == current_a: continue
            trial_la = dict(best_la)
            trial_la[(block, layer_offset)] = a
            r = evaluate(pair_blocks, trial_la, f'refine {block_name} L{layer_offset}={a}')
            all_results.append(r)
            if r['combined'] > best_combined:
                best_combined = r['combined']
                best_la = dict(trial_la)
                improved = True
                print(f'    >>> Improved! {block_name} L{layer_offset}={a} -> {best_combined:.2f}', flush=True)

if not improved:
    print(f'  No improvement in joint refinement — fully converged', flush=True)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}', flush=True)

block07_final = ', '.join(f'L{i}={best_la[(block_07,i)]:.1f}' for i in range(block_size))
block4552_final = ', '.join(f'L{i}={best_la[(block_4552,i)]:.1f}' for i in range(block_size))

print(f'\nBest per-layer alphas for pair (0,7)+(45,52):', flush=True)
print(f'  Block (0,7):  [{block07_final}]', flush=True)
print(f'  Block (45,52): [{block4552_final}]', flush=True)
print(f'  Combined score: {best_combined:.2f}', flush=True)

print(f'\nTop 20 configurations:', flush=True)
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:20]:
    print(f'  {r[\"name\"]:70s}: combined={r[\"combined\"]:.2f}', flush=True)

print(f'\nSensitivity ranking (most to least critical):', flush=True)
for key, c in sorted(sensitivity.items(), key=lambda x: x[1]):
    block, offset = key
    global_idx = block[0] + offset
    block_name = f'({block[0]},{block[1]})'
    print(f'  {block_name} L{offset} (global {global_idx}): {c:.2f}', flush=True)

# Save results
os.makedirs('results/data/72b/per_layer_alpha', exist_ok=True)
with open('results/data/72b/per_layer_alpha/pair_results.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'blocks': [[0, 7], [45, 52]],
        'best_per_layer_07': {f'L{i}': best_la[(block_07, i)] for i in range(block_size)},
        'best_per_layer_4552': {f'L{i}': best_la[(block_4552, i)] for i in range(block_size)},
        'best_combined': best_combined,
        'sensitivity': {f'({k[0][0]},{k[0][1]})_L{k[1]}': v for k, v in sensitivity.items()},
        'all_results': all_results,
    }, f, indent=2)
print(f'Saved to results/data/72b/per_layer_alpha/pair_results.json', flush=True)
"

echo "=== Done at $(date) ==="
