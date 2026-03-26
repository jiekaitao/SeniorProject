#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_per_layer_alpha_%j.log
#SBATCH --job-name=deeppass_pla

# Per-LAYER alpha: instead of one alpha per block, each layer in the
# second pass gets its own weight. For block (45,52) that's 7 separate alphas.
# This is the finest granularity possible.
#
# Approach:
# 1. Start with uniform alpha=1.0 for all layers in all blocks
# 2. Sweep each layer individually (coordinate descent)
# 3. Also test structured patterns: linear decay, V-shape, etc.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Per-Layer Alpha Optimization ==="
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
print('PER-LAYER ALPHA OPTIMIZATION')
print('Each layer in the second pass gets its own alpha weight')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

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
    layer_alphas: dict mapping (block, layer_offset) -> alpha
    For each layer in the second pass, applies:
      h = h_before + alpha * (h_after - h_before)
    where alpha is per-layer, not per-block.
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
                if count[idx] == 2:  # second occurrence
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
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# =====================================================================
# TEST 1: Single block (45,52) — 7 layers, per-layer alpha
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 1: Per-layer alpha on (45,52) — 7 layers')
print(f'{\"=\" * 70}', flush=True)

block = (45, 52)
block_size = 7

# Baseline: uniform alpha=1.0
la_uniform = {(block, i): 1.0 for i in range(block_size)}
r = evaluate([block], la_uniform, 'uniform alpha=1.0')
all_results.append(r)

# Uniform alpha=1.15 (known best per-block)
la_115 = {(block, i): 1.15 for i in range(block_size)}
r = evaluate([block], la_115, 'uniform alpha=1.15')
all_results.append(r)

# Pattern 1: Linear decay (first layer high, last layer low)
for high, low in [(1.5, 0.5), (1.3, 0.7), (1.2, 0.9)]:
    alphas = np.linspace(high, low, block_size)
    la = {(block, i): float(alphas[i]) for i in range(block_size)}
    r = evaluate([block], la, f'linear {high}->{low}')
    all_results.append(r)

# Pattern 2: Linear ramp (first layer low, last layer high)
for low, high in [(0.5, 1.5), (0.7, 1.3), (0.9, 1.2)]:
    alphas = np.linspace(low, high, block_size)
    la = {(block, i): float(alphas[i]) for i in range(block_size)}
    r = evaluate([block], la, f'ramp {low}->{high}')
    all_results.append(r)

# Pattern 3: V-shape (high at edges, low in middle)
for edge, mid in [(1.5, 0.5), (1.3, 0.7)]:
    alphas = [edge + (mid - edge) * 2 * abs(i - (block_size-1)/2) / (block_size-1) for i in range(block_size)]
    # Fix: this gives mid at edges, edge at center. Invert:
    alphas = [edge - (edge - mid) * 2 * abs(i - (block_size-1)/2) / (block_size-1) for i in range(block_size)]
    la = {(block, i): float(alphas[i]) for i in range(block_size)}
    r = evaluate([block], la, f'V-shape edge={edge} mid={mid}')
    all_results.append(r)

# Pattern 4: Only first/last layer of block duplicated (rest at alpha=0)
for active_layers in [[0], [6], [0, 6], [0, 1, 2], [4, 5, 6]]:
    la = {(block, i): (1.0 if i in active_layers else 0.0) for i in range(block_size)}
    active_str = ','.join(str(i) for i in active_layers)
    r = evaluate([block], la, f'only layers [{active_str}]')
    all_results.append(r)

# Pattern 5: Coordinate descent — sweep each layer's alpha while holding others at 1.0
print(f'\\n  --- Coordinate descent (per-layer) ---', flush=True)
best_la = {(block, i): 1.0 for i in range(block_size)}
best_combined = 0

# First: find which layers matter most (sensitivity analysis)
layer_sensitivity = {}
for layer_offset in range(block_size):
    # Set this layer to 0 (disable it)
    la = {(block, i): 1.0 for i in range(block_size)}
    la[(block, layer_offset)] = 0.0
    r = evaluate([block], la, f'layer {layer_offset} disabled (alpha=0)')
    layer_sensitivity[layer_offset] = r['combined']
    all_results.append(r)

print(f'\\n  Layer sensitivity (combined when disabled):', flush=True)
for l, c in sorted(layer_sensitivity.items(), key=lambda x: x[1]):
    print(f'    Layer {l} (global {block[0]+l}): {c:.2f} — {\"CRITICAL\" if c < 70 else \"important\" if c < 75 else \"dispensable\"}', flush=True)

# Coordinate descent: optimize each layer
for iteration in range(2):
    improved = False
    for layer_offset in range(block_size):
        current_best_a = best_la[(block, layer_offset)]
        for a in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
            if a == current_best_a: continue
            trial_la = dict(best_la)
            trial_la[(block, layer_offset)] = a
            r = evaluate([block], trial_la, f'iter{iteration} L{layer_offset}={a}')
            all_results.append(r)
            if r['combined'] > best_combined:
                best_combined = r['combined']
                best_la = dict(trial_la)
                improved = True
                print(f'    >>> Improved! L{layer_offset}={a} -> {best_combined:.2f}', flush=True)
    if not improved:
        print(f'  Converged at iteration {iteration}', flush=True)
        break

best_alphas_str = ', '.join(f'L{i}={best_la[(block,i)]:.1f}' for i in range(block_size))
print(f'\\n  Best per-layer alphas: [{best_alphas_str}] = {best_combined:.2f}', flush=True)

# =====================================================================
# TEST 2: Apply best per-layer pattern to the PAIR
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 2: Per-layer alpha on pair (0,7)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

pair_blocks = [(0, 7), (45, 52)]

# Transfer best per-layer alphas from (45,52) to the pair
# Keep (0,7) at uniform alpha=0.9 (known best) or also optimize
pair_la = {}
for i in range(7):  # (0,7) block
    pair_la[((0,7), i)] = 0.9
for i in range(7):  # (45,52) block — use optimized per-layer
    pair_la[((45,52), i)] = best_la.get(((45,52), i), best_la.get((block, i), 1.0))

r = evaluate(pair_blocks, pair_la, f'pair: (0,7)@0.9 + (45,52) per-layer optimized')
all_results.append(r)

# Also try: both blocks with per-layer optimization of (0,7)
# Quick sweep: which layers in (0,7) matter?
print(f'\\n  --- Layer sensitivity for (0,7) in pair context ---', flush=True)
for layer_offset in range(7):
    trial_la = dict(pair_la)
    trial_la[((0,7), layer_offset)] = 0.0
    r = evaluate(pair_blocks, trial_la, f'pair: (0,7) L{layer_offset} disabled')
    all_results.append(r)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:20]:
    print(f'  {r[\"name\"]:65s}: combined={r[\"combined\"]:.2f}', flush=True)

print(f'\\nBest per-layer config: [{best_alphas_str}] = {best_combined:.2f}')
print(f'vs uniform alpha=1.15 = {la_115}', flush=True)

os.makedirs('results/data/72b/per_layer_alpha', exist_ok=True)
with open('results/data/72b/per_layer_alpha/results.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'best_per_layer': {f'L{i}': best_la[(block, i)] for i in range(block_size)},
        'best_combined': best_combined,
        'layer_sensitivity': layer_sensitivity,
        'all_results': all_results,
    }, f, indent=2)
print(f'Saved to results/data/72b/per_layer_alpha/results.json', flush=True)
"

echo "=== Done at $(date) ==="
