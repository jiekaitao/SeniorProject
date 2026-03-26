#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_pla_triple_%j.log
#SBATCH --job-name=deeppass_plat

# Per-LAYER alpha optimization for the FULL TRIPLE: (0,7)+(20,27)+(45,52)
# 21 total alphas: 7 per block.
#
# Starting points (known good per-block alphas):
#   (0,7)   all layers at 0.9
#   (20,27) all layers at 0.15
#   (45,52) optimized: [1.1, 1.0, 0.5, 1.3, 1.0, 0.9, 1.1]
#
# Coordinate descent: sweep each of the 21 layer-alphas one at a time.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Per-Layer Alpha Triple Optimization ==="
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
print('PER-LAYER ALPHA OPTIMIZATION — FULL TRIPLE')
print('Blocks: (0,7) + (20,27) + (45,52) = 21 layer-alphas')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

# =====================================================================
# Block definitions and sweep ranges
# =====================================================================

BLOCKS = [(0, 7), (20, 27), (45, 52)]
BLOCK_SIZE = 7  # all blocks are size 7

# Per-block sweep ranges
SWEEP_RANGES = {
    (0, 7):   [0.5, 0.7, 0.9, 1.0, 1.1, 1.2],
    (20, 27): [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    (45, 52): [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
}

# Starting per-layer alphas (known good per-block values)
INIT_ALPHAS_45_52 = [1.1, 1.0, 0.5, 1.3, 1.0, 0.9, 1.1]

initial_layer_alphas = {}
for offset in range(BLOCK_SIZE):
    initial_layer_alphas[((0, 7), offset)]   = 0.9
    initial_layer_alphas[((20, 27), offset)]  = 0.15
    initial_layer_alphas[((45, 52), offset)]  = INIT_ALPHAS_45_52[offset]

print('Initial per-layer alphas:', flush=True)
for block in BLOCKS:
    vals = [initial_layer_alphas[(block, i)] for i in range(BLOCK_SIZE)]
    print(f'  ({block[0]},{block[1]}): {vals}', flush=True)

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
    For each layer in the second pass, applies:
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
        if next_token.item() == tokenizer.eos_token_id:
            break
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

all_results = []

# =====================================================================
# BASELINE: evaluate with initial per-layer alphas
# =====================================================================
print(f'\n{\"=\" * 70}')
print('BASELINE: Initial per-layer alphas')
print(f'{\"=\" * 70}', flush=True)

r = evaluate(BLOCKS, initial_layer_alphas, 'initial: (0,7)@0.9 (20,27)@0.15 (45,52)@optimized')
all_results.append(r)
best_combined = r['combined']
best_la = dict(initial_layer_alphas)
print(f'  Baseline combined: {best_combined:.2f}', flush=True)

# =====================================================================
# COORDINATE DESCENT: sweep each of 21 layer-alphas
# =====================================================================
print(f'\n{\"=\" * 70}')
print('COORDINATE DESCENT: 21 layer-alphas')
print('Order: (0,7) L0-L6, then (20,27) L0-L6, then (45,52) L0-L6')
print(f'{\"=\" * 70}', flush=True)

for iteration in range(3):
    print(f'\n--- Iteration {iteration} ---', flush=True)
    improved_this_iter = False

    for block in BLOCKS:
        sweep_range = SWEEP_RANGES[block]
        print(f'\n  Block ({block[0]},{block[1]}), sweep range: {sweep_range}', flush=True)

        for layer_offset in range(BLOCK_SIZE):
            current_alpha = best_la[(block, layer_offset)]
            global_layer = block[0] + layer_offset
            print(f'\n    Layer {layer_offset} (global {global_layer}), current alpha={current_alpha}', flush=True)

            for a in sweep_range:
                if a == current_alpha:
                    continue
                trial_la = dict(best_la)
                trial_la[(block, layer_offset)] = a
                name = f'iter{iteration} ({block[0]},{block[1]}) L{layer_offset}={a}'
                r = evaluate(BLOCKS, trial_la, name)
                all_results.append(r)
                if r['combined'] > best_combined:
                    best_combined = r['combined']
                    best_la = dict(trial_la)
                    improved_this_iter = True
                    print(f'      >>> IMPROVED! L{layer_offset}={a} -> combined={best_combined:.2f}', flush=True)

    # Print current best after each full iteration
    print(f'\n  === End of iteration {iteration}, best combined: {best_combined:.2f} ===', flush=True)
    for block in BLOCKS:
        vals = [best_la[(block, i)] for i in range(BLOCK_SIZE)]
        print(f'    ({block[0]},{block[1]}): {vals}', flush=True)

    if not improved_this_iter:
        print(f'  Converged at iteration {iteration}', flush=True)
        break

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\n{\"=\" * 70}')
print('FINAL SUMMARY')
print(f'{\"=\" * 70}', flush=True)

print(f'\nBest combined score: {best_combined:.2f}', flush=True)
print(f'\nOptimized per-layer alphas (21 total):', flush=True)
for block in BLOCKS:
    vals = [best_la[(block, i)] for i in range(BLOCK_SIZE)]
    print(f'  ({block[0]},{block[1]}): {vals}', flush=True)

print(f'\nTop 20 configurations:', flush=True)
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:20]:
    print(f'  {r[\"name\"]:70s}: combined={r[\"combined\"]:.2f}', flush=True)

# Save results
os.makedirs('results/data/72b/per_layer_alpha_triple', exist_ok=True)
save_data = {
    'date': datetime.now().isoformat(),
    'blocks': [[b[0], b[1]] for b in BLOCKS],
    'initial_alphas': {f'({b[0]},{b[1]})_L{i}': initial_layer_alphas[(b, i)]
                       for b in BLOCKS for i in range(BLOCK_SIZE)},
    'best_alphas': {f'({b[0]},{b[1]})_L{i}': best_la[(b, i)]
                    for b in BLOCKS for i in range(BLOCK_SIZE)},
    'best_alphas_by_block': {
        f'({b[0]},{b[1]})': [best_la[(b, i)] for i in range(BLOCK_SIZE)]
        for b in BLOCKS
    },
    'best_combined': best_combined,
    'sweep_ranges': {f'({k[0]},{k[1]})': v for k, v in SWEEP_RANGES.items()},
    'all_results': all_results,
}
with open('results/data/72b/per_layer_alpha_triple/results.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print(f'\nSaved to results/data/72b/per_layer_alpha_triple/results.json', flush=True)
"

echo "=== Done at $(date) ==="
