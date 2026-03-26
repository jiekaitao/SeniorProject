#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_alpha_opt_%j.log
#SBATCH --job-name=deeppass_aopt

# Experiment 1: Find optimal alpha per block across top-8 72B blocks
# Experiment 2: Multi-block stacking with per-block alpha tuning
#   - Stack 2,3,4 blocks greedily
#   - Coordinate-descent optimization of per-block alphas
#   - Can partial alphas unlock triples that failed at alpha=1.0?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Alpha Optimization + Multi-Block Alpha Tuning ==="
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
print('ALPHA OPTIMIZATION + MULTI-BLOCK ALPHA TUNING')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

# =====================================================================
# Core: generate with per-block alphas
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

def find_seams(layer_order, blocks):
    \"\"\"Find (first_pass_end, second_pass_end) step indices for each block.\"\"\"
    seams = []
    for block in sorted(blocks):
        i, j = block
        last_layer = j - 1
        occurrences = [step for step, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) >= 2:
            seams.append((occurrences[0], occurrences[1]))
        else:
            seams.append(None)
    return seams

def generate_multi_alpha(prompt, blocks, alphas, max_new_tokens=64):
    \"\"\"Generate with different alpha at each block's seam.\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    # Map alphas to sorted order
    block_to_alpha = {}
    for b, a in zip(blocks, alphas):
        block_to_alpha[b] = a

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
                    if seam is None:
                        continue
                    first_end, second_end = seam
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
                    if step_idx == second_end and si in saved_h1:
                        a = sorted_alphas[si]
                        h1 = saved_h1[si]
                        h2 = h
                        h = h1 + a * (h2 - h1)
                        del saved_h1[si]

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


def evaluate(blocks, alphas, name):
    gen = lambda p: generate_multi_alpha(p, blocks, alphas, max_new_tokens=64)
    gen_long = lambda p: generate_multi_alpha(p, blocks, alphas, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:55s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alphas': list(alphas),
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# Baseline
from layer_duplicator import generate_no_cache
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

all_results = []

# =====================================================================
# PART 1: Optimal alpha per single block (fine sweep)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PART 1: Optimal alpha per block (fine sweep)')
print(f'{\"=\" * 70}', flush=True)

TOP_BLOCKS = [(0,7), (45,52), (50,60), (15,20)]
ALPHA_GRID = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]

optimal_alphas = {}
for block in TOP_BLOCKS:
    print(f'\\n  Block ({block[0]},{block[1]}):', flush=True)
    best_alpha = 1.0
    best_combined = 0
    for alpha in ALPHA_GRID:
        name = f'({block[0]},{block[1]}) alpha={alpha}'
        r = evaluate([block], [alpha], name)
        all_results.append(r)
        if r['combined'] > best_combined:
            best_combined = r['combined']
            best_alpha = alpha
    optimal_alphas[block] = best_alpha
    print(f'  >>> Best alpha for ({block[0]},{block[1]}): {best_alpha} -> combined={best_combined:.2f}', flush=True)

print(f'\\nOptimal alphas: {optimal_alphas}', flush=True)

# =====================================================================
# PART 2: Multi-block stacking with alpha tuning
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PART 2: Multi-block greedy stacking with per-block alpha tuning')
print('Hypothesis: triples fail at alpha=1.0, but partial alphas may rescue them')
print(f'{\"=\" * 70}', flush=True)

# Known best blocks in priority order (from spectral + probe results)
STACK_CANDIDATES = [(45,52), (0,7), (50,60), (15,20), (55,62), (10,17)]

# Greedy stacking: add blocks one at a time, optimize alpha at each step
# For depth=2, we know (0,7)+(45,52) is best
# For depth=3+, add next non-overlapping block, then tune all alphas

def blocks_overlap(b1, b2):
    return not (b1[1] <= b2[0] or b2[1] <= b1[0])

def coordinate_descent(blocks, initial_alphas, alpha_grid, max_iters=3):
    \"\"\"Optimize per-block alphas via coordinate descent.\"\"\"
    alphas = list(initial_alphas)
    best_combined = 0
    best_alphas = list(alphas)

    # Initial evaluation
    name = '+'.join(f'({b[0]},{b[1]})@{a}' for b, a in zip(blocks, alphas))
    r = evaluate(blocks, alphas, f'init: {name}')
    best_combined = r['combined']
    best_alphas = list(alphas)

    for iteration in range(max_iters):
        improved = False
        for block_idx in range(len(blocks)):
            current_best_a = alphas[block_idx]
            for a in alpha_grid:
                if a == alphas[block_idx]:
                    continue
                trial_alphas = list(alphas)
                trial_alphas[block_idx] = a
                name = '+'.join(f'({b[0]},{b[1]})@{ta}' for b, ta in zip(blocks, trial_alphas))
                r = evaluate(blocks, trial_alphas, f'iter{iteration} blk{block_idx}: {name}')
                if r['combined'] > best_combined:
                    best_combined = r['combined']
                    best_alphas = list(trial_alphas)
                    alphas[block_idx] = a
                    improved = True
                    print(f'    >>> Improved! New best: {best_combined:.2f} alphas={best_alphas}', flush=True)
        if not improved:
            print(f'    Converged at iteration {iteration}', flush=True)
            break

    return best_alphas, best_combined

# --- Depth 2: (0,7)+(45,52) with alpha tuning ---
print(f'\\n--- Depth 2: (0,7)+(45,52) ---', flush=True)
blocks_2 = [(0,7), (45,52)]
# Use optimal single-block alphas as starting point
init_alphas_2 = [optimal_alphas.get(b, 1.0) for b in blocks_2]
best_alphas_2, best_combined_2 = coordinate_descent(
    blocks_2, init_alphas_2, [0.8, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25], max_iters=2)
print(f'  Best depth-2: alphas={best_alphas_2} combined={best_combined_2:.2f}', flush=True)

# --- Depth 3: add best non-overlapping third block ---
print(f'\\n--- Depth 3: finding third block ---', flush=True)
third_candidates = [b for b in STACK_CANDIDATES if not any(blocks_overlap(b, eb) for eb in blocks_2)]
print(f'  Third block candidates: {third_candidates}', flush=True)

# Quick screen: test each third block at alpha=1.0 first
depth3_results = []
for third in third_candidates[:4]:
    blocks_3 = blocks_2 + [third]
    # All at alpha=1.0 first
    r = evaluate(blocks_3, [1.0, 1.0, 1.0], f'depth3 all@1.0: +({third[0]},{third[1]})')
    depth3_results.append((third, r['combined']))

# Pick best third block (even if negative) and optimize alphas
depth3_results.sort(key=lambda x: x[1], reverse=True)
best_third = depth3_results[0][0]
print(f'\\n  Best third block (at alpha=1.0): ({best_third[0]},{best_third[1]}) combined={depth3_results[0][1]:.2f}', flush=True)

blocks_3 = blocks_2 + [best_third]
print(f'\\n--- Depth 3: alpha tuning on {[\",\".join(str(x) for x in b) for b in blocks_3]} ---', flush=True)
# Start with optimal pair alphas + 1.0 for third
init_alphas_3 = list(best_alphas_2) + [1.0]
best_alphas_3, best_combined_3 = coordinate_descent(
    blocks_3, init_alphas_3, [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], max_iters=2)
print(f'  Best depth-3: alphas={best_alphas_3} combined={best_combined_3:.2f}', flush=True)

# --- Depth 4: add fourth block ---
print(f'\\n--- Depth 4: finding fourth block ---', flush=True)
fourth_candidates = [b for b in STACK_CANDIDATES if not any(blocks_overlap(b, eb) for eb in blocks_3)]
if fourth_candidates:
    fourth = fourth_candidates[0]
    blocks_4 = blocks_3 + [fourth]
    r = evaluate(blocks_4, [1.0]*4, f'depth4 all@1.0: +({fourth[0]},{fourth[1]})')

    print(f'\\n--- Depth 4: alpha tuning ---', flush=True)
    init_alphas_4 = list(best_alphas_3) + [1.0]
    best_alphas_4, best_combined_4 = coordinate_descent(
        blocks_4, init_alphas_4, [0.3, 0.5, 0.7, 0.9, 1.0, 1.1], max_iters=2)
    print(f'  Best depth-4: alphas={best_alphas_4} combined={best_combined_4:.2f}', flush=True)
else:
    print('  No non-overlapping fourth block available', flush=True)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}')
print(f'Baseline:    {baseline:.2f}')
print(f'Best single: (45,52)@{optimal_alphas.get((45,52),1.0)} = ?')
print(f'Best depth-2: alphas={best_alphas_2} combined={best_combined_2:.2f}')
print(f'Best depth-3: alphas={best_alphas_3} combined={best_combined_3:.2f}')
if fourth_candidates:
    print(f'Best depth-4: alphas={best_alphas_4} combined={best_combined_4:.2f}')
print(f'\\nKey question: can alpha tuning rescue triples?')
print(f'Depth-3 vs depth-2: {best_combined_3 - best_combined_2:+.2f}', flush=True)

# Save
os.makedirs('results/data/72b/alpha_optimization', exist_ok=True)
with open('results/data/72b/alpha_optimization/results.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'baseline': baseline,
        'optimal_alphas': {f'({k[0]},{k[1]})': v for k, v in optimal_alphas.items()},
        'all_results': all_results,
        'depth2': {'blocks': [list(b) for b in blocks_2], 'alphas': best_alphas_2, 'combined': best_combined_2},
        'depth3': {'blocks': [list(b) for b in blocks_3], 'alphas': best_alphas_3, 'combined': best_combined_3},
    }, f, indent=2)
print(f'\\nSaved to results/data/72b/alpha_optimization/results.json', flush=True)
"

echo "=== Done at $(date) ==="
