#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_xlyr_%j.log
#SBATCH --job-name=deeppass_g3xl

# Cross-layer duplication on Gemma3-27B (62 layers)
# Does G(F(h)) work on Gemma3? First pass through block A, second pass uses block B's weights.
# Layer order: [0..j-1, A_layers, B_layers, j..N-1]
#
# IMPORTANT: Gemma3 manual layer-by-layer loop DOES NOT WORK (sliding window attention breaks).
# All experiments use full-model forward with ModuleList swapping.
# Alpha is not possible without manual loop, so all tests at alpha=1.0 only.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3-27B Cross-Layer Duplication ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/cross_layer'
os.makedirs(SAVE_DIR, exist_ok=True)

print('=' * 70, flush=True)
print('GEMMA3-27B CROSS-LAYER DUPLICATION', flush=True)
print('Does G(F(h)) work on a different architecture?', flush=True)
print(f'Date: {datetime.now().isoformat()}', flush=True)
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded Gemma3-27B: {N} layers on {device}', flush=True)

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    elif hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n

def build_cross_layer_order(first_block, second_block_weights, N):
    \"\"\"
    Cross-layer duplication: first pass through block A, second pass uses block B's weights.
    Layer order: [0..j-1, A_layers, B_layers, j..N-1]
    where first_block = (i,j) and second_block_weights = (a,b).
    The insertion point is after j-1 (same as self-duplication).
    \"\"\"
    i, j = first_block
    a, b = second_block_weights
    order = list(range(j))          # layers before and including first pass
    order += list(range(i, j))      # first pass (block A)
    order += list(range(a, b))      # second pass (block B's weights!)
    order += list(range(j, N))      # layers after block
    return order

def build_self_dup_order(block, N):
    \"\"\"Standard self-duplication: [0..j-1, i..j-1, j..N-1].\"\"\"
    i, j = block
    order = list(range(j)) + list(range(i, j)) + list(range(j, N))
    return order

def apply_order(order):
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

def evaluate_order(order, name):
    \"\"\"Evaluate a custom layer order using full-model forward.\"\"\"
    apply_order(order)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# ======================================================================
# Baseline
# ======================================================================
print('\\n=== Baseline ===', flush=True)
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f}', flush=True)

# ======================================================================
# Reference: self-duplication on (20,21)
# ======================================================================
print('\\n=== Reference: Self-Duplication ===', flush=True)
ref_order = build_self_dup_order((20, 21), N)
r = evaluate_order(ref_order, 'REFERENCE: (20,21) self-dup')
r['type'] = 'self_dup_reference'
all_results.append(r)
self_dup_score = r['combined']

ref_order2 = build_self_dup_order((12, 13), N)
r2 = evaluate_order(ref_order2, 'REFERENCE: (12,13) self-dup')
r2['type'] = 'self_dup_reference'
all_results.append(r2)

# ======================================================================
# Experiment 1: First pass (20,21), second pass from various blocks
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Experiment 1: First pass (20,21), second pass from different blocks', flush=True)
print(f'{\"=\" * 70}', flush=True)

FIRST_BLOCK = (20, 21)
CROSS_CANDIDATES_1 = [
    (4, 5), (8, 9), (12, 13), (16, 17),
    (24, 25), (28, 29), (32, 33), (40, 41), (50, 51),
]

for cross in CROSS_CANDIDATES_1:
    order = build_cross_layer_order(FIRST_BLOCK, cross, N)
    name = f'(20,21)->({cross[0]},{cross[1]}) @1.0'
    r = evaluate_order(order, name)
    r['type'] = 'cross_layer'
    r['first_block'] = list(FIRST_BLOCK)
    r['second_weights'] = list(cross)
    r['delta_vs_self'] = r['combined'] - self_dup_score
    all_results.append(r)

# Find best cross-layer with (20,21) as first pass
exp1_cross = [r for r in all_results if r.get('type') == 'cross_layer' and r.get('first_block') == [20, 21]]
if exp1_cross:
    best_exp1 = max(exp1_cross, key=lambda x: x['combined'])
    print(f'\\nBest cross-layer from (20,21): {best_exp1[\"name\"]} = {best_exp1[\"combined\"]:.2f} (self-dup: {self_dup_score:.2f}, delta: {best_exp1[\"delta_vs_self\"]:+.2f})', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_exp1.json', 'w') as f:
    json.dump({'baseline': baseline, 'results': all_results}, f, indent=2)

# ======================================================================
# Experiment 2: First pass (12,13), second pass from various blocks
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Experiment 2: First pass (12,13), second pass from different blocks', flush=True)
print(f'{\"=\" * 70}', flush=True)

FIRST_BLOCK_2 = (12, 13)
self_dup_12_score = r2['combined']

CROSS_CANDIDATES_2 = [
    (4, 5), (8, 9), (16, 17), (20, 21),
    (24, 25), (28, 29), (32, 33), (40, 41), (50, 51),
]

for cross in CROSS_CANDIDATES_2:
    order = build_cross_layer_order(FIRST_BLOCK_2, cross, N)
    name = f'(12,13)->({cross[0]},{cross[1]}) @1.0'
    r = evaluate_order(order, name)
    r['type'] = 'cross_layer'
    r['first_block'] = list(FIRST_BLOCK_2)
    r['second_weights'] = list(cross)
    r['delta_vs_self'] = r['combined'] - self_dup_12_score
    all_results.append(r)

exp2_cross = [r for r in all_results if r.get('type') == 'cross_layer' and r.get('first_block') == [12, 13]]
if exp2_cross:
    best_exp2 = max(exp2_cross, key=lambda x: x['combined'])
    print(f'\\nBest cross-layer from (12,13): {best_exp2[\"name\"]} = {best_exp2[\"combined\"]:.2f} (self-dup: {self_dup_12_score:.2f}, delta: {best_exp2[\"delta_vs_self\"]:+.2f})', flush=True)

# ======================================================================
# Experiment 3: Reverse direction — does B->A beat A->B?
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Experiment 3: Reverse direction — does B->A beat A->B?', flush=True)
print(f'{\"=\" * 70}', flush=True)

# Take the top-3 cross configs from each experiment and test reverse
top_cross_1 = sorted(exp1_cross, key=lambda x: x['combined'], reverse=True)[:3] if exp1_cross else []
top_cross_2 = sorted(exp2_cross, key=lambda x: x['combined'], reverse=True)[:3] if exp2_cross else []

for r_orig in top_cross_1:
    first = tuple(r_orig['first_block'])
    second = tuple(r_orig['second_weights'])
    # Reverse: second block as first pass, first block's weights as second pass
    order = build_cross_layer_order(second, first, N)
    name = f'REVERSE: ({second[0]},{second[1]})->({first[0]},{first[1]}) @1.0'
    r = evaluate_order(order, name)
    r['type'] = 'cross_layer_reverse'
    r['first_block'] = list(second)
    r['second_weights'] = list(first)
    all_results.append(r)

for r_orig in top_cross_2:
    first = tuple(r_orig['first_block'])
    second = tuple(r_orig['second_weights'])
    order = build_cross_layer_order(second, first, N)
    name = f'REVERSE: ({second[0]},{second[1]})->({first[0]},{first[1]}) @1.0'
    r = evaluate_order(order, name)
    r['type'] = 'cross_layer_reverse'
    r['first_block'] = list(second)
    r['second_weights'] = list(first)
    all_results.append(r)

# ======================================================================
# Experiment 4: Cross-layer with 2-layer blocks
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Experiment 4: Cross-layer with 2-layer blocks', flush=True)
print(f'{\"=\" * 70}', flush=True)

FIRST_BLOCK_3 = (20, 22)  # 2-layer block
ref_order3 = build_self_dup_order(FIRST_BLOCK_3, N)
r3 = evaluate_order(ref_order3, 'REFERENCE: (20,22) self-dup')
r3['type'] = 'self_dup_reference'
all_results.append(r3)
self_dup_2layer_score = r3['combined']

CROSS_2LAYER = [
    (4, 6), (8, 10), (12, 14), (16, 18),
    (24, 26), (28, 30), (32, 34), (40, 42),
]

for cross in CROSS_2LAYER:
    order = build_cross_layer_order(FIRST_BLOCK_3, cross, N)
    name = f'(20,22)->({cross[0]},{cross[1]}) @1.0'
    r = evaluate_order(order, name)
    r['type'] = 'cross_layer_2wide'
    r['first_block'] = list(FIRST_BLOCK_3)
    r['second_weights'] = list(cross)
    r['delta_vs_self'] = r['combined'] - self_dup_2layer_score
    all_results.append(r)

# ======================================================================
# Final Summary
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('FINAL SUMMARY', flush=True)
print(f'{\"=\" * 70}', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Self-dup (20,21): {self_dup_score:.2f}', flush=True)
print(f'Self-dup (12,13): {self_dup_12_score:.2f}', flush=True)

# All cross-layer results
cross_only = [r for r in all_results if 'cross_layer' in r.get('type', '')]
self_dup_only = [r for r in all_results if r.get('type') == 'self_dup_reference']

if cross_only:
    best_cross = max(cross_only, key=lambda x: x['combined'])
    best_self = max(self_dup_only, key=lambda x: x['combined'])

    print(f'\\nBest self-dup: {best_self[\"name\"]} = {best_self[\"combined\"]:.2f}', flush=True)
    print(f'Best cross-layer: {best_cross[\"name\"]} = {best_cross[\"combined\"]:.2f}', flush=True)

    cross_beats_self = best_cross['combined'] > best_self['combined']
    print(f'\\nCross-layer beats self-duplication on Gemma3: {\"YES\" if cross_beats_self else \"NO\"}', flush=True)

    print(f'\\nTop 15 configs ranked by combined score:', flush=True)
    sorted_all = sorted(all_results, key=lambda x: x['combined'], reverse=True)
    for r in sorted_all[:15]:
        marker = '***' if 'cross_layer' in r.get('type', '') and r['combined'] > best_self['combined'] else '   '
        print(f'{marker} {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f}', flush=True)

# Save final results
output = {
    'date': datetime.now().isoformat(),
    'model': 'gemma-3-27b-it',
    'num_layers': N,
    'baseline': baseline,
    'self_dup_20_21': self_dup_score,
    'self_dup_12_13': self_dup_12_score,
    'cross_beats_self': cross_beats_self if cross_only else None,
    'results': all_results,
}
with open(f'{SAVE_DIR}/cross_layer_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/cross_layer_results.json', flush=True)
"

echo "=== Done at $(date) ==="
