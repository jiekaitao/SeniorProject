#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_mega_%j.log
#SBATCH --job-name=deeppass_g3mega

# Mega stacking on Gemma3-27B (62 layers)
# Test extreme stacking depths (5-10 blocks) since Gemma3 handles alpha=1.0.
#
# Strategy:
# 1. Start from known best quad: (4,5)+(12,13)+(16,17)+(20,21) = 85.58
# 2. Greedy 5th, 6th, 7th... blocks until diminishing returns
# 3. Alternative anchor: build from (12,13) instead of (20,21)
# 4. "Every other layer" strategy: 6 evenly-spread single-layer blocks
#
# IMPORTANT: Gemma3 manual layer loop DOES NOT WORK (sliding window attention).
# All experiments use full-model forward with ModuleList swapping.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3-27B Mega Stacking ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/mega_stacking'
os.makedirs(SAVE_DIR, exist_ok=True)

print('=' * 70, flush=True)
print('GEMMA3-27B MEGA STACKING (5-10 blocks)', flush=True)
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

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def apply_blocks(blocks):
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

def blocks_overlap(b1, b2):
    return not (b1[1] <= b2[0] or b2[1] <= b1[0])

def evaluate(blocks, name):
    apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()
    n_extra = sum(b[1] - b[0] for b in blocks)
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} +{n_extra}layers ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks],
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined,
            'extra_layers': n_extra, 'total_layers': N + n_extra}

all_results = {}

# ======================================================================
# Baseline
# ======================================================================
print('\\n=== Baseline ===', flush=True)
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f}', flush=True)
all_results['baseline'] = {'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline}

# ======================================================================
# Known best configs (verify)
# ======================================================================
print('\\n=== Verify Known Best Configs ===', flush=True)
KNOWN_QUAD = [(4, 5), (12, 13), (16, 17), (20, 21)]
r_quad = evaluate(KNOWN_QUAD, 'KNOWN quad (4,5)+(12,13)+(16,17)+(20,21)')
all_results['known_quad'] = r_quad
print(f'Known quad: {r_quad[\"combined\"]:.2f} (expected ~85.58)', flush=True)

# ======================================================================
# Phase 1: Greedy 5th block from best quad
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Phase 1: Add 5th block to known quad', flush=True)
print(f'{\"=\" * 70}', flush=True)

FIFTH_CANDIDATES = [
    (8, 9), (24, 25), (28, 29),
    (6, 7), (10, 11), (14, 15), (18, 19),
    (22, 23), (26, 27), (30, 31),
]

quint_results = []
for fifth in FIFTH_CANDIDATES:
    if any(blocks_overlap(fifth, qb) for qb in KNOWN_QUAD):
        print(f'  Skipping ({fifth[0]},{fifth[1]}): overlaps with quad', flush=True)
        continue
    quint = sorted(list(KNOWN_QUAD) + [fifth])
    name = '+'.join(f'({b[0]},{b[1]})' for b in quint)
    r = evaluate(quint, f'quint {name}')
    r['added_block'] = list(fifth)
    r['delta_vs_quad'] = r['combined'] - r_quad['combined']
    quint_results.append(r)

all_results['quints'] = sorted(quint_results, key=lambda x: x['combined'], reverse=True)

best_quint = max(quint_results, key=lambda x: x['combined']) if quint_results else None
if best_quint:
    print(f'\\nBest quint: {best_quint[\"name\"]} = {best_quint[\"combined\"]:.2f} (delta vs quad: {best_quint[\"delta_vs_quad\"]:+.2f})', flush=True)
    quint_improves = best_quint['combined'] > r_quad['combined']
    print(f'Quint improves over quad: {quint_improves}', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_quint.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ======================================================================
# Phase 2: If quint improves, add 6th block
# ======================================================================
sext_results = []
best_sext = None

if best_quint and best_quint['combined'] > r_quad['combined']:
    print(f'\\n{\"=\" * 70}', flush=True)
    print('Phase 2: Add 6th block to best quint', flush=True)
    print(f'{\"=\" * 70}', flush=True)

    quint_blocks = [tuple(b) for b in best_quint['blocks']]

    SIXTH_CANDIDATES = []
    for start in range(0, N):
        block = (start, start + 1)
        if block[1] <= N and not any(blocks_overlap(block, qb) for qb in quint_blocks):
            SIXTH_CANDIDATES.append(block)
    # Also try some 2-layer blocks
    for start in range(0, N - 1, 2):
        block = (start, start + 2)
        if block[1] <= N and not any(blocks_overlap(block, qb) for qb in quint_blocks):
            SIXTH_CANDIDATES.append(block)

    print(f'Testing {len(SIXTH_CANDIDATES)} 6th block candidates...', flush=True)
    for sixth in SIXTH_CANDIDATES:
        sext = sorted(list(quint_blocks) + [sixth])
        name = '+'.join(f'({b[0]},{b[1]})' for b in sext)
        r = evaluate(sext, f'sext {name}')
        r['added_block'] = list(sixth)
        r['delta_vs_quint'] = r['combined'] - best_quint['combined']
        sext_results.append(r)

    all_results['sexts'] = sorted(sext_results, key=lambda x: x['combined'], reverse=True)
    best_sext = max(sext_results, key=lambda x: x['combined']) if sext_results else None

    if best_sext:
        print(f'\\nBest sext: {best_sext[\"name\"]} = {best_sext[\"combined\"]:.2f} (delta vs quint: {best_sext[\"delta_vs_quint\"]:+.2f})', flush=True)

    with open(f'{SAVE_DIR}/checkpoint_sext.json', 'w') as f:
        json.dump(all_results, f, indent=2)
else:
    print('\\nQuint did not improve. Skipping 6th block.', flush=True)

# ======================================================================
# Phase 3: If 6 blocks improves, try 7th
# ======================================================================
sept_results = []
best_sept = None

if best_sext and best_sext['combined'] > best_quint['combined']:
    print(f'\\n{\"=\" * 70}', flush=True)
    print('Phase 3: Add 7th block to best sext', flush=True)
    print(f'{\"=\" * 70}', flush=True)

    sext_blocks = [tuple(b) for b in best_sext['blocks']]

    SEVENTH_CANDIDATES = []
    for start in range(0, N):
        block = (start, start + 1)
        if block[1] <= N and not any(blocks_overlap(block, sb) for sb in sext_blocks):
            SEVENTH_CANDIDATES.append(block)

    print(f'Testing {len(SEVENTH_CANDIDATES)} 7th block candidates...', flush=True)
    for seventh in SEVENTH_CANDIDATES:
        sept = sorted(list(sext_blocks) + [seventh])
        name = '+'.join(f'({b[0]},{b[1]})' for b in sept)
        r = evaluate(sept, f'sept {name}')
        r['added_block'] = list(seventh)
        r['delta_vs_sext'] = r['combined'] - best_sext['combined']
        sept_results.append(r)

    all_results['septs'] = sorted(sept_results, key=lambda x: x['combined'], reverse=True)
    best_sept = max(sept_results, key=lambda x: x['combined']) if sept_results else None

    if best_sept:
        print(f'\\nBest sept: {best_sept[\"name\"]} = {best_sept[\"combined\"]:.2f} (delta vs sext: {best_sept[\"delta_vs_sext\"]:+.2f})', flush=True)

    # Continue to 8th if improving
    if best_sept and best_sept['combined'] > best_sext['combined']:
        print(f'\\n{\"=\" * 70}', flush=True)
        print('Phase 3b: Add 8th block to best sept', flush=True)
        print(f'{\"=\" * 70}', flush=True)

        sept_blocks = [tuple(b) for b in best_sept['blocks']]
        oct_results = []

        EIGHTH_CANDIDATES = []
        for start in range(0, N):
            block = (start, start + 1)
            if block[1] <= N and not any(blocks_overlap(block, sb) for sb in sept_blocks):
                EIGHTH_CANDIDATES.append(block)

        print(f'Testing {len(EIGHTH_CANDIDATES)} 8th block candidates...', flush=True)
        for eighth in EIGHTH_CANDIDATES:
            octet = sorted(list(sept_blocks) + [eighth])
            name = '+'.join(f'({b[0]},{b[1]})' for b in octet)
            r = evaluate(octet, f'oct {name}')
            r['added_block'] = list(eighth)
            r['delta_vs_sept'] = r['combined'] - best_sept['combined']
            oct_results.append(r)

        all_results['octs'] = sorted(oct_results, key=lambda x: x['combined'], reverse=True)
        if oct_results:
            best_oct = max(oct_results, key=lambda x: x['combined'])
            print(f'\\nBest oct: {best_oct[\"name\"]} = {best_oct[\"combined\"]:.2f} (delta vs sept: {best_oct[\"delta_vs_sept\"]:+.2f})', flush=True)

    with open(f'{SAVE_DIR}/checkpoint_deep.json', 'w') as f:
        json.dump(all_results, f, indent=2)
else:
    if best_sext:
        print(f'\\nSext did not improve over quint. Stopping greedy stacking.', flush=True)
    else:
        print(f'\\nNo sext results. Stopping greedy stacking.', flush=True)

# ======================================================================
# Phase 4: Alternative anchor — build from (12,13) instead of (20,21)
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Phase 4: Alternative anchor — greedy stacking from (12,13)', flush=True)
print(f'{\"=\" * 70}', flush=True)

alt_anchor = (12, 13)
r_alt_single = evaluate([alt_anchor], f'alt single ({alt_anchor[0]},{alt_anchor[1]})')
all_results['alt_anchor_single'] = r_alt_single

# Greedy second block
print('\\n  Searching for best second block...', flush=True)
alt_second_results = []
for start in range(0, N):
    for size in [1, 2]:
        block = (start, min(start + size, N))
        if block[1] <= N and not blocks_overlap(block, alt_anchor):
            pair = sorted([alt_anchor, block])
            name = '+'.join(f'({b[0]},{b[1]})' for b in pair)
            r = evaluate(pair, f'alt pair {name}')
            r['added_block'] = list(block)
            r['delta_vs_single'] = r['combined'] - r_alt_single['combined']
            alt_second_results.append(r)

best_alt_pair = max(alt_second_results, key=lambda x: x['combined']) if alt_second_results else None
all_results['alt_pairs'] = sorted(alt_second_results, key=lambda x: x['combined'], reverse=True)[:20]

if best_alt_pair:
    print(f'\\nBest alt pair: {best_alt_pair[\"name\"]} = {best_alt_pair[\"combined\"]:.2f}', flush=True)

    # Greedy third
    alt_pair_blocks = [tuple(b) for b in best_alt_pair['blocks']]
    print('  Searching for best third block...', flush=True)
    alt_third_results = []
    for start in range(0, N):
        block = (start, start + 1)
        if block[1] <= N and not any(blocks_overlap(block, pb) for pb in alt_pair_blocks):
            triple = sorted(list(alt_pair_blocks) + [block])
            name = '+'.join(f'({b[0]},{b[1]})' for b in triple)
            r = evaluate(triple, f'alt triple {name}')
            r['delta_vs_pair'] = r['combined'] - best_alt_pair['combined']
            alt_third_results.append(r)

    best_alt_triple = max(alt_third_results, key=lambda x: x['combined']) if alt_third_results else None
    all_results['alt_triples'] = sorted(alt_third_results, key=lambda x: x['combined'], reverse=True)[:20]

    if best_alt_triple:
        print(f'Best alt triple: {best_alt_triple[\"name\"]} = {best_alt_triple[\"combined\"]:.2f}', flush=True)

        # Greedy fourth
        alt_triple_blocks = [tuple(b) for b in best_alt_triple['blocks']]
        print('  Searching for best fourth block...', flush=True)
        alt_fourth_results = []
        for start in range(0, N):
            block = (start, start + 1)
            if block[1] <= N and not any(blocks_overlap(block, tb) for tb in alt_triple_blocks):
                quad = sorted(list(alt_triple_blocks) + [block])
                name = '+'.join(f'({b[0]},{b[1]})' for b in quad)
                r = evaluate(quad, f'alt quad {name}')
                r['delta_vs_triple'] = r['combined'] - best_alt_triple['combined']
                alt_fourth_results.append(r)

        best_alt_quad = max(alt_fourth_results, key=lambda x: x['combined']) if alt_fourth_results else None
        all_results['alt_quads'] = sorted(alt_fourth_results, key=lambda x: x['combined'], reverse=True)[:20]
        if best_alt_quad:
            print(f'Best alt quad: {best_alt_quad[\"name\"]} = {best_alt_quad[\"combined\"]:.2f}', flush=True)

with open(f'{SAVE_DIR}/checkpoint_alt_anchor.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ======================================================================
# Phase 5: "Every other layer" strategy
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Phase 5: Every-Other-Layer Strategy', flush=True)
print(f'{\"=\" * 70}', flush=True)

# 6 single-layer blocks spread evenly across the model
evenly_spread = [(4, 5), (8, 9), (12, 13), (16, 17), (20, 21), (24, 25)]
r_even6 = evaluate(evenly_spread, 'even6: (4,5)+(8,9)+(12,13)+(16,17)+(20,21)+(24,25)')
all_results['even_6blocks'] = r_even6

# Wider spread
wider_spread = [(4, 5), (12, 13), (20, 21), (28, 29), (36, 37), (44, 45)]
r_wide6 = evaluate(wider_spread, 'wide6: (4,5)+(12,13)+(20,21)+(28,29)+(36,37)+(44,45)')
all_results['wide_6blocks'] = r_wide6

# Dense early layers
dense_early = [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
r_dense = evaluate(dense_early, 'dense_early: (4,5)+(6,7)+(8,9)+(10,11)+(12,13)+(14,15)')
all_results['dense_early_6blocks'] = r_dense

# Dense around best region (12-24)
dense_mid = [(12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23)]
r_dmid = evaluate(dense_mid, 'dense_mid: (12,13)+(14,15)+(16,17)+(18,19)+(20,21)+(22,23)')
all_results['dense_mid_6blocks'] = r_dmid

# 8-block mega stack
mega8 = [(4, 5), (8, 9), (12, 13), (16, 17), (20, 21), (24, 25), (28, 29), (32, 33)]
r_mega8 = evaluate(mega8, 'mega8: 8 blocks every-4')
all_results['mega_8blocks'] = r_mega8

# 10-block extreme
mega10 = [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (16, 17), (20, 21), (24, 25), (28, 29), (32, 33)]
r_mega10 = evaluate(mega10, 'mega10: 10 blocks')
all_results['mega_10blocks'] = r_mega10

# ======================================================================
# Phase 6: Vary block sizes in mega stacks
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('Phase 6: Varying block sizes in mega stacks', flush=True)
print(f'{\"=\" * 70}', flush=True)

# 3 x 2-layer blocks
three_2layer = [(4, 6), (12, 14), (20, 22)]
r_3x2 = evaluate(three_2layer, '3x2layer: (4,6)+(12,14)+(20,22)')
all_results['3x2layer'] = r_3x2

# 2 x 3-layer blocks
two_3layer = [(12, 15), (20, 23)]
r_2x3 = evaluate(two_3layer, '2x3layer: (12,15)+(20,23)')
all_results['2x3layer'] = r_2x3

# 4 x 2-layer blocks
four_2layer = [(4, 6), (12, 14), (20, 22), (28, 30)]
r_4x2 = evaluate(four_2layer, '4x2layer: (4,6)+(12,14)+(20,22)+(28,30)')
all_results['4x2layer'] = r_4x2

# Mixed: 2-layer core + 1-layer support
mixed = [(4, 5), (12, 14), (20, 22), (28, 29)]
r_mixed = evaluate(mixed, 'mixed: (4,5)+(12,14)+(20,22)+(28,29)')
all_results['mixed'] = r_mixed

# ======================================================================
# Final Summary
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('FINAL SUMMARY', flush=True)
print(f'{\"=\" * 70}', flush=True)
print(f'Model: Gemma3-27B ({N} layers)', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Known quad: {r_quad[\"combined\"]:.2f} (target to beat: 85.58)', flush=True)

# Collect all evaluated configs with their scores
all_scored = []
for key, val in all_results.items():
    if isinstance(val, dict) and 'combined' in val:
        all_scored.append(val)
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, dict) and 'combined' in item:
                all_scored.append(item)

# Sort by combined score
all_scored_sorted = sorted(all_scored, key=lambda x: x.get('combined', 0), reverse=True)

print(f'\\nTop 20 configs overall:', flush=True)
for r in all_scored_sorted[:20]:
    n_blocks = len(r.get('blocks', [])) if 'blocks' in r else 0
    extra = r.get('extra_layers', '?')
    print(f'  {r.get(\"name\", \"?\"):65s}: combined={r[\"combined\"]:.2f} ({n_blocks} blocks, +{extra} layers)', flush=True)

# Stacking progression
print(f'\\nStacking depth analysis:', flush=True)
depth_best = {}
for r in all_scored:
    if 'blocks' in r:
        depth = len(r['blocks'])
        if depth not in depth_best or r['combined'] > depth_best[depth]['combined']:
            depth_best[depth] = r

for depth in sorted(depth_best.keys()):
    r = depth_best[depth]
    delta = r['combined'] - baseline
    print(f'  {depth} blocks: {r.get(\"name\", \"?\"):60s} combined={r[\"combined\"]:.2f} delta={delta:+.2f}', flush=True)

# Key question: at what depth does performance peak?
if depth_best:
    peak_depth = max(depth_best.items(), key=lambda x: x[1]['combined'])
    print(f'\\nPeak depth: {peak_depth[0]} blocks (combined={peak_depth[1][\"combined\"]:.2f})', flush=True)
    print(f'Extra layers at peak: +{peak_depth[1].get(\"extra_layers\", \"?\")} (total: {peak_depth[1].get(\"total_layers\", \"?\")})', flush=True)

# Save final results
all_results['summary'] = {
    'date': datetime.now().isoformat(),
    'baseline': baseline,
    'known_quad_score': r_quad['combined'],
    'peak_depth': peak_depth[0] if depth_best else None,
    'peak_score': peak_depth[1]['combined'] if depth_best else None,
    'depth_progression': {str(d): r['combined'] for d, r in sorted(depth_best.items())},
}

with open(f'{SAVE_DIR}/mega_stacking_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/mega_stacking_results.json', flush=True)
"

echo "=== Done at $(date) ==="
