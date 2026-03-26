#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_comp_%j.log
#SBATCH --job-name=deeppass_g3comp

# Comprehensive block search on Gemma3-27B (62 layers)
# Find absolute best configs at every depth:
#   1. Spectral screen ALL blocks step=1, sizes [1..7]
#   2. Dual probe top-15 singles
#   3. ALL non-overlapping pairs from top-15
#   4. Greedy stacking: triple, quad, quint, sext
# Known results to beat: pair=84.42, triple=85.43, quad=85.58

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3-27B Comprehensive Block Search ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, gc, torch, torch.nn as nn, numpy as np
from datetime import datetime
from itertools import combinations

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/comprehensive'
os.makedirs(SAVE_DIR, exist_ok=True)

print('=' * 70, flush=True)
print('GEMMA3-27B COMPREHENSIVE BLOCK SEARCH', flush=True)
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

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

def compute_rho(block):
    \"\"\"Displacement rho via full model forward passes (architecture-safe).\"\"\"
    i, j = block
    rhos = []
    for prompt in cal_prompts[:4]:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))

            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()

            inner.layers = nn.ModuleList(original_layers)
            set_num_layers(N)

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)
    return float(np.mean(rhos)) if rhos else 1.0

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

def evaluate(blocks, name):
    apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks],
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

def blocks_overlap(b1, b2):
    return not (b1[1] <= b2[0] or b2[1] <= b1[0])

all_results = {}

# ======================================================================
# Step 0: Baseline
# ======================================================================
print('\\n=== Step 0: Baseline ===', flush=True)
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f}', flush=True)
all_results['baseline'] = {'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline}

# ======================================================================
# Step 1: Spectral screen ALL blocks (step=1, sizes 1-7)
# ======================================================================
print('\\n=== Step 1: Comprehensive Spectral Screen (step=1, sizes 1-7) ===', flush=True)
candidates = []
for start in range(0, N):
    for size in [1, 2, 3, 4, 5, 6, 7]:
        end = start + size
        if end <= N:
            candidates.append((start, end))

print(f'Screening {len(candidates)} blocks...', flush=True)
block_rhos = {}
t_screen = time.time()
for idx, block in enumerate(candidates):
    block_rhos[block] = compute_rho(block)
    if (idx + 1) % 20 == 0:
        elapsed = time.time() - t_screen
        rate = (idx + 1) / elapsed * 60
        print(f'  [{idx+1}/{len(candidates)}] ({block[0]:2d},{block[1]:2d}) size={block[1]-block[0]}: rho={block_rhos[block]:.4f}  ({rate:.0f} blocks/min)', flush=True)

sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
print(f'\\nScreening done in {(time.time()-t_screen)/60:.1f} min', flush=True)
print('Top 20 blocks by rho (lowest = most stable):')
for b, r in sorted_blocks[:20]:
    print(f'  ({b[0]:2d},{b[1]:2d}) size={b[1]-b[0]}: rho={r:.6f}', flush=True)

all_results['spectral_screen'] = [{'block': list(b), 'size': b[1]-b[0], 'rho': r} for b, r in sorted_blocks]

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_spectral.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Checkpoint saved.', flush=True)

# ======================================================================
# Step 2: Evaluate top-15 singles with dual probe
# ======================================================================
print('\\n=== Step 2: Dual Probe Top-15 Singles ===', flush=True)
top15 = [b for b, r in sorted_blocks[:15]]
single_scores = {}

for block in top15:
    r = evaluate([block], f'single ({block[0]},{block[1]}) size={block[1]-block[0]}')
    single_scores[block] = r
    r['delta'] = r['combined'] - baseline

all_results['singles'] = [
    {**single_scores[b], 'rho': block_rhos[b], 'delta': single_scores[b]['combined'] - baseline}
    for b in top15
]

sorted_singles = sorted(single_scores.items(), key=lambda x: x[1]['combined'], reverse=True)
print('\\nSingles ranked by combined score:')
for b, r in sorted_singles:
    print(f'  ({b[0]:2d},{b[1]:2d}): combined={r[\"combined\"]:.2f} delta={r[\"combined\"]-baseline:+.2f}', flush=True)

best_single = sorted_singles[0]
print(f'\\nBest single: ({best_single[0][0]},{best_single[0][1]}) combined={best_single[1][\"combined\"]:.2f}', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_singles.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ======================================================================
# Step 3: ALL non-overlapping pairs from top-15
# ======================================================================
print('\\n=== Step 3: All Non-Overlapping Pairs from Top-15 ===', flush=True)
pair_results = []
pair_count = 0

for b1, b2 in combinations(top15, 2):
    if blocks_overlap(b1, b2):
        continue
    pair = sorted([b1, b2])
    name = '+'.join(f'({b[0]},{b[1]})' for b in pair)
    pair_count += 1

pair_combos = []
for b1, b2 in combinations(top15, 2):
    if not blocks_overlap(b1, b2):
        pair_combos.append(sorted([b1, b2]))

print(f'Testing {len(pair_combos)} non-overlapping pairs...', flush=True)

for idx, pair in enumerate(pair_combos):
    name = '+'.join(f'({b[0]},{b[1]})' for b in pair)
    r = evaluate(pair, f'pair {name}')
    r['delta'] = r['combined'] - baseline
    pair_results.append(r)
    if (idx + 1) % 10 == 0:
        best_so_far = max(pair_results, key=lambda x: x['combined'])
        print(f'  [{idx+1}/{len(pair_combos)}] best pair so far: {best_so_far[\"name\"]} = {best_so_far[\"combined\"]:.2f}', flush=True)

all_results['pairs'] = sorted(pair_results, key=lambda x: x['combined'], reverse=True)

sorted_pairs = sorted(pair_results, key=lambda x: x['combined'], reverse=True)
print('\\nTop 10 pairs:')
for r in sorted_pairs[:10]:
    print(f'  {r[\"name\"]:50s}: combined={r[\"combined\"]:.2f} delta={r[\"delta\"]:+.2f}', flush=True)

best_pair = sorted_pairs[0]
print(f'\\nBest pair: {best_pair[\"name\"]} combined={best_pair[\"combined\"]:.2f} (known best: 84.42)', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_pairs.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ======================================================================
# Step 4: Greedy stacking — find best third block for top-5 pairs
# ======================================================================
print('\\n=== Step 4: Greedy Stacking — Best Third Block for Top-5 Pairs ===', flush=True)
top5_pairs = sorted_pairs[:5]
triple_results = []

# Candidate third blocks: top-15 singles + extra coverage
third_candidates = list(top15)
# Add some extra blocks for coverage
for start in range(0, N, 2):
    for size in [1, 2]:
        block = (start, min(start + size, N))
        if block not in third_candidates and block[1] <= N:
            third_candidates.append(block)

for pair_r in top5_pairs:
    pair_blocks = [tuple(b) for b in pair_r['blocks']]
    pair_name = pair_r['name'].replace('pair ', '')
    print(f'\\n  Pair anchor: {pair_name} (combined={pair_r[\"combined\"]:.2f})', flush=True)

    tested_thirds = []
    for third in third_candidates:
        # Check non-overlap with both pair blocks
        if any(blocks_overlap(third, pb) for pb in pair_blocks):
            continue
        triple = sorted(list(pair_blocks) + [third])
        name = '+'.join(f'({b[0]},{b[1]})' for b in triple)
        r = evaluate(triple, f'triple {name}')
        r['delta_vs_pair'] = r['combined'] - pair_r['combined']
        r['delta_vs_baseline'] = r['combined'] - baseline
        tested_thirds.append(r)
        triple_results.append(r)

    if tested_thirds:
        best_third = max(tested_thirds, key=lambda x: x['combined'])
        print(f'  Best triple from {pair_name}: {best_third[\"name\"]} = {best_third[\"combined\"]:.2f} (delta vs pair: {best_third[\"delta_vs_pair\"]:+.2f})', flush=True)

all_results['triples'] = sorted(triple_results, key=lambda x: x['combined'], reverse=True)

sorted_triples = sorted(triple_results, key=lambda x: x['combined'], reverse=True)
print('\\nTop 10 triples:')
for r in sorted_triples[:10]:
    print(f'  {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f}', flush=True)

best_triple = sorted_triples[0] if sorted_triples else None
if best_triple:
    print(f'\\nBest triple: {best_triple[\"name\"]} combined={best_triple[\"combined\"]:.2f} (known best: 85.43)', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_triples.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ======================================================================
# Step 5: Best fourth block from best triple
# ======================================================================
print('\\n=== Step 5: Best Fourth Block from Best Triple ===', flush=True)
quad_results = []

if best_triple:
    triple_blocks = [tuple(b) for b in best_triple['blocks']]
    triple_name = best_triple['name'].replace('triple ', '')
    print(f'  Triple anchor: {triple_name} (combined={best_triple[\"combined\"]:.2f})', flush=True)

    for fourth in third_candidates:
        if any(blocks_overlap(fourth, tb) for tb in triple_blocks):
            continue
        quad = sorted(list(triple_blocks) + [fourth])
        name = '+'.join(f'({b[0]},{b[1]})' for b in quad)
        r = evaluate(quad, f'quad {name}')
        r['delta_vs_triple'] = r['combined'] - best_triple['combined']
        r['delta_vs_baseline'] = r['combined'] - baseline
        quad_results.append(r)

all_results['quads'] = sorted(quad_results, key=lambda x: x['combined'], reverse=True)

sorted_quads = sorted(quad_results, key=lambda x: x['combined'], reverse=True)
if sorted_quads:
    print('\\nTop 10 quads:')
    for r in sorted_quads[:10]:
        print(f'  {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f} delta_vs_triple={r[\"delta_vs_triple\"]:+.2f}', flush=True)

best_quad = sorted_quads[0] if sorted_quads else None
if best_quad:
    print(f'\\nBest quad: {best_quad[\"name\"]} combined={best_quad[\"combined\"]:.2f} (known best: 85.58)', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_quads.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# ======================================================================
# Step 6: Best fifth block from best quad (quint)
# ======================================================================
print('\\n=== Step 6: Quint — Best Fifth Block ===', flush=True)
quint_results = []

if best_quad:
    quad_blocks = [tuple(b) for b in best_quad['blocks']]
    quad_name = best_quad['name'].replace('quad ', '')
    print(f'  Quad anchor: {quad_name} (combined={best_quad[\"combined\"]:.2f})', flush=True)

    for fifth in third_candidates:
        if any(blocks_overlap(fifth, qb) for qb in quad_blocks):
            continue
        quint = sorted(list(quad_blocks) + [fifth])
        name = '+'.join(f'({b[0]},{b[1]})' for b in quint)
        r = evaluate(quint, f'quint {name}')
        r['delta_vs_quad'] = r['combined'] - best_quad['combined']
        r['delta_vs_baseline'] = r['combined'] - baseline
        quint_results.append(r)

all_results['quints'] = sorted(quint_results, key=lambda x: x['combined'], reverse=True)

sorted_quints = sorted(quint_results, key=lambda x: x['combined'], reverse=True)
best_quint = sorted_quints[0] if sorted_quints else None

if best_quint:
    print(f'\\nBest quint: {best_quint[\"name\"]} combined={best_quint[\"combined\"]:.2f}', flush=True)
    quint_improves = best_quint['combined'] > best_quad['combined'] if best_quad else False
    print(f'Quint improves over quad: {quint_improves}', flush=True)

    # ======================================================================
    # Step 7: Try sixth block if quint improves
    # ======================================================================
    if quint_improves:
        print('\\n=== Step 7: Sext — Best Sixth Block ===', flush=True)
        sext_results = []
        quint_blocks = [tuple(b) for b in best_quint['blocks']]
        quint_name = best_quint['name'].replace('quint ', '')
        print(f'  Quint anchor: {quint_name} (combined={best_quint[\"combined\"]:.2f})', flush=True)

        for sixth in third_candidates:
            if any(blocks_overlap(sixth, qb) for qb in quint_blocks):
                continue
            sext = sorted(list(quint_blocks) + [sixth])
            name = '+'.join(f'({b[0]},{b[1]})' for b in sext)
            r = evaluate(sext, f'sext {name}')
            r['delta_vs_quint'] = r['combined'] - best_quint['combined']
            r['delta_vs_baseline'] = r['combined'] - baseline
            sext_results.append(r)

        all_results['sexts'] = sorted(sext_results, key=lambda x: x['combined'], reverse=True)
        if sext_results:
            best_sext = max(sext_results, key=lambda x: x['combined'])
            print(f'Best sext: {best_sext[\"name\"]} combined={best_sext[\"combined\"]:.2f}', flush=True)
            print(f'Sext improves over quint: {best_sext[\"combined\"] > best_quint[\"combined\"]}', flush=True)
    else:
        print('\\nQuint did not improve over quad. Skipping sixth block search.', flush=True)
else:
    print('No quint results. Skipping further stacking.', flush=True)

# ======================================================================
# Final Summary
# ======================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('FINAL SUMMARY', flush=True)
print(f'{\"=\" * 70}', flush=True)
print(f'Model: Gemma3-27B ({N} layers)', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Best single: ({best_single[0][0]},{best_single[0][1]}) = {best_single[1][\"combined\"]:.2f} delta={best_single[1][\"combined\"]-baseline:+.2f}', flush=True)
if best_pair:
    print(f'Best pair:   {best_pair[\"name\"]} = {best_pair[\"combined\"]:.2f} delta={best_pair[\"combined\"]-baseline:+.2f} (known: 84.42)', flush=True)
if best_triple:
    print(f'Best triple: {best_triple[\"name\"]} = {best_triple[\"combined\"]:.2f} delta={best_triple[\"combined\"]-baseline:+.2f} (known: 85.43)', flush=True)
if best_quad:
    print(f'Best quad:   {best_quad[\"name\"]} = {best_quad[\"combined\"]:.2f} delta={best_quad[\"combined\"]-baseline:+.2f} (known: 85.58)', flush=True)
if best_quint:
    print(f'Best quint:  {best_quint[\"name\"]} = {best_quint[\"combined\"]:.2f} delta={best_quint[\"combined\"]-baseline:+.2f}', flush=True)
if 'sexts' in all_results and all_results['sexts']:
    best_sext = all_results['sexts'][0]
    print(f'Best sext:   {best_sext[\"name\"]} = {best_sext[\"combined\"]:.2f} delta={best_sext[\"combined\"]-baseline:+.2f}', flush=True)

# Progression
print(f'\\nProgression:', flush=True)
scores = [baseline]
labels = ['baseline']
if best_single:
    scores.append(best_single[1]['combined']); labels.append('single')
if best_pair:
    scores.append(best_pair['combined']); labels.append('pair')
if best_triple:
    scores.append(best_triple['combined']); labels.append('triple')
if best_quad:
    scores.append(best_quad['combined']); labels.append('quad')
if best_quint:
    scores.append(best_quint['combined']); labels.append('quint')
for i, (label, score) in enumerate(zip(labels, scores)):
    delta = f' (+{score - scores[i-1]:.2f})' if i > 0 else ''
    print(f'  {label:10s}: {score:.2f}{delta}', flush=True)

# Save final results
all_results['summary'] = {
    'baseline': baseline,
    'best_single': {'block': list(best_single[0]), 'combined': best_single[1]['combined']} if best_single else None,
    'best_pair': best_pair,
    'best_triple': best_triple,
    'best_quad': best_quad,
    'best_quint': best_quint if best_quint else None,
    'date': datetime.now().isoformat(),
}

with open(f'{SAVE_DIR}/comprehensive_search.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/comprehensive_search.json', flush=True)
"

echo "=== Done at $(date) ==="
