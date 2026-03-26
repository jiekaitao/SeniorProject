#!/bin/bash
# Overnight experiment queue — run sequentially after 7B sweeps complete
# Designed to maximize GPU utilization over ~10 hours

module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
cd /blue/cis4914/jietao/DeepPass

echo "=============================================="
echo "OVERNIGHT QUEUE — Started at $(date)"
echo "=============================================="

# Phase 1: 72B pairwise stacking (the Ng-beating experiment)
echo ""
echo "=== Phase 1: 72B Pair Sweep — Can we beat Ng? ==="
echo "Started at $(date)"
python -c "
import sys, torch, torch.nn as nn, copy, json
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache, apply_layer_duplication
from math_probe import run_math_probe

model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
device = next(model.parameters()).device
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)

# Baseline
baseline = run_math_probe(gen, verbose=False)
print(f'Baseline: {baseline[\"score\"]:.4f}')

# (45,52) alone — Ng's config
order_ng = list(range(52)) + list(range(45, 52)) + list(range(52, N))
inner.layers = nn.ModuleList([original_layers[i] for i in order_ng])
model.config.num_hidden_layers = len(order_ng)
r_ng = run_math_probe(gen, verbose=False)
print(f'Ng (45,52) alone: {r_ng[\"score\"]:.4f} ({r_ng[\"score\"]-baseline[\"score\"]:+.4f})')
inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

# (50,60) alone — our config
order_ours = list(range(60)) + list(range(50, 60)) + list(range(60, N))
inner.layers = nn.ModuleList([original_layers[i] for i in order_ours])
model.config.num_hidden_layers = len(order_ours)
r_ours = run_math_probe(gen, verbose=False)
print(f'Ours (50,60) alone: {r_ours[\"score\"]:.4f} ({r_ours[\"score\"]-baseline[\"score\"]:+.4f})')
inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

# Test key pairs in the 45-60 region
test_pairs = [
    ((45, 52), (55, 60)),  # Ng + deeper
    ((45, 52), (60, 65)),  # Ng + suffix region
    ((45, 52), (35, 40)),  # Ng + prefix region
    ((50, 60), (35, 45)),  # Ours + earlier
    ((50, 60), (60, 65)),  # Ours + suffix
    ((45, 50), (55, 60)),  # Split Ng's region into two
    ((40, 45), (50, 55)),  # Earlier + middle
    ((45, 52), (52, 58)),  # Ng + adjacent (like extending)
]

results = []
for (a_i, a_j), (b_i, b_j) in test_pairs:
    if a_j > N or b_j > N:
        continue
    # Check non-overlapping
    if not (a_j <= b_i or b_j <= a_i):
        print(f'  SKIP ({a_i},{a_j})+({b_i},{b_j}) — overlapping')
        continue

    blocks = sorted([(a_i, a_j), (b_i, b_j)])
    order = []
    prev = 0
    for (i, j) in blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))

    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)
    r = run_math_probe(gen, verbose=False)

    result = {'a': [a_i,a_j], 'b': [b_i,b_j], 'score': r['score'], 'delta': r['score']-baseline['score']}
    results.append(result)
    beats_ng = ' ***BEATS NG***' if r['score'] > r_ng['score'] else ''
    print(f'  ({a_i},{a_j})+({b_i},{b_j}): {r[\"score\"]:.4f} ({r[\"score\"]-baseline[\"score\"]:+.4f}){beats_ng}')

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

with open('results/72b_pair_sweep.json', 'w') as f:
    json.dump({'baseline': baseline['score'], 'ng_score': r_ng['score'], 'our_score': r_ours['score'], 'pairs': results}, f, indent=2)
print('Saved to results/72b_pair_sweep.json')
" 2>&1 | tee results/72b_pair_sweep.log

echo "Phase 1 done at $(date)"

# Phase 2: Greedy stacking on 72B — spectral screen then probe
echo ""
echo "=== Phase 2: 72B Greedy Stacking ==="
echo "Started at $(date)"
python scripts/experiments/spectral/corrected_pipeline.py \
    --model models/full/calme-2.1-qwen2-72b \
    --top-k 5 \
    --max-iterations 2 \
    --step 3 \
    --block-sizes "5,7,10" 2>&1 | tee results/72b_corrected_pipeline.log

echo "Phase 2 done at $(date)"

echo ""
echo "=============================================="
echo "ALL DONE at $(date)"
echo "=============================================="
