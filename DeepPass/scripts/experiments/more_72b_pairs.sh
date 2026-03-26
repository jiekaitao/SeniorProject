#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_more_pairs_%j.log
#SBATCH --job-name=deeppass_more_pairs

# Evaluate 10 more 72B pairs for DICE validation
# Mix of predicted-good, predicted-bad, and coverage sampling
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

def build_order(blocks, N):
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))
    return order

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Pairs to test (mix of regions and distances)
# We already have: (0,7)+(45,52), (15,20)+(50,60), (0,7)+(50,60), (15,20)+(45,52)
# And same-region pairs from 72b_pair_sweep.json
# Need: cross-region pairs we haven't tested
test_pairs = [
    # Near pairs (same corridor hypothesis)
    [(45, 52), (52, 60)],   # adjacent deep
    [(40, 47), (50, 57)],   # overlapping deep region
    # Medium distance
    [(20, 27), (45, 52)],   # mid + Ng's
    [(10, 17), (50, 60)],   # early-mid + our best
    [(25, 32), (50, 60)],   # mid + our best
    # Far cross-region
    [(0, 7), (55, 62)],     # very early + very deep
    [(5, 12), (45, 52)],    # early + Ng's
    [(0, 7), (30, 37)],     # early + mid
    # Weak block pairs (DICE predicts these might stack)
    [(10, 17), (30, 37)],   # two medium blocks
    [(20, 27), (55, 62)],   # mid + deep
]

results = []
for blocks in test_pairs:
    # Check non-overlapping
    if blocks[0][1] > blocks[1][0]:
        blocks = [blocks[1], blocks[0]]
    if blocks[0][1] > blocks[1][0]:
        print(f'  Skipping {blocks} — overlapping')
        continue

    name = '+'.join(f'({b[0]},{b[1]})' for b in blocks)
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    print(f'  {name:30s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}')
    results.append({
        'blocks': [list(b) for b in blocks],
        'name': name,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined
    })

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

os.makedirs('results/data/72b/pairs', exist_ok=True)
with open('results/data/72b/pairs/more_pairs_round2.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved')
"
