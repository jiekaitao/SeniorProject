#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_eq_sweep_72b_%j.log
#SBATCH --job-name=deeppass_eq_sweep

# Run EQ-bench on key 72B single-block configs we're missing EQ scores for.
# We have EQ-bench for: baseline, (45,52), (50,60), (0,7)+(45,52), (15,20)+(50,60)
# We're MISSING: (0,7) alone, (15,20) alone — needed for epistasis calculation

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== EQ-bench on missing 72B single configs ==="
echo "Started: $(date)"

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

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Configs to test (ones we're missing EQ-bench for)
configs = [
    ('(0,7)', [(0, 7)]),
    ('(15,20)', [(15, 20)]),
    ('(35,40)', [(35, 40)]),
    ('(55,60)', [(55, 60)]),
    ('(45,52)+(55,60)', [(45, 52), (55, 60)]),
]

results = []
for name, blocks in configs:
    if blocks:
        sorted_blocks = sorted(blocks)
        order = []
        prev = 0
        for (i, j) in sorted_blocks:
            order.extend(list(range(prev, j)))
            order.extend(list(range(i, j)))
            prev = j
        order.extend(list(range(prev, N)))
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    print(f'{name:25s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}')
    results.append({'name': name, 'blocks': [list(b) for b in blocks], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

    if blocks:
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

os.makedirs('results/data/72b/singles', exist_ok=True)
with open('results/data/72b/singles/missing_eq_bench.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved')
"

echo "=== Done at $(date) ==="
