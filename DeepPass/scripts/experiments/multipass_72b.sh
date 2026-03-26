#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_multipass_72b_%j.log
#SBATCH --job-name=deeppass_multipass

# Multi-pass test on 72B: does running (45,52) 3x or 4x beat 2x?
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

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Test multi-pass: run block (45,52) N times
block = (45, 52)
results = []

for n_passes in [1, 2, 3, 4]:
    # Build order: layers before block + block repeated n_passes times + layers after
    order = list(range(block[1]))  # 0..51
    for _ in range(n_passes - 1):  # extra passes (1 already in the order)
        order.extend(list(range(block[0], block[1])))
    order.extend(list(range(block[1], N)))

    if n_passes == 1:
        # Just use original model
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N
    else:
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    total_layers = len(order) if n_passes > 1 else N

    print(f'  (45,52) x{n_passes} ({total_layers} layers): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}')
    results.append({
        'block': list(block), 'passes': n_passes, 'total_layers': total_layers,
        'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined
    })

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

# Also test (50,60) multi-pass
block2 = (50, 60)
for n_passes in [2, 3]:
    order = list(range(block2[1]))
    for _ in range(n_passes - 1):
        order.extend(list(range(block2[0], block2[1])))
    order.extend(list(range(block2[1], N)))

    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    total_layers = len(order)

    print(f'  (50,60) x{n_passes} ({total_layers} layers): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}')
    results.append({
        'block': list(block2), 'passes': n_passes, 'total_layers': total_layers,
        'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined
    })

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

os.makedirs('results/data/72b/multipass', exist_ok=True)
with open('results/data/72b/multipass/multipass_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved')
"
