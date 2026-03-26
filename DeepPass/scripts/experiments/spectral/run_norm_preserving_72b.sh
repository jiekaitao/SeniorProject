#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_norm_pres_72b_%j.log
#SBATCH --job-name=deeppass_normpres

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Norm-Preserving on 72B ==="
echo "Started: $(date)"

# Reuse the existing script but override model path and output path
# Test on (45,52) — Ng's config — and (50,60) — our best single
$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
import numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/experiments/spectral')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe
from norm_preserving_test import (
    NORM_VARIANTS, build_layer_order, find_seam_positions,
    generate_with_norm_intervention, evaluate_variant,
    evaluate_baseline_no_dup, analyze_norms_at_seam
)

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCKS_TO_TEST = [(45, 52), (50, 60)]

print('=' * 70)
print('NORM-PRESERVING PROJECTION — 72B')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}')

inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

# Baseline
print('\n  [Baseline — no duplication]')
baseline_r = evaluate_baseline_no_dup(model, tokenizer)
baseline_combined = baseline_r['combined']

all_block_results = {}

for block in BLOCKS_TO_TEST:
    print(f'\n{\"=\" * 70}')
    print(f'BLOCK ({block[0]},{block[1]})')
    print('=' * 70)

    # Norm analysis
    print('\n  [Norm analysis at seam]')
    CAL_PROMPTS = [
        'What is 127 * 348?', 'What is 99999 * 99999?',
        'Calculate 15! / 13!', 'What is 2^16?',
        'What is the sum of all integers from 1 to 100?',
        'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
        'What emotion would someone feel after losing a close friend?',
        'How would a parent feel seeing their child graduate?',
    ]
    norm_stats = analyze_norms_at_seam(inner, original_layers, device, tokenizer, block, CAL_PROMPTS)
    for k, v in norm_stats.items():
        print(f'    {k:20s}: mean={v[\"mean\"]:.4f} ratio={v.get(\"mean\",0):.4f}')

    layer_order = build_layer_order(block, N)
    results = [baseline_r]

    # alpha=0
    print('\n  [alpha=0]')
    r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers, block, 'alpha0', 'alpha=0 (h1 only)')
    results.append(r)

    # alpha=1
    print('\n  [alpha=1 — standard]')
    r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers, block, None, 'alpha=1 (h2 only, standard)')
    results.append(r)

    # Norm variants
    print('\n  [Norm variants]')
    for vname, vfn in NORM_VARIANTS.items():
        r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers, block, vfn, vname)
        results.append(r)

    # Summary for this block
    sorted_r = sorted(results, key=lambda x: x['combined'], reverse=True)
    print(f'\n  --- Block ({block[0]},{block[1]}) Summary ---')
    for r in sorted_r:
        delta = r['combined'] - baseline_combined
        print(f'    {r[\"name\"]:40s}: combined={r[\"combined\"]:.2f} delta={delta:+.2f}')

    all_block_results[f'({block[0]},{block[1]})'] = {
        'block': list(block),
        'norm_analysis': norm_stats,
        'results': results,
        'best_variant': sorted_r[0]['name'],
        'best_combined': sorted_r[0]['combined'],
    }

# Save
os.makedirs('results/data/72b/norm_preserving', exist_ok=True)
output = {
    'model': MODEL_PATH,
    'num_layers': N,
    'date': datetime.now().isoformat(),
    'baseline': baseline_r,
    'blocks': all_block_results,
}
with open('results/data/72b/norm_preserving/norm_preserving_72b.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\nSaved to results/data/72b/norm_preserving/norm_preserving_72b.json')
"

echo "=== Done at $(date) ==="
