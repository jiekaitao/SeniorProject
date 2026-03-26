#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_direction_72b_%j.log
#SBATCH --job-name=deeppass_dir72b

# Direction-aware seam interventions on 72B
# Motivated by: 72B norm_ratio=1.04, cosine=0.997
# The 0.3% direction change is where all the action is

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Direction-Aware Interventions on 72B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/experiments/spectral')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe
from norm_preserving_test import (
    build_layer_order, find_seam_positions,
    generate_with_norm_intervention, evaluate_variant,
    evaluate_baseline_no_dup
)

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCKS = [(45, 52), (50, 60)]

# =====================================================================
# Direction intervention functions
# =====================================================================

def alpha_overshoot(alpha):
    \"\"\"h_patched = h1 + alpha * (h2 - h1). Alpha > 1 overshoots the second pass.\"\"\"
    def fn(h1, h2):
        return h1 + alpha * (h2 - h1)
    fn.__name__ = f'alpha={alpha}'
    return fn

def per_dim_scale(h1, h2):
    \"\"\"Scale each dimension by how much it changed (amplify active dims).
    gate_d = |h2_d - h1_d| / mean(|h2 - h1|) — dims that changed more get amplified.
    h_patched = h1 + gate * (h2 - h1)
    \"\"\"
    delta = h2 - h1
    abs_delta = delta.abs()
    # Per-dimension gate: normalize by mean change
    mean_change = abs_delta.mean(dim=-1, keepdim=True).clamp(min=1e-8)
    gate = abs_delta / mean_change  # > 1 for active dims, < 1 for quiet dims
    gate = gate.clamp(max=3.0)  # cap to avoid explosion
    return h1 + gate * delta

def per_dim_scale_soft(h1, h2):
    \"\"\"Softer version: gate = sigmoid(log(|delta_d| / mean(|delta|)))\"\"\"
    delta = h2 - h1
    abs_delta = delta.abs()
    mean_change = abs_delta.mean(dim=-1, keepdim=True).clamp(min=1e-8)
    log_ratio = torch.log(abs_delta / mean_change + 1e-8)
    gate = torch.sigmoid(log_ratio)  # 0.5 for average, >0.5 for active, <0.5 for quiet
    gate = gate * 2  # rescale so average dim gets gate=1
    return h1 + gate * delta

def adaptive_norm(beta):
    \"\"\"h_patched = (||h1|| + beta*(||h2||-||h1||)) * (h2/||h2||)
    beta=0 -> h1 norm, h2 dir (norm-preserving)
    beta=1 -> h2 norm, h2 dir (standard alpha=1)
    beta>1 -> amplified norm in h2 direction
    \"\"\"
    def fn(h1, h2):
        h1_norm = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h2_norm = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        target_norm = h1_norm + beta * (h2_norm - h1_norm)
        return target_norm * (h2 / h2_norm)
    fn.__name__ = f'adaptive_norm_beta={beta}'
    return fn

def direction_amplify(gamma):
    \"\"\"Amplify ONLY the direction change, keep norm at h2 level.
    delta_dir = normalize(h2 - h1)
    h_patched = h2 + gamma * ||h2-h1|| * delta_dir
    Pushes further in the direction h2 moved from h1.
    \"\"\"
    def fn(h1, h2):
        delta = h2 - h1
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_dir = delta / delta_norm
        return h2 + gamma * delta_norm * delta_dir
    fn.__name__ = f'dir_amplify_gamma={gamma}'
    return fn

def topk_direction(k_frac):
    \"\"\"Only keep the top-k% of dimensions by change magnitude, zero the rest.
    h_patched = h1 + mask * (h2 - h1) where mask selects top-k% changed dims.
    \"\"\"
    def fn(h1, h2):
        delta = h2 - h1
        abs_delta = delta.abs()
        # Per-token top-k
        k = max(1, int(h1.shape[-1] * k_frac))
        threshold = abs_delta.topk(k, dim=-1).values[..., -1:]
        mask = (abs_delta >= threshold).float()
        return h1 + mask * delta
    fn.__name__ = f'topk_dir_{int(k_frac*100)}pct'
    return fn


# =====================================================================
# All variants to test
# =====================================================================
VARIANTS = {
    # Alpha overshoot (the main hypothesis)
    'alpha=1.0 (standard)': None,  # baseline dup
    'alpha=1.25': alpha_overshoot(1.25),
    'alpha=1.5': alpha_overshoot(1.5),
    'alpha=2.0': alpha_overshoot(2.0),
    'alpha=0.75': alpha_overshoot(0.75),
    # Per-dimension scaling
    'per_dim_scale': per_dim_scale,
    'per_dim_scale_soft': per_dim_scale_soft,
    # Adaptive norm (interpolate between h1 and h2 norms)
    'adaptive_norm_beta=0.5': adaptive_norm(0.5),
    'adaptive_norm_beta=1.5': adaptive_norm(1.5),
    'adaptive_norm_beta=2.0': adaptive_norm(2.0),
    # Direction amplification (push further in h2-h1 direction)
    'dir_amplify_gamma=0.25': direction_amplify(0.25),
    'dir_amplify_gamma=0.5': direction_amplify(0.5),
    'dir_amplify_gamma=1.0': direction_amplify(1.0),
    # Top-k direction (only keep most changed dimensions)
    'topk_dir_50pct': topk_direction(0.50),
    'topk_dir_25pct': topk_direction(0.25),
    'topk_dir_10pct': topk_direction(0.10),
}

# =====================================================================
# Run
# =====================================================================
print('=' * 70)
print('DIRECTION-AWARE SEAM INTERVENTIONS — 72B')
print(f'Motivation: norm_ratio=1.04, cosine=0.997')
print(f'The 0.3% direction change is where all the action is')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)

inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

# Baseline
print('\\n  [Baseline — no duplication]', flush=True)
baseline_r = evaluate_baseline_no_dup(model, tokenizer)
baseline_combined = baseline_r['combined']

all_block_results = {}

for block in BLOCKS:
    print(f'\\n{\"=\" * 70}')
    print(f'BLOCK ({block[0]},{block[1]})')
    print(f'{\"=\" * 70}', flush=True)

    layer_order = build_layer_order(block, N)
    results = [baseline_r]

    # alpha=0 reference
    print('\\n  [alpha=0]', flush=True)
    r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers,
                         block, 'alpha0', 'alpha=0 (h1 only)')
    results.append(r)

    # All variants
    for vname, vfn in VARIANTS.items():
        print(f'\\n  [{vname}]', flush=True)
        r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers,
                             block, vfn, vname)
        results.append(r)

    # Summary
    sorted_r = sorted(results, key=lambda x: x['combined'], reverse=True)
    alpha1 = next(r for r in results if r['name'] == 'alpha=1.0 (standard)')

    print(f'\\n  --- Block ({block[0]},{block[1]}) Results ---')
    print(f'  {\"Variant\":45s} {\"Math\":>8s} {\"EQ\":>8s} {\"Combined\":>10s} {\"vs std\":>8s}')
    print('  ' + '-' * 85)
    for r in sorted_r:
        vs_std = r['combined'] - alpha1['combined']
        marker = ' ***' if r == sorted_r[0] and r['name'] != 'alpha=1.0 (standard)' else ''
        print(f'  {r[\"name\"]:45s} {r[\"math\"]:8.4f} {r[\"eq\"]:8.1f} {r[\"combined\"]:10.2f} {vs_std:+8.2f}{marker}', flush=True)

    all_block_results[f'({block[0]},{block[1]})'] = {
        'block': list(block),
        'results': results,
        'best_variant': sorted_r[0]['name'],
        'best_combined': sorted_r[0]['combined'],
        'alpha1_combined': alpha1['combined'],
    }

# Save
os.makedirs('results/data/72b/direction_interventions', exist_ok=True)
output = {
    'model': MODEL_PATH,
    'num_layers': N,
    'date': datetime.now().isoformat(),
    'motivation': 'norm_ratio=1.04 cosine=0.997: direction is everything on 72B',
    'baseline': baseline_r,
    'blocks': all_block_results,
}
with open('results/data/72b/direction_interventions/results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to results/data/72b/direction_interventions/results.json', flush=True)
"

echo "=== Done at $(date) ==="
