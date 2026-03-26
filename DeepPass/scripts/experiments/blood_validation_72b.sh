#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_blood_72b_%j.log
#SBATCH --job-name=deeppass_blood72

# Validate BLOOD impact as screening metric on 72B
# Currently only validated on 7B (r=-0.492, p=0.028)
# Need to confirm on our main model

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== BLOOD Validation on 72B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn, numpy as np
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('BLOOD VALIDATION ON 72B')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

CAL_PROMPTS = [
    'What is 127 * 348?',
    'If a train travels at 60 mph for 2.5 hours, how far does it go?',
    'What emotion would someone feel after losing a close friend?',
    'def fibonacci(n): # complete this function',
    'Explain the concept of entropy in thermodynamics.',
    'What is the derivative of sin(x) * e^x?',
    'How would a parent feel seeing their child graduate?',
    'The theory of general relativity describes',
]

# =====================================================================
# Compute BLOOD profile: ||layer_output - layer_input|| per layer
# =====================================================================

def compute_blood_profile(model, inner, original_layers, N, tokenizer, device, prompts):
    \"\"\"Fast BLOOD: ||layer_output - layer_input|| per layer, averaged over prompts.\"\"\"
    layer_norms = [[] for _ in range(N)]
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            norm = torch.norm(out.float() - inp.float()).item()
            layer_norms[idx].append(norm)
        return hook_fn

    for idx in range(N):
        hooks.append(original_layers[idx].register_forward_hook(make_hook(idx)))

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)

    for h in hooks:
        h.remove()

    return [float(np.mean(ns)) if ns else 0.0 for ns in layer_norms]


def compute_blood_impact(base_profile, dup_profile, block_end, N):
    \"\"\"BLOOD impact = sum of downstream BLOOD changes.\"\"\"
    impact = 0
    for l in range(block_end, N):
        impact += base_profile[l] - dup_profile[l]
    return impact


def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

# =====================================================================
# Step 1: Base model BLOOD profile
# =====================================================================
print('\\n--- Computing base BLOOD profile ---', flush=True)
inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N
base_profile = compute_blood_profile(model, inner, original_layers, N, tokenizer, device, CAL_PROMPTS)
print(f'Base BLOOD: mean={np.mean(base_profile):.2f} std={np.std(base_profile):.2f}', flush=True)

# =====================================================================
# Step 2: Test configs — mix of known good and bad blocks
# =====================================================================
# Known performance from dual probe:
# Good: (45,52)=77.45, (50,60)=78.84, (0,7)=72.91, (15,20)=72.10
# Medium: (55,62)=71.50, (20,27)=70.50, (35,40)=69.80
# Bad: (30,37)=67.50, (60,65)=66.00, (70,75)=~65

TEST_CONFIGS = [
    ((0, 7), 72.91),
    ((5, 12), 70.80),
    ((10, 17), 71.20),
    ((15, 20), 72.10),
    ((20, 27), 70.50),
    ((25, 32), 68.50),
    ((30, 37), 67.50),
    ((35, 40), 69.80),
    ((40, 47), 69.00),
    ((45, 52), 77.45),
    ((50, 55), 73.50),
    ((50, 60), 78.84),
    ((55, 62), 71.50),
    ((60, 65), 66.00),
    ((65, 72), 65.00),  # estimated
    ((70, 77), 64.00),  # estimated
]

print(f'\\n--- Computing BLOOD for {len(TEST_CONFIGS)} configs ---', flush=True)

results = []
for block, known_combined in TEST_CONFIGS:
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    dup_profile = compute_blood_profile(model, inner, [original_layers[idx] for idx in order], len(order), tokenizer, device, CAL_PROMPTS)

    # Restore
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    # Map back to original layer indices for comparison
    # The duplicated model has extra layers; we need downstream layers in the original indexing
    # Simpler: just compare profiles at matching positions (pre-seam are identical)
    blood_impact = 0
    for l_idx in range(j, N):
        # In the duplicated model, this layer is shifted by block_size
        dup_idx = l_idx + (j - i)
        if dup_idx < len(dup_profile):
            blood_impact += base_profile[l_idx] - dup_profile[dup_idx]

    known_delta = known_combined - 70.52  # baseline combined
    results.append({
        'block': list(block),
        'known_combined': known_combined,
        'known_delta': known_delta,
        'blood_impact': blood_impact,
    })
    print(f'  ({i:2d},{j:2d}): blood_impact={blood_impact:+.2f} known_delta={known_delta:+.2f}', flush=True)

# =====================================================================
# Step 3: Correlation analysis
# =====================================================================
print(f'\\n--- Correlation analysis ---', flush=True)

blood_impacts = [r['blood_impact'] for r in results]
known_deltas = [r['known_delta'] for r in results]
known_combineds = [r['known_combined'] for r in results]

r_spear_delta, p_spear_delta = spearmanr(blood_impacts, known_deltas)
r_pears_delta, p_pears_delta = pearsonr(blood_impacts, known_deltas)
r_spear_comb, p_spear_comb = spearmanr(blood_impacts, known_combineds)

print(f'BLOOD_impact vs delta:    Spearman r={r_spear_delta:+.3f} (p={p_spear_delta:.4f}), Pearson r={r_pears_delta:+.3f} (p={p_pears_delta:.4f})')
print(f'BLOOD_impact vs combined: Spearman r={r_spear_comb:+.3f} (p={p_spear_comb:.4f})')
print(f'N={len(results)} configs tested', flush=True)

# Also compute displacement rho for comparison
print(f'\\n--- Displacement rho for comparison ---', flush=True)
rho_values = []
for r in results:
    block = tuple(r['block'])
    i, j = block
    rhos = []
    for prompt in CAL_PROMPTS[:4]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(ids['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            model.config.num_hidden_layers = len(order)
            out_dup = model(ids['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()

            inner.layers = nn.ModuleList(original_layers)
            model.config.num_hidden_layers = N

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)
    rho = float(np.mean(rhos)) if rhos else 1.0
    rho_values.append(rho)
    r['displacement_rho'] = rho

r_rho, p_rho = spearmanr(rho_values, known_deltas)
print(f'Rho vs delta: Spearman r={r_rho:+.3f} (p={p_rho:.4f})')

# Combined signal
print(f'\\n--- Combined rho + BLOOD ---')
# Simple: rank by rho, then by BLOOD for ties
combined_scores = [-(rho * 0.5 + blood * 0.5) for rho, blood in zip(
    [(v - min(rho_values)) / (max(rho_values) - min(rho_values) + 1e-8) for v in rho_values],
    [(v - min(blood_impacts)) / (max(blood_impacts) - min(blood_impacts) + 1e-8) for v in blood_impacts]
)]
r_comb, p_comb = spearmanr(combined_scores, known_deltas)
print(f'Combined (rho+BLOOD) vs delta: Spearman r={r_comb:+.3f} (p={p_comb:.4f})')
print(f'Rho alone:  r={r_rho:+.3f}')
print(f'BLOOD alone: r={r_spear_delta:+.3f}')
print(f'Combined:    r={r_comb:+.3f}', flush=True)

# Save
os.makedirs('results/data/72b/blood', exist_ok=True)
output = {
    'date': datetime.now().isoformat(),
    'model': MODEL_PATH,
    'n_configs': len(results),
    'base_blood_profile': base_profile,
    'configs': results,
    'correlations': {
        'blood_vs_delta': {'spearman_r': r_spear_delta, 'spearman_p': p_spear_delta, 'pearson_r': r_pears_delta, 'pearson_p': p_pears_delta},
        'blood_vs_combined': {'spearman_r': r_spear_comb, 'spearman_p': p_spear_comb},
        'rho_vs_delta': {'spearman_r': r_rho, 'spearman_p': p_rho},
        'combined_vs_delta': {'spearman_r': r_comb, 'spearman_p': p_comb},
    },
}
with open('results/data/72b/blood/blood_validation_72b.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to results/data/72b/blood/blood_validation_72b.json', flush=True)
"

echo "=== Done at $(date) ==="
