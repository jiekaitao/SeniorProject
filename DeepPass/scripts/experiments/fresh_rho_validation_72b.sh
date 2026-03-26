#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_fresh_rho_%j.log
#SBATCH --job-name=deeppass_frho

# CRITICAL: Fresh rho/BLOOD revalidation on 72B
# Previous validation used stale/estimated "known_combined" scores → p=0.20
# This run: fresh contemporaneous dual-probe evaluations in a single session
# with consistent random seed, for 25 blocks spanning the full model

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Fresh Rho/BLOOD Revalidation on 72B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

# Fix random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('FRESH RHO/BLOOD REVALIDATION ON 72B')
print('All evaluations done in ONE session with consistent random seed')
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
# Step 1: Compute rho for all candidate blocks
# =====================================================================
# 25 blocks spanning the full model: mix of sizes and positions
TEST_BLOCKS = [
    # Early (0-20)
    (0, 3), (0, 7), (4, 9), (5, 12), (8, 13), (10, 17), (15, 20),
    # Mid-early (20-40)
    (20, 27), (22, 27), (25, 32), (28, 33), (30, 37), (35, 40),
    # Mid-deep (40-55)
    (40, 45), (40, 47), (42, 49), (45, 50), (45, 52),
    # Deep (55-80)
    (50, 55), (50, 60), (55, 60), (55, 62), (60, 65), (65, 72), (70, 77),
]

print(f'\\n--- Step 1: Computing displacement rho for {len(TEST_BLOCKS)} blocks ---', flush=True)

rho_results = {}
for idx, block in enumerate(TEST_BLOCKS):
    i, j = block
    rhos = []
    for prompt in CAL_PROMPTS:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(ids['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
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
    rho_results[block] = rho
    print(f'  [{idx+1}/{len(TEST_BLOCKS)}] ({i:2d},{j:2d}) rho={rho:.4f}', flush=True)

# =====================================================================
# Step 2: BLOOD profiles (hooks-based, fast)
# =====================================================================
print(f'\\n--- Step 2: Computing BLOOD profiles ---', flush=True)

def compute_blood_profile(layer_list, n_layers):
    layer_norms = [[] for _ in range(n_layers)]
    hooks = []
    def make_hook(idx):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            norm = torch.norm(out.float() - inp.float()).item()
            layer_norms[idx].append(norm)
        return hook_fn
    for idx in range(n_layers):
        hooks.append(layer_list[idx].register_forward_hook(make_hook(idx)))
    for prompt in CAL_PROMPTS[:4]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)
    for h in hooks:
        h.remove()
    return [float(np.mean(ns)) if ns else 0.0 for ns in layer_norms]

# Base BLOOD
inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N
base_blood = compute_blood_profile(original_layers, N)

blood_impacts = {}
for idx, block in enumerate(TEST_BLOCKS):
    i, j = block
    order = build_order([block], N)
    dup_layers = [original_layers[idx2] for idx2 in order]
    inner.layers = nn.ModuleList(dup_layers)
    model.config.num_hidden_layers = len(order)
    dup_blood = compute_blood_profile(dup_layers, len(order))
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    impact = 0
    for l in range(j, N):
        dup_idx = l + (j - i)
        if dup_idx < len(dup_blood):
            impact += base_blood[l] - dup_blood[dup_idx]
    blood_impacts[block] = impact
    if (idx + 1) % 5 == 0:
        print(f'  [{idx+1}/{len(TEST_BLOCKS)}] blood done', flush=True)

# =====================================================================
# Step 3: FRESH dual-probe evaluation of ALL blocks (the critical step)
# =====================================================================
print(f'\\n--- Step 3: Fresh dual-probe evaluation of all {len(TEST_BLOCKS)} blocks ---', flush=True)
print('This is the expensive step (~25 min per block)', flush=True)

gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Baseline first
print('\\n  Evaluating baseline...', flush=True)
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'  Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f}', flush=True)

fresh_results = []
for idx, block in enumerate(TEST_BLOCKS):
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
    model.config.num_hidden_layers = len(order)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - baseline
    elapsed = time.time() - t0

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    fresh_results.append({
        'block': list(block),
        'rho': rho_results[block],
        'blood_impact': blood_impacts[block],
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined,
        'delta': delta,
    })
    print(f'  [{idx+1}/{len(TEST_BLOCKS)}] ({i:2d},{j:2d}): rho={rho_results[block]:.4f} combined={combined:.2f} delta={delta:+.2f} ({elapsed:.0f}s)', flush=True)

# =====================================================================
# Step 4: Correlation analysis
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('CORRELATION ANALYSIS (all fresh, contemporaneous measurements)')
print(f'{\"=\" * 70}', flush=True)

rhos = [r['rho'] for r in fresh_results]
bloods = [r['blood_impact'] for r in fresh_results]
deltas = [r['delta'] for r in fresh_results]
combineds = [r['combined'] for r in fresh_results]

# Primary: rho vs delta
r_rho_d, p_rho_d = spearmanr(rhos, deltas)
r_rho_c, p_rho_c = spearmanr(rhos, combineds)
print(f'Rho vs delta:      Spearman r={r_rho_d:+.3f} (p={p_rho_d:.4f})')
print(f'Rho vs combined:   Spearman r={r_rho_c:+.3f} (p={p_rho_c:.4f})')

# BLOOD
r_blood_d, p_blood_d = spearmanr(bloods, deltas)
print(f'BLOOD vs delta:    Spearman r={r_blood_d:+.3f} (p={p_blood_d:.4f})')

# Pearson
r_rho_d_p, p_rho_d_p = pearsonr(rhos, deltas)
r_blood_d_p, p_blood_d_p = pearsonr(bloods, deltas)
print(f'Rho vs delta (Pearson):   r={r_rho_d_p:+.3f} (p={p_rho_d_p:.4f})')
print(f'BLOOD vs delta (Pearson): r={r_blood_d_p:+.3f} (p={p_blood_d_p:.4f})')

# Combined
rho_norm = [(v - min(rhos)) / (max(rhos) - min(rhos) + 1e-8) for v in rhos]
blood_norm = [(v - min(bloods)) / (max(bloods) - min(bloods) + 1e-8) for v in bloods]
combined_metric = [-(r * 0.5 + b * 0.5) for r, b in zip(rho_norm, blood_norm)]
r_comb, p_comb = spearmanr(combined_metric, deltas)
print(f'Combined (rho+BLOOD): Spearman r={r_comb:+.3f} (p={p_comb:.4f})')

# Top-k precision: does rho correctly identify top-5 blocks?
print(f'\\n--- Top-k Precision ---')
sorted_by_rho = sorted(range(len(fresh_results)), key=lambda i: rhos[i])
sorted_by_delta = sorted(range(len(fresh_results)), key=lambda i: deltas[i], reverse=True)
top5_rho = set(sorted_by_rho[:5])
top5_delta = set(sorted_by_delta[:5])
top10_rho = set(sorted_by_rho[:10])
top10_delta = set(sorted_by_delta[:10])
print(f'Top-5 overlap (rho vs delta): {len(top5_rho & top5_delta)}/5')
print(f'Top-10 overlap (rho vs delta): {len(top10_rho & top10_delta)}/10')

# Print ranked lists
print(f'\\nBy rho (best first):')
for i in sorted_by_rho[:10]:
    r = fresh_results[i]
    print(f'  ({r[\"block\"][0]:2d},{r[\"block\"][1]:2d}): rho={r[\"rho\"]:.4f} delta={r[\"delta\"]:+.2f}')
print(f'\\nBy delta (best first):')
for i in sorted_by_delta[:10]:
    r = fresh_results[i]
    print(f'  ({r[\"block\"][0]:2d},{r[\"block\"][1]:2d}): rho={r[\"rho\"]:.4f} delta={r[\"delta\"]:+.2f}')

print(flush=True)

# Save
os.makedirs('results/data/72b/fresh_validation', exist_ok=True)
output = {
    'date': datetime.now().isoformat(),
    'model': MODEL_PATH,
    'n_blocks': len(TEST_BLOCKS),
    'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline},
    'results': fresh_results,
    'correlations': {
        'rho_vs_delta': {'spearman_r': r_rho_d, 'spearman_p': p_rho_d, 'pearson_r': r_rho_d_p, 'pearson_p': p_rho_d_p},
        'blood_vs_delta': {'spearman_r': r_blood_d, 'spearman_p': p_blood_d, 'pearson_r': r_blood_d_p, 'pearson_p': p_blood_d_p},
        'rho_vs_combined': {'spearman_r': r_rho_c, 'spearman_p': p_rho_c},
        'combined_metric_vs_delta': {'spearman_r': r_comb, 'spearman_p': p_comb},
    },
    'topk_precision': {
        'top5_overlap': len(top5_rho & top5_delta),
        'top10_overlap': len(top10_rho & top10_delta),
    },
    'base_blood_profile': base_blood,
}
with open('results/data/72b/fresh_validation/results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'Saved to results/data/72b/fresh_validation/results.json', flush=True)
"

echo "=== Done at $(date) ==="
