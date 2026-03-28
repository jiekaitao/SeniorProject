#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_phase3_causal_%j.log
#SBATCH --job-name=dp_patch

# Phase 3: Causal Mediation Patching (HCMP)
# Patches FFN neuron groups between attn-only and whisper runs to find
# which neurons CAUSE reasoning improvement vs factual harm.
# Also tests HCES (cross-entropy search) over grouped masks.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Phase 3: Causal Patching + HCES ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe, MATH_QUESTIONS
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = 'models/full/gemma-3-27b-it'
BLOCKS = [(0, 2), (12, 13), (47, 48)]
SAVE_DIR = 'results/data/gemma3_27b/neuron_analysis'
os.makedirs(SAVE_DIR, exist_ok=True)

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

eq_all = _load_questions()
math_subset = MATH_QUESTIONS[:10]
eq_subset = eq_all[:10]

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    if hasattr(model.config, 'num_hidden_layers'):
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

def apply_dup():
    order = build_order(BLOCKS, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

sorted_blocks = sorted(BLOCKS)
dup_layers = []
for (i, j) in sorted_blocks:
    for l in range(i, j):
        dup_layers.append(l)

# ======================================================================
# CAUSAL PATCHING: Per-layer FFN contribution analysis
# ======================================================================
print('\\n=== Causal Patching: Per-Layer FFN Contribution ===', flush=True)
print('Testing: patch one layers FFN from whisper into attn-only', flush=True)

# For each dup layer, test: what happens if we enable ONLY that layers FFN?
# Base: attn-only (all FFN β=0)
# Patch: set FFN β=0.2 for ONE layer at a time

patch_results = []

# First: baseline attn-only
apply_dup()
hooks_base = []
for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    counter = [0]
    def make_zero_hook(ctr):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (0.0 * output[0],) + output[1:]
                return 0.0 * output
            return output
        return hook
    h = module.mlp.register_forward_hook(make_zero_hook(counter))
    hooks_base.append(h)

t0 = time.time()
math_r = run_math_probe(gen, questions=math_subset, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
base_combined = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'  attn_only base: combined={base_combined:.2f} ({time.time()-t0:.0f}s)', flush=True)

for h in hooks_base:
    h.remove()
restore()

# Now test each layer individually
for target_layer in dup_layers:
    for beta in [0.1, 0.2, 0.5, 1.0]:
        apply_dup()
        hooks = []
        for layer_idx in dup_layers:
            module = original_layers[layer_idx]
            counter = [0]

            if layer_idx == target_layer:
                # Enable this layers FFN at beta
                b = beta
                def make_beta_hook(ctr, b):
                    def hook(module, input, output):
                        ctr[0] += 1
                        if ctr[0] % 2 == 0:
                            if isinstance(output, tuple):
                                return (b * output[0],) + output[1:]
                            return b * output
                        return output
                    return hook
                h = module.mlp.register_forward_hook(make_beta_hook(counter, b))
            else:
                # Zero all other FFNs
                def make_zero(ctr):
                    def hook(module, input, output):
                        ctr[0] += 1
                        if ctr[0] % 2 == 0:
                            if isinstance(output, tuple):
                                return (0.0 * output[0],) + output[1:]
                            return 0.0 * output
                        return output
                    return hook
                h = module.mlp.register_forward_hook(make_zero(counter))
            hooks.append(h)

        t0 = time.time()
        math_r = run_math_probe(gen, questions=math_subset, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        delta = combined - base_combined
        elapsed = time.time() - t0

        print(f'  L{target_layer} FFN@{beta}: combined={combined:.2f} delta={delta:+.2f} ({elapsed:.0f}s)', flush=True)
        patch_results.append({
            'layer': target_layer, 'beta': beta,
            'math': math_r['score'], 'eq': eq_r['score'],
            'combined': combined, 'delta_vs_attn_only': delta,
        })

        for h in hooks:
            h.remove()
        restore()

# ======================================================================
# HCES: Cross-Entropy Search over grouped masks
# ======================================================================
print('\\n=== HCES: Cross-Entropy Search ===', flush=True)

# Groups: 4 layers × {attn, ffn} = 8 groups
# Levels: {0.0, 0.2, 0.5, 1.0}
# CE optimization

import random
random.seed(42)

levels = [0.0, 0.2, 0.5, 1.0]
n_groups = len(dup_layers) * 2  # attn + ffn per layer
group_names = []
for l in dup_layers:
    group_names.append(f'attn_L{l}')
    group_names.append(f'ffn_L{l}')

# Initialize uniform distribution
q = {g: np.ones(len(levels)) / len(levels) for g in group_names}

N_POP = 6
T_ITER = 5
ELITE_FRAC = 0.33
ETA = 0.7

best_mask_ever = None
best_score_ever = -1e9
hces_log = []

for t in range(T_ITER):
    samples = []
    for n in range(N_POP):
        # Sample a mask
        mask = {}
        for g in group_names:
            idx = np.random.choice(len(levels), p=q[g])
            mask[g] = levels[idx]

        # Evaluate
        apply_dup()
        hooks = []
        for layer_idx in dup_layers:
            module = original_layers[layer_idx]

            # Attention hook
            attn_alpha = mask[f'attn_L{layer_idx}']
            attn_ctr = [0]
            def make_attn_hook(ctr, a):
                def hook(module, input, output):
                    ctr[0] += 1
                    if ctr[0] % 2 == 0:
                        if isinstance(output, tuple):
                            return (a * output[0],) + output[1:]
                        return a * output
                    return output
                return hook
            h = module.self_attn.register_forward_hook(make_attn_hook(attn_ctr, attn_alpha))
            hooks.append(h)

            # FFN hook
            ffn_beta = mask[f'ffn_L{layer_idx}']
            ffn_ctr = [0]
            def make_ffn_hook(ctr, b):
                def hook(module, input, output):
                    ctr[0] += 1
                    if ctr[0] % 2 == 0:
                        if isinstance(output, tuple):
                            return (b * output[0],) + output[1:]
                        return b * output
                    return output
                return hook
            h = module.mlp.register_forward_hook(make_ffn_hook(ffn_ctr, ffn_beta))
            hooks.append(h)

        math_r = run_math_probe(gen, questions=math_subset, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
        score = math_r['score'] * 50 + eq_r['score'] * 0.5

        for h in hooks:
            h.remove()
        restore()

        samples.append((mask, score))
        if score > best_score_ever:
            best_score_ever = score
            best_mask_ever = mask.copy()

    # Update distribution from elites
    samples.sort(key=lambda x: x[1], reverse=True)
    n_elite = max(1, int(N_POP * ELITE_FRAC))
    elites = samples[:n_elite]

    for g in group_names:
        freq = np.zeros(len(levels))
        for mask, _ in elites:
            idx = levels.index(mask[g])
            freq[idx] += 1
        freq = freq / freq.sum()
        q[g] = (1 - ETA) * q[g] + ETA * freq

    elite_scores = [s for _, s in elites]
    print(f'  Iter {t}: best={samples[0][1]:.2f} mean_elite={np.mean(elite_scores):.2f} '
          f'best_ever={best_score_ever:.2f}', flush=True)

    hces_log.append({
        'iter': t,
        'best': samples[0][1],
        'mean_elite': float(np.mean(elite_scores)),
        'best_mask': samples[0][0],
    })

print(f'\\nHCES best: {best_score_ever:.2f}', flush=True)
print(f'Best mask: {best_mask_ever}', flush=True)

# ======================================================================
# Validate best HCES mask with full probes
# ======================================================================
print('\\n=== Validating HCES Best ===', flush=True)
apply_dup()
hooks = []
for layer_idx in dup_layers:
    module = original_layers[layer_idx]
    attn_alpha = best_mask_ever[f'attn_L{layer_idx}']
    ffn_beta = best_mask_ever[f'ffn_L{layer_idx}']

    attn_ctr = [0]
    def make_attn_v(ctr, a):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (a * output[0],) + output[1:]
                return a * output
            return output
        return hook
    h = module.self_attn.register_forward_hook(make_attn_v(attn_ctr, attn_alpha))
    hooks.append(h)

    ffn_ctr = [0]
    def make_ffn_v(ctr, b):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (b * output[0],) + output[1:]
                return b * output
            return output
        return hook
    h = module.mlp.register_forward_hook(make_ffn_v(ffn_ctr, ffn_beta))
    hooks.append(h)

t0 = time.time()
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
combined = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'HCES validated: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)

for h in hooks:
    h.remove()
restore()

# ======================================================================
# SAVE
# ======================================================================
results = {
    'causal_patching': {
        'attn_only_base': base_combined,
        'per_layer_patches': patch_results,
    },
    'hces': {
        'n_pop': N_POP, 'n_iter': T_ITER,
        'best_score_reduced': best_score_ever,
        'best_mask': best_mask_ever,
        'validated_combined': combined,
        'log': hces_log,
    },
}
with open(f'{SAVE_DIR}/phase3_causal_hces.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/phase3_causal_hces.json', flush=True)
"

echo "=== Finished: $(date) ==="
