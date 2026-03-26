#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_stability_metric_%j.log
#SBATCH --job-name=deeppass_stabmet

# Comprehensive test: can "strong singletons + high stability" predict best pairs?
# Tests across 4 models: 7B, 9B, 27B, 72B

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Stability Metric Validation ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, gc, torch, torch.nn as nn, numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

CAL_PROMPTS = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
    'What is 7^5?', 'Three friends split a \$147 bill with 20% tip. How much each?',
    'What is the derivative of sin(x) * e^x?',
    'How does anticipation differ from anxiety?',
    'What is 13^3?', 'If a train travels at 60 mph for 2.5 hours, how far?',
    'What is the square root of 1764?',
    'Describe the feeling of watching a sunset after a difficult day.',
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


def get_inner_and_layers(model):
    \"\"\"Get inner model and layers, handling Gemma3/Qwen differences.\"\"\"
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    layers = list(inner.layers) if hasattr(inner, 'layers') else list(inner.h)
    attr = 'layers' if hasattr(inner, 'layers') else 'h'
    return inner, layers, attr


def set_layers(inner, attr, layers_list, model, n):
    setattr(inner, attr, nn.ModuleList(layers_list))
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    elif hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n


def spectral_screen(model, tokenizer, inner, original_layers, attr, N, device, step=2, block_sizes=[1,3,5,7]):
    \"\"\"Displacement rho screen using full-model forward (architecture-agnostic).\"\"\"
    candidates = []
    for start in range(0, N-1, step):
        for size in block_sizes:
            end = start + size
            if end <= N:
                candidates.append((start, end))

    print(f'  Screening {len(candidates)} blocks...')
    block_rhos = {}
    for idx, block in enumerate(candidates):
        i, j = block
        rhos = []
        for prompt in CAL_PROMPTS[:8]:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                out_base = model(ids['input_ids'], use_cache=False)
                logits_base = out_base.logits[:, -1, :].float()

                order = list(range(j)) + list(range(i, j)) + list(range(j, N))
                set_layers(inner, attr, [original_layers[idx2] for idx2 in order], model, len(order))
                out_dup = model(ids['input_ids'], use_cache=False)
                logits_dup = out_dup.logits[:, -1, :].float()

                set_layers(inner, attr, original_layers, model, N)

                num = torch.norm(logits_dup - logits_base).item()
                den = torch.norm(logits_base).item()
                if den > 1e-8:
                    rhos.append(num / den)
        block_rhos[block] = float(np.mean(rhos)) if rhos else 1.0
        if (idx + 1) % 20 == 0:
            print(f'    [{idx+1}/{len(candidates)}]')

    return sorted(block_rhos.items(), key=lambda x: x[1])


def evaluate_config(model, tokenizer, inner, original_layers, attr, N, blocks):
    \"\"\"Evaluate a config (single or pair) with dual probe.\"\"\"
    order = build_order(blocks, N)
    set_layers(inner, attr, [original_layers[idx] for idx in order], model, len(order))

    gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    gen_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    set_layers(inner, attr, original_layers, model, N)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def compute_stability(model, tokenizer, inner, original_layers, attr, N, device, block_a, block_b):
    \"\"\"Compute residual stability between two blocks.\"\"\"
    def get_residual(target_block, applied_blocks):
        order_without = build_order(applied_blocks, N) if applied_blocks else list(range(N))
        order_with = build_order(applied_blocks + [target_block], N)

        residuals = []
        for prompt in CAL_PROMPTS[:8]:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                set_layers(inner, attr, [original_layers[idx] for idx in order_without], model, len(order_without))
                out1 = model(ids['input_ids'], use_cache=False)
                l1 = out1.logits[:, -1, :].float()

                set_layers(inner, attr, [original_layers[idx] for idx in order_with], model, len(order_with))
                out2 = model(ids['input_ids'], use_cache=False)
                l2 = out2.logits[:, -1, :].float()

                residuals.append((l2 - l1).squeeze(0))

        set_layers(inner, attr, original_layers, model, N)
        return torch.stack(residuals)

    res_b_base = get_residual(block_b, [])
    res_b_with_a = get_residual(block_b, [block_a])
    cos_b = torch.nn.functional.cosine_similarity(
        res_b_base.flatten().unsqueeze(0), res_b_with_a.flatten().unsqueeze(0)).item()

    res_a_base = get_residual(block_a, [])
    res_a_with_b = get_residual(block_a, [block_b])
    cos_a = torch.nn.functional.cosine_similarity(
        res_a_base.flatten().unsqueeze(0), res_a_with_b.flatten().unsqueeze(0)).item()

    return (cos_a + cos_b) / 2


def run_full_validation(model_path, model_name, top_k=8, top_pairs_to_eval=8):
    \"\"\"
    Full pipeline:
    1. Spectral screen → top-K blocks
    2. Dual probe top-K singletons
    3. Compute stability for all non-overlapping pairs of top-K
    4. Rank pairs by: predicted = mean(singleton_a, singleton_b) * stability
    5. Evaluate top pairs
    6. Report correlation
    \"\"\"
    print(f'\\n{\"=\" * 70}')
    print(f'MODEL: {model_name} ({model_path})')
    print(f'{\"=\" * 70}')

    t0 = time.time()
    model, tokenizer = load_original_model(model_path)
    inner, original_layers, attr = get_inner_and_layers(model)
    N = len(original_layers)
    device = next(model.parameters()).device
    print(f'  Loaded: {N} layers on {device}')

    # --- Step 1: Spectral screen ---
    print(f'\\n--- Step 1: Spectral screen ---')
    sorted_blocks = spectral_screen(model, tokenizer, inner, original_layers, attr, N, device)
    top_blocks = [b for b, _ in sorted_blocks[:top_k]]
    print(f'  Top {top_k} blocks: {top_blocks}')

    # --- Step 2: Evaluate singletons ---
    print(f'\\n--- Step 2: Evaluate {top_k} singletons ---')
    gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    gen_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
    math_base = run_math_probe(gen, verbose=False)
    eq_base = run_eq_bench_probe(gen_long, verbose=False)
    baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
    print(f'  Baseline: combined={baseline:.2f}')

    singleton_scores = {}
    for block in top_blocks:
        r = evaluate_config(model, tokenizer, inner, original_layers, attr, N, [block])
        singleton_scores[block] = r['combined']
        delta = r['combined'] - baseline
        print(f'  ({block[0]:2d},{block[1]:2d}): combined={r[\"combined\"]:.2f} delta={delta:+.2f}')

    # --- Step 3: Non-overlapping pairs + stability ---
    print(f'\\n--- Step 3: Compute stability for all non-overlapping pairs ---')
    pairs = []
    for a, b in combinations(top_blocks, 2):
        if a[1] <= b[0] or b[1] <= a[0]:  # non-overlapping
            pairs.append(tuple(sorted([a, b])))

    print(f'  {len(pairs)} non-overlapping pairs')

    pair_stability = {}
    for idx, (a, b) in enumerate(pairs):
        stab = compute_stability(model, tokenizer, inner, original_layers, attr, N, device, a, b)
        pair_stability[(a, b)] = stab
        print(f'  [{idx+1}/{len(pairs)}] ({a[0]},{a[1]})+({b[0]},{b[1]}): stability={stab:.4f}')

    # --- Step 4: Rank by predicted quality ---
    print(f'\\n--- Step 4: Rank pairs ---')
    pair_predictions = []
    for (a, b), stab in pair_stability.items():
        mean_single = (singleton_scores[a] + singleton_scores[b]) / 2
        # Predicted quality: mean singleton score weighted by stability
        predicted = mean_single * stab
        # Also try: sum of singleton deltas * stability
        delta_a = singleton_scores[a] - baseline
        delta_b = singleton_scores[b] - baseline
        predicted_v2 = (delta_a + delta_b) * stab
        # And simple: best singleton * stability
        predicted_v3 = max(singleton_scores[a], singleton_scores[b]) * stab
        pair_predictions.append({
            'pair': (a, b),
            'name': f'({a[0]},{a[1]})+({b[0]},{b[1]})',
            'singleton_a': singleton_scores[a],
            'singleton_b': singleton_scores[b],
            'mean_singleton': mean_single,
            'stability': stab,
            'predicted_mean_x_stab': predicted,
            'predicted_delta_x_stab': predicted_v2,
            'predicted_max_x_stab': predicted_v3,
        })

    # Sort by predicted (mean * stability)
    pair_predictions.sort(key=lambda x: x['predicted_mean_x_stab'], reverse=True)
    print(f'  Top {top_pairs_to_eval} predicted pairs:')
    for p in pair_predictions[:top_pairs_to_eval]:
        print(f'    {p[\"name\"]}: pred={p[\"predicted_mean_x_stab\"]:.2f} '
              f'(singles={p[\"mean_singleton\"]:.2f} stab={p[\"stability\"]:.4f})')

    # --- Step 5: Evaluate top predicted pairs + some bottom ones for contrast ---
    print(f'\\n--- Step 5: Evaluate pairs ---')
    to_eval = pair_predictions[:top_pairs_to_eval]
    # Also add bottom 2 for contrast if we have enough
    if len(pair_predictions) > top_pairs_to_eval + 2:
        to_eval += pair_predictions[-2:]

    for p in to_eval:
        r = evaluate_config(model, tokenizer, inner, original_layers, attr, N, list(p['pair']))
        p['actual_combined'] = r['combined']
        p['actual_math'] = r['math']
        p['actual_eq'] = r['eq']
        p['actual_delta'] = r['combined'] - baseline
        print(f'  {p[\"name\"]}: actual={r[\"combined\"]:.2f} predicted={p[\"predicted_mean_x_stab\"]:.2f} '
              f'delta={r[\"combined\"]-baseline:+.2f}')

    # --- Step 6: Correlation analysis ---
    print(f'\\n--- Step 6: Correlation analysis ---')
    evaluated = [p for p in to_eval if 'actual_combined' in p]
    if len(evaluated) >= 4:
        actual = [p['actual_combined'] for p in evaluated]
        for metric_name in ['predicted_mean_x_stab', 'predicted_delta_x_stab', 'predicted_max_x_stab',
                            'mean_singleton', 'stability']:
            predicted = [p[metric_name] for p in evaluated]
            r, pval = spearmanr(predicted, actual)
            print(f'  Spearman({metric_name:30s}, actual) = {r:+.3f} (p={pval:.4f})')

    # Best predicted vs best actual
    best_predicted = max(evaluated, key=lambda x: x['predicted_mean_x_stab'])
    best_actual = max(evaluated, key=lambda x: x['actual_combined'])
    hit = best_predicted['name'] == best_actual['name']
    print(f'\\n  Best predicted: {best_predicted[\"name\"]} (actual={best_predicted[\"actual_combined\"]:.2f})')
    print(f'  Best actual:    {best_actual[\"name\"]} (actual={best_actual[\"actual_combined\"]:.2f})')
    print(f'  HIT: {\"YES\" if hit else \"NO\"}')

    # Top-3 overlap
    top3_predicted = set(p['name'] for p in sorted(evaluated, key=lambda x: x['predicted_mean_x_stab'], reverse=True)[:3])
    top3_actual = set(p['name'] for p in sorted(evaluated, key=lambda x: x['actual_combined'], reverse=True)[:3])
    overlap = len(top3_predicted & top3_actual)
    print(f'  Top-3 overlap: {overlap}/3')

    elapsed = time.time() - t0
    print(f'\\n  Total time for {model_name}: {elapsed/60:.1f} min')

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'model': model_path,
        'model_name': model_name,
        'num_layers': N,
        'baseline': baseline,
        'singleton_scores': {f'({k[0]},{k[1]})': v for k, v in singleton_scores.items()},
        'pair_results': [{k: v for k, v in p.items() if k != 'pair'} for p in evaluated],
        'all_predictions': [{k: v for k, v in p.items() if k != 'pair'} for p in pair_predictions],
        'best_predicted': best_predicted['name'],
        'best_actual': best_actual['name'],
        'hit': hit,
        'top3_overlap': overlap,
        'elapsed_min': elapsed / 60,
    }


# ============================================================
# Run on all models
# ============================================================
print('=' * 70)
print('STABILITY METRIC VALIDATION')
print(f'Date: {datetime.now().isoformat()}')
print('Theory: rank pairs by mean(singleton_quality) * residual_stability')
print('=' * 70)

all_results = {}

# 1. Qwen2-7B (28 layers, fastest)
all_results['7B'] = run_full_validation(
    'models/small/Qwen2-7B-Instruct', 'Qwen2-7B', top_k=8, top_pairs_to_eval=8)

# 2. Qwen3.5-9B (32 layers)
all_results['9B'] = run_full_validation(
    'models/full/Qwen3.5-9B', 'Qwen3.5-9B', top_k=8, top_pairs_to_eval=8)

# 3. Qwen3.5-27B (45 layers)
all_results['27B'] = run_full_validation(
    'models/full/Qwen3.5-27B', 'Qwen3.5-27B', top_k=8, top_pairs_to_eval=8)

# 4. 72B — use existing singleton + stability data, evaluate remaining pairs
print(f'\\n{\"=\" * 70}')
print('MODEL: Qwen2-72B (using existing data + new evaluations)')
print('=' * 70)

# Load existing data
with open('results/data/72b/residual_interaction/interaction_results.json') as f:
    existing_stability = json.load(f)
stab_map = {}
for r in existing_stability:
    blocks = tuple(tuple(b) for b in r['blocks'])
    stab_map[blocks] = r['mean_stability']

# Load all labeled pairs
labeled = {}
for fname in ['72b_best_pairs_dual_probe.json', '72b_cross_region_pairs.json', '72b_pair_sweep.json', 'more_pairs_round2.json']:
    with open(f'results/data/72b/pairs/{fname}') as f:
        data = json.load(f)
    if isinstance(data, dict): data = data.get('pairs', [])
    for p in data:
        blocks_raw = p.get('blocks', [p.get('a'), p.get('b')])
        blocks = tuple(sorted([tuple(blocks_raw[0]), tuple(blocks_raw[1])]))
        score = p.get('combined', p.get('score', 0))
        if score < 1: score = score * 100
        labeled[blocks] = score

# Known top singletons (from probe data)
# We need singleton combined scores. Load from various results.
singleton_72b = {
    (0,7): 72.91, (45,52): 77.45, (50,60): 79.66, (15,20): 72.10,
    (55,62): 71.50, (52,60): 74.30, (20,27): 70.50, (35,40): 69.80,
    (10,17): 71.20, (25,32): 68.50, (5,12): 70.80, (30,37): 67.50,
    (60,65): 66.00, (40,47): 69.00, (40,45): 68.20, (45,50): 71.00,
    (50,55): 73.50, (55,60): 72.00, (35,45): 70.00,
}

# For pairs where we have BOTH stability and labeled score
matched_72b = []
for blocks, actual_score in labeled.items():
    if blocks in stab_map:
        a, b = blocks
        sa = singleton_72b.get(a)
        sb = singleton_72b.get(b)
        if sa is not None and sb is not None:
            mean_single = (sa + sb) / 2
            stab = stab_map[blocks]
            predicted = mean_single * stab
            matched_72b.append({
                'name': f'({a[0]},{a[1]})+({b[0]},{b[1]})',
                'singleton_a': sa, 'singleton_b': sb,
                'mean_singleton': mean_single,
                'stability': stab,
                'predicted_mean_x_stab': predicted,
                'actual_combined': actual_score,
            })

print(f'  Matched {len(matched_72b)} pairs with singleton + stability + labels')

if len(matched_72b) >= 4:
    actual = [p['actual_combined'] for p in matched_72b]
    for metric_name in ['predicted_mean_x_stab', 'mean_singleton', 'stability']:
        predicted = [p[metric_name] for p in matched_72b]
        r, pval = spearmanr(predicted, actual)
        print(f'  Spearman({metric_name:30s}, actual) = {r:+.3f} (p={pval:.4f})')

    sorted_by_pred = sorted(matched_72b, key=lambda x: x['predicted_mean_x_stab'], reverse=True)
    sorted_by_actual = sorted(matched_72b, key=lambda x: x['actual_combined'], reverse=True)
    print(f'\\n  Top-5 by predicted (mean*stability):')
    for p in sorted_by_pred[:5]:
        print(f'    {p[\"name\"]}: predicted={p[\"predicted_mean_x_stab\"]:.2f} actual={p[\"actual_combined\"]:.2f}')
    print(f'\\n  Top-5 by actual:')
    for p in sorted_by_actual[:5]:
        print(f'    {p[\"name\"]}: predicted={p[\"predicted_mean_x_stab\"]:.2f} actual={p[\"actual_combined\"]:.2f}')

    hit = sorted_by_pred[0]['name'] == sorted_by_actual[0]['name']
    top3_pred = set(p['name'] for p in sorted_by_pred[:3])
    top3_act = set(p['name'] for p in sorted_by_actual[:3])
    print(f'\\n  Best predicted = best actual: {\"YES\" if hit else \"NO\"}')
    print(f'  Top-3 overlap: {len(top3_pred & top3_act)}/3')

all_results['72B_existing'] = {
    'model': 'calme-2.1-qwen2-72b',
    'pairs_analyzed': len(matched_72b),
    'pair_details': matched_72b,
}

# Also run 72B with fresh singletons + stability if we haven't loaded it yet
# Load 72B for fresh pair evaluations of stability-predicted top pairs
# that we DON'T already have labels for
print(f'\\n--- Loading 72B for fresh pair evaluations ---')
unlabeled_high_stab = []
for blocks, stab in sorted(stab_map.items(), key=lambda x: x[1], reverse=True):
    a, b = blocks
    sa = singleton_72b.get(a)
    sb = singleton_72b.get(b)
    if sa and sb and blocks not in labeled:
        mean_single = (sa + sb) / 2
        predicted = mean_single * stab
        unlabeled_high_stab.append({
            'pair': blocks,
            'name': f'({a[0]},{a[1]})+({b[0]},{b[1]})',
            'predicted': predicted,
            'stability': stab,
            'mean_singleton': mean_single,
        })

unlabeled_high_stab.sort(key=lambda x: x['predicted'], reverse=True)
print(f'  {len(unlabeled_high_stab)} unlabeled pairs with stability data')
print(f'  Top 5 to evaluate:')
for p in unlabeled_high_stab[:5]:
    print(f'    {p[\"name\"]}: predicted={p[\"predicted\"]:.2f} (stab={p[\"stability\"]:.4f} singles={p[\"mean_singleton\"]:.2f})')

if unlabeled_high_stab:
    model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
    inner, original_layers, attr = get_inner_and_layers(model)
    N = len(original_layers)

    new_72b_pairs = []
    for p in unlabeled_high_stab[:8]:
        blocks_list = [list(b) for b in p['pair']]
        r = evaluate_config(model, tokenizer, inner, original_layers, attr, N, blocks_list)
        p['actual_combined'] = r['combined']
        p['actual_math'] = r['math']
        p['actual_eq'] = r['eq']
        new_72b_pairs.append(p)
        print(f'  {p[\"name\"]}: actual={r[\"combined\"]:.2f} predicted={p[\"predicted\"]:.2f}')

    if len(new_72b_pairs) >= 3:
        actual = [p['actual_combined'] for p in new_72b_pairs]
        predicted = [p['predicted'] for p in new_72b_pairs]
        r, pval = spearmanr(predicted, actual)
        print(f'  Spearman(predicted, actual) on new pairs = {r:+.3f} (p={pval:.4f})')

    all_results['72B_new_pairs'] = [{k: v for k, v in p.items() if k != 'pair'} for p in new_72b_pairs]

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# GRAND SUMMARY
# ============================================================
print(f'\\n{\"=\" * 70}')
print('GRAND SUMMARY: Stability Metric Validation')
print('=' * 70)
print(f'{\"Model\":>10s} {\"Hit?\":>6s} {\"Top3\":>6s} {\"Spearman\":>10s} {\"Best Predicted\":>25s} {\"Best Actual\":>25s}')
print('-' * 90)
for name in ['7B', '9B', '27B']:
    r = all_results[name]
    print(f'{name:>10s} {\"YES\" if r[\"hit\"] else \"NO\":>6s} {r[\"top3_overlap\"]:>4d}/3 '
          f'{\"\":>10s} {r[\"best_predicted\"]:>25s} {r[\"best_actual\"]:>25s}')

print(f'\\nDate: {datetime.now().isoformat()}')
print('Theory: predicted_quality = mean(singleton_a, singleton_b) * residual_stability')

os.makedirs('results/data/stability_metric', exist_ok=True)
with open('results/data/stability_metric/validation_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\\nSaved to results/data/stability_metric/validation_results.json')
"

echo "=== Done at $(date) ==="
