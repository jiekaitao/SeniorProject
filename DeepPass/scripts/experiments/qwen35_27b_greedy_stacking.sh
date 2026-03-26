#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_qwen35_27b_stacking_%j.log
#SBATCH --job-name=deeppass_q35stack

# Greedy stacking on Qwen3.5-27B
# Best single from spectral screen: (25,30) combined=78.30
# Now find a complementary second block

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Qwen3.5-27B Greedy Stacking ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn, numpy as np
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

model, tokenizer = load_original_model('models/full/Qwen3.5-27B')
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded Qwen3.5-27B: {N} layers')

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

def compute_rho(block):
    i, j = block
    # Use full-model forward for architecture safety
    rhos = []
    for prompt in cal_prompts:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            # Base model forward
            out_base = model(ids['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            # Duplicated forward
            order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            if hasattr(model.config, 'text_config'):
                model.config.text_config.num_hidden_layers = len(order)
            else:
                model.config.num_hidden_layers = len(order)

            out_dup = model(ids['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()

            # Restore
            inner.layers = nn.ModuleList(original_layers)
            if hasattr(model.config, 'text_config'):
                model.config.text_config.num_hidden_layers = N
            else:
                model.config.num_hidden_layers = N

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)
    return float(np.mean(rhos)) if rhos else 1.0

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Known best single from spectral screen
best_single = (25, 30)

# Top blocks from spectral screen (from generalization_results.json)
spectral_top = [(25,32),(25,35),(20,30),(45,55),(55,62),(25,30),(55,60),(20,27),(40,50),(40,47),(50,55),(30,40),(35,45),(50,60)]

print(f'Best single: ({best_single[0]},{best_single[1]})')

# Baseline
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
combined_base = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}')

# Best single score
order = build_order([best_single], N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
if hasattr(model.config, 'text_config'):
    model.config.text_config.num_hidden_layers = len(order)
else:
    model.config.num_hidden_layers = len(order)

math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, verbose=False)
best_score = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Best single ({best_single[0]},{best_single[1]}): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={best_score:.2f}')

# Restore
inner.layers = nn.ModuleList(original_layers)
if hasattr(model.config, 'text_config'):
    model.config.text_config.num_hidden_layers = N
else:
    model.config.num_hidden_layers = N

# Screen for second block on modified model
print('\n=== Screening second blocks on modified model ===')
second_candidates = [b for b in spectral_top if b[1] <= best_single[0] or b[0] >= best_single[1]]

# Apply best single first
order_a = build_order([best_single], N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
if hasattr(model.config, 'text_config'):
    model.config.text_config.num_hidden_layers = len(order_a)
else:
    model.config.num_hidden_layers = len(order_a)

second_rhos = {}
for block in second_candidates:
    second_rhos[block] = compute_rho(block)

# Restore
inner.layers = nn.ModuleList(original_layers)
if hasattr(model.config, 'text_config'):
    model.config.text_config.num_hidden_layers = N
else:
    model.config.num_hidden_layers = N

sorted_second = sorted(second_rhos.items(), key=lambda x: x[1])
print('Top 8 second blocks:')
for b, r in sorted_second[:8]:
    print(f'  ({b[0]:2d},{b[1]:2d}): rho={r:.4f}')

# Evaluate top-8 pairs
pair_results = []
for block_b, _ in sorted_second[:8]:
    pair = sorted([best_single, block_b])
    name = '+'.join(f'({b[0]},{b[1]})' for b in pair)

    order = build_order(pair, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = len(order)
    else:
        model.config.num_hidden_layers = len(order)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - best_score

    print(f'  {name}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} vs best single: {delta:+.2f}')
    pair_results.append({'blocks': [list(b) for b in pair], 'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta_vs_best_single': delta})

    inner.layers = nn.ModuleList(original_layers)
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = N
    else:
        model.config.num_hidden_layers = N

# Summary
print('\n=== SUMMARY ===')
print(f'Model: Qwen3.5-27B ({N} layers)')
print(f'Baseline: combined={combined_base:.2f}')
print(f'Best single: ({best_single[0]},{best_single[1]}) combined={best_score:.2f}')
best_pair = max(pair_results, key=lambda x: x['combined']) if pair_results else None
if best_pair:
    print(f'Best pair: {best_pair[\"name\"]} combined={best_pair[\"combined\"]:.2f}')
    if best_pair['combined'] > best_score:
        print('*** PAIR BEATS SINGLE! Greedy stacking works on Qwen3.5-27B! ***')
    else:
        print('Pair does not beat single.')

os.makedirs('results/data/qwen35', exist_ok=True)
results = {
    'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': combined_base},
    'best_single': {'block': list(best_single), 'combined': best_score},
    'second_block_screen': [{'block': list(b), 'rho': r} for b, r in sorted_second],
    'pairs': pair_results,
}
with open('results/data/qwen35/greedy_stacking.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to results/data/qwen35/greedy_stacking.json')
"

echo "=== Done at $(date) ==="
