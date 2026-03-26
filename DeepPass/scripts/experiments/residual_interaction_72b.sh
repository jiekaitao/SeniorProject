#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_residual_interaction_%j.log
#SBATCH --job-name=deeppass_resid_int

# Residual interaction experiment on 72B:
# Cheaply measure pairwise block interaction by comparing how block B's
# second-pass residual changes when block A is already applied.
# ~1 min per pair vs ~8 min for full probe. ~20 min total for ~20 pairs.
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Residual Interaction Experiment (72B) ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn, numpy as np
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded calme-2.1-qwen2-72b: {N} layers')

# 32 calibration prompts (math + reasoning, same spirit as DICE)
cal_prompts = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'Calculate 15! / 13!',
    'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
    'What is 256 * 512?',
    'What is the derivative of x^3 + 2x^2 - 5x + 3?',
    'Solve: 2x + 7 = 23',
    'What is the square root of 1764?',
    'If a train travels 60 mph for 2.5 hours, how far does it go?',
    'What is 17^2 + 13^2?',
    'Calculate the area of a circle with radius 7.',
    'What is 1000 / 7 rounded to two decimal places?',
    'What is the 10th term of the Fibonacci sequence?',
    'If you flip a coin 3 times, how many possible outcomes?',
    'What is the GCD of 48 and 180?',
    'Simplify: (3/4) * (8/9)',
    'A rectangle has perimeter 30 and width 5. What is its area?',
    'What is log base 2 of 1024?',
    'How many prime numbers are between 1 and 50?',
    'What is 3^5 - 2^8?',
    'Convert 0.375 to a fraction in lowest terms.',
    'What is the volume of a cube with side length 6?',
    'If 5x - 3 = 2x + 9, what is x?',
    'What is the median of [3, 7, 1, 9, 4, 6, 2]?',
    'Calculate: 123 + 456 + 789',
    'What is 7! (7 factorial)?',
    'A car depreciates 15% per year. After 2 years of a 20000 car, what is it worth?',
    'What is the sum of interior angles of a hexagon?',
]

# Top blocks from our 72B results
TOP_BLOCKS = [
    (0, 7), (45, 52), (50, 60), (15, 20), (55, 62),
    (52, 60), (20, 27), (35, 40), (10, 17), (25, 32),
    (5, 12), (30, 37), (60, 65), (40, 47), (40, 45),
    (45, 50), (50, 55), (55, 60), (35, 45),
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

def collect_residuals(block_b, applied_blocks=[]):
    \"\"\"Collect h2-h1 for block_b, optionally with other blocks already applied.

    h2-h1 is measured as the logit-space difference caused by block_b's duplication.
    We run the model twice: once with only applied_blocks, once with applied_blocks + block_b.
    The difference in last-token logits is the 'residual' of block_b's second pass.
    \"\"\"
    i, j = block_b

    residuals = []
    for prompt in cal_prompts:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)

        # Build layer order with applied_blocks only (no block_b duplication)
        order_without = build_order(applied_blocks, N) if applied_blocks else list(range(N))
        # Build layer order with applied_blocks + block_b
        order_with = build_order(applied_blocks + [block_b], N)

        # Forward with just applied_blocks
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_without])
        model.config.num_hidden_layers = len(order_without)
        with torch.no_grad():
            out1 = model(ids['input_ids'], use_cache=False)
            logits1 = out1.logits[:, -1, :].float()  # last token logits

        # Forward with applied_blocks + block_b
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_with])
        model.config.num_hidden_layers = len(order_with)
        with torch.no_grad():
            out2 = model(ids['input_ids'], use_cache=False)
            logits2 = out2.logits[:, -1, :].float()

        # Residual = logit difference caused by block_b's duplication
        residual = (logits2 - logits1).squeeze(0)
        residuals.append(residual)

    # Restore original model
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    return torch.stack(residuals)  # [n_prompts, vocab_size]

# For each pair of non-overlapping blocks, measure interaction
results = []
n_pairs = 0

# Pre-count pairs for progress reporting
total_pairs = 0
for idx_a, block_a in enumerate(TOP_BLOCKS):
    for block_b in TOP_BLOCKS[idx_a+1:]:
        if block_b[0] < block_a[1] and block_a[0] < block_b[1]:
            continue  # overlapping
        total_pairs += 1
print(f'Will evaluate {total_pairs} non-overlapping pairs')

for idx_a, block_a in enumerate(TOP_BLOCKS):
    # Collect block_a's residual on base model
    res_a_base = collect_residuals(block_a, applied_blocks=[])

    for block_b in TOP_BLOCKS[idx_a+1:]:
        # Skip overlapping blocks
        if block_b[0] < block_a[1] and block_a[0] < block_b[1]:
            continue

        n_pairs += 1
        pair = sorted([block_a, block_b])
        name = f'({pair[0][0]},{pair[0][1]})+({pair[1][0]},{pair[1][1]})'
        print(f'[{n_pairs}/{total_pairs}] {name} ...', flush=True)

        # Block B's residual on base model
        res_b_base = collect_residuals(block_b, applied_blocks=[])

        # Block B's residual with block A applied
        res_b_with_a = collect_residuals(block_b, applied_blocks=[block_a])

        # Block A's residual with block B applied
        res_a_with_b = collect_residuals(block_a, applied_blocks=[block_b])

        # Compute interaction metrics
        # Flatten residuals for cosine similarity
        cos_b = torch.nn.functional.cosine_similarity(
            res_b_base.flatten().unsqueeze(0),
            res_b_with_a.flatten().unsqueeze(0)
        ).item()

        cos_a = torch.nn.functional.cosine_similarity(
            res_a_base.flatten().unsqueeze(0),
            res_a_with_b.flatten().unsqueeze(0)
        ).item()

        # Also compute per-prompt cosine similarity for block B
        per_prompt_cos_b = []
        for p in range(len(cal_prompts)):
            c = torch.nn.functional.cosine_similarity(
                res_b_base[p].unsqueeze(0), res_b_with_a[p].unsqueeze(0)
            ).item()
            per_prompt_cos_b.append(c)

        # Per-prompt cosine similarity for block A
        per_prompt_cos_a = []
        for p in range(len(cal_prompts)):
            c = torch.nn.functional.cosine_similarity(
                res_a_base[p].unsqueeze(0), res_a_with_b[p].unsqueeze(0)
            ).item()
            per_prompt_cos_a.append(c)

        mean_stability = (cos_a + cos_b) / 2

        results.append({
            'name': name,
            'blocks': [list(b) for b in pair],
            'cos_b_stability': cos_b,   # how stable B's residual is when A is applied
            'cos_a_stability': cos_a,   # how stable A's residual is when B is applied
            'mean_stability': mean_stability,
            'per_prompt_cos_b': per_prompt_cos_b,
            'per_prompt_cos_a': per_prompt_cos_a,
            'per_prompt_cos_b_std': float(np.std(per_prompt_cos_b)),
            'per_prompt_cos_a_std': float(np.std(per_prompt_cos_a)),
        })

        print(f'  cos_b={cos_b:.4f} cos_a={cos_a:.4f} mean={mean_stability:.4f}')

# Sort by stability (highest = least interference = most independent)
results.sort(key=lambda x: x['mean_stability'], reverse=True)

print(f'\\n=== RESULTS (sorted by stability, {len(results)} pairs) ===')
print(f'{\"Pair\":<30s} {\"cos_b\":>8s} {\"cos_a\":>8s} {\"mean\":>8s}')
print('-' * 58)
for r in results:
    print(f'{r[\"name\"]:<30s} {r[\"cos_b_stability\"]:8.4f} {r[\"cos_a_stability\"]:8.4f} {r[\"mean_stability\"]:8.4f}')

print('\\n=== TOP 10 MOST INDEPENDENT PAIRS ===')
for r in results[:10]:
    print(f'  {r[\"name\"]}: stability={r[\"mean_stability\"]:.4f}')

print('\\n=== TOP 5 MOST INTERACTING PAIRS ===')
for r in results[-5:]:
    print(f'  {r[\"name\"]}: stability={r[\"mean_stability\"]:.4f}')

os.makedirs('results/data/72b/residual_interaction', exist_ok=True)
with open('results/data/72b/residual_interaction/interaction_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\\nSaved to results/data/72b/residual_interaction/interaction_results.json')
"

echo "=== Done at $(date) ==="
