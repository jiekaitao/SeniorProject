#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_alpha_crossval_%j.log
#SBATCH --job-name=deeppass_axval

# CRITICAL: Cross-validate alpha optimization
# Alphas were optimized on 16 math questions (set A) + 20 EQ-bench questions
# Test: do they generalize to UNSEEN questions?
# 1. Evaluate on completely different math questions (set C from prompt sensitivity)
# 2. Evaluate on full 171-question EQ-bench (not just 20)
# 3. Test metric weighting sensitivity (math-only, eq-only, different ratios)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Alpha Cross-Validation ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('ALPHA CROSS-VALIDATION')
print('Do alphas optimized on Set A generalize to unseen questions?')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

# Import math_probe internals for custom question sets
from math_probe import calculate_score, extract_number, USER_TEMPLATE, SYSTEM_PROMPT

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def generate_per_layer_alpha(prompt, blocks, layer_alphas, max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)
    step_alphas = {}
    for block in sorted_blocks:
        i, j = block
        block_layers = list(range(i, j))
        count = {}
        offset = 0
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:
                    step_alphas[step] = layer_alphas.get((block, offset), 1.0)
                    offset += 1
    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                h_before = h.clone()
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h_after = out[0] if isinstance(out, tuple) else out
                if step_idx in step_alphas:
                    h = h_before + step_alphas[step_idx] * (h_after - h_before)
                else:
                    h = h_after
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def run_custom_math_probe(gen_fn, questions):
    scores = []
    for q, expected in questions:
        full_prompt = f'{SYSTEM_PROMPT}\\n\\n{USER_TEMPLATE.format(question=q)}'
        response = gen_fn(full_prompt)
        extracted = extract_number(response)
        score = calculate_score(extracted, expected)
        scores.append(score)
    return {'score': float(np.mean(scores)), 'n': len(scores), 'scores': scores}

# =====================================================================
# Question Sets
# =====================================================================

# Set A: Original 16 questions (used for optimization)
SET_A = [
    ('What is 127 * 348?', 44196), ('What is 99999 * 99999?', 9999800001),
    ('If a train travels at 60 mph for 2.5 hours, how far does it go?', 150),
    ('What is the square root of 1764?', 42), ('Calculate 15! / 13!', 210),
    ('A rectangle has area 48 and perimeter 28. What are its dimensions? Give the larger one.', 8),
    ('What is 2^16?', 65536),
    ('If you have 3 red balls and 5 blue balls, what is the probability of drawing 2 red? Give as decimal.', 0.107),
    ('What is the sum of all integers from 1 to 100?', 5050),
    ('A car depreciates 15% per year. After 3 years, what fraction of value remains? Give as decimal.', 0.614),
    ('What is 7^5?', 16807), ('If f(x) = 3x^2 - 2x + 1, what is f(5)?', 66),
    ('A store has a 30% sale, then another 20% off. What is the total discount percentage?', 44),
    ('What is the derivative of x^3 at x=4?', 48),
    ('Three friends split a 147 dollar bill with 20% tip. How much does each pay?', 58.8),
    ('What is 1/7 as a decimal to 6 places?', 0.142857),
]

# Set C: Completely different questions (NEVER seen during optimization)
SET_C = [
    ('What is 847 * 293?', 248171), ('Calculate 17^3', 4913),
    ('What is 65536 / 256?', 256), ('What is the cube root of 27000?', 30),
    ('If a pool fills at 3 gallons per minute, how many minutes to fill 450 gallons?', 150),
    ('What is 11! / 9!', 110), ('Calculate 2^20', 1048576),
    ('A circle has radius 7. What is its area? Use pi=3.14159', 153.938),
    ('What is 999 * 1001?', 999999), ('What is 13^2 + 14^2?', 365),
    ('If you invest 1000 at 5% compound interest for 3 years, what is the total?', 1157.625),
    ('What is the sum of all odd numbers from 1 to 50?', 625),
    ('A triangle has sides 3, 4, 5. What is its area?', 6),
    ('What is 19 * 23 * 2?', 874), ('Calculate 100! / 98!', 9900),
    ('What is 2.5^3?', 15.625),
]

# Set D: Word problems (different style)
SET_D = [
    ('A factory makes 340 widgets per hour. How many in 7.5 hours?', 2550),
    ('If 8 workers finish a job in 6 days, how many days for 12 workers?', 4),
    ('A recipe needs 2.5 cups of flour for 12 cookies. How much for 30 cookies?', 6.25),
    ('Train A goes 80 mph, Train B goes 60 mph. They start 420 miles apart heading toward each other. When do they meet? Give hours.', 3),
    ('A shirt costs 45 dollars after a 25% discount. What was the original price?', 60),
    ('You flip a fair coin 3 times. What is the probability of exactly 2 heads? Give as decimal.', 0.375),
    ('A ladder 13 feet long leans against a wall. The base is 5 feet from the wall. How high does it reach?', 12),
    ('Population doubles every 15 years. Starting at 1000, what is it after 45 years?', 8000),
    ('A car gets 32 mpg. Gas costs 3.50 per gallon. How much to drive 560 miles?', 61.25),
    ('What is 15% of 15% of 10000?', 225),
    ('A box is 4x5x6 inches. What is its volume?', 120),
    ('If x + y = 10 and x - y = 4, what is x?', 7),
    ('A 20% tip on a 85 dollar meal is how much?', 17),
    ('What is the average of 23, 45, 67, 89, 11?', 47),
    ('How many seconds in 2.5 hours?', 9000),
    ('A square has diagonal 10. What is its area?', 50),
]

# =====================================================================
# Configs to test
# =====================================================================

BLOCKS_TRIPLE = [(0, 7), (20, 27), (45, 52)]

# Best per-layer alphas from grid search (optimized on Set A)
BEST_ALPHAS = {}
# (0,7) block
for i, a in enumerate([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]):
    BEST_ALPHAS[((0,7), i)] = a
# (20,27) block
for i, a in enumerate([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]):
    BEST_ALPHAS[((20,27), i)] = a
# (45,52) block — per-layer optimized
for i, a in enumerate([1.1, 1.0, 0.5, 1.3, 1.0, 0.9, 1.1]):
    BEST_ALPHAS[((45,52), i)] = a

# Uniform alpha=1.0 for comparison
UNIFORM_ALPHAS = {(b, i): 1.0 for b in BLOCKS_TRIPLE for i in range(b[1]-b[0])}

CONFIGS = [
    ('baseline', None, None),
    ('triple @1.0 uniform', BLOCKS_TRIPLE, UNIFORM_ALPHAS),
    ('triple per-layer optimized', BLOCKS_TRIPLE, BEST_ALPHAS),
    ('single (45,52) per-layer', [(45,52)], {((45,52), i): a for i, a in enumerate([1.1, 1.0, 0.5, 1.3, 1.0, 0.9, 1.1])}),
]

all_results = {}

for config_name, blocks, alphas in CONFIGS:
    print(f'\\n--- {config_name} ---', flush=True)

    if blocks is None:
        gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
        gen_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
    else:
        gen = lambda p, b=blocks, a=alphas: generate_per_layer_alpha(p, b, a, max_new_tokens=64)
        gen_long = lambda p, b=blocks, a=alphas: generate_per_layer_alpha(p, b, a, max_new_tokens=128)

    results = {}

    # Math on all 3 question sets
    for set_name, questions in [('Set_A (train)', SET_A), ('Set_C (unseen)', SET_C), ('Set_D (word)', SET_D)]:
        r = run_custom_math_probe(gen, questions)
        results[set_name] = r
        print(f'  {set_name}: {r[\"score\"]:.4f} ({r[\"n\"]} questions)', flush=True)

    # EQ-bench (standard 20 questions)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    results['eq_bench_20'] = eq_r['score']
    print(f'  EQ-bench (20q): {eq_r[\"score\"]:.1f}', flush=True)

    # Combined scores at different weightings
    for math_set in ['Set_A (train)', 'Set_C (unseen)', 'Set_D (word)']:
        math_s = results[math_set]['score']
        eq_s = results['eq_bench_20']
        c_default = math_s * 50 + eq_s * 0.5  # our standard
        c_math_heavy = math_s * 80 + eq_s * 0.2
        c_eq_heavy = math_s * 20 + eq_s * 0.8
        c_math_only = math_s * 100
        c_eq_only = eq_s
        results[f'{math_set}_combined'] = c_default
        results[f'{math_set}_math_heavy'] = c_math_heavy
        results[f'{math_set}_eq_heavy'] = c_eq_heavy

    all_results[config_name] = results

# =====================================================================
# ANALYSIS
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('CROSS-VALIDATION ANALYSIS')
print(f'{\"=\" * 70}', flush=True)

print(f'\\n--- Does the improvement generalize to unseen questions? ---')
for config_name in ['baseline', 'triple per-layer optimized']:
    r = all_results[config_name]
    print(f'{config_name}:')
    for s in ['Set_A (train)', 'Set_C (unseen)', 'Set_D (word)']:
        print(f'  {s}: math={r[s][\"score\"]:.4f} combined={r[f\"{s}_combined\"]:.2f}')

print(f'\\n--- Deltas (optimized - baseline) ---')
bl = all_results['baseline']
opt = all_results['triple per-layer optimized']
for s in ['Set_A (train)', 'Set_C (unseen)', 'Set_D (word)']:
    d_math = opt[s]['score'] - bl[s]['score']
    d_combined = opt[f'{s}_combined'] - bl[f'{s}_combined']
    print(f'  {s}: math delta={d_math:+.4f} combined delta={d_combined:+.2f}')

# Check: does config ranking change with different weightings?
print(f'\\n--- Config ranking stability across metric weightings ---')
for weighting in ['_combined', '_math_heavy', '_eq_heavy']:
    ranking = []
    for config_name in ['baseline', 'triple @1.0 uniform', 'triple per-layer optimized', 'single (45,52) per-layer']:
        r = all_results[config_name]
        score = r.get(f'Set_C (unseen){weighting}', 0)
        ranking.append((config_name, score))
    ranking.sort(key=lambda x: x[1], reverse=True)
    w_name = weighting.replace('_', '')
    print(f'  {w_name}: {\" > \".join(f\"{n}({s:.1f})\" for n, s in ranking)}')

# Bootstrap CIs on the key comparison (Set C)
print(f'\\n--- Bootstrap 95% CIs on Set C ---')
for config_name in ['baseline', 'triple per-layer optimized']:
    scores = all_results[config_name]['Set_C (unseen)']['scores']
    bootstraps = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(bootstraps, [2.5, 97.5])
    print(f'  {config_name}: math={np.mean(scores):.4f} 95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]')

os.makedirs('results/data/72b/alpha_crossval', exist_ok=True)
with open('results/data/72b/alpha_crossval/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'results': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in all_results.items()}}, f, indent=2, default=str)
print(f'\\nSaved to results/data/72b/alpha_crossval/results.json', flush=True)
"

echo "=== Done at $(date) ==="
