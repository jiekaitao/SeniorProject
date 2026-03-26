#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_prompt_sens_%j.log
#SBATCH --job-name=deeppass_psens

# Prompt Sensitivity Ablation
# Tests whether config rankings are robust to calibration prompt selection.
# 5 prompt sets x 4 configs = 20 math probe evaluations + 4 EQ-bench evaluations.
# Reports Spearman rank correlation of config rankings across prompt sets.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Prompt Sensitivity Ablation ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, random, torch, torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import calculate_score, extract_number, SYSTEM_PROMPT, USER_TEMPLATE
from eq_bench_probe import run_eq_bench_probe

# ============================================================
# Define 5 prompt sets (each 16 questions)
# ============================================================

# Set A: original 16 from math_probe.py
SET_A = [
    {'question': 'What is 78313086360375 multiplied by 88537453126609?', 'answer': 6933174468959498727528375},
    {'question': 'What is the cube root of 74088893247?', 'answer': 4201},
    {'question': 'What is the cube root of 18228885506341?', 'answer': 26317},
    {'question': 'What is the cube root of 844178022493, multiplied by 43?', 'answer': 40549},
    {'question': 'What is 9999999 multiplied by 9999999?', 'answer': 99999980000001},
    {'question': 'What is 123456789 multiplied by 987654321?', 'answer': 121932631112635269},
    {'question': 'What is the square root of 152399025?', 'answer': 12345},
    {'question': 'What is 7777777 multiplied by 3333333?', 'answer': 25925923703641},
    {'question': 'What is 456789 raised to the power of 2?', 'answer': 208655854521},
    {'question': 'What is 11111111 multiplied by 11111111?', 'answer': 123456787654321},
    {'question': 'What is the cube root of 2744000?', 'answer': 140},
    {'question': 'What is 314159 multiplied by 271828?', 'answer': 85397342252},
    {'question': 'What is 999999999999 divided by 142857, rounded to the nearest integer?', 'answer': 6999993},
    {'question': 'What is 2 raised to the power of 48?', 'answer': 281474976710656},
    {'question': 'What is the square root of 99980001?', 'answer': 9999},
    {'question': 'What is 54321 multiplied by 12345?', 'answer': 670592745},
]

# Set B: replace 8 questions with new hard arithmetic (keep first 8 from A, swap last 8)
SET_B = SET_A[:8] + [
    {'question': 'What is 847 multiplied by 293?', 'answer': 248171},
    {'question': 'What is 17 raised to the power of 3?', 'answer': 4913},
    {'question': 'What is 65536 multiplied by 65536?', 'answer': 4294967296},
    {'question': 'What is the square root of 5764801?', 'answer': 2401},
    {'question': 'What is 88888 multiplied by 77777?', 'answer': 6913580136},
    {'question': 'What is 3 raised to the power of 20?', 'answer': 3486784401},
    {'question': 'What is 142857 multiplied by 7?', 'answer': 999999},
    {'question': 'What is the cube root of 8000000?', 'answer': 200},
]

# Set C: completely different 16 arithmetic questions (no overlap with A)
SET_C = [
    {'question': 'What is 847293 multiplied by 192837?', 'answer': 163399178841},
    {'question': 'What is 19 raised to the power of 4?', 'answer': 130321},
    {'question': 'What is the square root of 1522756?', 'answer': 1234},
    {'question': 'What is 777777 multiplied by 222222?', 'answer': 172838827494},
    {'question': 'What is 2 raised to the power of 32?', 'answer': 4294967296},
    {'question': 'What is 555555 multiplied by 444444?', 'answer': 246913086420},
    {'question': 'What is the cube root of 12167000?', 'answer': 230},
    {'question': 'What is 98765 multiplied by 56789?', 'answer': 5609791985},
    {'question': 'What is 13 raised to the power of 5?', 'answer': 371293},
    {'question': 'What is 333333 multiplied by 666666?', 'answer': 222221777778},
    {'question': 'What is the square root of 1048576?', 'answer': 1024},
    {'question': 'What is 876543 multiplied by 345678?', 'answer': 302895884454},
    {'question': 'What is 7 raised to the power of 8?', 'answer': 5764801},
    {'question': 'What is 999999 multiplied by 111111?', 'answer': 111110888889},
    {'question': 'What is the cube root of 27000000?', 'answer': 300},
    {'question': 'What is 246813 multiplied by 135792?', 'answer': 33517718496},
]

# Set D: 16 questions mixing arithmetic with word problems
SET_D = [
    {'question': 'What is 847 multiplied by 293?', 'answer': 248171},
    {'question': 'A factory produces 4567 widgets per hour. How many widgets in 89 hours?', 'answer': 406463},
    {'question': 'What is the square root of 7744?', 'answer': 88},
    {'question': 'If a train travels 234 miles every 3 hours, how many miles in 17 hours? Round to the nearest integer.', 'answer': 1326},
    {'question': 'What is 5 raised to the power of 9?', 'answer': 1953125},
    {'question': 'A store sells 1247 items per day. How many items in 365 days?', 'answer': 455155},
    {'question': 'What is the cube root of 3375000?', 'answer': 150},
    {'question': 'If you invest 10000 at 8 percent annual compound interest, what is the value after 10 years? Round to the nearest integer.', 'answer': 21589},
    {'question': 'What is 88888 multiplied by 11111?', 'answer': 987642568},
    {'question': 'A pool fills at 347 gallons per hour. How many gallons in one week (168 hours)?', 'answer': 58296},
    {'question': 'What is 23 raised to the power of 3?', 'answer': 12167},
    {'question': 'A car uses 3.7 gallons per 100 miles. How many gallons for 2750 miles? Round to the nearest integer.', 'answer': 102},
    {'question': 'What is 654321 multiplied by 123456?', 'answer': 80779853376},
    {'question': 'If 15 workers can build a wall in 48 hours, how many hours for 36 workers? Round to the nearest integer.', 'answer': 20},
    {'question': 'What is the square root of 16384?', 'answer': 128},
    {'question': 'A warehouse ships 9876 packages per week. How many in 52 weeks?', 'answer': 513552},
]

# Set E: random subsets of 12 from original 16 (5 random draws)
# We do 5 random sub-samples of size 12 from SET_A and average the results.
random.seed(42)
SET_E_DRAWS = []
for _ in range(5):
    indices = sorted(random.sample(range(16), 12))
    SET_E_DRAWS.append([SET_A[i] for i in indices])

PROMPT_SETS = {
    'A_original': SET_A,
    'B_half_replaced': SET_B,
    'C_all_new': SET_C,
    'D_mixed_word': SET_D,
    # Set E is handled specially (5 draws, averaged)
}

# ============================================================
# Configs to test
# ============================================================
CONFIGS = {
    'baseline': [],
    'ng_45_52': [(45, 52)],
    'ours_50_60': [(50, 60)],
    'pair_0_7_45_52': [(0, 7), (45, 52)],
}


def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order


def run_custom_math_probe(generate_fn, questions, verbose=False):
    \"\"\"Run math probe with custom question set (same logic as math_probe.run_math_probe).\"\"\"
    scores = []
    details = []
    for idx, q in enumerate(questions):
        prompt = USER_TEMPLATE.format(question=q['question'])
        full_prompt = f'System: {SYSTEM_PROMPT}\\n\\nUser: {prompt}\\n\\nAssistant:'
        response = generate_fn(full_prompt)
        estimated = extract_number(response)
        try:
            score = calculate_score(q['answer'], estimated)
        except Exception:
            score = 0.0
        scores.append(score)
        details.append({
            'question': q['question'],
            'actual': q['answer'],
            'estimated': estimated,
            'raw_response': response,
            'score': score,
        })
        if verbose:
            status = 'OK' if score > 0.5 else 'MISS'
            print(f'  [{idx+1:2d}/{len(questions)}] {status} score={score:.4f} actual={q[\"answer\"]} got={estimated}', flush=True)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {'score': avg_score, 'scores': scores, 'details': details}


# ============================================================
# Main
# ============================================================
print('=' * 70, flush=True)
print('PROMPT SENSITIVITY ABLATION', flush=True)
print(f'Date: {datetime.now().isoformat()}', flush=True)
print(f'Prompt sets: A (original), B (half replaced), C (all new), D (mixed word), E (5x random 12-subset)', flush=True)
print(f'Configs: {list(CONFIGS.keys())}', flush=True)
print('=' * 70, flush=True)

t0 = time.time()

# Load model
print('\\nLoading calme-2.1-qwen2-72b...', flush=True)
model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
print(f'Loaded: {N} layers', flush=True)

def gen(p):
    return generate_no_cache(model, tokenizer, p, max_new_tokens=64)

def gen_long(p):
    return generate_no_cache(model, tokenizer, p, max_new_tokens=128)


def apply_config(config_blocks):
    \"\"\"Apply a config (list of (i,j) blocks) or restore baseline.\"\"\"
    if config_blocks:
        order = build_order(config_blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)
    else:
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N


# ============================================================
# Phase 1: Math probe across all prompt sets x configs
# ============================================================
all_math_results = {}  # {prompt_set_name: {config_name: score}}

# Sets A-D
for set_name, questions in PROMPT_SETS.items():
    print(f'\\n{\"=\" * 50}', flush=True)
    print(f'PROMPT SET: {set_name} ({len(questions)} questions)', flush=True)
    print('=' * 50, flush=True)
    all_math_results[set_name] = {}

    for config_name, blocks in CONFIGS.items():
        apply_config(blocks)
        nlayers = model.config.num_hidden_layers
        print(f'  Config {config_name:20s} (layers={nlayers}): ', end='', flush=True)
        result = run_custom_math_probe(gen, questions, verbose=False)
        all_math_results[set_name][config_name] = result['score']
        print(f'math={result[\"score\"]:.4f}', flush=True)
        # Restore baseline between configs
        apply_config([])

# Set E: 5 random draws of 12 questions, average per config
print(f'\\n{\"=\" * 50}', flush=True)
print(f'PROMPT SET: E_random_12 (5 draws x 12 questions, averaged)', flush=True)
print('=' * 50, flush=True)
e_scores_per_config = {cn: [] for cn in CONFIGS}
for draw_idx, draw_questions in enumerate(SET_E_DRAWS):
    print(f'  Draw {draw_idx+1}/5:', flush=True)
    for config_name, blocks in CONFIGS.items():
        apply_config(blocks)
        result = run_custom_math_probe(gen, draw_questions, verbose=False)
        e_scores_per_config[config_name].append(result['score'])
        print(f'    {config_name:20s}: math={result[\"score\"]:.4f}', flush=True)
        apply_config([])

all_math_results['E_random_12'] = {}
for config_name, scores_list in e_scores_per_config.items():
    avg = sum(scores_list) / len(scores_list)
    std = float(np.std(scores_list))
    all_math_results['E_random_12'][config_name] = avg
    print(f'  E avg {config_name:20s}: {avg:.4f} +/- {std:.4f}', flush=True)


# ============================================================
# Phase 2: EQ-bench (fixed prompts) — just to confirm config ranking
# ============================================================
print(f'\\n{\"=\" * 50}', flush=True)
print('EQ-BENCH (fixed prompts, for reference)', flush=True)
print('=' * 50, flush=True)

eq_results = {}
for config_name, blocks in CONFIGS.items():
    apply_config(blocks)
    nlayers = model.config.num_hidden_layers
    print(f'  Config {config_name:20s} (layers={nlayers}): ', end='', flush=True)
    result = run_eq_bench_probe(gen_long, verbose=False)
    eq_results[config_name] = result['score']
    print(f'eq={result[\"score\"]:.1f}/100 parse_rate={result[\"parse_rate\"]:.0%}', flush=True)
    apply_config([])


# ============================================================
# Phase 3: Analysis — Spearman rank correlation across prompt sets
# ============================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('ANALYSIS: Config ranking consistency across prompt sets', flush=True)
print('=' * 70, flush=True)

config_names = list(CONFIGS.keys())
set_names = list(all_math_results.keys())

# Build ranking matrix: for each prompt set, rank the 4 configs
# rank 1 = best (highest score)
rankings = {}  # {set_name: [rank_of_config_0, rank_of_config_1, ...]}
for set_name in set_names:
    scores = [all_math_results[set_name][cn] for cn in config_names]
    # argsort descending: best gets rank 1
    order = np.argsort(scores)[::-1]
    rank = np.empty(len(scores), dtype=int)
    for r, idx in enumerate(order):
        rank[idx] = r + 1
    rankings[set_name] = list(rank)

print(f'\\nConfig rankings (1=best) by prompt set:', flush=True)
header = f'{\"\":>20s}' + ''.join(f'{cn:>20s}' for cn in config_names)
print(header, flush=True)
print('-' * len(header), flush=True)
for set_name in set_names:
    scores_str = ''.join(f'{all_math_results[set_name][cn]:>15.4f}(#{rankings[set_name][i]})' for i, cn in enumerate(config_names))
    print(f'{set_name:>20s}{scores_str}', flush=True)

# Pairwise Spearman between all prompt set pairs
print(f'\\nPairwise Spearman rank correlation between prompt sets:', flush=True)
rho_matrix = np.zeros((len(set_names), len(set_names)))
for i, sn1 in enumerate(set_names):
    for j, sn2 in enumerate(set_names):
        scores1 = [all_math_results[sn1][cn] for cn in config_names]
        scores2 = [all_math_results[sn2][cn] for cn in config_names]
        rho, pval = spearmanr(scores1, scores2)
        rho_matrix[i, j] = rho

print(f'{\"\":>20s}' + ''.join(f'{sn:>18s}' for sn in set_names), flush=True)
for i, sn1 in enumerate(set_names):
    row = f'{sn1:>20s}'
    for j, sn2 in enumerate(set_names):
        row += f'{rho_matrix[i,j]:>18.3f}'
    print(row, flush=True)

# Mean pairwise rho (off-diagonal)
off_diag = []
for i in range(len(set_names)):
    for j in range(len(set_names)):
        if i != j:
            off_diag.append(rho_matrix[i, j])
mean_rho = np.mean(off_diag)
min_rho = np.min(off_diag)
max_rho = np.max(off_diag)
print(f'\\nMean pairwise Spearman rho: {mean_rho:.3f} (min={min_rho:.3f}, max={max_rho:.3f})', flush=True)

# Check: does every prompt set agree on best config?
best_per_set = {}
for set_name in set_names:
    scores = {cn: all_math_results[set_name][cn] for cn in config_names}
    best = max(scores, key=scores.get)
    best_per_set[set_name] = best
    print(f'  Best config for {set_name:>20s}: {best} (score={scores[best]:.4f})', flush=True)

all_agree = len(set(best_per_set.values())) == 1
print(f'\\nAll prompt sets agree on best config: {\"YES\" if all_agree else \"NO\"}', flush=True)
if all_agree:
    print(f'  Unanimous best: {list(best_per_set.values())[0]}', flush=True)
else:
    from collections import Counter
    counts = Counter(best_per_set.values())
    print(f'  Best config votes: {dict(counts)}', flush=True)

# EQ-bench ranking for reference
eq_ranking = sorted(config_names, key=lambda cn: eq_results[cn], reverse=True)
print(f'\\nEQ-bench ranking: {\" > \".join(eq_ranking)}', flush=True)
for cn in eq_ranking:
    print(f'  {cn:>20s}: {eq_results[cn]:.1f}/100', flush=True)


# ============================================================
# Phase 4: Kendall W (coefficient of concordance) across all sets
# ============================================================
print(f'\\n{\"=\" * 50}', flush=True)
print('Kendall W (concordance) across all 5 prompt sets', flush=True)
print('=' * 50, flush=True)

# Kendall W: measures agreement among m judges ranking n items
# W = 12 * S / (m^2 * (n^3 - n))  where S = sum((Rj - Rbar)^2)
m = len(set_names)  # judges (prompt sets)
n = len(config_names)  # items (configs)
R = np.zeros(n)  # sum of ranks for each config
for set_name in set_names:
    for ci, cn in enumerate(config_names):
        R[ci] += rankings[set_name][ci]

Rbar = np.mean(R)
S = np.sum((R - Rbar) ** 2)
W = 12 * S / (m ** 2 * (n ** 3 - n))
print(f'  Kendall W = {W:.4f} (1.0 = perfect agreement, 0.0 = no agreement)', flush=True)
if W > 0.7:
    print(f'  Interpretation: STRONG agreement -- results are robust to prompt selection', flush=True)
elif W > 0.4:
    print(f'  Interpretation: MODERATE agreement -- some sensitivity to prompt selection', flush=True)
else:
    print(f'  Interpretation: WEAK agreement -- results are sensitive to prompt selection', flush=True)

# Set E per-draw detail
print(f'\\n--- Set E per-draw detail ---', flush=True)
for draw_idx in range(5):
    draw_scores = {cn: e_scores_per_config[cn][draw_idx] for cn in config_names}
    best = max(draw_scores, key=draw_scores.get)
    scores_str = ' '.join(f'{cn}={draw_scores[cn]:.4f}' for cn in config_names)
    print(f'  Draw {draw_idx+1}: best={best:20s} | {scores_str}', flush=True)


# ============================================================
# Save results
# ============================================================
elapsed = time.time() - t0
print(f'\\nTotal elapsed: {elapsed/60:.1f} minutes', flush=True)

output = {
    'date': datetime.now().isoformat(),
    'model': 'calme-2.1-qwen2-72b',
    'num_layers': N,
    'configs': {cn: [list(b) for b in blocks] for cn, blocks in CONFIGS.items()},
    'prompt_sets': {
        'A_original': {'description': 'Original 16 questions from math_probe.py', 'num_questions': 16},
        'B_half_replaced': {'description': 'First 8 from A + 8 new hard arithmetic', 'num_questions': 16},
        'C_all_new': {'description': 'Completely different 16 arithmetic questions', 'num_questions': 16},
        'D_mixed_word': {'description': '16 questions mixing arithmetic with word problems', 'num_questions': 16},
        'E_random_12': {'description': '5 random draws of 12 from original 16, averaged', 'num_draws': 5, 'draw_size': 12},
    },
    'math_scores': all_math_results,
    'set_e_per_draw': {cn: scores for cn, scores in e_scores_per_config.items()},
    'eq_bench_scores': eq_results,
    'rankings': {sn: {cn: rankings[sn][ci] for ci, cn in enumerate(config_names)} for sn in set_names},
    'pairwise_spearman': {
        sn1: {sn2: float(rho_matrix[i, j]) for j, sn2 in enumerate(set_names)}
        for i, sn1 in enumerate(set_names)
    },
    'mean_pairwise_spearman': float(mean_rho),
    'min_pairwise_spearman': float(min_rho),
    'max_pairwise_spearman': float(max_rho),
    'kendall_w': float(W),
    'best_per_set': best_per_set,
    'all_agree_on_best': all_agree,
    'elapsed_minutes': elapsed / 60,
}

os.makedirs('results/data/72b/prompt_sensitivity', exist_ok=True)
outpath = 'results/data/72b/prompt_sensitivity/results.json'
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to {outpath}', flush=True)
print('DONE', flush=True)
"

echo "=== Done at $(date) ==="
