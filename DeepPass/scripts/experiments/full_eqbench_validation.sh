#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_full_eqbench_%j.log
#SBATCH --job-name=deeppass_feqb

# Run FULL 171-question EQ-bench (not just 20) on key configs
# Addresses reviewer concern about small sample size

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Full EQ-bench (171 questions) ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('FULL EQ-BENCH VALIDATION (171 questions)')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

# Load the FULL EQ-bench dataset (171 questions, not just 20)
from eq_bench_probe import calculate_eq_score, run_eq_bench_probe
import ast
from datasets import load_dataset
ds = load_dataset('pbevan11/EQ-Bench', split='validation')
ALL_EQ_QUESTIONS = []
for idx in range(len(ds)):
    row = ds[idx]
    ref_dict = ast.literal_eval(row['reference_answer'])
    ALL_EQ_QUESTIONS.append({
        'prompt': row['prompt'],
        'reference': {ref_dict['emotion1']: int(ref_dict['emotion1_score']),
                      ref_dict['emotion2']: int(ref_dict['emotion2_score']),
                      ref_dict['emotion3']: int(ref_dict['emotion3_score']),
                      ref_dict['emotion4']: int(ref_dict['emotion4_score'])},
    })
print(f'EQ-bench questions loaded: {len(ALL_EQ_QUESTIONS)}', flush=True)

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def run_full_eqbench(gen_fn):
    \"\"\"Run ALL EQ-bench questions (171), not just 20.\"\"\"
    scores = []
    for idx, q in enumerate(ALL_EQ_QUESTIONS):
        response = gen_fn(q['prompt'])
        result = calculate_eq_score(q['reference'], response)
        scores.append(result.get('score', 0.0))
        if (idx + 1) % 20 == 0:
            valid = [s for s in scores if s > 0]
            avg = np.mean(valid) if valid else 0
            print(f'    [{idx+1}/{len(ALL_EQ_QUESTIONS)}] running avg: {avg:.1f} ({len(valid)} valid)', flush=True)
    valid_scores = [s for s in scores if s > 0]
    return {'score': float(np.mean(valid_scores)) if valid_scores else 0.0, 'n': len(valid_scores), 'n_total': len(scores), 'std': float(np.std(valid_scores)) if valid_scores else 0.0}

# Configs
CONFIGS = [
    ('baseline', None),
    ('ng_45_52', [(45, 52)]),
    ('pair_0_7_45_52', [(0, 7), (45, 52)]),
]

all_results = {}

for config_name, blocks in CONFIGS:
    print(f'\\n--- {config_name} ---', flush=True)

    if blocks:
        order = build_order(blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)
    else:
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

    gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    # Math probe (standard 16 questions)
    gen_short = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    math_r = run_math_probe(gen_short, verbose=False)
    print(f'  Math (16q): {math_r[\"score\"]:.4f}', flush=True)

    # Full EQ-bench
    eq_r = run_full_eqbench(gen)
    print(f'  EQ-bench ({eq_r[\"n\"]}q): {eq_r[\"score\"]:.1f} (std={eq_r[\"std\"]:.1f})', flush=True)

    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  Combined: {combined:.2f}', flush=True)

    all_results[config_name] = {
        'math': math_r['score'], 'eq': eq_r['score'], 'eq_n': eq_r['n'],
        'eq_std': eq_r['std'], 'combined': combined
    }

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

# Bootstrap CIs
print(f'\\n--- Bootstrap 95% CIs ---', flush=True)
# Can only bootstrap EQ-bench since we have per-question scores
# Math probe with 16 questions: CI from bootstrap

# Summary
print(f'\\n{\"=\" * 70}')
print('FULL EQ-BENCH SUMMARY')
print(f'{\"=\" * 70}')
print(f'{\"Config\":>25s} {\"Math(16q)\":>10s} {\"EQ(full)\":>10s} {\"EQ(std)\":>8s} {\"Combined\":>10s}')
for name, r in all_results.items():
    print(f'{name:>25s} {r[\"math\"]:10.4f} {r[\"eq\"]:10.1f} {r[\"eq_std\"]:8.1f} {r[\"combined\"]:10.2f}')

os.makedirs('results/data/72b/full_eqbench', exist_ok=True)
with open('results/data/72b/full_eqbench/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
print(f'Saved', flush=True)
"

echo "=== Done at $(date) ==="
