#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_proper_15pct_%j.log
#SBATCH --job-name=deeppass_p15

# Proper eval: 15% subsample, standard HFLM, with leaderboard normalization
# Baseline + Ng (deep copy) + Pair (deep copy)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Proper 15% Eval with Leaderboard Normalization ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, gc, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

def normalize(value, lower_bound):
    if value < lower_bound:
        return 0
    return (value - lower_bound) / (1.0 - lower_bound) * 100

def compute_leaderboard_scores(results):
    \"\"\"Apply Open LLM Leaderboard v2 normalization.\"\"\"
    r = results['results']
    scores = {}

    # IFEval: avg of inst_strict and prompt_strict (lower_bound=0)
    ifeval = r.get('leaderboard_ifeval', {})
    inst_s = ifeval.get('inst_level_strict_acc,none', 0)
    prompt_s = ifeval.get('prompt_level_strict_acc,none', 0)
    scores['IFEval'] = (normalize(inst_s, 0) + normalize(prompt_s, 0)) / 2

    # BBH: acc_norm already baseline-adjusted, *100
    scores['BBH'] = r.get('leaderboard_bbh', {}).get('acc_norm,none', 0) * 100

    # MATH: exact_match, lower_bound=0
    scores['MATH'] = normalize(r.get('leaderboard_math_hard', {}).get('exact_match,none', 0), 0)

    # MuSR: per-subtask normalization
    murder = r.get('leaderboard_musr_murder_mysteries', {}).get('acc_norm,none', 0)
    obj = r.get('leaderboard_musr_object_placements', {}).get('acc_norm,none', 0)
    team = r.get('leaderboard_musr_team_allocation', {}).get('acc_norm,none', 0)
    scores['MuSR'] = (normalize(murder, 0.5) + normalize(obj, 0.2) + normalize(team, 0.333)) / 3

    # MMLU-PRO: num_choices=10, lower_bound=0.1
    scores['MMLU-PRO'] = normalize(r.get('leaderboard_mmlu_pro', {}).get('acc,none', 0), 0.1)

    scores['Average'] = sum(scores.values()) / len(scores)
    return scores

all_results = {}

# =====================================================================
# 1. Baseline
# =====================================================================
print('=== BASELINE ===', flush=True)
lm = HFLM(pretrained=MODEL_PATH, dtype='bfloat16', batch_size='auto', trust_remote_code=True)
r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.15)
base_scores = compute_leaderboard_scores(r)
print(f'Baseline normalized:', flush=True)
for k, v in base_scores.items():
    print(f'  {k}: {v:.2f}', flush=True)
all_results['baseline'] = {'normalized': base_scores, 'raw': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}}
del lm; gc.collect(); torch.cuda.empty_cache()

# =====================================================================
# 2. Ng (45,52) — deep copy with fixed config
# =====================================================================
print(f'\\n=== NG (45,52) ===', flush=True)
print('Loading and duplicating...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='cpu', dtype=torch.bfloat16, trust_remote_code=True)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

i, j = 45, 52
new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    if idx == j - 1:
        for dup_idx in range(i, j):
            new_layers.append(copy.deepcopy(original_layers[dup_idx]))

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
model.config.layer_types = None
for pos, layer in enumerate(inner.layers):
    if hasattr(layer, 'self_attn'):
        layer.self_attn.layer_idx = pos

print(f'Built: {len(new_layers)} layers', flush=True)
model = model.cuda()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.15)
ng_scores = compute_leaderboard_scores(r)
print(f'Ng normalized:', flush=True)
for k, v in ng_scores.items():
    print(f'  {k}: {v:.2f}', flush=True)
all_results['ng'] = {'normalized': ng_scores, 'raw': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}}

del model, lm; gc.collect(); torch.cuda.empty_cache()

# =====================================================================
# 3. Pair (0,7)+(45,52) — deep copy
# =====================================================================
print(f'\\n=== PAIR (0,7)+(45,52) ===', flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='cpu', dtype=torch.bfloat16, trust_remote_code=True)
inner = model.model
original_layers = list(inner.layers)

blocks = [(0, 7), (45, 52)]
new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    for (bi, bj) in sorted(blocks):
        if idx == bj - 1:
            for dup_idx in range(bi, bj):
                new_layers.append(copy.deepcopy(original_layers[dup_idx]))

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
model.config.layer_types = None
for pos, layer in enumerate(inner.layers):
    if hasattr(layer, 'self_attn'):
        layer.self_attn.layer_idx = pos

print(f'Built: {len(new_layers)} layers', flush=True)
model = model.cuda()

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.15)
pair_scores = compute_leaderboard_scores(r)
print(f'Pair normalized:', flush=True)
for k, v in pair_scores.items():
    print(f'  {k}: {v:.2f}', flush=True)
all_results['pair'] = {'normalized': pair_scores, 'raw': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}}

# =====================================================================
# Summary
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('LEADERBOARD-NORMALIZED COMPARISON (15% subsample)')
print(f'{\"=\" * 70}')
ng_reported = {'IFEval': 79.96, 'BBH': 58.77, 'MATH': 38.97, 'MuSR': 23.72, 'MMLU-PRO': 49.20, 'Average': 44.75}

print(f'{\"Task\":>10s} {\"Baseline\":>10s} {\"Ng ours\":>10s} {\"Pair ours\":>10s} {\"Ng reported\":>12s} {\"Ng delta\":>10s} {\"Pair delta\":>10s}')
print('-' * 75)
for task in ['IFEval', 'BBH', 'MATH', 'MuSR', 'MMLU-PRO', 'Average']:
    b = base_scores.get(task, 0)
    n = ng_scores.get(task, 0)
    p = pair_scores.get(task, 0)
    nr = ng_reported.get(task, 0)
    print(f'{task:>10s} {b:10.2f} {n:10.2f} {p:10.2f} {nr:12.2f} {n-b:+10.2f} {p-b:+10.2f}', flush=True)

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/leaderboard_15pct.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\\nSaved!', flush=True)
"

echo "=== Done at $(date) ==="
