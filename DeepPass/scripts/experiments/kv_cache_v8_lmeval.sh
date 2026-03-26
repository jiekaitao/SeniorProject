#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_v8_%j.log
#SBATCH --job-name=deeppass_kvv8

# v8: Just run lm-eval the standard way — HFLM from path, 1% subsample
# Compare baseline vs Ng's saved model to match his reported numbers

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV v8: Standard HFLM lm-eval ==="
echo "Started: $(date)"

$PYTHON -c "
import os, json
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

# Baseline
print('=== BASELINE ===', flush=True)
lm = HFLM(
    pretrained='models/full/calme-2.1-qwen2-72b',
    dtype='bfloat16',
    batch_size='auto',
    trust_remote_code=True,
)
r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)
print(f'\\nBaseline results:', flush=True)
for task in TASKS:
    data = r['results'].get(task, {})
    for m in sorted(data.keys()):
        if isinstance(data[m], (int, float)) and 'stderr' not in m:
            print(f'  {task}/{m}: {data[m]:.4f} (x100={data[m]*100:.1f})', flush=True)

base_results = {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}

# Free memory
del lm
import gc, torch
gc.collect()
torch.cuda.empty_cache()
print(f'\\nFreed VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB remaining', flush=True)

# Ng's saved model (if it exists from sev2 job)
SAVE_DIR = 'models/saved/ng_45_52_proper'
if os.path.exists(os.path.join(SAVE_DIR, 'config.json')):
    print(f'\\n=== NG SAVED MODEL ===', flush=True)
    lm2 = HFLM(
        pretrained=SAVE_DIR,
        dtype='bfloat16',
        batch_size='auto',
        trust_remote_code=True,
    )
    r2 = evaluator.simple_evaluate(model=lm2, tasks=TASKS, limit=0.01)
    print(f'\\nNg saved model results:', flush=True)
    for task in TASKS:
        data = r2['results'].get(task, {})
        for m in sorted(data.keys()):
            if isinstance(data[m], (int, float)) and 'stderr' not in m:
                print(f'  {task}/{m}: {data[m]:.4f} (x100={data[m]*100:.1f})', flush=True)
    ng_results = {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r2['results'].items()}
else:
    print(f'\\nNo saved Ng model found at {SAVE_DIR}', flush=True)
    ng_results = None

# Ng's reported numbers
print(f'\\n=== COMPARISON ===', flush=True)
ng_reported = {
    'leaderboard_ifeval': 79.96,
    'leaderboard_bbh': 58.77,
    'leaderboard_math_hard': 38.97,
    'leaderboard_musr': 23.72,
    'leaderboard_mmlu_pro': 49.20,
}
ng_baseline_est = {
    'leaderboard_ifeval': 82.01,
    'leaderboard_bbh': 56.26,
    'leaderboard_math_hard': 30.81,
    'leaderboard_musr': 5.99,
    'leaderboard_mmlu_pro': 48.89,
}
print(f'{\"Task\":>25s} {\"Our base\":>9s} {\"Ng base(est)\":>12s} {\"Ng RYS\":>9s}', flush=True)
for task in TASKS:
    our = 0
    bd = base_results.get(task, {})
    # Try different metrics
    for metric in ['acc_norm,none', 'exact_match,none', 'prompt_level_strict_acc,none',
                   'inst_level_strict_acc,none', 'acc,none']:
        if metric in bd:
            our = bd[metric] * 100
            break
    ng_b = ng_baseline_est.get(task, 0)
    ng_r = ng_reported.get(task, 0)
    print(f'{task:>25s} {our:9.2f} {ng_b:12.2f} {ng_r:9.2f}', flush=True)

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/kv_v8.json', 'w') as f:
    json.dump({'baseline': base_results, 'ng_saved': ng_results}, f, indent=2)
print('\\nSaved!', flush=True)
"

echo "=== Done at $(date) ==="
