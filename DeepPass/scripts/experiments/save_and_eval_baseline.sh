#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_proper_baseline_%j.log
#SBATCH --job-name=deeppass_pbase

# Proper baseline eval with same lm-eval setup as the saved model evals
# 1% test to verify numbers match Ng's scale

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Proper Baseline Eval (1% test) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, json
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

print('Loading baseline model via HFLM (standard, with KV cache)...', flush=True)
lm = HFLM(
    pretrained=MODEL_PATH,
    dtype='bfloat16',
    batch_size='auto',
    device_map_option='auto',
    trust_remote_code=True,
)

print('Running lm-eval (1% subsample)...', flush=True)
results = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

print(f'\\n=== BASELINE RESULTS (1% test, KV cache ON) ===', flush=True)
for task in TASKS:
    data = results['results'].get(task, {})
    for metric in sorted(data.keys()):
        if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/baseline_1pct.json', 'w') as f:
    json.dump({
        'method': 'standard HFLM, KV cache ON, batch_size auto',
        'subsample': 0.01,
        'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                    for k, v in results['results'].items()},
    }, f, indent=2)
print('Saved!', flush=True)
"

echo "=== Done at $(date) ==="
