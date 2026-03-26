#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_baseline_%j.log
#SBATCH --job-name=deeppass_lmeval_base

# lm-eval on baseline (no duplication) — clean comparison through same pipeline

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval baseline (no duplication) ==="
$PYTHON -c "
import sys, os, json
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
print('Running baseline lm-eval...')

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

results = evaluator.simple_evaluate(
    model=lm,
    tasks=['leaderboard_ifeval','leaderboard_bbh','leaderboard_math_hard','leaderboard_musr','leaderboard_mmlu_pro'],
    limit=0.15,
    batch_size='auto',
)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print('=== RESULTS ===')
for k, v in sorted(scores.items()):
    print(f'  {k}: {v:.4f}')

os.makedirs('results/data/72b/lm_eval', exist_ok=True)
with open('results/data/72b/lm_eval/baseline.json', 'w') as f:
    json.dump({'model': 'calme-2.1-qwen2-72b', 'blocks': [], 'scores': scores}, f, indent=2)
print('Saved')
"
echo "=== Done at $(date) ==="
