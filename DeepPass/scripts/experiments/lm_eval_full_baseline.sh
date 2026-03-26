#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_full_baseline_%j.log
#SBATCH --job-name=deeppass_fbase

# Full lm-eval (NO subsample) on BASELINE — definitive paper numbers

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Full lm-eval BASELINE (no subsample) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, json, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys; sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

# No cache patching needed for baseline — but keep consistent
orig_gen = model.generate
def patched_gen(*a, **kw):
    kw['use_cache'] = False
    return orig_gen(*a, **kw)
model.generate = patched_gen

print('Running full lm-eval baseline...', flush=True)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard', 'leaderboard_musr', 'leaderboard_mmlu_pro'],
    limit=None,
)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print('\\n=== FULL BASELINE RESULTS ===', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k:
        print(f'  {k}: {v:.4f}', flush=True)

os.makedirs('results/data/72b/lm_eval/full', exist_ok=True)
with open('results/data/72b/lm_eval/full/baseline.json', 'w') as f:
    json.dump({'config': 'baseline_full', 'scores': scores}, f, indent=2)
print('Saved', flush=True)
"

echo "=== Done at $(date) ==="
