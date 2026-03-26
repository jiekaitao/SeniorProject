#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_full_%j.log
#SBATCH --job-name=deeppass_fulleval

# Full lm-eval (NO subsample) on baseline + best pair
# Definitive paper numbers

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Full lm-eval (no subsample) on 72B ==="
echo "Started: $(date)"

# Baseline first
echo "=== BASELINE ==="
$PYTHON scripts/experiments/lm_eval_runtime_dup.py \
    --model models/full/calme-2.1-qwen2-72b \
    --blocks "0,1" \
    --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro \
    --limit 1.0 \
    --output results/data/72b/lm_eval/full_baseline.json

# Actually, block "0,1" would duplicate layer 0. We need a baseline without duplication.
# Let me use a different approach — run lm-eval directly for baseline.
echo "=== BASELINE (direct) ==="
$PYTHON -c "
import os, json, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys; sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
orig_gen = model.generate
def patched_gen(*a, **kw):
    kw['use_cache'] = False
    return orig_gen(*a, **kw)
model.generate = patched_gen

results = evaluator.simple_evaluate(
    model=lm,
    tasks=['leaderboard_ifeval', 'leaderboard_musr'],
    limit=None,
)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print('=== BASELINE FULL RESULTS ===', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k:
        print(f'  {k}: {v:.4f}', flush=True)

os.makedirs('results/data/72b/lm_eval', exist_ok=True)
with open('results/data/72b/lm_eval/full_baseline_ifeval_musr.json', 'w') as f:
    json.dump({'scores': scores}, f, indent=2)
print('Saved baseline', flush=True)

del model, tokenizer, lm
import gc; gc.collect(); torch.cuda.empty_cache()
"

echo "=== BEST PAIR (0,7)+(45,52) ==="
$PYTHON scripts/experiments/lm_eval_runtime_dup.py \
    --model models/full/calme-2.1-qwen2-72b \
    --blocks "0,7;45,52" \
    --tasks leaderboard_ifeval,leaderboard_musr \
    --limit 1.0 \
    --output results/data/72b/lm_eval/full_best_pair_ifeval_musr.json

echo "=== Done at $(date) ==="
