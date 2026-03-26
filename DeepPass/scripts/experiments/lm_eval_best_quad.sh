#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_quad_%j.log
#SBATCH --job-name=deeppass_evquad

# lm-eval on best quad config: (0,7)@0.9 + (15,20)@0.1 + (20,27)@0.15 + (45,52)@1.0
# This is our new flagship result (combined=82.58)
# Tests IFEval + MuSR (where we win) at 15% subsample

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval on Best Quad Config ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, torch, torch.nn as nn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCKS = [(0, 7), (15, 20), (20, 27), (45, 52)]
ALPHAS = [0.9, 0.1, 0.15, 1.0]

# For lm-eval we can't use per-block alpha — lm-eval calls model() directly.
# So we apply standard duplication (alpha=1.0 for all) and note this is a limitation.
# Actually, we CAN use hooks to apply alpha blending at seams during standard forward.

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

# Build duplicated layer order
def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

order = build_order(BLOCKS, N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
model.config.num_hidden_layers = len(order)
print(f'Applied quad duplication: {N} -> {len(order)} layers', flush=True)

# Install hooks to apply alpha blending at seams
# Find seam positions
seam_info = []
block_to_alpha = dict(zip(BLOCKS, ALPHAS))
for block in sorted(BLOCKS):
    last_layer = block[1] - 1
    occurrences = [step for step, idx in enumerate(order) if idx == last_layer]
    if len(occurrences) >= 2:
        alpha = block_to_alpha[block]
        seam_info.append((occurrences[0], occurrences[1], alpha))

print(f'Seam info: {seam_info}', flush=True)

# Register hooks on seam layers
saved_h1 = {}
hooks = []

def make_first_pass_hook(seam_idx, first_end_step):
    step_counter = [0]
    def hook_fn(module, input, output):
        # We need to track which step we're at — but hooks don't know the step index.
        # This approach won't work cleanly with lm-eval since it batches differently.
        pass
    return hook_fn

# Actually, applying per-block alpha via hooks in lm-eval is extremely complex because
# lm-eval's forward passes don't go through our manual layer loop.
# The pragmatic approach: run lm-eval with standard duplication (alpha=1.0 for all blocks)
# and note that the alpha-tuned results are from our dual probe.
# This still shows: does a 4-block duplicated model help on lm-eval?

print('\\nNote: lm-eval runs standard duplication (alpha=1.0 for all 4 blocks).', flush=True)
print('Alpha tuning is tested via dual probe. lm-eval tests the general direction.', flush=True)

# Wrap for lm-eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

orig_gen = model.generate
def patched_gen(*a, **kw):
    kw['use_cache'] = False
    return orig_gen(*a, **kw)
model.generate = patched_gen

TASKS = 'leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'

print(f'\\nRunning lm-eval on quad config (4 blocks, alpha=1.0)...', flush=True)
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print(f'\\n=== RESULTS (quad 4-block, alpha=1.0) ===', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k:
        print(f'  {k}: {v:.4f}', flush=True)

os.makedirs('results/data/72b/lm_eval', exist_ok=True)
with open('results/data/72b/lm_eval/quad_4block.json', 'w') as f:
    json.dump({'config': 'quad_4block_alpha1', 'blocks': [list(b) for b in BLOCKS],
               'note': 'alpha=1.0 for all blocks (lm-eval limitation)',
               'scores': scores}, f, indent=2)
print(f'Saved to results/data/72b/lm_eval/quad_4block.json', flush=True)
"

echo "=== Done at $(date) ==="
