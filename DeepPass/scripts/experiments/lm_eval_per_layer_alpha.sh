#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_pla_%j.log
#SBATCH --job-name=deeppass_evpla

# lm-eval on the per-layer alpha config (82.77)
# Uses hooks to apply per-layer alpha during lm-eval's forward passes

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval on Per-Layer Alpha Config ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, torch, torch.nn as nn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCK = (45, 52)
# Best per-layer alphas from per_layer_alpha experiment
PER_LAYER_ALPHAS = [1.1, 1.0, 0.5, 1.3, 1.0, 0.9, 1.1]

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

# Build duplicated model
def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

order = build_order([BLOCK], N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
model.config.num_hidden_layers = len(order)
print(f'Applied duplication: {N} -> {len(order)} layers', flush=True)

# Install persistent hooks to apply per-layer alpha during ALL forward passes
# Find second-pass steps
i, j = BLOCK
second_pass_steps = {}
block_layers = list(range(i, j))
count = {}
for step, idx in enumerate(order):
    if idx in block_layers:
        count[idx] = count.get(idx, 0) + 1
        if count[idx] == 2:
            offset = block_layers.index(idx)
            second_pass_steps[step] = PER_LAYER_ALPHAS[offset]

print(f'Second-pass steps with alphas: {second_pass_steps}', flush=True)

# Register hooks on each second-pass layer
hooks = []
for step, alpha in second_pass_steps.items():
    layer_module = inner.layers[step]

    def make_alpha_hook(a):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            blended = inp + a * (out - inp)
            if isinstance(output, tuple):
                return (blended,) + output[1:]
            return blended
        return hook_fn

    hooks.append(layer_module.register_forward_hook(make_alpha_hook(alpha)))

print(f'Installed {len(hooks)} per-layer alpha hooks', flush=True)

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

print(f'Running lm-eval with per-layer alpha...', flush=True)
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print(f'\\n=== RESULTS (per-layer alpha single block) ===', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k:
        print(f'  {k}: {v:.4f}', flush=True)

# Clean up hooks
for h in hooks:
    h.remove()

os.makedirs('results/data/72b/lm_eval', exist_ok=True)
with open('results/data/72b/lm_eval/per_layer_alpha_45_52.json', 'w') as f:
    json.dump({'config': 'per_layer_alpha_45_52', 'block': list(BLOCK),
               'per_layer_alphas': PER_LAYER_ALPHAS,
               'scores': scores}, f, indent=2)
print(f'Saved', flush=True)
"

echo "=== Done at $(date) ==="
