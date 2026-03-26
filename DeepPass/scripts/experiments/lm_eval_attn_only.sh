#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_attn_only_%j.log
#SBATCH --job-name=deeppass_evao

# lm-eval on ATTENTION-ONLY duplication (skip FFN on second pass)
# Hypothesis: FFN re-retrieval corrupts factual knowledge
# Attention-only should preserve MATH/MMLU while keeping reasoning gains

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval Attention-Only Duplication ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, torch, torch.nn as nn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCK = (45, 52)

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

# Find second-pass steps
i, j = BLOCK
block_layers = list(range(i, j))
second_pass_steps = set()
count = {}
for step, idx in enumerate(order):
    if idx in block_layers:
        count[idx] = count.get(idx, 0) + 1
        if count[idx] == 2:
            second_pass_steps.add(step)

print(f'Second-pass steps: {len(second_pass_steps)}', flush=True)

# Install hooks: on second-pass layers, keep attention but SKIP FFN
# Each layer does: h = h + attn(norm(h)); h = h + ffn(norm(h))
# We want: h = h + attn(norm(h)); h = h  (skip FFN residual)
hooks = []
for step in second_pass_steps:
    layer_module = inner.layers[step]

    # Hook the MLP to make it a no-op (return zeros)
    def make_ffn_skip_hook():
        def hook_fn(module, input, output):
            # Return zeros so the residual connection adds nothing
            return torch.zeros_like(output)
        return hook_fn

    hooks.append(layer_module.mlp.register_forward_hook(make_ffn_skip_hook()))

print(f'Installed {len(hooks)} FFN-skip hooks on second-pass layers', flush=True)

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
orig_gen = model.generate
def patched_gen(*a, **kw):
    kw['use_cache'] = False
    return orig_gen(*a, **kw)
model.generate = patched_gen

TASKS = 'leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'

print(f'Running lm-eval with attention-only duplication...', flush=True)
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print(f'\\n=== RESULTS (attention-only duplication) ===', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k:
        print(f'  {k}: {v:.4f}', flush=True)

for h in hooks: h.remove()

os.makedirs('results/data/72b/lm_eval', exist_ok=True)
with open('results/data/72b/lm_eval/attn_only_45_52.json', 'w') as f:
    json.dump({'config': 'attn_only_45_52', 'block': list(BLOCK),
               'hypothesis': 'FFN re-retrieval corrupts facts; attention-only preserves knowledge while helping reasoning',
               'scores': scores}, f, indent=2)
print('Saved', flush=True)
"

echo "=== Done at $(date) ==="
