#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_crosslayer_%j.log
#SBATCH --job-name=deeppass_evxl

# lm-eval on best cross-layer config: (45,52)->(20,27) @1.15
# Also compare to standard self-dup (45,52) and attention-only

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval Cross-Layer Duplication ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, torch, torch.nn as nn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

# Cross-layer order: first pass (45,52), second pass uses (20,27) weights
first_block = (45, 52)
cross_block = (20, 27)
i, j = first_block
a, b = cross_block
order = list(range(j)) + list(range(i, j)) + list(range(a, b)) + list(range(j, N))

inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
model.config.num_hidden_layers = len(order)
print(f'Cross-layer duplication: {N} -> {len(order)} layers', flush=True)
print(f'First pass: ({i},{j}), Second pass weights: ({a},{b})', flush=True)

# Hook for alpha=1.15 at the seam
first_pass_end = j - 1
second_pass_end = j + (b - a) - 1

seam_h1 = {}

def make_seam_hooks():
    hooks = []
    def save_h1(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        seam_h1['h1'] = out.clone()
    def apply_alpha(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if 'h1' in seam_h1:
            blended = seam_h1['h1'] + 1.15 * (out - seam_h1['h1'])
            del seam_h1['h1']
            if isinstance(output, tuple):
                return (blended,) + output[1:]
            return blended
    hooks.append(inner.layers[first_pass_end].register_forward_hook(save_h1))
    hooks.append(inner.layers[second_pass_end].register_forward_hook(apply_alpha))
    return hooks

hooks = make_seam_hooks()
print(f'Installed seam hooks at steps {first_pass_end} and {second_pass_end}', flush=True)

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
orig_gen = model.generate
def patched_gen(*a, **kw):
    kw['use_cache'] = False
    return orig_gen(*a, **kw)
model.generate = patched_gen

TASKS = 'leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'

print(f'Running lm-eval on cross-layer (45,52)->(20,27) @1.15...', flush=True)
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)

scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

print(f'\\n=== RESULTS (cross-layer duplication) ===', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k:
        print(f'  {k}: {v:.4f}', flush=True)

for h in hooks: h.remove()

os.makedirs('results/data/72b/lm_eval', exist_ok=True)
with open('results/data/72b/lm_eval/cross_layer_45_52_to_20_27.json', 'w') as f:
    json.dump({'config': 'cross_layer_45_52_to_20_27_alpha_1.15',
               'first_block': list(first_block), 'cross_block': list(cross_block), 'alpha': 1.15,
               'scores': scores}, f, indent=2)
print('Saved', flush=True)
"

echo "=== Done at $(date) ==="
