#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_proper_eval_%j.log
#SBATCH --job-name=deeppass_peval

# PROPER evaluation: save_pretrained with deep-copied layers, then run lm-eval
# with KV cache enabled, matching Open LLM Leaderboard setup exactly.
# 1% subsample to verify it works before full runs.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Proper save_pretrained + lm-eval (1% test) ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, torch, torch.nn as nn
sys.path.insert(0, 'scripts')

print('=' * 70)
print('STEP 1: Save duplicated model with deep-copied layers')
print('=' * 70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
SAVE_DIR = 'models/saved/ng_45_52_proper'

print('Loading base model...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
print(f'Loaded: {N} layers', flush=True)

# Build duplicated layer list with DEEP COPIES
# Ng's config: duplicate block [45, 52)
i, j = 45, 52
print(f'Duplicating block [{i}, {j}) with deep copies...', flush=True)

new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    if idx == j - 1:
        # Insert deep copies of layers i..j-1 after layer j-1
        for dup_idx in range(i, j):
            print(f'  Deep copying layer {dup_idx}...', flush=True)
            new_layers.append(copy.deepcopy(original_layers[dup_idx]))

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
print(f'New model: {len(new_layers)} layers', flush=True)

# Fix layer_types if present (Qwen2 has this)
if hasattr(model.config, 'layer_types') and model.config.layer_types:
    orig_types = model.config.layer_types
    new_types = []
    for idx in range(N):
        new_types.append(orig_types[idx])
        if idx == j - 1:
            for dup_idx in range(i, j):
                new_types.append(orig_types[dup_idx])
    model.config.layer_types = new_types

# Save
os.makedirs(SAVE_DIR, exist_ok=True)
print(f'Saving to {SAVE_DIR}...', flush=True)
model.save_pretrained(SAVE_DIR, max_shard_size='10GB')
tokenizer.save_pretrained(SAVE_DIR)
print('Model saved!', flush=True)

# Free memory
del model
import gc; gc.collect(); torch.cuda.empty_cache()

print()
print('=' * 70)
print('STEP 2: Verify KV cache works')
print('=' * 70, flush=True)

# Reload the saved model
model2 = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer2 = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)

print(f'Reloaded: {model2.config.num_hidden_layers} layers', flush=True)

# Test generation WITH cache
input_ids = tokenizer2('What is 2+2?', return_tensors='pt').to(model2.device)
print('Testing generation with KV cache...', flush=True)
output = model2.generate(**input_ids, max_new_tokens=20, use_cache=True)
response = tokenizer2.decode(output[0], skip_special_tokens=True)
print(f'  Response: {response[:100]}', flush=True)
print('KV cache works!', flush=True)

del model2; gc.collect(); torch.cuda.empty_cache()

print()
print('=' * 70)
print('STEP 3: Run lm-eval properly (1% subsample, with KV cache)')
print('=' * 70, flush=True)

# Run lm-eval the standard way — point it at the saved model
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Load as standard HFLM — this uses KV cache by default
lm = HFLM(
    pretrained=SAVE_DIR,
    dtype='bfloat16',
    batch_size='auto',
    device_map_option='auto',
    trust_remote_code=True,
)

TASKS = [
    'leaderboard_ifeval',
    'leaderboard_bbh',
    'leaderboard_math_hard',
    'leaderboard_musr',
    'leaderboard_mmlu_pro',
]

print(f'Running lm-eval on saved model (1% subsample)...', flush=True)
print(f'Tasks: {TASKS}', flush=True)

results = evaluator.simple_evaluate(
    model=lm,
    tasks=TASKS,
    limit=0.01,  # 1% subsample just to verify
    batch_size='auto',
)

print(f'\\n=== RESULTS (1% subsample, KV cache ON, saved model) ===', flush=True)
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {value:.4f}', flush=True)

# Also compare: what does Ng's leaderboard use?
# The Open LLM Leaderboard v2 normalizes scores differently
# Let's print all metrics so we can identify which ones match Ng's
print(f'\\n=== ALL METRICS (for matching with Ng) ===', flush=True)
for task, data in sorted(results['results'].items()):
    print(f'  {task}:', flush=True)
    for metric, value in sorted(data.items()):
        if isinstance(value, (int, float)):
            print(f'    {metric}: {value}', flush=True)

# Save
os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
scores = {}
for task, data in results['results'].items():
    for metric, value in data.items():
        if isinstance(value, (int, float)):
            scores[f'{task}/{metric}'] = value

with open('results/data/72b/lm_eval/proper/ng_45_52_1pct_test.json', 'w') as f:
    json.dump({
        'config': 'ng_45_52_save_pretrained',
        'method': 'save_pretrained with deep-copied layers, KV cache ON',
        'subsample': 0.01,
        'scores': scores,
        'full_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                        for k, v in results['results'].items()},
    }, f, indent=2)
print(f'\\nSaved to results/data/72b/lm_eval/proper/', flush=True)
"

echo "=== Done at $(date) ==="
