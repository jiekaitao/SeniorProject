#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_fix_eval_%j.log
#SBATCH --job-name=deeppass_kvfix

# Fix KV cache for duplicated layers using thin wrappers (shared weights)
# Then run lm-eval properly with cache enabled. 1% test first.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache Fix + Proper lm-eval ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn
sys.path.insert(0, 'scripts')

print('=' * 70)
print('KV CACHE FIX: Thin wrappers for shared-weight layer duplication')
print('=' * 70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('Loading base model...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
print(f'Loaded: {N} layers', flush=True)

# =====================================================================
# Thin wrapper: gives a shared layer a unique identity for KV cache
# =====================================================================
class LayerWrapper(nn.Module):
    \"\"\"Wraps an existing layer without copying weights.
    Gives it a unique module identity so KV cache doesn't collide.
    Delegates all forward calls to the wrapped layer.\"\"\"
    def __init__(self, layer):
        super().__init__()
        self._wrapped = layer

    def forward(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def __getattr__(self, name):
        if name == '_wrapped':
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._wrapped, name)

# Build duplicated model: layers 0-51, wrapped copies of 45-51, then 52-79
i, j = 45, 52
print(f'Building duplicated model with wrapped layers [{i},{j})...', flush=True)

new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    if idx == j - 1:
        for dup_idx in range(i, j):
            new_layers.append(LayerWrapper(original_layers[dup_idx]))

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
print(f'New model: {len(new_layers)} layers ({j-i} wrapped duplicates)', flush=True)
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB (no extra memory for wrappers)', flush=True)

# =====================================================================
# Test KV cache
# =====================================================================
print(f'\\nTesting generation with KV cache...', flush=True)
input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)

try:
    output = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'  KV cache ON: {response[:100]}', flush=True)
    kv_works = True
except Exception as e:
    print(f'  KV cache FAILED: {e}', flush=True)
    kv_works = False

# Also test without cache for comparison
output_nc = model.generate(**input_ids, max_new_tokens=30, use_cache=False)
response_nc = tokenizer.decode(output_nc[0], skip_special_tokens=True)
print(f'  KV cache OFF: {response_nc[:100]}', flush=True)

if kv_works:
    # Compare outputs
    match = (output[0] == output_nc[0]).all().item() if output.shape == output_nc.shape else False
    print(f'  Outputs match: {match}', flush=True)
    if not match:
        print(f'  WARNING: KV cache gives different output! May need deeper fix.', flush=True)

# =====================================================================
# Speed comparison
# =====================================================================
import time

prompt = 'The theory of general relativity, proposed by Albert Einstein in 1915, fundamentally changed our understanding of gravity, space, and time.'
input_ids = tokenizer(prompt, return_tensors='pt').to(model.device)

# Warmup
model.generate(**input_ids, max_new_tokens=5, use_cache=kv_works)

if kv_works:
    t0 = time.time()
    for _ in range(3):
        model.generate(**input_ids, max_new_tokens=64, use_cache=True)
    cache_time = (time.time() - t0) / 3
    print(f'\\nSpeed with KV cache: {64/cache_time:.1f} tok/s ({cache_time:.1f}s for 64 tokens)', flush=True)

t0 = time.time()
for _ in range(3):
    model.generate(**input_ids, max_new_tokens=64, use_cache=False)
nocache_time = (time.time() - t0) / 3
print(f'Speed without KV cache: {64/nocache_time:.1f} tok/s ({nocache_time:.1f}s for 64 tokens)', flush=True)

if kv_works:
    print(f'Speedup: {nocache_time/cache_time:.1f}x', flush=True)

# =====================================================================
# Run lm-eval (1% subsample)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print(f'Running lm-eval (1% subsample, KV cache={\"ON\" if kv_works else \"OFF\"})')
print(f'{\"=\" * 70}', flush=True)

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    batch_size='auto' if kv_works else 1,
)

# If KV cache doesn't work with wrappers, patch generate
if not kv_works:
    orig_gen = model.generate
    def patched_gen(*a, **kw):
        kw['use_cache'] = False
        return orig_gen(*a, **kw)
    model.generate = patched_gen

TASKS = [
    'leaderboard_ifeval',
    'leaderboard_bbh',
    'leaderboard_math_hard',
    'leaderboard_musr',
    'leaderboard_mmlu_pro',
]

results = evaluator.simple_evaluate(
    model=lm,
    tasks=TASKS,
    limit=0.01,
)

print(f'\\n=== RESULTS (1% subsample) ===', flush=True)
print(f'KV cache: {\"ON\" if kv_works else \"OFF (wrapper failed)\"}', flush=True)

# Print ALL metrics to match against Ng's numbers
for task in sorted(results['results'].keys()):
    data = results['results'][task]
    print(f'\\n  {task}:', flush=True)
    for metric in sorted(data.keys()):
        value = data[metric]
        if isinstance(value, (int, float)):
            print(f'    {metric}: {value:.4f}', flush=True)

# Also run baseline for comparison
print(f'\\n{\"=\" * 70}')
print('Running BASELINE for comparison...')
print(f'{\"=\" * 70}', flush=True)

# Restore original layers
inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

lm_base = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')

results_base = evaluator.simple_evaluate(
    model=lm_base,
    tasks=TASKS,
    limit=0.01,
)

print(f'\\n=== BASELINE (1% subsample) ===', flush=True)
for task in sorted(results_base['results'].keys()):
    data = results_base['results'][task]
    print(f'\\n  {task}:', flush=True)
    for metric in sorted(data.keys()):
        value = data[metric]
        if isinstance(value, (int, float)):
            print(f'    {metric}: {value:.4f}', flush=True)

# Save
os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/kv_fix_test.json', 'w') as f:
    json.dump({
        'kv_cache_works': kv_works,
        'speedup': nocache_time/cache_time if kv_works else None,
        'ng_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results['results'].items()},
        'baseline_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results_base['results'].items()},
    }, f, indent=2)
print(f'\\nSaved', flush=True)
"

echo "=== Done at $(date) ==="
