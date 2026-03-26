#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_v7_%j.log
#SBATCH --job-name=deeppass_kvv7

# v7: Debug why cache gives no speedup, and match Ng's metrics exactly

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache v7: Debug speedup + match Ng metrics ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, gc, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70, flush=True)
print('DEBUG 1: Why is cache speedup 1.0x?', flush=True)
print('=' * 70, flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Try loading WITHOUT device_map='auto' — use single GPU directly
print('Loading with device_map=None (single GPU)...', flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
).cuda()

print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)
print(f'Model device: {next(model.parameters()).device}', flush=True)

# Test baseline speed
prompt = 'The theory of general relativity describes gravity as curvature of spacetime caused by mass and energy distribution throughout the universe.'
inp = tokenizer(prompt, return_tensors='pt').to('cuda')

# Warmup
model.generate(**inp, max_new_tokens=5, use_cache=True)
torch.cuda.synchronize()

# Cache ON
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    model.generate(**inp, max_new_tokens=64, use_cache=True)
    torch.cuda.synchronize()
cache_time = (time.time() - t0) / 5

# Cache OFF
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    model.generate(**inp, max_new_tokens=64, use_cache=False)
    torch.cuda.synchronize()
nocache_time = (time.time() - t0) / 5

print(f'Baseline (single GPU, no device_map):', flush=True)
print(f'  Cache ON:  {64/cache_time:.1f} tok/s ({cache_time:.2f}s)', flush=True)
print(f'  Cache OFF: {64/nocache_time:.1f} tok/s ({nocache_time:.2f}s)', flush=True)
print(f'  Speedup:   {nocache_time/cache_time:.1f}x', flush=True)

# Now duplicate with deep copy
print(f'\\n--- Duplicating (45,52) with deep copy ---', flush=True)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

i, j = 45, 52
new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    if idx == j - 1:
        for dup_idx in range(i, j):
            new_layers.append(copy.deepcopy(original_layers[dup_idx].cuda()))

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
model.config.layer_types = None
for pos, layer in enumerate(inner.layers):
    if hasattr(layer, 'self_attn'):
        layer.self_attn.layer_idx = pos
print(f'Built: {len(new_layers)} layers', flush=True)

# Speed test on duplicated
model.generate(**inp, max_new_tokens=5, use_cache=True)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(5):
    model.generate(**inp, max_new_tokens=64, use_cache=True)
    torch.cuda.synchronize()
dup_cache_time = (time.time() - t0) / 5

t0 = time.time()
for _ in range(5):
    model.generate(**inp, max_new_tokens=64, use_cache=False)
    torch.cuda.synchronize()
dup_nocache_time = (time.time() - t0) / 5

print(f'Duplicated (single GPU):', flush=True)
print(f'  Cache ON:  {64/dup_cache_time:.1f} tok/s ({dup_cache_time:.2f}s)', flush=True)
print(f'  Cache OFF: {64/dup_nocache_time:.1f} tok/s ({dup_nocache_time:.2f}s)', flush=True)
print(f'  Speedup:   {dup_nocache_time/dup_cache_time:.1f}x', flush=True)

# =====================================================================
# DEBUG 2: Run lm-eval with HFLM loading from path (standard way)
# This is how Ng's model was evaluated on the leaderboard
# =====================================================================
print(f'\\n{\"=\" * 70}', flush=True)
print('DEBUG 2: lm-eval the standard way (HFLM from path)', flush=True)
print(f'{\"=\" * 70}', flush=True)

del model; gc.collect(); torch.cuda.empty_cache()

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

# Standard HFLM loading — this is exactly how the leaderboard does it
print('Loading via HFLM (standard leaderboard way)...', flush=True)
lm = HFLM(
    pretrained=MODEL_PATH,
    dtype='bfloat16',
    batch_size='auto',
    trust_remote_code=True,
)

TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

print('Running baseline 1%...', flush=True)
r_base = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

print(f'\\n=== BASELINE (standard HFLM, 1%) ===', flush=True)
for task in TASKS:
    data = r_base['results'].get(task, {})
    for m in sorted(data.keys()):
        if isinstance(data[m], (int, float)) and 'stderr' not in m:
            # Print both raw and *100 to figure out Ng's scale
            val = data[m]
            print(f'  {task}/{m}: {val:.4f} (x100: {val*100:.2f})', flush=True)

# Compare with Ng's reported baseline
# Ng's RYS improvement was RELATIVE to his baseline
# His baseline (calme-2.1) would be: IFEval=82.01, BBH=56.26, MATH=30.81, etc.
# (from 79.96-(-2.05), 58.77-2.51, etc.)
ng_baseline = {
    'IFEval': 82.01, 'BBH': 56.26, 'MATH': 30.81,
    'GPQA': 15.32, 'MuSR': 5.99, 'MMLU-PRO': 48.89
}
print(f'\\nNg baseline (estimated): {ng_baseline}', flush=True)
print(f'Ng average baseline: {sum(ng_baseline.values())/6:.2f}', flush=True)

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/kv_v7_debug.json', 'w') as f:
    json.dump({
        'baseline_speedup': nocache_time/cache_time,
        'dup_speedup': dup_nocache_time/dup_cache_time,
        'baseline_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r_base['results'].items()},
    }, f, indent=2)
print('Saved!', flush=True)
"

echo "=== Done at $(date) ==="
