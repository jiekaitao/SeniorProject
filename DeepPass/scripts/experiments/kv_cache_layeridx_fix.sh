#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_layeridx_%j.log
#SBATCH --job-name=deeppass_kvli

# THE FIX: patch layer_idx on duplicated layers so KV cache slots don't collide
# No deep copy, no wrapper, no extra memory — just reassign an int per layer

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache Fix: layer_idx patching ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('Loading model...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
print(f'Loaded: {N} layers', flush=True)

# =====================================================================
# Build duplicated model (pointer sharing, NOT deep copy)
# =====================================================================
def build_duplicated_model(model, inner, original_layers, blocks, N):
    \"\"\"Build duplicated model with shared weights but fixed KV cache.\"\"\"
    sorted_blocks = sorted(blocks)
    new_layers = []
    for idx in range(N):
        new_layers.append(original_layers[idx])
        for (bi, bj) in sorted_blocks:
            if idx == bj - 1:
                for dup_idx in range(bi, bj):
                    new_layers.append(original_layers[dup_idx])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    # THE FIX: patch layer_idx on EVERY layer to its position in the new list
    for pos, layer in enumerate(inner.layers):
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_idx'):
            layer.self_attn.layer_idx = pos

    print(f'Built: {len(new_layers)} layers, all layer_idx patched', flush=True)
    print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB (no extra memory)', flush=True)

    # Verify unique layer_idx
    idxs = [l.self_attn.layer_idx for l in inner.layers if hasattr(l, 'self_attn')]
    assert len(set(idxs)) == len(idxs), f'layer_idx not unique! {idxs}'
    print(f'layer_idx range: {min(idxs)} to {max(idxs)} (all unique)', flush=True)

    return new_layers

# =====================================================================
# Test 1: Ng's (45,52) config
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 1: Ng (45,52) with layer_idx fix')
print(f'{\"=\" * 70}', flush=True)

new_layers = build_duplicated_model(model, inner, original_layers, [(45, 52)], N)

# Test KV cache
input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)

print('\\nComparing cache ON vs OFF...', flush=True)
out_cache = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
out_nocache = model.generate(**input_ids, max_new_tokens=30, use_cache=False)

resp_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
resp_nocache = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'  Cache ON:  {resp_cache[:100]}', flush=True)
print(f'  Cache OFF: {resp_nocache[:100]}', flush=True)

# Check token-level match
min_len = min(len(out_cache[0]), len(out_nocache[0]))
match = (out_cache[0][:min_len] == out_nocache[0][:min_len]).all().item()
print(f'  Token match: {match}', flush=True)

# Speed comparison
prompt = 'The theory of general relativity describes gravity as curvature of spacetime.'
inp = tokenizer(prompt, return_tensors='pt').to(model.device)
model.generate(**inp, max_new_tokens=5, use_cache=True)  # warmup

t0 = time.time()
for _ in range(3):
    model.generate(**inp, max_new_tokens=64, use_cache=True)
cache_time = (time.time() - t0) / 3

t0 = time.time()
for _ in range(3):
    model.generate(**inp, max_new_tokens=64, use_cache=False)
nocache_time = (time.time() - t0) / 3

print(f'  Cache ON:  {64/cache_time:.1f} tok/s ({cache_time:.1f}s)', flush=True)
print(f'  Cache OFF: {64/nocache_time:.1f} tok/s ({nocache_time:.1f}s)', flush=True)
print(f'  Speedup:   {nocache_time/cache_time:.1f}x', flush=True)

# =====================================================================
# Test 2: lm-eval 1% with KV cache
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 2: lm-eval 1% with KV cache (Ng config)')
print(f'{\"=\" * 70}', flush=True)

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')

TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

results_ng = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

print(f'\\n=== NG (45,52) with layer_idx fix, KV cache ON ===', flush=True)
for task in TASKS:
    data = results_ng['results'].get(task, {})
    for metric in sorted(data.keys()):
        if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

# =====================================================================
# Test 3: Restore baseline and eval
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 3: Baseline (restored)')
print(f'{\"=\" * 70}', flush=True)

inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N
for pos, layer in enumerate(inner.layers):
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_idx'):
        layer.self_attn.layer_idx = pos

lm_base = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
results_base = evaluator.simple_evaluate(model=lm_base, tasks=TASKS, limit=0.01)

print(f'\\n=== BASELINE with KV cache ON ===', flush=True)
for task in TASKS:
    data = results_base['results'].get(task, {})
    for metric in sorted(data.keys()):
        if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

# =====================================================================
# Test 4: Pair config
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 4: Pair (0,7)+(45,52) with layer_idx fix')
print(f'{\"=\" * 70}', flush=True)

new_layers = build_duplicated_model(model, inner, original_layers, [(0, 7), (45, 52)], N)

lm_pair = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
results_pair = evaluator.simple_evaluate(model=lm_pair, tasks=TASKS, limit=0.01)

print(f'\\n=== PAIR with layer_idx fix, KV cache ON ===', flush=True)
for task in TASKS:
    data = results_pair['results'].get(task, {})
    for metric in sorted(data.keys()):
        if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

# =====================================================================
# Comparison with Ng's reported numbers
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('COMPARISON WITH NG REPORTED NUMBERS')
print(f'{\"=\" * 70}', flush=True)
ng_reported = {'IFEval': 79.96, 'BBH': 58.77, 'MATH': 38.97, 'MuSR': 23.72, 'MMLU-PRO': 49.20}
print(f'Ng reported (scale 0-100): {ng_reported}')
print(f'NOTE: 1% subsample — numbers will be noisy. Full run needed for comparison.', flush=True)

# Save
os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/kv_layeridx_fix.json', 'w') as f:
    json.dump({
        'method': 'layer_idx patching (shared weights, no deep copy)',
        'kv_cache_match': match,
        'cache_speedup': nocache_time / cache_time,
        'ng_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results_ng['results'].items()},
        'baseline_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results_base['results'].items()},
        'pair_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results_pair['results'].items()},
    }, f, indent=2)
print('\\nSaved!', flush=True)
"

echo "=== Done at $(date) ==="
