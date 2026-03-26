#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_hook_%j.log
#SBATCH --job-name=deeppass_kvhk

# FIX: Use pre-forward hooks to temporarily set layer_idx before each call
# This handles shared layer objects correctly

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache Fix: Hook-based layer_idx ==="
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

def build_duplicated_model_with_kv_fix(model, inner, original_layers, blocks, N):
    \"\"\"Build duplicated model with shared weights and KV cache fix via hooks.\"\"\"
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
    print(f'Built: {len(new_layers)} layers (shared weights)', flush=True)

    # Install pre-forward hooks that set layer_idx just before each call
    # This is safe because hooks fire sequentially during forward pass
    hooks = []
    for pos in range(len(new_layers)):
        layer = new_layers[pos]
        def make_hook(target_idx):
            def hook_fn(module, args, kwargs=None):
                module.self_attn.layer_idx = target_idx
            return hook_fn
        hooks.append(layer.register_forward_pre_hook(make_hook(pos)))

    print(f'Installed {len(hooks)} pre-forward hooks for layer_idx', flush=True)
    return new_layers, hooks

# =====================================================================
# Test: Ng (45,52)
# =====================================================================
print(f'\\nBuilding Ng (45,52)...', flush=True)
new_layers, hooks = build_duplicated_model_with_kv_fix(
    model, inner, original_layers, [(45, 52)], N
)

# Verify layer_idx changes dynamically
print(f'Before forward: layer 45 idx = {inner.layers[45].self_attn.layer_idx}', flush=True)

# Test generation
input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)

print('\\nTesting cache ON vs OFF...', flush=True)
out_cache = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
resp_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
print(f'  Cache ON:  {resp_cache[:100]}', flush=True)

out_nocache = model.generate(**input_ids, max_new_tokens=30, use_cache=False)
resp_nocache = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'  Cache OFF: {resp_nocache[:100]}', flush=True)

min_len = min(len(out_cache[0]), len(out_nocache[0]))
match = (out_cache[0][:min_len] == out_nocache[0][:min_len]).all().item()
print(f'  Token match: {match}', flush=True)

if not match:
    # Show where they diverge
    for i in range(min_len):
        if out_cache[0][i] != out_nocache[0][i]:
            print(f'  First divergence at token {i}: cache={out_cache[0][i].item()} nocache={out_nocache[0][i].item()}', flush=True)
            break

# Speed test
prompt = 'The theory of general relativity describes gravity as curvature of spacetime.'
inp = tokenizer(prompt, return_tensors='pt').to(model.device)
model.generate(**inp, max_new_tokens=5, use_cache=True)

t0 = time.time()
for _ in range(3):
    model.generate(**inp, max_new_tokens=64, use_cache=True)
cache_time = (time.time() - t0) / 3

t0 = time.time()
for _ in range(3):
    model.generate(**inp, max_new_tokens=64, use_cache=False)
nocache_time = (time.time() - t0) / 3

print(f'\\n  Cache ON:  {64/cache_time:.1f} tok/s ({cache_time:.1f}s)', flush=True)
print(f'  Cache OFF: {64/nocache_time:.1f} tok/s ({nocache_time:.1f}s)', flush=True)
print(f'  Speedup:   {nocache_time/cache_time:.1f}x', flush=True)

# lm-eval 1% test
print(f'\\n--- lm-eval 1% (Ng with hook fix) ---', flush=True)
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

results_ng = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)
for task in TASKS:
    data = results_ng['results'].get(task, {})
    for metric in sorted(data.keys()):
        if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

# Clean up hooks
for h in hooks:
    h.remove()

# =====================================================================
# Compare with deep-copy results
# =====================================================================
print(f'\\n--- Comparing with deep-copy eval ---', flush=True)
try:
    dc = json.load(open('results/data/72b/lm_eval/proper/ng_saved_1pct.json'))
    print('Deep-copy results found, comparing...', flush=True)
    for task in TASKS:
        dc_data = dc.get('results', {}).get(task, {})
        hook_data = results_ng['results'].get(task, {})
        for metric in ['acc_norm,none', 'exact_match,none', 'inst_level_loose_acc,none', 'acc,none']:
            if metric in dc_data and metric in hook_data:
                dc_val = dc_data[metric]
                hk_val = hook_data[metric]
                match_str = 'MATCH' if abs(dc_val - hk_val) < 0.001 else f'DIFF ({dc_val:.4f} vs {hk_val:.4f})'
                print(f'  {task}/{metric}: {match_str}', flush=True)
except:
    print('No deep-copy results to compare yet.', flush=True)

# Save
os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/kv_hook_fix.json', 'w') as f:
    json.dump({
        'method': 'pre-forward hook layer_idx patching (shared weights)',
        'kv_cache_match': match,
        'cache_speedup': nocache_time / cache_time,
        'ng_results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results_ng['results'].items()},
    }, f, indent=2)
print('\\nSaved!', flush=True)
"

echo "=== Done at $(date) ==="
