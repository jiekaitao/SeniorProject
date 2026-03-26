#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_v4_%j.log
#SBATCH --job-name=deeppass_kvv4

# KV cache fix v4: patch layer_idx hooks + extend layer_types list

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache Fix v4 ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('Loading model...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True
)

inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

def apply_duplication_with_kv_fix(model, inner, original_layers, blocks, N):
    \"\"\"Apply layer duplication with proper KV cache fix. Returns hook handles.\"\"\"
    sorted_blocks = sorted(blocks)
    new_layers = []
    # Track which original indices go where
    for idx in range(N):
        new_layers.append(original_layers[idx])
        for (bi, bj) in sorted_blocks:
            if idx == bj - 1:
                for dup_idx in range(bi, bj):
                    new_layers.append(original_layers[dup_idx])

    NEW_N = len(new_layers)
    inner.layers = nn.ModuleList(new_layers)

    # FIX 1: Update num_hidden_layers
    model.config.num_hidden_layers = NEW_N

    # FIX 2: Extend layer_types to match new layer count
    if hasattr(model.config, 'layer_types') and model.config.layer_types is not None:
        orig_types = model.config.layer_types
        new_types = []
        for idx in range(N):
            new_types.append(orig_types[idx])
            for (bi, bj) in sorted_blocks:
                if idx == bj - 1:
                    for dup_idx in range(bi, bj):
                        new_types.append(orig_types[dup_idx])
        model.config.layer_types = new_types
        print(f'  layer_types extended: {len(orig_types)} -> {len(new_types)}', flush=True)

    # FIX 3: Pre-forward hooks for layer_idx
    hooks = []
    for pos in range(NEW_N):
        layer = new_layers[pos]
        def make_hook(target_idx):
            def hook_fn(module, args):
                module.self_attn.layer_idx = target_idx
            return hook_fn
        hooks.append(layer.register_forward_pre_hook(make_hook(pos)))

    # Verify cache size
    test_cache = DynamicCache(config=model.config)
    assert len(test_cache.layers) == NEW_N, f'Cache size {len(test_cache.layers)} != {NEW_N}'
    print(f'  Cache size: {len(test_cache.layers)} (correct!)', flush=True)

    print(f'  Built {NEW_N} layers, {len(hooks)} hooks, cache verified', flush=True)
    return new_layers, hooks

# =====================================================================
# Test Ng (45,52)
# =====================================================================
print(f'\\n--- Ng (45,52) ---', flush=True)
new_layers, hooks = apply_duplication_with_kv_fix(model, inner, original_layers, [(45, 52)], N)

input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)
out_cache = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
out_nocache = model.generate(**input_ids, max_new_tokens=30, use_cache=False)
resp_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
resp_nocache = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'  Cache ON:  {resp_cache[:100]}', flush=True)
print(f'  Cache OFF: {resp_nocache[:100]}', flush=True)
min_len = min(len(out_cache[0]), len(out_nocache[0]))
match = (out_cache[0][:min_len] == out_nocache[0][:min_len]).all().item()
print(f'  MATCH: {match}', flush=True)

# Speed
prompt = 'The theory of general relativity describes gravity as curvature of spacetime.'
inp = tokenizer(prompt, return_tensors='pt').to(model.device)
model.generate(**inp, max_new_tokens=5, use_cache=True)
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=True)
cache_time = (time.time() - t0) / 3
t0 = time.time()
for _ in range(3): model.generate(**inp, max_new_tokens=64, use_cache=False)
nocache_time = (time.time() - t0) / 3
print(f'  Cache: {64/cache_time:.1f} tok/s  NoCache: {64/nocache_time:.1f} tok/s  Speedup: {nocache_time/cache_time:.1f}x', flush=True)

# Clean up
for h in hooks: h.remove()

# =====================================================================
# If match, run lm-eval 1% on Ng + baseline + pair
# =====================================================================
if match:
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
             'leaderboard_musr', 'leaderboard_mmlu_pro']
    all_results = {}

    # Baseline
    print(f'\\n--- Baseline lm-eval 1% ---', flush=True)
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N
    if hasattr(model.config, 'layer_types'):
        model.config.layer_types = ['full_attention'] * N
    for pos, layer in enumerate(inner.layers):
        layer.self_attn.layer_idx = pos
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
    r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)
    for task in TASKS:
        data = r['results'].get(task, {})
        for m in sorted(data.keys()):
            if isinstance(data[m], (int, float)) and 'stderr' not in m:
                print(f'  {task}/{m}: {data[m]:.4f}', flush=True)
    all_results['baseline'] = {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}

    # Ng
    print(f'\\n--- Ng (45,52) lm-eval 1% ---', flush=True)
    new_layers, hooks = apply_duplication_with_kv_fix(model, inner, original_layers, [(45, 52)], N)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
    r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)
    for task in TASKS:
        data = r['results'].get(task, {})
        for m in sorted(data.keys()):
            if isinstance(data[m], (int, float)) and 'stderr' not in m:
                print(f'  {task}/{m}: {data[m]:.4f}', flush=True)
    all_results['ng'] = {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}
    for h in hooks: h.remove()

    # Pair
    print(f'\\n--- Pair (0,7)+(45,52) lm-eval 1% ---', flush=True)
    new_layers, hooks = apply_duplication_with_kv_fix(model, inner, original_layers, [(0,7),(45,52)], N)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
    r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)
    for task in TASKS:
        data = r['results'].get(task, {})
        for m in sorted(data.keys()):
            if isinstance(data[m], (int, float)) and 'stderr' not in m:
                print(f'  {task}/{m}: {data[m]:.4f}', flush=True)
    all_results['pair'] = {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in r['results'].items()}
    for h in hooks: h.remove()

    # Save
    os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
    with open('results/data/72b/lm_eval/proper/kv_v4_results.json', 'w') as f:
        json.dump({'method': 'hook layer_idx + layer_types fix', 'match': match,
                   'speedup': nocache_time/cache_time, 'results': all_results}, f, indent=2)
    print('\\nSaved!', flush=True)
else:
    print('\\nMATCH FAILED — not running lm-eval.', flush=True)
"

echo "=== Done at $(date) ==="
