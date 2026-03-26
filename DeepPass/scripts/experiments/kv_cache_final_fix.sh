#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_final_%j.log
#SBATCH --job-name=deeppass_kvfn

# Final KV cache fix: hook-based layer_idx + ensure cache is sized correctly

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache Final Fix ==="
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
print(f'Loaded: {N} layers', flush=True)
print(f'Config num_hidden_layers: {model.config.num_hidden_layers}', flush=True)

# Build duplicated model
sorted_blocks = [(45, 52)]
new_layers = []
for idx in range(N):
    new_layers.append(original_layers[idx])
    for (bi, bj) in sorted_blocks:
        if idx == bj - 1:
            for dup_idx in range(bi, bj):
                new_layers.append(original_layers[dup_idx])

NEW_N = len(new_layers)
inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = NEW_N
print(f'Built: {NEW_N} layers', flush=True)
print(f'Config updated to: {model.config.num_hidden_layers}', flush=True)

# Install pre-forward hooks for layer_idx
hooks = []
for pos in range(NEW_N):
    layer = new_layers[pos]
    def make_hook(target_idx):
        def hook_fn(module, args):
            module.self_attn.layer_idx = target_idx
        return hook_fn
    hooks.append(layer.register_forward_pre_hook(make_hook(pos)))
print(f'Installed {len(hooks)} hooks', flush=True)

# Debug: check what DynamicCache sees
print(f'\\nDebug cache creation...', flush=True)
test_cache = DynamicCache(config=model.config)
print(f'  Cache layers count: {len(test_cache.layers)}', flush=True)
print(f'  Expected: {NEW_N}', flush=True)

if len(test_cache.layers) < NEW_N:
    print(f'  PROBLEM: Cache too small! Investigating...', flush=True)
    # Check what config the cache sees
    import inspect
    src = inspect.getsource(DynamicCache.__init__)
    # Find num_hidden_layers usage
    for line in src.split('\\n'):
        if 'num_hidden' in line or 'num_layer' in line or 'n_layer' in line:
            print(f'    {line.strip()}', flush=True)

    # Try to find what attribute it reads
    decoder_config = model.config.get_text_config(decoder=True) if hasattr(model.config, 'get_text_config') else model.config
    print(f'  decoder_config.num_hidden_layers: {decoder_config.num_hidden_layers}', flush=True)

    # Check if there's a text_config
    if hasattr(model.config, 'text_config'):
        print(f'  text_config.num_hidden_layers: {model.config.text_config.num_hidden_layers}', flush=True)
        model.config.text_config.num_hidden_layers = NEW_N
        print(f'  Fixed text_config to {NEW_N}', flush=True)

    # Also try setting on decoder config
    decoder_config.num_hidden_layers = NEW_N

    # Re-test
    test_cache2 = DynamicCache(config=model.config)
    print(f'  Cache layers after fix: {len(test_cache2.layers)}', flush=True)

# If cache is still wrong, monkey-patch DynamicCache
if len(DynamicCache(config=model.config).layers) < NEW_N:
    print(f'  Still wrong. Patching DynamicCache.__init__...', flush=True)
    _orig_dc_init = DynamicCache.__init__
    def _patched_dc_init(self, *args, **kwargs):
        _orig_dc_init(self, *args, **kwargs)
        while len(self.layers) < NEW_N:
            from transformers.cache_utils import DefaultCache
            self.layers.append(DefaultCache())
    DynamicCache.__init__ = _patched_dc_init
    test_cache3 = DynamicCache(config=model.config)
    print(f'  After monkey-patch: {len(test_cache3.layers)} layers', flush=True)

# Now test
print(f'\\n--- Testing generation ---', flush=True)
input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)

try:
    out_cache = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
    resp_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
    print(f'  Cache ON:  {resp_cache[:100]}', flush=True)
    cache_works = True
except Exception as e:
    print(f'  Cache ON FAILED: {e}', flush=True)
    cache_works = False

out_nocache = model.generate(**input_ids, max_new_tokens=30, use_cache=False)
resp_nocache = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'  Cache OFF: {resp_nocache[:100]}', flush=True)

if cache_works:
    min_len = min(len(out_cache[0]), len(out_nocache[0]))
    match = (out_cache[0][:min_len] == out_nocache[0][:min_len]).all().item()
    print(f'  Token match: {match}', flush=True)
    if not match:
        for i in range(min_len):
            if out_cache[0][i] != out_nocache[0][i]:
                print(f'  Diverge at token {i}: cache={tokenizer.decode([out_cache[0][i].item()])} nocache={tokenizer.decode([out_nocache[0][i].item()])}', flush=True)
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

    print(f'  Cache ON:  {64/cache_time:.1f} tok/s', flush=True)
    print(f'  Cache OFF: {64/nocache_time:.1f} tok/s', flush=True)
    print(f'  Speedup:   {nocache_time/cache_time:.1f}x', flush=True)

    # lm-eval 1%
    if match:
        print(f'\\n--- lm-eval 1% ---', flush=True)
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
        TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
                 'leaderboard_musr', 'leaderboard_mmlu_pro']
        results = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

        for task in TASKS:
            data = results['results'].get(task, {})
            for metric in sorted(data.keys()):
                if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
                    print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

        os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
        with open('results/data/72b/lm_eval/proper/kv_final_fix.json', 'w') as f:
            json.dump({
                'method': 'hook layer_idx + cache size fix',
                'match': match, 'speedup': nocache_time/cache_time,
                'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                           for k, v in results['results'].items()},
            }, f, indent=2)
    else:
        print('\\nSkipping lm-eval — outputs dont match.', flush=True)
else:
    print('\\nCache failed. Need different approach.', flush=True)

# Cleanup hooks
for h in hooks:
    h.remove()

print('\\nDone!', flush=True)
"

echo "=== Done at $(date) ==="
