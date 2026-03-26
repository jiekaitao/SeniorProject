#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_v9_%j.log
#SBATCH --job-name=deeppass_kvv9

# v9: Patch DynamicCache to remap shared layer_idx to unique positions
# Instead of changing layer_idx on shared objects (impossible),
# intercept the cache.update() call and remap the index.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache v9: Cache-side remapping ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

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
# THE FIX: Subclass DynamicCache with a layer_idx remapping table
# =====================================================================

class RemappedDynamicCache(DynamicCache):
    \"\"\"DynamicCache that remaps layer_idx for duplicated (shared) layers.

    When layer 45 appears at both position 45 and position 80,
    self_attn.layer_idx is 45 for both calls (shared object).
    This cache intercepts update() and remaps 45 -> the correct unique slot
    based on a call counter.
    \"\"\"
    def __init__(self, num_layers, layer_idx_map=None, **kwargs):
        # Don't pass config — we'll manually create layers
        super().__init__(**kwargs)
        # Manually create the right number of cache layers
        self.layers = [DynamicLayer() for _ in range(num_layers)]
        # Map: original_layer_idx -> list of unique positions
        # e.g. {45: [45, 80], 46: [46, 81], ...}
        self._idx_map = layer_idx_map or {}
        # Track call count per original layer_idx within a forward pass
        self._call_counts = {}

    def reset_call_counts(self):
        self._call_counts = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Remap layer_idx based on how many times this idx has been called
        if layer_idx in self._idx_map:
            positions = self._idx_map[layer_idx]
            count = self._call_counts.get(layer_idx, 0)
            if count < len(positions):
                remapped_idx = positions[count]
            else:
                remapped_idx = layer_idx  # fallback
            self._call_counts[layer_idx] = count + 1
        else:
            remapped_idx = layer_idx

        return super().update(key_states, value_states, remapped_idx, cache_kwargs)


def build_duplicated_model_with_cache_fix(model, inner, original_layers, blocks, N):
    \"\"\"Build duplicated model and return the layer_idx remapping table.\"\"\"
    sorted_blocks = sorted(blocks)
    new_layers = []
    # Track: for each original layer_idx, what positions does it appear at?
    idx_positions = {}

    for idx in range(N):
        pos = len(new_layers)
        new_layers.append(original_layers[idx])
        if idx not in idx_positions:
            idx_positions[idx] = []
        idx_positions[idx].append(pos)

        for (bi, bj) in sorted_blocks:
            if idx == bj - 1:
                for dup_idx in range(bi, bj):
                    pos = len(new_layers)
                    new_layers.append(original_layers[dup_idx])
                    idx_positions[dup_idx].append(pos)

    NEW_N = len(new_layers)
    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = NEW_N
    model.config.layer_types = None

    # Build remapping: only for layers that appear more than once
    layer_idx_map = {k: v for k, v in idx_positions.items() if len(v) > 1}

    print(f'Built: {NEW_N} layers', flush=True)
    print(f'Remapped layers: {list(layer_idx_map.keys())}', flush=True)

    return new_layers, layer_idx_map, NEW_N


# =====================================================================
# Monkey-patch model to use RemappedDynamicCache
# =====================================================================

def patch_model_cache(model, layer_idx_map, num_layers):
    \"\"\"Patch the model to use our remapped cache.\"\"\"

    # Patch _get_cache or the generate method to use our cache
    _orig_prepare = model.prepare_inputs_for_generation

    def patched_prepare(*args, **kwargs):
        result = _orig_prepare(*args, **kwargs)
        # Reset call counts at the start of each forward pass
        past_kv = result.get('past_key_values', None)
        if isinstance(past_kv, RemappedDynamicCache):
            past_kv.reset_call_counts()
        return result

    model.prepare_inputs_for_generation = patched_prepare

    # Also need to ensure our cache class is used instead of DynamicCache
    # Patch the model's _get_cache method
    if hasattr(model, '_get_cache'):
        _orig_get_cache = model._get_cache
        def patched_get_cache(*args, **kwargs):
            cache = RemappedDynamicCache(
                num_layers=num_layers,
                layer_idx_map=layer_idx_map,
            )
            return cache
        model._get_cache = patched_get_cache

    # Also try patching at generate level
    import transformers.cache_utils as cu
    _orig_dc = cu.DynamicCache

    class PatchedDC(RemappedDynamicCache):
        def __init__(self, *args, **kwargs):
            # Ignore config-based init, use our fixed size
            nn.Module.__init__(self) if hasattr(nn.Module, '__init__') else None
            super(DynamicCache, self).__init__()
            self.layers = [DynamicLayer() for _ in range(num_layers)]
            self._idx_map = layer_idx_map
            self._call_counts = {}

    cu.DynamicCache = PatchedDC
    print(f'Patched DynamicCache globally with {num_layers} layers, {len(layer_idx_map)} remapped', flush=True)

    return _orig_dc  # save for restoration


# =====================================================================
# Test: Ng (45,52)
# =====================================================================
print(f'\\n--- Building Ng (45,52) ---', flush=True)
new_layers, layer_idx_map, NEW_N = build_duplicated_model_with_cache_fix(
    model, inner, original_layers, [(45, 52)], N
)

orig_dc = patch_model_cache(model, layer_idx_map, NEW_N)

# Test generation
input_ids = tokenizer('What is the capital of France?', return_tensors='pt').to(model.device)

print('Testing cache ON...', flush=True)
try:
    out_cache = model.generate(**input_ids, max_new_tokens=30, use_cache=True)
    resp_cache = tokenizer.decode(out_cache[0], skip_special_tokens=True)
    print(f'  Cache ON:  {resp_cache[:100]}', flush=True)
    cache_works = True
except Exception as e:
    print(f'  Cache ON FAILED: {e}', flush=True)
    cache_works = False

print('Testing cache OFF...', flush=True)
out_nocache = model.generate(**input_ids, max_new_tokens=30, use_cache=False)
resp_nocache = tokenizer.decode(out_nocache[0], skip_special_tokens=True)
print(f'  Cache OFF: {resp_nocache[:100]}', flush=True)

if cache_works:
    min_len = min(len(out_cache[0]), len(out_nocache[0]))
    match = (out_cache[0][:min_len] == out_nocache[0][:min_len]).all().item()
    print(f'  MATCH: {match}', flush=True)

    # Speed test
    prompt = 'The theory of general relativity describes gravity as curvature of spacetime.'
    inp = tokenizer(prompt, return_tensors='pt').to(model.device)
    model.generate(**inp, max_new_tokens=5, use_cache=True)

    t0 = time.time()
    for _ in range(5):
        model.generate(**inp, max_new_tokens=64, use_cache=True)
    cache_time = (time.time() - t0) / 5

    t0 = time.time()
    for _ in range(5):
        model.generate(**inp, max_new_tokens=64, use_cache=False)
    nocache_time = (time.time() - t0) / 5

    print(f'  Cache: {64/cache_time:.1f} tok/s  NoCache: {64/nocache_time:.1f} tok/s  Speedup: {nocache_time/cache_time:.1f}x', flush=True)

    # If working, run lm-eval 1%
    if match or True:  # Run even if not exact match — check scores
        print(f'\\n--- lm-eval 1% ---', flush=True)
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
        TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
                 'leaderboard_musr', 'leaderboard_mmlu_pro']
        r = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

        for task in TASKS:
            data = r['results'].get(task, {})
            for m in sorted(data.keys()):
                if isinstance(data[m], (int, float)) and 'stderr' not in m:
                    print(f'  {task}/{m}: {data[m]:.4f}', flush=True)

        os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
        with open('results/data/72b/lm_eval/proper/kv_v9_remap.json', 'w') as f:
            json.dump({
                'method': 'RemappedDynamicCache with call-count based idx remapping',
                'match': match if cache_works else False,
                'speedup': nocache_time/cache_time if cache_works else 0,
                'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                           for k, v in r['results'].items()},
            }, f, indent=2)
        print('Saved!', flush=True)

# Restore
import transformers.cache_utils as cu
cu.DynamicCache = orig_dc

print('\\nDone!', flush=True)
"

echo "=== Done at $(date) ==="
