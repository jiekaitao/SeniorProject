#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_kv_cache_fix_%j.log
#SBATCH --job-name=kv_fix

# KV Cache Fix for Layer Duplication
# Problem: shared modules have same layer_idx → cache slot collision
# Fix: thin wrapper that temporarily swaps layer_idx during forward pass
# Tests: cached vs uncached generation, correctness check, speed benchmark

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== KV Cache Fix Test ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/kv_cache_fix'
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================================
# The Fix: LayerIdxWrapper
# ======================================================================
class LayerIdxWrapper(nn.Module):
    \"\"\"Wraps a decoder layer to give it a unique layer_idx for KV cache.

    Since duplicated layers share the same module object, we can't just
    set layer_idx permanently. Instead, temporarily swap it during forward.
    \"\"\"
    def __init__(self, layer, new_layer_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_layer_idx
        # Store original for restoration
        self.original_layer_idx = layer.layer_idx
        self.original_attn_idx = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None

    def forward(self, *args, **kwargs):
        # Swap layer_idx to our unique index
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'):
            self.layer.self_attn.layer_idx = self.new_layer_idx
        elif hasattr(self.layer, 'attention'):
            self.layer.attention.layer_idx = self.new_layer_idx
        try:
            return self.layer(*args, **kwargs)
        finally:
            # Restore original (so the other copy still works)
            self.layer.layer_idx = self.original_layer_idx
            if self.original_attn_idx is not None:
                self.layer.self_attn.layer_idx = self.original_attn_idx
            elif hasattr(self.layer, 'attention'):
                self.layer.attention.layer_idx = self.original_layer_idx

    # Delegate attribute access to wrapped layer for compatibility
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


def apply_duplication_with_cache_fix(model, blocks):
    \"\"\"Apply multi-block duplication with KV cache fix.

    Duplicated layers get wrapped with LayerIdxWrapper so each physical
    position in the ModuleList has a unique layer_idx for the cache.
    \"\"\"
    inner = model.model
    if hasattr(inner, 'language_model'):
        inner = inner.language_model

    original_layers = list(inner.layers)
    N = len(original_layers)

    # Build execution order
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))

    new_N = len(order)

    # Track which positions are duplicates (second occurrence of a layer)
    seen = set()
    is_duplicate = []
    for idx in order:
        is_duplicate.append(idx in seen)
        seen.add(idx)

    # Build new layer list: wrap DUPLICATES with LayerIdxWrapper
    new_layers = []
    for physical_idx, (orig_idx, is_dup) in enumerate(zip(order, is_duplicate)):
        layer = original_layers[orig_idx]
        if is_dup:
            # This is a duplicate — wrap it with unique layer_idx
            wrapped = LayerIdxWrapper(layer, physical_idx)
            new_layers.append(wrapped)
        else:
            # Original — just patch layer_idx directly (safe, first occurrence)
            layer.layer_idx = physical_idx
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = physical_idx
            elif hasattr(layer, 'attention'):
                layer.attention.layer_idx = physical_idx
            new_layers.append(layer)

    inner.layers = nn.ModuleList(new_layers)

    # Update all configs
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = new_N
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[orig_idx] for orig_idx in order]
        # ENABLE cache
        if hasattr(cfg, 'use_cache'):
            cfg.use_cache = True

    print(f'Applied duplication: {N} -> {new_N} layers ({sum(is_duplicate)} wrapped duplicates)', flush=True)
    print(f'Layer idx mapping: {[(i, order[i], \"DUP\" if is_duplicate[i] else \"orig\") for i in range(min(new_N, 10))]}...', flush=True)

    return original_layers, N

# ======================================================================
# Load model
# ======================================================================
print('Loading Gemma3-27B...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
device = next(model.parameters()).device

# ======================================================================
# Test 1: Simple generation WITHOUT cache (baseline)
# ======================================================================
print('\\n=== Test 1: Uncached generation (baseline) ===', flush=True)
BLOCKS = [(12, 13)]  # Simple single-block test first

# Save original state
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
orig_layers_backup = list(inner.layers)
orig_N = len(orig_layers_backup)

# Apply duplication WITHOUT cache fix (old method)
from layer_duplicator import generate_no_cache
order = []
prev = 0
for (i, j) in sorted(BLOCKS):
    order.extend(range(prev, j))
    order.extend(range(i, j))
    prev = j
order.extend(range(prev, orig_N))

inner.layers = nn.ModuleList([orig_layers_backup[idx] for idx in order])
for cfg in [model.config, getattr(model.config, 'text_config', None)]:
    if cfg is None: continue
    if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = len(order)
    if hasattr(cfg, 'layer_types') and cfg.layer_types:
        cfg.layer_types = [cfg.layer_types[idx] for idx in order]
    if hasattr(cfg, 'use_cache'): cfg.use_cache = False

test_prompt = 'What is 127 * 348? Answer with just the number:'
print(f'Prompt: {test_prompt}', flush=True)

t0 = time.time()
nocache_output = generate_no_cache(model, tokenizer, test_prompt, max_new_tokens=32)
nocache_time = time.time() - t0
print(f'No-cache output: {nocache_output}', flush=True)
print(f'No-cache time: {nocache_time:.1f}s', flush=True)

# Restore
inner.layers = nn.ModuleList(orig_layers_backup)
for cfg in [model.config, getattr(model.config, 'text_config', None)]:
    if cfg is None: continue
    if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = orig_N

# ======================================================================
# Test 2: Generation WITH cache fix
# ======================================================================
print('\\n=== Test 2: Cached generation (KV fix) ===', flush=True)

original_layers, orig_N = apply_duplication_with_cache_fix(model, BLOCKS)

t0 = time.time()
try:
    inputs = tokenizer(test_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )
    cached_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    cached_time = time.time() - t0
    print(f'Cached output: {cached_output}', flush=True)
    print(f'Cached time: {cached_time:.1f}s', flush=True)
    cache_works = True
except Exception as e:
    print(f'CACHE FAILED: {e}', flush=True)
    import traceback
    traceback.print_exc()
    cached_output = 'FAILED'
    cached_time = -1
    cache_works = False

# ======================================================================
# Test 3: Output consistency check
# ======================================================================
print('\\n=== Test 3: Consistency ===', flush=True)
if cache_works:
    match = nocache_output.strip()[:20] == cached_output.strip()[:20]
    print(f'No-cache first 20 chars: {nocache_output.strip()[:20]}', flush=True)
    print(f'Cached first 20 chars:   {cached_output.strip()[:20]}', flush=True)
    print(f'Match: {match}', flush=True)
    if not match:
        print('WARNING: Outputs differ! Cache may be corrupting generation.', flush=True)
else:
    print('Skipped (cache failed)', flush=True)

# ======================================================================
# Test 4: Speed benchmark (5 prompts)
# ======================================================================
if cache_works:
    print('\\n=== Test 4: Speed Benchmark ===', flush=True)
    speed_prompts = [
        'What is 99999 * 99999?',
        'Explain quantum entanglement in one sentence.',
        'What is the capital of Australia?',
        'Write a haiku about rain.',
        'What is 2 raised to the power of 16?',
    ]

    cached_times = []
    for prompt in speed_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
        cached_times.append(time.time() - t0)

    # Restore for uncached benchmark
    inner.layers = nn.ModuleList(orig_layers_backup)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = orig_N
        if hasattr(cfg, 'use_cache'): cfg.use_cache = False
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = cfg.layer_types[:orig_N]

    # Re-apply duplication for uncached
    inner.layers = nn.ModuleList([orig_layers_backup[idx] for idx in order])
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = len(order)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[idx % len(cfg.layer_types)] for idx in order]

    uncached_times = []
    for prompt in speed_prompts:
        t0 = time.time()
        generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
        uncached_times.append(time.time() - t0)

    import numpy as np
    avg_cached = np.mean(cached_times)
    avg_uncached = np.mean(uncached_times)
    speedup = avg_uncached / avg_cached if avg_cached > 0 else 0

    print(f'Avg cached: {avg_cached:.1f}s', flush=True)
    print(f'Avg uncached: {avg_uncached:.1f}s', flush=True)
    print(f'Speedup: {speedup:.1f}x', flush=True)

# ======================================================================
# Test 5: Multi-block test
# ======================================================================
print('\\n=== Test 5: Multi-block cache fix ===', flush=True)

# Restore original
inner.layers = nn.ModuleList(orig_layers_backup)
for cfg in [model.config, getattr(model.config, 'text_config', None)]:
    if cfg is None: continue
    if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = orig_N

MULTI_BLOCKS = [(0, 2), (12, 13), (47, 48)]
original_layers, orig_N = apply_duplication_with_cache_fix(model, MULTI_BLOCKS)

try:
    inputs = tokenizer('What is the meaning of life?', return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    multi_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'Multi-block output: {multi_output}', flush=True)
    print('Multi-block cache: SUCCESS', flush=True)
    multi_works = True
except Exception as e:
    print(f'Multi-block FAILED: {e}', flush=True)
    import traceback
    traceback.print_exc()
    multi_works = False

# ======================================================================
# Test 6: Math probe with cache (the real test)
# ======================================================================
if multi_works:
    print('\\n=== Test 6: Math Probe with Cache ===', flush=True)

    def gen_cached(prompt):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
        return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen_cached, verbose=True)
    cached_math_time = time.time() - t0
    print(f'\\nCached math score: {math_r[\"score\"]:.4f} ({cached_math_time:.0f}s)', flush=True)

# ======================================================================
# SUMMARY
# ======================================================================
print(f'\\n{\"=\" * 50}', flush=True)
print('KV CACHE FIX RESULTS', flush=True)
print(f'{\"=\" * 50}', flush=True)
print(f'Single-block cache: {\"SUCCESS\" if cache_works else \"FAILED\"}', flush=True)
print(f'Multi-block cache:  {\"SUCCESS\" if multi_works else \"FAILED\"}', flush=True)
if cache_works:
    print(f'Speedup: {speedup:.1f}x', flush=True)
    print(f'Output consistency: {\"MATCH\" if match else \"MISMATCH\"}', flush=True)
if multi_works:
    print(f'Cached math score: {math_r[\"score\"]:.4f}', flush=True)
print('COMPLETE', flush=True)

# Save
results = {
    'single_block_works': cache_works,
    'multi_block_works': multi_works,
    'speedup': float(speedup) if cache_works else None,
    'output_match': match if cache_works else None,
    'cached_math_score': math_r['score'] if multi_works else None,
    'nocache_output': nocache_output,
    'cached_output': cached_output if cache_works else None,
}
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
