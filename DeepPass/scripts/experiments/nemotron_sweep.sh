#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_nemotron_%j.log
#SBATCH --job-name=nemo_sw

# Nemotron-H 120B (12B active) — Hybrid Mamba+MoE+Attention architecture
# Tests layer duplication on a non-standard architecture:
#   M=Mamba, E=MoE, *=Full Attention
#   Pattern: MEMEMEM*EMEMEMEM*E...
# Key question: which layer TYPES benefit from duplication?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Nemotron-H 120B Hybrid Architecture Sweep ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np

sys.path.insert(0, 'scripts')
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

# Use HF cache path
MODEL_PATH = 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8'
CACHE_DIR = '/blue/cis4914/jietao/hf_cache'
SAVE_DIR = 'results/data/nemotron_h'
os.makedirs(SAVE_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print('Loading Nemotron-H config...', flush=True)
config = AutoConfig.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=True)
pattern = config.hybrid_override_pattern if hasattr(config, 'hybrid_override_pattern') else ''
print(f'Hybrid pattern: {pattern}', flush=True)
print(f'Pattern length: {len(pattern)}, Layers: {config.num_hidden_layers}', flush=True)

# Map layer types
layer_types = []
for i, c in enumerate(pattern):
    if c == 'M':
        layer_types.append('mamba')
    elif c == 'E':
        layer_types.append('moe')
    elif c == '*':
        layer_types.append('attention')
    else:
        layer_types.append(f'unknown_{c}')

type_counts = {}
for t in layer_types:
    type_counts[t] = type_counts.get(t, 0) + 1
print(f'Layer type counts: {type_counts}', flush=True)

# Find attention (*) layer indices
attn_layers = [i for i, t in enumerate(layer_types) if t == 'attention']
moe_layers = [i for i, t in enumerate(layer_types) if t == 'moe']
mamba_layers = [i for i, t in enumerate(layer_types) if t == 'mamba']
print(f'Attention layers: {attn_layers}', flush=True)
print(f'MoE layers (first 10): {moe_layers[:10]}...', flush=True)
print(f'Mamba layers (first 10): {mamba_layers[:10]}...', flush=True)

print('\\nLoading model (FP8, ~120GB)...', flush=True)
t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, cache_dir=CACHE_DIR, device_map='auto',
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    print(f'Loaded in {time.time()-t0:.0f}s', flush=True)
except Exception as e:
    print(f'Failed to load: {e}', flush=True)
    import traceback
    traceback.print_exc()
    # Try without trust_remote_code
    print('Retrying without trust_remote_code...', flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, cache_dir=CACHE_DIR, device_map='auto',
            torch_dtype=torch.bfloat16,
        )
        print(f'Loaded (retry) in {time.time()-t0:.0f}s', flush=True)
    except Exception as e2:
        print(f'Retry also failed: {e2}', flush=True)
        traceback.print_exc()
        sys.exit(1)

# Discover layer structure
inner = None
for attr in ['model', 'transformer', 'backbone']:
    if hasattr(model, attr):
        inner = getattr(model, attr)
        break

if inner is None:
    print('Could not find inner model. Attributes:', flush=True)
    for attr in dir(model):
        if not attr.startswith('_'):
            obj = getattr(model, attr)
            if isinstance(obj, nn.Module) and not callable(obj):
                print(f'  model.{attr}: {type(obj).__name__}', flush=True)
    sys.exit(1)

layers_attr = None
for attr in ['layers', 'h', 'blocks']:
    if hasattr(inner, attr):
        layers_attr = attr
        break

if layers_attr is None:
    print('Could not find layers. Inner model attributes:', flush=True)
    for attr in dir(inner):
        if not attr.startswith('_'):
            obj = getattr(inner, attr)
            if isinstance(obj, nn.Module):
                print(f'  inner.{attr}: {type(obj).__name__}', flush=True)
    sys.exit(1)

original_layers = list(getattr(inner, layers_attr))
N = len(original_layers)
print(f'Found {N} layers via inner.{layers_attr}', flush=True)
print(f'Layer 0 type: {type(original_layers[0]).__name__}', flush=True)
if N > 7:
    print(f'Layer 7 type: {type(original_layers[7]).__name__}', flush=True)
    print(f'Layer 8 type: {type(original_layers[8]).__name__}', flush=True)

eq_all = _load_questions()

def set_num_layers(n):
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def apply_blocks(blocks):
    order = build_order(blocks, N)
    setattr(inner, layers_attr, nn.ModuleList([original_layers[idx] for idx in order]))
    set_num_layers(len(order))
    # Update hybrid pattern if exists
    if hasattr(model.config, 'hybrid_override_pattern') and model.config.hybrid_override_pattern:
        orig_pattern = list(pattern)
        new_pattern = [orig_pattern[idx] for idx in order]
        model.config.hybrid_override_pattern = ''.join(new_pattern)

def restore():
    setattr(inner, layers_attr, nn.ModuleList(original_layers))
    set_num_layers(N)
    if hasattr(model.config, 'hybrid_override_pattern'):
        model.config.hybrid_override_pattern = pattern

def generate_no_cache(prompt, max_new_tokens=64):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def gen(p): return generate_no_cache(p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(p, max_new_tokens=128)

def evaluate(blocks, name):
    if blocks:
        apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    if blocks:
        restore()
    ltype = layer_types[blocks[0][0]] if blocks else 'none'
    print(f'  {name:45s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} type={ltype} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks] if blocks else [],
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined,
            'layer_type': ltype}

def save_checkpoint(data, name):
    with open(f'{SAVE_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [SAVED] {name}.json', flush=True)

# ======================================================================
# 1. Baseline
# ======================================================================
print('\\n=== Baseline ===', flush=True)
baseline = evaluate([], 'baseline')
save_checkpoint(baseline, 'baseline')

# ======================================================================
# 2. Sweep by layer type: test one of each type
# ======================================================================
print('\\n=== Layer Type Sweep ===', flush=True)

# Test representative layers from each type
test_indices = []
# Attention layers (*)
for idx in attn_layers[:4]:
    test_indices.append((idx, 'attention'))
# MoE layers (E) - sample every 10th
for idx in moe_layers[::10][:4]:
    test_indices.append((idx, 'moe'))
# Mamba layers (M) - sample every 10th
for idx in mamba_layers[::10][:4]:
    test_indices.append((idx, 'mamba'))

type_results = []
for idx, ltype in test_indices:
    block = [(idx, idx + 1)]
    r = evaluate(block, f'single L{idx} ({ltype})')
    r['layer_type'] = ltype
    type_results.append(r)

# Sort by combined
type_results.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nResults by type:', flush=True)
for r in type_results:
    delta = r['combined'] - baseline['combined']
    print(f'  {r[\"name\"]:45s} combined={r[\"combined\"]:.2f} delta={delta:+.2f}', flush=True)

# Aggregate by type
type_avg = {}
for r in type_results:
    t = r['layer_type']
    if t not in type_avg:
        type_avg[t] = []
    type_avg[t].append(r['combined'] - baseline['combined'])

print(f'\\nAverage delta by layer type:', flush=True)
for t, deltas in type_avg.items():
    print(f'  {t:12s}: avg_delta={np.mean(deltas):+.2f} (n={len(deltas)})', flush=True)

save_checkpoint({
    'baseline': baseline,
    'layer_types': layer_types,
    'type_results': type_results,
    'type_averages': {t: {'mean_delta': float(np.mean(d)), 'n': len(d)} for t, d in type_avg.items()},
}, 'layer_type_sweep')

# ======================================================================
# 3. Best block deep dive — attn-only vs full
# ======================================================================
if type_results:
    best = type_results[0]
    best_block = [tuple(b) for b in best['blocks']]
    best_idx = best_block[0][0]
    best_type = best['layer_type']
    print(f'\\n=== Best Block Deep Dive: L{best_idx} ({best_type}) ===', flush=True)

    # Check if this layer has separable attn + mlp
    module = original_layers[best_idx]
    has_attn = hasattr(module, 'self_attn') or hasattr(module, 'attention')
    has_mlp = hasattr(module, 'mlp') or hasattr(module, 'feed_forward')
    print(f'  has_attn={has_attn} has_mlp={has_mlp}', flush=True)
    print(f'  Submodules: {[n for n, _ in module.named_children()]}', flush=True)

    if has_mlp:
        mlp_attr = 'mlp' if hasattr(module, 'mlp') else 'feed_forward'
        mlp_module = getattr(module, mlp_attr)

        # Attn-only test
        apply_blocks(best_block)
        counter = [0]
        def make_zero(ctr):
            def hook(mod, inp, out):
                ctr[0] += 1
                if ctr[0] % 2 == 0:
                    if isinstance(out, tuple):
                        return (0.0 * out[0],) + out[1:]
                    return 0.0 * out
                return out
            return hook
        h = mlp_module.register_forward_hook(make_zero(counter))
        t0 = time.time()
        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        attn_only = math_r['score'] * 50 + eq_r['score'] * 0.5
        h.remove()
        restore()
        print(f'  attn_only: combined={attn_only:.2f} ({time.time()-t0:.0f}s)', flush=True)
        print(f'  FFN impact: {attn_only - best[\"combined\"]:+.2f}', flush=True)

        save_checkpoint({
            'best_layer': best_idx, 'best_type': best_type,
            'full_dup': best['combined'],
            'attn_only': attn_only,
            'ffn_impact': attn_only - best['combined'],
        }, 'sublayer_analysis')

print(f'\\n{\"=\" * 60}', flush=True)
print('COMPLETE', flush=True)
print(f'{\"=\" * 60}', flush=True)
"

echo "=== Finished: $(date) ==="
