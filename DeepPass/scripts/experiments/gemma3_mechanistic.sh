#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_mechanistic_%j.log
#SBATCH --job-name=g3_mech

# Mechanistic analysis on Gemma3-27B:
# 1. Attention-only vs full duplication (per-layer decomposition)
# 2. Jaccard instability (gate overlap between passes)
# 3. FFN danger scores per layer
# 4. Full EQ-bench (171 questions) on best configs
# 5. Inference speed benchmark

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3-27B Mechanistic Analysis ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/mechanistic'
os.makedirs(SAVE_DIR, exist_ok=True)

print('=' * 70, flush=True)
print('GEMMA3-27B MECHANISTIC ANALYSIS', flush=True)
print(f'Date: {datetime.now().isoformat()}', flush=True)
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)

eq_all = _load_questions()

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
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
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))
    return order

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

def evaluate(blocks, name, full_eq=False):
    apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all if full_eq else None, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()
    print(f'  {name:55s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks],
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# ======================================================================
# 0. Baseline
# ======================================================================
print('\\n=== Baseline ===', flush=True)
t0 = time.time()
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline (full EQ 20q): math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)

# ======================================================================
# 1. ATTENTION-ONLY vs FULL DUPLICATION
# ======================================================================
print('\\n=== 1. Attention-Only vs Full Duplication ===', flush=True)
# Test on the best triple's individual blocks: (0,2), (12,13), (47,48)
# and on the full best triple
test_blocks = [
    [(0, 2)],
    [(12, 13)],
    [(47, 48)],
    [(0, 2), (12, 13), (47, 48)],
]

attn_only_results = []

for blocks in test_blocks:
    name_b = '+'.join(f'({i},{j})' for i, j in blocks)

    # Full duplication
    full_r = evaluate(blocks, f'full {name_b}')

    # Attention-only: hook MLPs to zero their output on second pass
    order = apply_blocks(blocks)
    sorted_b = sorted(blocks)
    dup_layers = []
    for (i, j) in sorted_b:
        for l in range(i, j):
            dup_layers.append(l)

    hooks = []
    for layer_idx in dup_layers:
        module = original_layers[layer_idx]
        counter = [0]
        def make_hook(ctr):
            def hook(module, input, output):
                ctr[0] += 1
                if ctr[0] % 2 == 0:  # second pass: zero FFN
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                return output
            return hook
        h = module.mlp.register_forward_hook(make_hook(counter))
        hooks.append(h)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    attn_combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0

    for h in hooks:
        h.remove()
    restore()

    print(f'  attn-only {name_b:44s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={attn_combined:.2f} ({elapsed:.0f}s)', flush=True)

    delta = attn_combined - full_r['combined']
    print(f'    FFN impact: {delta:+.2f} (positive = FFN hurts)', flush=True)

    attn_only_results.append({
        'blocks': [list(b) for b in blocks],
        'full': full_r,
        'attn_only': {'math': math_r['score'], 'eq': eq_r['score'], 'combined': attn_combined},
        'ffn_impact': delta,
    })

# ======================================================================
# 2. JACCARD INSTABILITY (gate overlap between passes)
# ======================================================================
print('\\n=== 2. Jaccard Instability ===', flush=True)

# Measure on the best triple's layers
triple_blocks = [(0, 2), (12, 13), (47, 48)]
test_layers = [0, 1, 12, 47]
prompts = [
    'What is 127 * 348?',
    'What is the square root of 152399025?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'How would a parent feel seeing their child graduate?',
    'What is the capital of France?',
    'Explain the concept of entropy in thermodynamics.',
]

jaccard_results = {}
for layer_idx in test_layers:
    block = (layer_idx, layer_idx + 1)
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    gate_overlaps = []
    module = original_layers[layer_idx]

    for prompt in prompts:
        gate_acts = []
        counter = [0]

        def make_gate_hook(ctr, acts):
            def hook(module, input, output):
                ctr[0] += 1
                # Capture gate activation pattern (which neurons are active)
                with torch.no_grad():
                    if hasattr(module, 'gate_proj'):
                        gate_in = input[0] if isinstance(input, tuple) else input
                        gate_out = module.gate_proj(gate_in)
                        active = (gate_out > 0).float().cpu()
                        acts.append(active)
                return output
            return hook

        h = module.mlp.register_forward_hook(make_gate_hook(counter, gate_acts))

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(inputs['input_ids'], use_cache=False)

        h.remove()

        # gate_acts should have 2 entries (first and second pass)
        if len(gate_acts) >= 2:
            a1 = gate_acts[0].flatten()
            a2 = gate_acts[1].flatten()
            intersection = (a1 * a2).sum().item()
            union = ((a1 + a2) > 0).float().sum().item()
            jaccard = intersection / union if union > 0 else 1.0
            gate_overlaps.append(jaccard)

    restore()

    mean_jaccard = float(np.mean(gate_overlaps)) if gate_overlaps else 0.0
    jaccard_results[layer_idx] = {
        'mean_jaccard': mean_jaccard,
        'per_prompt': gate_overlaps,
        'n_prompts': len(gate_overlaps),
    }
    stability = 'stable' if mean_jaccard > 0.6 else ('moderate' if mean_jaccard > 0.4 else 'unstable')
    print(f'  Layer {layer_idx:3d}: Jaccard={mean_jaccard:.4f} ({stability})', flush=True)

# ======================================================================
# 3. FFN DANGER SCORES
# ======================================================================
print('\\n=== 3. FFN Danger Scores (per-layer in best triple) ===', flush=True)

# For each layer in the triple, measure: how much does FFN hurt on second pass?
# Compare full duplication vs attention-only for single-layer blocks
ffn_danger = {}
for layer_idx in test_layers:
    block = [(layer_idx, layer_idx + 1)]

    # Full duplication
    full_r = evaluate(block, f'full L{layer_idx}')

    # Attention-only
    order = apply_blocks(block)
    hooks = []
    module = original_layers[layer_idx]
    counter = [0]
    def make_zero_hook(ctr):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return output
        return hook
    h = module.mlp.register_forward_hook(make_zero_hook(counter))
    hooks.append(h)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    attn_combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    for hh in hooks:
        hh.remove()
    restore()

    ffn_harm = full_r['combined'] - attn_combined  # negative = FFN helps, positive = FFN hurts
    danger = abs(ffn_harm) * (1.0 - jaccard_results.get(layer_idx, {}).get('mean_jaccard', 0.5))

    ffn_danger[layer_idx] = {
        'full_combined': full_r['combined'],
        'attn_only_combined': attn_combined,
        'ffn_harm': ffn_harm,
        'jaccard': jaccard_results.get(layer_idx, {}).get('mean_jaccard', 0),
        'danger_score': danger,
    }
    print(f'  Layer {layer_idx:3d}: full={full_r[\"combined\"]:.2f} attn_only={attn_combined:.2f} ffn_harm={ffn_harm:+.2f} danger={danger:.2f}', flush=True)

# ======================================================================
# 4. INFERENCE SPEED BENCHMARK
# ======================================================================
print('\\n=== 4. Inference Speed Benchmark ===', flush=True)

speed_prompts = [
    'Explain quantum entanglement in simple terms.',
    'Write a Python function that finds all prime numbers up to N.',
    'What are the main differences between capitalism and socialism?',
]

configs_speed = [
    ('baseline', []),
    ('single (12,13)', [(12, 13)]),
    ('single (47,48)', [(47, 48)]),
    ('triple (0,2)+(12,13)+(47,48)', [(0, 2), (12, 13), (47, 48)]),
]

speed_results = []
for name, blocks in configs_speed:
    if blocks:
        apply_blocks(blocks)

    times = []
    tokens_total = 0
    for prompt in speed_prompts:
        t0 = time.time()
        output = generate_no_cache(model, tokenizer, prompt, max_new_tokens=128)
        elapsed = time.time() - t0
        n_tokens = len(tokenizer.encode(output))
        times.append(elapsed)
        tokens_total += n_tokens

    if blocks:
        restore()

    avg_time = np.mean(times)
    tokens_per_sec = tokens_total / sum(times) if sum(times) > 0 else 0
    n_extra = sum(j - i for i, j in blocks)
    print(f'  {name:45s}: avg={avg_time:.1f}s  tok/s={tokens_per_sec:.1f}  +{n_extra} layers', flush=True)

    speed_results.append({
        'name': name, 'blocks': [list(b) for b in blocks],
        'avg_time_s': float(avg_time),
        'tokens_per_sec': float(tokens_per_sec),
        'extra_layers': n_extra,
    })

# Compute slowdown
if speed_results:
    base_tps = speed_results[0]['tokens_per_sec']
    for r in speed_results:
        r['slowdown_pct'] = ((base_tps - r['tokens_per_sec']) / base_tps * 100) if base_tps > 0 else 0
        print(f'    {r[\"name\"]}: {r[\"slowdown_pct\"]:.1f}% slower', flush=True)

# ======================================================================
# 5. FULL EQ-BENCH (171 questions) on best configs
# ======================================================================
print('\\n=== 5. Full EQ-Bench (all 20 probe questions) on key configs ===', flush=True)

full_eq_configs = [
    ('baseline', []),
    ('best_single (12,13)', [(12, 13)]),
    ('best_triple (0,2)+(12,13)+(47,48)', [(0, 2), (12, 13), (47, 48)]),
]

full_eq_results = []
for name, blocks in full_eq_configs:
    r = evaluate(blocks, name, full_eq=True)
    full_eq_results.append(r)

# ======================================================================
# 6. DEEPER STACKING (5-6 blocks with whisper alpha)
# ======================================================================
print('\\n=== 6. Deep Stacking (5-6 blocks with whisper alpha) ===', flush=True)

# Start from best triple, add blocks with whisper alpha hooks
anchor = [(0, 2), (12, 13), (47, 48)]
# Promising 4th/5th block candidates (from various positions)
extra_candidates = [
    (6, 7), (8, 9), (20, 21), (25, 26), (30, 31), (35, 36), (40, 41), (54, 55),
]

print('Testing 5-block configs (anchor + 2 extras @ alpha=0.15)...', flush=True)
from itertools import combinations

deep_results = []
for extras in combinations(extra_candidates, 2):
    all_blocks = list(anchor) + list(extras)
    # Check no overlaps
    sorted_b = sorted(all_blocks)
    overlap = False
    for k in range(len(sorted_b) - 1):
        if sorted_b[k][1] > sorted_b[k+1][0]:
            overlap = True
            break
    if overlap:
        continue

    order = build_order(all_blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    # Apply whisper alpha hooks on the extra blocks only
    hooks = []
    for block in extras:
        for l_idx in range(block[0], block[1]):
            module = original_layers[l_idx]
            counter = [0]
            def make_whisper_hook(ctr, alpha=0.15):
                def hook(module, input, output):
                    ctr[0] += 1
                    if ctr[0] % 2 == 0:
                        h_in = input[0]
                        if isinstance(output, tuple):
                            h_out = output[0]
                            blended = h_in + alpha * (h_out - h_in)
                            return (blended,) + output[1:]
                        return h_in + alpha * (output - h_in)
                    return output
                return hook
            h = module.register_forward_hook(make_whisper_hook(counter))
            hooks.append(h)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0

    for h in hooks:
        h.remove()
    restore()

    name = '+'.join(f'({i},{j})' for i, j in sorted(all_blocks))
    n_extra = sum(j - i for i, j in all_blocks)
    print(f'  {name:55s}: combined={combined:.2f} +{n_extra}layers ({elapsed:.0f}s)', flush=True)
    deep_results.append({
        'blocks': [list(b) for b in sorted(all_blocks)],
        'combined': combined, 'math': math_r['score'], 'eq': eq_r['score'],
    })

deep_results.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\nTop 5 deep stacking:', flush=True)
for r in deep_results[:5]:
    b_str = '+'.join(f'({b[0]},{b[1]})' for b in r['blocks'])
    print(f'  {b_str}: combined={r[\"combined\"]:.2f}', flush=True)

# ======================================================================
# SAVE ALL RESULTS
# ======================================================================
print('\\n=== Saving Results ===', flush=True)

all_results = {
    'date': datetime.now().isoformat(),
    'model': 'gemma-3-27b-it',
    'num_layers': N,
    'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline},
    'attn_only_vs_full': attn_only_results,
    'jaccard_instability': {str(k): v for k, v in jaccard_results.items()},
    'ffn_danger_scores': {str(k): v for k, v in ffn_danger.items()},
    'inference_speed': speed_results,
    'full_eq_bench': full_eq_results,
    'deep_stacking': {
        'anchor': [list(b) for b in anchor],
        'whisper_alpha': 0.15,
        'top_10': deep_results[:10],
    },
}

with open(f'{SAVE_DIR}/comprehensive_analysis.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved to {SAVE_DIR}/comprehensive_analysis.json', flush=True)

# Summary
print(f'\\n{\"=\" * 70}', flush=True)
print('SUMMARY', flush=True)
print(f'{\"=\" * 70}', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Attn-only analysis: {len(attn_only_results)} configs tested', flush=True)
for r in attn_only_results:
    b = '+'.join(f'({b[0]},{b[1]})' for b in r['blocks'])
    print(f'  {b}: full={r[\"full\"][\"combined\"]:.2f} attn_only={r[\"attn_only\"][\"combined\"]:.2f} ffn_impact={r[\"ffn_impact\"]:+.2f}', flush=True)
print(f'Jaccard instability:', flush=True)
for l_idx, data in sorted(jaccard_results.items()):
    print(f'  Layer {l_idx}: {data[\"mean_jaccard\"]:.4f}', flush=True)
print(f'FFN danger:', flush=True)
for l_idx, data in sorted(ffn_danger.items()):
    print(f'  Layer {l_idx}: harm={data[\"ffn_harm\"]:+.2f} danger={data[\"danger_score\"]:.2f}', flush=True)
print(f'Speed: triple is {speed_results[-1][\"slowdown_pct\"]:.1f}% slower than baseline', flush=True)
print(f'Deep stacking best: {deep_results[0][\"combined\"]:.2f}' if deep_results else 'No deep results', flush=True)
"

echo "=== Finished: $(date) ==="
