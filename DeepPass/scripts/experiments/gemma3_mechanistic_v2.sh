#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_mech_v2_%j.log
#SBATCH --job-name=g3_mech2

# Mechanistic analysis v2 — saves checkpoints after each section
# 1. Baseline + attn-only vs full (4 configs)
# 2. Jaccard instability
# 3. FFN danger scores
# 4. Speed benchmark
# 5. Full-probe validation of best configs

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3-27B Mechanistic Analysis v2 ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/mechanistic'
os.makedirs(SAVE_DIR, exist_ok=True)

def save_checkpoint(data, name):
    with open(f'{SAVE_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  [CHECKPOINT] Saved {name}.json', flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

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

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# ======================================================================
# 0. Baseline
# ======================================================================
print('\\n=== 0. Baseline ===', flush=True)
t0 = time.time()
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)
save_checkpoint({'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline}, 'baseline')

# ======================================================================
# 1. ATTENTION-ONLY vs FULL — single-layer blocks from best triple
# ======================================================================
print('\\n=== 1. Attention-Only vs Full Duplication ===', flush=True)
test_layers = [0, 1, 12, 47]
attn_results = []

for layer_idx in test_layers:
    block = [(layer_idx, layer_idx + 1)]

    # Full duplication
    apply_blocks(block)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    full_combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    restore()
    print(f'  L{layer_idx} full: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={full_combined:.2f} ({time.time()-t0:.0f}s)', flush=True)

    # Attention-only: zero FFN on second pass
    apply_blocks(block)
    counter = [0]
    module = original_layers[layer_idx]
    def make_zero_ffn(ctr):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            return output
        return hook
    h = module.mlp.register_forward_hook(make_zero_ffn(counter))
    t0 = time.time()
    math_r2 = run_math_probe(gen, verbose=False)
    eq_r2 = run_eq_bench_probe(gen_long, verbose=False)
    attn_combined = math_r2['score'] * 50 + eq_r2['score'] * 0.5
    h.remove()
    restore()

    ffn_impact = attn_combined - full_combined
    print(f'  L{layer_idx} attn-only: combined={attn_combined:.2f} ffn_impact={ffn_impact:+.2f} ({time.time()-t0:.0f}s)', flush=True)

    attn_results.append({
        'layer': layer_idx,
        'full': {'math': math_r['score'], 'eq': eq_r['score'], 'combined': full_combined},
        'attn_only': {'math': math_r2['score'], 'eq': eq_r2['score'], 'combined': attn_combined},
        'ffn_impact': ffn_impact,
    })

save_checkpoint(attn_results, 'attn_only_vs_full')

# ======================================================================
# 2. JACCARD INSTABILITY
# ======================================================================
print('\\n=== 2. Jaccard Instability ===', flush=True)
prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'Explain the concept of entropy.', 'How would a parent feel seeing their child graduate?',
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

        if len(gate_acts) >= 2:
            a1 = gate_acts[0].flatten()
            a2 = gate_acts[1].flatten()
            intersection = (a1 * a2).sum().item()
            union = ((a1 + a2) > 0).float().sum().item()
            jaccard = intersection / union if union > 0 else 1.0
            gate_overlaps.append(jaccard)

    restore()
    mean_j = float(np.mean(gate_overlaps)) if gate_overlaps else 0.0
    jaccard_results[str(layer_idx)] = {'mean_jaccard': mean_j, 'per_prompt': gate_overlaps}
    stability = 'stable' if mean_j > 0.6 else ('moderate' if mean_j > 0.4 else 'unstable')
    print(f'  Layer {layer_idx}: Jaccard={mean_j:.4f} ({stability})', flush=True)

save_checkpoint(jaccard_results, 'jaccard_instability')

# ======================================================================
# 3. FFN DANGER SCORES
# ======================================================================
print('\\n=== 3. FFN Danger Scores ===', flush=True)
ffn_danger = {}
for r in attn_results:
    layer_idx = r['layer']
    j = jaccard_results.get(str(layer_idx), {}).get('mean_jaccard', 0.5)
    danger = abs(r['ffn_impact']) * (1.0 - j)
    ffn_danger[str(layer_idx)] = {
        'ffn_impact': r['ffn_impact'],
        'jaccard': j,
        'danger_score': danger,
    }
    print(f'  Layer {layer_idx}: ffn_impact={r[\"ffn_impact\"]:+.2f} jaccard={j:.4f} danger={danger:.2f}', flush=True)

save_checkpoint(ffn_danger, 'ffn_danger_scores')

# ======================================================================
# 4. SPEED BENCHMARK
# ======================================================================
print('\\n=== 4. Speed Benchmark ===', flush=True)
speed_prompts = [
    'Explain quantum entanglement in simple terms.',
    'Write a Python function that finds all prime numbers up to N.',
    'What are the main differences between capitalism and socialism?',
]
configs_speed = [
    ('baseline', []),
    ('single (12,13)', [(12, 13)]),
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
    avg_time = float(np.mean(times))
    tps = tokens_total / sum(times) if sum(times) > 0 else 0
    print(f'  {name:45s}: avg={avg_time:.1f}s tok/s={tps:.1f}', flush=True)
    speed_results.append({'name': name, 'avg_time_s': avg_time, 'tokens_per_sec': float(tps), 'extra_layers': sum(j-i for i,j in blocks)})

if speed_results:
    base_tps = speed_results[0]['tokens_per_sec']
    for r in speed_results:
        r['slowdown_pct'] = float((base_tps - r['tokens_per_sec']) / base_tps * 100) if base_tps > 0 else 0

save_checkpoint(speed_results, 'speed_benchmark')

# ======================================================================
# 5. FULL VALIDATION of best configs (full EQ-bench)
# ======================================================================
print('\\n=== 5. Full-Probe Validation ===', flush=True)
val_configs = [
    ('baseline', []),
    ('best_single_12_13', [(12, 13)]),
    ('best_triple', [(0, 2), (12, 13), (47, 48)]),
]
val_results = []
for name, blocks in val_configs:
    if blocks:
        apply_blocks(blocks)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    if blocks:
        restore()
    print(f'  {name:30s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    val_results.append({'name': name, 'blocks': [list(b) for b in blocks], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

save_checkpoint(val_results, 'full_validation')

# Final summary
print(f'\\n{\"=\" * 60}', flush=True)
print('COMPLETE — All checkpoints saved', flush=True)
print(f'{\"=\" * 60}', flush=True)
"

echo "=== Finished: $(date) ==="
