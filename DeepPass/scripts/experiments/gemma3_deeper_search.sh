#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_deep_%j.log
#SBATCH --job-name=deeppass_g3deep

# Gemma3 deeper search: larger block sizes + quantization test
# Previous search only found 1-layer blocks. 72B's best blocks are 7-10 layers.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3 Deeper Search + Quantization Test ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, gc, torch, torch.nn as nn, numpy as np
from datetime import datetime
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

# =========================================================================
# PART 1: Larger block search in bfloat16
# =========================================================================
print('=' * 70)
print('PART 1: LARGER BLOCK SEARCH (bfloat16)')
print('=' * 70, flush=True)

model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    else:
        model.config.num_hidden_layers = n

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
]

def compute_rho(block):
    i, j = block
    rhos = []
    for prompt in cal_prompts:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()
            order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))
            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()
            inner.layers = nn.ModuleList(original_layers)
            set_num_layers(N)
            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)
    return float(np.mean(rhos)) if rhos else 1.0

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Screen LARGER blocks specifically (5, 7, 10, 14 layers), step=2
print('\\n--- Screening larger blocks (sizes 5,7,10,14, step=2) ---', flush=True)
candidates = []
for start in range(0, N-1, 2):
    for size in [5, 7, 10, 14]:
        end = start + size
        if end <= N:
            candidates.append((start, end))

print(f'Screening {len(candidates)} large blocks...', flush=True)
block_rhos = {}
for idx, block in enumerate(candidates):
    block_rhos[block] = compute_rho(block)
    if (idx + 1) % 5 == 0:
        print(f'  [{idx+1}/{len(candidates)}] ({block[0]:2d},{block[1]:2d}) size={block[1]-block[0]} rho={block_rhos[block]:.4f}', flush=True)

sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
print('\\nTop 12 large blocks:')
for b, r in sorted_blocks[:12]:
    print(f'  ({b[0]:2d},{b[1]:2d}) size={b[1]-b[0]} rho={r:.4f}', flush=True)

# Baseline
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
combined_base = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'\\nBaseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}', flush=True)

# Evaluate top-10 large singles
print('\\n--- Evaluating top-10 large singles ---', flush=True)
single_results = {}
top_10 = [b for b, _ in sorted_blocks[:10]]
for block in top_10:
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - combined_base
    print(f'  ({i:2d},{j:2d}) size={j-i}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}', flush=True)
    single_results[block] = {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta': delta}
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

# Find best large single
best_large = max(single_results.items(), key=lambda x: x[1]['combined'])
print(f'\\nBest large block: ({best_large[0][0]},{best_large[0][1]}) combined={best_large[1][\"combined\"]:.2f}', flush=True)

# Previous best (small blocks)
prev_best_single_combined = 83.76  # (20,21)
prev_best_pair_combined = 84.42    # (4,5)+(20,21)

# Greedy stacking from best large single
print('\\n--- Greedy stacking from best large single ---', flush=True)
best_block = best_large[0]
best_score = best_large[1]['combined']

# Screen for second block on modified model
order_a = build_order([best_block], N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
set_num_layers(len(order_a))

second_candidates = [b for b in top_10 if b[1] <= best_block[0] or b[0] >= best_block[1]]
# Also add the previous best small blocks
for sb in [(4,5), (8,9), (12,13), (16,17), (20,21), (24,25)]:
    if sb[1] <= best_block[0] or sb[0] >= best_block[1]:
        if sb not in second_candidates:
            second_candidates.append(sb)

second_rhos = {}
for block in second_candidates:
    second_rhos[block] = compute_rho(block)

inner.layers = nn.ModuleList(original_layers)
set_num_layers(N)

sorted_second = sorted(second_rhos.items(), key=lambda x: x[1])
print(f'Top 5 second blocks:')
for b, r in sorted_second[:5]:
    print(f'  ({b[0]:2d},{b[1]:2d}) rho={r:.4f}', flush=True)

# Evaluate top-5 pairs
pair_results = []
for block_b, _ in sorted_second[:5]:
    pair = sorted([best_block, block_b])
    name = '+'.join(f'({b[0]},{b[1]})' for b in pair)
    order = build_order(pair, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta_vs_single = combined - best_score
    print(f'  {name}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} vs single: {delta_vs_single:+.2f}', flush=True)
    pair_results.append({'name': name, 'blocks': [list(b) for b in pair], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

# Also try cross-mixing: best large + best small blocks
print('\\n--- Cross-mixing: large + small blocks ---', flush=True)
cross_pairs = []
for small in [(4,5), (8,9), (12,13), (16,17), (20,21), (24,25)]:
    if small[1] <= best_block[0] or small[0] >= best_block[1]:
        pair = sorted([best_block, small])
        name = '+'.join(f'({b[0]},{b[1]})' for b in pair)
        # Skip if already evaluated
        if any(p['name'] == name for p in pair_results):
            continue
        order = build_order(pair, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        set_num_layers(len(order))
        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        print(f'  {name}: combined={combined:.2f}', flush=True)
        cross_pairs.append({'name': name, 'blocks': [list(b) for b in pair], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})
        inner.layers = nn.ModuleList(original_layers)
        set_num_layers(N)

all_pairs = pair_results + cross_pairs
best_pair = max(all_pairs, key=lambda x: x['combined'])

print(f'\\n=== PART 1 SUMMARY ===')
print(f'Baseline: {combined_base:.2f}')
print(f'Previous best single (20,21): {prev_best_single_combined:.2f}')
print(f'Previous best pair (4,5)+(20,21): {prev_best_pair_combined:.2f}')
print(f'Best large single: ({best_large[0][0]},{best_large[0][1]}) = {best_large[1][\"combined\"]:.2f}')
print(f'Best new pair: {best_pair[\"name\"]} = {best_pair[\"combined\"]:.2f}', flush=True)

# Save part 1
part1 = {
    'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': combined_base},
    'large_spectral': [{'block': list(b), 'rho': r} for b, r in sorted_blocks],
    'large_singles': {f'({k[0]},{k[1]})': v for k, v in single_results.items()},
    'pairs': all_pairs,
    'best_large_single': {'block': list(best_large[0]), **best_large[1]},
    'best_pair': best_pair,
}

# Cleanup bf16 model
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# =========================================================================
# PART 2: Quantized (4-bit NF4) test
# =========================================================================
print(f'\\n{\"=\" * 70}')
print('PART 2: QUANTIZED (4-bit NF4) TEST')
print('Does duplication benefit survive quantization?')
print('=' * 70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print('Loading Gemma3 in 4-bit NF4...', flush=True)
tokenizer = AutoTokenizer.from_pretrained('models/full/gemma-3-27b-it', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'models/full/gemma-3-27b-it',
    quantization_config=quant_config,
    device_map='auto',
    trust_remote_code=True,
)

inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded 4-bit: {N} layers, ~{torch.cuda.memory_allocated()/1e9:.1f} GB VRAM', flush=True)

def set_num_layers_q(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    else:
        model.config.num_hidden_layers = n

def gen_q(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_q_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Baseline (quantized)
print('\\n--- Quantized baseline ---', flush=True)
math_r = run_math_probe(gen_q, verbose=False)
eq_r = run_eq_bench_probe(gen_q_long, verbose=False)
q_baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'4-bit baseline: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={q_baseline:.2f}', flush=True)

# Test configs: best small single, best small pair, best large single, best large pair
configs_to_test = [
    ('(20,21)', [(20, 21)]),
    ('(4,5)+(20,21)', [(4, 5), (20, 21)]),
    (f'({best_large[0][0]},{best_large[0][1]})', [best_large[0]]),
]
if best_pair:
    bp_blocks = [tuple(b) for b in best_pair['blocks']]
    configs_to_test.append((best_pair['name'], bp_blocks))

q_results = [{'name': 'baseline', 'math': math_r['score'], 'eq': eq_r['score'], 'combined': q_baseline}]

for name, blocks in configs_to_test:
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers_q(len(order))
    math_r = run_math_probe(gen_q, verbose=False)
    eq_r = run_eq_bench_probe(gen_q_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - q_baseline
    print(f'4-bit {name}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}', flush=True)
    q_results.append({'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta': delta})
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers_q(N)

print(f'\\n=== PART 2 SUMMARY ===')
print(f'VRAM used: ~{torch.cuda.memory_allocated()/1e9:.1f} GB')
print(f'bf16 baseline: {combined_base:.2f} vs 4-bit baseline: {q_baseline:.2f} (degradation: {q_baseline-combined_base:+.2f})')
for r in q_results[1:]:
    bf16_equiv = part1.get('large_singles', {}).get(r['name'], {}).get('combined', 'N/A')
    print(f'{r[\"name\"]:30s}: 4-bit={r[\"combined\"]:.2f} delta={r[\"delta\"]:+.2f}')
print(f'Duplication benefit survives quantization: {q_results[1][\"delta\"] > 0}', flush=True)

# Save everything
output = {
    'date': datetime.now().isoformat(),
    'part1_larger_blocks': part1,
    'part2_quantized': {
        'vram_gb': torch.cuda.memory_allocated()/1e9,
        'results': q_results,
        'bf16_baseline': combined_base,
        'q4_baseline': q_baseline,
    },
}
os.makedirs('results/data/gemma3_27b', exist_ok=True)
with open('results/data/gemma3_27b/deeper_search_and_quant.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\\nSaved to results/data/gemma3_27b/deeper_search_and_quant.json', flush=True)
"

echo "=== Done at $(date) ==="
