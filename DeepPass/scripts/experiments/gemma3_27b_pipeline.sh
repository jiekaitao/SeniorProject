#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_27b_%j.log
#SBATCH --job-name=deeppass_gemma3

# Full pipeline on Gemma-3-27B:
# 1. Spectral screen all blocks
# 2. Test top-8 singles with dual probe
# 3. Greedy stacking — find best pair
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma-3-27B Full Pipeline ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn, numpy as np
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
inner = model.model if hasattr(model, 'model') else model.transformer
# Gemma3 nests the text model: model.model.language_model.layers
if hasattr(inner, 'language_model'):
    inner = inner.language_model
attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
original_layers = list(getattr(inner, attr_name))
N = len(original_layers)
device = next(model.parameters()).device

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    else:
        model.config.num_hidden_layers = n
print(f'Loaded Gemma-3-27B: {N} layers')

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

def compute_rho(block):
    \"\"\"Architecture-agnostic rho via full model forward passes.
    Computes displacement = ||logits_dup - logits_base|| / ||logits_base||
    by running the full model with and without block duplication.
    Works across Qwen, Gemma, LLaMA, etc. without touching rotary_emb internals.
    \"\"\"
    i, j = block
    rhos = []
    for prompt in cal_prompts[:4]:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            # 1. Base model forward
            out_base = model(**inputs, use_cache=False)
            logits_base = out_base.logits.detach()

            # 2. Duplicate block [i,j) and forward
            layer_order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            dup_layers = [original_layers[k] for k in layer_order]
            setattr(inner, attr_name, nn.ModuleList(dup_layers))
            set_num_layers(len(dup_layers))

            out_dup = model(**inputs, use_cache=False)
            logits_dup = out_dup.logits.detach()

            # 3. Restore original layers
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            set_num_layers(N)

            # 4. Compute displacement rho
            diff_norm = (logits_dup - logits_base).float().norm(dim=-1).mean().item()
            base_norm = logits_base.float().norm(dim=-1).mean().item()
            if base_norm > 1e-8:
                rhos.append(diff_norm / base_norm)
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

# ==========================================
# Step 1: Spectral screen
# ==========================================
print('\\n=== Step 1: Spectral Screen ===')
candidates = []
for start in range(0, N-1, 4):
    for size in [1, 3, 5, 7]:
        end = start + size
        if end <= N:
            candidates.append((start, end))

print(f'Screening {len(candidates)} blocks...', flush=True)
import sys
block_rhos = {}
for idx, block in enumerate(candidates):
    block_rhos[block] = compute_rho(block)
    if (idx + 1) % 5 == 0:
        print(f'  [{idx+1}/{len(candidates)}] ({block[0]:2d},{block[1]:2d}): rho={block_rhos[block]:.4f}', flush=True)

sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
print('Top 15:')
for b, r in sorted_blocks[:15]:
    print(f'  ({b[0]:2d},{b[1]:2d}): rho={r:.4f}')

# ==========================================
# Step 2: Baseline + top-8 singles
# ==========================================
print('\\n=== Step 2: Baseline + Top 8 Singles ===')
math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
combined_base = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}')

results = {'baseline': {'math': math_base['score'], 'eq': eq_base['score'], 'combined': combined_base}}
single_scores = {}

top_8 = [b for b, r in sorted_blocks[:8]]
for block in top_8:
    i, j = block
    order = build_order([block], N)
    setattr(inner, attr_name, nn.ModuleList([original_layers[idx] for idx in order]))
    set_num_layers(len(order))

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - combined_base

    print(f'  ({i:2d},{j:2d}): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}')
    single_scores[block] = combined
    results[f'({i},{j})'] = {'block': list(block), 'rho': block_rhos[block], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta': delta}

    setattr(inner, attr_name, nn.ModuleList(original_layers))
    set_num_layers(N)

# ==========================================
# Step 3: Greedy stacking from best single
# ==========================================
print('\\n=== Step 3: Greedy Stacking ===')
best_single = max(single_scores, key=single_scores.get)
best_score = single_scores[best_single]
print(f'Best single: ({best_single[0]},{best_single[1]}) combined={best_score:.2f}')

# Apply best single, screen for second block
order_a = build_order([best_single], N)
setattr(inner, attr_name, nn.ModuleList([original_layers[idx] for idx in order_a]))
set_num_layers(len(order_a))

# Screen top-10 candidates for second block
second_candidates = [b for b in [x[0] for x in sorted_blocks[:15]]
                     if b[1] <= best_single[0] or b[0] >= best_single[1]]

print(f'Screening {len(second_candidates)} second-block candidates on modified model...')
second_rhos = {}
for block in second_candidates:
    second_rhos[block] = compute_rho(block)

setattr(inner, attr_name, nn.ModuleList(original_layers))
set_num_layers(N)

sorted_second = sorted(second_rhos.items(), key=lambda x: x[1])
print('Top 5 second blocks:')
for b, r in sorted_second[:5]:
    print(f'  ({b[0]:2d},{b[1]:2d}): rho={r:.4f}')

# Evaluate top-5 pairs
pair_results = []
for block_b, _ in sorted_second[:5]:
    pair = sorted([best_single, block_b])
    name = '+'.join(f'({b[0]},{b[1]})' for b in pair)

    order = build_order(pair, N)
    setattr(inner, attr_name, nn.ModuleList([original_layers[idx] for idx in order]))
    set_num_layers(len(order))

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - best_score

    print(f'  {name}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} vs best single: {delta:+.2f}')
    pair_results.append({'blocks': [list(b) for b in pair], 'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta_vs_best_single': delta})

    setattr(inner, attr_name, nn.ModuleList(original_layers))
    set_num_layers(N)

results['pairs'] = pair_results
results['spectral'] = [{'block': list(b), 'rho': r} for b, r in sorted_blocks]

# Summary
print('\\n=== SUMMARY ===')
print(f'Model: Gemma-3-27B ({N} layers)')
print(f'Baseline: combined={combined_base:.2f}')
print(f'Best single: ({best_single[0]},{best_single[1]}) combined={best_score:.2f} delta={best_score-combined_base:+.2f}')
best_pair = max(pair_results, key=lambda x: x['combined']) if pair_results else None
if best_pair:
    print(f'Best pair: {best_pair[\"name\"]} combined={best_pair[\"combined\"]:.2f} delta={best_pair[\"combined\"]-combined_base:+.2f}')
    if best_pair['combined'] > best_score:
        print('*** PAIR BEATS SINGLE! Stacking works on Gemma-3-27B! ***')
    else:
        print('Pair does not beat single.')

os.makedirs('results/data/gemma3_27b', exist_ok=True)
with open('results/data/gemma3_27b/full_pipeline.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to results/data/gemma3_27b/full_pipeline.json')
"

echo "=== Done at $(date) ==="
