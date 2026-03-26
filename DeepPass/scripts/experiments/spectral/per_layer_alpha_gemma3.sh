#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_pla_gemma3_%j.log
#SBATCH --job-name=deeppass_plag

# Per-block alpha tuning + whisper triples on Gemma3-27B (62 layers)
#
# Known results:
#   Baseline:                         80.54
#   Best single (20,21):              83.76
#   Best pair (4,5)+(20,21):          84.42
#   Best triple (4,5)+(12,13)+(20,21): 85.43
#
# Strategy:
#   1. Alpha-tuned pairs: (4,5)+(20,21) with different alpha combos
#   2. More triple candidates: (4,5)+(X)+(20,21) for various X
#   3. Sweep 3rd block alpha on best triple
#   4. Quad: add a 4th block at low alpha
#
# Gemma3's sliding window attention breaks with manual layer loops,
# so we use ModuleList swapping for alpha=1.0 tests. For per-block alpha
# we attempt the manual approach first; if it fails, fall back to
# alpha=1.0 ModuleList-only testing.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Per-Block Alpha Tuning on Gemma3-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'

print('=' * 70)
print('PER-BLOCK ALPHA TUNING + WHISPER TRIPLES ON GEMMA3-27B')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
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
    elif hasattr(model.config, 'num_hidden_layers'):
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

def find_seams(layer_order, blocks):
    \"\"\"For each block, find the step indices of the 1st and 2nd occurrence
    of the last layer in the block (i.e., where 1st pass ends and 2nd pass ends).\"\"\"
    seams = []
    for block in sorted(blocks):
        last_layer = block[1] - 1
        occurrences = [step for step, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) >= 2:
            seams.append((occurrences[0], occurrences[1]))
        else:
            seams.append(None)
    return seams

# =====================================================================
# Generation: full-model forward with ModuleList swapping (alpha=1.0)
# =====================================================================
def generate_full_model(prompt, blocks, max_new_tokens=64):
    \"\"\"Duplication via ModuleList swapping. No per-block alpha (all at 1.0).\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
            logits = out.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

# =====================================================================
# Generation: manual layer loop with per-block alpha (may fail on Gemma3)
# =====================================================================
def generate_multi_alpha(prompt, blocks, alphas, max_new_tokens=64):
    \"\"\"Manual layer loop with per-block alpha weighting at seams.
    For Gemma3 we try position_ids + cache_position calling convention.\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    block_to_alpha = {b: a for b, a in zip(blocks, alphas)}
    layer_order = build_order(sorted_blocks, N)
    seams = find_seams(layer_order, sorted_blocks)
    sorted_alphas = [block_to_alpha[b] for b in sorted_blocks]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            cache_position = torch.arange(seq_len, device=device)

            saved_h1 = {}
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                try:
                    out = layer(h, position_ids=pos_ids, cache_position=cache_position, use_cache=False)
                except TypeError:
                    try:
                        out = layer(h, position_ids=pos_ids, use_cache=False)
                    except TypeError:
                        out = layer(h, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                for si, seam in enumerate(seams):
                    if seam is None:
                        continue
                    if step_idx == seam[0]:
                        saved_h1[si] = h.clone()
                    if step_idx == seam[1] and si in saved_h1:
                        h = saved_h1[si] + sorted_alphas[si] * (h - saved_h1[si])
                        del saved_h1[si]

            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

# =====================================================================
# Evaluation helpers
# =====================================================================
def evaluate_full(blocks, name):
    gen = lambda p: generate_full_model(p, blocks, max_new_tokens=64)
    gen_long = lambda p: generate_full_model(p, blocks, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks],
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

def evaluate_alpha(blocks, alphas, name):
    gen = lambda p: generate_multi_alpha(p, blocks, alphas, max_new_tokens=64)
    gen_long = lambda p: generate_multi_alpha(p, blocks, alphas, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alphas': list(alphas),
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# =====================================================================
# BASELINE
# =====================================================================
print(f'\\n--- Baseline ---', flush=True)
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

BEST_SINGLE = (20, 21)
BEST_PAIR = [(4, 5), (20, 21)]
BEST_TRIPLE = [(4, 5), (12, 13), (20, 21)]

# =====================================================================
# TEST: does manual layer loop work on Gemma3?
# =====================================================================
print(f'\\n--- Testing manual layer loop on Gemma3 ---', flush=True)
manual_works = False
try:
    test_r = evaluate_alpha([BEST_SINGLE], [1.0], 'test: manual (20,21)@1.0')
    manual_works = True
    print('  Manual layer loop WORKS on Gemma3!', flush=True)
except Exception as e:
    print(f'  Manual layer loop FAILED: {e}', flush=True)
    print('  Will use full-model forward for alpha=1.0 tests only.', flush=True)

# =====================================================================
# PHASE 1: Alpha-tuned pairs — (4,5)+(20,21)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PHASE 1: Alpha-tuned pair (4,5)+(20,21)')
print(f'{\"=\" * 70}', flush=True)

if manual_works:
    # Test alpha combinations on the known best pair
    pair_alpha_combos = [
        ([1.0, 1.0], 'pair @1.0/1.0'),
        ([0.9, 1.0], 'pair @0.9/1.0'),
        ([1.0, 1.15], 'pair @1.0/1.15'),
        ([0.9, 1.15], 'pair @0.9/1.15'),
        ([0.9, 0.9], 'pair @0.9/0.9'),
        ([1.1, 1.0], 'pair @1.1/1.0'),
        ([1.0, 1.1], 'pair @1.0/1.1'),
        ([1.1, 1.1], 'pair @1.1/1.1'),
        ([0.8, 1.15], 'pair @0.8/1.15'),
        ([1.15, 1.15], 'pair @1.15/1.15'),
        ([0.9, 1.2], 'pair @0.9/1.2'),
        ([1.0, 1.25], 'pair @1.0/1.25'),
    ]
    for alphas, name in pair_alpha_combos:
        r = evaluate_alpha(BEST_PAIR, alphas, name)
        all_results.append(r)

    # Find best pair alpha combo
    best_pair_result = max(all_results, key=lambda x: x['combined'])
    print(f'\\n  Best pair alpha: {best_pair_result[\"name\"]} = {best_pair_result[\"combined\"]:.2f}', flush=True)
else:
    # Full-model forward: pair at alpha=1.0 only
    r = evaluate_full(BEST_PAIR, 'pair (4,5)+(20,21) @1.0')
    all_results.append(r)
    print(f'  Pair @1.0: {r[\"combined\"]:.2f}', flush=True)

# =====================================================================
# PHASE 2: Triple candidates — (4,5)+(X)+(20,21) at alpha=1.0
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PHASE 2: Triple candidates (4,5)+(X)+(20,21)')
print(f'{\"=\" * 70}', flush=True)

THIRD_CANDIDATES = [(8, 9), (12, 13), (14, 15), (16, 17), (24, 25)]

for third in THIRD_CANDIDATES:
    # Check no overlap with existing blocks
    triple = list(BEST_PAIR) + [third]
    overlap = any(not (third[1] <= b[0] or third[0] >= b[1]) for b in BEST_PAIR)
    if overlap:
        print(f'  Skipping ({third[0]},{third[1]}): overlaps with pair', flush=True)
        continue

    if manual_works:
        r = evaluate_alpha(triple, [1.0, 1.0, 1.0], f'triple +({third[0]},{third[1]}) @1.0/1.0/1.0')
    else:
        r = evaluate_full(triple, f'triple +({third[0]},{third[1]}) @1.0')
    all_results.append(r)

# Find best triple candidate
triple_results = [r for r in all_results if 'triple' in r['name']]
if triple_results:
    best_triple_result = max(triple_results, key=lambda x: x['combined'])
    print(f'\\n  Best triple candidate: {best_triple_result[\"name\"]} = {best_triple_result[\"combined\"]:.2f}', flush=True)

    # Also confirm the known best triple
    if manual_works:
        r = evaluate_alpha(BEST_TRIPLE, [1.0, 1.0, 1.0], 'known best triple (4,5)+(12,13)+(20,21) @1.0')
    else:
        r = evaluate_full(BEST_TRIPLE, 'known best triple (4,5)+(12,13)+(20,21) @1.0')
    all_results.append(r)

# =====================================================================
# PHASE 3: Sweep 3rd block alpha on best triple
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PHASE 3: Sweep 3rd block alpha on best triple')
print(f'{\"=\" * 70}', flush=True)

if manual_works:
    # Use the known best triple (4,5)+(12,13)+(20,21) and sweep alpha on (12,13)
    sweep_triple = BEST_TRIPLE
    print(f'  Sweeping alpha on middle block (12,13) in triple', flush=True)

    for a_mid in [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]:
        r = evaluate_alpha(sweep_triple, [1.0, a_mid, 1.0],
                           f'triple (4,5)+(12,13)+(20,21) @1.0/{a_mid}/1.0')
        all_results.append(r)

    # Also try sweeping first block alpha
    print(f'\\n  Sweeping alpha on first block (4,5) in triple', flush=True)
    for a_first in [0.5, 0.7, 0.9, 1.15]:
        r = evaluate_alpha(sweep_triple, [a_first, 1.0, 1.0],
                           f'triple (4,5)+(12,13)+(20,21) @{a_first}/1.0/1.0')
        all_results.append(r)

    # Sweep last block alpha
    print(f'\\n  Sweeping alpha on last block (20,21) in triple', flush=True)
    for a_last in [0.9, 1.1, 1.15, 1.2]:
        r = evaluate_alpha(sweep_triple, [1.0, 1.0, a_last],
                           f'triple (4,5)+(12,13)+(20,21) @1.0/1.0/{a_last}')
        all_results.append(r)

    # Best of each sweep combined
    best_triple_alpha = max(
        [r for r in all_results if 'triple' in r['name'] and 'alphas' in r],
        key=lambda x: x['combined']
    )
    print(f'\\n  Best alpha-tuned triple: {best_triple_alpha[\"name\"]} = {best_triple_alpha[\"combined\"]:.2f}', flush=True)
else:
    print('  Manual loop unavailable — skipping alpha sweep', flush=True)

# =====================================================================
# PHASE 4: Quad — add a 4th block at low alpha
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PHASE 4: Quad — best triple + 4th block')
print(f'{\"=\" * 70}', flush=True)

FOURTH_CANDIDATES = [(8, 9), (14, 15), (16, 17), (24, 25), (28, 29), (30, 31)]

for fourth in FOURTH_CANDIDATES:
    # Check no overlap with best triple blocks
    overlap = any(not (fourth[1] <= b[0] or fourth[0] >= b[1]) for b in BEST_TRIPLE)
    if overlap:
        print(f'  Skipping ({fourth[0]},{fourth[1]}): overlaps', flush=True)
        continue

    quad = list(BEST_TRIPLE) + [fourth]

    if manual_works:
        # First at alpha=1.0
        r = evaluate_alpha(quad, [1.0, 1.0, 1.0, 1.0],
                           f'quad +({fourth[0]},{fourth[1]}) @1.0/1.0/1.0/1.0')
        all_results.append(r)

        # Low alpha on 4th block (whisper)
        for a4 in [0.05, 0.1, 0.2, 0.5]:
            r = evaluate_alpha(quad, [1.0, 1.0, 1.0, a4],
                               f'quad +({fourth[0]},{fourth[1]}) @1.0/1.0/1.0/{a4}')
            all_results.append(r)
    else:
        r = evaluate_full(quad, f'quad +({fourth[0]},{fourth[1]}) @1.0')
        all_results.append(r)

# =====================================================================
# PHASE 5: If manual works, try whisper-alpha triples with non-(12,13) middles
# =====================================================================
if manual_works:
    print(f'\\n{\"=\" * 70}')
    print('PHASE 5: Whisper-alpha triples — low alpha on 3rd block')
    print(f'{\"=\" * 70}', flush=True)

    WHISPER_THIRDS = [(8, 9), (14, 15), (16, 17), (24, 25)]
    for third in WHISPER_THIRDS:
        overlap = any(not (third[1] <= b[0] or third[0] >= b[1]) for b in BEST_PAIR)
        if overlap:
            continue
        triple = list(BEST_PAIR) + [third]
        for a3 in [0.05, 0.1, 0.2]:
            r = evaluate_alpha(triple, [1.0, 1.0, a3],
                               f'whisper triple +({third[0]},{third[1]}) @1.0/1.0/{a3}')
            all_results.append(r)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
print(f'Baseline: {baseline:.2f}')
print(f'Known results: single=83.76, pair=84.42, triple=85.43')
print(f'Manual layer loop: {\"WORKS\" if manual_works else \"FAILED\"}')
print(f'\\nTop 20 results:')

sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for i, r in enumerate(sorted_r[:20]):
    marker = '***' if r['combined'] > 85.43 else '   '
    print(f'{marker} {i+1:2d}. {r[\"name\"]:65s}: combined={r[\"combined\"]:.2f}', flush=True)

# Categorized bests
for category in ['pair', 'triple', 'quad', 'whisper']:
    cat_results = [r for r in all_results if category in r.get('name', '')]
    if cat_results:
        best_cat = max(cat_results, key=lambda x: x['combined'])
        print(f'\\nBest {category}: {best_cat[\"name\"]} = {best_cat[\"combined\"]:.2f}', flush=True)

# Save results
os.makedirs('results/data/gemma3_27b/per_layer_alpha', exist_ok=True)
with open('results/data/gemma3_27b/per_layer_alpha/results.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'model': MODEL_PATH,
        'num_layers': N,
        'baseline': baseline,
        'manual_layer_loop_works': manual_works,
        'known_results': {
            'baseline': 80.54,
            'best_single_20_21': 83.76,
            'best_pair_4_5_20_21': 84.42,
            'best_triple_4_5_12_13_20_21': 85.43,
        },
        'results': all_results,
    }, f, indent=2)
print(f'\\nSaved to results/data/gemma3_27b/per_layer_alpha/results.json', flush=True)
print(f'\\nFinished: {datetime.now().isoformat()}', flush=True)
"

echo "=== Done at $(date) ==="
