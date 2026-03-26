#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ext_triple_%j.log
#SBATCH --job-name=deeppass_xtrip

# Extended triple search: now that alpha=0.1 on third block works,
# 1. Try many third block candidates at alpha=0.1
# 2. Use optimized pair alphas (0,7)@0.9 + (45,52)@1.0 as base
# 3. Try quads: add a FOURTH block at alpha=0.05
# 4. Fine-tune alpha of best third block (0.05, 0.08, 0.1, 0.12, 0.15)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Extended Triple + Quad Search ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('EXTENDED TRIPLE + QUAD SEARCH')
print('Base: (0,7)@0.9 + (45,52)@1.0 = 81.24 (optimized pair)')
print('Breakthrough: (15,20)@0.1 as third block → 81.85')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

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
    seams = []
    for block in sorted(blocks):
        last_layer = block[1] - 1
        occurrences = [step for step, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) >= 2:
            seams.append((occurrences[0], occurrences[1]))
        else:
            seams.append(None)
    return seams

def generate_multi_alpha(prompt, blocks, alphas, max_new_tokens=64):
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
            pos_embeds = inner.rotary_emb(h, pos_ids)
            saved_h1 = {}
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                for si, seam in enumerate(seams):
                    if seam is None: continue
                    first_end, second_end = seam
                    if step_idx == first_end: saved_h1[si] = h.clone()
                    if step_idx == second_end and si in saved_h1:
                        a = sorted_alphas[si]
                        h = saved_h1[si] + a * (h - saved_h1[si])
                        del saved_h1[si]
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def evaluate(blocks, alphas, name):
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

# Reference: optimized pair
r = evaluate([(0,7), (45,52)], [0.9, 1.0], 'REFERENCE: (0,7)@0.9+(45,52)@1.0')
pair_combined = r['combined']
all_results.append(r)

# =====================================================================
# PART 1: Many third blocks at alpha=0.1 with OPTIMIZED pair alphas
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('PART 1: Third block candidates at alpha=0.1')
print('Base: (0,7)@0.9 + (45,52)@1.0')
print(f'{\"=\" * 70}', flush=True)

THIRD_CANDIDATES = [
    (10, 17), (15, 20), (20, 27), (25, 32), (30, 37),
    (35, 40), (50, 55), (50, 60), (55, 62), (60, 65),
    (8, 13), (10, 15), (12, 17), (18, 23), (22, 27),
]

for third in THIRD_CANDIDATES:
    # Skip if overlapping with (0,7) or (45,52)
    if third[1] > 0 and third[0] < 7: continue
    if third[1] > 45 and third[0] < 52: continue
    blocks = [(0, 7), third, (45, 52)]
    r = evaluate(blocks, [0.9, 0.1, 1.0], f'triple +({third[0]},{third[1]})@0.1')
    all_results.append(r)

# Find best third block
triple_results = [r for r in all_results if 'triple' in r['name']]
if triple_results:
    best_triple = max(triple_results, key=lambda x: x['combined'])
    print(f'\\n*** Best third block: {best_triple[\"name\"]} = {best_triple[\"combined\"]:.2f} ***', flush=True)

    # =====================================================================
    # PART 2: Fine-tune alpha of best third block
    # =====================================================================
    best_third_block = tuple(best_triple['blocks'][1])  # middle block
    print(f'\\n{\"=\" * 70}')
    print(f'PART 2: Fine-tune alpha for ({best_third_block[0]},{best_third_block[1]})')
    print(f'{\"=\" * 70}', flush=True)

    for a3 in [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]:
        blocks = [(0, 7), best_third_block, (45, 52)]
        r = evaluate(blocks, [0.9, a3, 1.0], f'triple +({best_third_block[0]},{best_third_block[1]})@{a3} fine')
        all_results.append(r)

    # =====================================================================
    # PART 3: QUAD — add a fourth block at very low alpha
    # =====================================================================
    print(f'\\n{\"=\" * 70}')
    print('PART 3: Quad search — can we add a FOURTH block?')
    print(f'{\"=\" * 70}', flush=True)

    FOURTH_CANDIDATES = [
        (20, 27), (25, 32), (30, 37), (35, 40),
        (50, 55), (55, 62), (60, 65),
    ]

    for fourth in FOURTH_CANDIDATES:
        # Skip overlapping
        existing = [(0, 7), best_third_block, (45, 52)]
        overlap = any(not (fourth[1] <= b[0] or fourth[0] >= b[1]) for b in existing)
        if overlap: continue
        blocks = existing + [fourth]
        # Fourth block at alpha=0.05 (even gentler)
        alphas = [0.9, 0.1, 1.0, 0.05]
        r = evaluate(blocks, alphas, f'quad +({fourth[0]},{fourth[1]})@0.05')
        all_results.append(r)

    # Also try fourth at alpha=0.1
    quad_results = [r for r in all_results if 'quad' in r['name']]
    if quad_results:
        best_quad = max(quad_results, key=lambda x: x['combined'])
        best_fourth = tuple(best_quad['blocks'][3] if len(best_quad['blocks']) > 3 else best_quad['blocks'][-1])
        for a4 in [0.02, 0.05, 0.1, 0.15]:
            blocks = existing + [best_fourth]
            alphas = [0.9, 0.1, 1.0, a4]
            r = evaluate(blocks, alphas, f'quad +({best_fourth[0]},{best_fourth[1]})@{a4} fine')
            all_results.append(r)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:20]:
    beat = '***' if r['combined'] > pair_combined else '   '
    print(f'{beat} {r[\"name\"]:65s}: combined={r[\"combined\"]:.2f}', flush=True)

os.makedirs('results/data/72b/deeper_stacking', exist_ok=True)
with open('results/data/72b/deeper_stacking/extended_results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'pair_reference': pair_combined,
               'all_results': all_results}, f, indent=2)
print(f'\\nSaved to results/data/72b/deeper_stacking/extended_results.json', flush=True)
"

echo "=== Done at $(date) ==="
