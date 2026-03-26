#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_deep5_%j.log
#SBATCH --job-name=deeppass_deep5

# 5-block stacking: (0,7) + (15,20) + (20,27) + (35,40) + (45,52)
# Each additional block at decreasing alphas
# Pattern: core blocks at high alpha, additional blocks at very low alpha

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Deep 5-Block Stacking ==="
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
print('DEEP 5-BLOCK STACKING')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

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
        if len(occurrences) >= 2: seams.append((occurrences[0], occurrences[1]))
        else: seams.append(None)
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
                    if step_idx == seam[0]: saved_h1[si] = h.clone()
                    if step_idx == seam[1] and si in saved_h1:
                        h = saved_h1[si] + sorted_alphas[si] * (h - saved_h1[si])
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
    print(f'  {name:70s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alphas': list(alphas),
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# Reference: best triple
r = evaluate([(0,7), (20,27), (45,52)], [0.9, 0.15, 1.0], 'REFERENCE: best triple (0,7)@0.9+(20,27)@0.15+(45,52)@1.0')
best_triple = r['combined']
all_results.append(r)

# =====================================================================
# Quad configurations — build on best triple
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('QUADS: Add 4th block to best triple')
print(f'{\"=\" * 70}', flush=True)

QUAD_CANDIDATES = [(10, 15), (15, 20), (30, 37), (35, 40), (55, 60), (55, 62), (60, 65)]

for fourth in QUAD_CANDIDATES:
    # Check overlap with existing blocks
    existing = [(0,7), (20,27), (45,52)]
    overlap = any(not (fourth[1] <= b[0] or fourth[0] >= b[1]) for b in existing)
    if overlap: continue
    for a4 in [0.02, 0.05, 0.1]:
        blocks = existing + [fourth]
        alphas = [0.9, 0.15, 1.0, a4]
        r = evaluate(blocks, alphas, f'quad +({fourth[0]},{fourth[1]})@{a4}')
        all_results.append(r)

# Find best quad
quad_results = [r for r in all_results if 'quad' in r['name']]
if quad_results:
    best_quad_r = max(quad_results, key=lambda x: x['combined'])
    print(f'\\n*** Best quad: {best_quad_r[\"name\"]} = {best_quad_r[\"combined\"]:.2f} ***', flush=True)

    # =====================================================================
    # Quint: Add 5th block
    # =====================================================================
    print(f'\\n{\"=\" * 70}')
    print('QUINTS: Add 5th block to best quad')
    print(f'{\"=\" * 70}', flush=True)

    best_quad_blocks = [tuple(b) for b in best_quad_r['blocks']]
    QUINT_CANDIDATES = [(10, 15), (15, 20), (30, 37), (35, 40), (55, 60), (60, 65)]

    for fifth in QUINT_CANDIDATES:
        overlap = any(not (fifth[1] <= b[0] or fifth[0] >= b[1]) for b in best_quad_blocks)
        if overlap: continue
        blocks = best_quad_blocks + [fifth]
        alphas = list(best_quad_r['alphas']) + [0.02]
        r = evaluate(blocks, alphas, f'quint +({fifth[0]},{fifth[1]})@0.02')
        all_results.append(r)

# =====================================================================
# Alternative: many weak blocks at alpha=0.05
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('ALTERNATIVE: Many weak blocks at uniform low alpha')
print(f'{\"=\" * 70}', flush=True)

# 3 blocks all at alpha=0.1
many_configs = [
    ('3x@0.1: early+mid+deep', [(0,7), (20,27), (55,62)], [0.1, 0.1, 0.1]),
    ('4x@0.05: spread', [(0,7), (15,20), (35,40), (55,62)], [0.05, 0.05, 0.05, 0.05]),
    ('5x@0.03: full spread', [(0,7), (15,20), (30,37), (45,52), (60,65)], [0.03, 0.03, 0.03, 0.03, 0.03]),
    ('5x mixed: core+whisper', [(0,7), (15,20), (30,37), (45,52), (60,65)], [0.9, 0.05, 0.05, 1.0, 0.02]),
    ('5x decay: 0.5→0.1', [(0,7), (15,20), (30,37), (45,52), (60,65)], [0.5, 0.3, 0.15, 0.1, 0.05]),
]

for name, blocks, alphas in many_configs:
    r = evaluate(blocks, alphas, name)
    all_results.append(r)

# Summary
print(f'\\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    beat = '***' if r['combined'] > best_triple else '   '
    n_blocks = len(r.get('blocks', r.get('alphas', [])))
    print(f'{beat} [{n_blocks}blk] {r[\"name\"]:65s}: combined={r[\"combined\"]:.2f}', flush=True)

os.makedirs('results/data/72b/deeper_stacking', exist_ok=True)
with open('results/data/72b/deeper_stacking/deep5_results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'best_triple_ref': best_triple,
               'all_results': all_results}, f, indent=2)
print(f'\\nSaved to results/data/72b/deeper_stacking/deep5_results.json', flush=True)
"

echo "=== Done at $(date) ==="
