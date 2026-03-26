#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_alpha_stack_27b_%j.log
#SBATCH --job-name=deeppass_as27

# Per-block alpha stacking on Qwen3.5-27B (64 layers)
# Known: best single (25,30) = 78.30, no pair beat single at alpha=1.0
# Test: can whisper-alpha triples beat the single?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Alpha-Tuned Stacking on Qwen3.5-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/Qwen3.5-27B'

print('=' * 70)
print('ALPHA-TUNED STACKING ON QWEN3.5-27B')
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
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alphas': list(alphas),
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# Baseline
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

# Known best single
BEST_SINGLE = (25, 30)
r = evaluate([BEST_SINGLE], [1.0], f'single ({BEST_SINGLE[0]},{BEST_SINGLE[1]})')
all_results.append(r)
best_single_score = r['combined']

# Alpha-tuned single
for a in [0.9, 1.05, 1.1, 1.15, 1.2, 1.25]:
    r = evaluate([BEST_SINGLE], [a], f'single @{a}')
    all_results.append(r)

# Alpha-tuned pairs (previously all pairs lost to single at alpha=1.0)
print(f'\\n--- Alpha-tuned pairs ---', flush=True)
PAIR_CANDIDATES = [(45, 55), (55, 60), (20, 30), (40, 50)]
for second in PAIR_CANDIDATES:
    if second[1] <= BEST_SINGLE[0] or second[0] >= BEST_SINGLE[1]:
        # Standard pair
        r = evaluate([BEST_SINGLE, second], [1.0, 1.0], f'pair +({second[0]},{second[1]}) @1.0/1.0')
        all_results.append(r)
        # Alpha-tuned: dampen second block
        for a2 in [0.1, 0.2, 0.5, 0.7]:
            r = evaluate([BEST_SINGLE, second], [1.0, a2], f'pair +({second[0]},{second[1]}) @1.0/{a2}')
            all_results.append(r)

# Whisper triples
print(f'\\n--- Whisper triples ---', flush=True)
best_pair = max([r for r in all_results if 'pair' in r['name']], key=lambda x: x['combined'], default=None)
if best_pair:
    print(f'Best pair so far: {best_pair[\"name\"]} = {best_pair[\"combined\"]:.2f}', flush=True)
    bp_blocks = [tuple(b) for b in best_pair['blocks']]
    bp_alphas = best_pair['alphas']

    THIRD_CANDIDATES = [(10, 15), (15, 20), (35, 40), (50, 55)]
    for third in THIRD_CANDIDATES:
        overlap = any(not (third[1] <= b[0] or third[0] >= b[1]) for b in bp_blocks)
        if overlap: continue
        triple = list(bp_blocks) + [third]
        for a3 in [0.05, 0.1, 0.15, 1.0]:
            alphas = list(bp_alphas) + [a3]
            r = evaluate(triple, alphas, f'triple +({third[0]},{third[1]})@{a3}')
            all_results.append(r)

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
print(f'Baseline: {baseline:.2f}')
print(f'Best single: {best_single_score:.2f}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    beat = '***' if r['combined'] > best_single_score else '   '
    print(f'{beat} {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f}', flush=True)

triple_beats = any(r['combined'] > best_single_score for r in all_results if 'triple' in r.get('name', ''))
pair_beats = any(r['combined'] > best_single_score for r in all_results if 'pair' in r.get('name', ''))
print(f'\\nPair beats single: {\"YES\" if pair_beats else \"NO\"}')
print(f'Triple beats single: {\"YES\" if triple_beats else \"NO\"}', flush=True)

os.makedirs('results/data/qwen35/alpha_stacking', exist_ok=True)
with open('results/data/qwen35/alpha_stacking/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'baseline': baseline,
               'best_single': best_single_score, 'results': all_results}, f, indent=2)
print(f'Saved to results/data/qwen35/alpha_stacking/results.json', flush=True)
"

echo "=== Done at $(date) ==="
