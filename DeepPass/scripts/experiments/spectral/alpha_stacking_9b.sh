#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_alpha_stack_9b_%j.log
#SBATCH --job-name=deeppass_as9b

# Test alpha-tuned stacking on Qwen3.5-9B (32 layers)
# Does the "whisper alpha" approach generalize to a smaller model?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Alpha-Tuned Stacking on Qwen3.5-9B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/Qwen3.5-9B'

print('=' * 70)
print('ALPHA-TUNED STACKING ON QWEN3.5-9B')
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

# Step 1: Quick spectral screen for best blocks
print(f'\\n--- Spectral screen ---', flush=True)
CAL_PROMPTS = ['What is 127 * 348?', 'What is 99999 * 99999?', 'Calculate 15! / 13!', 'What is 2^16?']
block_rhos = {}
candidates = []
for start in range(0, N-1, 2):
    for size in [1, 3, 5, 7]:
        end = start + size
        if end <= N: candidates.append((start, end))

for idx, block in enumerate(candidates):
    i, j = block
    rhos = []
    for prompt in CAL_PROMPTS:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(ids['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()
            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
            if hasattr(model.config, 'num_hidden_layers'):
                model.config.num_hidden_layers = len(order)
            out_dup = model(ids['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()
            inner.layers = nn.ModuleList(original_layers)
            model.config.num_hidden_layers = N
            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8: rhos.append(num / den)
    block_rhos[block] = float(np.mean(rhos)) if rhos else 1.0

sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
print(f'Top 8 blocks by rho:', flush=True)
top8 = [b for b, _ in sorted_blocks[:8]]
for b, r in sorted_blocks[:8]:
    print(f'  ({b[0]:2d},{b[1]:2d}) rho={r:.4f}', flush=True)

# Step 2: Evaluate top-6 singles
print(f'\\n--- Singles ---', flush=True)
single_scores = {}
for block in top8[:6]:
    r = evaluate([block], [1.0], f'single ({block[0]},{block[1]})')
    single_scores[block] = r['combined']
    all_results.append(r)

best_single = max(single_scores.items(), key=lambda x: x[1])
print(f'Best single: ({best_single[0][0]},{best_single[0][1]}) = {best_single[1]:.2f}', flush=True)

# Step 3: Best pair at alpha=1.0
print(f'\\n--- Pairs ---', flush=True)
from itertools import combinations
pair_results = []
for a, b in combinations(top8[:6], 2):
    if a[1] <= b[0] or b[1] <= a[0]:  # non-overlapping
        r = evaluate([a, b], [1.0, 1.0], f'pair ({a[0]},{a[1]})+({b[0]},{b[1]})')
        pair_results.append(r)
        all_results.append(r)

best_pair = max(pair_results, key=lambda x: x['combined']) if pair_results else None
if best_pair:
    print(f'Best pair: {best_pair[\"name\"]} = {best_pair[\"combined\"]:.2f}', flush=True)
    pair_blocks = [tuple(b) for b in best_pair['blocks']]

    # Step 4: Alpha-tuned pair
    print(f'\\n--- Alpha-tuned pair ---', flush=True)
    for a0 in [0.8, 0.9, 1.0, 1.1, 1.15]:
        for a1 in [0.8, 0.9, 1.0, 1.1, 1.15]:
            if a0 == 1.0 and a1 == 1.0: continue  # already tested
            r = evaluate(pair_blocks, [a0, a1], f'pair @{a0}/{a1}')
            all_results.append(r)

    best_pair_tuned = max([r for r in all_results if 'pair' in r['name']], key=lambda x: x['combined'])
    print(f'Best tuned pair: {best_pair_tuned[\"name\"]} = {best_pair_tuned[\"combined\"]:.2f}', flush=True)

    # Step 5: Triple with whisper alpha
    print(f'\\n--- Triple with whisper alpha ---', flush=True)
    third_candidates = [b for b in top8[:6] if b not in pair_blocks and
                        all(b[1] <= pb[0] or b[0] >= pb[1] for pb in pair_blocks)]

    for third in third_candidates[:4]:
        triple = list(pair_blocks) + [third]
        for a3 in [0.05, 0.1, 0.15, 0.2, 1.0]:
            alphas = [1.0, 1.0, a3]
            r = evaluate(triple, alphas, f'triple +({third[0]},{third[1]})@{a3}')
            all_results.append(r)

    # Step 6: Quad?
    triple_results = [r for r in all_results if 'triple' in r['name']]
    if triple_results:
        best_triple = max(triple_results, key=lambda x: x['combined'])
        print(f'Best triple: {best_triple[\"name\"]} = {best_triple[\"combined\"]:.2f}', flush=True)

        if best_triple['combined'] > best_pair['combined']:
            print('*** TRIPLE BEATS PAIR ON 9B! Whisper alpha generalizes! ***', flush=True)

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
print(f'Baseline: {baseline:.2f}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    print(f'  {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f}', flush=True)

os.makedirs('results/data/qwen35_9b/alpha_stacking', exist_ok=True)
with open('results/data/qwen35_9b/alpha_stacking/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'baseline': baseline, 'results': all_results}, f, indent=2)
print(f'\\nSaved to results/data/qwen35_9b/alpha_stacking/results.json', flush=True)
"

echo "=== Done at $(date) ==="
