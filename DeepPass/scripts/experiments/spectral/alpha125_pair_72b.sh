#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_alpha125_pair_%j.log
#SBATCH --job-name=deeppass_a125

# Test alpha=1.25 on:
# 1. Single (50,60) — missing from crashed dir72b run
# 2. Best pair (0,7)+(45,52) — the big test
# 3. Alpha sweep on the pair to find optimal

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Alpha=1.25 on Singles + Best Pair ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'scripts/experiments/spectral')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe
from norm_preserving_test import (
    build_layer_order, find_seam_positions,
    evaluate_baseline_no_dup
)

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('ALPHA=1.25 ON SINGLES + BEST PAIR')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N

# =====================================================================
# Multi-block seam intervention
# =====================================================================

def generate_with_multi_seam(model, inner, tokenizer, prompt, layer_order,
                              original_layers, blocks, alpha, max_new_tokens=64):
    \"\"\"Generate with alpha blending at EVERY seam in a multi-block config.\"\"\"
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # Find seam positions for each block
    seam_info = []
    for block in blocks:
        i, j = block
        last_layer = j - 1
        occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
        if len(occurrences) >= 2:
            seam_info.append((occurrences[0], occurrences[1]))  # (first_pass_end, second_pass_end)

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

                for si, (first_end, second_end) in enumerate(seam_info):
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
                    if step_idx == second_end and si in saved_h1:
                        h1 = saved_h1[si]
                        h2 = h
                        h = h1 + alpha * (h2 - h1)
                        del saved_h1[si]

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    generated = input_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_config(blocks, alpha, name):
    \"\"\"Evaluate a config with given alpha at all seams.\"\"\"
    layer_order = build_layer_order(tuple(blocks[0]) if len(blocks) == 1 else None, N) if len(blocks) == 1 else None

    # Build layer order for multi-block
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    layer_order = order

    def gen(p):
        return generate_with_multi_seam(model, inner, tokenizer, p, layer_order,
                                         original_layers, blocks, alpha, max_new_tokens=64)
    def gen_long(p):
        return generate_with_multi_seam(model, inner, tokenizer, p, layer_order,
                                         original_layers, blocks, alpha, max_new_tokens=128)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0

    print(f'  {name:50s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alpha': alpha,
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# Baseline
print('\\n--- Baseline ---', flush=True)
baseline_r = evaluate_baseline_no_dup(model, tokenizer)
baseline = baseline_r['combined']

all_results = [baseline_r]

# =====================================================================
# Test 1: Alpha sweep on single (50,60)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 1: Alpha sweep on single (50,60)')
print(f'{\"=\" * 70}', flush=True)

for alpha in [1.0, 1.1, 1.25, 1.35, 1.5]:
    r = evaluate_config([(50, 60)], alpha, f'(50,60) alpha={alpha}')
    all_results.append(r)

# =====================================================================
# Test 2: Alpha sweep on single (45,52)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 2: Alpha sweep on single (45,52) — verify dir72b result')
print(f'{\"=\" * 70}', flush=True)

for alpha in [1.0, 1.15, 1.25, 1.35]:
    r = evaluate_config([(45, 52)], alpha, f'(45,52) alpha={alpha}')
    all_results.append(r)

# =====================================================================
# Test 3: Alpha sweep on BEST PAIR (0,7)+(45,52)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 3: Alpha sweep on BEST PAIR (0,7)+(45,52)')
print('Current best: combined=79.91 at alpha=1.0')
print(f'{\"=\" * 70}', flush=True)

for alpha in [1.0, 1.1, 1.15, 1.2, 1.25, 1.35]:
    r = evaluate_config([(0, 7), (45, 52)], alpha, f'(0,7)+(45,52) alpha={alpha}')
    all_results.append(r)

# =====================================================================
# Test 4: Per-block alpha — different alpha for each block in the pair
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 4: Asymmetric alpha on pair')
print(f'{\"=\" * 70}', flush=True)

# Test: alpha=1.25 on deep block only, 1.0 on early block
# This requires a modified generation function
def generate_asymmetric(model, inner, tokenizer, prompt, layer_order,
                         original_layers, blocks, alphas, max_new_tokens=64):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    seam_info = []
    for block in blocks:
        i, j = block
        last_layer = j - 1
        occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
        if len(occurrences) >= 2:
            seam_info.append((occurrences[0], occurrences[1]))

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

                for si, (first_end, second_end) in enumerate(seam_info):
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
                    if step_idx == second_end and si in saved_h1:
                        a = alphas[si]
                        h1 = saved_h1[si]
                        h2 = h
                        h = h1 + a * (h2 - h1)
                        del saved_h1[si]

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    generated = input_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)

blocks_pair = [(0, 7), (45, 52)]
sorted_pair = sorted(blocks_pair)
order = []
prev = 0
for (i, j) in sorted_pair:
    order.extend(range(prev, j))
    order.extend(range(i, j))
    prev = j
order.extend(range(prev, N))

asymmetric_configs = [
    ([1.0, 1.25], '(0,7)@1.0 + (45,52)@1.25'),
    ([1.25, 1.0], '(0,7)@1.25 + (45,52)@1.0'),
    ([1.25, 1.25], '(0,7)@1.25 + (45,52)@1.25'),
    ([1.15, 1.25], '(0,7)@1.15 + (45,52)@1.25'),
    ([1.1, 1.15], '(0,7)@1.1 + (45,52)@1.15'),
]

for alphas, name in asymmetric_configs:
    def gen(p, a=alphas):
        return generate_asymmetric(model, inner, tokenizer, p, order,
                                    original_layers, sorted_pair, a, max_new_tokens=64)
    def gen_long(p, a=alphas):
        return generate_asymmetric(model, inner, tokenizer, p, order,
                                    original_layers, sorted_pair, a, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:50s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    all_results.append({'name': name, 'alphas': alphas, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# Summary
print(f'\\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    delta = r['combined'] - baseline
    print(f'  {r[\"name\"]:50s}: combined={r[\"combined\"]:.2f} delta={delta:+.2f}', flush=True)

# Save
os.makedirs('results/data/72b/direction_interventions', exist_ok=True)
with open('results/data/72b/direction_interventions/alpha125_pair_results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'baseline': baseline, 'results': all_results}, f, indent=2)
print(f'\\nSaved to results/data/72b/direction_interventions/alpha125_pair_results.json', flush=True)
"

echo "=== Done at $(date) ==="
