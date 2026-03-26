#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_crosslayer_%j.log
#SBATCH --job-name=deeppass_xlyr

# Cross-layer duplication: second pass uses DIFFERENT block's weights
# Instead of F(F(h)), try G(F(h)) where G is from a different region
# Novel: nobody has tested this

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Cross-Layer Duplication ==="
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
print('CROSS-LAYER DUPLICATION')
print('Second pass uses weights from a DIFFERENT block')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

def generate_cross_layer(prompt, first_block, second_block_weights, alpha=1.0, max_new_tokens=64):
    \"\"\"
    Run first_block normally, then run second_block_weights at the same insertion point.
    Layer order: [0..j-1, first_block, second_block_weights, j..N-1]
    where first_block = (i,j) and second_block_weights = (a,b) provides layers a..b-1 for the second pass.
    \"\"\"
    i, j = first_block
    a, b = second_block_weights
    # Build custom layer order
    order = list(range(j))  # layers before block
    order += list(range(i, j))  # first pass (original block)
    order += list(range(a, b))  # second pass (DIFFERENT block's weights!)
    order += list(range(j, N))  # layers after block

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # Find seam: where first pass ends and second pass begins
    # First pass ends at position j-1 in order (index j-1)
    # Second pass ends at position j + (b-a) - 1
    first_pass_end = j - 1  # last step of first pass
    second_pass_end = j + (b - a) - 1  # last step of second pass

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_after_first = None
            for step_idx, layer_idx in enumerate(order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                if step_idx == first_pass_end:
                    h_after_first = h.clone()
                if step_idx == second_pass_end and h_after_first is not None:
                    # Apply alpha blending at the seam
                    h = h_after_first + alpha * (h - h_after_first)

            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

from layer_duplicator import generate_no_cache

def evaluate(first_block, second_weights, alpha, name):
    gen = lambda p: generate_cross_layer(p, first_block, second_weights, alpha, max_new_tokens=64)
    gen_long = lambda p: generate_cross_layer(p, first_block, second_weights, alpha, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:55s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# Baseline
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

# Reference: standard self-duplication (45,52) -> (45,52)
r = evaluate((45, 52), (45, 52), 1.0, 'REFERENCE: (45,52) self-dup @1.0')
all_results.append(r)

# =====================================================================
# Cross-layer: first pass (45,52), second pass uses different blocks
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('First pass: (45,52). Second pass weights from different blocks.')
print(f'{\"=\" * 70}', flush=True)

CROSS_CANDIDATES = [
    (0, 7), (5, 12), (10, 17), (15, 22), (20, 27),
    (25, 32), (30, 37), (35, 42), (50, 57), (55, 62),
    (60, 67), (73, 80),
]

for cross in CROSS_CANDIDATES:
    r = evaluate((45, 52), cross, 1.0, f'(45,52)->({cross[0]},{cross[1]}) @1.0')
    all_results.append(r)

# Best cross-layer config with alpha tuning
cross_results = [r for r in all_results if 'self-dup' not in r['name'] and 'REFERENCE' not in r['name']]
if cross_results:
    best_cross = max(cross_results, key=lambda x: x['combined'])
    print(f'\\nBest cross-layer: {best_cross[\"name\"]} = {best_cross[\"combined\"]:.2f}', flush=True)

    # Alpha sweep on best cross config
    # Extract blocks from name
    import re
    match = re.search(r'\((\d+),(\d+)\)->\\((\d+),(\d+)\\)', best_cross['name'])
    if not match:
        # Try simpler parse
        parts = best_cross['name'].split('->')
        if len(parts) == 2:
            first = (45, 52)
            cross_match = re.search(r'\((\d+),(\d+)\)', parts[1])
            if cross_match:
                best_cross_block = (int(cross_match.group(1)), int(cross_match.group(2)))
                print(f'\\n--- Alpha sweep on best cross config ---', flush=True)
                for alpha in [0.1, 0.3, 0.5, 0.7, 1.0, 1.25]:
                    r = evaluate(first, best_cross_block, alpha, f'cross ({first[0]},{first[1]})->({best_cross_block[0]},{best_cross_block[1]}) @{alpha}')
                    all_results.append(r)

# =====================================================================
# Also test: first pass (0,7), second pass from different blocks
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('First pass: (0,7). Second pass weights from different blocks.')
print(f'{\"=\" * 70}', flush=True)

r = evaluate((0, 7), (0, 7), 1.0, 'REFERENCE: (0,7) self-dup @1.0')
all_results.append(r)

for cross in [(10, 17), (20, 27), (30, 37), (45, 52), (55, 62)]:
    r = evaluate((0, 7), cross, 1.0, f'(0,7)->({cross[0]},{cross[1]}) @1.0')
    all_results.append(r)

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
print(f'Baseline: {baseline:.2f}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    beat = '***' if r['combined'] > all_results[0]['combined'] else '   '
    print(f'{beat} {r[\"name\"]:55s}: combined={r[\"combined\"]:.2f}', flush=True)

cross_beats_self = any(r['combined'] > all_results[0]['combined'] for r in cross_results) if cross_results else False
print(f'\\nCross-layer beats self-duplication: {\"YES\" if cross_beats_self else \"NO\"}', flush=True)

os.makedirs('results/data/72b/cross_layer', exist_ok=True)
with open('results/data/72b/cross_layer/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'baseline': baseline, 'results': all_results,
               'cross_beats_self': cross_beats_self}, f, indent=2)
print(f'Saved to results/data/72b/cross_layer/results.json', flush=True)
"

echo "=== Done at $(date) ==="
