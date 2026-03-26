#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_deeper_stack_%j.log
#SBATCH --job-name=deeppass_dstack

# Three approaches to break the two-block ceiling:
# 1. Low-alpha third block sweep (the EQ-bench destruction is from overshooting)
# 2. Residual scaling within the second pass (distributed dampening)
# 3. Mean-shift correction between blocks (re-center distribution)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Deeper Stacking: Breaking the Two-Block Ceiling ==="
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
print('DEEPER STACKING: THREE APPROACHES')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

CAL_PROMPTS = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'Calculate 15! / 13!', 'What is 2^16?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
    'Describe the feeling of watching a sunset after a difficult day.',
    'What is the derivative of sin(x) * e^x?',
]

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
        i, j = block
        last_layer = j - 1
        occurrences = [step for step, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) >= 2:
            seams.append((occurrences[0], occurrences[1]))
        else:
            seams.append(None)
    return seams

def find_second_pass_ranges(layer_order, blocks):
    \"\"\"Find step index ranges where each block's SECOND pass occurs.\"\"\"
    ranges = []
    for block in sorted(blocks):
        i, j = block
        # Find all steps that correspond to this block's layers
        block_layers = set(range(i, j))
        steps_in_block = [step for step, idx in enumerate(layer_order) if idx in block_layers]
        # First pass = first (j-i) steps, second pass = remaining
        block_size = j - i
        if len(steps_in_block) >= 2 * block_size:
            second_pass_steps = steps_in_block[block_size:]
            ranges.append((min(second_pass_steps), max(second_pass_steps)))
        else:
            ranges.append(None)
    return ranges

# =====================================================================
# Approach 1: Low-alpha third block sweep
# =====================================================================
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
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
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

# =====================================================================
# Approach 2: Residual scaling within second pass
# =====================================================================
def generate_residual_scaled(prompt, blocks, alphas, betas, max_new_tokens=64):
    \"\"\"Like generate_multi_alpha but also scales residuals within second pass.
    betas: dict mapping block_index -> beta for residual scaling during 2nd pass.
    h = h_in + beta * (layer(h_in) - h_in) instead of h = layer(h_in)
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    block_to_alpha = {b: a for b, a in zip(blocks, alphas)}
    block_to_beta = {b: bt for b, bt in zip(blocks, betas)}
    layer_order = build_order(sorted_blocks, N)
    seams = find_seams(layer_order, sorted_blocks)
    second_pass_ranges = find_second_pass_ranges(layer_order, sorted_blocks)
    sorted_alphas = [block_to_alpha[b] for b in sorted_blocks]
    sorted_betas = [block_to_beta[b] for b in sorted_blocks]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            saved_h1 = {}
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                h_before = h.clone()
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h_after = out[0] if isinstance(out, tuple) else out

                # Check if we're in a second pass and should apply residual scaling
                in_second_pass = False
                for si, spr in enumerate(second_pass_ranges):
                    if spr is not None and spr[0] <= step_idx <= spr[1]:
                        beta = sorted_betas[si]
                        if beta < 1.0:
                            h = h_before + beta * (h_after - h_before)
                            in_second_pass = True
                        break
                if not in_second_pass:
                    h = h_after

                # Seam alpha blending
                for si, seam in enumerate(seams):
                    if seam is None: continue
                    first_end, second_end = seam
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
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

# =====================================================================
# Approach 3: Mean-shift correction
# =====================================================================
def collect_seam_deltas(blocks, alphas):
    \"\"\"Collect mean(h2 - h1) at each seam over calibration prompts.\"\"\"
    sorted_blocks = sorted(blocks)
    block_to_alpha = {b: a for b, a in zip(blocks, alphas)}
    layer_order = build_order(sorted_blocks, N)
    seams = find_seams(layer_order, sorted_blocks)
    sorted_alphas = [block_to_alpha[b] for b in sorted_blocks]

    seam_deltas = [[] for _ in range(len(sorted_blocks))]

    for prompt in CAL_PROMPTS:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = inner.embed_tokens(ids['input_ids'])
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
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
                    if step_idx == second_end and si in saved_h1:
                        delta = h - saved_h1[si]
                        seam_deltas[si].append(delta.mean(dim=1).squeeze(0))  # avg over seq
                        a = sorted_alphas[si]
                        h = saved_h1[si] + a * (h - saved_h1[si])
                        del saved_h1[si]

    mean_deltas = []
    for deltas in seam_deltas:
        if deltas:
            mean_deltas.append(torch.stack(deltas).mean(dim=0))
        else:
            mean_deltas.append(None)
    return mean_deltas

def generate_mean_shift_corrected(prompt, blocks, alphas, corrections, correction_scales, max_new_tokens=64):
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
                    if step_idx == first_end:
                        saved_h1[si] = h.clone()
                    if step_idx == second_end and si in saved_h1:
                        a = sorted_alphas[si]
                        h = saved_h1[si] + a * (h - saved_h1[si])
                        # Apply mean-shift correction (skip first block)
                        if si > 0 and corrections[si] is not None:
                            scale = correction_scales[si]
                            h = h - scale * corrections[si].unsqueeze(0).unsqueeze(0).to(h.device)
                        del saved_h1[si]
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

# =====================================================================
# Evaluation helper
# =====================================================================
def evaluate_blocks(blocks, alphas, name, gen_func=None, gen_func_long=None):
    \"\"\"Evaluate a block config with dual probe. If gen_func not provided, uses generate_multi_alpha.\"\"\"
    if gen_func is None:
        gen_func = lambda p: generate_multi_alpha(p, blocks, alphas, max_new_tokens=64)
        gen_func_long = lambda p: generate_multi_alpha(p, blocks, alphas, max_new_tokens=128)
    elif gen_func_long is None:
        gen_func_long = gen_func
    t0 = time.time()
    math_r = run_math_probe(gen_func, verbose=False)
    eq_r = run_eq_bench_probe(gen_func_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# Baseline
from layer_duplicator import generate_no_cache
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

# Reference: best pair at alpha=1.0
pair_blocks = [(0, 7), (45, 52)]
pair_r = evaluate_blocks(pair_blocks, [1.0, 1.0], 'REFERENCE: (0,7)+(45,52) pair @1.0')

all_results = [pair_r]

# =====================================================================
# APPROACH 1: Low-alpha third block sweep
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('APPROACH 1: Low-alpha third block')
print('Fix (0,7)@1.0 + (45,52)@1.0, sweep third block alpha')
print(f'{\"=\" * 70}', flush=True)

THIRD_BLOCK_CANDIDATES = [(15, 20), (10, 17), (50, 60)]

for third in THIRD_BLOCK_CANDIDATES:
    print(f'\\n  Third block: ({third[0]},{third[1]})', flush=True)
    triple_blocks = [(0, 7), third, (45, 52)]

    # Standard alpha=1.0 for reference
    r = evaluate_blocks(triple_blocks, [1.0, 1.0, 1.0], f'triple +({third[0]},{third[1]}) all@1.0')
    all_results.append(r)

    # Sweep low alphas for third block
    for a3 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        r = evaluate_blocks(triple_blocks, [1.0, a3, 1.0], f'triple +({third[0]},{third[1]})@{a3}')
        all_results.append(r)

# =====================================================================
# APPROACH 2: Residual scaling within second pass
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('APPROACH 2: Residual scaling within second pass of third block')
print(f'{\"=\" * 70}', flush=True)

# Use best third block from approach 1 (or default to (15,20))
best_third = (15, 20)
triple_blocks = [(0, 7), best_third, (45, 52)]

for beta3 in [0.3, 0.5, 0.7]:
    def gen_resid(p, bt=triple_blocks, b3=beta3):
        return generate_residual_scaled(p, bt, [1.0, 1.0, 1.0], [1.0, 1.0, b3])
    def gen_resid_long(p, bt=triple_blocks, b3=beta3):
        return generate_residual_scaled(p, bt, [1.0, 1.0, 1.0], [1.0, 1.0, b3], max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen_resid, verbose=False)
    eq_r = run_eq_bench_probe(gen_resid_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    name = f'triple +({best_third[0]},{best_third[1]}) resid_beta={b3}'
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    all_results.append({'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# Also test residual scaling on ALL blocks in the triple
for beta_all in [0.5, 0.7]:
    def gen_resid_all(p, bt=triple_blocks, ba=beta_all):
        return generate_residual_scaled(p, bt, [1.0, 1.0, 1.0], [ba, ba, ba])
    def gen_resid_all_long(p, bt=triple_blocks, ba=beta_all):
        return generate_residual_scaled(p, bt, [1.0, 1.0, 1.0], [ba, ba, ba], max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen_resid_all, verbose=False)
    eq_r = run_eq_bench_probe(gen_resid_all_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    name = f'triple all_resid_beta={ba}'
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    all_results.append({'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# =====================================================================
# APPROACH 3: Mean-shift correction
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('APPROACH 3: Mean-shift correction between blocks')
print(f'{\"=\" * 70}', flush=True)

print('  Collecting seam deltas on calibration prompts...', flush=True)
mean_deltas = collect_seam_deltas(triple_blocks, [1.0, 1.0, 1.0])
print(f'  Got {len(mean_deltas)} seam delta vectors', flush=True)

for corr_scale in [0.1, 0.3, 0.5, 0.7, 1.0]:
    def gen_corr(p, tb=triple_blocks, cs=corr_scale, md=mean_deltas):
        return generate_mean_shift_corrected(p, tb, [1.0, 1.0, 1.0], md, [0, cs, cs])
    def gen_corr_long(p, tb=triple_blocks, cs=corr_scale, md=mean_deltas):
        return generate_mean_shift_corrected(p, tb, [1.0, 1.0, 1.0], md, [0, cs, cs], max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen_corr, verbose=False)
    eq_r = run_eq_bench_probe(gen_corr_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    name = f'triple mean_shift_corr={cs}'
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    all_results.append({'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('GRAND SUMMARY — Deeper Stacking Attempts')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
print(f'Baseline: {baseline:.2f}')
print(f'Best pair (reference): {pair_r[\"combined\"]:.2f}')
print()
for r in sorted_r[:15]:
    beat_pair = '***' if r['combined'] > pair_r['combined'] else '   '
    print(f'{beat_pair} {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f} (math={r[\"math\"]:.4f} eq={r[\"eq\"]:.1f})', flush=True)

any_beats_pair = any(r['combined'] > pair_r['combined'] for r in all_results if 'triple' in r.get('name', ''))
print(f'\\nAny triple beats pair: {\"YES!!!\" if any_beats_pair else \"NO\"}', flush=True)

# Save
os.makedirs('results/data/72b/deeper_stacking', exist_ok=True)
with open('results/data/72b/deeper_stacking/results.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'baseline': baseline,
        'pair_reference': pair_r,
        'all_results': all_results,
        'any_triple_beats_pair': any_beats_pair,
    }, f, indent=2)
print(f'Saved to results/data/72b/deeper_stacking/results.json', flush=True)
"

echo "=== Done at $(date) ==="
