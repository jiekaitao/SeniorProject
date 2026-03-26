#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_resid_scale_%j.log
#SBATCH --job-name=deeppass_rscal

# Residual scaling within second pass — mechanistically distinct from alpha blending
# Alpha blending operates at the SEAM (between passes)
# Residual scaling operates WITHIN each layer of the second pass
# h = h_in + beta * (layer(h_in) - h_in) for each layer during 2nd pass

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Residual Scaling Within Second Pass ==="
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
print('RESIDUAL SCALING WITHIN SECOND PASS')
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

def find_second_pass_steps(layer_order, blocks):
    \"\"\"For each block, find which steps in layer_order are the SECOND pass.\"\"\"
    second_pass = set()
    for block in sorted(blocks):
        i, j = block
        block_layers = list(range(i, j))
        block_size = j - i
        count = {}
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:  # second occurrence
                    second_pass.add(step)
    return second_pass

def generate_residual_scaled(prompt, blocks, betas, max_new_tokens=64):
    \"\"\"Generate with residual scaling during second pass.
    betas: dict mapping block -> beta for residual scaling.
    During the second pass of a block, each layer's output is:
      h = h_in + beta * (layer(h_in) - h_in)
    instead of h = layer(h_in)
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)

    # Map each second-pass step to its beta
    step_betas = {}
    for block in sorted_blocks:
        i, j = block
        block_layers = list(range(i, j))
        count = {}
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:
                    step_betas[step] = betas.get(block, 1.0)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                h_before = h.clone()
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h_after = out[0] if isinstance(out, tuple) else out

                if step_idx in step_betas:
                    beta = step_betas[step_idx]
                    h = h_before + beta * (h_after - h_before)
                else:
                    h = h_after

            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def evaluate(blocks, betas, name):
    gen = lambda p: generate_residual_scaled(p, blocks, betas, max_new_tokens=64)
    gen_long = lambda p: generate_residual_scaled(p, blocks, betas, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# =====================================================================
# Test 1: Residual scaling on PAIR (0,7)+(45,52)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 1: Residual scaling on pair')
print(f'{\"=\" * 70}', flush=True)

pair = [(0, 7), (45, 52)]
for beta in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    betas = {(0,7): beta, (45,52): beta}
    r = evaluate(pair, betas, f'pair both@beta={beta}')
    all_results.append(r)

# Asymmetric: scale only early block
for beta in [0.5, 0.7, 0.9]:
    betas = {(0,7): beta, (45,52): 1.0}
    r = evaluate(pair, betas, f'pair (0,7)@beta={beta} (45,52)@1.0')
    all_results.append(r)

# =====================================================================
# Test 2: Residual scaling on TRIPLE with (15,20) as third block
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 2: Residual scaling on triple (0,7)+(15,20)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

triple = [(0, 7), (15, 20), (45, 52)]

# Scale only third block
for beta3 in [0.1, 0.2, 0.3, 0.5, 0.7]:
    betas = {(0,7): 1.0, (15,20): beta3, (45,52): 1.0}
    r = evaluate(triple, betas, f'triple (15,20)@beta={beta3}')
    all_results.append(r)

# Scale all blocks
for beta_all in [0.3, 0.5, 0.7, 0.9]:
    betas = {(0,7): beta_all, (15,20): beta_all, (45,52): beta_all}
    r = evaluate(triple, betas, f'triple all@beta={beta_all}')
    all_results.append(r)

# =====================================================================
# Test 3: Residual scaling on triple with (20,27)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 3: Residual scaling on triple (0,7)+(20,27)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

triple2 = [(0, 7), (20, 27), (45, 52)]
for beta3 in [0.1, 0.2, 0.3, 0.5]:
    betas = {(0,7): 1.0, (20,27): beta3, (45,52): 1.0}
    r = evaluate(triple2, betas, f'triple2 (20,27)@beta={beta3}')
    all_results.append(r)

# =====================================================================
# Test 4: COMBINED — alpha blending + residual scaling
# Use alpha=0.1 at seam AND residual scaling within second pass
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('TEST 4: Combined alpha + residual scaling')
print('Can we get even better than alpha=0.1 alone?')
print(f'{\"=\" * 70}', flush=True)

# This needs a combined generator
def generate_combined(prompt, blocks, seam_alphas, resid_betas, max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)

    # Seam positions
    seams = []
    for block in sorted_blocks:
        last_layer = block[1] - 1
        occurrences = [step for step, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) >= 2:
            seams.append((occurrences[0], occurrences[1]))
        else:
            seams.append(None)
    sorted_seam_alphas = [seam_alphas.get(b, 1.0) for b in sorted_blocks]

    # Second pass step betas
    step_betas = {}
    for block in sorted_blocks:
        i, j = block
        block_layers = list(range(i, j))
        count = {}
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:
                    step_betas[step] = resid_betas.get(block, 1.0)

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
                if step_idx in step_betas:
                    beta = step_betas[step_idx]
                    h = h_before + beta * (h_after - h_before)
                else:
                    h = h_after
                for si, seam in enumerate(seams):
                    if seam is None: continue
                    if step_idx == seam[0]: saved_h1[si] = h.clone()
                    if step_idx == seam[1] and si in saved_h1:
                        a = sorted_seam_alphas[si]
                        h = saved_h1[si] + a * (h - saved_h1[si])
                        del saved_h1[si]
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

# Best known triple: alpha=0.1 on third block. Can residual scaling improve further?
for combo in [
    ('alpha=0.1+beta=0.5', {(15,20): 0.1}, {(15,20): 0.5}),
    ('alpha=0.1+beta=0.3', {(15,20): 0.1}, {(15,20): 0.3}),
    ('alpha=0.2+beta=0.5', {(15,20): 0.2}, {(15,20): 0.5}),
    ('alpha=1.0+beta=0.1', {(15,20): 1.0}, {(15,20): 0.1}),
    ('alpha=1.0+beta=0.2', {(15,20): 1.0}, {(15,20): 0.2}),
]:
    name, sa, rb = combo
    full_sa = {(0,7): 1.0, (45,52): 1.0}; full_sa.update(sa)
    full_rb = {(0,7): 1.0, (45,52): 1.0}; full_rb.update(rb)
    gen = lambda p, s=full_sa, r=full_rb: generate_combined(p, triple, s, r, max_new_tokens=64)
    gen_long = lambda p, s=full_sa, r=full_rb: generate_combined(p, triple, s, r, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  combined {name:45s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    all_results.append({'name': f'combined {name}', 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    print(f'  {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f}', flush=True)

os.makedirs('results/data/72b/residual_scaling', exist_ok=True)
with open('results/data/72b/residual_scaling/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
print(f'\\nSaved to results/data/72b/residual_scaling/results.json', flush=True)
"

echo "=== Done at $(date) ==="
