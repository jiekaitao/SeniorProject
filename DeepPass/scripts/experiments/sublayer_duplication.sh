#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sublayer_%j.log
#SBATCH --job-name=deeppass_subl

# Sublayer Duplication Experiment (72B)
#
# Tests whether the duplication benefit comes from attention or FFN sublayers.
# For block (45,52), during the SECOND pass only:
#   1. Duplicate ONLY attention (skip FFN re-computation)
#   2. Duplicate ONLY FFN (skip attention re-computation)
#   3. Both with independent attn_alpha and ffn_alpha
#
# Instead of calling layer(h) for second-pass layers, manually calls
# self_attn and mlp separately with per-sublayer alpha blending.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Sublayer Duplication Experiment ==="
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
print('SUBLAYER DUPLICATION EXPERIMENT')
print('Does the duplication benefit come from attention, FFN, or both?')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

# =====================================================================
# Core: build layer execution order with second-pass tracking
# =====================================================================

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def identify_second_pass_steps(layer_order, blocks):
    \"\"\"Return set of step indices that are second-pass layers, keyed by block.
    Returns dict: block -> list of (step_idx, layer_offset) for second-pass steps.
    \"\"\"
    sorted_blocks = sorted(blocks)
    second_pass = {}
    for block in sorted_blocks:
        i, j = block
        block_layers = list(range(i, j))
        count = {}
        second_pass[block] = []
        for step_idx, layer_idx in enumerate(layer_order):
            if layer_idx in block_layers:
                count[layer_idx] = count.get(layer_idx, 0) + 1
                if count[layer_idx] == 2:
                    offset = layer_idx - i
                    second_pass[block].append((step_idx, offset))
    return second_pass

# =====================================================================
# Sublayer-controlled generation
# =====================================================================

def generate_sublayer(prompt, blocks, sublayer_config, max_new_tokens=64):
    \"\"\"
    Generate with per-sublayer alpha control on second-pass layers.

    sublayer_config: dict mapping (block, layer_offset) -> (attn_alpha, ffn_alpha)
    For first-pass layers, run normally via layer(h).
    For second-pass layers, manually split into attn + FFN with alpha blending.
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)
    second_pass_info = identify_second_pass_steps(layer_order, sorted_blocks)

    # Build a lookup: step_idx -> (block, offset, attn_alpha, ffn_alpha)
    step_to_config = {}
    for block, steps in second_pass_info.items():
        for step_idx, offset in steps:
            key = (block, offset)
            attn_a, ffn_a = sublayer_config.get(key, (1.0, 1.0))
            step_to_config[step_idx] = (block, offset, attn_a, ffn_a)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]

                if step_idx in step_to_config:
                    # Second-pass layer: manual sublayer control
                    _, offset, attn_alpha, ffn_alpha = step_to_config[step_idx]

                    # Attention sublayer
                    residual = h
                    normed = layer.input_layernorm(h)
                    attn_out = layer.self_attn(normed, position_embeddings=pos_embeds, attention_mask=None, use_cache=False)
                    attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
                    h = residual + attn_alpha * attn_out

                    # FFN sublayer
                    residual = h
                    normed = layer.post_attention_layernorm(h)
                    ffn_out = layer.mlp(normed)
                    h = residual + ffn_alpha * ffn_out
                else:
                    # First-pass layer: normal forward
                    out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                    h = out[0] if isinstance(out, tuple) else out

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


def evaluate(blocks, sublayer_config, name):
    gen = lambda p: generate_sublayer(p, blocks, sublayer_config, max_new_tokens=64)
    gen_long = lambda p: generate_sublayer(p, blocks, sublayer_config, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


all_results = []
block = (45, 52)
block_size = 7  # layers 45,46,47,48,49,50,51

# =====================================================================
# TEST 0: Baselines
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 0: Baselines')
print(f'{\"=\" * 70}', flush=True)

# Baseline: no duplication (run first N layers normally)
# Use sublayer_config with empty dict and empty blocks
from layer_duplicator import generate_no_cache
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline_combined = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'  Baseline (no dup): math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={baseline_combined:.2f}', flush=True)
all_results.append({'name': 'baseline_no_dup', 'math': math_base['score'], 'eq': eq_base['score'], 'combined': baseline_combined})

# Full-layer duplication at alpha=1.0 (standard RYS)
full_layer_config = {(block, i): (1.0, 1.0) for i in range(block_size)}
r = evaluate([block], full_layer_config, 'full dup alpha=1.0 (both attn+ffn)')
all_results.append(r)

# Full-layer duplication at alpha=1.15 (known best for this block)
full_115_config = {(block, i): (1.15, 1.15) for i in range(block_size)}
r = evaluate([block], full_115_config, 'full dup alpha=1.15 (known best)')
all_results.append(r)

# =====================================================================
# TEST 1: Sensitivity — attn-only vs ffn-only vs both for each layer
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 1: Per-layer sublayer sensitivity')
print('For each of 7 layers: test attn-only, ffn-only, both')
print(f'{\"=\" * 70}', flush=True)

layer_sensitivity = {}

for offset in range(block_size):
    global_layer = block[0] + offset
    print(f'\n  --- Layer {offset} (global {global_layer}) ---', flush=True)

    # attn-only: attn_alpha=1.0, ffn_alpha=0.0
    cfg = {(block, i): (1.0, 1.0) for i in range(block_size)}
    cfg[(block, offset)] = (1.0, 0.0)
    r = evaluate([block], cfg, f'L{offset} attn-only (ffn_alpha=0)')
    all_results.append(r)
    attn_only_score = r['combined']

    # ffn-only: attn_alpha=0.0, ffn_alpha=1.0
    cfg = {(block, i): (1.0, 1.0) for i in range(block_size)}
    cfg[(block, offset)] = (0.0, 1.0)
    r = evaluate([block], cfg, f'L{offset} ffn-only (attn_alpha=0)')
    all_results.append(r)
    ffn_only_score = r['combined']

    # both disabled: attn_alpha=0.0, ffn_alpha=0.0 (effectively skip this layer's 2nd pass)
    cfg = {(block, i): (1.0, 1.0) for i in range(block_size)}
    cfg[(block, offset)] = (0.0, 0.0)
    r = evaluate([block], cfg, f'L{offset} both disabled (skip 2nd pass)')
    all_results.append(r)
    skip_score = r['combined']

    layer_sensitivity[offset] = {
        'global_layer': global_layer,
        'attn_only': attn_only_score,
        'ffn_only': ffn_only_score,
        'both_disabled': skip_score,
    }

# Summary of per-layer sensitivity
print(f'\n  --- Sublayer sensitivity summary ---', flush=True)
print(f'  {\"Layer\":>6s} {\"Global\":>7s} {\"Attn-only\":>12s} {\"FFN-only\":>12s} {\"Disabled\":>12s} {\"Dominant\":>10s}', flush=True)
print(f'  ' + '-' * 65, flush=True)
for offset in range(block_size):
    s = layer_sensitivity[offset]
    attn = s['attn_only']
    ffn = s['ffn_only']
    skip = s['both_disabled']
    dominant = 'attn' if attn > ffn else 'ffn' if ffn > attn else 'tie'
    print(f'  {offset:>6d} {s[\"global_layer\"]:>7d} {attn:>12.2f} {ffn:>12.2f} {skip:>12.2f} {dominant:>10s}', flush=True)

# =====================================================================
# TEST 2: Uniform sublayer ablation across all 7 layers
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 2: Uniform sublayer modes across entire block')
print(f'{\"=\" * 70}', flush=True)

# All layers attn-only
attn_only_cfg = {(block, i): (1.0, 0.0) for i in range(block_size)}
r = evaluate([block], attn_only_cfg, 'ALL layers attn-only (ffn_alpha=0)')
all_results.append(r)

# All layers ffn-only
ffn_only_cfg = {(block, i): (0.0, 1.0) for i in range(block_size)}
r = evaluate([block], ffn_only_cfg, 'ALL layers ffn-only (attn_alpha=0)')
all_results.append(r)

# Attn at higher alpha, FFN at lower (test relative importance)
for attn_a, ffn_a in [(1.3, 0.7), (0.7, 1.3), (1.5, 0.5), (0.5, 1.5), (1.15, 0.0), (0.0, 1.15)]:
    cfg = {(block, i): (attn_a, ffn_a) for i in range(block_size)}
    r = evaluate([block], cfg, f'ALL layers attn_a={attn_a} ffn_a={ffn_a}')
    all_results.append(r)

# =====================================================================
# TEST 3: Best sublayer config via coordinate descent
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 3: Coordinate descent over attn_alpha and ffn_alpha per layer')
print(f'{\"=\" * 70}', flush=True)

# Determine which sublayer is dominant per layer from TEST 1
# Start coordinate descent with informed initial values
best_cfg = {}
for offset in range(block_size):
    s = layer_sensitivity[offset]
    if s['attn_only'] > s['ffn_only']:
        # Attention-dominant: start with higher attn_alpha
        best_cfg[(block, offset)] = (1.15, 0.5)
    else:
        # FFN-dominant: start with higher ffn_alpha
        best_cfg[(block, offset)] = (0.5, 1.15)

# Evaluate the initial config
cfg_str = ', '.join(f'L{i}=({best_cfg[(block,i)][0]:.1f},{best_cfg[(block,i)][1]:.1f})' for i in range(block_size))
r = evaluate([block], best_cfg, f'initial informed: [{cfg_str}]')
all_results.append(r)
best_combined = r['combined']

# Coordinate descent
attn_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.15, 1.3, 1.5]
ffn_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.15, 1.3, 1.5]

for iteration in range(3):
    improved = False
    print(f'\n  --- Coordinate descent iteration {iteration} ---', flush=True)

    for offset in range(block_size):
        current_attn, current_ffn = best_cfg[(block, offset)]

        # Sweep attn_alpha for this layer (hold ffn_alpha fixed)
        for a in attn_grid:
            if a == current_attn:
                continue
            trial_cfg = dict(best_cfg)
            trial_cfg[(block, offset)] = (a, current_ffn)
            r = evaluate([block], trial_cfg, f'iter{iteration} L{offset} attn_a={a} ffn_a={current_ffn}')
            all_results.append(r)
            if r['combined'] > best_combined:
                best_combined = r['combined']
                best_cfg = dict(trial_cfg)
                current_attn = a
                improved = True
                print(f'    >>> Improved! L{offset} attn_a={a} -> {best_combined:.2f}', flush=True)

        # Sweep ffn_alpha for this layer (hold attn_alpha at current best)
        current_attn = best_cfg[(block, offset)][0]
        for f in ffn_grid:
            if f == best_cfg[(block, offset)][1]:
                continue
            trial_cfg = dict(best_cfg)
            trial_cfg[(block, offset)] = (current_attn, f)
            r = evaluate([block], trial_cfg, f'iter{iteration} L{offset} attn_a={current_attn} ffn_a={f}')
            all_results.append(r)
            if r['combined'] > best_combined:
                best_combined = r['combined']
                best_cfg = dict(trial_cfg)
                improved = True
                print(f'    >>> Improved! L{offset} ffn_a={f} -> {best_combined:.2f}', flush=True)

    if not improved:
        print(f'  Converged at iteration {iteration}', flush=True)
        break

best_cfg_str = ', '.join(f'L{i}=({best_cfg[(block,i)][0]:.2f},{best_cfg[(block,i)][1]:.2f})' for i in range(block_size))
print(f'\n  Best sublayer config: [{best_cfg_str}]', flush=True)
print(f'  Best combined score: {best_combined:.2f}', flush=True)

# =====================================================================
# TEST 4: Compare best sublayer config vs best full-layer alpha
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 4: Final comparison — sublayer vs full-layer')
print(f'{\"=\" * 70}', flush=True)

# Re-evaluate the best sublayer config (for clean comparison)
r_sublayer = evaluate([block], best_cfg, f'BEST sublayer config')
all_results.append(r_sublayer)

# Best full-layer alpha=1.15 (already tested, but re-run for clean comparison)
r_full = evaluate([block], full_115_config, f'BEST full-layer alpha=1.15')
all_results.append(r_full)

print(f'\n  Full-layer alpha=1.15:  combined={r_full[\"combined\"]:.2f}', flush=True)
print(f'  Sublayer optimized:     combined={r_sublayer[\"combined\"]:.2f}', flush=True)
delta = r_sublayer['combined'] - r_full['combined']
print(f'  Delta (sublayer - full): {delta:+.2f}', flush=True)

# =====================================================================
# TEST 5: Apply best sublayer config to pair (0,7)+(45,52)
# =====================================================================
print(f'\n{\"=\" * 70}')
print('TEST 5: Sublayer config on pair (0,7)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

pair_blocks = [(0, 7), (45, 52)]

# Transfer best sublayer config from (45,52), use standard for (0,7)
pair_cfg = {}
for i in range(7):
    pair_cfg[((0, 7), i)] = (1.0, 1.0)  # standard for first block
for i in range(7):
    pair_cfg[((45, 52), i)] = best_cfg.get(((45, 52), i), best_cfg.get((block, i), (1.0, 1.0)))

r = evaluate(pair_blocks, pair_cfg, 'pair: (0,7)@standard + (45,52)@sublayer_opt')
all_results.append(r)

# Also test attn-only and ffn-only on the pair's (45,52)
pair_attn_cfg = {}
for i in range(7):
    pair_attn_cfg[((0, 7), i)] = (1.0, 1.0)
for i in range(7):
    pair_attn_cfg[((45, 52), i)] = (1.0, 0.0)

r = evaluate(pair_blocks, pair_attn_cfg, 'pair: (0,7)@standard + (45,52)@attn-only')
all_results.append(r)

pair_ffn_cfg = {}
for i in range(7):
    pair_ffn_cfg[((0, 7), i)] = (1.0, 1.0)
for i in range(7):
    pair_ffn_cfg[((45, 52), i)] = (0.0, 1.0)

r = evaluate(pair_blocks, pair_ffn_cfg, 'pair: (0,7)@standard + (45,52)@ffn-only')
all_results.append(r)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}', flush=True)

sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for rank, r in enumerate(sorted_r[:25], 1):
    print(f'  #{rank:2d} {r[\"name\"]:65s}: combined={r[\"combined\"]:.2f} (math={r[\"math\"]:.4f} eq={r[\"eq\"]:.1f})', flush=True)

# Per-layer summary: which sublayer matters more?
print(f'\n  Per-layer sublayer dominance:', flush=True)
attn_wins = 0
ffn_wins = 0
for offset in range(block_size):
    s = layer_sensitivity[offset]
    dominant = 'ATTN' if s['attn_only'] > s['ffn_only'] else 'FFN'
    margin = abs(s['attn_only'] - s['ffn_only'])
    if dominant == 'ATTN':
        attn_wins += 1
    else:
        ffn_wins += 1
    print(f'    Layer {offset} (global {s[\"global_layer\"]}): {dominant} dominant (margin={margin:.2f})', flush=True)
print(f'  Overall: {attn_wins} attn-dominant, {ffn_wins} ffn-dominant layers', flush=True)

# Save
os.makedirs('results/data/72b/sublayer', exist_ok=True)
outpath = 'results/data/72b/sublayer/results.json'

# Convert tuple keys to strings for JSON
json_cfg = {}
for k, v in best_cfg.items():
    block_k, offset_k = k
    key_str = f'({block_k[0]},{block_k[1]})_L{offset_k}'
    json_cfg[key_str] = {'attn_alpha': v[0], 'ffn_alpha': v[1]}

with open(outpath, 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'model': 'calme-2.1-qwen2-72b',
        'block': list(block),
        'block_size': block_size,
        'baseline_combined': baseline_combined,
        'best_sublayer_config': json_cfg,
        'best_sublayer_combined': best_combined,
        'layer_sensitivity': {str(k): v for k, v in layer_sensitivity.items()},
        'all_results': all_results,
    }, f, indent=2)
print(f'\nSaved to {outpath}', flush=True)
print('DONE', flush=True)
"

echo "=== Done at $(date) ==="
