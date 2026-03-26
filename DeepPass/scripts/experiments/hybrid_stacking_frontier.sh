#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_hybrid_stack_%j.log
#SBATCH --job-name=deeppass_hstack

# Compute-Normalized Stacking Frontier (72B)
#
# Compare full duplication, attention-only duplication, and hybrid stacks
# at roughly similar FLOP budgets. Plots the Pareto frontier of
# combined score vs extra FLOPs.
#
# Key question: can attention-only stacking achieve comparable gains
# at ~1/3 the FLOP cost of full duplication?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Hybrid Stacking Frontier ==="
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
print('COMPUTE-NORMALIZED STACKING FRONTIER')
print('Full vs Attention-Only vs Hybrid at matched FLOP budgets')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)

# =====================================================================
# FLOP estimation
# For Qwen2-72B: hidden=8192, intermediate=24576, num_heads=64, kv_heads=8
# Per-layer FLOPs (approx per token):
#   Attention: 4*h*h + 2*h*kv_h = 4*8192^2 + 2*8192*(8192*8/64) ~= 277M
#   FFN: 3*h*intermediate = 3*8192*24576 ~= 604M
#   Total per layer: ~881M
# Attention is ~31% of a full layer, FFN is ~69%
# =====================================================================

FLOPS_PER_LAYER = 1.0   # normalize: 1 full layer = 1.0 unit
ATTN_FRAC = 0.31        # attention fraction of layer FLOPs
FFN_FRAC = 0.69         # FFN fraction of layer FLOPs

# =====================================================================
# Core: multi-block layer order and second-pass tracking
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
    \"\"\"Return dict: block -> list of (step_idx, layer_offset) for second-pass steps.\"\"\"
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
# Generation with per-sublayer alpha control
# =====================================================================

def generate_sublayer(prompt, blocks, sublayer_config, max_new_tokens=64):
    \"\"\"
    Generate with per-sublayer (attn_alpha, ffn_alpha) control on second-pass layers.
    sublayer_config: dict mapping (block_tuple, layer_offset) -> (attn_alpha, ffn_alpha)
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)
    second_pass_info = identify_second_pass_steps(layer_order, sorted_blocks)

    # Build step -> config lookup
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


# =====================================================================
# Generation with whisper alpha (full-layer alpha blending at seam)
# For multi-block with different alphas
# =====================================================================

def generate_whisper(prompt, block_alphas, max_new_tokens=64):
    \"\"\"
    Generate with full-layer alpha blending.
    block_alphas: dict mapping block_tuple -> alpha (applied at block seam).
    h_patched = h1 + alpha * (h2 - h1)
    \"\"\"
    blocks = list(block_alphas.keys())
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)

    # Find seam positions for each block
    seam_info = {}
    for block in sorted_blocks:
        last_layer = block[1] - 1
        occurrences = [s for s, l in enumerate(layer_order) if l == last_layer]
        if len(occurrences) >= 2:
            seam_info[block] = (occurrences[0], occurrences[1])

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_first_pass = {}
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                # Check if this is a first-pass end for any block
                for block, (first_end, second_end) in seam_info.items():
                    if step_idx == first_end:
                        h_first_pass[block] = h.clone()
                    if step_idx == second_end and block in h_first_pass:
                        alpha = block_alphas[block]
                        h1 = h_first_pass[block]
                        h = h1 + alpha * (h - h1)

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


# =====================================================================
# Evaluation helper
# =====================================================================

def evaluate(gen_fn, name, extra_flops):
    gen = lambda p: gen_fn(p, 64)
    gen_long = lambda p: gen_fn(p, 128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} '
          f'combined={combined:.2f} flops={extra_flops:.2f} ({elapsed:.0f}s)', flush=True)
    return {
        'name': name,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined,
        'extra_flops': extra_flops,
        'elapsed': elapsed,
    }


# =====================================================================
# Configuration definitions
# =====================================================================

configs = []

# --- Config 0: Baseline (no duplication) ---
print(f'\n{\"=\" * 70}')
print('CONFIG 0: Baseline (no duplication)')
print(f'{\"=\" * 70}', flush=True)

from layer_duplicator import generate_no_cache
def gen_baseline(p, max_t):
    return generate_no_cache(model, tokenizer, p, max_new_tokens=max_t)

r = evaluate(gen_baseline, 'baseline (no dup)', 0.0)
configs.append(r)
baseline_combined = r['combined']

# --- Config 1: Single block attention-only (45,52) ---
print(f'\n{\"=\" * 70}')
print('CONFIG 1: Single block attention-only (45,52)')
print(f'{\"=\" * 70}', flush=True)

block_45_52 = (45, 52)
block_size_45 = 7
attn_only_cfg_45 = {(block_45_52, i): (1.0, 0.0) for i in range(block_size_45)}
extra_flops_1 = block_size_45 * ATTN_FRAC  # ~2.17

def gen_1(p, max_t):
    return generate_sublayer(p, [block_45_52], attn_only_cfg_45, max_new_tokens=max_t)

r = evaluate(gen_1, 'single attn-only (45,52)', extra_flops_1)
configs.append(r)

# --- Config 2: Single block full dup (45,52) ---
print(f'\n{\"=\" * 70}')
print('CONFIG 2: Single block full dup (45,52)')
print(f'{\"=\" * 70}', flush=True)

full_cfg_45 = {(block_45_52, i): (1.0, 1.0) for i in range(block_size_45)}
extra_flops_2 = block_size_45 * FLOPS_PER_LAYER  # 7.0

def gen_2(p, max_t):
    return generate_sublayer(p, [block_45_52], full_cfg_45, max_new_tokens=max_t)

r = evaluate(gen_2, 'single full dup (45,52)', extra_flops_2)
configs.append(r)

# --- Config 3: Pair attention-only (0,7)+(45,52) ---
print(f'\n{\"=\" * 70}')
print('CONFIG 3: Pair attention-only (0,7)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

block_0_7 = (0, 7)
block_size_0 = 7
pair_blocks = [block_0_7, block_45_52]
pair_attn_cfg = {}
for i in range(block_size_0):
    pair_attn_cfg[(block_0_7, i)] = (1.0, 0.0)
for i in range(block_size_45):
    pair_attn_cfg[(block_45_52, i)] = (1.0, 0.0)
extra_flops_3 = (block_size_0 + block_size_45) * ATTN_FRAC  # ~4.34

def gen_3(p, max_t):
    return generate_sublayer(p, pair_blocks, pair_attn_cfg, max_new_tokens=max_t)

r = evaluate(gen_3, 'pair attn-only (0,7)+(45,52)', extra_flops_3)
configs.append(r)

# --- Config 4: Pair full dup (0,7)+(45,52) ---
print(f'\n{\"=\" * 70}')
print('CONFIG 4: Pair full dup (0,7)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

pair_full_cfg = {}
for i in range(block_size_0):
    pair_full_cfg[(block_0_7, i)] = (1.0, 1.0)
for i in range(block_size_45):
    pair_full_cfg[(block_45_52, i)] = (1.0, 1.0)
extra_flops_4 = (block_size_0 + block_size_45) * FLOPS_PER_LAYER  # 14.0

def gen_4(p, max_t):
    return generate_sublayer(p, pair_blocks, pair_full_cfg, max_new_tokens=max_t)

r = evaluate(gen_4, 'pair full dup (0,7)+(45,52)', extra_flops_4)
configs.append(r)

# --- Config 5: Triple attention-only (0,7)+(20,27)+(45,52) ---
print(f'\n{\"=\" * 70}')
print('CONFIG 5: Triple attention-only (0,7)+(20,27)+(45,52)')
print(f'{\"=\" * 70}', flush=True)

block_20_27 = (20, 27)
block_size_20 = 7
triple_blocks = [block_0_7, block_20_27, block_45_52]
triple_attn_cfg = {}
for b, bs in [(block_0_7, block_size_0), (block_20_27, block_size_20), (block_45_52, block_size_45)]:
    for i in range(bs):
        triple_attn_cfg[(b, i)] = (1.0, 0.0)
extra_flops_5 = (block_size_0 + block_size_20 + block_size_45) * ATTN_FRAC  # ~6.51

def gen_5(p, max_t):
    return generate_sublayer(p, triple_blocks, triple_attn_cfg, max_new_tokens=max_t)

r = evaluate(gen_5, 'triple attn-only (0,7)+(20,27)+(45,52)', extra_flops_5)
configs.append(r)

# --- Config 6: Triple full dup with whisper alphas ---
print(f'\n{\"=\" * 70}')
print('CONFIG 6: Triple full dup with whisper (0,7)@0.9+(20,27)@0.15+(45,52)@1.0')
print(f'{\"=\" * 70}', flush=True)

triple_whisper_alphas = {
    block_0_7: 0.9,
    block_20_27: 0.15,
    block_45_52: 1.0,
}
extra_flops_6 = (block_size_0 + block_size_20 + block_size_45) * FLOPS_PER_LAYER  # 21.0

def gen_6(p, max_t):
    return generate_whisper(p, triple_whisper_alphas, max_new_tokens=max_t)

r = evaluate(gen_6, 'triple whisper (0,7)@0.9+(20,27)@0.15+(45,52)@1.0', extra_flops_6)
configs.append(r)

# --- Config 7: Hybrid: (45,52) attn-only + (0,7) full dup ---
print(f'\n{\"=\" * 70}')
print('CONFIG 7: Hybrid: (45,52) attn-only + (0,7) full dup')
print(f'{\"=\" * 70}', flush=True)

hybrid_cfg_7 = {}
for i in range(block_size_0):
    hybrid_cfg_7[(block_0_7, i)] = (1.0, 1.0)   # full dup
for i in range(block_size_45):
    hybrid_cfg_7[(block_45_52, i)] = (1.0, 0.0)  # attn-only
extra_flops_7 = block_size_0 * FLOPS_PER_LAYER + block_size_45 * ATTN_FRAC  # 7 + 2.17 = 9.17

def gen_7(p, max_t):
    return generate_sublayer(p, pair_blocks, hybrid_cfg_7, max_new_tokens=max_t)

r = evaluate(gen_7, 'hybrid: (0,7)full + (45,52)attn-only', extra_flops_7)
configs.append(r)

# --- Config 8: Hybrid per-sublayer: (45,52) with selective FFN suppression ---
print(f'\n{\"=\" * 70}')
print('CONFIG 8: Hybrid per-sublayer: (45,52) attn=1.15, FFN selective')
print('  FFN=0 on L2/L5/L6 (harmful), FFN=1.0 on L1/L3/L4 (neutral/helpful)')
print(f'{\"=\" * 70}', flush=True)

# L0 (layer 45): FFN neutral -> keep at 1.0
# L1 (layer 46): FFN slight harm from removing -> keep at 1.0
# L2 (layer 47): FFN VERY harmful -> suppress to 0.0
# L3 (layer 48): FFN moderate harm -> keep at 1.0
# L4 (layer 49): FFN very harmful -> keep at 1.0 (close call, but interacts with L3)
# L5 (layer 50): FFN most harmful -> suppress to 0.0
# L6 (layer 51): FFN very harmful -> suppress to 0.0
hybrid_cfg_8 = {}
ffn_suppress_layers = {2, 5, 6}  # layer offsets to suppress FFN
ffn_keep_layers = {0, 1, 3, 4}   # layer offsets to keep FFN
for offset in range(block_size_45):
    if offset in ffn_suppress_layers:
        hybrid_cfg_8[(block_45_52, offset)] = (1.15, 0.0)
    else:
        hybrid_cfg_8[(block_45_52, offset)] = (1.15, 1.0)

# FLOP cost: 7 * attn (at alpha=1.15, FLOPs same as 1.0) + 4 * FFN
extra_flops_8 = block_size_45 * ATTN_FRAC + len(ffn_keep_layers) * FFN_FRAC  # 2.17 + 2.76 = 4.93

def gen_8(p, max_t):
    return generate_sublayer(p, [block_45_52], hybrid_cfg_8, max_new_tokens=max_t)

r = evaluate(gen_8, 'hybrid: (45,52) attn=1.15, ffn=0@L2/L5/L6', extra_flops_8)
configs.append(r)

# =====================================================================
# SUMMARY: Pareto frontier
# =====================================================================

print(f'\n{\"=\" * 70}')
print('PARETO FRONTIER: Combined Score vs Extra FLOPs')
print(f'{\"=\" * 70}', flush=True)

# Sort by extra FLOPs
sorted_configs = sorted(configs, key=lambda x: x['extra_flops'])

print(f'  {\"Config\":65s} {\"FLOPs\":>7s} {\"Combined\":>10s} {\"Delta\":>8s} {\"Eff\":>10s}', flush=True)
print(f'  ' + '-' * 105, flush=True)

for c in sorted_configs:
    delta = c['combined'] - baseline_combined
    efficiency = delta / max(c['extra_flops'], 0.01)
    print(f'  {c[\"name\"]:65s} {c[\"extra_flops\"]:7.2f} {c[\"combined\"]:10.2f} '
          f'{delta:+8.2f} {efficiency:+10.3f}', flush=True)

# Compute Pareto frontier
print(f'\n  --- Pareto-Optimal Configs ---', flush=True)
pareto = []
best_so_far = -float('inf')
for c in sorted_configs:
    if c['combined'] > best_so_far:
        pareto.append(c)
        best_so_far = c['combined']
        print(f'    PARETO: {c[\"name\"]:55s} flops={c[\"extra_flops\"]:.2f} combined={c[\"combined\"]:.2f}', flush=True)

# Dominated configs
dominated = [c for c in sorted_configs if c not in pareto]
if dominated:
    print(f'\n  --- Dominated (NOT on Pareto frontier) ---', flush=True)
    for c in dominated:
        # Find which Pareto config dominates it
        dominator = None
        for p in pareto:
            if p['extra_flops'] <= c['extra_flops'] and p['combined'] >= c['combined']:
                dominator = p
                break
        dom_name = dominator['name'][:35] + '...' if dominator and len(dominator['name']) > 35 else (dominator['name'] if dominator else 'N/A')
        print(f'    {c[\"name\"]:55s} (dominated by: {dom_name})', flush=True)

# Best efficiency (score per FLOP)
print(f'\n  --- Best Efficiency (delta per extra FLOP) ---', flush=True)
with_flops = [c for c in configs if c['extra_flops'] > 0]
if with_flops:
    efficiencies = [(c, (c['combined'] - baseline_combined) / c['extra_flops']) for c in with_flops]
    efficiencies.sort(key=lambda x: x[1], reverse=True)
    for rank, (c, eff) in enumerate(efficiencies[:5], 1):
        print(f'    #{rank}: {c[\"name\"]:55s} eff={eff:+.3f}/FLOP', flush=True)

# =====================================================================
# Save
# =====================================================================

os.makedirs('results/data/72b/mechanistic', exist_ok=True)
outpath = 'results/data/72b/mechanistic/hybrid_stacking_frontier.json'

with open(outpath, 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'model': 'calme-2.1-qwen2-72b',
        'num_layers': N,
        'flop_model': {
            'attn_fraction': ATTN_FRAC,
            'ffn_fraction': FFN_FRAC,
            'unit': '1.0 = one full layer',
        },
        'baseline_combined': baseline_combined,
        'configs': configs,
        'pareto_frontier': [c['name'] for c in pareto],
    }, f, indent=2)
print(f'\nSaved to {outpath}', flush=True)
print('DONE', flush=True)
"

echo "=== Done at $(date) ==="
