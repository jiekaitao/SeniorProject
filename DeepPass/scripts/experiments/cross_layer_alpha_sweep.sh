#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_xlyr_alpha_%j.log
#SBATCH --job-name=deeppass_xlya

# Follow-up: alpha sweep on best cross-layer configs
# Also test: combining cross-layer with standard pair duplication

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Cross-Layer Alpha Sweep ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('CROSS-LAYER ALPHA SWEEP + COMBINATION')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

def generate_cross_layer(prompt, first_block, second_block_weights, alpha=1.0, max_new_tokens=64):
    i, j = first_block
    a, b = second_block_weights
    order = list(range(j)) + list(range(i, j)) + list(range(a, b)) + list(range(j, N))
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    first_pass_end = j - 1
    second_pass_end = j + (b - a) - 1
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
                    h = h_after_first + alpha * (h - h_after_first)
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def evaluate_cross(first, second, alpha, name):
    gen = lambda p: generate_cross_layer(p, first, second, alpha, 64)
    gen_long = lambda p: generate_cross_layer(p, first, second, alpha, 128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  {name:55s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# =====================================================================
# 1. Alpha sweep on best cross-layer: (45,52) -> (20,27)
# =====================================================================
print(f'\\n--- Alpha sweep: (45,52)->(20,27) ---', flush=True)
for alpha in [0.3, 0.5, 0.7, 0.9, 1.0, 1.15, 1.25, 1.5]:
    r = evaluate_cross((45,52), (20,27), alpha, f'(45,52)->(20,27) @{alpha}')
    all_results.append(r)

# =====================================================================
# 2. Alpha sweep on (45,52) -> (5,12)
# =====================================================================
print(f'\\n--- Alpha sweep: (45,52)->(5,12) ---', flush=True)
for alpha in [0.5, 0.7, 1.0, 1.15, 1.25]:
    r = evaluate_cross((45,52), (5,12), alpha, f'(45,52)->(5,12) @{alpha}')
    all_results.append(r)

# =====================================================================
# 3. Combined: standard pair dup + cross-layer
# First: (0,7) self-dup + (45,52) cross-dup with (20,27)
# =====================================================================
print(f'\\n--- Combination: pair dup + cross-layer ---', flush=True)

def generate_combined_cross(prompt, max_new_tokens=64):
    \"\"\"(0,7) self-dup at alpha=0.9 + (45,52) cross-dup with (20,27) at alpha=1.0\"\"\"
    # Layer order: [0..6, 0..6(dup), 7..51, 45..51(1st pass), 20..26(cross 2nd pass), 52..79]
    order = (list(range(7)) + list(range(0, 7)) +  # (0,7) self-dup
             list(range(7, 52)) +                    # layers 7-51
             list(range(45, 52)) +                    # (45,52) first pass
             list(range(20, 27)) +                    # (20,27) cross second pass
             list(range(52, N)))                      # remaining

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # Seam positions
    # (0,7) seam: first pass ends at step 6, second pass ends at step 13
    # (45,52) seam: first pass ends at step 13+45=58(?), second pass ends later
    # Actually need to track more carefully with occurrence counting
    seam_07_first = 6   # last step of (0,7) first pass
    seam_07_second = 13  # last step of (0,7) second pass (dup)
    # After that, 7..51 = 45 layers (steps 14..58)
    # Then 45..51 first pass = 7 layers (steps 59..65)
    seam_45_first = 13 + 45 + 7 - 1  # = 64
    # Then 20..26 cross pass = 7 layers (steps 66..72)
    seam_45_second = 13 + 45 + 7 + 7 - 1  # = 71

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            h_seam_07 = None
            h_seam_45 = None
            for step_idx, layer_idx in enumerate(order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                # (0,7) seam
                if step_idx == seam_07_first:
                    h_seam_07 = h.clone()
                if step_idx == seam_07_second and h_seam_07 is not None:
                    h = h_seam_07 + 0.9 * (h - h_seam_07)  # alpha=0.9
                # (45,52) seam
                if step_idx == seam_45_first:
                    h_seam_45 = h.clone()
                if step_idx == seam_45_second and h_seam_45 is not None:
                    h = h_seam_45 + 1.0 * (h - h_seam_45)  # alpha=1.0
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

gen_combo = lambda p: generate_combined_cross(p, 64)
gen_combo_long = lambda p: generate_combined_cross(p, 128)
t0 = time.time()
math_r = run_math_probe(gen_combo, verbose=False)
eq_r = run_eq_bench_probe(gen_combo_long, verbose=False)
combined = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'  COMBO (0,7)@0.9 self + (45,52)->(20,27)@1.0   : math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}', flush=True)
all_results.append({'name': 'COMBO pair+cross', 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:10]:
    print(f'  {r[\"name\"]:55s}: combined={r[\"combined\"]:.2f}', flush=True)

os.makedirs('results/data/72b/cross_layer', exist_ok=True)
with open('results/data/72b/cross_layer/alpha_sweep_results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
print(f'Saved', flush=True)
"

echo "=== Done at $(date) ==="
