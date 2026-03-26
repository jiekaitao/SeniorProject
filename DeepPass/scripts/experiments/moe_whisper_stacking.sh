#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_moe_whisper_%j.log
#SBATCH --job-name=deeppass_moew

# Whisper-alpha stacking on Qwen3-30B-A3B (MoE model)
# From basic MoE experiment:
#   Baseline: combined=27.76
#   Best single (8,9): combined=40.42
#   Pairs at alpha=1.0 don't beat single
#
# Test whisper-alpha pairs and triples, plus alpha tuning on best single.

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== MoE Whisper-Alpha Stacking: Qwen3-30B-A3B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/Qwen3-30B-A3B'

print('=' * 70)
print('WHISPER-ALPHA STACKING ON QWEN3-30B-A3B (MoE)')
print('Best single (8,9): combined=40.42')
print('Pairs at alpha=1.0 do not beat single — try whisper-alpha')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)
print(f'VRAM: ~{torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)

# Verify Qwen3 MoE uses the same rotary_emb interface
print(f'inner.embed_tokens: {type(inner.embed_tokens).__name__}', flush=True)
print(f'inner.rotary_emb: {type(inner.rotary_emb).__name__}', flush=True)
print(f'inner.norm: {type(inner.norm).__name__}', flush=True)
print(f'model.lm_head: {type(model.lm_head).__name__}', flush=True)

# Quick test: verify rotary_emb call signature
test_h = inner.embed_tokens(torch.tensor([[1]], device=device))
test_pos = torch.arange(1, device=device).unsqueeze(0)
test_pe = inner.rotary_emb(test_h, test_pos)
pe_len = len(test_pe) if isinstance(test_pe, tuple) else 'N/A'
print(f'rotary_emb output type: {type(test_pe).__name__}, len={pe_len}', flush=True)
# Verify layer call signature
test_out = original_layers[0](test_h, position_embeddings=test_pe, use_cache=False)
print(f'Layer output type: {type(test_out).__name__}', flush=True)
del test_h, test_pos, test_pe, test_out
print('Rotary embedding interface verified OK', flush=True)

# =====================================================================
# Core functions
# =====================================================================

BEST_SINGLE = (8, 9)
SECOND_CANDIDATES = [(28, 29), (16, 17), (4, 5), (32, 33), (44, 45)]

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
    \"\"\"Find (first_pass_end, second_pass_end) step indices for each block.\"\"\"
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

def generate_multi_alpha(prompt, blocks, alphas, max_new_tokens=64):
    \"\"\"Generate with per-block alpha weighting via manual layer-by-layer forward.\"\"\"
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
                    if seam is None:
                        continue
                    if step_idx == seam[0]:
                        saved_h1[si] = h.clone()
                    if step_idx == seam[1] and si in saved_h1:
                        h = saved_h1[si] + sorted_alphas[si] * (h - saved_h1[si])
                        del saved_h1[si]

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
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
    print(f'  {name:65s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alphas': list(alphas),
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

all_results = []

# =====================================================================
# STEP 1: Baseline — verify known results
# =====================================================================
print(f'\n{\"=\" * 70}')
print('STEP 1: Baseline + Best Single Verification')
print(f'{\"=\" * 70}', flush=True)

r = evaluate([BEST_SINGLE], [1.0], f'single ({BEST_SINGLE[0]},{BEST_SINGLE[1]}) @1.0')
all_results.append(r)
best_single_score = r['combined']
print(f'Best single score: {best_single_score:.2f}', flush=True)

# =====================================================================
# STEP 2: Alpha tuning on the best single (8,9)
# =====================================================================
print(f'\n{\"=\" * 70}')
print('STEP 2: Alpha Tuning on Best Single (8,9)')
print(f'{\"=\" * 70}', flush=True)

SINGLE_ALPHAS = [0.9, 1.05, 1.1, 1.15, 1.2]
best_single_alpha = 1.0
best_single_combined = best_single_score

for a in SINGLE_ALPHAS:
    r = evaluate([BEST_SINGLE], [a], f'single (8,9) @{a}')
    all_results.append(r)
    if r['combined'] > best_single_combined:
        best_single_combined = r['combined']
        best_single_alpha = a
        print(f'  >>> NEW BEST single alpha: {a} -> combined={best_single_combined:.2f}', flush=True)

print(f'\nBest single alpha: {best_single_alpha} -> combined={best_single_combined:.2f}', flush=True)

# =====================================================================
# STEP 3: Whisper-alpha pairs — (8,9) + second block at low alpha
# =====================================================================
print(f'\n{\"=\" * 70}')
print('STEP 3: Whisper-Alpha Pairs')
print(f'Best single (8,9) at alpha={best_single_alpha}')
print(f'Second block candidates: {SECOND_CANDIDATES}')
print(f'Second alpha values: [0.1, 0.2, 0.5]')
print(f'{\"=\" * 70}', flush=True)

WHISPER_ALPHAS = [0.1, 0.2, 0.5]
best_pair_score = best_single_combined
best_pair_config = None
pair_results = []

for second in SECOND_CANDIDATES:
    print(f'\n  --- Pair: (8,9) + ({second[0]},{second[1]}) ---', flush=True)
    for a2 in WHISPER_ALPHAS:
        blocks = [BEST_SINGLE, second]
        alphas = [best_single_alpha, a2]
        name = f'pair (8,9)@{best_single_alpha}+({second[0]},{second[1]})@{a2}'
        r = evaluate(blocks, alphas, name)
        all_results.append(r)
        pair_results.append(r)
        if r['combined'] > best_pair_score:
            best_pair_score = r['combined']
            best_pair_config = (blocks, alphas)
            print(f'  >>> NEW BEST pair: {name} -> combined={best_pair_score:.2f}', flush=True)

pair_beats_single = best_pair_score > best_single_combined
print(f'\nPair beats single? {\"YES\" if pair_beats_single else \"NO\"}', flush=True)
print(f'Best single: {best_single_combined:.2f}', flush=True)
print(f'Best pair:   {best_pair_score:.2f}', flush=True)

# =====================================================================
# STEP 4: Whisper-alpha triples (if pair beats single)
# =====================================================================
print(f'\n{\"=\" * 70}')
print('STEP 4: Whisper-Alpha Triples')
print(f'{\"=\" * 70}', flush=True)

triple_results = []

if pair_beats_single and best_pair_config is not None:
    print('Pair beats single! Trying triples...', flush=True)
    bp_blocks, bp_alphas = best_pair_config
    used_blocks = set(tuple(b) for b in bp_blocks)

    # Third block candidates: all second candidates not already used
    third_candidates = [b for b in SECOND_CANDIDATES if tuple(b) not in used_blocks]

    for third in third_candidates:
        for a3 in WHISPER_ALPHAS:
            blocks = list(bp_blocks) + [third]
            alphas = list(bp_alphas) + [a3]
            name = f'triple +({third[0]},{third[1]})@{a3}'
            r = evaluate(blocks, alphas, name)
            all_results.append(r)
            triple_results.append(r)
            if r['combined'] > best_pair_score:
                print(f'  >>> TRIPLE BEATS PAIR: {name} -> combined={r[\"combined\"]:.2f}', flush=True)
else:
    print('Pair does not beat single. Skipping triples.', flush=True)
    print('Instead, trying a few triples directly with heavy whisper (alpha=0.1) on both secondary blocks...', flush=True)

    # Even if pair doesn't beat single, try a few triples with very low alpha
    # Maybe two whisper blocks together create a synergy
    for i, second in enumerate(SECOND_CANDIDATES[:3]):
        for third in SECOND_CANDIDATES[i+1:4]:
            blocks = [BEST_SINGLE, second, third]
            alphas = [best_single_alpha, 0.1, 0.1]
            name = f'triple (8,9)@{best_single_alpha}+({second[0]},{second[1]})@0.1+({third[0]},{third[1]})@0.1'
            r = evaluate(blocks, alphas, name)
            all_results.append(r)
            triple_results.append(r)
            if r['combined'] > best_single_combined:
                print(f'  >>> TRIPLE BEATS SINGLE: {name} -> combined={r[\"combined\"]:.2f}', flush=True)

# =====================================================================
# SUMMARY
# =====================================================================
print(f'\n{\"=\" * 70}')
print('SUMMARY: MoE Whisper-Alpha Stacking')
print(f'{\"=\" * 70}', flush=True)

print(f'Model: Qwen3-30B-A3B ({N} layers)', flush=True)

# Best from each category
print(f'\nBest single: (8,9) @{best_single_alpha} -> combined={best_single_combined:.2f}', flush=True)

if pair_results:
    best_pair_r = max(pair_results, key=lambda x: x['combined'])
    print(f'Best pair: {best_pair_r[\"name\"]} -> combined={best_pair_r[\"combined\"]:.2f}', flush=True)

if triple_results:
    best_triple_r = max(triple_results, key=lambda x: x['combined'])
    print(f'Best triple: {best_triple_r[\"name\"]} -> combined={best_triple_r[\"combined\"]:.2f}', flush=True)

# Overall best
overall_best = max(all_results, key=lambda x: x['combined'])
print(f'\nOverall best: {overall_best[\"name\"]} -> combined={overall_best[\"combined\"]:.2f}', flush=True)

print(f'\nAll results ranked:', flush=True)
sorted_all = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_all:
    print(f'  {r[\"name\"]:65s}: combined={r[\"combined\"]:.2f}', flush=True)

# Save results
os.makedirs('results/data/moe', exist_ok=True)
save_data = {
    'date': datetime.now().isoformat(),
    'model': 'Qwen3-30B-A3B',
    'num_layers': N,
    'best_single': {'block': list(BEST_SINGLE), 'alpha': best_single_alpha, 'combined': best_single_combined},
    'pair_beats_single': pair_beats_single,
    'best_pair': best_pair_r if pair_results else None,
    'best_triple': best_triple_r if triple_results else None,
    'overall_best': overall_best,
    'all_results': all_results,
}
with open('results/data/moe/whisper_stacking.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f'\nSaved to results/data/moe/whisper_stacking.json', flush=True)
"

echo "=== Done at $(date) ==="
