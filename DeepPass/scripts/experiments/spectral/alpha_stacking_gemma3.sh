#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_alpha_stack_gemma3_%j.log
#SBATCH --job-name=deeppass_asg3

# Per-block alpha stacking on Gemma3-27B
# Test if the "whisper alpha" approach that works on 72B generalizes
# Known: pair (4,5)+(20,21) = 84.42, single (20,21) = 83.76

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Alpha-Tuned Stacking on Gemma3-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'

print('=' * 70)
print('ALPHA-TUNED STACKING ON GEMMA3-27B (62 layers)')
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

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    elif hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n

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

    # Gemma3 uses position_ids not position_embeddings
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Use full model forward isn't possible with manual layers on Gemma3
            # due to sliding window attention. Use simple embed + layer loop.
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Gemma3 layers need position_ids and cache_position
            cache_position = torch.arange(seq_len, device=device)

            saved_h1 = {}
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                # Try different calling conventions
                try:
                    out = layer(h, position_ids=pos_ids, cache_position=cache_position, use_cache=False)
                except TypeError:
                    try:
                        out = layer(h, position_ids=pos_ids, use_cache=False)
                    except TypeError:
                        out = layer(h, use_cache=False)
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

# Gemma3 might not work with manual layer loop due to sliding window attention.
# Fall back to full-model forward with ModuleList swapping if manual fails.

def generate_full_model(prompt, blocks, max_new_tokens=64):
    \"\"\"Standard duplication via ModuleList swapping (no per-block alpha).\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
            logits = out.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def evaluate_full(blocks, name):
    gen = lambda p: generate_full_model(p, blocks, max_new_tokens=64)
    gen_long = lambda p: generate_full_model(p, blocks, max_new_tokens=128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {name:60s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

def evaluate_alpha(blocks, alphas, name):
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

# Known best configs from previous experiments
BEST_SINGLE = (20, 21)
BEST_PAIR = [(4, 5), (20, 21)]

# Test if manual layer loop works for Gemma3
print(f'\\n--- Testing manual layer loop on Gemma3 ---', flush=True)
try:
    test_r = evaluate_alpha([BEST_SINGLE], [1.0], 'test: manual (20,21)@1.0')
    manual_works = True
    print('  Manual layer loop WORKS on Gemma3!', flush=True)
except Exception as e:
    print(f'  Manual layer loop FAILED: {e}', flush=True)
    print('  Falling back to full-model forward (no per-block alpha)', flush=True)
    manual_works = False

if manual_works:
    # =====================================================================
    # Alpha tuning on pair
    # =====================================================================
    print(f'\\n{\"=\" * 70}')
    print('Alpha tuning on pair (4,5)+(20,21)')
    print(f'{\"=\" * 70}', flush=True)

    r = evaluate_alpha(BEST_PAIR, [1.0, 1.0], 'pair @1.0/1.0')
    all_results.append(r)

    for a0, a1 in [(0.9, 1.0), (1.0, 0.9), (0.9, 1.15), (1.15, 1.0), (0.9, 0.9), (1.1, 1.1)]:
        r = evaluate_alpha(BEST_PAIR, [a0, a1], f'pair @{a0}/{a1}')
        all_results.append(r)

    # =====================================================================
    # Triple with whisper alpha
    # =====================================================================
    print(f'\\n{\"=\" * 70}')
    print('Triple with whisper alpha')
    print(f'{\"=\" * 70}', flush=True)

    THIRD_CANDIDATES = [(8, 9), (12, 13), (16, 17), (24, 25), (6, 11), (14, 19)]

    for third in THIRD_CANDIDATES:
        if third[1] <= BEST_PAIR[0][0] or third[0] >= BEST_PAIR[1][1] or \
           (third[0] < BEST_PAIR[0][1] and third[1] > BEST_PAIR[0][0]) or \
           (third[0] < BEST_PAIR[1][1] and third[1] > BEST_PAIR[1][0]):
            # Check non-overlap more carefully
            overlap = any(not (third[1] <= b[0] or third[0] >= b[1]) for b in BEST_PAIR)
            if overlap: continue

        triple = list(BEST_PAIR) + [third]
        # Standard alpha=1.0 for reference
        r = evaluate_alpha(triple, [1.0, 1.0, 1.0], f'triple +({third[0]},{third[1]}) @1.0')
        all_results.append(r)
        # Whisper alphas
        for a3 in [0.05, 0.1, 0.15, 0.2]:
            r = evaluate_alpha(triple, [1.0, 1.0, a3], f'triple +({third[0]},{third[1]}) @{a3}')
            all_results.append(r)

else:
    # Full-model forward only (no alpha tuning, just test more block combos)
    print('\\nUsing full-model forward — testing block combinations at alpha=1.0', flush=True)

    r = evaluate_full([BEST_SINGLE], 'single (20,21)')
    all_results.append(r)
    r = evaluate_full(BEST_PAIR, 'pair (4,5)+(20,21)')
    all_results.append(r)

    THIRD_CANDIDATES = [(8, 9), (12, 13), (16, 17), (24, 25)]
    for third in THIRD_CANDIDATES:
        overlap = any(not (third[1] <= b[0] or third[0] >= b[1]) for b in BEST_PAIR)
        if overlap: continue
        triple = list(BEST_PAIR) + [third]
        r = evaluate_full(triple, f'triple +({third[0]},{third[1]})')
        all_results.append(r)

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY')
print(f'{\"=\" * 70}')
print(f'Baseline: {baseline:.2f}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r[:15]:
    print(f'  {r[\"name\"]:60s}: combined={r[\"combined\"]:.2f}', flush=True)

os.makedirs('results/data/gemma3_27b/alpha_stacking', exist_ok=True)
with open('results/data/gemma3_27b/alpha_stacking/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'baseline': baseline,
               'manual_layer_loop_works': manual_works, 'results': all_results}, f, indent=2)
print(f'Saved to results/data/gemma3_27b/alpha_stacking/results.json', flush=True)
"

echo "=== Done at $(date) ==="
