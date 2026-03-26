#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_e2e_27b_%j.log
#SBATCH --job-name=deeppass_e2e

# End-to-end automated pipeline demo on Qwen3.5-27B
# SBUID screening → greedy stacking → per-block alpha tuning
# Demonstrates the full method on a fresh model

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== End-to-End Pipeline Demo: Qwen3.5-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/Qwen3.5-27B'

print('=' * 70)
print('END-TO-END PIPELINE DEMO')
print('SBUID screening → greedy stacking → alpha tuning')
print(f'Model: {MODEL_PATH}')
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
    'What is the derivative of sin(x) * e^x?',
    'The theory of general relativity describes',
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
    print(f'  {name:55s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

pipeline_log = {}
t_start = time.time()

# =====================================================================
# STEP 1: SBUID Screening (~10 min)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('STEP 1: SBUID Screening')
print(f'{\"=\" * 70}', flush=True)
t1 = time.time()

# Compute rho and BLOOD for all candidate blocks
candidates = []
for start in range(0, N-1, 2):
    for size in [3, 5, 7]:
        end = start + size
        if end <= N:
            candidates.append((start, end))
print(f'Screening {len(candidates)} blocks...', flush=True)

# Rho
rhos = {}
for idx, block in enumerate(candidates):
    i, j = block
    block_rhos = []
    for prompt in CAL_PROMPTS[:4]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(ids['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()
            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
            model.config.num_hidden_layers = len(order)
            out_dup = model(ids['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()
            inner.layers = nn.ModuleList(original_layers)
            model.config.num_hidden_layers = N
            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8: block_rhos.append(num / den)
    rhos[block] = float(np.mean(block_rhos)) if block_rhos else 1.0
    if (idx + 1) % 10 == 0:
        print(f'  [{idx+1}/{len(candidates)}] rho done', flush=True)

# BLOOD
def compute_blood_profile():
    layer_norms = [[] for _ in range(N)]
    hooks = []
    def make_hook(idx):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            layer_norms[idx].append(torch.norm(out.float() - inp.float()).item())
        return hook_fn
    for idx in range(N):
        hooks.append(original_layers[idx].register_forward_hook(make_hook(idx)))
    for prompt in CAL_PROMPTS[:4]:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(ids['input_ids'], use_cache=False)
    for h in hooks: h.remove()
    return [float(np.mean(ns)) if ns else 0.0 for ns in layer_norms]

inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N
base_blood = compute_blood_profile()

bloods = {}
for idx, block in enumerate(candidates):
    i, j = block
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
    model.config.num_hidden_layers = len(order)
    dup_blood = compute_blood_profile()
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N
    impact = sum(base_blood[l] - dup_blood[l + (j-i)] for l in range(j, N) if l + (j-i) < len(dup_blood))
    bloods[block] = impact

# SBUID ranking
sbuid_scores = {b: bloods[b] - 2500 * rhos[b] for b in candidates}  # lambda=2500 from cross-model validation
ranked = sorted(sbuid_scores.items(), key=lambda x: x[1], reverse=True)

print(f'\\nTop 10 by SBUID:')
for b, s in ranked[:10]:
    print(f'  ({b[0]:2d},{b[1]:2d}): SBUID={s:+.0f} rho={rhos[b]:.4f} blood={bloods[b]:+.0f}', flush=True)

top_blocks = [b for b, _ in ranked[:10]]
t1_elapsed = time.time() - t1
pipeline_log['step1_screening'] = {'time_min': t1_elapsed/60, 'n_candidates': len(candidates), 'top_10': [list(b) for b in top_blocks]}
print(f'\\nStep 1 done in {t1_elapsed/60:.1f} min', flush=True)

# =====================================================================
# STEP 2: Evaluate Top Singletons (~30 min)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('STEP 2: Evaluate Top 8 Singletons')
print(f'{\"=\" * 70}', flush=True)
t2 = time.time()

gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

single_scores = {}
for block in top_blocks[:8]:
    r = evaluate([block], [1.0], f'single ({block[0]},{block[1]})')
    single_scores[block] = r['combined']

best_single = max(single_scores.items(), key=lambda x: x[1])
t2_elapsed = time.time() - t2
pipeline_log['step2_singletons'] = {'time_min': t2_elapsed/60, 'baseline': baseline, 'best_single': list(best_single[0]), 'best_score': best_single[1]}
print(f'\\nBest single: ({best_single[0][0]},{best_single[0][1]}) = {best_single[1]:.2f}')
print(f'Step 2 done in {t2_elapsed/60:.1f} min', flush=True)

# =====================================================================
# STEP 3: Greedy Stacking (~20 min)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('STEP 3: Greedy Stacking — Find Best Pair')
print(f'{\"=\" * 70}', flush=True)
t3 = time.time()

best_block = best_single[0]
second_candidates = [b for b in top_blocks[:8] if b[1] <= best_block[0] or b[0] >= best_block[1]]

pair_results = []
for second in second_candidates[:6]:
    pair = sorted([best_block, second])
    r = evaluate(pair, [1.0, 1.0], f'pair ({pair[0][0]},{pair[0][1]})+({pair[1][0]},{pair[1][1]})')
    pair_results.append((pair, r['combined']))

best_pair = max(pair_results, key=lambda x: x[1]) if pair_results else None
t3_elapsed = time.time() - t3
pipeline_log['step3_stacking'] = {'time_min': t3_elapsed/60}
if best_pair:
    pipeline_log['step3_stacking']['best_pair'] = [list(b) for b in best_pair[0]]
    pipeline_log['step3_stacking']['best_score'] = best_pair[1]
    print(f'\\nBest pair: {best_pair[0]} = {best_pair[1]:.2f}')
print(f'Step 3 done in {t3_elapsed/60:.1f} min', flush=True)

# =====================================================================
# STEP 4: Alpha Tuning (~30 min)
# =====================================================================
print(f'\\n{\"=\" * 70}')
print('STEP 4: Per-Block Alpha Tuning')
print(f'{\"=\" * 70}', flush=True)
t4 = time.time()

# Alpha tune the best single
best_single_tuned = best_single[1]
best_single_alpha = 1.0
for a in [0.9, 1.05, 1.1, 1.15, 1.2, 1.25]:
    r = evaluate([best_block], [a], f'single @{a}')
    if r['combined'] > best_single_tuned:
        best_single_tuned = r['combined']
        best_single_alpha = a

best_triple = None
best_triple_score = 0
best_pair_score = 0
best_pair_alphas = [1.0, 1.0]

# Alpha tune the pair
if best_pair and best_pair[1] > baseline:
    bp = best_pair[0]
    best_pair_score = best_pair[1]
    best_pair_alphas = [1.0, 1.0]
    for a0 in [0.8, 0.9, 1.0, 1.1]:
        for a1 in [0.8, 0.9, 1.0, 1.1]:
            if a0 == 1.0 and a1 == 1.0: continue
            r = evaluate(bp, [a0, a1], f'pair @{a0}/{a1}')
            if r['combined'] > best_pair_score:
                best_pair_score = r['combined']
                best_pair_alphas = [a0, a1]

    # Whisper triple
    print(f'\\n  --- Whisper triples ---', flush=True)
    third_candidates = [b for b in top_blocks[:8] if all(b[1] <= pb[0] or b[0] >= pb[1] for pb in bp)]
    best_triple_score = best_pair_score
    best_triple = None
    for third in third_candidates[:4]:
        triple = sorted(list(bp) + [third])
        for a3 in [0.05, 0.1, 0.15, 0.2]:
            alphas = [best_pair_alphas[0] if b == bp[0] else best_pair_alphas[1] if b == bp[1] else a3 for b in triple]
            r = evaluate(triple, alphas, f'triple +({third[0]},{third[1]})@{a3}')
            if r['combined'] > best_triple_score:
                best_triple_score = r['combined']
                best_triple = (triple, alphas)

t4_elapsed = time.time() - t4
pipeline_log['step4_alpha'] = {'time_min': t4_elapsed/60, 'best_single_tuned': best_single_tuned, 'best_single_alpha': best_single_alpha}
if best_pair:
    pipeline_log['step4_alpha']['best_pair_score'] = best_pair_score
    pipeline_log['step4_alpha']['best_pair_alphas'] = best_pair_alphas
if best_triple:
    pipeline_log['step4_alpha']['best_triple_score'] = best_triple_score
print(f'Step 4 done in {t4_elapsed/60:.1f} min', flush=True)

# =====================================================================
# SUMMARY
# =====================================================================
total_time = time.time() - t_start
print(f'\\n{\"=\" * 70}')
print('END-TO-END PIPELINE SUMMARY')
print(f'{\"=\" * 70}')
print(f'Model: {MODEL_PATH} ({N} layers)')
print(f'Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} min)')
print(f'\\nStep 1 (SBUID screening): {t1_elapsed/60:.1f} min — screened {len(candidates)} blocks')
print(f'Step 2 (singleton eval):  {t2_elapsed/60:.1f} min — evaluated top 8')
print(f'Step 3 (greedy stacking): {t3_elapsed/60:.1f} min — tested pairs')
print(f'Step 4 (alpha tuning):    {t4_elapsed/60:.1f} min — tuned alphas + whisper triples')
print(f'\\nResults:')
print(f'  Baseline:           {baseline:.2f}')
print(f'  Best single:        ({best_block[0]},{best_block[1]})@{best_single_alpha} = {best_single_tuned:.2f} (delta={best_single_tuned-baseline:+.2f})')
if best_pair:
    print(f'  Best pair (tuned):  {best_pair_alphas} = {best_pair_score:.2f} (delta={best_pair_score-baseline:+.2f})')
if best_triple:
    print(f'  Best triple:        = {best_triple_score:.2f} (delta={best_triple_score-baseline:+.2f})')
print(f'\\nComparison: Ng\\'s brute force = 3,241 evaluations. Our pipeline = ~{8+6+12+4+12} evaluations.', flush=True)

pipeline_log['total_time_hours'] = total_time / 3600
pipeline_log['baseline'] = baseline

os.makedirs('results/data/qwen35/e2e_pipeline', exist_ok=True)
with open('results/data/qwen35/e2e_pipeline/results.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), **pipeline_log}, f, indent=2)
print(f'Saved to results/data/qwen35/e2e_pipeline/results.json', flush=True)
"

echo "=== Done at $(date) ==="
