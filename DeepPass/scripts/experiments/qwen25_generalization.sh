#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_qwen35_generalization_%j.log
#SBATCH --job-name=deeppass_qwen35

# Generalization test: does spectral screening + layer duplication work on Qwen2.5-72B?
# Different model from our Qwen2-based calme-2.1, same architecture family but different training.
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Qwen2.5-72B Generalization Test ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, torch, torch.nn as nn, numpy as np
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

model, tokenizer = load_original_model('models/full/Qwen3.5-27B')
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded Qwen2.5-72B: {N} layers')

# Calibration prompts for spectral screening
cal_prompts = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'Calculate 15! / 13!',
    'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]

# Step 1: Spectral screen candidate blocks
print('\\n=== Step 1: Spectral Screening ===')
candidates = []
for start in range(0, 65, 5):
    for size in [5, 7, 10]:
        end = start + size
        if end <= N:
            candidates.append((start, end))
# Add Ng-equivalent positions
for b in [(0, 7), (10, 15), (15, 20), (45, 52), (50, 60)]:
    if b not in candidates and b[1] <= N:
        candidates.append(b)

print(f'Screening {len(candidates)} candidate blocks...')
block_rhos = {}
for block in candidates:
    i, j = block
    rhos = []
    for prompt in cal_prompts:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = inner.embed_tokens(ids['input_ids'])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for layer_idx in range(i):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            h_input = h.clone()
            h1 = h_input.clone()
            for layer_idx in range(i, j):
                out = inner.layers[layer_idx](h1, position_embeddings=pos_embeds, use_cache=False)
                h1 = out[0] if isinstance(out, tuple) else out
            h2 = h1.clone()
            for layer_idx in range(i, j):
                out = inner.layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out
            num = torch.norm(h2 - h1).item()
            den = torch.norm(h1 - h_input).item()
            if den > 1e-8:
                rhos.append(num / den)
    block_rhos[block] = float(np.mean(rhos)) if rhos else 1.0

sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
print('Top 15 by displacement rho:')
for b, r in sorted_blocks[:15]:
    print(f'  ({b[0]:2d},{b[1]:2d}): rho={r:.4f}')

# Step 2: Baseline
print('\\n=== Step 2: Baseline ===')
def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

math_base = run_math_probe(gen, verbose=False)
eq_base = run_eq_bench_probe(gen_long, verbose=False)
combined_base = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: math={math_base[\"score\"]:.4f} eq={eq_base[\"score\"]:.1f} combined={combined_base:.2f}')

# Step 3: Test top 8 spectral candidates
print('\\n=== Step 3: Evaluate Top 8 Candidates ===')
results = [{'block': 'baseline', 'math': math_base['score'], 'eq': eq_base['score'], 'combined': combined_base}]

top_8 = [b for b, r in sorted_blocks[:8]]
for block in top_8:
    i, j = block
    order = list(range(j)) + list(range(i, j)) + list(range(j, N))
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - combined_base

    print(f'  ({i:2d},{j:2d}): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} delta={delta:+.2f}')
    results.append({'block': list(block), 'rho': block_rhos[block], 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined, 'delta': delta})

    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

# Step 4: Test Ng's equivalent position
print('\\n=== Step 4: Ng-equivalent position (45,52) ===')
if (45, 52) not in top_8:
    block = (45, 52)
    order = list(range(52)) + list(range(45, 52)) + list(range(52, N))
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  (45,52): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f}')
    results.append({'block': [45, 52], 'rho': block_rhos.get((45,52), -1), 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined})
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

# Save
os.makedirs('results/data/qwen35', exist_ok=True)
with open('results/data/qwen35/generalization_results.json', 'w') as f:
    json.dump({'spectral': [{'block': list(b), 'rho': r} for b, r in sorted_blocks], 'probes': results}, f, indent=2)
print('\\nSaved to results/data/qwen35/generalization_results.json')
print(f'\\n=== DONE at $(date) ===')
"
