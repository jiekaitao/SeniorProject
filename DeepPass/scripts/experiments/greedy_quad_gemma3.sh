#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_greedy_quad_g3_%j.log
#SBATCH --job-name=g3_quad

# Greedy quad search on Gemma3-27B
# Anchor: alpha-tuned triple (0,2)+(12,13)+(47,48)
# Sweep 4th block candidates with whisper alpha (0.1-0.5)
# Then Bayesian optimize all 5 alphas (4 from triple + 1 new)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Greedy Quad Search: Gemma3-27B ==="
echo "Anchor: (0,2)+(12,13)+(47,48) with alpha"
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe, MATH_QUESTIONS
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = 'models/full/gemma-3-27b-it'
ANCHOR_BLOCKS = [(0, 2), (12, 13), (47, 48)]
ANCHOR_ALPHAS = {0: 0.8797, 1: 0.8063, 12: 1.4507, 47: 0.9453}
SAVE_DIR = 'results/data/gemma3_27b/greedy_quad'
os.makedirs(SAVE_DIR, exist_ok=True)

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)

inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
print(f'Loaded: {N} layers', flush=True)

eq_all = _load_questions()
eq_subset = eq_all[:10]
math_subset = MATH_QUESTIONS[:10]

def set_num_layers(n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    if hasattr(model.config, 'num_hidden_layers'):
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

def blocks_overlap(b1, b2):
    return not (b1[1] <= b2[0] or b2[1] <= b1[0])

def apply_with_alpha(blocks, alphas):
    \"\"\"Apply blocks with per-layer alpha hooks. Returns list of hook handles.\"\"\"
    order = build_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    sorted_b = sorted(blocks)
    dup_layers = []
    for (i, j) in sorted_b:
        for l in range(i, j):
            dup_layers.append(l)

    hooks = []
    for layer_idx in dup_layers:
        alpha_val = alphas.get(layer_idx, 1.0)
        module = original_layers[layer_idx]
        counter = [0]
        ac = [alpha_val]

        def make_hook(ctr, ac):
            def hook(module, input, output):
                ctr[0] += 1
                if ctr[0] % 2 == 0:
                    h_in = input[0]
                    if isinstance(output, tuple):
                        h_out = output[0]
                        blended = h_in + ac[0] * (h_out - h_in)
                        return (blended,) + output[1:]
                    blended = h_in + ac[0] * (output - h_in)
                    return blended
                return output
            return hook

        h = module.register_forward_hook(make_hook(counter, ac))
        hooks.append(h)
    return hooks

def restore(hooks):
    for h in hooks:
        h.remove()
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

def evaluate_reduced(blocks, alphas, name):
    hooks = apply_with_alpha(blocks, alphas)
    t0 = time.time()
    math_r = run_math_probe(gen, questions=math_subset, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore(hooks)
    n_extra = sum(j - i for i, j in blocks)
    print(f'  {name:50s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} +{n_extra}layers ({elapsed:.0f}s)', flush=True)
    return {'name': name, 'blocks': [list(b) for b in blocks], 'alphas': {str(k): v for k, v in alphas.items()},
            'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

# ======================================================================
# Step 1: Evaluate anchor triple with alpha
# ======================================================================
print('\\n=== Step 1: Anchor triple baseline ===', flush=True)
anchor_result = evaluate_reduced(ANCHOR_BLOCKS, ANCHOR_ALPHAS, 'anchor (0,2)+(12,13)+(47,48)')
anchor_score = anchor_result['combined']

# ======================================================================
# Step 2: Sweep 4th block candidates
# ======================================================================
print('\\n=== Step 2: Sweep 4th block candidates ===', flush=True)

# Generate candidates: size 1-3, step 2, avoiding overlap with anchor
candidates = []
for size in [1, 2, 3]:
    for start in range(0, N - size, 2):
        block = (start, start + size)
        overlaps = any(blocks_overlap(block, ab) for ab in ANCHOR_BLOCKS)
        if not overlaps:
            candidates.append(block)

print(f'Testing {len(candidates)} candidates with whisper alpha=0.3...', flush=True)

results_4th = []
for cand in candidates:
    all_blocks = ANCHOR_BLOCKS + [cand]
    # Use whisper alpha for new block, anchor alphas for existing
    alphas = dict(ANCHOR_ALPHAS)
    for l in range(cand[0], cand[1]):
        alphas[l] = 0.3  # whisper alpha for new block
    name = f'quad +({cand[0]},{cand[1]}) @0.3'
    r = evaluate_reduced(all_blocks, alphas, name)
    results_4th.append(r)

results_4th.sort(key=lambda x: x['combined'], reverse=True)

print(f'\\n--- Top 10 4th blocks ---', flush=True)
for r in results_4th[:10]:
    print(f'  {r[\"name\"]:50s} combined={r[\"combined\"]:.2f}', flush=True)

# Save checkpoint
with open(f'{SAVE_DIR}/checkpoint_4th_sweep.json', 'w') as f:
    json.dump({'anchor': anchor_result, 'sweep': results_4th}, f, indent=2)

# ======================================================================
# Step 3: Test top 5 at different whisper alphas
# ======================================================================
print('\\n=== Step 3: Alpha sweep on top 5 candidates ===', flush=True)
best_quads = []
for r in results_4th[:5]:
    cand_block = [b for b in [tuple(b) for b in r['blocks']] if tuple(b) not in [tuple(b) for b in ANCHOR_BLOCKS]][0]
    for whisper in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        all_blocks = ANCHOR_BLOCKS + [cand_block]
        alphas = dict(ANCHOR_ALPHAS)
        for l in range(cand_block[0], cand_block[1]):
            alphas[l] = whisper
        name = f'quad +({cand_block[0]},{cand_block[1]}) @{whisper}'
        r2 = evaluate_reduced(all_blocks, alphas, name)
        best_quads.append(r2)

best_quads.sort(key=lambda x: x['combined'], reverse=True)
print(f'\\n--- Top 10 quads (alpha-swept) ---', flush=True)
for r in best_quads[:10]:
    print(f'  {r[\"name\"]:50s} combined={r[\"combined\"]:.2f}', flush=True)

# ======================================================================
# Step 4: Validate top 3 with full probes
# ======================================================================
print('\\n=== Step 4: Full-probe validation (top 3 + anchor) ===', flush=True)

to_validate = [anchor_result] + best_quads[:3]
validated = []
for r in to_validate:
    blocks = [tuple(b) for b in r['blocks']]
    alphas = {int(k): v for k, v in r['alphas'].items()}
    hooks = apply_with_alpha(blocks, alphas)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore(hooks)

    print(f'  {r[\"name\"]:50s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    validated.append({
        'name': r['name'], 'blocks': r['blocks'], 'alphas': r['alphas'],
        'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined,
    })

validated.sort(key=lambda x: x['combined'], reverse=True)

print(f'\\n{\"=\" * 60}', flush=True)
print(f'BEST: {validated[0][\"name\"]} combined={validated[0][\"combined\"]:.2f}', flush=True)
print(f'{\"=\" * 60}', flush=True)

# Save final results
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump({
        'anchor': anchor_result,
        'sweep_top_10': results_4th[:10],
        'alpha_swept_top_10': best_quads[:10],
        'validated': validated,
        'best': validated[0],
    }, f, indent=2)
print(f'Saved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
