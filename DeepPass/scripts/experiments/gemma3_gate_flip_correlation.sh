#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_gate_corr_%j.log
#SBATCH --job-name=g3_gcorr

# Compute gate flip rate for ALL blocks we already evaluated on Gemma3
# Then correlate with actual dual-probe scores to test gate flip as Tier 1 screener
# Compare: SBUID vs gate_flip_rate vs rho vs BLOOD — which predicts best?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gate Flip Correlation: Gemma3-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from scipy import stats

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/gate_flip_correlation'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load previously evaluated blocks from comprehensive search
comp_file = 'results/data/gemma3_27b/comprehensive/checkpoint_singles.json'
if not os.path.exists(comp_file):
    # Try alternative locations
    for f in ['results/data/gemma3_27b/comprehensive/checkpoint_pairs.json',
              'results/data/gemma3_27b/mega_stacking/alt_anchor_results.json']:
        if os.path.exists(f):
            comp_file = f
            break

print(f'Loading evaluated blocks from {comp_file}...', flush=True)
with open(comp_file) as f:
    comp_data = json.load(f)

# Extract blocks that have combined scores
evaluated_blocks = []
if isinstance(comp_data, dict) and 'singles' in comp_data:
    for entry in comp_data['singles']:
        if 'combined' in entry and 'blocks' in entry:
            evaluated_blocks.append(entry)
elif isinstance(comp_data, dict) and 'best_alt_triples' in comp_data:
    # Alt anchor format — use individual block data
    pass
elif isinstance(comp_data, list):
    for entry in comp_data:
        if isinstance(entry, dict) and 'combined' in entry:
            evaluated_blocks.append(entry)

print(f'Found {len(evaluated_blocks)} previously evaluated blocks', flush=True)

# If not enough blocks from files, use a set of representative blocks
if len(evaluated_blocks) < 10:
    print('Not enough pre-evaluated blocks. Will evaluate + measure gate flip together.', flush=True)
    # Generate candidate blocks
    test_blocks = [(i, i+1) for i in range(0, 62, 3)] + [(i, i+2) for i in range(0, 60, 6)]
    need_eval = True
else:
    test_blocks = [tuple(e['blocks'][0]) if isinstance(e['blocks'][0], list) else (e['blocks'][0], e['blocks'][1]) for e in evaluated_blocks]
    need_eval = False

print(f'Will measure gate flip for {len(test_blocks)} blocks', flush=True)

# Load model
print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

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

def restore():
    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

cal_prompts = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'What is the capital of France?',
    'Explain entropy simply.',
    'Who wrote Romeo and Juliet?',
    'What is 2^16?',
]

# If we need to evaluate, import probes
if need_eval:
    from math_probe import run_math_probe
    from eq_bench_probe import run_eq_bench_probe
    from layer_duplicator import generate_no_cache
    def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# ======================================================================
# Measure gate flip rate for each block
# ======================================================================
print('\\n=== Measuring Gate Flip Rates ===', flush=True)

results = []
for idx, block in enumerate(test_blocks):
    if isinstance(block, (list, tuple)) and len(block) == 2:
        i, j = int(block[0]), int(block[1])
    else:
        continue

    if i >= N or j > N:
        continue

    # Apply duplication
    order = build_order([(i, j)], N)
    inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
    set_num_layers(len(order))

    # Measure gate flip for each layer in the block
    block_flip_rates = []
    for layer_idx in range(i, j):
        module = original_layers[layer_idx]
        if not hasattr(module.mlp, 'gate_proj'):
            continue
        gate_proj = module.mlp.gate_proj

        flip_rates = []
        for prompt in cal_prompts:
            gate_vals = []
            counter = [0]
            def make_capture(ctr, gvals, gp):
                def hook(mod, inp, out):
                    ctr[0] += 1
                    with torch.no_grad():
                        inp_t = inp[0] if isinstance(inp, tuple) else inp
                        gvals.append(gp(inp_t[:, -1, :]).cpu().float().squeeze(0))
                    return out
                return hook
            h = module.mlp.register_forward_hook(make_capture(counter, gate_vals, gate_proj))
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                model(inputs['input_ids'], use_cache=False)
            h.remove()

            if len(gate_vals) >= 2:
                flips = ((gate_vals[0] > 0) != (gate_vals[1] > 0)).float().mean().item()
                flip_rates.append(flips)

        if flip_rates:
            block_flip_rates.append(float(np.mean(flip_rates)))

    restore()

    avg_flip = float(np.mean(block_flip_rates)) if block_flip_rates else -1

    entry = {'block': [i, j], 'gate_flip_rate': avg_flip}

    # If we need dual-probe eval too
    if need_eval:
        order = build_order([(i, j)], N)
        inner.layers = nn.ModuleList([original_layers[idx2] for idx2 in order])
        set_num_layers(len(order))
        t0 = time.time()
        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        restore()
        entry['math'] = math_r['score']
        entry['eq'] = eq_r['score']
        entry['combined'] = combined
        print(f'  ({i},{j}): flip={avg_flip:.4f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    else:
        # Match with pre-existing combined score
        for ev in evaluated_blocks:
            ev_block = ev['blocks'][0] if isinstance(ev['blocks'][0], list) else ev['blocks']
            if [i, j] == list(ev_block)[:2]:
                entry['combined'] = ev['combined']
                entry['math'] = ev.get('math', 0)
                entry['eq'] = ev.get('eq', 0)
                break
        print(f'  ({i},{j}): flip={avg_flip:.4f} combined={entry.get(\"combined\", \"?\"):.2f}' if 'combined' in entry else f'  ({i},{j}): flip={avg_flip:.4f}', flush=True)

    results.append(entry)

    if idx % 5 == 0:
        # Save checkpoint
        with open(f'{SAVE_DIR}/checkpoint.json', 'w') as f:
            json.dump(results, f, indent=2)

# ======================================================================
# Correlation Analysis
# ======================================================================
print('\\n=== Correlation Analysis ===', flush=True)

# Filter to entries with both flip rate and combined score
valid = [r for r in results if 'combined' in r and r['gate_flip_rate'] >= 0]
print(f'Valid entries: {len(valid)}', flush=True)

if len(valid) >= 5:
    flips = [r['gate_flip_rate'] for r in valid]
    combineds = [r['combined'] for r in valid]

    sr, sp = stats.spearmanr(flips, combineds)
    print(f'Gate flip rate vs combined: Spearman r={sr:.3f}, p={sp:.4f}', flush=True)
    print(f'Direction: {\"high flip = bad (expected)\" if sr < 0 else \"high flip = good (unexpected)\"}', flush=True)

    # Also load SBUID data for comparison if available
    sbuid_file = 'results/data/gemma3_27b/sbuid_validation/results.json'
    if os.path.exists(sbuid_file):
        with open(sbuid_file) as f:
            sbuid_data = json.load(f)
        print(f'\\nComparison with SBUID:', flush=True)
        print(f'  SBUID correlation: loaded from file', flush=True)

    print(f'\\n=== SUMMARY ===', flush=True)
    print(f'Gate flip rate as Tier 1 screener on Gemma3-27B:', flush=True)
    print(f'  Spearman r = {sr:.3f}', flush=True)
    print(f'  p-value = {sp:.4f}', flush=True)
    print(f'  Significant (p<0.05): {sp < 0.05}', flush=True)
    print(f'  SBUID on same model: r=-0.25, p=0.29 (not significant)', flush=True)
    print(f'  Gate flip is {\"BETTER\" if abs(sr) > 0.25 and sp < 0.1 else \"NOT BETTER\"} than SBUID on Gemma3', flush=True)

# Save final
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump({
        'n_blocks': len(results),
        'n_valid': len(valid),
        'correlation': {'spearman_r': float(sr), 'p_value': float(sp)} if len(valid) >= 5 else None,
        'blocks': results,
    }, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
