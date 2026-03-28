#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_sbuid_%j.log
#SBATCH --job-name=g3_sbuid

# SBUID screening validation on Gemma3-27B
# Compute SBUID for all single blocks, compare with actual dual-probe scores
# Tests cross-architecture generalization of the screening metric

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SBUID Validation on Gemma3-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from scipy import stats
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/sbuid_validation'
os.makedirs(SAVE_DIR, exist_ok=True)

print('Loading model...', flush=True)
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

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Calibration prompts for spectral analysis
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

# ======================================================================
# 1. Compute SBUID for all single-layer blocks
# ======================================================================
print('\\n=== Computing SBUID for single-layer blocks ===', flush=True)

sbuid_data = []
for start in range(N - 1):
    block = (start, start + 1)
    rhos = []
    blood_impacts = []

    for prompt in cal_prompts[:4]:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            # Baseline logits
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            # Measure hidden states at block exit (BLOOD)
            # Hook to capture output of the duplicated block's last layer
            hidden_before = [None]
            hidden_after = [None]
            counter = [0]

            def make_blood_hook(ctr, hb, ha):
                def hook(module, input, output):
                    ctr[0] += 1
                    h = output[0] if isinstance(output, tuple) else output
                    if ctr[0] % 2 == 1:  # first pass
                        hb[0] = h.detach().float()
                    else:  # second pass
                        ha[0] = h.detach().float()
                    return output
                return hook

            hook = original_layers[start].register_forward_hook(
                make_blood_hook(counter, hidden_before, hidden_after)
            )

            # Duplicated forward
            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))

            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()

            hook.remove()
            inner.layers = nn.ModuleList(original_layers)
            set_num_layers(N)

            # Displacement rho
            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8:
                rhos.append(num / den)

            # BLOOD impact
            if hidden_before[0] is not None and hidden_after[0] is not None:
                diff = torch.norm(hidden_after[0] - hidden_before[0]).item()
                blood_impacts.append(diff)

    mean_rho = float(np.mean(rhos)) if rhos else 1.0
    mean_blood = float(np.mean(blood_impacts)) if blood_impacts else 0.0
    sbuid = mean_blood - 6000 * mean_rho  # SBUID formula from 72B

    sbuid_data.append({
        'block': list(block),
        'layer': start,
        'rho': mean_rho,
        'blood_impact': mean_blood,
        'sbuid': sbuid,
    })

    if start % 10 == 0:
        print(f'  Layer {start:3d}: rho={mean_rho:.4f} blood={mean_blood:.1f} sbuid={sbuid:.1f}', flush=True)

print(f'Computed SBUID for {len(sbuid_data)} blocks', flush=True)

# ======================================================================
# 2. Evaluate top and bottom SBUID blocks with dual probe
# ======================================================================
print('\\n=== Evaluating top/bottom SBUID blocks ===', flush=True)

sbuid_data.sort(key=lambda x: x['sbuid'], reverse=True)
# Test top 15 and bottom 5
to_test = sbuid_data[:15] + sbuid_data[-5:]

for entry in to_test:
    block = [tuple(entry['block'])]
    order = build_order(block, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0

    inner.layers = nn.ModuleList(original_layers)
    set_num_layers(N)

    entry['math'] = math_r['score']
    entry['eq'] = eq_r['score']
    entry['combined'] = combined
    print(f'  Layer {entry[\"layer\"]:3d}: sbuid={entry[\"sbuid\"]:7.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)

# ======================================================================
# 3. Compute correlation
# ======================================================================
print('\\n=== SBUID Correlation ===', flush=True)

evaluated = [e for e in sbuid_data if 'combined' in e]
if len(evaluated) >= 5:
    sbuids = [e['sbuid'] for e in evaluated]
    combineds = [e['combined'] for e in evaluated]
    rho_vals = [e['rho'] for e in evaluated]
    blood_vals = [e['blood_impact'] for e in evaluated]

    sr, sp = stats.spearmanr(sbuids, combineds)
    print(f'SBUID vs combined: Spearman r={sr:.3f}, p={sp:.4f}', flush=True)

    sr2, sp2 = stats.spearmanr(rho_vals, combineds)
    print(f'Rho vs combined:   Spearman r={sr2:.3f}, p={sp2:.4f}', flush=True)

    sr3, sp3 = stats.spearmanr(blood_vals, combineds)
    print(f'BLOOD vs combined: Spearman r={sr3:.3f}, p={sp3:.4f}', flush=True)

    # Also try different lambda values
    print(f'\\nLambda sweep:', flush=True)
    best_lambda = 6000
    best_r = abs(sr)
    for lam in [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]:
        sbuid_lam = [e['blood_impact'] - lam * e['rho'] for e in evaluated]
        sr_l, sp_l = stats.spearmanr(sbuid_lam, combineds)
        sig = '*' if sp_l < 0.05 else ''
        print(f'  lambda={lam:6d}: r={sr_l:.3f} p={sp_l:.4f} {sig}', flush=True)
        if abs(sr_l) > best_r:
            best_r = abs(sr_l)
            best_lambda = lam

    print(f'\\nBest lambda for Gemma3: {best_lambda} (r={best_r:.3f})', flush=True)

# Save
results = {
    'date': datetime.now().isoformat(),
    'model': 'gemma-3-27b-it',
    'num_layers': N,
    'sbuid_all': sbuid_data,
    'evaluated': evaluated,
    'correlation': {
        'sbuid_spearman_r': float(sr) if len(evaluated) >= 5 else None,
        'sbuid_spearman_p': float(sp) if len(evaluated) >= 5 else None,
        'n_evaluated': len(evaluated),
    },
}

with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
