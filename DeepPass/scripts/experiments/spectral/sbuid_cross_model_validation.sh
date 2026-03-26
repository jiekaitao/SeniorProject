#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sbuid_xmodel_%j.log
#SBATCH --job-name=deeppass_sxm

# Validate SBUID_0 screening metric across models
# SBUID_0 = BLOOD_impact - lambda * rho works on 72B (p=0.008)
# Does it generalize to 7B, 9B, 27B?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SBUID Cross-Model Validation ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, gc, torch, torch.nn as nn, numpy as np
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

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

def validate_sbuid_on_model(model_path, model_name, n_test_blocks=15):
    print(f'\\n{\"=\" * 70}')
    print(f'MODEL: {model_name}')
    print(f'{\"=\" * 70}', flush=True)

    model, tokenizer = load_original_model(model_path)
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    attr = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr))
    N = len(original_layers)
    device = next(model.parameters()).device
    print(f'  Loaded: {N} layers', flush=True)

    def set_nl(n):
        if hasattr(model.config, 'text_config'):
            model.config.text_config.num_hidden_layers = n
        elif hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = n

    # Generate test blocks spanning the model
    test_blocks = []
    for start in range(0, N - 3, max(1, N // n_test_blocks)):
        for size in [3, 5, 7]:
            end = start + size
            if end <= N and (start, end) not in test_blocks:
                test_blocks.append((start, end))
                break
    test_blocks = test_blocks[:n_test_blocks]
    print(f'  Testing {len(test_blocks)} blocks', flush=True)

    # Compute rho for each block
    rhos = {}
    for block in test_blocks:
        i, j = block
        block_rhos = []
        for prompt in CAL_PROMPTS[:4]:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                out_base = model(ids['input_ids'], use_cache=False)
                logits_base = out_base.logits[:, -1, :].float()
                order = build_order([block], N)
                setattr(inner, attr, nn.ModuleList([original_layers[idx] for idx in order]))
                set_nl(len(order))
                out_dup = model(ids['input_ids'], use_cache=False)
                logits_dup = out_dup.logits[:, -1, :].float()
                setattr(inner, attr, nn.ModuleList(original_layers))
                set_nl(N)
                num = torch.norm(logits_dup - logits_base).item()
                den = torch.norm(logits_base).item()
                if den > 1e-8:
                    block_rhos.append(num / den)
        rhos[block] = float(np.mean(block_rhos)) if block_rhos else 1.0

    # Compute BLOOD for each block
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
        for h in hooks:
            h.remove()
        return [float(np.mean(ns)) if ns else 0.0 for ns in layer_norms]

    setattr(inner, attr, nn.ModuleList(original_layers))
    set_nl(N)
    base_blood = compute_blood_profile()

    bloods = {}
    for block in test_blocks:
        i, j = block
        order = build_order([block], N)
        setattr(inner, attr, nn.ModuleList([original_layers[idx] for idx in order]))
        set_nl(len(order))
        dup_blood = compute_blood_profile()
        setattr(inner, attr, nn.ModuleList(original_layers))
        set_nl(N)
        impact = sum(base_blood[l] - dup_blood[l + (j-i)] for l in range(j, N) if l + (j-i) < len(dup_blood))
        bloods[block] = impact

    # Evaluate each block with dual probe
    gen = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    gen_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
    math_base = run_math_probe(gen, verbose=False)
    eq_base = run_eq_bench_probe(gen_long, verbose=False)
    baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
    print(f'  Baseline: {baseline:.2f}', flush=True)

    deltas = {}
    for bi, block in enumerate(test_blocks):
        i, j = block
        order = build_order([block], N)
        setattr(inner, attr, nn.ModuleList([original_layers[idx] for idx in order]))
        set_nl(len(order))
        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        deltas[block] = combined - baseline
        setattr(inner, attr, nn.ModuleList(original_layers))
        set_nl(N)
        print(f'  [{bi+1}/{len(test_blocks)}] ({i:2d},{j:2d}): rho={rhos[block]:.4f} blood={bloods[block]:+.0f} delta={deltas[block]:+.2f}', flush=True)

    # Correlations
    d_list = [deltas[b] for b in test_blocks]
    r_list = [rhos[b] for b in test_blocks]
    b_list = [bloods[b] for b in test_blocks]

    # Find best lambda for SBUID
    best_r, best_p, best_lam = 0, 1, 0
    for lam in np.arange(0, 20000, 500):
        scores = [bloods[b] - lam * rhos[b] for b in test_blocks]
        r, p = spearmanr(scores, d_list)
        if r > best_r:
            best_r, best_p, best_lam = r, p, lam

    print(f'\\n  Rho vs delta: Spearman r={spearmanr(r_list, d_list)[0]:+.3f} (p={spearmanr(r_list, d_list)[1]:.4f})')
    print(f'  BLOOD vs delta: Spearman r={spearmanr(b_list, d_list)[0]:+.3f} (p={spearmanr(b_list, d_list)[1]:.4f})')
    print(f'  SBUID best: lam={best_lam:.0f} Spearman r={best_r:+.3f} (p={best_p:.4f})', flush=True)

    result = {
        'model': model_path, 'model_name': model_name, 'n_layers': N,
        'baseline': baseline, 'n_blocks': len(test_blocks),
        'sbuid_best_lambda': best_lam, 'sbuid_spearman_r': best_r, 'sbuid_p': best_p,
        'rho_spearman': spearmanr(r_list, d_list)[0], 'rho_p': spearmanr(r_list, d_list)[1],
        'blood_spearman': spearmanr(b_list, d_list)[0], 'blood_p': spearmanr(b_list, d_list)[1],
        'blocks': [{'block': list(b), 'rho': rhos[b], 'blood': bloods[b], 'delta': deltas[b]} for b in test_blocks],
    }

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result

all_results = {}

# Test on 3 models (7B is fast, 9B medium, 27B slower)
all_results['7B'] = validate_sbuid_on_model('models/small/Qwen2-7B-Instruct', 'Qwen2-7B', n_test_blocks=15)
all_results['9B'] = validate_sbuid_on_model('models/full/Qwen3.5-9B', 'Qwen3.5-9B', n_test_blocks=12)
all_results['27B'] = validate_sbuid_on_model('models/full/Qwen3.5-27B', 'Qwen3.5-27B', n_test_blocks=10)

# Summary
print(f'\\n{\"=\" * 70}')
print('CROSS-MODEL SBUID VALIDATION SUMMARY')
print(f'{\"=\" * 70}')
print(f'{\"Model\":>10s} {\"N\":>4s} {\"Rho r\":>8s} {\"Rho p\":>8s} {\"BLOOD r\":>8s} {\"BLOOD p\":>8s} {\"SBUID r\":>8s} {\"SBUID p\":>8s} {\"λ\":>8s}')
for name, r in all_results.items():
    print(f'{name:>10s} {r[\"n_layers\"]:4d} {r[\"rho_spearman\"]:+8.3f} {r[\"rho_p\"]:8.4f} '
          f'{r[\"blood_spearman\"]:+8.3f} {r[\"blood_p\"]:8.4f} '
          f'{r[\"sbuid_spearman_r\"]:+8.3f} {r[\"sbuid_p\"]:8.4f} {r[\"sbuid_best_lambda\"]:8.0f}', flush=True)

os.makedirs('results/data/sbuid_validation', exist_ok=True)
with open('results/data/sbuid_validation/cross_model.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
print(f'\\nSaved to results/data/sbuid_validation/cross_model.json', flush=True)
"

echo "=== Done at $(date) ==="
