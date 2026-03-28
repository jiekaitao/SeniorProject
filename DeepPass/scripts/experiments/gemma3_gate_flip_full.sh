#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_gemma3_gate_full_%j.log
#SBATCH --job-name=g3_gfull

# Full validation: gate flip rate vs dual probe on ALL 62 single-layer blocks
# This gives n=62 for proper statistical power (p<0.05 needs ~n>20 for r>0.4)
# Also computes SBUID for each block for head-to-head comparison
# Saves checkpoint every 10 blocks

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Full Gate Flip Validation: Gemma3-27B (n=62) ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from scipy import stats

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe, MATH_QUESTIONS
from eq_bench_probe import run_eq_bench_probe, _load_questions

MODEL_PATH = 'models/full/gemma-3-27b-it'
SAVE_DIR = 'results/data/gemma3_27b/gate_flip_full'
os.makedirs(SAVE_DIR, exist_ok=True)

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

math_sub = MATH_QUESTIONS[:10]
eq_sub = _load_questions()[:10]

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

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

cal_prompts = [
    'What is 127 * 348?', 'What is 99999 * 99999?',
    'What is the capital of France?', 'Explain entropy simply.',
    'Who wrote Romeo and Juliet?', 'What is 2^16?',
]

# Check for existing checkpoint
checkpoint_file = f'{SAVE_DIR}/checkpoint.json'
if os.path.exists(checkpoint_file):
    with open(checkpoint_file) as f:
        results = json.load(f)
    done_blocks = {tuple(r['block']) for r in results}
    print(f'Resuming from checkpoint: {len(results)} blocks done', flush=True)
else:
    results = []
    done_blocks = set()

# Baseline
print('\\n=== Baseline ===', flush=True)
t0 = time.time()
math_r = run_math_probe(gen, questions=math_sub, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_sub, verbose=False)
baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Baseline: {baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)

# ======================================================================
# Evaluate ALL single-layer blocks: gate flip + SBUID + dual probe
# ======================================================================
print(f'\\n=== Evaluating {N-1} single-layer blocks ===', flush=True)

for start in range(N - 1):
    block = (start, start + 1)
    if block in done_blocks:
        continue

    entry = {'block': list(block), 'layer': start}

    # 1. Gate flip rate
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    module = original_layers[start]
    if hasattr(module.mlp, 'gate_proj'):
        gate_proj = module.mlp.gate_proj
        flip_rates = []
        for prompt in cal_prompts:
            gate_vals = []
            counter = [0]
            def make_cap(ctr, gv, gp):
                def hook(mod, inp, out):
                    ctr[0] += 1
                    with torch.no_grad():
                        i = inp[0] if isinstance(inp, tuple) else inp
                        gv.append(gp(i[:, -1, :]).cpu().float().squeeze(0))
                    return out
                return hook
            h = module.mlp.register_forward_hook(make_cap(counter, gate_vals, gate_proj))
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                model(inputs['input_ids'], use_cache=False)
            h.remove()
            if len(gate_vals) >= 2:
                flips = ((gate_vals[0] > 0) != (gate_vals[1] > 0)).float().mean().item()
                flip_rates.append(flips)
        entry['gate_flip_rate'] = float(np.mean(flip_rates)) if flip_rates else -1
    else:
        entry['gate_flip_rate'] = -1

    # 2. SBUID (rho + BLOOD)
    rhos = []
    bloods = []
    for prompt in cal_prompts[:4]:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            # Baseline logits
            restore()
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            # BLOOD hook
            hb = [None]; ha = [None]; ctr = [0]
            def make_bh(c, b, a):
                def hook(mod, inp, out):
                    c[0] += 1
                    h = out[0] if isinstance(out, tuple) else out
                    if c[0] % 2 == 1: b[0] = h.detach().float()
                    else: a[0] = h.detach().float()
                    return out
                return hook
            bh = original_layers[start].register_forward_hook(make_bh(ctr, hb, ha))

            order = build_order([block], N)
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            set_num_layers(len(order))
            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()
            bh.remove()
            restore()

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8: rhos.append(num / den)
            if hb[0] is not None and ha[0] is not None:
                bloods.append(torch.norm(ha[0] - hb[0]).item())

    entry['rho'] = float(np.mean(rhos)) if rhos else 1.0
    entry['blood'] = float(np.mean(bloods)) if bloods else 0.0
    entry['sbuid_6k'] = entry['blood'] - 6000 * entry['rho']

    # 3. Dual probe (reduced — 10 math + 10 eq)
    order = build_order([block], N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    set_num_layers(len(order))

    t0 = time.time()
    math_r = run_math_probe(gen, questions=math_sub, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_sub, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    restore()

    entry['math'] = math_r['score']
    entry['eq'] = eq_r['score']
    entry['combined'] = combined
    entry['delta'] = combined - baseline

    results.append(entry)
    done_blocks.add(block)

    print(f'  L{start:2d}: flip={entry[\"gate_flip_rate\"]:.4f} sbuid={entry[\"sbuid_6k\"]:8.0f} combined={combined:.2f} delta={entry[\"delta\"]:+.2f} ({elapsed:.0f}s)', flush=True)

    # Checkpoint every 10
    if len(results) % 10 == 0:
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  [CHECKPOINT] {len(results)} blocks saved', flush=True)

# ======================================================================
# Final Correlation Analysis
# ======================================================================
print(f'\\n{\"=\" * 60}', flush=True)
print(f'CORRELATION ANALYSIS (n={len(results)})', flush=True)
print(f'{\"=\" * 60}', flush=True)

valid = [r for r in results if r['gate_flip_rate'] >= 0 and 'combined' in r]
n = len(valid)

if n >= 10:
    flips = [r['gate_flip_rate'] for r in valid]
    sbuids = [r['sbuid_6k'] for r in valid]
    rhos = [r['rho'] for r in valid]
    bloods = [r['blood'] for r in valid]
    combineds = [r['combined'] for r in valid]

    metrics = {
        'gate_flip_rate': (flips, 'negative expected'),
        'sbuid_6k': (sbuids, 'positive expected'),
        'rho': (rhos, 'unclear'),
        'blood': (bloods, 'unclear'),
    }

    print(f'\\n{\"Metric\":>20s}  {\"Spearman r\":>10s}  {\"p-value\":>8s}  {\"Sig?\":>5s}  Direction')
    print('-' * 75)
    for name, (vals, expected) in metrics.items():
        sr, sp = stats.spearmanr(vals, combineds)
        sig = 'YES' if sp < 0.05 else 'no'
        print(f'{name:>20s}  {sr:>10.3f}  {sp:>8.4f}  {sig:>5s}  {expected}', flush=True)

    # Best metric
    best_name = max(metrics.keys(), key=lambda k: abs(stats.spearmanr(metrics[k][0], combineds)[0]))
    best_r, best_p = stats.spearmanr(metrics[best_name][0], combineds)
    print(f'\\nBest predictor: {best_name} (r={best_r:.3f}, p={best_p:.4f})', flush=True)

# Save final
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump({
        'baseline': baseline,
        'n_blocks': len(results),
        'correlations': {
            name: {
                'spearman_r': float(stats.spearmanr(vals, combineds)[0]),
                'p_value': float(stats.spearmanr(vals, combineds)[1]),
            }
            for name, (vals, _) in metrics.items()
        } if n >= 10 else {},
        'blocks': results,
    }, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
