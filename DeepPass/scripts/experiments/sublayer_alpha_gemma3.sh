#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sublayer_alpha_g3_%j.log
#SBATCH --job-name=g3_sub

# Per-sublayer (attention vs FFN) alpha optimization for Gemma3-27B
# Triple (0,2)+(12,13)+(47,48): 8 params (attn + ffn per duplicated layer)
# Tests FFN re-retrieval hypothesis: if ffn_alpha < attn_alpha, FFN is harmful

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Per-Sublayer Alpha Optimization: Gemma3-27B ==="
echo "Blocks: (0,2)+(12,13)+(47,48) — 8 params (attn+ffn per layer)"
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
BLOCKS = [(0, 2), (12, 13), (47, 48)]
SAVE_DIR = 'results/data/gemma3_27b/sublayer_alpha'
os.makedirs(SAVE_DIR, exist_ok=True)

# 8 parameters: attn + ffn per duplicated layer
PARAM_NAMES = [
    'attn_L0', 'ffn_L0',
    'attn_L1', 'ffn_L1',
    'attn_L12', 'ffn_L12',
    'attn_L47', 'ffn_L47',
]
ALPHA_RANGE = (0.0, 2.0)

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)

inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)

# Load probes
eq_all = _load_questions()
eq_subset = eq_all[:10]
math_subset = MATH_QUESTIONS[:10]

# Build execution order
def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

# Map duplicated layers to param indices
sorted_blocks = sorted(BLOCKS)
dup_layers = []  # list of layer indices that are duplicated
for (i, j) in sorted_blocks:
    for l in range(i, j):
        dup_layers.append(l)
# dup_layers = [0, 1, 12, 47]

# Mutable alpha containers: [attn_alpha, ffn_alpha] per duplicated layer
alpha_containers = [[1.0, 1.0] for _ in dup_layers]

# Apply duplication
order = build_order(BLOCKS, N)
new_layers = [original_layers[idx] for idx in order]
inner.layers = nn.ModuleList(new_layers)
if hasattr(model.config, 'text_config'):
    model.config.text_config.num_hidden_layers = len(new_layers)
if hasattr(model.config, 'num_hidden_layers'):
    model.config.num_hidden_layers = len(new_layers)
print(f'Applied {len(BLOCKS)} blocks: {N} -> {len(order)} layers', flush=True)

# Register sublayer hooks
# For each duplicated layer, hook self_attn and mlp separately
# On the second pass, scale the sublayer output by its alpha
hooks = []
for idx, layer_idx in enumerate(dup_layers):
    module = original_layers[layer_idx]
    ac = alpha_containers[idx]  # [attn_alpha, ffn_alpha]

    # Attention hook
    attn_counter = [0]
    def make_attn_hook(ctr, ac):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:  # second pass
                alpha = ac[0]  # attn_alpha
                if isinstance(output, tuple):
                    return (alpha * output[0],) + output[1:]
                return alpha * output
            return output
        return hook
    h = module.self_attn.register_forward_hook(make_attn_hook(attn_counter, ac))
    hooks.append(h)

    # MLP/FFN hook
    ffn_counter = [0]
    def make_ffn_hook(ctr, ac):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:  # second pass
                alpha = ac[1]  # ffn_alpha
                if isinstance(output, tuple):
                    return (alpha * output[0],) + output[1:]
                return alpha * output
            return output
        return hook
    h = module.mlp.register_forward_hook(make_ffn_hook(ffn_counter, ac))
    hooks.append(h)

print(f'Registered {len(hooks)} sublayer hooks ({len(dup_layers)} layers x 2)', flush=True)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Sanity: all alphas = 1.0 should match ~87.68
print('Sanity check (all sublayer alphas=1.0)...', flush=True)
t0 = time.time()
math_r = run_math_probe(gen, questions=math_subset, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Baseline: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)

# Bayesian optimization
study = optuna.create_study(
    study_name='gemma3_sublayer_alpha',
    storage=f'sqlite:///{SAVE_DIR}/optuna_study.db',
    sampler=TPESampler(seed=42),
    direction='maximize',
    load_if_exists=True,
)

# Seed with all-1.0 and known best per-layer config
study.enqueue_trial({n: 1.0 for n in PARAM_NAMES})
# From per-layer results: L0=0.88, L1=0.81, L12=1.45, L47=0.95
# Start sublayer search near these (same for attn and ffn)
study.enqueue_trial({
    'attn_L0': 0.88, 'ffn_L0': 0.88,
    'attn_L1': 0.81, 'ffn_L1': 0.81,
    'attn_L12': 1.45, 'ffn_L12': 1.45,
    'attn_L47': 0.95, 'ffn_L47': 0.95,
})
# Hypothesis: attn high, ffn low
study.enqueue_trial({
    'attn_L0': 1.0, 'ffn_L0': 0.5,
    'attn_L1': 1.0, 'ffn_L1': 0.5,
    'attn_L12': 1.5, 'ffn_L12': 0.8,
    'attn_L47': 1.0, 'ffn_L47': 0.5,
})

trial_times = []

def objective(trial):
    params = {}
    for name in PARAM_NAMES:
        params[name] = trial.suggest_float(name, ALPHA_RANGE[0], ALPHA_RANGE[1])

    # Update alpha containers
    for i, layer_idx in enumerate(dup_layers):
        attn_name = f'attn_L{layer_idx}'
        ffn_name = f'ffn_L{layer_idx}'
        alpha_containers[i][0] = params[attn_name]
        alpha_containers[i][1] = params[ffn_name]

    t0 = time.time()
    math_r = run_math_probe(gen, questions=math_subset, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    trial_times.append(elapsed)

    p_str = ' '.join(f'{n}={params[n]:.2f}' for n in PARAM_NAMES)
    print(f'Trial {trial.number} ({len(study.trials)} total): {p_str} => combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return combined

N_TRIALS = 60
print(f'\\nStarting {N_TRIALS} Optuna trials...', flush=True)
t_start = time.time()
study.optimize(objective, n_trials=N_TRIALS)
total_time = time.time() - t_start

best = study.best_trial
avg_t = sum(trial_times) / len(trial_times) if trial_times else 0
print(f'\\n=== SEARCH DONE === {N_TRIALS} trials in {total_time:.0f}s (avg {avg_t:.0f}s/trial)', flush=True)
print(f'Best search: combined={best.value:.2f}', flush=True)
for n in PARAM_NAMES:
    print(f'  {n} = {best.params[n]:.4f}', flush=True)

# Validate top 5 with FULL probes
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
completed.sort(key=lambda t: t.value, reverse=True)

print(f'\\n=== VALIDATION (full probes, top 5) ===', flush=True)
validation = []
for rank, t in enumerate(completed[:5]):
    for i, layer_idx in enumerate(dup_layers):
        alpha_containers[i][0] = t.params[f'attn_L{layer_idx}']
        alpha_containers[i][1] = t.params[f'ffn_L{layer_idx}']

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0

    p_str = ' '.join(f'{n}={t.params[n]:.2f}' for n in PARAM_NAMES)
    print(f'  #{rank+1} (trial {t.number}): math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    print(f'    {p_str}', flush=True)

    entry = {
        'trial': t.number, 'search_combined': t.value,
        'validated_math': math_r['score'], 'validated_eq': eq_r['score'],
        'validated_combined': combined,
        'params': {n: t.params[n] for n in PARAM_NAMES},
    }
    validation.append(entry)

# Analyze: is attn_alpha > ffn_alpha on average? (FFN hypothesis)
best_v = validation[0]
print(f'\\n=== BEST VALIDATED ===', flush=True)
print(f'Combined: {best_v[\"validated_combined\"]:.2f}', flush=True)
attn_avg = sum(best_v['params'][f'attn_L{l}'] for l in dup_layers) / len(dup_layers)
ffn_avg = sum(best_v['params'][f'ffn_L{l}'] for l in dup_layers) / len(dup_layers)
print(f'Avg attn_alpha: {attn_avg:.3f}', flush=True)
print(f'Avg ffn_alpha:  {ffn_avg:.3f}', flush=True)
print(f'FFN hypothesis supported: {ffn_avg < attn_avg}', flush=True)

results = {
    'blocks': [list(b) for b in BLOCKS],
    'dup_layers': dup_layers,
    'param_names': PARAM_NAMES,
    'n_trials': len(completed),
    'search_top_10': [
        {'trial': t.number, 'combined': t.value, 'params': {n: t.params[n] for n in PARAM_NAMES}}
        for t in completed[:10]
    ],
    'validation': validation,
    'best': best_v,
    'ffn_hypothesis': {
        'avg_attn_alpha': attn_avg,
        'avg_ffn_alpha': ffn_avg,
        'supported': bool(ffn_avg < attn_avg),
    },
}
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/results.json', flush=True)

# Clean up
for h in hooks:
    h.remove()
print('Done.', flush=True)
"

echo "=== Finished: $(date) ==="
