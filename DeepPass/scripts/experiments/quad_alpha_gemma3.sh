#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_quad_alpha_g3_%j.log
#SBATCH --job-name=g3_qalpha

# Alpha-tune the best quad: (0,2)+(12,13)+(47,48)+(54,55)
# 5 params: L0, L1, L12, L47, L54
# Bayesian optimization with 40 trials

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Quad Alpha Optimization: (0,2)+(12,13)+(47,48)+(54,55) ==="
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
BLOCKS = [(0, 2), (12, 13), (47, 48), (54, 55)]
SAVE_DIR = 'results/data/gemma3_27b/quad_alpha'
os.makedirs(SAVE_DIR, exist_ok=True)

ALPHA_NAMES = ['alpha_L0', 'alpha_L1', 'alpha_L12', 'alpha_L47', 'alpha_L54']

print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
original_layers = list(inner.layers)
N = len(original_layers)

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

# Apply duplication
sorted_blocks = sorted(BLOCKS)
dup_layers = []
for (i, j) in sorted_blocks:
    for l in range(i, j):
        dup_layers.append(l)

alpha_containers = [[1.0] for _ in dup_layers]

order = build_order(BLOCKS, N)
inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
set_num_layers(len(order))

# Register alpha hooks
hooks = []
for idx, layer_idx in enumerate(dup_layers):
    module = original_layers[layer_idx]
    counter = [0]
    ac = alpha_containers[idx]
    def make_hook(ctr, ac):
        def hook(module, input, output):
            ctr[0] += 1
            if ctr[0] % 2 == 0:
                h_in = input[0]
                if isinstance(output, tuple):
                    h_out = output[0]
                    blended = h_in + ac[0] * (h_out - h_in)
                    return (blended,) + output[1:]
                return h_in + ac[0] * (output - h_in)
            return output
        return hook
    h = module.register_forward_hook(make_hook(counter, ac))
    hooks.append(h)
print(f'Applied quad: {N} -> {len(order)} layers, {len(hooks)} alpha hooks', flush=True)

def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

# Sanity check
print('Sanity check (all alphas=1.0)...', flush=True)
t0 = time.time()
math_r = run_math_probe(gen, questions=math_subset, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Baseline (reduced): combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)

study = optuna.create_study(
    study_name='gemma3_quad_alpha',
    storage=f'sqlite:///{SAVE_DIR}/optuna_study.db',
    sampler=TPESampler(seed=42),
    direction='maximize',
    load_if_exists=True,
)

# Seed with known good configs
study.enqueue_trial({n: 1.0 for n in ALPHA_NAMES})
# Triple best alphas + 0.3 whisper for L54
study.enqueue_trial({
    'alpha_L0': 0.88, 'alpha_L1': 0.81, 'alpha_L12': 1.45,
    'alpha_L47': 0.95, 'alpha_L54': 0.3,
})

def objective(trial):
    params = {}
    for name in ALPHA_NAMES:
        params[name] = trial.suggest_float(name, 0.0, 2.0)
    for i, layer_idx in enumerate(dup_layers):
        alpha_containers[i][0] = params[f'alpha_L{layer_idx}']
    t0 = time.time()
    math_r = run_math_probe(gen, questions=math_subset, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    a_str = ' '.join(f'{n}={params[n]:.2f}' for n in ALPHA_NAMES)
    print(f'Trial {trial.number}: {a_str} => combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return combined

N_TRIALS = 40
print(f'Starting {N_TRIALS} trials...', flush=True)
study.optimize(objective, n_trials=N_TRIALS)

# Report and save
best = study.best_trial
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
completed.sort(key=lambda t: t.value, reverse=True)

print(f'\\nBest: combined={best.value:.2f}', flush=True)
for n in ALPHA_NAMES:
    print(f'  {n} = {best.params[n]:.4f}', flush=True)

# Validate top 3 with full probes
print('\\nValidating top 3 with full probes...', flush=True)
val_results = []
for t in completed[:3]:
    for i, layer_idx in enumerate(dup_layers):
        alpha_containers[i][0] = t.params[f'alpha_L{layer_idx}']
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  trial {t.number}: combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    val_results.append({
        'trial': t.number, 'search_combined': t.value,
        'validated_combined': combined, 'math': math_r['score'], 'eq': eq_r['score'],
        'params': {n: t.params[n] for n in ALPHA_NAMES},
    })

val_results.sort(key=lambda x: x['validated_combined'], reverse=True)
results = {
    'blocks': [list(b) for b in BLOCKS],
    'n_trials': len(completed),
    'best_search': {'combined': best.value, 'params': {n: best.params[n] for n in ALPHA_NAMES}},
    'validation': val_results,
    'best_validated': val_results[0] if val_results else None,
}
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
