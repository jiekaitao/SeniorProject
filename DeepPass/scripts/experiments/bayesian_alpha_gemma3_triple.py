#!/usr/bin/env python3
"""
Bayesian Alpha Optimization for Gemma3-27B Triple (0,2)+(12,13)+(47,48)

Uses Optuna TPE sampler with 2-GPU parallel workers via shared SQLite study.
Reduced probes (10 math + 10 EQ-bench) for search speed, full probes for validation.

Usage:
    # From sbatch (2 workers launched in parallel):
    CUDA_VISIBLE_DEVICES=0 python script.py --n-trials 20 --worker 0 &
    CUDA_VISIBLE_DEVICES=1 python script.py --n-trials 20 --worker 1 &
    wait
    python script.py --validate --gpu 0
"""

import sys, os, json, time, argparse
import torch
import torch.nn as nn

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe, MATH_QUESTIONS
from eq_bench_probe import run_eq_bench_probe

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_PATH = 'models/full/gemma-3-27b-it'
BLOCKS = [(0, 2), (12, 13), (47, 48)]
RESULTS_DIR = 'results/data/gemma3_27b/bayesian_alpha_triple'
STUDY_DB_PATH = os.path.join(RESULTS_DIR, 'optuna_study.db')

# Alpha params: one per duplicated layer
# Block (0,2) → layers 0,1 → alpha_L0, alpha_L1
# Block (12,13) → layer 12 → alpha_L12
# Block (47,48) → layer 47 → alpha_L47
ALPHA_NAMES = ['alpha_L0', 'alpha_L1', 'alpha_L12', 'alpha_L47']
ALPHA_RANGE = (0.0, 2.0)

# Reduced probes for fast search (10 math + 10 EQ-bench ≈ 110s per eval)
MATH_SUBSET = MATH_QUESTIONS[:10]


def build_order(blocks, N):
    """Build execution order with duplicated blocks."""
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order


def get_inner_model(model):
    """Navigate Gemma3's nested model structure to find the layer container."""
    inner = model.model
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    return inner


def set_num_layers(model, n):
    """Update num_hidden_layers in config (handles text_config nesting)."""
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n


def setup_model_with_alpha_hooks(model, blocks, alpha_containers):
    """
    Apply multi-block duplication and register alpha blending hooks.

    Alpha blending at each duplicated layer's second pass:
        h_out = h_in + alpha * (h_layer_out - h_in)

    alpha_containers: list of [alpha_value] mutable lists (one per dup layer).
    Updated in-place between trials; hooks auto-use new values.

    Returns: (original_layers, hooks, orig_N)
    """
    inner = get_inner_model(model)
    original_layers = list(inner.layers)
    N = len(original_layers)

    # Map: duplicated layer_idx → alpha_containers index
    sorted_blocks = sorted(blocks)
    dup_layer_to_alpha_idx = {}
    alpha_idx = 0
    for (i, j) in sorted_blocks:
        for l in range(i, j):
            dup_layer_to_alpha_idx[l] = alpha_idx
            alpha_idx += 1

    # Swap ModuleList to duplicated order
    order = build_order(blocks, N)
    new_layers = [original_layers[idx] for idx in order]
    inner.layers = nn.ModuleList(new_layers)
    set_num_layers(model, len(new_layers))

    # Register alpha hooks on each duplicated layer module
    # Hook fires on every call to that module's forward.
    # Shared module fires 2x per model forward (1st pass + duplicate).
    # Counter tracks: odd fire = 1st pass (no alpha), even fire = 2nd pass (apply alpha).
    hooks = []
    for layer_idx, a_idx in dup_layer_to_alpha_idx.items():
        module = original_layers[layer_idx]
        counter = [0]
        ac = alpha_containers[a_idx]

        def make_hook(ctr, ac):
            def hook(module, input, output):
                ctr[0] += 1
                if ctr[0] % 2 == 0:  # second pass → apply alpha blending
                    h_in = input[0]
                    if isinstance(output, tuple):
                        h_out = output[0]
                        blended = h_in + ac[0] * (h_out - h_in)
                        return (blended,) + output[1:]
                    else:
                        blended = h_in + ac[0] * (output - h_in)
                        return blended
                return output
            return hook

        h = module.register_forward_hook(make_hook(counter, ac))
        hooks.append(h)

    print(f'Applied {len(BLOCKS)} blocks, {len(hooks)} alpha hooks, '
          f'{len(order)} total layers (was {N})', flush=True)
    return original_layers, hooks, N


def run_worker(args):
    """Run Optuna trials on the visible GPU."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    worker_id = args.worker

    print(f'[W{worker_id}] Loading Gemma3-27B...', flush=True)
    t_load = time.time()
    model, tokenizer = load_original_model(MODEL_PATH)
    print(f'[W{worker_id}] Model loaded in {time.time()-t_load:.0f}s', flush=True)

    # Load EQ-bench questions once (avoid re-downloading per trial)
    from eq_bench_probe import _load_questions
    eq_all = _load_questions()
    eq_subset = eq_all[:10]

    # Create mutable alpha containers (hooks read from these)
    alpha_containers = [[1.0] for _ in ALPHA_NAMES]

    # Apply duplication + hooks
    original_layers, hooks, orig_N = setup_model_with_alpha_hooks(
        model, BLOCKS, alpha_containers
    )

    def gen(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    def gen_long(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    # Sanity check: baseline with all alphas = 1.0
    print(f'[W{worker_id}] Sanity check (all alphas=1.0)...', flush=True)
    t0 = time.time()
    math_r = run_math_probe(gen, questions=MATH_SUBSET, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
    baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'[W{worker_id}] Baseline (reduced probe): math={math_r["score"]:.4f} '
          f'eq={eq_r["score"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)',
          flush=True)

    # Create/load shared Optuna study
    storage = f'sqlite:///{STUDY_DB_PATH}'
    study = optuna.create_study(
        study_name='gemma3_triple_alpha',
        storage=storage,
        sampler=TPESampler(seed=42 + worker_id),
        direction='maximize',
        load_if_exists=True,
    )

    # Also seed the study with the known good config (all 1.0 = 87.80)
    if worker_id == 0 and len(study.trials) == 0:
        study.enqueue_trial({n: 1.0 for n in ALPHA_NAMES})

    trial_times = []

    def objective(trial):
        alphas = []
        for name in ALPHA_NAMES:
            a = trial.suggest_float(name, ALPHA_RANGE[0], ALPHA_RANGE[1])
            alphas.append(a)

        # Update alpha containers in-place (hooks read from these)
        for i, a in enumerate(alphas):
            alpha_containers[i][0] = a

        t0 = time.time()
        math_r = run_math_probe(gen, questions=MATH_SUBSET, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, questions=eq_subset, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        elapsed = time.time() - t0
        trial_times.append(elapsed)

        alpha_str = ' '.join(f'{n}={a:.3f}' for n, a in zip(ALPHA_NAMES, alphas))
        total_done = len(study.trials)
        print(f'[W{worker_id}] Trial {trial.number} ({total_done} total): '
              f'{alpha_str} => combined={combined:.2f} ({elapsed:.0f}s)', flush=True)

        return combined

    n = args.n_trials
    print(f'[W{worker_id}] Starting {n} Optuna trials...', flush=True)
    t_start = time.time()
    study.optimize(objective, n_trials=n)
    total_time = time.time() - t_start

    # Report worker results
    best = study.best_trial
    avg_trial = sum(trial_times) / len(trial_times) if trial_times else 0
    print(f'\n[W{worker_id}] === DONE === {n} trials in {total_time:.0f}s '
          f'(avg {avg_trial:.0f}s/trial)', flush=True)
    print(f'[W{worker_id}] Best so far: combined={best.value:.2f}', flush=True)
    for name in ALPHA_NAMES:
        print(f'  {name} = {best.params[name]:.4f}', flush=True)

    # Clean up
    for h in hooks:
        h.remove()


def validate(args):
    """Validate top configs from the Optuna study with full probes."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load study
    storage = f'sqlite:///{STUDY_DB_PATH}'
    study = optuna.load_study(study_name='gemma3_triple_alpha', storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f'\n{"="*70}')
    print(f'BAYESIAN ALPHA OPTIMIZATION — VALIDATION PHASE')
    print(f'Triple: {BLOCKS}')
    print(f'Total search trials: {len(completed)}')
    print(f'{"="*70}')

    # Show top 10 from search
    print(f'\nTop 10 from search (reduced probes):')
    for t in completed[:10]:
        alphas = [f'{t.params[n]:.3f}' for n in ALPHA_NAMES]
        print(f'  #{t.number}: combined={t.value:.2f}  alphas=[{", ".join(alphas)}]')

    # Validate top 5 with full probes
    print(f'\nLoading model for full validation...', flush=True)
    model, tokenizer = load_original_model(MODEL_PATH)

    from eq_bench_probe import _load_questions
    eq_all = _load_questions()

    alpha_containers = [[1.0] for _ in ALPHA_NAMES]
    original_layers, hooks, orig_N = setup_model_with_alpha_hooks(
        model, BLOCKS, alpha_containers
    )

    def gen(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    def gen_long(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    # Also evaluate baseline (no duplication)
    inner = get_inner_model(model)

    # Validate top 5 + all-1.0 baseline
    configs_to_validate = []
    configs_to_validate.append(('all_1.0', {n: 1.0 for n in ALPHA_NAMES}))
    for t in completed[:5]:
        configs_to_validate.append((f'trial_{t.number}', {n: t.params[n] for n in ALPHA_NAMES}))

    validation_results = []
    print(f'\nValidating {len(configs_to_validate)} configs with FULL probes:')
    for name, params in configs_to_validate:
        for i, n in enumerate(ALPHA_NAMES):
            alpha_containers[i][0] = params[n]

        t0 = time.time()
        math_r = run_math_probe(gen, verbose=False)
        eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
        combined = math_r['score'] * 50 + eq_r['score'] * 0.5
        elapsed = time.time() - t0

        alpha_str = ' '.join(f'{n}={params[n]:.3f}' for n in ALPHA_NAMES)
        print(f'  {name:15s}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} '
              f'combined={combined:.2f} ({elapsed:.0f}s)  [{alpha_str}]', flush=True)

        validation_results.append({
            'name': name,
            'alphas': params,
            'math': math_r['score'],
            'eq': eq_r['score'],
            'combined': combined,
        })

    # Sort by combined
    validation_results.sort(key=lambda x: x['combined'], reverse=True)

    best = validation_results[0]
    print(f'\n{"="*70}')
    print(f'BEST VALIDATED CONFIG: {best["name"]}')
    print(f'  Combined: {best["combined"]:.2f}')
    print(f'  Math:     {best["math"]:.4f}')
    print(f'  EQ-bench: {best["eq"]:.1f}')
    print(f'  Alphas:')
    for n in ALPHA_NAMES:
        print(f'    {n} = {best["alphas"][n]:.4f}')
    print(f'{"="*70}')

    # Save full results
    results = {
        'blocks': [list(b) for b in BLOCKS],
        'alpha_names': ALPHA_NAMES,
        'n_search_trials': len(completed),
        'search_top_10': [
            {'trial': t.number, 'combined': t.value,
             'alphas': {n: t.params[n] for n in ALPHA_NAMES}}
            for t in completed[:10]
        ],
        'validation': validation_results,
        'best': best,
    }
    outpath = os.path.join(RESULTS_DIR, 'results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {outpath}')

    # Clean up
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, default=0, help='Worker ID (for seed)')
    parser.add_argument('--n-trials', type=int, default=20, help='Trials per worker')
    parser.add_argument('--validate', action='store_true', help='Validate top configs')
    args = parser.parse_args()

    if args.validate:
        validate(args)
    else:
        run_worker(args)
