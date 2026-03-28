#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_g3_fixed_%j.log
#SBATCH --job-name=g3_fix

# Fixed lm-eval for Gemma3-27B with layer duplication
# Fixes:
#   1. Update layer_types to match duplicated layer order
#   2. Disable cache for ALL calls (forward + generate), not just generate
#   3. Patch inner model's config too (text_config)
# Tests with 2 samples first, then runs full 15% subsample

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval Gemma3 FIXED (sliding window + cache fix) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

BLOCKS = [(0, 2), (12, 13), (47, 48)]
BEST_ALPHAS = {0: 0.8797, 1: 0.8063, 12: 1.4507, 47: 0.9453}
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
SAVE_DIR = 'results/data/gemma3_27b/lm_eval'
os.makedirs(SAVE_DIR, exist_ok=True)

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def get_inner(model):
    inner = model.model
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    return inner

def apply_duplication_with_fixes(model, blocks, alphas=None):
    \"\"\"Apply layer duplication with full Gemma3 sliding window fix.\"\"\"
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)

    if not blocks:
        return [], N

    order = build_order(blocks, N)
    new_N = len(order)

    # 1. Swap ModuleList
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])

    # 2. Update num_hidden_layers in ALL config locations
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None:
            continue
        if hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = new_N

        # 3. FIX: Update layer_types to match new order
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            orig_types = cfg.layer_types
            new_types = [orig_types[idx] for idx in order]
            cfg.layer_types = new_types
            print(f'  Updated layer_types: {len(orig_types)} -> {len(new_types)}', flush=True)
            # Verify: count types
            n_full = sum(1 for t in new_types if t == 'full_attention')
            n_slide = sum(1 for t in new_types if t == 'sliding_attention')
            print(f'  Types: {n_full} full + {n_slide} sliding = {n_full+n_slide}', flush=True)

        # 4. FIX: Disable cache globally (prevents DynamicCache creation)
        if hasattr(cfg, 'use_cache'):
            cfg.use_cache = False

    # Also set on the inner model's config if it has its own
    if hasattr(inner, 'config'):
        inner.config.use_cache = False
        if hasattr(inner.config, 'layer_types') and inner.config.layer_types:
            inner.config.layer_types = [inner.config.layer_types[idx] if idx < len(inner.config.layer_types) else inner.config.layer_types[idx % len(inner.config.layer_types)] for idx in order]
            inner.config.num_hidden_layers = new_N

    print(f'  Applied duplication: {N} -> {new_N} layers', flush=True)

    # 5. Apply alpha hooks if needed
    hooks = []
    if alphas:
        sorted_blocks = sorted(blocks)
        dup_layers = []
        for (i, j) in sorted_blocks:
            for l in range(i, j):
                dup_layers.append(l)

        for layer_idx in dup_layers:
            alpha_val = alphas.get(layer_idx, 1.0)
            if abs(alpha_val - 1.0) < 1e-6:
                continue
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
                        return h_in + ac[0] * (output - h_in)
                    return output
                return hook
            h = module.register_forward_hook(make_hook(counter, ac))
            hooks.append(h)
        print(f'  Registered {len(hooks)} alpha hooks', flush=True)

    return hooks, N

def setup_lm_eval(model, tokenizer):
    \"\"\"Wrap model for lm-eval with cache disabled everywhere.\"\"\"
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    # Patch generate to disable cache
    orig_gen = model.generate
    def patched_gen(*a, **kw):
        kw['use_cache'] = False
        return orig_gen(*a, **kw)
    model.generate = patched_gen

    # Patch forward to disable cache for loglikelihood calls
    orig_forward = model.forward
    @functools.wraps(orig_forward)
    def patched_forward(*args, **kwargs):
        kwargs['use_cache'] = False
        return orig_forward(*args, **kwargs)
    model.forward = patched_forward

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    return lm, evaluator

# ======================================================================
# Phase 1: Quick test (2 samples) to verify fix works
# ======================================================================
print('\\n' + '=' * 60, flush=True)
print('PHASE 1: Quick test (limit=2) on triple @1.0', flush=True)
print('=' * 60, flush=True)

model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
hooks, orig_N = apply_duplication_with_fixes(model, BLOCKS)

lm, evaluator = setup_lm_eval(model, tokenizer)

print('Running quick test (2 samples, BBH only)...', flush=True)
t0 = time.time()
try:
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=['leaderboard_bbh'],
        limit=2,
    )
    print(f'SUCCESS! Quick test passed in {time.time()-t0:.0f}s', flush=True)
    for k, v in sorted(results['results'].items()):
        for mk, mv in v.items():
            if isinstance(mv, (int, float)) and 'stderr' not in mk:
                print(f'  {k}/{mk}: {mv:.4f}', flush=True)
except Exception as e:
    print(f'FAILED: {e}', flush=True)
    print('\\nAttempting fallback: set layer_types=None...', flush=True)

    # Fallback: remove layer_types entirely (forces full attention everywhere)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'layer_types'):
            cfg.layer_types = None
    if hasattr(get_inner(model), 'config') and hasattr(get_inner(model).config, 'layer_types'):
        get_inner(model).config.layer_types = None

    # Need new HFLM since model changed
    del lm
    lm, evaluator = setup_lm_eval(model, tokenizer)

    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=['leaderboard_bbh'],
            limit=2,
        )
        print(f'FALLBACK SUCCESS! in {time.time()-t0:.0f}s', flush=True)
    except Exception as e2:
        print(f'FALLBACK ALSO FAILED: {e2}', flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Clean up test
for h in hooks:
    h.remove()
del model, tokenizer, lm
gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# Phase 2: Full benchmark on all configs
# ======================================================================
configs = [
    ('triple_alpha1', BLOCKS, {}),
    ('triple_alpha_tuned', BLOCKS, BEST_ALPHAS),
]

for config_name, blocks, alphas in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'\\nSkipping {config_name} — already exists', flush=True)
        continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)
    print(f'  Blocks: {blocks}', flush=True)
    if alphas:
        print(f'  Alphas: {alphas}', flush=True)
    print(f'{\"=\" * 60}', flush=True)

    model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
    hooks, orig_N = apply_duplication_with_fixes(model, blocks, alphas if alphas else None)
    lm, evaluator = setup_lm_eval(model, tokenizer)

    task_list = TASKS.split(',')
    LIMIT = 0.15
    print(f'Tasks: {task_list}  Limit: {LIMIT}', flush=True)

    t0 = time.time()
    try:
        results = evaluator.simple_evaluate(model=lm, tasks=task_list, limit=LIMIT)
        elapsed = time.time() - t0

        scores = {}
        for task, data in results['results'].items():
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    scores[f'{task}/{metric}'] = value

        print(f'\\n=== RESULTS ({config_name}) === [{elapsed:.0f}s]', flush=True)
        for k, v in sorted(scores.items()):
            if 'stderr' not in k:
                print(f'  {k}: {v:.4f}', flush=True)

        with open(outpath, 'w') as f:
            json.dump({
                'config': config_name, 'blocks': [list(b) for b in blocks],
                'alphas': {str(k): v for k, v in alphas.items()} if alphas else {},
                'tasks': task_list, 'limit': LIMIT,
                'scores': scores, 'elapsed_s': elapsed,
                'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results['results'].items()},
            }, f, indent=2)
        print(f'SAVED to {outpath}', flush=True)

    except Exception as e:
        print(f'FAILED on {config_name}: {e}', flush=True)
        import traceback
        traceback.print_exc()

    for h in hooks:
        h.remove()
    del model, tokenizer, lm
    gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Finished: $(date) ==="
