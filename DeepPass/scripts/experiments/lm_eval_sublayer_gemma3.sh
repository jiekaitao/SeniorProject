#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_sublayer_g3_%j.log
#SBATCH --job-name=g3_sublm

# lm-eval with sublayer alpha control (attention vs FFN)
# Tests: attention-only, attention-heavy (ffn=0.2), and Bayesian-optimized sublayer
# Key question: does dampening FFN preserve MMLU-PRO while keeping reasoning gains?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval: Sublayer Alpha (Attention vs FFN) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

BLOCKS = [(0, 2), (12, 13), (47, 48)]
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

def apply_dup_with_sublayer(model, blocks, sublayer_alphas):
    \"\"\"Apply duplication with per-sublayer alpha hooks.
    sublayer_alphas: dict of layer_idx -> (attn_alpha, ffn_alpha)
    \"\"\"
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)
    order = build_order(blocks, N)

    # Swap ModuleList
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])

    # Update configs
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = len(order)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[idx] for idx in order]
        if hasattr(cfg, 'use_cache'): cfg.use_cache = False
    if hasattr(inner, 'config'):
        inner.config.use_cache = False
        if hasattr(inner.config, 'layer_types') and inner.config.layer_types:
            inner.config.layer_types = [inner.config.layer_types[idx] for idx in order]
            inner.config.num_hidden_layers = len(order)

    # Identify duplicated layers
    sorted_blocks = sorted(blocks)
    dup_layers = []
    for (i, j) in sorted_blocks:
        for l in range(i, j):
            dup_layers.append(l)

    # Register sublayer hooks
    hooks = []
    for layer_idx in dup_layers:
        attn_alpha, ffn_alpha = sublayer_alphas.get(layer_idx, (1.0, 1.0))
        module = original_layers[layer_idx]

        # Attention hook: scale attn output on second pass
        attn_ctr = [0]
        aa = [attn_alpha]
        def make_attn_hook(ctr, aa):
            def hook(module, input, output):
                ctr[0] += 1
                if ctr[0] % 2 == 0:
                    if isinstance(output, tuple):
                        return (aa[0] * output[0],) + output[1:]
                    return aa[0] * output
                return output
            return hook
        h = module.self_attn.register_forward_hook(make_attn_hook(attn_ctr, aa))
        hooks.append(h)

        # FFN hook: scale FFN output on second pass
        ffn_ctr = [0]
        fa = [ffn_alpha]
        def make_ffn_hook(ctr, fa):
            def hook(module, input, output):
                ctr[0] += 1
                if ctr[0] % 2 == 0:
                    if isinstance(output, tuple):
                        return (fa[0] * output[0],) + output[1:]
                    return fa[0] * output
                return output
            return hook
        h = module.mlp.register_forward_hook(make_ffn_hook(ffn_ctr, fa))
        hooks.append(h)

    print(f'  Applied: {N} -> {len(order)} layers, {len(hooks)} sublayer hooks', flush=True)
    return hooks, original_layers, N

# Configs to test: (name, {layer_idx: (attn_alpha, ffn_alpha)})
configs = [
    ('attn_only', {0: (1.0, 0.0), 1: (1.0, 0.0), 12: (1.0, 0.0), 47: (1.0, 0.0)}),
    ('attn_heavy_ffn02', {0: (1.0, 0.2), 1: (1.0, 0.2), 12: (1.0, 0.2), 47: (1.0, 0.2)}),
    ('bayesian_sublayer', {
        0: (1.7213, 1.3462), 1: (1.4581, 1.9862),
        12: (1.7850, 0.8191), 47: (0.5063, 0.4485),
    }),
]

for config_name, sublayer_alphas in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'Skipping {config_name} — exists', flush=True)
        continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)
    for l, (a, f) in sorted(sublayer_alphas.items()):
        print(f'  L{l}: attn={a:.2f} ffn={f:.2f}', flush=True)
    print(f'{\"=\" * 60}', flush=True)

    model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
    hooks, orig_layers, orig_N = apply_dup_with_sublayer(model, BLOCKS, sublayer_alphas)

    # Patch forward + generate for no cache
    orig_gen = model.generate
    def patched_gen(*a, **kw):
        kw['use_cache'] = False
        return orig_gen(*a, **kw)
    model.generate = patched_gen

    orig_fwd = model.forward
    @functools.wraps(orig_fwd)
    def patched_fwd(*a, **kw):
        kw['use_cache'] = False
        return orig_fwd(*a, **kw)
    model.forward = patched_fwd

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    t0 = time.time()
    try:
        results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)
        elapsed = time.time() - t0

        scores = {}
        for task, data in results['results'].items():
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    scores[f'{task}/{metric}'] = value

        print(f'\\n=== RESULTS ({config_name}) === [{elapsed:.0f}s]', flush=True)
        for k, v in sorted(scores.items()):
            if 'stderr' not in k and k.count('/') == 1:
                print(f'  {k}: {v:.4f}', flush=True)

        with open(outpath, 'w') as f:
            json.dump({
                'config': config_name, 'blocks': [list(b) for b in BLOCKS],
                'sublayer_alphas': {str(k): list(v) for k, v in sublayer_alphas.items()},
                'scores': scores, 'elapsed_s': elapsed,
            }, f, indent=2)
        print(f'SAVED {outpath}', flush=True)

    except Exception as e:
        print(f'FAILED: {e}', flush=True)
        import traceback
        traceback.print_exc()

    for h in hooks:
        h.remove()
    del model, tokenizer, lm
    gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Finished: $(date) ==="
