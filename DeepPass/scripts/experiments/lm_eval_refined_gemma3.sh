#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_refined_g3_%j.log
#SBATCH --job-name=g3_refin

# lm-eval on refined configs informed by neuron analysis:
# 1. pair_only: (0,2)+(12,13) — drop L47 entirely (the harmful layer)
# 2. hces_best: HCES-optimized sublayer mask (L47 off, L12 full attn, L1 full ffn)
# 3. pair_whisper: (0,2)+(12,13) with whisper FFN β=0.2
# All use the sliding window fix

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval: Refined Configs (drop L47) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

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

def apply_dup_sublayer(model, blocks, sublayer_alphas=None):
    \"\"\"Apply duplication with optional per-sublayer hooks.
    sublayer_alphas: dict of layer_idx -> (attn_alpha, ffn_alpha)
    \"\"\"
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)
    order = build_order(blocks, N)

    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
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

    hooks = []
    if sublayer_alphas:
        sorted_blocks = sorted(blocks)
        dup_layers = []
        for (i, j) in sorted_blocks:
            for l in range(i, j):
                dup_layers.append(l)

        for layer_idx in dup_layers:
            if layer_idx not in sublayer_alphas:
                continue
            attn_a, ffn_b = sublayer_alphas[layer_idx]
            module = original_layers[layer_idx]

            # Attention hook
            if abs(attn_a - 1.0) > 1e-6:
                actr = [0]
                aa = attn_a
                def make_ah(c, a):
                    def hook(mod, inp, out):
                        c[0] += 1
                        if c[0] % 2 == 0:
                            if isinstance(out, tuple):
                                return (a * out[0],) + out[1:]
                            return a * out
                        return out
                    return hook
                hooks.append(module.self_attn.register_forward_hook(make_ah(actr, aa)))

            # FFN hook
            if abs(ffn_b - 1.0) > 1e-6:
                fctr = [0]
                fb = ffn_b
                def make_fh(c, b):
                    def hook(mod, inp, out):
                        c[0] += 1
                        if c[0] % 2 == 0:
                            if isinstance(out, tuple):
                                return (b * out[0],) + out[1:]
                            return b * out
                        return out
                    return hook
                hooks.append(module.mlp.register_forward_hook(make_fh(fctr, fb)))

    print(f'  Applied: {N} -> {len(order)} layers, {len(hooks)} sublayer hooks', flush=True)
    return hooks, original_layers, N

# Configs to test
configs = [
    # 1. Just the pair — no L47 at all
    ('pair_0_2_12_13', [(0, 2), (12, 13)], None),

    # 2. Pair with whisper FFN
    ('pair_whisper_ffn02', [(0, 2), (12, 13)], {
        0: (1.0, 0.2), 1: (1.0, 0.2), 12: (1.0, 0.2),
    }),

    # 3. HCES best: full triple but L47 completely off
    ('hces_best', [(0, 2), (12, 13), (47, 48)], {
        0: (0.5, 0.2), 1: (0.5, 1.0), 12: (1.0, 0.2), 47: (0.0, 0.0),
    }),

    # 4. Single (12,13) with whisper FFN
    ('single_12_13_ffn02', [(12, 13)], {
        12: (1.0, 0.2),
    }),
]

for config_name, blocks, sublayer_alphas in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'\\nSkipping {config_name} — exists', flush=True)
        continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)
    print(f'  Blocks: {blocks}', flush=True)
    if sublayer_alphas:
        for l, (a, f) in sorted(sublayer_alphas.items()):
            print(f'  L{l}: attn={a} ffn={f}', flush=True)
    print(f'{\"=\" * 60}', flush=True)

    model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
    hooks, orig_layers, orig_N = apply_dup_sublayer(model, blocks, sublayer_alphas)

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
                'config': config_name, 'blocks': [list(b) for b in blocks],
                'sublayer_alphas': {str(k): list(v) for k, v in sublayer_alphas.items()} if sublayer_alphas else {},
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
