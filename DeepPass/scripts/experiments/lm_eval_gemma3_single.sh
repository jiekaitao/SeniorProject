#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_g3_single_%j.log
#SBATCH --job-name=g3_sing

# lm-eval on individual blocks to understand WHERE duplication helps/hurts
# Tests: single (12,13), single (47,48), and the best pair (0,2)+(12,13)
# Uses the sliding window fix from lm_eval_gemma3_fixed.sh

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval: Single/Pair Block Analysis ==="
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

def apply_dup(model, blocks):
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)
    if not blocks:
        return N
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
    return N

configs = [
    ('single_12_13', [(12, 13)]),
    ('single_47_48', [(47, 48)]),
    ('pair_0_2_12_13', [(0, 2), (12, 13)]),
]

for config_name, blocks in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'Skipping {config_name} — exists', flush=True)
        continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name} blocks={blocks}', flush=True)

    model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
    apply_dup(model, blocks)

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
    results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)
    elapsed = time.time() - t0

    scores = {}
    for task, data in results['results'].items():
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                scores[f'{task}/{metric}'] = value

    print(f'=== RESULTS ({config_name}) === [{elapsed:.0f}s]', flush=True)
    for k, v in sorted(scores.items()):
        if 'stderr' not in k and k.count('/') == 1:
            print(f'  {k}: {v:.4f}', flush=True)

    with open(outpath, 'w') as f:
        json.dump({'config': config_name, 'blocks': [list(b) for b in blocks],
                   'scores': scores, 'elapsed_s': elapsed}, f, indent=2)
    print(f'SAVED {outpath}', flush=True)

    del model, tokenizer, lm
    gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Finished: $(date) ==="
