#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_gemma3_%j.log
#SBATCH --job-name=deeppass_g3eval

# lm-eval on Gemma3-27B: baseline, best single (20,21), best pair (4,5)+(20,21)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval on Gemma3-27B ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, torch, torch.nn as nn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

TASKS = 'leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
LIMIT = 0.15

configs = [
    ('baseline', []),
    ('single_20_21', [(20, 21)]),
    ('pair_4_5_20_21', [(4, 5), (20, 21)]),
]

for config_name, blocks in configs:
    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name} blocks={blocks}', flush=True)
    print(f'{\"=\" * 60}', flush=True)

    model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
    inner = model.model
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    original_layers = list(inner.layers)
    N = len(original_layers)

    if blocks:
        order = build_order(blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        if hasattr(model.config, 'text_config'):
            model.config.text_config.num_hidden_layers = len(order)
        else:
            model.config.num_hidden_layers = len(order)
        print(f'Applied duplication: {N} -> {len(order)} layers', flush=True)

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    # Disable cache
    orig_gen = model.generate
    def patched_gen(*a, **kw):
        kw['use_cache'] = False
        return orig_gen(*a, **kw)
    model.generate = patched_gen

    task_list = TASKS.split(',')
    print(f'Tasks: {task_list}', flush=True)
    print(f'Limit: {LIMIT}', flush=True)

    results = evaluator.simple_evaluate(model=lm, tasks=task_list, limit=LIMIT)

    scores = {}
    for task, data in results['results'].items():
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                scores[f'{task}/{metric}'] = value

    print(f'\\n=== RESULTS ({config_name}) ===', flush=True)
    for k, v in sorted(scores.items()):
        if 'stderr' not in k:
            print(f'  {k}: {v:.4f}', flush=True)

    os.makedirs('results/data/gemma3_27b/lm_eval', exist_ok=True)
    with open(f'results/data/gemma3_27b/lm_eval/{config_name}.json', 'w') as f:
        json.dump({'config': config_name, 'blocks': [list(b) for b in blocks],
                   'scores': scores, 'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results['results'].items()}}, f, indent=2)
    print(f'Saved to results/data/gemma3_27b/lm_eval/{config_name}.json', flush=True)

    # Cleanup for next config
    del model, tokenizer, lm
    import gc; gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Done at $(date) ==="
