#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_alpha_g3_%j.log
#SBATCH --job-name=g3_lmeval

# Full lm-eval on Gemma3-27B with alpha-tuned triple (0,2)+(12,13)+(47,48)
# Configs: baseline, all-1.0 triple, alpha-tuned triple
# Tasks: Open LLM Leaderboard v2 (BBH, MMLU-PRO, IFEval, MATH, MuSR) — 15% subsample

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== lm-eval: Gemma3-27B Alpha-Tuned Triple ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

BLOCKS = [(0, 2), (12, 13), (47, 48)]
BEST_ALPHAS = {0: 0.8797, 1: 0.8063, 12: 1.4507, 47: 0.9453}
TASKS = 'leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
LIMIT = 0.15
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

def set_num_layers(model, n):
    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = n
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = n

configs = [
    ('baseline', [], {}),
    ('triple_alpha1', BLOCKS, {0: 1.0, 1: 1.0, 12: 1.0, 47: 1.0}),
    ('triple_alpha_tuned', BLOCKS, BEST_ALPHAS),
]

for config_name, blocks, alphas in configs:
    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)
    if blocks:
        print(f'  Blocks: {blocks}', flush=True)
        print(f'  Alphas: {alphas}', flush=True)
    print(f'{\"=\" * 60}', flush=True)

    model, tokenizer = load_original_model('models/full/gemma-3-27b-it')
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)

    hooks = []
    if blocks:
        order = build_order(blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        set_num_layers(model, len(order))
        print(f'Applied duplication: {N} -> {len(order)} layers', flush=True)

        # Apply alpha hooks if not all 1.0
        sorted_blocks = sorted(blocks)
        dup_layers = []
        for (i, j) in sorted_blocks:
            for l in range(i, j):
                dup_layers.append(l)

        for layer_idx in dup_layers:
            alpha_val = alphas.get(layer_idx, 1.0)
            if abs(alpha_val - 1.0) < 1e-6:
                continue  # no hook needed for alpha=1.0

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
                        else:
                            blended = h_in + ac[0] * (output - h_in)
                            return blended
                    return output
                return hook

            h = module.register_forward_hook(make_hook(counter, ac))
            hooks.append(h)
        print(f'Registered {len(hooks)} alpha hooks', flush=True)

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

    t0 = time.time()
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

    with open(f'{SAVE_DIR}/{config_name}.json', 'w') as f:
        json.dump({
            'config': config_name, 'blocks': [list(b) for b in blocks],
            'alphas': {str(k): v for k, v in alphas.items()},
            'scores': scores, 'elapsed_s': elapsed,
            'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))} for k, v in results['results'].items()},
        }, f, indent=2)
    print(f'Saved to {SAVE_DIR}/{config_name}.json', flush=True)

    # Cleanup
    for h in hooks:
        h.remove()
    del model, tokenizer, lm
    gc.collect(); torch.cuda.empty_cache()

print('\\n=== All configs done ===', flush=True)
"

echo "=== Finished: $(date) ==="
