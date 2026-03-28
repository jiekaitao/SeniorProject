#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama3_70b_lmeval_%j.log
#SBATCH --job-name=ll3_eval

# lm-eval on LLaMA 3 70B: baseline, best single (10,11), best pair (10,11)+(61,62)
# KEY QUESTION: Does duplication improve lm-eval on a full-attention model?
# (On Gemma3 it regressed. On LLaMA3, FFN helps, so maybe benchmarks improve too.)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 70B lm-eval ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-Instruct-hf'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
LIMIT = 0.15
SAVE_DIR = 'results/data/llama3_70b/lm_eval'
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

configs = [
    ('baseline', []),
    ('single_10_11', [(10, 11)]),
    ('pair_10_11_61_62', [(10, 11), (61, 62)]),
]

for config_name, blocks in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'\\nSkipping {config_name} — exists', flush=True)
        continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name} blocks={blocks}', flush=True)
    print(f'{\"=\" * 60}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map='auto', dtype=torch.bfloat16,
    )
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)

    if blocks:
        order = build_order(blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)
        print(f'Applied duplication: {N} -> {len(order)} layers', flush=True)

    # Disable cache for all calls
    model.config.use_cache = False

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

    task_list = TASKS.split(',')
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
            if 'stderr' not in k and k.count('/') == 1:
                print(f'  {k}: {v:.4f}', flush=True)

        with open(outpath, 'w') as f:
            json.dump({
                'config': config_name,
                'blocks': [list(b) for b in blocks],
                'tasks': task_list, 'limit': LIMIT,
                'scores': scores, 'elapsed_s': elapsed,
            }, f, indent=2)
        print(f'SAVED to {outpath}', flush=True)

    except Exception as e:
        print(f'FAILED: {e}', flush=True)
        import traceback
        traceback.print_exc()

    del model, tokenizer, lm
    gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Finished: $(date) ==="
