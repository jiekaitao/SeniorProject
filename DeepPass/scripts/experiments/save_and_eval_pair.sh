#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_save_eval_pair_%j.log
#SBATCH --job-name=deeppass_sepair

# Save pair (0,7)+(45,52) with deep-copied layers, eval with KV cache

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Save + Eval Pair (0,7)+(45,52) ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, copy, gc, torch, torch.nn as nn
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
SAVE_DIR = 'models/saved/pair_0_7_45_52_deepcopy'
BLOCKS = [(0, 7), (45, 52)]

if os.path.exists(os.path.join(SAVE_DIR, 'config.json')):
    print(f'Model already saved, skipping.', flush=True)
else:
    print('Loading base model on CPU...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map='cpu', dtype=torch.bfloat16, trust_remote_code=True
    )
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)

    # Build new layer list with deep copies for duplicated blocks
    sorted_blocks = sorted(BLOCKS)
    new_layers = []
    dup_positions = set()
    for idx in range(N):
        new_layers.append(original_layers[idx])
        for (bi, bj) in sorted_blocks:
            if idx == bj - 1:
                for dup_idx in range(bi, bj):
                    print(f'  Deep copying layer {dup_idx}...', flush=True)
                    new_layers.append(copy.deepcopy(original_layers[dup_idx]))

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    print(f'New model: {len(new_layers)} layers', flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR, max_shard_size='10GB')
    tokenizer.save_pretrained(SAVE_DIR)
    print('Saved!', flush=True)
    del model; gc.collect()

# Run lm-eval
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

print(f'\\nLoading saved pair model for lm-eval...', flush=True)
lm = HFLM(
    pretrained=SAVE_DIR,
    dtype='bfloat16',
    batch_size='auto',
    device_map_option='auto',
    trust_remote_code=True,
)

TASKS = ['leaderboard_ifeval', 'leaderboard_bbh', 'leaderboard_math_hard',
         'leaderboard_musr', 'leaderboard_mmlu_pro']

results = evaluator.simple_evaluate(model=lm, tasks=TASKS, limit=0.01)

print(f'\\n=== PAIR RESULTS (1% test, KV cache ON) ===', flush=True)
for task in TASKS:
    data = results['results'].get(task, {})
    for metric in sorted(data.keys()):
        if isinstance(data[metric], (int, float)) and 'stderr' not in metric:
            print(f'  {task}/{metric}: {data[metric]:.4f}', flush=True)

os.makedirs('results/data/72b/lm_eval/proper', exist_ok=True)
with open('results/data/72b/lm_eval/proper/pair_1pct.json', 'w') as f:
    json.dump({
        'method': 'save_pretrained deep copy, KV cache ON',
        'blocks': [list(b) for b in BLOCKS],
        'subsample': 0.01,
        'results': {k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float, str))}
                    for k, v in results['results'].items()},
    }, f, indent=2)
print('Saved!', flush=True)
"

echo "=== Done at $(date) ==="
