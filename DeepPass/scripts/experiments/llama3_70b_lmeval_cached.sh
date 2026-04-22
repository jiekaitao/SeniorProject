#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama3_70b_lmeval_cached_%j.log
#SBATCH --job-name=ll3_kv

# LLaMA 3 70B lm-eval WITH KV cache fix (LayerIdxWrapper)
# Previously failed due to cache collision making it 30s/it instead of 3s/it
# Now should run at normal speed with cached generation

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 70B lm-eval WITH KV Cache Fix ==="
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

# ======================================================================
# LayerIdxWrapper (the KV cache fix)
# ======================================================================
class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_layer_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_layer_idx
        self.original_layer_idx = layer.layer_idx
        self.original_attn_idx = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None

    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'):
            self.layer.self_attn.layer_idx = self.new_layer_idx
        try:
            return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.original_layer_idx
            if self.original_attn_idx is not None:
                self.layer.self_attn.layer_idx = self.original_attn_idx

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

def apply_duplication_with_cache(model, blocks):
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)

    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))

    seen = set()
    is_duplicate = []
    for idx in order:
        is_duplicate.append(idx in seen)
        seen.add(idx)

    new_layers = []
    for physical_idx, (orig_idx, is_dup) in enumerate(zip(order, is_duplicate)):
        layer = original_layers[orig_idx]
        if is_dup:
            new_layers.append(LayerIdxWrapper(layer, physical_idx))
        else:
            layer.layer_idx = physical_idx
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = physical_idx
            new_layers.append(layer)

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    # Enable cache
    model.config.use_cache = True

    print(f'Applied: {N} -> {len(new_layers)} layers, cache ENABLED', flush=True)
    return original_layers, N

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

# ======================================================================
# Configs: baseline (already saved), single (10,11), pair (10,11)+(61,62)
# ======================================================================
configs = [
    ('single_10_11_cached', [(10, 11)]),
    ('pair_10_11_61_62_cached', [(10, 11), (61, 62)]),
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

    original_layers, orig_N = apply_duplication_with_cache(model, blocks)

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
                'kv_cache': True,
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
