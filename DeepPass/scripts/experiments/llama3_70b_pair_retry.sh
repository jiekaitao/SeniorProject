#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ll3_pair_retry_%j.log
#SBATCH --job-name=ll3_pair

# LLaMA 3 70B pair lm-eval retry (timed out last time)
# Only runs the pair config (single already saved)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 70B Pair lm-eval Retry ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-70B-Instruct-hf'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
SAVE_DIR = 'results/data/llama3_70b/lm_eval'
os.makedirs(SAVE_DIR, exist_ok=True)

outpath = f'{SAVE_DIR}/pair_10_11_61_62_cached.json'
if os.path.exists(outpath):
    print('Already exists, skipping', flush=True)
    import sys; sys.exit(0)

class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None
    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'): self.layer.self_attn.layer_idx = self.new_layer_idx
        try: return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None: self.layer.self_attn.layer_idx = self.orig_attn
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self.layer, name)

def build_order(blocks, N):
    s = sorted(blocks); order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j)); order.extend(range(i, j)); prev = j
    order.extend(range(prev, N)); return order

blocks = [(10, 11), (61, 62)]
print(f'Config: pair {blocks}', flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)
inner = model.model; orig = list(inner.layers); N = len(orig)
order = build_order(blocks, N)

seen = set(); is_dup = []
for idx in order: is_dup.append(idx in seen); seen.add(idx)
new_layers = []
for pi, (oi, d) in enumerate(zip(order, is_dup)):
    l = orig[oi]
    if d: new_layers.append(LayerIdxWrapper(l, pi))
    else:
        l.layer_idx = pi
        if hasattr(l, 'self_attn'): l.self_attn.layer_idx = pi
        new_layers.append(l)
inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
model.config.use_cache = True

from lm_eval.models.huggingface import HFLM; from lm_eval import evaluator
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

t0 = time.time()
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)
elapsed = time.time() - t0

scores = {f'{t}/{m}': v for t, d in results['results'].items() for m, v in d.items() if isinstance(v, (int, float))}
print(f'\\n=== RESULTS === [{elapsed:.0f}s]', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k and k.count('/') == 1: print(f'  {k}: {v:.4f}', flush=True)

with open(outpath, 'w') as f:
    json.dump({'config': 'pair_10_11_61_62_cached', 'blocks': [list(b) for b in blocks],
               'scores': scores, 'elapsed_s': elapsed, 'kv_cache': True}, f, indent=2)
print(f'SAVED {outpath}', flush=True)
"

echo "=== Finished: $(date) ==="
