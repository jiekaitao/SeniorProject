#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_mistral_7b_%j.log
#SBATCH --job-name=mis_eval

# Mistral 7B lm-eval — sliding window model, compare with Gemma3
# Best block from sweep: (28,29) = +2.96

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Mistral 7B lm-eval ===" && echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
CACHE_DIR = '/blue/cis4914/jietao/hf_cache'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
LIMIT = 0.15
SAVE_DIR = 'results/data/mistral_7b/lm_eval'
os.makedirs(SAVE_DIR, exist_ok=True)

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

def apply_dup(model, blocks):
    inner = model.model
    orig = list(inner.layers); N = len(orig)
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
    # Mistral has sliding window — update if layer_types exists
    if hasattr(model.config, 'sliding_window'):
        pass  # sliding_window is a scalar, not per-layer
    print(f'Applied: {N} -> {len(new_layers)} layers, cache ON', flush=True)
    return orig, N

configs = [
    ('baseline', []),
    ('single_28_29', [(28, 29)]),
]

for name, blocks in configs:
    outpath = f'{SAVE_DIR}/{name}.json'
    if os.path.exists(outpath):
        print(f'Skipping {name}', flush=True); continue
    print(f'\\n=== {name} ===', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True)
    if blocks: apply_dup(model, blocks)
    from lm_eval.models.huggingface import HFLM; from lm_eval import evaluator
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    t0 = time.time()
    try:
        results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=LIMIT)
        elapsed = time.time() - t0
        scores = {f'{t}/{m}': v for t, d in results['results'].items() for m, v in d.items() if isinstance(v, (int, float))}
        print(f'RESULTS ({name}) [{elapsed:.0f}s]', flush=True)
        for k, v in sorted(scores.items()):
            if 'stderr' not in k and k.count('/') == 1: print(f'  {k}: {v:.4f}', flush=True)
        with open(outpath, 'w') as f:
            json.dump({'config': name, 'blocks': [list(b) for b in blocks], 'scores': scores, 'elapsed_s': elapsed}, f, indent=2)
        print(f'SAVED {outpath}', flush=True)
    except Exception as e:
        print(f'FAILED: {e}', flush=True); import traceback; traceback.print_exc()
    del model, tokenizer, lm; gc.collect(); torch.cuda.empty_cache()
print('=== All done ===', flush=True)
"
echo "=== Finished: $(date) ==="
