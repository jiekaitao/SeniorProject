#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_g3_cached_%j.log
#SBATCH --job-name=g3_kv

# Gemma3 lm-eval WITH KV cache fix
# Re-runs configs that timed out + tests if cached generation changes results
# Tests: hces_best, single_12_13_ffn02, triple_cached (compare vs uncached)

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3-27B lm-eval WITH KV Cache ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/gemma-3-27b-it'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
LIMIT = 0.15
SAVE_DIR = 'results/data/gemma3_27b/lm_eval'
os.makedirs(SAVE_DIR, exist_ok=True)

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

def apply_dup_cached(model, blocks, sublayer_alphas=None):
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)
    order = build_order(blocks, N)

    seen = set()
    is_dup = []
    for idx in order:
        is_dup.append(idx in seen)
        seen.add(idx)

    new_layers = []
    for phys_idx, (orig_idx, dup) in enumerate(zip(order, is_dup)):
        layer = original_layers[orig_idx]
        if dup:
            new_layers.append(LayerIdxWrapper(layer, phys_idx))
        else:
            layer.layer_idx = phys_idx
            if hasattr(layer, 'self_attn'): layer.self_attn.layer_idx = phys_idx
            new_layers.append(layer)

    inner.layers = nn.ModuleList(new_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = len(new_layers)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[orig_idx] for orig_idx in order]
        if hasattr(cfg, 'use_cache'): cfg.use_cache = True

    # Sublayer hooks
    hooks = []
    if sublayer_alphas:
        sorted_blocks = sorted(blocks)
        dup_layers = []
        for (i, j) in sorted_blocks:
            for l in range(i, j): dup_layers.append(l)
        for layer_idx in dup_layers:
            if layer_idx not in sublayer_alphas: continue
            attn_a, ffn_b = sublayer_alphas[layer_idx]
            module = original_layers[layer_idx]
            if abs(attn_a - 1.0) > 1e-6:
                actr = [0]; aa = attn_a
                def make_ah(c, a):
                    def hook(mod, inp, out):
                        c[0] += 1
                        if c[0] % 2 == 0:
                            return (a * out[0],) + out[1:] if isinstance(out, tuple) else a * out
                        return out
                    return hook
                hooks.append(module.self_attn.register_forward_hook(make_ah(actr, aa)))
            if abs(ffn_b - 1.0) > 1e-6:
                fctr = [0]; fb = ffn_b
                def make_fh(c, b):
                    def hook(mod, inp, out):
                        c[0] += 1
                        if c[0] % 2 == 0:
                            return (b * out[0],) + out[1:] if isinstance(out, tuple) else b * out
                        return out
                    return hook
                hooks.append(module.mlp.register_forward_hook(make_fh(fctr, fb)))

    print(f'Applied: {N} -> {len(new_layers)} layers, cache ON, {len(hooks)} sublayer hooks', flush=True)
    return hooks, original_layers, N

BLOCKS = [(0, 2), (12, 13), (47, 48)]

configs = [
    # Triple with cache (compare vs uncached triple_alpha1 result)
    ('triple_cached', BLOCKS, None),

    # HCES best mask (timed out before)
    ('hces_best_cached', BLOCKS, {
        0: (0.5, 0.2), 1: (0.5, 1.0), 12: (1.0, 0.2), 47: (0.0, 0.0),
    }),

    # Single (12,13) with whisper FFN (timed out before)
    ('single_12_13_ffn02_cached', [(12, 13)], {
        12: (1.0, 0.2),
    }),
]

for config_name, blocks, sublayer_alphas in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'Skipping {config_name}', flush=True)
        continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)

    model, tokenizer = load_original_model(MODEL_PATH)
    hooks, orig_layers, orig_N = apply_dup_cached(model, blocks, sublayer_alphas)

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    t0 = time.time()
    try:
        results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=LIMIT)
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
                       'sublayer_alphas': {str(k): list(v) for k, v in sublayer_alphas.items()} if sublayer_alphas else {},
                       'scores': scores, 'elapsed_s': elapsed, 'kv_cache': True}, f, indent=2)
        print(f'SAVED {outpath}', flush=True)
    except Exception as e:
        print(f'FAILED: {e}', flush=True)
        import traceback; traceback.print_exc()

    for h in hooks: h.remove()
    del model, tokenizer, lm; gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Finished: $(date) ==="
