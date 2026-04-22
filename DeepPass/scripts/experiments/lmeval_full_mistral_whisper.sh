#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_full_mis_whisper_%j.log
#SBATCH --job-name=mis_full

# Mistral 7B: FULL lm-eval (no subsample) on whisper FFN β=0.2
# Our best raw-dup result: MATH +2.5%, MuSR +2.6%, BBH +0.3%, MMLU -0.1%

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Mistral 7B FULL lm-eval (Whisper FFN) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'sirt')
from recursion_finetune import LayerIdxWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
CACHE_DIR = '/blue/cis4914/jietao/hf_cache'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
SAVE_DIR = 'results/data/mistral_7b/lm_eval_full'
os.makedirs(SAVE_DIR, exist_ok=True)

def build_order(blocks, N):
    s = sorted(blocks); order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j)); order.extend(range(i, j)); prev = j
    order.extend(range(prev, N)); return order

def apply_dup_with_whisper(model, blocks, beta=0.2):
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

    # Whisper FFN hooks on second pass
    hooks = []
    sorted_blocks = sorted(blocks)
    dup_layers = []
    for (i, j) in sorted_blocks:
        for l in range(i, j): dup_layers.append(l)
    for layer_idx in dup_layers:
        module = orig[layer_idx]
        counter = [0]
        def make_whisper(ctr, b=beta):
            def hook(mod, inp, out):
                ctr[0] += 1
                if ctr[0] % 2 == 0:
                    if isinstance(out, tuple):
                        return (b * out[0],) + out[1:]
                    return b * out
                return out
            return hook
        h = module.mlp.register_forward_hook(make_whisper(counter))
        hooks.append(h)

    return orig, N, hooks

configs = [
    ('baseline', [], None),
    ('whisper_ffn02_28_29', [(28, 29)], 0.2),
]

for config_name, blocks, beta in configs:
    outpath = f'{SAVE_DIR}/{config_name}.json'
    if os.path.exists(outpath):
        print(f'Skipping {config_name}', flush=True); continue

    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, device_map='auto', dtype=torch.bfloat16, trust_remote_code=True)

    hooks = []
    if blocks:
        orig, N, hooks = apply_dup_with_whisper(model, blocks, beta)

    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    print(f'Running FULL lm-eval (limit=None)...', flush=True)
    t0 = time.time()
    results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=None)
    elapsed = time.time() - t0

    scores = {f'{t}/{m}': v for t, d in results['results'].items() for m, v in d.items() if isinstance(v, (int, float))}
    print(f'\\n=== RESULTS ({config_name}) === [{elapsed:.0f}s]', flush=True)
    for k, v in sorted(scores.items()):
        if 'stderr' not in k and k.count('/') == 1:
            print(f'  {k}: {v:.4f}', flush=True)

    with open(outpath, 'w') as f:
        json.dump({'config': config_name, 'blocks': [list(b) for b in blocks],
                   'beta': beta, 'limit': None, 'scores': scores, 'elapsed_s': elapsed}, f, indent=2)
    print(f'SAVED {outpath}', flush=True)

    for h in hooks: h.remove()
    del model, tokenizer, lm; gc.collect(); torch.cuda.empty_cache()

print('\\n=== All done ===', flush=True)
"

echo "=== Finished: $(date) ==="
