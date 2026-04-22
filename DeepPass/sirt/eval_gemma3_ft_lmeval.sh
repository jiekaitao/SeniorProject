#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_g3_ft_lmeval_%j.log
#SBATCH --job-name=g3ft_ev

# Gemma3-27B: lm-eval on recursion-finetuned model at K=2
# The model showed +4.11 on dual probe with K=2 post-ft
# Does it also improve on standardized benchmarks?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Gemma3 Recursion-FT lm-eval ===" && echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')
from recursion_finetune import LayerIdxWrapper, build_recursive_model, restore_model
from layer_duplicator import load_original_model
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = 'models/full/gemma-3-27b-it'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
SAVE_DIR = 'sirt/recursion_ft/gemma3_27b_v3'
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Load model
print('Loading Gemma3-27B...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
device = next(model.parameters()).device

inner = model.model
if hasattr(inner, 'language_model'):
    inner = inner.language_model
N = len(inner.layers)

# Step 2: Fine-tune core layers [11,14) with recursion curriculum (same as v3)
print('Fine-tuning core [11,14) with K=1-3 curriculum...', flush=True)
original_layers = list(inner.layers)

# Freeze all, unfreeze core
for param in model.parameters():
    param.requires_grad = False
for i in range(11, 14):
    for param in original_layers[i].parameters():
        param.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-6, weight_decay=0.01,
)

import random, numpy as np

ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
model.train()
step = 0
token_buffer = []

for example in ds:
    if step >= 300: break
    text = example.get('text', '')
    if not text or len(text) < 100: continue
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2048)
    if tokenizer.eos_token_id: tokens.append(tokenizer.eos_token_id)
    token_buffer.extend(tokens)

    while len(token_buffer) >= 513 and step < 300:
        chunk = torch.tensor([token_buffer[:513]], dtype=torch.long).to(device)
        token_buffer = token_buffer[512:]
        input_ids = chunk[:, :-1]
        labels = chunk[:, 1:]

        K = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        orig, orig_N, new_N = build_recursive_model(model, 11, 14, K)

        try:
            fwd_kwargs = {'input_ids': input_ids, 'use_cache': False,
                         'token_type_ids': torch.zeros_like(input_ids)}
            outputs = model(**fwd_kwargs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),
                labels.reshape(-1), ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            optimizer.zero_grad(set_to_none=True)

        restore_model(model, orig, orig_N)
        step += 1
        if step % 100 == 0:
            print(f'  step {step} | loss {loss.item():.4f} | K={K}', flush=True)

for param in model.parameters():
    param.requires_grad = False
model.eval()
print(f'Fine-tune done: {step} steps', flush=True)

# Step 3: Apply K=2 recursion and run lm-eval
print('\\nApplying K=2 recursion for lm-eval...', flush=True)
orig, orig_N, new_N = build_recursive_model(model, 11, 14, 2)

# Fix layer_types for Gemma3 sliding window
for cfg in [model.config, getattr(model.config, 'text_config', None)]:
    if cfg and hasattr(cfg, 'use_cache'):
        cfg.use_cache = True

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

# Patch generate AND forward to disable cache (fixes sliding window mask mismatch)
import functools
orig_gen = model.generate
def patched_gen(*a, **kw):
    kw.pop('token_type_ids', None)
    kw['use_cache'] = False
    return orig_gen(*a, **kw)
model.generate = patched_gen

orig_fwd = model.forward
@functools.wraps(orig_fwd)
def patched_fwd(*a, **kw):
    kw['use_cache'] = False
    return orig_fwd(*a, **kw)
model.forward = patched_fwd

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

t0 = time.time()
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)
elapsed = time.time() - t0

scores = {f'{t}/{m}': v for t, d in results['results'].items() for m, v in d.items() if isinstance(v, (int, float))}
print(f'\\n=== RESULTS (gemma3_ft_k2) === [{elapsed:.0f}s]', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k and k.count('/') == 1:
        print(f'  {k}: {v:.4f}', flush=True)

with open(f'{SAVE_DIR}/lm_eval_ft_k2.json', 'w') as f:
    json.dump({'config': 'gemma3_ft_k2', 'scores': scores, 'elapsed_s': elapsed}, f, indent=2)
print(f'SAVED', flush=True)
"

echo "=== Finished: $(date) ==="
