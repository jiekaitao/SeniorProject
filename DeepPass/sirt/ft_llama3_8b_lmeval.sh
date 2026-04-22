#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ll3_ft_lmeval_%j.log
#SBATCH --job-name=ll3ft_ev

# LLaMA 3 8B: lm-eval on recursion-finetuned K=2
# v3 showed K=2=+3.22 on dual probe. Does it translate?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 8B FT K=2 lm-eval ===" && echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc, torch, torch.nn as nn, functools, random, numpy as np
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')
from recursion_finetune import LayerIdxWrapper, build_recursive_model, restore_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
TASKS = 'leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro'
SAVE_DIR = 'sirt/recursion_ft/llama3_8b_v3'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda')
print('Loading LLaMA 3 8B...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)

# Fine-tune core [10,13) with same v3 params
print('Fine-tuning...', flush=True)
for param in model.parameters(): param.requires_grad = False
for i in range(10, 13):
    for param in original_layers[i].parameters(): param.requires_grad = True

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-7, weight_decay=0.01)
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
model.train()
step = 0; token_buffer = []
for example in ds:
    if step >= 300: break
    text = example.get('text', '')
    if not text or len(text) < 100: continue
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2048)
    if tokenizer.eos_token_id: tokens.append(tokenizer.eos_token_id)
    token_buffer.extend(tokens)
    while len(token_buffer) >= 1025 and step < 300:
        chunk = torch.tensor([token_buffer[:1025]], dtype=torch.long).to(device)
        token_buffer = token_buffer[1024:]
        K = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        orig, orig_N, new_N = build_recursive_model(model, 10, 13, K)
        try:
            outputs = model(chunk[:, :-1], use_cache=False)
            loss = torch.nn.functional.cross_entropy(outputs.logits.reshape(-1, outputs.logits.size(-1)), chunk[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        except: optimizer.zero_grad(set_to_none=True)
        restore_model(model, orig, orig_N)
        step += 1
        if step % 100 == 0: print(f'  step {step} | loss {loss.item():.4f} | K={K}', flush=True)

for param in model.parameters(): param.requires_grad = False
model.eval()
print(f'Fine-tune done: {step} steps', flush=True)

# Apply K=2 and run lm-eval
print('Applying K=2 for lm-eval...', flush=True)
orig, orig_N, new_N = build_recursive_model(model, 10, 13, 2)
model.config.use_cache = True

from lm_eval.models.huggingface import HFLM; from lm_eval import evaluator
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

t0 = time.time()
results = evaluator.simple_evaluate(model=lm, tasks=TASKS.split(','), limit=0.15)
elapsed = time.time() - t0

scores = {f'{t}/{m}': v for t, d in results['results'].items() for m, v in d.items() if isinstance(v, (int, float))}
print(f'\\n=== RESULTS (llama3_ft_k2) === [{elapsed:.0f}s]', flush=True)
for k, v in sorted(scores.items()):
    if 'stderr' not in k and k.count('/') == 1: print(f'  {k}: {v:.4f}', flush=True)

with open(f'{SAVE_DIR}/lm_eval_ft_k2.json', 'w') as f:
    json.dump({'config': 'llama3_ft_k2', 'scores': scores, 'elapsed_s': elapsed}, f, indent=2)
print('SAVED', flush=True)
"

echo "=== Finished: $(date) ==="
