#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_ft_ll3_scaled_%j.log
#SBATCH --job-name=ll3_scl

# LLaMA 3 8B SCALED recursion fine-tune
# Push further: wider core, more steps, combine with whisper FFN
# Tests multiple configurations to find the best

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== LLaMA 3 8B Scaled Recursion FT ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, torch, torch.nn as nn, random, numpy as np
sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')
from recursion_finetune import *
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
SAVE_DIR = 'sirt/recursion_ft/llama3_8b_scaled'
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda')
eq_all = _load_questions()

# ======================================================================
# Config 1: Wider core [8,16] (8 layers), 500 steps, lr=5e-7
# ======================================================================
print('\\n=== Config 1: Wide core [8,16], 500 steps ===', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)
inner = model.model; original_layers = list(inner.layers); N = len(original_layers)

# Baseline
baseline = evaluate_model(model, tokenizer, device, 1, 'baseline')

# Pre-ft K=2
orig, oN, nN = build_recursive_model(model, 8, 16, 2)
pre_k2 = evaluate_model(model, tokenizer, device, 2, 'pre-ft K=2 [8,16]')
restore_model(model, orig, oN)

# Fine-tune
finetune_recursion(model, tokenizer, device, 'sirt/data', 8, 16,
                   max_steps=500, lr=5e-7, batch_size=1, seq_len=1024)

# Post-ft K=1
post_k1 = evaluate_model(model, tokenizer, device, 1, 'post-ft K=1')

# Post-ft K=2
orig, oN, nN = build_recursive_model(model, 8, 16, 2)
post_k2 = evaluate_model(model, tokenizer, device, 2, 'post-ft K=2 [8,16]')
restore_model(model, orig, oN)

result1 = {
    'config': 'wide_core_8_16', 'core': [8, 16], 'steps': 500,
    'baseline': baseline, 'pre_k2': pre_k2, 'post_k1': post_k1, 'post_k2': post_k2,
}
print(f'  Wide: baseline={baseline[\"combined\"]:.2f} post_k2={post_k2[\"combined\"]:.2f} delta={post_k2[\"combined\"]-baseline[\"combined\"]:+.2f}', flush=True)

del model, tokenizer; import gc; gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# Config 2: Original core [10,13] but 1000 steps (longer training)
# ======================================================================
print('\\n=== Config 2: Core [10,13], 1000 steps ===', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)

finetune_recursion(model, tokenizer, device, 'sirt/data', 10, 13,
                   max_steps=1000, lr=5e-7, batch_size=1, seq_len=1024)

post_k1_2 = evaluate_model(model, tokenizer, device, 1, 'post-ft K=1')
orig, oN, nN = build_recursive_model(model, 10, 13, 2)
post_k2_2 = evaluate_model(model, tokenizer, device, 2, 'post-ft K=2 [10,13]')
restore_model(model, orig, oN)
orig, oN, nN = build_recursive_model(model, 10, 13, 3)
post_k3_2 = evaluate_model(model, tokenizer, device, 3, 'post-ft K=3 [10,13]')
restore_model(model, orig, oN)

result2 = {
    'config': 'long_train_10_13', 'core': [10, 13], 'steps': 1000,
    'baseline': baseline, 'post_k1': post_k1_2, 'post_k2': post_k2_2, 'post_k3': post_k3_2,
}
print(f'  Long: baseline={baseline[\"combined\"]:.2f} post_k2={post_k2_2[\"combined\"]:.2f} delta={post_k2_2[\"combined\"]-baseline[\"combined\"]:+.2f}', flush=True)

del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# Config 3: Two recursive zones [4,7] + [10,13] (early + mid)
# ======================================================================
print('\\n=== Config 3: Two zones [4,7]+[10,13] ===', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', dtype=torch.bfloat16)
inner = model.model; original_layers = list(inner.layers); N = len(original_layers)

# Fine-tune both zones
for param in model.parameters(): param.requires_grad = False
for i in list(range(4, 7)) + list(range(10, 13)):
    for param in original_layers[i].parameters(): param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Trainable: {trainable:,}', flush=True)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-7, weight_decay=0.01)
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
model.train()
step = 0; token_buffer = []
for example in ds:
    if step >= 500: break
    text = example.get('text', '')
    if not text or len(text) < 100: continue
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2048)
    if tokenizer.eos_token_id: tokens.append(tokenizer.eos_token_id)
    token_buffer.extend(tokens)
    while len(token_buffer) >= 1025 and step < 500:
        chunk = torch.tensor([token_buffer[:1025]], dtype=torch.long).to(device)
        token_buffer = token_buffer[1024:]
        K = random.choices([1, 2], weights=[0.5, 0.5])[0]
        # Build order with two zones
        order = list(range(N))
        if K >= 2:
            # Duplicate both zones
            order = list(range(7)) + list(range(4, 7)) + list(range(7, 13)) + list(range(10, 13)) + list(range(13, N))
        seen = set(); is_dup = []
        for idx in order: is_dup.append(idx in seen); seen.add(idx)
        new_layers = []
        for pi, (oi, d) in enumerate(zip(order, is_dup)):
            l = original_layers[oi]
            if d: new_layers.append(LayerIdxWrapper(l, pi))
            else:
                l.layer_idx = pi
                if hasattr(l, 'self_attn'): l.self_attn.layer_idx = pi
                new_layers.append(l)
        inner.layers = nn.ModuleList(new_layers)
        model.config.num_hidden_layers = len(new_layers)
        try:
            outputs = model(chunk[:, :-1], use_cache=False)
            loss = torch.nn.functional.cross_entropy(outputs.logits.reshape(-1, outputs.logits.size(-1)), chunk[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        except: optimizer.zero_grad(set_to_none=True)
        # Restore
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N
        for i, l in enumerate(original_layers):
            l.layer_idx = i
            if hasattr(l, 'self_attn'): l.self_attn.layer_idx = i
        step += 1
        if step % 100 == 0: print(f'  step {step} | loss {loss.item():.4f} | K={K}', flush=True)

for param in model.parameters(): param.requires_grad = False
model.eval()

# Evaluate two-zone K=2
order = list(range(7)) + list(range(4, 7)) + list(range(7, 13)) + list(range(10, 13)) + list(range(13, N))
seen = set(); is_dup = []
for idx in order: is_dup.append(idx in seen); seen.add(idx)
new_layers = []
for pi, (oi, d) in enumerate(zip(order, is_dup)):
    l = original_layers[oi]
    if d: new_layers.append(LayerIdxWrapper(l, pi))
    else:
        l.layer_idx = pi
        if hasattr(l, 'self_attn'): l.self_attn.layer_idx = pi
        new_layers.append(l)
inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
model.config.use_cache = True

post_dual = evaluate_model(model, tokenizer, device, 2, 'post-ft dual-zone K=2')
inner.layers = nn.ModuleList(original_layers)
model.config.num_hidden_layers = N
for i, l in enumerate(original_layers):
    l.layer_idx = i
    if hasattr(l, 'self_attn'): l.self_attn.layer_idx = i

post_k1_3 = evaluate_model(model, tokenizer, device, 1, 'post-ft K=1')

result3 = {
    'config': 'dual_zone_4_7_10_13', 'steps': 500,
    'baseline': baseline, 'post_k1': post_k1_3, 'post_dual_k2': post_dual,
}
print(f'  Dual: baseline={baseline[\"combined\"]:.2f} dual_k2={post_dual[\"combined\"]:.2f} delta={post_dual[\"combined\"]-baseline[\"combined\"]:+.2f}', flush=True)

# ======================================================================
# Summary
# ======================================================================
print(f'\\n{\"=\" * 50}', flush=True)
print('SCALED RECURSION RESULTS — LLaMA 3 8B', flush=True)
print(f'Baseline: {baseline[\"combined\"]:.2f}', flush=True)
print(f'Config 1 (wide [8,16]):      K=2={post_k2[\"combined\"]:.2f} ({post_k2[\"combined\"]-baseline[\"combined\"]:+.2f})', flush=True)
print(f'Config 2 (long 1000 steps):  K=2={post_k2_2[\"combined\"]:.2f} ({post_k2_2[\"combined\"]-baseline[\"combined\"]:+.2f}) K=3={post_k3_2[\"combined\"]:.2f}', flush=True)
print(f'Config 3 (dual zone):        K=2={post_dual[\"combined\"]:.2f} ({post_dual[\"combined\"]-baseline[\"combined\"]:+.2f})', flush=True)
print(f'Previous v3 (core [10,13]):  K=2=66.98 (+3.22)', flush=True)
print('COMPLETE', flush=True)

all_results = {'config1': result1, 'config2': result2, 'config3': result3}
with open(f'{SAVE_DIR}/results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved to {SAVE_DIR}/results.json', flush=True)
"

echo "=== Finished: $(date) ==="
