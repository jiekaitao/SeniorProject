#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_attn_only_probe_%j.log
#SBATCH --job-name=deeppass_aodp

# Comprehensive attention-only vs FFN-only analysis on dual probe
# Test the hypothesis: attention = reasoning, FFN = retrieval
# Also test attention-only on multiple blocks and pairs

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Attention-Only Comprehensive Analysis ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('ATTENTION-ONLY vs FFN-ONLY COMPREHENSIVE')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def generate_sublayer_mode(prompt, blocks, mode='attn_only', max_new_tokens=64):
    \"\"\"
    mode: 'attn_only' = skip FFN on 2nd pass, 'ffn_only' = skip attn on 2nd pass, 'full' = normal
    \"\"\"
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)

    # Find second-pass steps
    second_pass_steps = set()
    for block in sorted_blocks:
        bi, bj = block
        block_layers = list(range(bi, bj))
        count = {}
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:
                    second_pass_steps.add(step)

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                if step_idx in second_pass_steps and mode != 'full':
                    # Manual sublayer forward
                    residual = h
                    normed = layer.input_layernorm(h)
                    if mode == 'ffn_only':
                        # Skip attention, keep FFN
                        h = residual  # no attention contribution
                    else:
                        attn_out = layer.self_attn(normed, position_embeddings=pos_embeds, attention_mask=None, use_cache=False)
                        attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
                        h = residual + attn_out

                    residual = h
                    normed = layer.post_attention_layernorm(h)
                    if mode == 'attn_only':
                        # Skip FFN
                        h = residual  # no FFN contribution
                    else:
                        ffn_out = layer.mlp(normed)
                        h = residual + ffn_out
                else:
                    out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                    h = out[0] if isinstance(out, tuple) else out

            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def evaluate(blocks, mode, name):
    gen = lambda p: generate_sublayer_mode(p, blocks, mode, 64)
    gen_long = lambda p: generate_sublayer_mode(p, blocks, mode, 128)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  {name:55s}: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    return {'name': name, 'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}

from layer_duplicator import generate_no_cache
gen_base = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=64)
gen_base_long = lambda p: generate_no_cache(model, tokenizer, p, max_new_tokens=128)
math_base = run_math_probe(gen_base, verbose=False)
eq_base = run_eq_bench_probe(gen_base_long, verbose=False)
baseline = math_base['score'] * 50 + eq_base['score'] * 0.5
print(f'Baseline: combined={baseline:.2f}', flush=True)

all_results = []

# Test on multiple blocks
print(f'\\n--- Single blocks: attn-only vs ffn-only vs full ---', flush=True)
BLOCKS = [(45, 52), (50, 60), (0, 7), (20, 27)]
for block in BLOCKS:
    for mode in ['full', 'attn_only', 'ffn_only']:
        r = evaluate([block], mode, f'({block[0]},{block[1]}) {mode}')
        all_results.append(r)

# Pair: attention-only on both blocks
print(f'\\n--- Pair (0,7)+(45,52): attn-only vs full ---', flush=True)
pair = [(0, 7), (45, 52)]
for mode in ['full', 'attn_only', 'ffn_only']:
    r = evaluate(pair, mode, f'pair {mode}')
    all_results.append(r)

# Summary
print(f'\\n{\"=\" * 70}')
print('SUMMARY: Does attention-only preserve knowledge while helping reasoning?')
print(f'{\"=\" * 70}')
print(f'Baseline: {baseline:.2f}')
sorted_r = sorted(all_results, key=lambda x: x['combined'], reverse=True)
for r in sorted_r:
    delta = r['combined'] - baseline
    print(f'  {r[\"name\"]:55s}: math={r[\"math\"]:.4f} eq={r[\"eq\"]:.1f} combined={r[\"combined\"]:.2f} delta={delta:+.2f}', flush=True)

os.makedirs('results/data/72b/attn_only', exist_ok=True)
with open('results/data/72b/attn_only/comprehensive.json', 'w') as f:
    json.dump({'date': datetime.now().isoformat(), 'baseline': baseline, 'results': all_results}, f, indent=2)
print(f'Saved', flush=True)
"

echo "=== Done at $(date) ==="
