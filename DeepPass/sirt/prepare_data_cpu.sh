#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_data_cpu_%j.log
#SBATCH --job-name=sirt_cpu

# Data prep on CPU — no GPU needed

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT Data Prep (CPU) ==="
echo "Started: $(date)"

# Run the same Python code from prepare_data.sh but inline
$PYTHON -c "
import os, time, numpy as np

DATA_DIR = 'sirt/data'
os.makedirs(DATA_DIR, exist_ok=True)

from transformers import AutoTokenizer
print('Loading tokenizer...', flush=True)
tok = AutoTokenizer.from_pretrained(
    '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
)
print(f'Vocab size: {tok.vocab_size}', flush=True)

from datasets import load_dataset
print('Loading fineweb-edu (streaming)...', flush=True)

TARGET_TOKENS = 6_000_000_000
SHARD_SIZE = 100_000_000

shard_idx = 0
token_buffer = []
total_tokens = 0
t0 = time.time()

existing = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.bin')])
if existing:
    for f in existing:
        total_tokens += os.path.getsize(os.path.join(DATA_DIR, f)) // 4
    shard_idx = len(existing)
    print(f'Resuming: {len(existing)} shards, {total_tokens:,} tokens', flush=True)
    if total_tokens >= TARGET_TOKENS:
        print('Already done!', flush=True)
        import sys; sys.exit(0)

ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                   split='train', streaming=True)

def save_shard(tokens, idx):
    arr = np.array(tokens, dtype=np.uint32)
    path = os.path.join(DATA_DIR, f'shard_{idx:04d}.bin')
    arr.tofile(path)
    print(f'  Saved shard {idx}: {len(tokens):,} tokens', flush=True)

for example in ds:
    if total_tokens >= TARGET_TOKENS: break
    text = example.get('text', '')
    if not text or len(text) < 50: continue
    tokens = tok.encode(text, add_special_tokens=False)
    tokens.append(tok.eos_token_id)
    token_buffer.extend(tokens)
    while len(token_buffer) >= SHARD_SIZE:
        save_shard(token_buffer[:SHARD_SIZE], shard_idx)
        token_buffer = token_buffer[SHARD_SIZE:]
        total_tokens += SHARD_SIZE
        shard_idx += 1
        elapsed = time.time() - t0
        rate = total_tokens / elapsed
        eta = (TARGET_TOKENS - total_tokens) / rate if rate > 0 else 0
        print(f'  {total_tokens/1e9:.2f}B / {TARGET_TOKENS/1e9:.0f}B ({total_tokens/TARGET_TOKENS*100:.1f}%) | {rate/1e6:.1f}M tok/s | ETA: {eta/60:.0f}min', flush=True)

if token_buffer:
    save_shard(token_buffer, shard_idx)
    total_tokens += len(token_buffer)

import json
with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
    json.dump({'total_tokens': total_tokens, 'n_shards': shard_idx + 1,
               'vocab_size': tok.vocab_size, 'source': 'fineweb-edu/sample-10BT'}, f, indent=2)
print(f'Done! {total_tokens:,} tokens in {shard_idx+1} shards', flush=True)
"

echo "=== Finished: $(date) ==="
