#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_data_%j.log
#SBATCH --job-name=sirt_dat

# Download and tokenize fineweb-edu for SIRT training
# Uses LLaMA 3 tokenizer (49152 vocab, pre-downloaded)
# Produces sharded .bin files for fast streaming

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT Data Preparation ==="
echo "Started: $(date)"

$PYTHON -c "
import os, time, struct, numpy as np, torch

DATA_DIR = 'sirt/data'
os.makedirs(DATA_DIR, exist_ok=True)

# Use LLaMA 3 tokenizer (pre-downloaded on HiPerGator)
from transformers import AutoTokenizer
print('Loading tokenizer...', flush=True)
tok = AutoTokenizer.from_pretrained(
    '/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf'
)
print(f'Vocab size: {tok.vocab_size}', flush=True)
# Note: LLaMA 3 has vocab_size=128256, not 49152
# We'll update the model config to match

# Stream and tokenize fineweb-edu
from datasets import load_dataset
print('Loading fineweb-edu (streaming)...', flush=True)

# Target: ~6B tokens in shards of ~100M tokens each
TARGET_TOKENS = 6_000_000_000
SHARD_SIZE = 100_000_000  # 100M tokens per shard
SEQ_LEN = 4097  # 4096 + 1 for labels

shard_idx = 0
token_buffer = []
total_tokens = 0
t0 = time.time()

# Check if we already have shards
existing = [f for f in os.listdir(DATA_DIR) if f.endswith('.bin')]
if existing:
    # Count existing tokens
    for f in sorted(existing):
        path = os.path.join(DATA_DIR, f)
        size = os.path.getsize(path)
        total_tokens += size // 4  # uint32 = 4 bytes
    shard_idx = len(existing)
    print(f'Found {len(existing)} existing shards, {total_tokens:,} tokens. Resuming from shard {shard_idx}.', flush=True)
    if total_tokens >= TARGET_TOKENS:
        print(f'Already have {total_tokens:,} >= {TARGET_TOKENS:,} target. Done!', flush=True)
        import sys; sys.exit(0)

ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                   split='train', streaming=True)

def save_shard(tokens, shard_idx):
    arr = np.array(tokens, dtype=np.uint32)
    path = os.path.join(DATA_DIR, f'shard_{shard_idx:04d}.bin')
    arr.tofile(path)
    print(f'  Saved shard {shard_idx}: {len(tokens):,} tokens ({os.path.getsize(path)/1e6:.1f}MB)', flush=True)

for example in ds:
    if total_tokens >= TARGET_TOKENS:
        break

    text = example.get('text', '')
    if not text or len(text) < 50:
        continue

    tokens = tok.encode(text, add_special_tokens=False)
    tokens.append(tok.eos_token_id)
    token_buffer.extend(tokens)

    while len(token_buffer) >= SHARD_SIZE:
        shard_tokens = token_buffer[:SHARD_SIZE]
        token_buffer = token_buffer[SHARD_SIZE:]
        save_shard(shard_tokens, shard_idx)
        total_tokens += SHARD_SIZE
        shard_idx += 1

        elapsed = time.time() - t0
        rate = total_tokens / elapsed
        eta = (TARGET_TOKENS - total_tokens) / rate if rate > 0 else 0
        print(f'  Total: {total_tokens/1e9:.2f}B / {TARGET_TOKENS/1e9:.0f}B tokens '
              f'({total_tokens/TARGET_TOKENS*100:.1f}%) | {rate/1e6:.1f}M tok/s | ETA: {eta/60:.0f}min',
              flush=True)

# Save remaining
if token_buffer:
    save_shard(token_buffer, shard_idx)
    total_tokens += len(token_buffer)

elapsed = time.time() - t0
print(f'\\nDone! {total_tokens:,} tokens in {shard_idx+1} shards ({elapsed/60:.1f} min)', flush=True)
print(f'Tokenizer vocab: {tok.vocab_size}', flush=True)

# Save metadata
import json
with open(os.path.join(DATA_DIR, 'metadata.json'), 'w') as f:
    json.dump({
        'total_tokens': total_tokens,
        'n_shards': shard_idx + 1,
        'shard_size': SHARD_SIZE,
        'vocab_size': tok.vocab_size,
        'tokenizer': 'meta-llama/Llama-3-8B-Instruct',
        'source': 'HuggingFaceFW/fineweb-edu (sample-10BT)',
    }, f, indent=2)
print('Saved metadata.json', flush=True)
"

echo "=== Finished: $(date) ==="
