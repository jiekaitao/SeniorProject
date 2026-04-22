"""
SIRT-170M Training Script

Stage 1: Fixed 1-recursion warmup (standard LM training)
Stage 2: Recurrence curriculum (1-3 recursions, random)
Stage 3: Adaptive halting (ACT loss enabled)

Usage:
    python train.py --stage 1 --tokens 2B
    python train.py --stage 2 --tokens 2B --resume checkpoints/stage1_final.pt
    python train.py --stage 3 --tokens 2B --resume checkpoints/stage2_final.pt
"""

import os
import sys
import json
import math
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from model import SIRTConfig, SIRTLM, create_model

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class ShardedTokenDataset(IterableDataset):
    """Reads pre-tokenized .bin shards (uint16) for fast training."""
    def __init__(self, data_dir, seq_len=4096, shuffle_shards=True):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.shards = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.bin')
        ])
        assert len(self.shards) > 0, f"No .bin shards in {data_dir}"

    def __iter__(self):
        import numpy as np
        shards = list(self.shards)
        if self.shuffle_shards:
            random.shuffle(shards)
        for shard_path in shards:
            data = np.fromfile(shard_path, dtype=np.uint32)
            data = torch.from_numpy(data.astype(np.int32)).long()
            # Chunk into sequences
            n_chunks = len(data) // (self.seq_len + 1)
            data = data[:n_chunks * (self.seq_len + 1)]
            data = data.reshape(n_chunks, self.seq_len + 1)
            # Shuffle chunks within shard
            indices = torch.randperm(n_chunks)
            for idx in indices:
                chunk = data[idx]
                yield {
                    "input_ids": chunk[:-1],
                    "labels": chunk[1:],
                }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset from HuggingFace (fallback if no shards)."""
    def __init__(self, tokenizer, seq_len=4096, split="train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split

    def __iter__(self):
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                          streaming=True, trust_remote_code=True)
        buffer = []
        for example in ds:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


class SyntheticDataset(IterableDataset):
    """Random data for smoke testing."""
    def __init__(self, vocab_size=49152, seq_len=512, n_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
            yield {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine decay with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_step(model, batch, optimizer, scheduler, cfg, stage, step, grad_accum_steps=1):
    """Single training step with gradient accumulation."""
    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Stage-dependent recursion config
    if stage == 1:
        fixed_k = 1
        train_halting = False
    elif stage == 2:
        fixed_k = random.choice([1, 2, 3])
        train_halting = False
    else:  # stage 3
        fixed_k = None
        train_halting = True

    # Forward
    logits, lm_loss, aux = model(
        input_ids, labels=labels,
        fixed_recursions=fixed_k,
        train_halting=train_halting,
    )

    # Total loss
    loss = lm_loss.float()
    if train_halting:
        halt_loss = 0.01 * aux['expected_steps'].float()
        beta_loss = 0.001 * aux['beta_mean'].float()
        loss = loss + halt_loss + beta_loss

    # Scale for gradient accumulation
    loss = loss / grad_accum_steps
    loss.backward()

    if (step + 1) % grad_accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()

    return {
        'loss': lm_loss.item(),
        'total_loss': (loss * grad_accum_steps).item(),
        'beta_mean': aux['beta_mean'].item(),
        'expected_steps': aux['expected_steps'].item() if isinstance(aux['expected_steps'], torch.Tensor) else aux['expected_steps'],
        'n_recursions': aux['n_recursions'],
    }


def train(args):
    # Config
    cfg = SIRTConfig(
        context_len=args.seq_len,
    )

    # Model
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model = create_model(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = create_model(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).to(torch.bfloat16)
    print(f"Device: {device}")

    # Dataset
    if args.synthetic:
        print("Using synthetic data (smoke test)")
        dataset = SyntheticDataset(cfg.vocab_size, args.seq_len, args.max_steps * args.batch_size * 2)
    elif os.path.exists(args.data_dir) and any(f.endswith('.bin') for f in os.listdir(args.data_dir)):
        print(f"Using sharded data from {args.data_dir}")
        dataset = ShardedTokenDataset(args.data_dir, args.seq_len)
    else:
        print("Using streaming from HuggingFace (slow, use prepare_data.sh first)")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf"
        )
        dataset = StreamingTextDataset(tokenizer, args.seq_len)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2 if not args.synthetic else 0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True if device.type == 'cuda' else False,
    )

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    step = 0
    t0 = time.time()
    running_loss = 0.0
    tokens_processed = 0

    print(f"\nTraining Stage {args.stage}")
    print(f"  Steps: {args.max_steps}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"  LR: {args.lr}, Grad Accum: {args.grad_accum}")
    print(f"  Effective batch tokens: {args.batch_size * args.seq_len * args.grad_accum:,}")

    for batch in dataloader:
        if step >= args.max_steps:
            break

        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        metrics = train_step(model, batch, optimizer, None, cfg, args.stage, step, args.grad_accum)
        running_loss += metrics['loss']
        tokens_processed += args.batch_size * args.seq_len

        step += 1

        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed
            print(f"  step {step:6d} | loss {avg_loss:.4f} | β {metrics['beta_mean']:.4f} | "
                  f"K={metrics['n_recursions']} | E[K]={metrics['expected_steps']:.2f} | "
                  f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s | {elapsed:.0f}s",
                  flush=True)
            running_loss = 0.0

        if step % args.save_interval == 0:
            path = f"{args.save_dir}/stage{args.stage}_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg.__dict__,
                'stage': args.stage,
            }, path)
            print(f"  [SAVED] {path}")

    # Final save
    path = f"{args.save_dir}/stage{args.stage}_final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg.__dict__,
        'stage': args.stage,
    }, path)
    print(f"\n[FINAL SAVE] {path}")
    print(f"Total: {step} steps, {tokens_processed:,} tokens, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--synthetic", action="store_true", help="Use random data for smoke test")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="sirt/checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="sirt/data")
    args = parser.parse_args()
    train(args)
