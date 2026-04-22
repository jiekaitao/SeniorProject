"""
PSRT Training: 3-Phase Curriculum

Phase 1: Fixed K=1 (standard LM training, warmup all weights)
Phase 2: Curriculum K in {1,2,3} (introduce recursion)
Phase 3: Adaptive halting with ACT loss

Uses GPT-2 tokenizer, streams from fineweb-edu.

Usage:
    python train.py --size 172m --total_steps 20000
    python train.py --size 1b --total_steps 100000 --batch_size 4
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import PSRT, PSRTConfig, psrt_172m, psrt_1b, create_model


def get_tokenizer():
    """Load GPT-2 tokenizer (50257 vocab, widely available)."""
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def stream_tokens(tokenizer, seq_len, batch_size):
    """Yield batches of (input_ids, labels) from fineweb-edu stream."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                      split='train', streaming=True)
    token_buffer = []

    for example in ds:
        text = example.get('text', '')
        if not text or len(text) < 50:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                  max_length=seq_len * 4)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= (seq_len + 1) * batch_size:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            t = torch.tensor(batch, dtype=torch.long)
            yield t[:, :-1], t[:, 1:]  # input_ids, labels


def evaluate_perplexity(model, tokenizer, device, n_batches=20, seq_len=512, batch_size=4):
    """Quick perplexity evaluation on held-out data."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    gen = stream_tokens(tokenizer, seq_len, batch_size)
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(gen):
            if i >= n_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            _, loss, _ = model(input_ids, labels=labels, fixed_K=1)
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model, cfg = create_model(args.size)
    model = model.to(device)

    # Tokenizer
    tokenizer = get_tokenizer()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.95))
    total_steps = args.total_steps
    warmup_steps = int(total_steps * 0.05)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Phase boundaries
    phase1_end = int(total_steps * 0.50)
    phase2_end = int(total_steps * 0.80)
    # phase3: phase2_end to total_steps

    save_dir = args.save_dir or f'psrt/checkpoints/{args.size}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== PSRT Training ===', flush=True)
    print(f'  Size: {args.size} ({model.count_params()/1e6:.0f}M params)', flush=True)
    print(f'  Steps: {total_steps} (P1: {phase1_end}, P2: {phase2_end}, P3: {total_steps})', flush=True)
    print(f'  Batch: {args.batch_size} x {args.seq_len} = {args.batch_size * args.seq_len} tokens/step', flush=True)
    print(f'  LR: {args.lr}', flush=True)
    print(f'  Device: {device}', flush=True)
    print(f'  Save: {save_dir}', flush=True)

    # Training loop
    gen = stream_tokens(tokenizer, args.seq_len, args.batch_size)
    model.train()

    step = 0
    running_loss = 0
    running_aux = {'alpha': 0, 'expected_steps': 0}
    t0 = time.time()
    best_ppl = float('inf')
    log_interval = 100
    eval_interval = 2000

    for input_ids, labels in gen:
        if step >= total_steps:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Determine phase and K
        if step < phase1_end:
            # Phase 1: fixed K=1, no halting
            fixed_K = 1
            train_halting = False
            phase = 1
        elif step < phase2_end:
            # Phase 2: curriculum K in {1,2,3}
            fixed_K = random.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0]
            train_halting = False
            phase = 2
        else:
            # Phase 3: adaptive halting
            fixed_K = None
            train_halting = True
            phase = 3

        logits, loss, aux = model(input_ids, labels=labels,
                                   fixed_K=fixed_K, train_halting=train_halting)

        # Add halt penalty in phase 3
        if train_halting:
            loss = loss + 0.01 * aux['expected_steps']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        running_aux['alpha'] += aux['alpha']
        es = aux['expected_steps']
        running_aux['expected_steps'] += es.item() if isinstance(es, torch.Tensor) else es

        step += 1

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_alpha = running_aux['alpha'] / log_interval
            avg_steps = running_aux['expected_steps'] / log_interval
            elapsed = time.time() - t0
            tokens_per_sec = step * args.batch_size * args.seq_len / elapsed
            lr = scheduler.get_last_lr()[0]
            K_str = f'K={fixed_K}' if fixed_K else 'adaptive'

            print(f'  step {step:6d} P{phase} | loss={avg_loss:.4f} | '
                  f'alpha={avg_alpha:.3f} | E[K]={avg_steps:.2f} | {K_str} | '
                  f'lr={lr:.2e} | {tokens_per_sec/1e3:.0f}K tok/s | {elapsed:.0f}s',
                  flush=True)
            running_loss = 0
            running_aux = {'alpha': 0, 'expected_steps': 0}

        # Evaluate and save
        if step % eval_interval == 0:
            ppl_k1 = evaluate_perplexity(model, tokenizer, device, n_batches=20,
                                          seq_len=args.seq_len, batch_size=args.batch_size)
            # Also evaluate K=2
            model.eval()
            total_loss_k2 = 0
            total_tok_k2 = 0
            gen_eval = stream_tokens(tokenizer, args.seq_len, args.batch_size)
            with torch.no_grad():
                for i, (ids, labs) in enumerate(gen_eval):
                    if i >= 20:
                        break
                    ids, labs = ids.to(device), labs.to(device)
                    _, loss_k2, _ = model(ids, labels=labs, fixed_K=2)
                    total_loss_k2 += loss_k2.item() * labs.numel()
                    total_tok_k2 += labs.numel()
            ppl_k2 = math.exp(total_loss_k2 / max(total_tok_k2, 1))
            model.train()

            print(f'  --- EVAL step {step}: PPL K=1={ppl_k1:.2f} K=2={ppl_k2:.2f} '
                  f'(delta={ppl_k2-ppl_k1:+.2f}) ---', flush=True)

            # Save if best
            if ppl_k1 < best_ppl:
                best_ppl = ppl_k1
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'config': cfg.__dict__,
                    'ppl_k1': ppl_k1,
                    'ppl_k2': ppl_k2,
                }, f'{save_dir}/best.pt')
                print(f'  --- SAVED best (PPL={ppl_k1:.2f}) ---', flush=True)

        # Phase transitions
        if step == phase1_end:
            print(f'\n=== Phase 2: Curriculum K={{1,2,3}} ===', flush=True)
        elif step == phase2_end:
            print(f'\n=== Phase 3: Adaptive Halting ===', flush=True)

    # Final save
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': cfg.__dict__,
    }, f'{save_dir}/final.pt')

    # Final evaluation
    ppl_final = evaluate_perplexity(model, tokenizer, device, n_batches=50,
                                     seq_len=args.seq_len, batch_size=args.batch_size)

    print(f'\n=== Training Complete ===', flush=True)
    print(f'  Steps: {step}', flush=True)
    print(f'  Final PPL (K=1): {ppl_final:.2f}', flush=True)
    print(f'  Best PPL: {best_ppl:.2f}', flush=True)
    print(f'  Saved to {save_dir}/', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='172m', choices=['172m', '1b'])
    parser.add_argument('--total_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
