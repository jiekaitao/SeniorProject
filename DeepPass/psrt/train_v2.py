"""
PSRT v2 Training: Mixed dataset with reasoning-heavy data

Key fix from v1: fineweb-edu alone is too easy — model learns E[K]=1.0
(never recurse). Mixed data forces the model to learn WHEN recursion helps.

Data mix:
  50% fineweb-edu (general fluency)
  25% OpenMathInstruct (math reasoning — needs multi-step thought)
  25% ARC/science QA (multi-step inference)

Also: lower halt penalty (0.001 instead of 0.01) to let the model
explore higher K before collapsing.

Usage:
    python train_v2.py --size 172m --total_steps 20000
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import PSRT, PSRTConfig, psrt_172m, psrt_1b, create_model


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def mixed_data_stream(tokenizer, seq_len, batch_size):
    """Stream from 3 sources with 50/25/25 mix."""
    from datasets import load_dataset

    # Load all three streams
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train',
                                streaming=True))
    science = iter(load_dataset('allenai/ai2_arc', 'ARC-Challenge',
                                split='train', streaming=True))

    token_buffer = []

    def get_text(source_name):
        """Get text from a source, handling different formats."""
        try:
            if source_name == 'general':
                ex = next(general)
                return ex.get('text', '')
            elif source_name == 'math':
                ex = next(math_ds)
                # OpenMathInstruct has 'problem' and 'generated_solution'
                problem = ex.get('problem', '')
                solution = ex.get('generated_solution', '')
                return f"Problem: {problem}\nSolution: {solution}"
            elif source_name == 'science':
                ex = next(science)
                q = ex.get('question', '')
                choices = ex.get('choices', {})
                labels = choices.get('label', [])
                texts = choices.get('text', [])
                answer_key = ex.get('answerKey', '')
                choice_str = ' '.join(f"({l}) {t}" for l, t in zip(labels, texts))
                answer_text = ''
                for l, t in zip(labels, texts):
                    if l == answer_key:
                        answer_text = t
                return f"Question: {q}\n{choice_str}\nAnswer: ({answer_key}) {answer_text}"
        except StopIteration:
            return ''
        except Exception:
            return ''

    while True:
        # Pick source: 50% general, 25% math, 25% science
        r = random.random()
        if r < 0.50:
            text = get_text('general')
        elif r < 0.75:
            text = get_text('math')
        else:
            text = get_text('science')

        if not text or len(text) < 50:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False,
                                  truncation=True, max_length=seq_len * 4)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= (seq_len + 1) * batch_size:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            t = torch.tensor(batch, dtype=torch.long)
            yield t[:, :-1], t[:, 1:]


def evaluate_perplexity(model, tokenizer, device, n_batches=20, seq_len=512, batch_size=4):
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Use general data for consistent eval
    from datasets import load_dataset
    ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                           split='train', streaming=True))
    token_buffer = []

    with torch.no_grad():
        for i in range(n_batches):
            # Fill buffer
            while len(token_buffer) < (seq_len + 1) * batch_size:
                ex = next(ds)
                text = ex.get('text', '')
                if not text or len(text) < 50:
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=False,
                                          truncation=True, max_length=seq_len * 4)
                tokens.append(tokenizer.eos_token_id)
                token_buffer.extend(tokens)

            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            t = torch.tensor(batch, dtype=torch.long).to(device)
            input_ids, labels = t[:, :-1], t[:, 1:]

            _, loss, _ = model(input_ids, labels=labels, fixed_K=1)
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, cfg = create_model(args.size)
    model = model.to(device)

    tokenizer = get_tokenizer()

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

    phase1_end = int(total_steps * 0.40)
    phase2_end = int(total_steps * 0.70)

    save_dir = args.save_dir or f'psrt/checkpoints/{args.size}_v2'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== PSRT v2 Training (Mixed Data) ===', flush=True)
    print(f'  Size: {args.size} ({model.count_params()/1e6:.0f}M params)', flush=True)
    print(f'  Steps: {total_steps} (P1: {phase1_end}, P2: {phase2_end}, P3: {total_steps})', flush=True)
    print(f'  Data: 50% fineweb-edu + 25% OpenMathInstruct + 25% ARC', flush=True)
    print(f'  Halt penalty: {args.halt_penalty} (v1 was 0.01)', flush=True)
    print(f'  Batch: {args.batch_size} x {args.seq_len}', flush=True)
    print(f'  Device: {device}', flush=True)

    gen = mixed_data_stream(tokenizer, args.seq_len, args.batch_size)
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

        if step < phase1_end:
            fixed_K = 1
            train_halting = False
            phase = 1
        elif step < phase2_end:
            # Phase 2: heavier on K=2,3 than v1 to force recursion learning
            fixed_K = random.choices([1, 2, 3], weights=[0.30, 0.45, 0.25])[0]
            train_halting = False
            phase = 2
        else:
            fixed_K = None
            train_halting = True
            phase = 3

        logits, loss, aux = model(input_ids, labels=labels,
                                   fixed_K=fixed_K, train_halting=train_halting)

        # Lower halt penalty than v1 (0.001 vs 0.01)
        if train_halting:
            loss = loss + args.halt_penalty * aux['expected_steps']

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

        if step % eval_interval == 0:
            ppl_k1 = evaluate_perplexity(model, tokenizer, device, n_batches=20,
                                          seq_len=args.seq_len, batch_size=args.batch_size)
            model.eval()
            # K=2 eval
            total_loss_k2 = 0
            total_tok_k2 = 0
            from datasets import load_dataset
            ds_eval = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                        split='train', streaming=True))
            eval_buf = []
            with torch.no_grad():
                for i in range(20):
                    while len(eval_buf) < (args.seq_len + 1) * args.batch_size:
                        ex = next(ds_eval)
                        text = ex.get('text', '')
                        if not text or len(text) < 50:
                            continue
                        toks = tokenizer.encode(text, add_special_tokens=False,
                                                truncation=True, max_length=args.seq_len * 4)
                        toks.append(tokenizer.eos_token_id)
                        eval_buf.extend(toks)
                    batch = []
                    for _ in range(args.batch_size):
                        chunk = eval_buf[:args.seq_len + 1]
                        eval_buf = eval_buf[args.seq_len:]
                        batch.append(chunk)
                    t = torch.tensor(batch, dtype=torch.long).to(device)
                    ids, labs = t[:, :-1], t[:, 1:]
                    _, loss_k2, _ = model(ids, labels=labs, fixed_K=2)
                    total_loss_k2 += loss_k2.item() * labs.numel()
                    total_tok_k2 += labs.numel()
            ppl_k2 = math.exp(total_loss_k2 / max(total_tok_k2, 1))
            model.train()

            print(f'  --- EVAL step {step}: PPL K=1={ppl_k1:.2f} K=2={ppl_k2:.2f} '
                  f'(delta={ppl_k2-ppl_k1:+.2f}) ---', flush=True)

            if ppl_k1 < best_ppl:
                best_ppl = ppl_k1
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'config': cfg.__dict__,
                    'ppl_k1': ppl_k1, 'ppl_k2': ppl_k2,
                }, f'{save_dir}/best.pt')
                print(f'  --- SAVED best (PPL={ppl_k1:.2f}) ---', flush=True)

        if step == phase1_end:
            print(f'\n=== Phase 2: Curriculum K={{1,2,3}} (heavier on K=2,3) ===', flush=True)
        elif step == phase2_end:
            print(f'\n=== Phase 3: Adaptive Halting (penalty={args.halt_penalty}) ===', flush=True)

    # Final save
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': cfg.__dict__,
    }, f'{save_dir}/final.pt')

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
    parser.add_argument('--halt_penalty', type=float, default=0.001)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
