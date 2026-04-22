"""
PSRT Reasoning Trainer: Train on synthetic reasoning tasks

Tasks where more thinking provably helps:
  - Arithmetic chains (variable length)
  - Grid mazes (variable size)
  - Logic deductions (variable chain length)
  - Counting/tracking (variable events)
  - Pattern completion (variable complexity)

Key metric: does the model learn higher E[K] for harder problems?

Mixed with 30% fineweb-edu for language fluency.

Usage:
    python train_reasoning.py --size 172m --total_steps 15000
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import PSRT, PSRTConfig, psrt_172m, create_model
from reasoning_data import generate_reasoning_example, stream_reasoning_data


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def mixed_stream(tokenizer, seq_len, batch_size):
    """70% reasoning tasks + 30% general text."""
    from datasets import load_dataset

    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    token_buffer = []

    while True:
        if random.random() < 0.70:
            # Reasoning task
            diff = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.20, 0.30, 0.25, 0.15])[0]
            ex = generate_reasoning_example(diff)
            text = ex['text']
        else:
            # General text
            try:
                ex = next(general)
                text = ex.get('text', '')
            except StopIteration:
                general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                            split='train', streaming=True))
                continue

        if not text or len(text) < 20:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False,
                                  truncation=True, max_length=seq_len * 2)
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


def evaluate_reasoning(model, tokenizer, device, n_per_difficulty=20):
    """Evaluate on reasoning tasks at each difficulty, measuring accuracy and E[K]."""
    model.eval()
    results = {}

    for diff in range(1, 6):
        correct = 0
        total = 0
        total_k = 0

        for _ in range(n_per_difficulty):
            ex = generate_reasoning_example(diff)
            prompt_tokens = tokenizer.encode(ex['prompt'], add_special_tokens=False,
                                             truncation=True, max_length=256)
            answer_tokens = tokenizer.encode(ex['answer'], add_special_tokens=False)

            input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

            with torch.no_grad():
                # Generate with adaptive K
                generated = []
                for _ in range(len(answer_tokens) + 10):
                    logits, _, aux = model(input_ids, fixed_K=None, train_halting=False)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated.append(next_token.item())
                    total_k += aux['n_recursions']
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    total += 1

                gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                if ex['answer'].strip() in gen_text[:len(ex['answer']) + 5]:
                    correct += 1

        avg_k = total_k / max(total, 1)
        acc = correct / n_per_difficulty * 100
        results[f'diff_{diff}'] = {'accuracy': acc, 'avg_K': avg_k, 'n': n_per_difficulty}
        print(f'  Difficulty {diff}: acc={acc:.0f}% E[K]={avg_k:.2f}', flush=True)

    model.train()
    return results


def evaluate_perplexity_by_k(model, tokenizer, device, n_batches=20, seq_len=512, batch_size=4):
    """PPL at K=1 and K=2 on reasoning data."""
    model.eval()

    results = {}
    for K in [1, 2]:
        total_loss = 0
        total_tokens = 0
        gen = stream_reasoning_data(tokenizer, seq_len, batch_size)
        with torch.no_grad():
            for i, (input_ids, labels) in enumerate(gen):
                if i >= n_batches:
                    break
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                _, loss, _ = model(input_ids, labels=labels, fixed_K=K)
                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()
        ppl = math.exp(total_loss / max(total_tokens, 1))
        results[f'K={K}'] = ppl

    model.train()
    return results


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

    phase1_end = int(total_steps * 0.35)
    phase2_end = int(total_steps * 0.65)

    save_dir = args.save_dir or f'psrt/checkpoints/reasoning'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== PSRT Reasoning Trainer ===', flush=True)
    print(f'  Size: {args.size} ({model.count_params()/1e6:.0f}M)', flush=True)
    print(f'  Steps: {total_steps} (P1:{phase1_end} P2:{phase2_end} P3:{total_steps})', flush=True)
    print(f'  Data: 70% reasoning (arith/maze/logic/counting/pattern) + 30% fineweb', flush=True)
    print(f'  Halt penalty: {args.halt_penalty}', flush=True)
    print(f'  Phase 2 K weights: [0.20, 0.45, 0.35] (heavy on K=2,3)', flush=True)

    gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
    model.train()

    step = 0
    running_loss = 0
    running_ek = 0
    t0 = time.time()
    best_ppl = float('inf')

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
            fixed_K = random.choices([1, 2, 3], weights=[0.20, 0.45, 0.35])[0]
            train_halting = False
            phase = 2
        else:
            fixed_K = None
            train_halting = True
            phase = 3

        logits, loss, aux = model(input_ids, labels=labels,
                                   fixed_K=fixed_K, train_halting=train_halting)

        if train_halting:
            loss = loss + args.halt_penalty * aux['expected_steps']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        es = aux['expected_steps']
        running_ek += es.item() if isinstance(es, torch.Tensor) else es

        step += 1

        if step % 100 == 0:
            avg_loss = running_loss / 100
            avg_ek = running_ek / 100
            elapsed = time.time() - t0
            K_str = f'K={fixed_K}' if fixed_K else 'adaptive'
            print(f'  step {step:6d} P{phase} | loss={avg_loss:.4f} | '
                  f'E[K]={avg_ek:.2f} | {K_str} | alpha={aux["alpha"]:.3f} | '
                  f'{elapsed:.0f}s', flush=True)
            running_loss = 0
            running_ek = 0

        if step % 2000 == 0:
            # PPL comparison
            ppl = evaluate_perplexity_by_k(model, tokenizer, device,
                                            n_batches=15, seq_len=args.seq_len,
                                            batch_size=args.batch_size)
            delta = ppl['K=2'] - ppl['K=1']
            print(f'  --- EVAL step {step}: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} '
                  f'(delta={delta:+.2f}) ---', flush=True)

            if ppl['K=1'] < best_ppl:
                best_ppl = ppl['K=1']
                torch.save({
                    'step': step, 'model_state': model.state_dict(),
                    'config': cfg.__dict__, 'ppl': ppl,
                }, f'{save_dir}/best.pt')
                print(f'  --- SAVED best ---', flush=True)

        if step % 5000 == 0:
            # Full reasoning eval
            print(f'  --- Reasoning eval step {step} ---', flush=True)
            reason_results = evaluate_reasoning(model, tokenizer, device, n_per_difficulty=15)
            with open(f'{save_dir}/reasoning_eval_{step}.json', 'w') as f:
                json.dump(reason_results, f, indent=2)

        if step == phase1_end:
            print(f'\n=== Phase 2: K={{1,2,3}} curriculum (heavy recursion) ===', flush=True)
        elif step == phase2_end:
            print(f'\n=== Phase 3: Adaptive Halting (penalty={args.halt_penalty}) ===', flush=True)

    # Final eval
    torch.save({'step': step, 'model_state': model.state_dict(),
                'config': cfg.__dict__}, f'{save_dir}/final.pt')

    print(f'\n=== Final Reasoning Evaluation ===', flush=True)
    final_reason = evaluate_reasoning(model, tokenizer, device, n_per_difficulty=30)

    print(f'\n=== Final PPL ===', flush=True)
    final_ppl = evaluate_perplexity_by_k(model, tokenizer, device, n_batches=30,
                                          seq_len=args.seq_len, batch_size=args.batch_size)
    delta = final_ppl['K=2'] - final_ppl['K=1']
    print(f'  PPL K=1={final_ppl["K=1"]:.2f} K=2={final_ppl["K=2"]:.2f} (delta={delta:+.2f})')

    print(f'\n=== Training Complete ===', flush=True)
    print(f'  Best PPL: {best_ppl:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='172m', choices=['172m', '1b'])
    parser.add_argument('--total_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--halt_penalty', type=float, default=0.0005)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
