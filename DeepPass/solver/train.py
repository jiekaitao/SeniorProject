"""
Prompt Solver Training: 4 experiments in parallel.

Exp 1: Full solver (K_outer=3, K_inner=6, grad_last_only=True)
Exp 2: Solver without gradient truncation (grad_last_only=False)
Exp 3: Solver with fewer cycles (K_outer=1, K_inner=6)
Exp 4: Solver K-scaling test (train K=3×6, eval K=1,2,3,4)
"""

import os, sys, time, math, random, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from model import PromptSolverLLM


def get_math_stream(tokenizer, seq_len, batch_size):
    """80% math + 20% general — hard tasks that need reasoning."""
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    token_buffer = []

    def get_text():
        r = random.random()
        try:
            if r < 0.8:
                ex = next(math_ds)
                return f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
            else:
                return next(general).get('text', '')
        except (StopIteration, Exception):
            return ''

    while True:
        text = get_text()
        if not text or len(text) < 50:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                  max_length=seq_len * 2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)
        while len(token_buffer) >= (seq_len + 1) * batch_size:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            yield torch.tensor(batch, dtype=torch.long)


def evaluate_k_scaling(model, tokenizer, device, n=15, seq_len=512, bs=2):
    """Evaluate at different K_outer values."""
    model.eval()
    ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    buf = []
    while len(buf) < (seq_len + 1) * bs * n + 500:
        ex = next(ds)
        text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
        toks = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                max_length=seq_len * 2)
        toks.append(tokenizer.eos_token_id)
        buf.extend(toks)

    results = {}
    for K_outer in [0, 1, 2, 3, 4]:
        total_loss = total_tok = 0
        buf_copy = list(buf)
        with torch.no_grad():
            for _ in range(n):
                batch = []
                for _ in range(bs):
                    batch.append(buf_copy[:seq_len + 1])
                    buf_copy = buf_copy[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                prompt_len = t.shape[1] // 2

                if K_outer == 0:
                    # Baseline: no solver, just frozen LLM
                    with torch.no_grad():
                        out = model.base_model(t[:, :-1], labels=t[:, :-1])
                    total_loss += out.loss.item() * t[:, 1:].numel()
                else:
                    _, loss = model(t[:, :-1], labels=t[:, :-1],
                                   prompt_len=prompt_len,
                                   K_inner=6, K_outer=K_outer,
                                   grad_last_only=True)
                    if loss is not None:
                        total_loss += loss.item() * (t.shape[1] - prompt_len)
                total_tok += t[:, 1:].numel() if K_outer == 0 else (t.shape[1] - prompt_len)
        results[f'K={K_outer}'] = math.exp(total_loss / max(total_tok, 1))

    model.train()
    return results


def train(args):
    device = torch.device('cuda')
    model_path = 'models/full/Llama-3.1-8B'

    print(f'Loading {model_path}...', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to(device)
    print(f'  Loaded in {time.time()-t0:.0f}s', flush=True)

    # Build solver
    K_outer = args.K_outer
    grad_last = args.grad_last
    model = PromptSolverLLM(base_model, solver_d=1024, solver_heads=16,
                             solver_ffn=2816, solver_L_layers=2, n_memory=16)

    # Move solver to device
    model.solver = model.solver.to(device=device, dtype=torch.bfloat16)

    n_train = model.count_trainable()
    print(f'  Solver params: {n_train:,} ({n_train/model.count_total()*100:.2f}%)', flush=True)
    print(f'  K_outer={K_outer}, K_inner=6, grad_last={grad_last}', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = args.steps
    warmup = min(1000, total_steps // 5)
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    save_dir = f'solver/checkpoints/{args.exp}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== Prompt Solver: {args.exp} ===', flush=True)
    print(f'  Data: 80% math + 20% general', flush=True)
    print(f'  Steps: {total_steps}, LR: {args.lr}', flush=True)

    data_stream = get_math_stream(tokenizer, seq_len=512, batch_size=2)
    model.train()
    step = 0; running_loss = 0; t0 = time.time()

    for batch in data_stream:
        if step >= total_steps: break
        batch = batch.to(device)
        input_ids = batch[:, :-1]
        prompt_len = input_ids.shape[1] // 2

        _, loss = model(input_ids, labels=batch, prompt_len=prompt_len,
                       K_inner=6, K_outer=K_outer, grad_last_only=grad_last)

        if loss is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if loss is not None:
            running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg = running_loss / 100
            print(f'  step {step:5d} | loss={avg:.4f} | {time.time()-t0:.0f}s', flush=True)
            running_loss = 0

        if step % 1000 == 0:
            results = evaluate_k_scaling(model, tokenizer, device,
                                         n=10, seq_len=512, bs=2)
            parts = [f"{k}={v:.2f}" for k, v in sorted(results.items())]
            print(f'  --- EVAL step {step}: {" | ".join(parts)} ---', flush=True)

            # Check K-scaling
            k0 = results.get('K=0', float('inf'))
            k1 = results.get('K=1', float('inf'))
            k2 = results.get('K=2', float('inf'))
            k3 = results.get('K=3', float('inf'))
            if k2 < k1 < k0:
                print(f'  >>> K-SCALING DETECTED! K=0={k0:.2f} > K=1={k1:.2f} > K=2={k2:.2f} <<<',
                      flush=True)

    # Final eval
    results = evaluate_k_scaling(model, tokenizer, device, n=20, seq_len=512, bs=2)
    parts = [f"{k}={v:.2f}" for k, v in sorted(results.items())]
    print(f'\n=== Final: {" | ".join(parts)} ===', flush=True)

    k0 = results.get('K=0', 0)
    k3 = results.get('K=3', 0)
    if k3 < k0:
        print(f'  >>> SOLVER BEATS BASE! K=0={k0:.2f} vs K=3={k3:.2f} (delta={k3-k0:+.2f}) <<<',
              flush=True)
    else:
        print(f'  >>> Solver does NOT beat base. K=0={k0:.2f} vs K=3={k3:.2f} <<<', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True)
    parser.add_argument('--K_outer', type=int, default=3)
    parser.add_argument('--grad_last', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    args.grad_last = bool(args.grad_last)
    train(args)
