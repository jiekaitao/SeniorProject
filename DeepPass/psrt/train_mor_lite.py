"""
PSRT-MoR-lite Training: 3-phase curriculum per GPT-5.4 Pro plan.

Phase A (3k steps): fixed K=2, uniform routing, only expert FFNs train
Phase B (5k steps): K={2,3} curriculum, soft routing, load balance
Phase C (4k steps): K={1-4}, top-2 routing, halting enabled

Mixed data: 50% fineweb-edu + 25% OpenMathInstruct + 25% ARC

Usage:
    python train_mor_lite.py --total_steps 12000
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from mor_lite import PSRTMoRLite, MoRLiteConfig, create_mor_lite


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def mixed_stream(tokenizer, seq_len, batch_size):
    from datasets import load_dataset
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    science = iter(load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train', streaming=True))

    token_buffer = []

    def get_text(source):
        try:
            if source == 'general':
                return next(general).get('text', '')
            elif source == 'math':
                ex = next(math_ds)
                return f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
            elif source == 'science':
                ex = next(science)
                q = ex.get('question', '')
                choices = ex.get('choices', {})
                labels = choices.get('label', [])
                texts = choices.get('text', [])
                ak = ex.get('answerKey', '')
                cs = ' '.join(f"({l}) {t}" for l, t in zip(labels, texts))
                at = next((t for l, t in zip(labels, texts) if l == ak), '')
                return f"Question: {q}\n{cs}\nAnswer: ({ak}) {at}"
        except (StopIteration, Exception):
            return ''

    while True:
        r = random.random()
        text = get_text('general' if r < 0.50 else ('math' if r < 0.75 else 'science'))
        if not text or len(text) < 30:
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
            t = torch.tensor(batch, dtype=torch.long)
            yield t[:, :-1], t[:, 1:]


def evaluate_ppl(model, tokenizer, device, n=20, seq_len=512, bs=4):
    model.eval()
    from datasets import load_dataset
    ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                           split='train', streaming=True))
    buf = []
    results = {}
    for K in [1, 2]:
        total_loss = total_tok = 0
        with torch.no_grad():
            for i in range(n):
                while len(buf) < (seq_len + 1) * bs:
                    ex = next(ds)
                    t = ex.get('text', '')
                    if t and len(t) > 30:
                        toks = tokenizer.encode(t, add_special_tokens=False, truncation=True,
                                                max_length=seq_len * 2)
                        toks.append(tokenizer.eos_token_id)
                        buf.extend(toks)
                batch = []
                for _ in range(bs):
                    batch.append(buf[:seq_len + 1])
                    buf = buf[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                _, loss, _ = model(t[:, :-1], labels=t[:, :-1], fixed_K=K)
                total_loss += loss.item() * t[:, 1:].numel()
                total_tok += t[:, 1:].numel()
        results[f'K={K}'] = math.exp(total_loss / max(total_tok, 1))
    model.train()
    return results


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = create_mor_lite(args.size)
    model = model.to(device)
    tokenizer = get_tokenizer()

    total = args.total_steps
    phase_a_end = int(total * 0.25)   # 3k of 12k
    phase_b_end = int(total * 0.67)   # 8k of 12k
    warmup = int(total * 0.05)

    # Phase A: only expert FFNs train, everything else frozen
    expert_params = []
    other_params = []
    for name, p in model.named_parameters():
        if 'expert_ffn' in name:
            expert_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': expert_params, 'lr': args.lr},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=0.01, betas=(0.9, 0.95))

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_schedule, lr_schedule])

    save_dir = args.save_dir or f'psrt/checkpoints/mor_lite'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== PSRT-MoR-lite Training ===', flush=True)
    print(f'  {model.count_params()/1e6:.0f}M params, {cfg.n_experts} experts', flush=True)
    print(f'  Steps: {total} (A:{phase_a_end} B:{phase_b_end} C:{total})', flush=True)
    print(f'  Data: 50% general + 25% math + 25% science', flush=True)

    gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
    model.train()

    # Phase A: freeze non-expert params
    for p in other_params:
        p.requires_grad = False
    for p in expert_params:
        p.requires_grad = True

    step = 0
    running_loss = 0
    running_ek = 0
    t0 = time.time()
    best_ppl = float('inf')

    for input_ids, labels in gen:
        if step >= total:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Phase transitions
        if step == phase_a_end:
            print(f'\n=== Phase B: K={{2,3}} + soft routing ===', flush=True)
            for p in other_params:
                p.requires_grad = True
            # Lower LR for shared params
            optimizer.param_groups[1]['lr'] = args.lr * 0.3

        if step == phase_b_end:
            print(f'\n=== Phase C: K={{1-4}} + top-2 + halting ===', flush=True)

        # Determine K and routing mode
        if step < phase_a_end:
            fixed_K = 2
            train_halting = False
            uniform_routing = True
            phase = 'A'
        elif step < phase_b_end:
            fixed_K = random.choices([2, 3], weights=[0.60, 0.40])[0]
            train_halting = False
            uniform_routing = False
            phase = 'B'
        else:
            fixed_K = random.choices([1, 2, 3, 4], weights=[0.15, 0.40, 0.30, 0.15])[0]
            train_halting = True
            uniform_routing = False
            phase = 'C'

        logits, loss, aux = model(input_ids, labels=labels, fixed_K=fixed_K,
                                   train_halting=train_halting, uniform_routing=uniform_routing)

        if train_halting:
            loss = loss + 0.0005 * aux['expected_steps']

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
            route = aux.get('route_dist', [])
            route_str = ' '.join(f'{r:.2f}' for r in route) if route else '?'
            print(f'  step {step:6d} P{phase} | loss={avg_loss:.4f} | E[K]={avg_ek:.2f} | '
                  f'K={fixed_K} | alpha={aux["alpha"]:.3f} | route=[{route_str}] | '
                  f'{elapsed:.0f}s', flush=True)
            running_loss = running_ek = 0

        if step % 2000 == 0:
            ppl = evaluate_ppl(model, tokenizer, device, n=15, seq_len=args.seq_len,
                               bs=args.batch_size)
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

    # Final
    torch.save({'step': step, 'model_state': model.state_dict(),
                'config': cfg.__dict__}, f'{save_dir}/final.pt')

    ppl = evaluate_ppl(model, tokenizer, device, n=30, seq_len=args.seq_len, bs=args.batch_size)
    delta = ppl['K=2'] - ppl['K=1']
    print(f'\n=== Complete: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} (delta={delta:+.2f}) ===',
          flush=True)
    print(f'  Best PPL: {best_ppl:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='172m', choices=['172m', '1b'])
    parser.add_argument('--total_steps', type=int, default=12000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
