"""
Train Dense Attention Replay (DAR) model.

Mixed K training: 50% K=1, 50% K=2. Explicit K=1 auxiliary loss prevents
replay gradients from ruining the dense path. MLP replay gates regularized
toward zero (attention replay only).

Usage:
    python train_dar.py --size 1b --total_steps 100000
"""

import os, sys, time, math, random, argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from replay_transformer import DenseAttentionReplay, DARConfig, create_dar


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
        text = get_text('general' if r < 0.45 else ('math' if r < 0.65 else 'science'))
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


_eval_cache = None

def evaluate_ppl(model, tokenizer, device, n=20, seq_len=512, bs=4):
    global _eval_cache
    model.eval()

    if _eval_cache is None or len(_eval_cache) < (seq_len + 1) * bs * n * 2 + 1000:
        import signal
        def _timeout_handler(signum, frame):
            raise TimeoutError("Dataset load timed out")
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)
            from datasets import load_dataset
            ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                   split='train', streaming=True))
            buf = []
            target = (seq_len + 1) * bs * n * 2 + 1000
            while len(buf) < target:
                ex = next(ds)
                t = ex.get('text', '')
                if t and len(t) > 30:
                    toks = tokenizer.encode(t, add_special_tokens=False,
                                            truncation=True, max_length=seq_len * 2)
                    toks.append(tokenizer.eos_token_id)
                    buf.extend(toks)
            signal.alarm(0)
            _eval_cache = buf
            print("  [eval: cached data from fineweb]", flush=True)
        except Exception as e:
            signal.alarm(0)
            print(f"  [eval: dataset failed ({e}), using random tokens]", flush=True)
            vocab_size = tokenizer.vocab_size or 32000
            _eval_cache = torch.randint(10, vocab_size, ((seq_len + 1) * bs * n * 2 + 1000,)).tolist()

    buf = list(_eval_cache)
    results = {}
    for K in [1, 2]:
        total_loss = total_tok = 0
        with torch.no_grad():
            for i in range(n):
                batch = []
                for _ in range(bs):
                    batch.append(buf[:seq_len + 1])
                    buf = buf[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                _, loss, _ = model(t[:, :-1], labels=t[:, :-1], K=K)
                total_loss += loss.item() * t[:, 1:].numel()
                total_tok += t[:, 1:].numel()
        results[f'K={K}'] = math.exp(total_loss / max(total_tok, 1))
    model.train()
    return results


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = create_dar(args.size)
    model = model.to(device)
    tokenizer = get_tokenizer()

    total = args.total_steps
    warmup = int(total * 0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01, betas=(0.9, 0.95))

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    save_dir = args.save_dir or 'psrt/checkpoints/dar'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== DAR Training ===', flush=True)
    print(f'  {model.count_params()/1e6:.0f}M params ({model.count_replay_params():,} replay-specific)', flush=True)
    print(f'  Steps: {total}, LR: {args.lr}', flush=True)
    print(f'  Replay layers: {cfg.replay_layers}', flush=True)
    print(f'  Data: 45% general + 20% math + 35% science/reasoning', flush=True)

    gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
    model.train()

    step = 0
    running_loss = 0
    t0 = time.time()
    best_ppl = float('inf')

    for input_ids, labels in gen:
        if step >= total:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Stochastic K: 50% K=1, 50% K=2
        # Keeps K=1 path healthy and avoids forcing compromise every batch
        use_replay = (random.random() < 0.5)
        K = 2 if use_replay else 1

        _, loss, aux = model(input_ids, labels=input_ids, K=K)

        # Explicit K=1 auxiliary loss when training with K=2
        # Prevents replay gradients from ruining the dense path
        if K == 2:
            _, loss_k1, _ = model(input_ids, labels=input_ids, K=1)
            loss = loss + 0.5 * loss_k1

        # Regularize replay MLP gates toward zero (attention replay only)
        mlp_pen = 0.0
        for blk in model.blocks:
            if hasattr(blk, 'replay_mlp_gate'):
                mlp_pen = mlp_pen + (blk.replay_mlp_gate ** 2).sum()
        loss = loss + 1e-4 * mlp_pen

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg_loss = running_loss / 100
            elapsed = time.time() - t0
            # Get gate values
            attn_gates = []
            mlp_gates = []
            for blk in model.blocks:
                if hasattr(blk, 'replay_attn_gate'):
                    attn_gates.append(f'{torch.tanh(blk.replay_attn_gate[0]).item():.4f}')
                    mlp_gates.append(f'{torch.tanh(blk.replay_mlp_gate[0]).item():.4f}')
            ag_str = ','.join(attn_gates) if attn_gates else '?'
            mg_str = ','.join(mlp_gates) if mlp_gates else '?'
            print(f'  step {step:6d} | loss={avg_loss:.4f} | K={K} | '
                  f'attn_g=[{ag_str}] | mlp_g=[{mg_str}] | {elapsed:.0f}s', flush=True)
            running_loss = 0

        if step % 2000 == 0:
            ppl = evaluate_ppl(model, tokenizer, device, n=15,
                               seq_len=args.seq_len, bs=args.batch_size)
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

        if step % 5000 == 0 and step > 0:
            torch.save({
                'step': step, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': cfg.__dict__,
            }, f'{save_dir}/step_{step}.pt')

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
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
