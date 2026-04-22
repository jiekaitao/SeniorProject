"""
Dense Baseline: Standard transformer trained on same data as ARR-PSRT.

For fair comparison: same params, same data, same compute, no recursion.
Trains both 172M and 1.1B configs.

Usage:
    python train_dense_baseline.py --size 172m --total_steps 12000
    python train_dense_baseline.py --size 1b --total_steps 100000
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
from arr_psrt import RMSNorm, GQAttention, TransformerBlock, precompute_rope, ARRConfig


@dataclass
class DenseConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    d_head: int = 64
    ffn_dim: int = 3072
    context_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    n_layers: int = 10
    dropout: float = 0.0


class DenseTransformer(nn.Module):
    """Standard dense transformer — the baseline to beat."""
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Convert to ARRConfig for block compatibility
        arr_cfg = ARRConfig(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            d_head=cfg.d_head, ffn_dim=cfg.ffn_dim,
            context_len=cfg.context_len, rope_base=cfg.rope_base,
            norm_eps=cfg.norm_eps,
        )

        self.layers = nn.ModuleList([TransformerBlock(arr_cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.vocab_size, cfg.d_model, bias=False)
        # Fix: lm_head should be d_model -> vocab_size
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.d_head, cfg.context_len, cfg.rope_base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.final_norm(h))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def dense_172m():
    return DenseConfig(d_model=1024, n_heads=16, n_kv_heads=4, d_head=64,
                       ffn_dim=3072, n_layers=10)

def dense_1b():
    return DenseConfig(d_model=2048, n_heads=32, n_kv_heads=8, d_head=64,
                       ffn_dim=5632, n_layers=24)


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
                batch.append(token_buffer[:seq_len + 1])
                token_buffer = token_buffer[seq_len:]
            t = torch.tensor(batch, dtype=torch.long)
            yield t[:, :-1], t[:, 1:]


def train(args):
    device = torch.device('cuda')
    cfg = dense_172m() if args.size == '172m' else dense_1b()
    model = DenseTransformer(cfg).to(device)
    tokenizer = get_tokenizer()

    n = model.count_params()
    print(f'\n=== Dense Baseline {n/1e6:.0f}M ===', flush=True)
    print(f'  d={cfg.d_model} layers={cfg.n_layers} ffn={cfg.ffn_dim}', flush=True)
    print(f'  Parameters: {n:,}', flush=True)
    print(f'  Steps: {args.total_steps}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                   betas=(0.9, 0.95))
    warmup = int(args.total_steps * 0.05)

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(args.total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    save_dir = args.save_dir or f'psrt/checkpoints/dense_{args.size}'
    os.makedirs(save_dir, exist_ok=True)

    gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
    model.train()

    step = 0
    running_loss = 0
    t0 = time.time()
    best_ppl = float('inf')

    for input_ids, labels in gen:
        if step >= args.total_steps:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        _, loss = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg = running_loss / 100
            elapsed = time.time() - t0
            tps = step * args.batch_size * args.seq_len / elapsed
            print(f'  step {step:6d} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | '
                  f'{tps/1e3:.0f}K tok/s | {elapsed:.0f}s', flush=True)
            running_loss = 0

        if step % 2000 == 0:
            model.eval()
            total_loss = total_tok = 0
            eval_gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
            with torch.no_grad():
                for i, (ids, labs) in enumerate(eval_gen):
                    if i >= 20:
                        break
                    ids, labs = ids.to(device), labs.to(device)
                    _, l = model(ids, labels=labs)
                    total_loss += l.item() * labs[:, 1:].numel()
                    total_tok += labs[:, 1:].numel()
            ppl = math.exp(total_loss / max(total_tok, 1))
            print(f'  --- EVAL step {step}: PPL={ppl:.2f} ---', flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save({'step': step, 'model_state': model.state_dict(),
                            'config': cfg.__dict__, 'ppl': ppl}, f'{save_dir}/best.pt')
                print(f'  --- SAVED best ---', flush=True)
            model.train()

    torch.save({'step': step, 'model_state': model.state_dict(),
                'config': cfg.__dict__}, f'{save_dir}/final.pt')

    model.eval()
    total_loss = total_tok = 0
    eval_gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
    with torch.no_grad():
        for i, (ids, labs) in enumerate(eval_gen):
            if i >= 30:
                break
            ids, labs = ids.to(device), labs.to(device)
            _, l = model(ids, labels=labs)
            total_loss += l.item() * labs[:, 1:].numel()
            total_tok += labs[:, 1:].numel()
    final_ppl = math.exp(total_loss / max(total_tok, 1))

    print(f'\n=== Complete: Final PPL={final_ppl:.2f}, Best PPL={best_ppl:.2f} ===', flush=True)


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
