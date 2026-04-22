"""
Text K-Scaling: Apply the local reachability lesson to text tasks.

The reachability experiment proved K-scaling works when:
1. One cycle is computationally insufficient (local receptive field too small)
2. The task has genuine iterative depth

For text, we create this by:
- Using a TINY solver with LOCAL attention (no global self-attn, only window attention)
- Using tasks with multi-hop reasoning chains
- Using the no-bypass setup (decoder depends on solver)

If local attention on text creates K-scaling like it did on reachability,
we've proven the principle transfers to language.
"""
import os, sys, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


class LocalAttentionBlock(nn.Module):
    """Attention with a LOCAL window — each token only sees W neighbors."""
    def __init__(self, d_model, n_heads, window=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.norm = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.wg = nn.Linear(d_model, d_model * 2, bias=False)
        self.wd = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Local window mask: each token attends to [i-W, i+W]
        W = self.window
        mask = torch.full((T, T), float('-inf'), device=x.device)
        for i in range(T):
            lo = max(0, i - W)
            hi = min(T, i + W + 1)
            mask[i, lo:hi] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(out)

        h = self.ffn_norm(x)
        x = x + self.wd(F.silu(self.wg(h)))
        return x


class LocalSolverCore(nn.Module):
    """Solver with LOCAL-only attention. K cycles expand receptive field."""
    def __init__(self, d_model=256, n_heads=4, window=32, n_memory=16):
        super().__init__()
        self.d_model = d_model
        self.n_memory = n_memory
        self.proj_in = nn.Linear(4096, d_model, bias=False)
        self.block = LocalAttentionBlock(d_model, n_heads, window)  # SHARED
        self.H_init = nn.Parameter(torch.randn(1, n_memory, d_model) * 0.02)
        # Global cross-attention for H to read from L
        self.H_norm = nn.LayerNorm(d_model)
        self.H_cross_q = nn.Linear(d_model, d_model, bias=False)
        self.H_cross_k = nn.Linear(d_model, d_model, bias=False)
        self.H_cross_v = nn.Linear(d_model, d_model, bias=False)
        self.H_cross_o = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, 4096, bias=False)
        self.out_norm = nn.LayerNorm(4096)

    def forward(self, prompt_embeddings, K_outer=3, K_inner=4, grad_last_only=True):
        B, T, _ = prompt_embeddings.shape
        e = self.proj_in(prompt_embeddings)
        z_L = 0.1 * e
        z_H = self.H_init.expand(B, -1, -1).clone()

        for s in range(K_outer):
            use_grad = (not grad_last_only) or (s == K_outer - 1)
            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                for _ in range(K_inner):
                    z_L = self.block(z_L + e)  # LOCAL attn + raw input injection

                # H reads from L via global cross-attention
                h_q = self.H_cross_q(self.H_norm(z_H))
                h_k = self.H_cross_k(z_L)
                h_v = self.H_cross_v(z_L)
                Nh = self.d_model // (self.d_model // 4)
                Dh = self.d_model // Nh
                h_q = h_q.view(B, self.n_memory, Nh, Dh).transpose(1, 2)
                h_k = h_k.view(B, T, Nh, Dh).transpose(1, 2)
                h_v = h_v.view(B, T, Nh, Dh).transpose(1, 2)
                h_out = F.scaled_dot_product_attention(h_q, h_k, h_v)
                h_out = h_out.transpose(1, 2).contiguous().view(B, self.n_memory, self.d_model)
                z_H = z_H + self.H_cross_o(h_out)

            if grad_last_only and s < K_outer - 1:
                z_L = z_L.detach()
                z_H = z_H.detach()

        return self.out_norm(self.proj_out(z_H))


class LocalSolverLLM(nn.Module):
    """Local solver + frozen decoder. No bypass — decoder sees only memory + stub."""
    def __init__(self, base_model, solver_d=256, n_heads=4, window=32, n_memory=16):
        super().__init__()
        self.base_model = base_model
        for p in base_model.parameters():
            p.requires_grad = False
        self.solver = LocalSolverCore(d_model=solver_d, n_heads=n_heads,
                                       window=window, n_memory=n_memory)

    def forward(self, input_ids, labels=None, prompt_len=None,
                K_inner=4, K_outer=3, grad_last_only=True, stub_len=32):
        base = self.base_model
        model = base.model
        B, T = input_ids.shape
        device = input_ids.device
        if prompt_len is None: prompt_len = T // 2

        with torch.no_grad():
            all_embeds = model.embed_tokens(input_ids)

        prompt_embeds = all_embeds[:, :prompt_len]
        memory = self.solver(prompt_embeds, K_outer=K_outer, K_inner=K_inner,
                            grad_last_only=grad_last_only)

        # No bypass: decoder sees [memory] + [stub] + [answer]
        sl = min(stub_len, prompt_len)
        stub = all_embeds[:, prompt_len - sl:prompt_len]
        answer_embeds = all_embeds[:, prompt_len:]
        decoder_input = torch.cat([memory, stub, answer_embeds], dim=1)

        M = memory.shape[1]
        T_dec = decoder_input.shape[1]
        position_ids = torch.arange(T_dec, device=device).unsqueeze(0)
        position_embeddings = model.rotary_emb(decoder_input, position_ids)

        h = decoder_input
        for layer in model.layers:
            h = layer(h, position_embeddings=position_embeddings)
        h = model.norm(h)
        logits = base.lm_head(h)

        answer_start = M + sl
        answer_logits = logits[:, answer_start - 1:-1]
        answer_labels = labels[:, prompt_len:] if labels is not None else None

        loss = None
        if answer_labels is not None:
            min_len = min(answer_logits.shape[1], answer_labels.shape[1])
            if min_len > 0:
                loss = F.cross_entropy(
                    answer_logits[:, :min_len].reshape(-1, logits.shape[-1]),
                    answer_labels[:, :min_len].reshape(-1), ignore_index=-100)
        return logits, loss

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_math_stream(tokenizer, seq_len, batch_size):
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    token_buffer = []
    while True:
        try:
            ex = next(math_ds)
            text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
        except: continue
        if len(text) < 50: continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len*2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)
        while len(token_buffer) >= (seq_len+1)*batch_size:
            batch = [token_buffer[:seq_len+1] for _ in range(batch_size)]
            token_buffer = token_buffer[seq_len*batch_size:]
            yield torch.tensor(batch, dtype=torch.long)


def evaluate_k(model, tokenizer, device, n=10, seq_len=512, bs=2):
    model.eval()
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    buf = []
    while len(buf) < (seq_len+1)*bs*n + 500:
        ex = next(math_ds)
        text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
        toks = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len*2)
        toks.append(tokenizer.eos_token_id)
        buf.extend(toks)
    results = {}
    for K in [1, 2, 4, 8]:
        total_loss = total_tok = 0
        bc = list(buf)
        with torch.no_grad():
            for _ in range(n):
                batch = [bc[:seq_len+1] for _ in range(bs)]
                bc = bc[seq_len*bs:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                pl = t.shape[1] // 2
                _, loss = model(t[:, :-1], labels=t, prompt_len=pl,
                               K_inner=4, K_outer=K, grad_last_only=True)
                if loss is not None:
                    total_loss += loss.item() * (t.shape[1] - pl)
                    total_tok += t.shape[1] - pl
        results[f'K={K}'] = math.exp(total_loss / max(total_tok, 1))
    model.train()
    return results


def main(window=32):
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained('models/full/Llama-3.1-8B',
                                                 dtype=torch.bfloat16).to(device)

    model = LocalSolverLLM(base, solver_d=256, n_heads=4, window=window, n_memory=16)
    model.solver = model.solver.to(device=device, dtype=torch.bfloat16)
    print(f'Trainable: {model.count_trainable():,}', flush=True)
    print(f'Local window: {window} tokens', flush=True)
    print(f'K=1 receptive field: {window*2+1} tokens', flush=True)
    print(f'K=4 receptive field: ~{(window*2+1)*4} tokens', flush=True)
    print(f'No bypass: decoder sees only [memory]+[stub]+[answer]', flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = 10000; warmup = 500
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    print(f'\n=== Local Solver on Text (window={window}) ===', flush=True)
    data_stream = get_math_stream(tokenizer, seq_len=512, batch_size=2)
    model.train(); step = 0; running_loss = 0; t0 = time.time()

    for batch in data_stream:
        if step >= total_steps: break
        batch = batch.to(device)
        pl = batch.shape[1] // 2
        K = random.choices([2, 4, 8], weights=[0.35, 0.40, 0.25])[0]
        _, loss = model(batch[:, :-1], labels=batch, prompt_len=pl,
                       K_inner=4, K_outer=K, grad_last_only=True)
        if loss is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if loss is not None: running_loss += loss.item()
        step += 1
        if step % 100 == 0:
            print(f'  step {step:5d} | loss={running_loss/100:.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)
            running_loss = 0
        if step % 1000 == 0:
            results = evaluate_k(model, tokenizer, device, n=10, seq_len=512, bs=2)
            parts = [f'{k}={v:.2f}' for k, v in sorted(results.items())]
            print(f'  --- EVAL step {step}: {" | ".join(parts)} ---', flush=True)
            k1 = results.get('K=1', 99); k2 = results.get('K=2', 99); k4 = results.get('K=4', 99)
            if k2 < k1 - 0.01:
                print(f'  >>> K-SCALING! K=1={k1:.2f} > K=2={k2:.2f} <<<', flush=True)
            if k4 < k2 - 0.01:
                print(f'  >>> MONOTONE! K=1={k1:.2f} > K=2={k2:.2f} > K=4={k4:.2f} <<<', flush=True)

    results = evaluate_k(model, tokenizer, device, n=20, seq_len=512, bs=2)
    parts = [f'{k}={v:.2f}' for k, v in sorted(results.items())]
    print(f'\n=== Final: {" | ".join(parts)} ===', flush=True)


if __name__ == '__main__':
    import sys
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    main(window=window)
