"""
No-Bypass Solver: decoder sees ONLY [memory_tokens][question_stub], NOT the raw prompt.
Forces the decoder to depend entirely on the solver's output.
If K-scaling appears here, the raw-prompt bypass was killing iteration.
"""
import os, sys, time, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore


class NoBypassSolverLLM(torch.nn.Module):
    """Solver feeds memory to decoder, but decoder does NOT see raw prompt."""
    def __init__(self, base_model, solver_d=1024, solver_heads=16,
                 solver_ffn=2816, solver_L_layers=2, n_memory=32):
        super().__init__()
        self.base_model = base_model
        for p in base_model.parameters():
            p.requires_grad = False

        self.solver = SolverCore(
            d_model=solver_d, n_heads=solver_heads, ffn_dim=solver_ffn,
            n_L_layers=solver_L_layers, n_memory_slots=n_memory,
        )

    def forward(self, input_ids, labels=None, prompt_len=None,
                K_inner=6, K_outer=3, grad_last_only=True, question_stub_len=32):
        base = self.base_model
        model = base.model
        B, T = input_ids.shape
        device = input_ids.device

        if prompt_len is None:
            prompt_len = T // 2

        # Get embeddings
        with torch.no_grad():
            all_embeds = model.embed_tokens(input_ids)

        prompt_embeds = all_embeds[:, :prompt_len]
        answer_embeds = all_embeds[:, prompt_len:]

        # Run solver on full prompt
        memory = self.solver(prompt_embeds, K_inner=K_inner, K_outer=K_outer,
                            grad_last_only=grad_last_only)

        # NO-BYPASS: decoder sees [memory] + [last N prompt tokens as stub] + [answer]
        # NOT the full prompt
        stub_len = min(question_stub_len, prompt_len)
        stub_embeds = all_embeds[:, prompt_len - stub_len:prompt_len]  # last N tokens of prompt

        decoder_input = torch.cat([memory, stub_embeds, answer_embeds], dim=1)
        M = memory.shape[1]
        T_dec = decoder_input.shape[1]

        position_ids = torch.arange(T_dec, device=device).unsqueeze(0)
        position_embeddings = model.rotary_emb(decoder_input, position_ids)

        h = decoder_input
        for layer in model.layers:
            h = layer(h, position_embeddings=position_embeddings)

        h = model.norm(h)
        logits = base.lm_head(h)

        # Loss on answer tokens only (after memory + stub)
        answer_start = M + stub_len
        answer_logits = logits[:, answer_start - 1:-1]
        answer_labels = labels[:, prompt_len:] if labels is not None else None

        loss = None
        if answer_labels is not None:
            min_len = min(answer_logits.shape[1], answer_labels.shape[1])
            if min_len > 0:
                loss = F.cross_entropy(
                    answer_logits[:, :min_len].reshape(-1, logits.shape[-1]),
                    answer_labels[:, :min_len].reshape(-1),
                    ignore_index=-100,
                )

        return logits, loss

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        return sum(p.numel() for p in self.parameters())


def get_math_stream(tokenizer, seq_len, batch_size):
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    token_buffer = []
    def get_text():
        try:
            if random.random() < 0.8:
                ex = next(math_ds)
                return f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
            else:
                return next(general).get('text', '')
        except: return ''
    while True:
        text = get_text()
        if not text or len(text) < 50: continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len*2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)
        while len(token_buffer) >= (seq_len+1)*batch_size:
            batch = [token_buffer[:seq_len+1] for _ in range(batch_size)]
            token_buffer = token_buffer[seq_len*batch_size:]
            yield torch.tensor(batch, dtype=torch.long)


def evaluate_k_scaling(model, tokenizer, device, n=10, seq_len=512, bs=2):
    model.eval()
    ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    buf = []
    while len(buf) < (seq_len+1)*bs*n + 500:
        ex = next(ds)
        text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
        toks = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len*2)
        toks.append(tokenizer.eos_token_id)
        buf.extend(toks)

    results = {}
    for K in [1, 2, 3, 4]:
        total_loss = total_tok = 0
        buf_copy = list(buf)
        with torch.no_grad():
            for _ in range(n):
                batch = [buf_copy[:seq_len+1] for _ in range(bs)]
                buf_copy = buf_copy[seq_len*bs:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                prompt_len = t.shape[1] // 2
                _, loss = model(t[:, :-1], labels=t, prompt_len=prompt_len,
                               K_inner=6, K_outer=K, grad_last_only=True)
                if loss is not None:
                    total_loss += loss.item() * (t.shape[1] - prompt_len)
                    total_tok += t.shape[1] - prompt_len
        results[f'K={K}'] = math.exp(total_loss / max(total_tok, 1))
    model.train()
    return results


def main():
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)

    model = NoBypassSolverLLM(base_model, solver_d=1024, solver_heads=16,
                               solver_ffn=2816, solver_L_layers=2, n_memory=32)
    model.solver = model.solver.to(device=device, dtype=torch.bfloat16)
    print(f'Trainable: {model.count_trainable():,}', flush=True)
    print(f'NO-BYPASS: decoder sees [32 memory] + [32 stub tokens] + [answer], NOT full prompt', flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = 10000; warmup = 1000
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    print(f'\n=== No-Bypass Solver (decoder depends entirely on solver) ===', flush=True)
    data_stream = get_math_stream(tokenizer, seq_len=512, batch_size=2)
    model.train(); step = 0; running_loss = 0; t0 = time.time()

    for batch in data_stream:
        if step >= total_steps: break
        batch = batch.to(device)
        prompt_len = batch.shape[1] // 2
        K = random.choices([2, 3, 4], weights=[0.35, 0.40, 0.25])[0]
        _, loss = model(batch[:, :-1], labels=batch, prompt_len=prompt_len,
                       K_inner=6, K_outer=K, grad_last_only=True)
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
            results = evaluate_k_scaling(model, tokenizer, device, n=10, seq_len=512, bs=2)
            parts = [f'{k}={v:.2f}' for k, v in sorted(results.items())]
            print(f'  --- EVAL step {step}: {" | ".join(parts)} ---', flush=True)
            k1 = results.get('K=1', 99); k2 = results.get('K=2', 99); k3 = results.get('K=3', 99)
            if k2 < k1 - 0.01:
                print(f'  >>> K-SCALING! K=1={k1:.2f} > K=2={k2:.2f} <<<', flush=True)
            if k3 < k2 - 0.01:
                print(f'  >>> MONOTONE K-SCALING! K=1={k1:.2f} > K=2={k2:.2f} > K=3={k3:.2f} <<<', flush=True)

    results = evaluate_k_scaling(model, tokenizer, device, n=20, seq_len=512, bs=2)
    parts = [f'{k}={v:.2f}' for k, v in sorted(results.items())]
    print(f'\n=== Final: {" | ".join(parts)} ===', flush=True)

if __name__ == '__main__':
    main()
