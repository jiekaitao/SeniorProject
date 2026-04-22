"""
Thinking Tokens: Instead of iterating in latent space, generate
special "thinking" tokens that feed back into the autoregressive loop.

The model generates N thinking tokens before the answer. These tokens
are produced by a small trainable head, fed back as input embeddings,
and the decoder uses them for reasoning. More thinking tokens = more
"external iteration" through the autoregressive loop.

This works WITH the LLM's existing recurrence (autoregression)
instead of fighting against it.
"""
import os, sys, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


class ThinkingTokenModule(nn.Module):
    """
    Generates thinking token embeddings that get prepended to the answer.
    The decoder processes [prompt][thinking_tokens][answer] autoregressively.
    More thinking tokens = more intermediate computation.
    """
    def __init__(self, base_model, n_thinking=8):
        super().__init__()
        self.base_model = base_model
        for p in base_model.parameters():
            p.requires_grad = False

        d = base_model.config.hidden_size  # 4096

        # Thinking token generator: takes last hidden state, produces N embeddings
        self.n_thinking = n_thinking
        self.think_proj = nn.Linear(d, d * n_thinking, bias=False)
        self.think_norm = nn.LayerNorm(d)

        # Each thinking token also gets a learned position bias
        self.think_pos = nn.Parameter(torch.randn(1, n_thinking, d) * 0.01)

    def forward(self, input_ids, labels=None, prompt_len=None, n_think=None):
        base = self.base_model
        model = base.model
        B, T = input_ids.shape
        device = input_ids.device

        if prompt_len is None:
            prompt_len = T // 2
        if n_think is None:
            n_think = self.n_thinking

        # Run frozen decoder on prompt to get prompt representation
        with torch.no_grad():
            all_embeds = model.embed_tokens(input_ids)

        prompt_embeds = all_embeds[:, :prompt_len]

        # Run prompt through frozen decoder to get contextualized representation
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        position_embeddings = model.rotary_emb(prompt_embeds, position_ids)
        h = prompt_embeds
        with torch.no_grad():
            for layer in model.layers:
                h = layer(h, position_embeddings=position_embeddings)
            h = model.norm(h)

        # Generate thinking tokens from last prompt position
        last_h = h[:, -1:]  # (B, 1, D)
        D = last_h.shape[-1]
        think_raw = self.think_proj(last_h).view(B, self.n_thinking, D)  # always reshape to max
        think_raw = think_raw[:, :n_think]  # slice to requested count
        think_embeds = self.think_norm(think_raw) + self.think_pos[:, :n_think]

        # Build full sequence: [prompt_embeds][thinking_tokens][answer_embeds]
        answer_embeds = all_embeds[:, prompt_len:]
        full_embeds = torch.cat([all_embeds[:, :prompt_len], think_embeds, answer_embeds], dim=1)

        # Run full decoder
        T_full = full_embeds.shape[1]
        position_ids_full = torch.arange(T_full, device=device).unsqueeze(0)
        position_embeddings_full = model.rotary_emb(full_embeds, position_ids_full)

        h = full_embeds
        for layer in model.layers:
            h = layer(h, position_embeddings=position_embeddings_full)
        h = model.norm(h)
        logits = base.lm_head(h)

        # Loss on answer tokens only (after prompt + thinking)
        answer_start = prompt_len + n_think
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


def get_math_stream(tokenizer, seq_len, batch_size):
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    token_buffer = []
    while True:
        try:
            ex = next(math_ds)
            text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
        except:
            continue
        if len(text) < 50: continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len*2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)
        while len(token_buffer) >= (seq_len+1)*batch_size:
            batch = [token_buffer[:seq_len+1] for _ in range(batch_size)]
            token_buffer = token_buffer[seq_len*batch_size:]
            yield torch.tensor(batch, dtype=torch.long)


def main():
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)

    model = ThinkingTokenModule(base_model, n_thinking=16)
    # Move trainable params to device
    model.think_proj = model.think_proj.to(device=device, dtype=torch.bfloat16)
    model.think_norm = model.think_norm.to(device=device, dtype=torch.bfloat16)
    model.think_pos = nn.Parameter(model.think_pos.to(device=device, dtype=torch.bfloat16))

    print(f'Trainable: {model.count_trainable():,}', flush=True)
    print(f'Max thinking tokens: 16', flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=1e-3, weight_decay=0.01)
    total_steps = 10000
    warmup = 500
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    print(f'\n=== Thinking Tokens: external iteration through autoregressive loop ===', flush=True)
    data_stream = get_math_stream(tokenizer, seq_len=512, batch_size=2)
    model.train()
    step = 0; running_loss = 0; t0 = time.time()

    for batch in data_stream:
        if step >= total_steps: break
        batch = batch.to(device)
        prompt_len = batch.shape[1] // 2

        # Train with variable number of thinking tokens
        n_think = random.choice([2, 4, 8, 16])
        _, loss = model(batch[:, :-1], labels=batch, prompt_len=prompt_len, n_think=n_think)

        if loss is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if loss is not None: running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            print(f'  step {step:5d} | loss={running_loss/100:.4f} | n_think={n_think} | {time.time()-t0:.0f}s',
                  flush=True)
            running_loss = 0

        if step % 1000 == 0:
            model.eval()
            # Eval: different numbers of thinking tokens
            ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
            buf = []
            while len(buf) < 513*2*10+500:
                ex = next(ds)
                text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
                toks = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=1024)
                toks.append(tokenizer.eos_token_id)
                buf.extend(toks)

            for nt in [0, 2, 4, 8, 16]:
                total_loss = total_tok = 0
                bc = list(buf)
                with torch.no_grad():
                    for _ in range(10):
                        b = [bc[:513] for _ in range(2)]
                        bc = bc[512*2:]
                        t = torch.tensor(b, dtype=torch.long).to(device)
                        pl = t.shape[1] // 2
                        if nt == 0:
                            out = base_model(t[:, :-1], labels=t[:, :-1])
                            total_loss += out.loss.item() * t[:, 1:].numel()
                            total_tok += t[:, 1:].numel()
                        else:
                            _, l = model(t[:, :-1], labels=t, prompt_len=pl, n_think=nt)
                            if l is not None:
                                total_loss += l.item() * (t.shape[1] - pl)
                                total_tok += t.shape[1] - pl
                ppl = math.exp(total_loss / max(total_tok, 1))
                print(f'  think={nt:2d}: PPL={ppl:.2f}', flush=True)
            model.train()

    print(f'\n=== Done ===', flush=True)

if __name__ == '__main__':
    main()
