"""
Direct soft-prefix optimization diagnostic.
Optimize memory tokens directly (no solver) to find the interface ceiling.
If optimized 16 tokens still hit 1.89, the interface IS the ceiling.
If 64 beats 16, width is the bottleneck.
If optimized 16 beats 1.89, solver/training dynamics are the bottleneck.
"""
import os, sys, time, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    device = torch.device('cuda')
    model_path = 'models/full/Llama-3.1-8B'
    print(f'Loading {model_path}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
    for p in model.parameters(): p.requires_grad = False

    # Build eval data
    ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    buf = []
    while len(buf) < 513 * 2 * 30 + 500:
        ex = next(ds)
        text = f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
        toks = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=1024)
        toks.append(tokenizer.eos_token_id)
        buf.extend(toks)

    # Build a fixed eval batch
    eval_batches = []
    buf_copy = list(buf)
    for _ in range(20):
        batch = []
        for _ in range(2):
            batch.append(buf_copy[:513])
            buf_copy = buf_copy[512:]
        eval_batches.append(torch.tensor(batch, dtype=torch.long).to(device))

    def eval_with_prefix(prefix_embeds):
        """Evaluate PPL with given prefix embeddings prepended."""
        total_loss = total_tok = 0
        with torch.no_grad():
            for t in eval_batches:
                input_ids = t[:, :-1]
                labels = t[:, 1:]
                embeds = model.model.embed_tokens(input_ids)
                B = embeds.shape[0]
                prefix = prefix_embeds.unsqueeze(0).expand(B, -1, -1)
                augmented = torch.cat([prefix, embeds], dim=1)
                M = prefix.shape[1]
                T_total = augmented.shape[1]
                position_ids = torch.arange(T_total, device=device).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(augmented, position_ids)
                h = augmented
                for layer in model.model.layers:
                    h = layer(h, position_embeddings=position_embeddings)
                h = model.model.norm(h)
                logits = model.lm_head(h)
                logits = logits[:, M:]  # remove prefix positions
                prompt_len = input_ids.shape[1] // 2
                answer_logits = logits[:, prompt_len-1:-1]
                answer_labels = labels[:, prompt_len:]
                min_len = min(answer_logits.shape[1], answer_labels.shape[1])
                if min_len > 0:
                    loss = F.cross_entropy(
                        answer_logits[:, :min_len].reshape(-1, logits.shape[-1]),
                        answer_labels[:, :min_len].reshape(-1))
                    total_loss += loss.item() * min_len * B
                    total_tok += min_len * B
        return math.exp(total_loss / max(total_tok, 1))

    # Baseline: no prefix
    print(f'\n=== Baseline (no prefix) ===', flush=True)
    dummy = torch.zeros(0, 4096, device=device, dtype=torch.bfloat16)
    # Can't prepend 0 tokens, just eval normally
    total_loss = total_tok = 0
    with torch.no_grad():
        for t in eval_batches:
            out = model(t[:, :-1], labels=t[:, :-1])
            total_loss += out.loss.item() * t[:, 1:].numel()
            total_tok += t[:, 1:].numel()
    baseline_ppl = math.exp(total_loss / max(total_tok, 1))
    print(f'  Baseline PPL: {baseline_ppl:.2f}', flush=True)

    # Test M=16, 32, 64, 128
    for M in [16, 32, 64, 128]:
        print(f'\n=== Direct optimize M={M} tokens ===', flush=True)
        prefix = torch.randn(M, 4096, device=device, dtype=torch.float32) * 0.01
        prefix.requires_grad = True
        opt = torch.optim.Adam([prefix], lr=0.01)

        for step in range(300):
            opt.zero_grad()
            # Forward through model with prefix
            total_loss_val = 0
            total_tok_val = 0
            # Use first 5 batches for training
            for t in eval_batches[:5]:
                input_ids = t[:, :-1]
                labels = t[:, 1:]
                embeds = model.model.embed_tokens(input_ids)
                B = embeds.shape[0]
                p = prefix.to(torch.bfloat16).unsqueeze(0).expand(B, -1, -1)
                augmented = torch.cat([p, embeds], dim=1)
                T_total = augmented.shape[1]
                position_ids = torch.arange(T_total, device=device).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(augmented, position_ids)
                h = augmented
                for layer in model.model.layers:
                    h = layer(h, position_embeddings=position_embeddings)
                h = model.model.norm(h)
                logits = model.lm_head(h)
                logits = logits[:, M:]
                prompt_len = input_ids.shape[1] // 2
                answer_logits = logits[:, prompt_len-1:-1]
                answer_labels = labels[:, prompt_len:]
                min_len = min(answer_logits.shape[1], answer_labels.shape[1])
                if min_len > 0:
                    loss = F.cross_entropy(
                        answer_logits[:, :min_len].reshape(-1, logits.shape[-1]),
                        answer_labels[:, :min_len].reshape(-1))
                    loss.backward()
                    total_loss_val += loss.item() * min_len * B
                    total_tok_val += min_len * B

            opt.step()

            if (step + 1) % 50 == 0:
                ppl = math.exp(total_loss_val / max(total_tok_val, 1))
                print(f'  step {step+1:3d} | train PPL={ppl:.2f} | prefix norm={prefix.data.norm():.2f}',
                      flush=True)

        # Final eval on held-out batches
        final_ppl = eval_with_prefix(prefix.data.to(torch.bfloat16))
        print(f'  M={M} final eval PPL: {final_ppl:.2f} (baseline: {baseline_ppl:.2f}, delta: {final_ppl - baseline_ppl:+.2f})',
              flush=True)

    print(f'\n=== Done ===', flush=True)

if __name__ == '__main__':
    main()
