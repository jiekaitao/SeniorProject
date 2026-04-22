"""
Text-Encoded Graph Reachability: convert the grid task to TEXT.
The grid is described as text tokens, solver processes text, decoder answers.
If K-scaling transfers from grids to text-encoded grids, text iteration works.

Format:
  "Grid: .#..#./.....#/##.#../...#..\nReachable from (0,0): "
  Answer: "1,0,0,1,0,0/0,1,1,1,1,0/..."
"""
import torch, torch.nn as nn, torch.nn.functional as F
import random, math, time
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import SolverCore

device = torch.device('cuda')
N = 8  # 8x8 grid as text

def make_text_grid_batch(tokenizer, bs, n=8):
    """Generate text-encoded grid reachability problems."""
    prompts = []
    answers = []
    for _ in range(bs):
        grid = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if random.random() < 0.3 and (i, j) != (0, 0):
                    grid[i][j] = 1
        # BFS reachability
        reach = [[0]*n for _ in range(n)]
        reach[0][0] = 1
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(n):
                    if reach[i][j] == 1 and grid[i][j] == 0:
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = i+di, j+dj
                            if 0 <= ni < n and 0 <= nj < n and reach[ni][nj] == 0 and grid[ni][nj] == 0:
                                reach[ni][nj] = 1
                                changed = True

        grid_str = '/'.join(''.join('#' if c else '.' for c in row) for row in grid)
        reach_str = '/'.join(''.join(str(c) for c in row) for row in reach)
        prompts.append(f"Grid {n}x{n}: {grid_str}\nReachable from (0,0): ")
        answers.append(reach_str)
    return prompts, answers

def main():
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters(): p.requires_grad = False

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=16).to(device=device, dtype=torch.bfloat16)
    print(f'Solver params: {sum(p.numel() for p in solver.parameters()):,}', flush=True)

    optimizer = torch.optim.AdamW([p for p in solver.parameters() if p.requires_grad],
                                   lr=1e-4, weight_decay=0.05)

    model = base_model.model
    print(f'\n=== Text-Encoded Graph Reachability ({N}x{N}) ===', flush=True)
    t0 = time.time()

    for step in range(5000):
        prompts, answers = make_text_grid_batch(tokenizer, bs=4, n=N)
        full_texts = [p + a for p, a in zip(prompts, answers)]

        enc = tokenizer(full_texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=512).to(device)
        input_ids = enc['input_ids']
        prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
        avg_prompt_len = sum(prompt_lens) // len(prompt_lens)

        with torch.no_grad():
            all_embeds = model.embed_tokens(input_ids)

        prompt_embeds = all_embeds[:, :avg_prompt_len]
        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
        memory = solver(prompt_embeds, K_inner=4, K_outer=K, grad_last_only=True)

        # No bypass: decoder sees [memory][stub][answer]
        stub = all_embeds[:, avg_prompt_len-16:avg_prompt_len]
        answer_embeds = all_embeds[:, avg_prompt_len:]
        dec_input = torch.cat([memory, stub, answer_embeds], dim=1)
        M = memory.shape[1]
        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = model.rotary_emb(dec_input, pos_ids)
        h = dec_input
        for layer in model.layers:
            h = layer(h, position_embeddings=pos_emb)
        h = model.norm(h)
        logits = base_model.lm_head(h)

        ans_start = M + 16
        ans_logits = logits[:, ans_start-1:-1]
        ans_labels = input_ids[:, avg_prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:,:ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:,:ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step+1) % 100 == 0:
            print(f'  step {step+1:5d} | loss={loss.item():.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)

        if (step+1) % 500 == 0:
            solver.eval()
            for Ke in [1, 2, 4]:
                prompts_e, answers_e = make_text_grid_batch(tokenizer, bs=8, n=N)
                full_e = [p + a for p, a in zip(prompts_e, answers_e)]
                enc_e = tokenizer(full_e, return_tensors='pt', padding=True,
                                 truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    emb_e = model.embed_tokens(enc_e['input_ids'])
                    pe = emb_e[:, :avg_prompt_len]
                    mem = solver(pe, K_inner=4, K_outer=Ke, grad_last_only=True)
                    stub_e = emb_e[:, avg_prompt_len-16:avg_prompt_len]
                    ans_e = emb_e[:, avg_prompt_len:]
                    di = torch.cat([mem, stub_e, ans_e], dim=1)
                    pi = torch.arange(di.shape[1], device=device).unsqueeze(0)
                    pe2 = model.rotary_emb(di, pi)
                    h2 = di
                    for layer in model.layers:
                        h2 = layer(h2, position_embeddings=pe2)
                    h2 = model.norm(h2)
                    lg = base_model.lm_head(h2)
                    al = lg[:, M+16-1:-1]
                    tgt = enc_e['input_ids'][:, avg_prompt_len:]
                    ml2 = min(al.shape[1], tgt.shape[1])
                    if ml2 > 0:
                        l = F.cross_entropy(al[:,:ml2].reshape(-1, lg.shape[-1]),
                                           tgt[:,:ml2].reshape(-1), ignore_index=tokenizer.pad_token_id)
                        ppl = math.exp(l.item())
                    else:
                        ppl = float('inf')
                print(f'  K={Ke}: PPL={ppl:.2f}', flush=True)
            solver.train()

    print(f'=== Done ({time.time()-t0:.0f}s) ===', flush=True)

if __name__ == '__main__':
    main()
