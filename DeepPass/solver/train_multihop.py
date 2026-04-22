"""
Train solver on multi-hop tasks with variable computational depth.
Tests K-scaling on pointer chasing, variable substitution, and text grids.
"""
import os, sys, time, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore
from multihop_tasks import generate_pointer_chase, generate_variable_sub, generate_text_grid


def main(task_type='mixed'):
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters(): p.requires_grad = False
    model = base_model.model

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=16).to(device=device, dtype=torch.bfloat16)
    print(f'Solver params: {sum(p.numel() for p in solver.parameters()):,}', flush=True)
    print(f'Task: {task_type}', flush=True)

    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    total_steps = 10000
    warmup = 500
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5*(1+math.cos(math.pi*(step-warmup)/(total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    def make_batch(bs=4, min_depth=2, max_depth=8):
        prompts, answers = [], []
        for _ in range(bs):
            depth = random.randint(min_depth, max_depth)
            if task_type == 'pointer':
                p, a, _ = generate_pointer_chase(depth=depth)
            elif task_type == 'variable':
                p, a, _ = generate_variable_sub(depth=depth)
            elif task_type == 'grid':
                p, a, _ = generate_text_grid(n=random.choice([8, 12, 16]))
            else:  # mixed
                t = random.choice(['pointer', 'variable', 'grid'])
                if t == 'pointer':
                    p, a, _ = generate_pointer_chase(depth=depth)
                elif t == 'variable':
                    p, a, _ = generate_variable_sub(depth=depth)
                else:
                    p, a, _ = generate_text_grid(n=random.choice([8, 12]))
            prompts.append(p)
            answers.append(a)
        return prompts, answers

    def forward_solver(prompts, answers, K_outer):
        full_texts = [p + a for p, a in zip(prompts, answers)]
        enc = tokenizer(full_texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=512).to(device)
        input_ids = enc['input_ids']
        prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
        avg_pl = sum(prompt_lens) // len(prompt_lens)

        with torch.no_grad():
            all_embeds = model.embed_tokens(input_ids)

        prompt_embeds = all_embeds[:, :avg_pl]
        memory = solver(prompt_embeds, K_inner=4, K_outer=K_outer, grad_last_only=True)

        M = memory.shape[1]
        stub_len = min(16, avg_pl)
        stub = all_embeds[:, avg_pl-stub_len:avg_pl]
        answer_embeds = all_embeds[:, avg_pl:]
        dec_input = torch.cat([memory, stub, answer_embeds], dim=1)

        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = model.rotary_emb(dec_input, pos_ids)
        h = dec_input
        for layer in model.layers:
            h = layer(h, position_embeddings=pos_emb)
        h = model.norm(h)
        logits = base_model.lm_head(h)

        ans_start = M + stub_len
        ans_logits = logits[:, ans_start-1:-1]
        ans_labels = input_ids[:, avg_pl:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(
                ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                ans_labels[:, :ml].reshape(-1),
                ignore_index=tokenizer.pad_token_id)
            return loss
        return None

    print(f'\n=== Multi-Hop Solver: {task_type} ===', flush=True)
    t0 = time.time()
    step = 0; running_loss = 0

    for step in range(total_steps):
        # Curriculum: start with shallow, increase depth over training
        max_depth = min(2 + step // 1000, 8)
        prompts, answers = make_batch(bs=4, min_depth=2, max_depth=max_depth)
        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]

        loss = forward_solver(prompts, answers, K)
        if loss is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
            optimizer.step(); scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if loss is not None: running_loss += loss.item()

        if (step+1) % 100 == 0:
            print(f'  step {step+1:5d} | loss={running_loss/100:.4f} | max_depth={max_depth} | K={K} | {time.time()-t0:.0f}s',
                  flush=True)
            running_loss = 0

        if (step+1) % 1000 == 0:
            solver.eval()
            # Test K-scaling at different depths
            for test_depth in [2, 4, 6, 8]:
                results = {}
                for Ke in [1, 2, 4, 8]:
                    total_loss = 0; total_tok = 0
                    for _ in range(10):
                        if task_type == 'pointer':
                            ps, ans = zip(*[generate_pointer_chase(depth=test_depth)[:2] for _ in range(4)])
                        elif task_type == 'variable':
                            ps, ans = zip(*[generate_variable_sub(depth=test_depth)[:2] for _ in range(4)])
                        elif task_type == 'grid':
                            ps, ans = zip(*[generate_text_grid(n=8)[:2] for _ in range(4)])
                        else:
                            ps, ans = zip(*[generate_pointer_chase(depth=test_depth)[:2] for _ in range(4)])
                        with torch.no_grad():
                            l = forward_solver(list(ps), list(ans), Ke)
                        if l is not None:
                            total_loss += l.item(); total_tok += 1
                    results[Ke] = math.exp(total_loss / max(total_tok, 1))
                parts = [f'K={k}={v:.2f}' for k, v in sorted(results.items())]
                k1 = results.get(1, 99); k4 = results.get(4, 99)
                flag = " <<< K-SCALING!" if k4 < k1 - 0.05 else ""
                print(f'  depth={test_depth}: {" | ".join(parts)}{flag}', flush=True)
            solver.train()

    print(f'\n=== Done ({time.time()-t0:.0f}s) ===', flush=True)


if __name__ == '__main__':
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else 'mixed'
    main(task_type=task)
