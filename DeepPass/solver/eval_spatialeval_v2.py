"""
SpatialEval v2 — BYPASS mode: decoder sees FULL maze text + solver memory.
The solver AUGMENTS the LLM's reasoning, doesn't replace the input.
Also: more training (5000 steps), larger eval (all 1500 samples).
"""
import os, sys, torch, json, random, math, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore

device = torch.device('cuda')


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def main():
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=16).to(device=device, dtype=torch.bfloat16)
    print(f'Solver params: {sum(p.numel() for p in solver.parameters()):,}', flush=True)

    maze_data = load_maze_nav()
    random.seed(42)

    # Split: 1000 train, 500 eval
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # ===== TRAINING with BYPASS mode =====
    # Decoder sees: [memory_tokens][full_prompt_embeddings][answer_embeddings]
    # This way solver AUGMENTS the prompt, doesn't replace it
    print(f'\n=== Training Solver (BYPASS mode, 5000 steps) ===', flush=True)
    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    t0 = time.time()

    for step in range(500):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        answer_text = f" {sample['oracle_option']}"
        full = text + "\nAnswer:" + answer_text

        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048,
                        padding=True).to(device)
        input_ids = enc['input_ids']
        prompt_text = text + "\nAnswer:"
        prompt_len = len(tokenizer.encode(prompt_text))

        with torch.no_grad():
            all_emb = lm_model.embed_tokens(input_ids)

        prompt_emb = all_emb[:, :prompt_len]
        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
        memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)

        # BYPASS: decoder sees [memory][full_prompt][answer]
        ans_emb = all_emb[:, prompt_len:]
        dec_in = torch.cat([memory, all_emb[:, :prompt_len], ans_emb], dim=1)
        M = memory.shape[1]
        T = dec_in.shape[1]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_in, pos_ids)
        h = dec_in
        for layer in lm_model.layers:
            h = layer(h, position_embeddings=pos_emb)
        h = lm_model.norm(h)
        logits = base_model.lm_head(h)

        # Loss on answer tokens only
        ans_start = M + prompt_len
        ans_logits = logits[:, ans_start-1:-1]
        ans_labels = input_ids[:, prompt_len:]
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
            print(f'  step {step+1} | loss={loss.item():.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)

    # ===== EVALUATION with K-scaling =====
    print(f'\n=== K-Scaling Evaluation (BYPASS mode) ===', flush=True)
    solver.eval()

    for K_eval in [0, 1, 2, 4]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option']
            prompt = text + "\nAnswer:"

            enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)

            if K_eval == 0:
                # Base LLM only
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                with torch.no_grad():
                    all_emb = lm_model.embed_tokens(enc['input_ids'])
                    mem = solver(all_emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                    # BYPASS: [memory][full_prompt]
                    dec_in = torch.cat([mem, all_emb], dim=1)
                    M = mem.shape[1]
                    T = dec_in.shape[1]
                    pos_ids = torch.arange(T, device=device).unsqueeze(0)
                    pe = lm_model.rotary_emb(dec_in, pos_ids)
                    h = dec_in
                    for layer in lm_model.layers:
                        h = layer(h, position_embeddings=pe)
                    h = lm_model.norm(h)
                    lg = base_model.lm_head(h)
                    pred_id = lg[0, -1].argmax().item()
                    answer = tokenizer.decode([pred_id]).strip()

            if oracle.upper() in answer.upper()[:10]:
                correct += 1

        acc = correct / n_eval
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    print(f'\nDone ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
