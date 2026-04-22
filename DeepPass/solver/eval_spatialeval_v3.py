"""
SpatialEval v3 — Longer training (2000 steps), bypass mode, proper curriculum.
v2 showed 33.4% → 65.0% with only 500 steps. This version:
  - 2000 training steps with cosine LR schedule
  - K-curriculum: start with K=1, ramp to K=4
  - Checkpoint saving for reproducibility
  - Full 500-sample eval at K=0,1,2,4,8
  - Also test with different memory slot counts (8, 16, 32)
"""
import os, sys, torch, json, random, math, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def train_and_eval(n_memory_slots=16, total_steps=2000, tag='v3'):
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=n_memory_slots).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in solver.parameters())
    print(f'Solver params: {n_params:,} | memory_slots={n_memory_slots}', flush=True)

    maze_data = load_maze_nav()
    random.seed(42)

    # Split: 1000 train, rest eval
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # Training with cosine LR + K curriculum
    print(f'\n=== Training ({total_steps} steps, bypass mode) ===', flush=True)
    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    warmup = 200

    def lr_sched(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)
    t0 = time.time()
    losses = []

    for step in range(total_steps):
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

        # K curriculum: start K=1, ramp up
        if step < total_steps // 4:
            K = random.choices([1, 2], weights=[0.7, 0.3])[0]
        elif step < total_steps // 2:
            K = random.choices([1, 2, 4], weights=[0.2, 0.5, 0.3])[0]
        else:
            K = random.choices([1, 2, 4], weights=[0.1, 0.3, 0.6])[0]

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
        ans_logits = logits[:, ans_start - 1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:, :ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            lr = scheduler.get_last_lr()[0]
            print(f'  step {step + 1} | loss={avg_loss:.4f} | K={K} | lr={lr:.6f} | {time.time() - t0:.0f}s',
                  flush=True)

    # Save checkpoint
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f'solver_{tag}_mem{n_memory_slots}.pt')
    torch.save(solver.state_dict(), ckpt_path)
    print(f'Saved checkpoint: {ckpt_path}', flush=True)

    # Evaluation with K-scaling
    print(f'\n=== K-Scaling Evaluation (bypass mode, {len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    results = {}

    for K_eval in [0, 1, 2, 4, 8]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option']
            prompt = text + "\nAnswer:"

            enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)

            if K_eval == 0:
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:],
                                              skip_special_tokens=True).strip()
            else:
                with torch.no_grad():
                    all_emb = lm_model.embed_tokens(enc['input_ids'])
                    mem = solver(all_emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
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
        results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    # Save results
    result_data = {
        'tag': tag,
        'n_memory_slots': n_memory_slots,
        'total_steps': total_steps,
        'solver_params': n_params,
        'final_loss': sum(losses[-50:]) / len(losses[-50:]),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    result_path = os.path.join(RESULTS_DIR, f'spatialeval_{tag}_mem{n_memory_slots}.json')
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'Saved results: {result_path}', flush=True)

    print(f'\n=== Summary ===', flush=True)
    for k, v in results.items():
        print(f'  {k}: {v["accuracy"]:.4f}', flush=True)
    print(f'Done ({time.time() - t0:.0f}s)', flush=True)

    return results


if __name__ == '__main__':
    import sys
    mem_slots = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    tag = sys.argv[3] if len(sys.argv) > 3 else 'v3'
    train_and_eval(n_memory_slots=mem_slots, total_steps=steps, tag=tag)
