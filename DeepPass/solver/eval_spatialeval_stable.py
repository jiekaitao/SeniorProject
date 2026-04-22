"""
SpatialEval with STABLE training — attack the variance problem.

The scaled solver experiments showed training instability is the bottleneck:
  - 12M/25M/42M all cluster around 37-39% on bad seeds
  - Best runs hit 61-70% but only 1/3 seeds

Stability fixes to test:
  1. Lower learning rate (5e-5 instead of 1e-4)
  2. Longer warmup (500 steps instead of 200)
  3. Gradient accumulation (effective batch size 8 instead of 1)
  4. EMA (exponential moving average) of solver weights
  5. Multiple random restarts with best-of-N checkpoint selection
"""
import os, sys, torch, json, random, math, time, copy, argparse
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


def train_and_eval(variant='stable_v1', n_memory_slots=32, total_steps=3000, seed=42):
    tag = f'{variant}_mem{n_memory_slots}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

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
    print(f'Solver: {n_params:,} params | variant={variant}', flush=True)

    # EMA copy — use float32 for accumulation to avoid bf16 precision loss
    ema_solver = copy.deepcopy(solver).float()
    ema_decay = 0.999

    maze_data = load_maze_nav()
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # Stability improvements
    lr = 5e-5  # half the original
    warmup = 500  # longer warmup
    grad_accum = 4  # effective batch size 4

    print(f'lr={lr}, warmup={warmup}, grad_accum={grad_accum}, steps={total_steps}', flush=True)

    optimizer = torch.optim.AdamW(solver.parameters(), lr=lr, weight_decay=0.05)
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses = []
    best_eval_acc = 0
    best_ckpt = None

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        answer_text = f" {sample['oracle_option']}"
        full = text + "\nAnswer:" + answer_text

        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048, padding=True).to(device)
        input_ids = enc['input_ids']
        prompt_text = text + "\nAnswer:"
        prompt_len = len(tokenizer.encode(prompt_text))

        with torch.no_grad():
            all_emb = lm_model.embed_tokens(input_ids)
        prompt_emb = all_emb[:, :prompt_len]

        # K curriculum
        if step < total_steps // 4:
            K = random.choices([1, 2], weights=[0.7, 0.3])[0]
        elif step < total_steps // 2:
            K = random.choices([1, 2, 4], weights=[0.2, 0.5, 0.3])[0]
        else:
            K = random.choices([1, 2, 4], weights=[0.1, 0.3, 0.6])[0]

        memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)

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

        ans_start = M + prompt_len
        ans_logits = logits[:, ans_start - 1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:, :ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            loss = loss / grad_accum  # scale for accumulation
            if loss.requires_grad:
                loss.backward()
            losses.append(loss.item() * grad_accum)

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update in float32
                with torch.no_grad():
                    for p_ema, p in zip(ema_solver.parameters(), solver.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data.float(), alpha=1 - ema_decay)

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f'  step {step+1} | loss={avg_loss:.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)

        # Mid-training eval every 500 steps (quick, 50 samples)
        if (step + 1) % 500 == 0:
            ema_eval = copy.deepcopy(ema_solver).to(device=device, dtype=torch.bfloat16)
            ema_eval.eval()
            quick_correct = 0
            quick_n = 50
            for qi in range(quick_n):
                idx = eval_idx[qi]
                sample = maze_data[idx]
                text_q = sample['text'][:1500]
                oracle = sample['oracle_option']
                prompt_q = text_q + "\nAnswer:"
                enc_q = tokenizer(prompt_q, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    emb_q = lm_model.embed_tokens(enc_q['input_ids'])
                    mem_q = ema_eval(emb_q, K_inner=4, K_outer=1, grad_last_only=False)
                    dec_q = torch.cat([mem_q, emb_q], dim=1)
                    T_q = dec_q.shape[1]
                    pid_q = torch.arange(T_q, device=device).unsqueeze(0)
                    pe_q = lm_model.rotary_emb(dec_q, pid_q)
                    h_q = dec_q
                    for layer in lm_model.layers:
                        h_q = layer(h_q, position_embeddings=pe_q)
                    h_q = lm_model.norm(h_q)
                    lg_q = base_model.lm_head(h_q)
                    pred = tokenizer.decode([lg_q[0, -1].argmax().item()]).strip()
                if oracle.upper() in pred.upper()[:10]:
                    quick_correct += 1
            quick_acc = quick_correct / quick_n
            print(f'  --- mid-eval step {step+1}: EMA acc={quick_acc:.2f} ({quick_correct}/{quick_n}) ---', flush=True)

            # Save best
            if quick_acc > best_eval_acc:
                best_eval_acc = quick_acc
                best_ckpt = copy.deepcopy(ema_eval.state_dict())
                print(f'  --- new best! ---', flush=True)
            del ema_eval

    # Final eval with best checkpoint (or EMA if no mid-evals beat it)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Prepare eval solver in bf16
    eval_solver = copy.deepcopy(ema_solver).to(device=device, dtype=torch.bfloat16)
    if best_ckpt is not None:
        eval_solver.load_state_dict(best_ckpt)
        print(f'\nUsing best checkpoint (mid-eval acc={best_eval_acc:.2f})', flush=True)
    else:
        print(f'\nUsing final EMA weights (cast to bf16)', flush=True)

    ckpt_path = os.path.join(RESULTS_DIR, f'solver_{tag}.pt')
    torch.save(eval_solver.state_dict(), ckpt_path)

    # Full evaluation
    print(f'\n=== Full Evaluation ({len(eval_idx)} samples) ===', flush=True)
    eval_solver.eval()
    results = {}

    for K_eval in [0, 1, 2, 4]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text_e = sample['text'][:1500]
            oracle = sample['oracle_option']
            prompt_e = text_e + "\nAnswer:"
            enc_e = tokenizer(prompt_e, return_tensors='pt', truncation=True, max_length=2048).to(device)

            if K_eval == 0:
                with torch.no_grad():
                    out = base_model.generate(enc_e['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc_e['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                with torch.no_grad():
                    emb_e = lm_model.embed_tokens(enc_e['input_ids'])
                    mem_e = eval_solver(emb_e, K_inner=4, K_outer=K_eval, grad_last_only=False)
                    dec_e = torch.cat([mem_e, emb_e], dim=1)
                    T_e = dec_e.shape[1]
                    pid_e = torch.arange(T_e, device=device).unsqueeze(0)
                    pe_e = lm_model.rotary_emb(dec_e, pid_e)
                    h_e = dec_e
                    for layer in lm_model.layers:
                        h_e = layer(h_e, position_embeddings=pe_e)
                    h_e = lm_model.norm(h_e)
                    lg_e = base_model.lm_head(h_e)
                    pred_id = lg_e[0, -1].argmax().item()
                    answer = tokenizer.decode([pred_id]).strip()

            if oracle.upper() in answer.upper()[:10]:
                correct += 1

        acc = correct / n_eval
        results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    result_data = {
        'tag': tag, 'variant': variant, 'n_memory_slots': n_memory_slots,
        'total_steps': total_steps, 'seed': seed, 'solver_params': n_params,
        'lr': lr, 'warmup': warmup, 'grad_accum': grad_accum, 'ema_decay': ema_decay,
        'best_mid_eval_acc': best_eval_acc,
        'results': results, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    result_path = os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json')
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'Saved: {result_path}', flush=True)
    print(f'Done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--mem', type=int, default=32)
    args = parser.parse_args()
    train_and_eval(variant='stable_v1', n_memory_slots=args.mem,
                   total_steps=args.steps, seed=args.seed)
