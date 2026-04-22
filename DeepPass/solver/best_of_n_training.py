"""
Best-of-N Training: Run N short trials, pick the best, continue training.

The training variance problem: avg ~40%, best ~70% across 30+ runs.
Hypothesis: good vs bad seeds diverge in the first few hundred steps.
Solution: screen many seeds cheaply, then invest compute in winners.

Phase 1: Train N=10 seeds for 500 steps each (~2 min per seed)
Phase 2: Pick top-K by quick eval
Phase 3: Continue top-K for 5000 more steps with best-of-run checkpointing
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


def quick_eval(solver, base_model, lm_model, tokenizer, maze_data, eval_idx, n_eval=50):
    """Fast eval on n_eval samples, returns accuracy."""
    solver.eval()
    correct = 0
    for i in range(min(n_eval, len(eval_idx))):
        idx = eval_idx[i]
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option']
        prompt = text + "\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            emb = lm_model.embed_tokens(enc['input_ids'])
            mem = solver(emb, K_inner=4, K_outer=1, grad_last_only=False)
            dec = torch.cat([mem, emb], dim=1)
            T = dec.shape[1]
            pid = torch.arange(T, device=device).unsqueeze(0)
            pe = lm_model.rotary_emb(dec, pid)
            h = dec
            for layer in lm_model.layers:
                h = layer(h, position_embeddings=pe)
            h = lm_model.norm(h)
            lg = base_model.lm_head(h)
            pred = tokenizer.decode([lg[0, -1].argmax().item()]).strip()
        if oracle.upper() in pred.upper()[:10]:
            correct += 1
    solver.train()
    return correct / min(n_eval, len(eval_idx))


def train_steps(solver, optimizer, scheduler, base_model, lm_model, tokenizer,
                maze_data, train_idx, start_step, end_step, t0):
    """Train solver for a range of steps. Returns list of losses."""
    losses = []
    for step in range(start_step, end_step):
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

        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
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
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f'  step {step+1} | loss={avg_loss:.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)

    return losses


def main(n_seeds=10, screen_steps=500, continue_steps=5000, top_k=2):
    t0 = time.time()
    print(f'=== Best-of-{n_seeds} Training (screen {screen_steps} steps, continue {continue_steps}, top-{top_k}) ===', flush=True)

    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    maze_data = load_maze_nav()
    indices = list(range(len(maze_data)))
    random.seed(0)
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # ========== PHASE 1: Screen N seeds ==========
    print(f'\n--- Phase 1: Screening {n_seeds} seeds for {screen_steps} steps each ---', flush=True)
    candidates = []

    for seed_i in range(n_seeds):
        seed = seed_i * 1000 + 7
        random.seed(seed)
        torch.manual_seed(seed)

        solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                            n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)

        total_steps = screen_steps
        optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
        warmup = min(200, screen_steps // 3)
        def lr_sched(step, ws=warmup, ts=total_steps):
            if step < ws: return step / ws
            return 0.5 * (1 + math.cos(math.pi * (step - ws) / (ts - ws)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

        print(f'\n  Seed {seed}:', flush=True)
        losses = train_steps(solver, optimizer, scheduler, base_model, lm_model,
                           tokenizer, maze_data, train_idx, 0, screen_steps, t0)

        # Quick eval
        acc = quick_eval(solver, base_model, lm_model, tokenizer, maze_data, eval_idx, n_eval=100)
        avg_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 99
        print(f'  Seed {seed}: acc={acc:.2%}, final_loss={avg_loss:.4f}', flush=True)

        candidates.append({
            'seed': seed,
            'accuracy': acc,
            'final_loss': avg_loss,
            'solver_state': copy.deepcopy(solver.state_dict()),
        })
        del solver, optimizer, scheduler
        torch.cuda.empty_cache()

    # Sort by accuracy
    candidates.sort(key=lambda c: c['accuracy'], reverse=True)
    print(f'\n--- Screening Results ---', flush=True)
    for i, c in enumerate(candidates):
        marker = ' <<<' if i < top_k else ''
        print(f'  #{i+1} seed={c["seed"]}: acc={c["accuracy"]:.2%}, loss={c["final_loss"]:.4f}{marker}', flush=True)

    # ========== PHASE 2: Continue top-K ==========
    print(f'\n--- Phase 2: Continuing top-{top_k} for {continue_steps} more steps ---', flush=True)
    final_results = []

    for rank in range(top_k):
        c = candidates[rank]
        seed = c['seed']
        random.seed(seed)
        torch.manual_seed(seed + 999)  # Different seed for continuation randomness

        solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                            n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
        solver.load_state_dict(c['solver_state'])

        total_steps = continue_steps
        optimizer = torch.optim.AdamW(solver.parameters(), lr=5e-5, weight_decay=0.05)
        warmup = 200
        def lr_sched(step, ws=warmup, ts=total_steps):
            if step < ws: return step / ws
            return 0.5 * (1 + math.cos(math.pi * (step - ws) / (ts - ws)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

        print(f'\n  Continuing seed {seed} (rank #{rank+1}, screen acc={c["accuracy"]:.2%}):', flush=True)

        # Train with periodic eval and best checkpoint
        best_acc = c['accuracy']
        best_state = copy.deepcopy(solver.state_dict())
        all_losses = []

        for phase_step in range(0, continue_steps, 500):
            end = min(phase_step + 500, continue_steps)
            losses = train_steps(solver, optimizer, scheduler, base_model, lm_model,
                               tokenizer, maze_data, train_idx, phase_step, end, t0)
            all_losses.extend(losses)

            # Eval every 500 steps
            acc = quick_eval(solver, base_model, lm_model, tokenizer, maze_data, eval_idx, n_eval=100)
            print(f'  step {end}: acc={acc:.2%}, loss={sum(losses[-50:])/max(len(losses[-50:]),1):.4f}', flush=True)
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(solver.state_dict())
                print(f'  --- new best! ---', flush=True)

        # Final full eval with best checkpoint
        solver.load_state_dict(best_state)
        solver.eval()

        os.makedirs(RESULTS_DIR, exist_ok=True)
        tag = f'bestofn_rank{rank}_seed{seed}'
        torch.save(best_state, os.path.join(RESULTS_DIR, f'solver_{tag}.pt'))

        print(f'\n  === Full eval for seed {seed} (best_mid_acc={best_acc:.2%}) ===', flush=True)
        results = {}
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
                    with torch.no_grad():
                        out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                                   pad_token_id=tokenizer.pad_token_id)
                        answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                else:
                    with torch.no_grad():
                        emb = lm_model.embed_tokens(enc['input_ids'])
                        mem = solver(emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                        dec = torch.cat([mem, emb], dim=1)
                        T = dec.shape[1]
                        pid = torch.arange(T, device=device).unsqueeze(0)
                        pe = lm_model.rotary_emb(dec, pid)
                        h = dec
                        for layer in lm_model.layers:
                            h = layer(h, position_embeddings=pe)
                        h = lm_model.norm(h)
                        lg = base_model.lm_head(h)
                        answer = tokenizer.decode([lg[0, -1].argmax().item()]).strip()
                if oracle.upper() in answer.upper()[:10]:
                    correct += 1
            acc = correct / n_eval
            results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
            print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

        final_results.append({
            'seed': seed, 'rank': rank, 'screen_acc': c['accuracy'],
            'best_mid_acc': best_acc, 'results': results,
        })

        result_data = {
            'tag': tag, 'method': 'best_of_n',
            'n_seeds': n_seeds, 'screen_steps': screen_steps,
            'continue_steps': continue_steps, 'top_k': top_k,
            'seed': seed, 'rank': rank,
            'screen_accuracy': c['accuracy'],
            'best_mid_accuracy': best_acc,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
            json.dump(result_data, f, indent=2)

        del solver
        torch.cuda.empty_cache()

    print(f'\n=== Final Summary ===', flush=True)
    for fr in final_results:
        k1 = fr['results'].get('K=1', {}).get('accuracy', 0)
        print(f'  Seed {fr["seed"]} (rank #{fr["rank"]+1}): screen={fr["screen_acc"]:.2%} → final K=1={k1:.2%}', flush=True)
    print(f'Done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--screen-steps', type=int, default=500)
    parser.add_argument('--continue-steps', type=int, default=5000)
    parser.add_argument('--top-k', type=int, default=2)
    args = parser.parse_args()
    main(n_seeds=args.n_seeds, screen_steps=args.screen_steps,
         continue_steps=args.continue_steps, top_k=args.top_k)
