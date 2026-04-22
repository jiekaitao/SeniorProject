"""
Hard Maze Curriculum — Train specifically on the mazes solvers usually get wrong.

The ensemble analysis showed 104/500 eval samples are "always wrong" (all 5
independently-trained solvers fail). These represent the ceiling.

Strategy: weighted sampling that oversamples hard mazes during training.
If the solver CAN learn the hard ones with enough focus, the ceiling is training
distribution, not capability. If not, it's fundamental.

Also implements multi-pass diversity: run solver multiple times with
different random perturbations and pick the best.
"""
import os, sys, torch, json, random, math, time, argparse, copy
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


def identify_hard_samples(base_model, lm_model, tokenizer, solver_ckpts, maze_data, eval_idx):
    """Run pre-trained solvers on eval to find always-wrong samples."""
    n_eval = len(eval_idx)
    per_sample_correct = [0] * n_eval

    for ckpt_path in solver_ckpts:
        solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                            n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
        solver.load_state_dict(torch.load(ckpt_path, map_location=device))
        solver.eval()

        for ei, idx in enumerate(eval_idx):
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
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
                answer = tokenizer.decode([lg[0, -1].argmax().item()]).strip()
            if oracle in answer.upper()[:10]:
                per_sample_correct[ei] += 1

        del solver
        torch.cuda.empty_cache()

    n_solvers = len(solver_ckpts)
    hard = [eval_idx[i] for i in range(n_eval) if per_sample_correct[i] == 0]
    easy = [eval_idx[i] for i in range(n_eval) if per_sample_correct[i] == n_solvers]
    medium = [eval_idx[i] for i in range(n_eval) if 0 < per_sample_correct[i] < n_solvers]

    print(f'  Hard (0/{n_solvers} correct): {len(hard)}', flush=True)
    print(f'  Medium: {len(medium)}', flush=True)
    print(f'  Easy ({n_solvers}/{n_solvers} correct): {len(easy)}', flush=True)

    return hard, medium, easy, per_sample_correct


def run_experiment(strategy, seed, total_steps, hard_weight, tokenizer, base_model,
                   lm_model, maze_data, train_idx, eval_idx, hard_train=None):
    tag = f'hardmaze_{strategy}_hw{hard_weight}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  Hard Maze: strategy={strategy}, hard_weight={hard_weight}, seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in solver.parameters())

    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # Build weighted training set
    if strategy == 'weighted' and hard_train:
        # Oversample hard training samples
        easy_train = [i for i in train_idx if i not in set(hard_train)]
        print(f'  Hard train samples: {len(hard_train)}, Easy: {len(easy_train)}', flush=True)
    else:
        hard_train = []
        easy_train = train_idx

    t0 = time.time()
    losses = []

    for step in range(total_steps):
        # Sample with curriculum weighting
        if strategy == 'weighted' and hard_train and random.random() < hard_weight:
            idx = random.choice(hard_train)
        else:
            idx = train_idx[step % len(train_idx)]

        sample = maze_data[idx]
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

        if strategy == 'diverse_pass':
            # Run solver twice with noise and pick the one with lower loss
            noise_scale = 0.01
            memory1 = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)
            # Add small noise to L_init for diversity
            orig_scale = solver.L_init_scale.data.clone()
            solver.L_init_scale.data = orig_scale * (1 + noise_scale * torch.randn_like(orig_scale))
            memory2 = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)
            solver.L_init_scale.data = orig_scale

            # Compute loss for both, use the better one
            def compute_loss(memory):
                dec_in = torch.cat([memory, all_emb], dim=1)
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
                    return F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                           ans_labels[:, :ml].reshape(-1),
                                           ignore_index=tokenizer.pad_token_id)
                return None

            loss1 = compute_loss(memory1)
            loss2 = compute_loss(memory2)
            if loss1 is not None and loss2 is not None:
                loss = loss1 if loss1.item() < loss2.item() else loss2
            elif loss1 is not None:
                loss = loss1
            else:
                loss = loss2
        else:
            memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)
            dec_in = torch.cat([memory, all_emb], dim=1)
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
            loss = None
            if ml > 0:
                loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                       ans_labels[:, :ml].reshape(-1),
                                       ignore_index=tokenizer.pad_token_id)

        if loss is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if loss is not None:
            losses.append(loss.item())

        if (step + 1) % 200 == 0:
            avg_loss = sum(losses[-200:]) / len(losses[-200:])
            print(f'  step {step+1} | loss={avg_loss:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    results = {}

    for K_eval in [0, 1, 2]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            prompt = text + "\nAnswer:"

            if K_eval == 0:
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
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

            if oracle in answer.upper()[:10]:
                correct += 1

        acc = correct / n_eval
        results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': f'hardmaze_{strategy}',
        'strategy': strategy, 'hard_weight': hard_weight,
        'seed': seed, 'total_steps': total_steps,
        'solver_params': n_params,
        'final_loss': sum(losses[-50:]) / max(len(losses[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del solver, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='weighted',
                        choices=['weighted', 'diverse_pass', 'baseline'])
    parser.add_argument('--hard-weight', type=float, default=0.5)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]

    print('Loading model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    maze_data = load_maze_nav()
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # Identify hard samples using existing checkpoints
    hard_train = None
    if args.strategy == 'weighted':
        ckpt_dir = RESULTS_DIR
        ckpts = []
        for f in os.listdir(ckpt_dir):
            if f.startswith('solver_bestofn_rank0') and f.endswith('.pt'):
                ckpts.append(os.path.join(ckpt_dir, f))
        ckpts = sorted(ckpts)[:3]  # Use top 3

        if ckpts:
            print(f'\nIdentifying hard samples using {len(ckpts)} checkpoints...', flush=True)
            hard_eval, medium_eval, easy_eval, _ = identify_hard_samples(
                base_model, lm_model, tokenizer, ckpts, maze_data, eval_idx)

            # Also identify hard TRAINING samples (run solvers on training data)
            hard_train_all, _, _, train_correctness = identify_hard_samples(
                base_model, lm_model, tokenizer, ckpts, maze_data, train_idx)
            hard_train = hard_train_all
            print(f'  Hard training samples: {len(hard_train)}/{len(train_idx)}', flush=True)
        else:
            print('No checkpoints found, using uniform sampling', flush=True)

    for seed in seeds:
        run_experiment(args.strategy, seed, args.steps, args.hard_weight,
                      tokenizer, base_model, lm_model, maze_data,
                      train_idx, eval_idx, hard_train)

    print('\n=== All hard maze experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
