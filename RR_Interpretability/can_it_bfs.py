"""
Can the Solver Rediscover BFS?

Direct test: train solver on text-encoded grid reachability,
then measure EXACT cell-level accuracy (not just PPL).
Does accuracy approach 100%? Does it improve with more K cycles?

Also test TRM with a non-linear (MLP) probe — maybe reachability
IS encoded but in a non-linear way our linear probe missed.
"""
import os, sys, torch, json, random, math, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

# ===== PART 1: TRM Non-Linear Probe =====

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)
from utils.functions import load_model_class
import yaml


class MLPProbe(nn.Module):
    """2-layer MLP probe — can detect non-linear features the linear probe missed."""
    def __init__(self, input_dim, hidden_dim=256, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def trm_nonlinear_probe():
    """Test if TRM z_H encodes reachability non-linearly."""
    print(f'\n{"="*60}', flush=True)
    print(f'  PART 1: TRM Non-Linear (MLP) Probe', flush=True)
    print(f'{"="*60}', flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for ckpt_name in ['ABLATION_BASELINE', 'ABLATION_FULL_COMBO']:
        print(f'\n--- {ckpt_name} ---', flush=True)

        ckpt_dir = os.path.join(TRM_ROOT, 'checkpoints/SeniorProjectTRM', ckpt_name)
        with open(os.path.join(ckpt_dir, 'all_config.yaml')) as f:
            cfg = yaml.safe_load(f)

        ac = cfg['arch']
        dp = os.path.join(TRM_ROOT, cfg.get('data_paths', ['data/maze-30x30-hard-1k'])[0])
        inputs = np.load(os.path.join(dp, 'train/all__inputs.npy'))
        labels = np.load(os.path.join(dp, 'train/all__labels.npy'))
        pids = np.load(os.path.join(dp, 'train/all__puzzle_identifiers.npy'))

        config = dict(ac)
        config['seq_len'] = inputs.shape[1]
        config['vocab_size'] = int(inputs.max()) + 1
        config['num_puzzle_identifiers'] = int(pids.max()) + 1
        config['batch_size'] = 16

        model = load_model_class(ac['name'])(config)
        ckf = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')],
                     key=lambda x: int(x.split('_')[1]))
        sd = torch.load(os.path.join(ckpt_dir, ckf[-1]), map_location=device)
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()

        inner = model.inner
        H_cycles = config.get('H_cycles', 3)
        L_cycles = config.get('L_cycles', 6)
        pel = getattr(inner, 'puzzle_emb_len', 16)
        tl = config['seq_len'] + pel
        hidden_size = config.get('hidden_size', 512)

        # Split data: 300 train, 100 test
        np.random.seed(42)
        idx = np.random.permutation(len(inputs))
        train_idx, test_idx = idx[:300], idx[300:400]

        # Extract z_H at each H-cycle
        def extract(data_idx, bs=16):
            z_H_all = {h: [] for h in range(H_cycles)}
            inp = torch.tensor(inputs[data_idx], dtype=torch.long, device=device)
            pid_t = torch.tensor(pids[data_idx], dtype=torch.long, device=device)
            for start in range(0, len(data_idx), bs):
                end = min(start + bs, len(data_idx))
                bi, bp = inp[start:end], pid_t[start:end]
                B = bi.shape[0]
                with torch.no_grad():
                    ie = inner._input_embeddings(bi, bp)
                    cs = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
                    si = {'cos_sin': cs}
                    zH = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
                    zL = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
                    for h in range(H_cycles):
                        for l in range(L_cycles):
                            zL = inner.L_level(zL, zH + ie, **si)
                        zH = inner.L_level(zH, zL, **si)
                        z_H_all[h].append(zH[:, pel:].float().cpu())
            for h in range(H_cycles):
                z_H_all[h] = torch.cat(z_H_all[h], dim=0)
            return z_H_all

        print(f'  Extracting representations...', flush=True)
        z_train = extract(train_idx)
        z_test = extract(test_idx)
        lbl_train = torch.tensor(labels[train_idx], dtype=torch.long)
        lbl_test = torch.tensor(labels[test_idx], dtype=torch.long)

        ckpt_results = {}
        for h in range(H_cycles):
            # Train MLP probe
            probe = MLPProbe(hidden_size, hidden_dim=256, n_classes=6)
            optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

            X_train = z_train[h].reshape(-1, hidden_size)
            y_train = lbl_train.reshape(-1)
            X_test = z_test[h].reshape(-1, hidden_size)
            y_test = lbl_test.reshape(-1)

            # Subsample training
            if X_train.shape[0] > 200000:
                si = torch.randperm(X_train.shape[0])[:200000]
                X_train, y_train = X_train[si], y_train[si]

            # Train
            for epoch in range(30):
                perm = torch.randperm(X_train.shape[0])
                for i in range(0, X_train.shape[0], 4096):
                    bi = perm[i:i+4096]
                    logits = probe(X_train[bi])
                    loss = F.cross_entropy(logits, y_train[bi])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Eval
            probe.eval()
            with torch.no_grad():
                test_logits = probe(X_test)
                preds = test_logits.argmax(dim=-1)
                overall = (preds == y_test).float().mean().item()
                per_label = {}
                for lv in range(6):
                    mask = (y_test == lv)
                    if mask.sum() > 0:
                        per_label[lv] = (preds[mask] == y_test[mask]).float().mean().item()

            print(f'  H-cycle {h}: MLP probe acc = {overall:.4f}', flush=True)
            print(f'    Per-label: {per_label}', flush=True)
            print(f'    REACHABLE (label=5): {per_label.get(5, "N/A")}', flush=True)
            ckpt_results[f'h_cycle_{h}'] = {
                'mlp_probe_acc': overall,
                'per_label': {str(k): v for k, v in per_label.items()},
            }

        results[ckpt_name] = ckpt_results
        del model; torch.cuda.empty_cache()

    return results


# ===== PART 2: Solver Accuracy on Reachability =====

def solver_reachability_accuracy():
    """
    Train solver on text-encoded grid reachability for 2000 steps,
    then measure EXACT cell-level accuracy for each K value.
    """
    print(f'\n{"="*60}', flush=True)
    print(f'  PART 2: Solver Reachability — Can It Hit 100%?', flush=True)
    print(f'{"="*60}', flush=True)

    sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/solver')
    from model import SolverCore

    device = torch.device('cuda')
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)

    N = 6  # Start with 6x6 (easier, faster to train)

    def make_grid_batch(bs=4, n=6):
        prompts, answers, grids, reaches = [], [], [], []
        for _ in range(bs):
            grid = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if random.random() < 0.25 and (i, j) != (0, 0):
                        grid[i][j] = 1
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
            grids.append(grid)
            reaches.append(reach)
        return prompts, answers, grids, reaches

    print(f'  Training solver on {N}x{N} grids for 3000 steps...', flush=True)
    t0 = time.time()

    for step in range(3000):
        prompts, answers, _, _ = make_grid_batch(bs=4, n=N)
        full_texts = [p + a for p, a in zip(prompts, answers)]
        enc = tokenizer(full_texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=512).to(device)
        input_ids = enc['input_ids']
        prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
        avg_pl = sum(prompt_lens) // len(prompt_lens)

        with torch.no_grad():
            all_emb = lm_model.embed_tokens(input_ids)

        prompt_emb = all_emb[:, :avg_pl]
        K = random.choices([1, 2, 4, 8], weights=[0.1, 0.3, 0.3, 0.3])[0]
        memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)

        stub = all_emb[:, avg_pl-16:avg_pl]
        ans_emb = all_emb[:, avg_pl:]
        dec_in = torch.cat([memory, stub, ans_emb], dim=1)
        M = memory.shape[1]
        T = dec_in.shape[1]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_in, pos_ids)
        h = dec_in
        for layer in lm_model.layers:
            h = layer(h, position_embeddings=pos_emb)
        h = lm_model.norm(h)
        logits = base_model.lm_head(h)

        ans_start = M + 16
        ans_logits = logits[:, ans_start-1:-1]
        ans_labels = input_ids[:, avg_pl:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:,:ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:,:ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step+1) % 500 == 0:
            print(f'  step {step+1} | loss={loss.item():.4f} | {time.time()-t0:.0f}s', flush=True)

    # === EVALUATION: Measure exact cell-level accuracy ===
    print(f'\n  === Cell-Level Accuracy Evaluation ===', flush=True)
    solver.eval()
    results = {}

    for K_eval in [1, 2, 4, 8]:
        correct_cells = 0
        total_cells = 0
        correct_reach = 0
        total_reach = 0
        correct_unreach = 0
        total_unreach = 0
        n_perfect = 0
        n_eval = 50

        for _ in range(n_eval):
            prompts, answers, grids, reaches = make_grid_batch(bs=1, n=N)
            full = [prompts[0] + answers[0]]
            enc = tokenizer(full, return_tensors='pt', padding=True,
                           truncation=True, max_length=512).to(device)
            prompt_len = len(tokenizer.encode(prompts[0]))

            with torch.no_grad():
                emb = lm_model.embed_tokens(enc['input_ids'])
                pe = emb[:, :prompt_len]
                mem = solver(pe, K_inner=4, K_outer=K_eval, grad_last_only=False)
                stub = emb[:, prompt_len-16:prompt_len]
                ans_emb = emb[:, prompt_len:]
                di = torch.cat([mem, stub, ans_emb], dim=1)
                M = mem.shape[1]
                pi = torch.arange(di.shape[1], device=device).unsqueeze(0)
                pe2 = lm_model.rotary_emb(di, pi)
                h = di
                for layer in lm_model.layers:
                    h = layer(h, position_embeddings=pe2)
                h = lm_model.norm(h)
                lg = base_model.lm_head(h)

                # Decode answer tokens
                ans_start = M + 16
                pred_ids = lg[:, ans_start-1:-1].argmax(dim=-1)[0]
                true_ids = enc['input_ids'][0, prompt_len:]

                # Decode to text and compare
                pred_text = tokenizer.decode(pred_ids[:len(true_ids)])
                true_text = answers[0]

                # Parse reachability predictions
                reach = reaches[0]
                # Count cell-level accuracy by parsing the answer string
                true_cells = []
                for row in reach:
                    true_cells.extend(row)

                # Try to parse predicted cells
                pred_cells = []
                for ch in pred_text:
                    if ch == '0':
                        pred_cells.append(0)
                    elif ch == '1':
                        pred_cells.append(1)

                n_cells = min(len(pred_cells), len(true_cells))
                for i in range(n_cells):
                    total_cells += 1
                    if pred_cells[i] == true_cells[i]:
                        correct_cells += 1
                    if true_cells[i] == 1:
                        total_reach += 1
                        if pred_cells[i] == 1:
                            correct_reach += 1
                    else:
                        total_unreach += 1
                        if pred_cells[i] == 0:
                            correct_unreach += 1

                if n_cells == N*N and all(pred_cells[i] == true_cells[i] for i in range(n_cells)):
                    n_perfect += 1

        cell_acc = correct_cells / max(total_cells, 1)
        reach_acc = correct_reach / max(total_reach, 1)
        unreach_acc = correct_unreach / max(total_unreach, 1)
        perfect_rate = n_perfect / n_eval

        print(f'  K={K_eval}: cell_acc={cell_acc:.4f} reach_acc={reach_acc:.4f} '
              f'unreach_acc={unreach_acc:.4f} perfect={perfect_rate:.2%} ({n_perfect}/{n_eval})',
              flush=True)
        results[K_eval] = {
            'cell_accuracy': cell_acc,
            'reachable_accuracy': reach_acc,
            'unreachable_accuracy': unreach_acc,
            'perfect_grid_rate': perfect_rate,
        }

    return results


def main():
    all_results = {}

    # Part 1: TRM non-linear probe
    trm_results = trm_nonlinear_probe()
    all_results['trm_mlp_probe'] = trm_results

    # Part 2: Solver reachability accuracy
    solver_results = solver_reachability_accuracy()
    all_results['solver_reachability'] = {str(k): v for k, v in solver_results.items()}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/can_it_bfs_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
