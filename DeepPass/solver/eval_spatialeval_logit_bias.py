"""
Logit Bias — Solver directly biases answer token logits.

Instead of producing memory tokens for the decoder to attend to, the solver
produces a 4-dim bias added to the frozen decoder's A/B/C/D logits.

Key advantages:
  - Very short gradient path: loss -> bias_head -> z_H -> solver
  - Frozen decoder runs with no_grad (fast, no autograd overhead)
  - Tests solver capability in isolation (can it solve the task?)

If logit_bias > 72%, solver CAN break the ceiling — the decoder interface was
the bottleneck. If logit_bias ~ 72%, solver itself is limited.

Also supports hybrid mode: memory tokens (decoder loss) + logit bias (aux loss).
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class LogitBiasHead(nn.Module):
    """Attention-pooled z_H -> 4-dim bias for A/B/C/D."""
    def __init__(self, d_model=512, n_choices=4):
        super().__init__()
        self.pool_w = nn.Linear(d_model, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, n_choices),
        )

    def forward(self, z_H):
        weights = F.softmax(self.pool_w(z_H).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * z_H).sum(dim=1)
        return self.head(pooled)  # (B, 4)


def solver_forward_with_z_H(solver, prompt_emb, K_inner=4, K_outer=1, grad_last_only=True):
    """Modified solver forward returning (memory, z_H_before_projection)."""
    B, T, _ = prompt_emb.shape
    e = solver.proj_in(prompt_emb)
    z_L = solver.L_init_scale * e
    z_H = solver.H_init.expand(B, -1, -1).clone()

    for s in range(K_outer):
        use_grad = (not grad_last_only) or (s == K_outer - 1)
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            for _ in range(K_inner):
                z_L_input = z_L + e
                z_L_input = z_L_input + solver.L_cross_H(z_L_input, z_H)
                for layer in solver.L_self:
                    z_L_input = layer(z_L_input)
                z_L = z_L_input
            z_H = z_H + solver.H_cross_L(z_H, z_L)
            z_H = solver.H_self(z_H)
        if grad_last_only and s < K_outer - 1:
            z_L = z_L.detach()
            z_H = z_H.detach()

    memory = solver.out_norm(solver.proj_out(z_H))
    return memory, z_H


def get_choice_token_ids(tokenizer):
    """Get token IDs for ' A', ' B', ' C', ' D' (with leading space)."""
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def run_experiment(mode, temperature, lambda_bias, seed, total_steps,
                   tokenizer, base_model, lm_model, maze_data, train_idx, eval_idx,
                   choice_token_ids):
    tag = f'logit_{mode}_t{temperature}_s{seed}'
    if mode == 'hybrid':
        tag = f'logit_hybrid_t{temperature}_lam{lambda_bias}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  Logit Bias: mode={mode}, temp={temperature}, seed={seed}', flush=True)
    if mode == 'hybrid':
        print(f'  lambda_bias={lambda_bias}', flush=True)
    print(f'{"="*60}', flush=True)

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
    bias_head = LogitBiasHead(d_model=512, n_choices=4).to(device=device, dtype=torch.bfloat16)

    all_params = list(solver.parameters()) + list(bias_head.parameters())
    n_solver = sum(p.numel() for p in solver.parameters())
    n_bias = sum(p.numel() for p in bias_head.parameters())
    print(f'  Solver: {n_solver:,} | Bias head: {n_bias:,}', flush=True)

    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids_tensor = torch.tensor(choice_token_ids, device=device)
    t0 = time.time()
    bias_losses, dec_losses = [], []

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)
        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        prompt_text = text + "\nAnswer:"
        enc_prompt = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=2048).to(device)

        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]

        # Get base model logits at answer position (no grad — fast)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(enc_prompt['input_ids'])
            base_out = base_model(enc_prompt['input_ids'])
            base_logits_answer = base_out.logits[0, -1, choice_ids_tensor].float()  # (4,)

        # Solver forward (with grad)
        _, z_H = solver_forward_with_z_H(solver, prompt_emb, K_inner=4, K_outer=K)
        bias = bias_head(z_H)  # (1, 4)

        # Biased logits
        biased_logits = base_logits_answer.unsqueeze(0) + temperature * bias  # (1, 4)
        bias_loss = F.cross_entropy(biased_logits, label_tensor)

        if mode == 'pure':
            total_loss = bias_loss
            dec_losses.append(0.0)
        else:
            # Hybrid: also compute decoder loss with memory tokens
            answer_text = f" {oracle}"
            full = prompt_text + answer_text
            enc_full = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048, padding=True).to(device)
            input_ids = enc_full['input_ids']
            prompt_len = len(tokenizer.encode(prompt_text))

            with torch.no_grad():
                all_emb = lm_model.embed_tokens(input_ids)

            memory = solver.out_norm(solver.proj_out(z_H))  # Reuse z_H
            dec_in = torch.cat([memory, all_emb], dim=1)
            M = memory.shape[1]
            T_dec = dec_in.shape[1]
            pos_ids = torch.arange(T_dec, device=device).unsqueeze(0)
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
            dec_loss = torch.tensor(0.0, device=device)
            if ml > 0:
                dec_loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                           ans_labels[:, :ml].reshape(-1),
                                           ignore_index=tokenizer.pad_token_id)
            total_loss = dec_loss + lambda_bias * bias_loss
            dec_losses.append(dec_loss.item())

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        bias_losses.append(bias_loss.item())

        if (step + 1) % 200 == 0:
            avg_bias = sum(bias_losses[-200:]) / len(bias_losses[-200:])
            avg_dec = sum(dec_losses[-200:]) / max(len(dec_losses[-200:]), 1)
            print(f'  step {step+1} | bias_loss={avg_bias:.4f} | dec_loss={avg_dec:.4f} | {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    bias_head.eval()
    results = {}

    for K_eval in [0, 1, 2]:
        bias_correct, dec_correct = 0, 0
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
                if oracle in answer.upper()[:10]:
                    bias_correct += 1
                    dec_correct += 1
            else:
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    emb = lm_model.embed_tokens(enc['input_ids'])
                    _, z_H = solver_forward_with_z_H(solver, emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                    bias = bias_head(z_H)

                    # Biased prediction
                    base_out = base_model(enc['input_ids'])
                    base_logits = base_out.logits[0, -1, choice_ids_tensor].float()
                    biased = base_logits + temperature * bias[0]
                    pred_choice = biased.argmax().item()
                    bias_answer = ['A', 'B', 'C', 'D'][pred_choice]

                    # Also decoder prediction (with memory prefix, for comparison)
                    memory = solver.out_norm(solver.proj_out(z_H))
                    dec = torch.cat([memory, emb], dim=1)
                    T_dec = dec.shape[1]
                    pid = torch.arange(T_dec, device=device).unsqueeze(0)
                    pe = lm_model.rotary_emb(dec, pid)
                    h = dec
                    for layer in lm_model.layers:
                        h = layer(h, position_embeddings=pe)
                    h = lm_model.norm(h)
                    lg = base_model.lm_head(h)
                    dec_answer = tokenizer.decode([lg[0, -1].argmax().item()]).strip()

                if oracle[0] == bias_answer:
                    bias_correct += 1
                if oracle in dec_answer.upper()[:10]:
                    dec_correct += 1

        bias_acc = bias_correct / n_eval
        dec_acc = dec_correct / n_eval
        results[f'K={K_eval}'] = {
            'bias_accuracy': bias_acc, 'bias_correct': bias_correct,
            'decoder_accuracy': dec_acc, 'decoder_correct': dec_correct,
            'total': n_eval,
        }
        if K_eval == 0:
            print(f'  K=0 (baseline): {bias_acc:.4f}', flush=True)
        else:
            print(f'  K={K_eval} bias: {bias_acc:.4f} | decoder: {dec_acc:.4f}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': f'logit_bias_{mode}',
        'mode': mode, 'temperature': temperature,
        'lambda_bias': lambda_bias if mode == 'hybrid' else None,
        'seed': seed, 'total_steps': total_steps,
        'solver_params': n_solver, 'bias_params': n_bias,
        'final_bias_loss': sum(bias_losses[-50:]) / max(len(bias_losses[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del solver, bias_head, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='pure', choices=['pure', 'hybrid'])
    parser.add_argument('--temperatures', type=str, default='0.5,1.0,2.0')
    parser.add_argument('--lambda-bias', type=float, default=0.5)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()

    temps = [float(x) for x in args.temperatures.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    print('Loading tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('Loading Llama 3.1 8B...', flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model
    print('Model loaded.', flush=True)

    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f'Choice token IDs: {dict(zip(["A","B","C","D"], choice_token_ids))}', flush=True)

    maze_data = load_maze_nav()
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    for temp in temps:
        for seed in seeds:
            run_experiment(args.mode, temp, args.lambda_bias, seed, args.steps,
                          tokenizer, base_model, lm_model, maze_data, train_idx, eval_idx,
                          choice_token_ids)

    print('\n=== All logit bias experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
