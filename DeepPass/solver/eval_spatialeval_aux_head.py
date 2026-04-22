"""
Auxiliary Choice Head — Force z_H to explicitly encode the answer.

z_H probe showed 33.7% (random chance): solver works through implicit attention
steering, never explicitly encoding the answer. This adds a direct classification
loss on z_H -> A/B/C/D to force explicit encoding.

If aux_head accuracy >> decoder accuracy: confirms decoder readout bottleneck.
If aux_head ~ decoder ~ 72%: solver itself is limited.

Combined loss: L_decoder + lambda_aux * L_aux
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


class AuxChoiceHead(nn.Module):
    """Attention-pooled classification head on z_H (solver space)."""
    def __init__(self, d_model=512, n_choices=4):
        super().__init__()
        self.pool_w = nn.Linear(d_model, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_choices),
        )

    def forward(self, z_H):
        # z_H: (B, M, d_model)
        weights = F.softmax(self.pool_w(z_H).squeeze(-1), dim=-1)  # (B, M)
        pooled = (weights.unsqueeze(-1) * z_H).sum(dim=1)  # (B, d_model)
        return self.head(pooled)  # (B, n_choices)


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


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def run_experiment(lambda_aux, seed, total_steps, tokenizer, base_model, lm_model,
                   maze_data, train_idx, eval_idx):
    tag = f'aux_head_lam{lambda_aux}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  Aux Head: lambda={lambda_aux}, seed={seed}, steps={total_steps}', flush=True)
    print(f'{"="*60}', flush=True)

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
    aux_head = AuxChoiceHead(d_model=512, n_choices=4).to(device=device, dtype=torch.bfloat16)

    all_params = list(solver.parameters()) + list(aux_head.parameters())
    n_solver = sum(p.numel() for p in solver.parameters())
    n_aux = sum(p.numel() for p in aux_head.parameters())
    print(f'  Solver: {n_solver:,} | Aux head: {n_aux:,} | Total: {n_solver+n_aux:,}', flush=True)

    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    dec_losses, aux_losses = [], []

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_text = f" {oracle}"
        answer_label = CHOICE_MAP.get(oracle[0], 0)

        full = text + "\nAnswer:" + answer_text
        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048, padding=True).to(device)
        input_ids = enc['input_ids']
        prompt_text = text + "\nAnswer:"
        prompt_len = len(tokenizer.encode(prompt_text))

        with torch.no_grad():
            all_emb = lm_model.embed_tokens(input_ids)
        prompt_emb = all_emb[:, :prompt_len]

        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
        memory, z_H = solver_forward_with_z_H(solver, prompt_emb, K_inner=4, K_outer=K)

        # --- Decoder loss (same as baseline) ---
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

        dec_loss = torch.tensor(0.0, device=device)
        if ml > 0:
            dec_loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                       ans_labels[:, :ml].reshape(-1),
                                       ignore_index=tokenizer.pad_token_id)

        # --- Aux loss ---
        choice_logits = aux_head(z_H)
        aux_loss = F.cross_entropy(choice_logits,
                                   torch.tensor([answer_label], device=device, dtype=torch.long))

        # --- Combined ---
        total_loss = dec_loss + lambda_aux * aux_loss
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        dec_losses.append(dec_loss.item())
        aux_losses.append(aux_loss.item())

        if (step + 1) % 200 == 0:
            avg_dec = sum(dec_losses[-200:]) / len(dec_losses[-200:])
            avg_aux = sum(aux_losses[-200:]) / len(aux_losses[-200:])
            print(f'  step {step+1} | dec_loss={avg_dec:.4f} | aux_loss={avg_aux:.4f} | {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    aux_head.eval()
    results = {}

    for K_eval in [0, 1, 2]:
        dec_correct, aux_correct = 0, 0
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
                    dec_correct += 1
                aux_correct = dec_correct  # no aux at K=0
            else:
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    emb = lm_model.embed_tokens(enc['input_ids'])
                    memory, z_H = solver_forward_with_z_H(solver, emb, K_inner=4, K_outer=K_eval, grad_last_only=False)

                    # Decoder prediction
                    dec = torch.cat([memory, emb], dim=1)
                    T = dec.shape[1]
                    pid = torch.arange(T, device=device).unsqueeze(0)
                    pe = lm_model.rotary_emb(dec, pid)
                    h = dec
                    for layer in lm_model.layers:
                        h = layer(h, position_embeddings=pe)
                    h = lm_model.norm(h)
                    lg = base_model.lm_head(h)
                    dec_answer = tokenizer.decode([lg[0, -1].argmax().item()]).strip()

                    # Aux head prediction
                    choice_logits = aux_head(z_H)
                    aux_pred = choice_logits.argmax(dim=-1).item()
                    aux_answer = ['A', 'B', 'C', 'D'][aux_pred]

                if oracle in dec_answer.upper()[:10]:
                    dec_correct += 1
                if oracle[0] == aux_answer:
                    aux_correct += 1

        dec_acc = dec_correct / n_eval
        aux_acc = aux_correct / n_eval
        results[f'K={K_eval}'] = {
            'decoder_accuracy': dec_acc, 'decoder_correct': dec_correct,
            'aux_accuracy': aux_acc, 'aux_correct': aux_correct,
            'total': n_eval,
        }
        if K_eval == 0:
            print(f'  K=0 (baseline): {dec_acc:.4f} ({dec_correct}/{n_eval})', flush=True)
        else:
            print(f'  K={K_eval} decoder: {dec_acc:.4f} | aux_head: {aux_acc:.4f}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'aux_head', 'lambda_aux': lambda_aux,
        'seed': seed, 'total_steps': total_steps,
        'solver_params': n_solver, 'aux_params': n_aux,
        'final_dec_loss': sum(dec_losses[-50:]) / max(len(dec_losses[-50:]), 1),
        'final_aux_loss': sum(aux_losses[-50:]) / max(len(aux_losses[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del solver, aux_head, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambdas', type=str, default='0.1,0.5,1.0,2.0')
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

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

    for lam in lambdas:
        for seed in seeds:
            run_experiment(lam, seed, args.steps, tokenizer, base_model, lm_model,
                          maze_data, train_idx, eval_idx)

    print('\n=== All aux head experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
