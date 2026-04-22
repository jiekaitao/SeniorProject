"""
Embedding Adapter — Transform frozen embeddings before the solver.

Root cause hypothesis: frozen LLM embeddings represent text, not spatial structure.
The solver's proj_in (4096→512) loses spatial info in the compression.

Fix: add a trainable adapter in 4096-dim space BEFORE the solver, giving it
richer input. Two modes:
  A. solver_only: adapter transforms embeddings for solver only, decoder sees originals
  B. shared: adapter transforms embeddings for both solver AND decoder

The adapter is a small residual MLP: emb + alpha * MLP(emb)
alpha initialized to 0.01 for stable start.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


class EmbeddingAdapter(nn.Module):
    """Residual adapter in LLM embedding space."""
    def __init__(self, d_model=4096, d_hidden=1024):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        h = self.norm(x)
        return x + self.gate * self.down(F.gelu(self.up(h)))


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def run_experiment(mode, d_hidden, seed, total_steps, tokenizer, base_model,
                   lm_model, maze_data, train_idx, eval_idx):
    tag = f'adapter_{mode}_dh{d_hidden}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  Adapter: mode={mode}, d_hidden={d_hidden}, seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
    adapter = EmbeddingAdapter(d_model=4096, d_hidden=d_hidden).to(device=device, dtype=torch.bfloat16)

    n_solver = sum(p.numel() for p in solver.parameters())
    n_adapter = sum(p.numel() for p in adapter.parameters())
    print(f'  Solver: {n_solver:,} | Adapter: {n_adapter:,} | Total: {n_solver+n_adapter:,}', flush=True)

    all_params = list(solver.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses = []

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

        # Apply adapter
        if mode == 'solver_only':
            # Adapter only for solver input, decoder sees original
            adapted_prompt = adapter(all_emb[:, :prompt_len])
            memory = solver(adapted_prompt, K_inner=4, K_outer=random.choices([1,2,4], weights=[0.2,0.4,0.4])[0],
                           grad_last_only=True)
            # Decoder: [memory | ORIGINAL embeddings]
            dec_in = torch.cat([memory, all_emb], dim=1)
        elif mode == 'shared':
            # Adapter for both solver and decoder
            adapted = adapter(all_emb)
            adapted_prompt = adapted[:, :prompt_len]
            memory = solver(adapted_prompt, K_inner=4, K_outer=random.choices([1,2,4], weights=[0.2,0.4,0.4])[0],
                           grad_last_only=True)
            # Decoder: [memory | ADAPTED embeddings]
            dec_in = torch.cat([memory, adapted], dim=1)
        else:  # baseline (no adapter, for comparison)
            memory = solver(all_emb[:, :prompt_len], K_inner=4,
                           K_outer=random.choices([1,2,4], weights=[0.2,0.4,0.4])[0],
                           grad_last_only=True)
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
            loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:, :ml].reshape(-1),
                                   ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        if (step + 1) % 200 == 0:
            avg_loss = sum(losses[-200:]) / len(losses[-200:])
            gate_val = adapter.gate.item()
            print(f'  step {step+1} | loss={avg_loss:.4f} | adapter_gate={gate_val:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    adapter.eval()
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
                    if mode == 'solver_only':
                        adapted_p = adapter(emb)
                        mem = solver(adapted_p, K_inner=4, K_outer=K_eval, grad_last_only=False)
                        dec = torch.cat([mem, emb], dim=1)
                    elif mode == 'shared':
                        adapted = adapter(emb)
                        mem = solver(adapted, K_inner=4, K_outer=K_eval, grad_last_only=False)
                        dec = torch.cat([mem, adapted], dim=1)
                    else:
                        mem = solver(emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                        dec = torch.cat([mem, emb], dim=1)
                    T_d = dec.shape[1]
                    pid = torch.arange(T_d, device=device).unsqueeze(0)
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
        'tag': tag, 'method': f'adapter_{mode}',
        'mode': mode, 'd_hidden': d_hidden,
        'seed': seed, 'total_steps': total_steps,
        'solver_params': n_solver, 'adapter_params': n_adapter,
        'final_gate': adapter.gate.item(),
        'final_loss': sum(losses[-50:]) / max(len(losses[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del solver, adapter, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='solver_only', choices=['solver_only', 'shared'])
    parser.add_argument('--d-hidden', type=str, default='1024', help='Adapter hidden dims, comma-sep')
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    d_hiddens = [int(x) for x in args.d_hidden.split(',')]
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

    for dh in d_hiddens:
        for seed in seeds:
            run_experiment(args.mode, dh, seed, args.steps,
                          tokenizer, base_model, lm_model, maze_data, train_idx, eval_idx)

    print('\n=== All adapter experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
