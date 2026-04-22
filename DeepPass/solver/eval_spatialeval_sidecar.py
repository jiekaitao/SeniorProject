"""
Cross-Attention Sidecars — CALM-style memory injection at frozen decoder layers.

Instead of prepending memory tokens (prefix mode), inject solver output via
learned cross-attention at specific frozen decoder layers.

Advantages over prefix prepending:
  - No position encoding clash (prompt stays at natural positions, BOS preserved)
  - No softmax competition between memory and prompt tokens
  - Memory injected via dedicated channel (separate from self-attention)
  - Shorter effective gradient path (sidecars near the output)

Based on GPT-5.4 Pro's recommendation #5 (CALM-style cross-attention).

Each sidecar: Q from decoder hidden, KV from solver memory.
Bottleneck projection (d_llm -> d_bn -> d_llm) keeps params small.
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


class CrossAttentionSidecar(nn.Module):
    """Bottleneck cross-attention: decoder hidden -> memory."""
    def __init__(self, d_llm=4096, d_bottleneck=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_bottleneck // n_heads
        self.q_proj = nn.Linear(d_llm, d_bottleneck, bias=False)
        self.k_proj = nn.Linear(d_llm, d_bottleneck, bias=False)
        self.v_proj = nn.Linear(d_llm, d_bottleneck, bias=False)
        self.o_proj = nn.Linear(d_bottleneck, d_llm, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.01))
        # Init o_proj near zero for stable start
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, hidden, memory):
        B, T, D = hidden.shape
        M = memory.shape[1]
        q = self.q_proj(hidden).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(memory).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(memory).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.gate * self.o_proj(attn)


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def run_experiment(sidecar_layer_ids, d_bottleneck, seed, total_steps,
                   tokenizer, base_model, lm_model, maze_data, train_idx, eval_idx):
    layers_str = '_'.join(map(str, sidecar_layer_ids))
    tag = f'sidecar_L{layers_str}_d{d_bottleneck}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  Sidecar: layers={sidecar_layer_ids}, d_bn={d_bottleneck}, seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)

    # Create sidecars for specified layers
    sidecars = nn.ModuleDict({
        str(i): CrossAttentionSidecar(d_llm=4096, d_bottleneck=d_bottleneck, n_heads=4)
        for i in sidecar_layer_ids
    }).to(device=device, dtype=torch.bfloat16)

    n_solver = sum(p.numel() for p in solver.parameters())
    n_sidecar = sum(p.numel() for p in sidecars.parameters())
    print(f'  Solver: {n_solver:,} | Sidecars: {n_sidecar:,} | Total: {n_solver+n_sidecar:,}', flush=True)

    # Separate LR: higher for sidecars (starting from scratch)
    optimizer = torch.optim.AdamW([
        {'params': solver.parameters(), 'lr': 1e-4},
        {'params': sidecars.parameters(), 'lr': 3e-4},
    ], weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    sidecar_set = set(sidecar_layer_ids)
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
        prompt_emb = all_emb[:, :prompt_len]

        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
        memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)

        # Decoder forward WITHOUT memory prefix — sidecars inject memory
        h = all_emb  # (B, T, 4096) — prompt at natural positions
        T_seq = h.shape[1]
        pos_ids = torch.arange(T_seq, device=device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(h, pos_ids)

        for i, layer in enumerate(lm_model.layers):
            h = layer(h, position_embeddings=pos_emb)
            if i in sidecar_set:
                h = h + sidecars[str(i)](h, memory)

        h = lm_model.norm(h)
        logits = base_model.lm_head(h)

        # Loss on answer tokens (NO memory prefix offset)
        ans_logits = logits[:, prompt_len - 1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:, :ml].reshape(-1),
                                   ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                all_params = list(solver.parameters()) + list(sidecars.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        if (step + 1) % 200 == 0:
            avg = sum(losses[-200:]) / len(losses[-200:])
            # Check gate values
            gates = {k: sidecars[k].gate.item() for k in sidecars}
            print(f'  step {step+1} | loss={avg:.4f} | gates={gates} | {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    sidecars.eval()
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

                    h = emb
                    T_seq = h.shape[1]
                    pid = torch.arange(T_seq, device=device).unsqueeze(0)
                    pe = lm_model.rotary_emb(h, pid)
                    for i, layer in enumerate(lm_model.layers):
                        h = layer(h, position_embeddings=pe)
                        if i in sidecar_set:
                            h = h + sidecars[str(i)](h, mem)
                    h = lm_model.norm(h)
                    lg = base_model.lm_head(h)
                    answer = tokenizer.decode([lg[0, -1].argmax().item()]).strip()

            if oracle in answer.upper()[:10]:
                correct += 1

        acc = correct / n_eval
        results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gate_vals = {k: sidecars[k].gate.item() for k in sidecars}
    result_data = {
        'tag': tag, 'method': 'sidecar',
        'sidecar_layers': sidecar_layer_ids, 'd_bottleneck': d_bottleneck,
        'seed': seed, 'total_steps': total_steps,
        'solver_params': n_solver, 'sidecar_params': n_sidecar,
        'final_gates': gate_vals,
        'final_loss': sum(losses[-50:]) / max(len(losses[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del solver, sidecars, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sidecar-layers', type=str, default='4,8,12,16',
                        help='Comma-separated decoder layer indices for sidecars')
    parser.add_argument('--d-bottleneck', type=int, default=256)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    layer_configs = [list(map(int, args.sidecar_layers.split(',')))]
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

    for layers in layer_configs:
        for seed in seeds:
            run_experiment(layers, args.d_bottleneck, seed, args.steps,
                          tokenizer, base_model, lm_model, maze_data, train_idx, eval_idx)

    print('\n=== All sidecar experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
