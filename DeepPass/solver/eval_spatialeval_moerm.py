"""
MoERM-Lite Training on SpatialEval — Mixture of External Reasoning Modules.

Tests whether routing between multiple specialists can break the 72% ceiling.
Key insight from ensemble: all 5 same-architecture solvers fail on the same mazes.
MoERM may help IF experts genuinely specialize (different reasoning strategies).

Configurations:
  1. Full MoERM: 4 separate SolverCore experts (~50M params)
  2. Shared-core: 1 SolverCore + 4 expert-specific inits (~16M params)
  3. Heterogeneous K: experts use K=1,2,4,8 (different reasoning depths)
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from moerm_lite import (MoERMLite, router_regularizers, diversity_loss,
                        compute_specialization_metrics)

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def moerm_forward_hetero_K(moerm, prompt_emb, K_list, K_inner=4, grad_last_only=True):
    """Forward with different K_outer per expert (heterogeneous depth)."""
    gate, gate_logits = moerm.router(prompt_emb)
    expert_mems = []

    if moerm.shared_core:
        for i in range(moerm.n_experts):
            mem_i = moerm.experts(prompt_emb, expert_idx=i,
                                 K_inner=K_inner, K_outer=K_list[i],
                                 grad_last_only=grad_last_only)
            expert_mems.append(mem_i)
    else:
        for i, expert in enumerate(moerm.experts):
            mem_i = expert(prompt_emb, K_inner=K_inner, K_outer=K_list[i],
                          grad_last_only=grad_last_only)
            expert_mems.append(mem_i)

    memory = moerm.fusion(expert_mems, gate)
    memory = moerm.out_norm(memory)
    return memory, gate, gate_logits, expert_mems


def run_experiment(config, seed, total_steps, tokenizer, base_model, lm_model,
                   maze_data, train_idx, eval_idx):
    shared_core = config.get('shared_core', False)
    n_experts = config.get('n_experts', 4)
    hetero_K = config.get('hetero_K', False)
    K_list = config.get('K_list', [1, 2, 4, 8])
    lambda_lb = config.get('lambda_lb', 1e-3)
    lambda_div = config.get('lambda_div', 1e-4)

    mode = 'shared' if shared_core else 'full'
    tag = f'moerm_{mode}_{n_experts}exp'
    if hetero_K:
        tag += '_hetK'
    tag += f'_s{seed}'

    random.seed(seed)
    torch.manual_seed(seed)
    print(f'\n{"="*60}', flush=True)
    print(f'  MoERM: {mode}, {n_experts} experts, hetero_K={hetero_K}, seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    moerm = MoERMLite(
        n_experts=n_experts, d_solver=512, n_heads=8, ffn_dim=1024,
        n_L_layers=2, n_memory_slots_per_expert=32, n_output_slots=32,
        llm_dim=4096, shared_core=shared_core
    ).to(device=device, dtype=torch.bfloat16)

    total_p, train_p = moerm.count_params()
    print(f'  Params: {total_p:,} total, {train_p:,} trainable ({train_p/1e6:.1f}M)', flush=True)

    optimizer = torch.optim.AdamW(moerm.parameters(), lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses, gate_history = [], []

    for step in range(total_steps):
        # Anneal router temperature
        progress = step / total_steps
        moerm.router.temperature = 2.0 - progress * 1.0  # 2.0 → 1.0

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

        if hetero_K:
            memory, gate, gate_logits, expert_mems = moerm_forward_hetero_K(
                moerm, prompt_emb, K_list, K_inner=4, grad_last_only=True)
        else:
            K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
            memory, gate, gate_logits = moerm(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)
            # Get expert_mems for diversity loss (re-extract from last forward)
            expert_mems = None  # skip diversity loss if not hetero (too expensive to re-compute)

        # Decoder forward
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

        # Decoder CE loss
        ans_start = M + prompt_len
        ans_logits = logits[:, ans_start - 1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        ce_loss = torch.tensor(0.0, device=device)
        if ml > 0:
            ce_loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                      ans_labels[:, :ml].reshape(-1),
                                      ignore_index=tokenizer.pad_token_id)

        # Router regularization
        lb, ent = router_regularizers(gate)
        total_loss = ce_loss + lambda_lb * lb

        # Diversity loss (only for hetero_K mode where we have expert_mems)
        if expert_mems is not None and lambda_div > 0:
            div = diversity_loss(expert_mems, gate)
            total_loss = total_loss + lambda_div * div

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(moerm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(ce_loss.item())
        if (step + 1) % 50 == 0:
            gate_history.append(gate[0].detach().cpu().tolist())

        if (step + 1) % 200 == 0:
            avg_loss = sum(losses[-200:]) / len(losses[-200:])
            g = gate[0].detach().cpu().tolist()
            gate_str = ' '.join(f'{v:.2f}' for v in g)
            metrics = compute_specialization_metrics(gate.detach())
            print(f'  step {step+1} | loss={avg_loss:.4f} | gate=[{gate_str}] '
                  f'| n_eff={metrics["n_eff"]:.2f} | T={moerm.router.temperature:.2f} '
                  f'| {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    moerm.eval()
    moerm.router.temperature = 1.0
    results = {}

    for K_eval in [0, 1, 2]:
        correct = 0
        n_eval = len(eval_idx)
        all_gates = []

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
                    if hetero_K:
                        mem, gate_e, _, _ = moerm_forward_hetero_K(
                            moerm, emb, K_list, K_inner=4, grad_last_only=False)
                    else:
                        mem, gate_e, _ = moerm(emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                    all_gates.append(gate_e[0].cpu().tolist())
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

        if all_gates and K_eval == 1:
            import numpy as np
            gates_arr = np.array(all_gates)
            mean_gate = gates_arr.mean(axis=0)
            print(f'    Mean gate: [{" ".join(f"{v:.3f}" for v in mean_gate)}]', flush=True)
            print(f'    Gate std: [{" ".join(f"{v:.3f}" for v in gates_arr.std(axis=0))}]', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'moerm',
        'config': config, 'seed': seed, 'total_steps': total_steps,
        'total_params': total_p, 'trainable_params': train_p,
        'final_loss': sum(losses[-50:]) / max(len(losses[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: spatialeval_{tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del moerm, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'shared', 'hetero'])
    parser.add_argument('--n-experts', type=int, default=4)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lambda-lb', type=float, default=1e-3)
    parser.add_argument('--lambda-div', type=float, default=1e-4)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]

    config = {
        'shared_core': args.mode == 'shared',
        'n_experts': args.n_experts,
        'hetero_K': args.mode == 'hetero',
        'K_list': [1, 2, 4, 8][:args.n_experts],
        'lambda_lb': args.lambda_lb,
        'lambda_div': args.lambda_div,
    }

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

    for seed in seeds:
        run_experiment(config, seed, args.steps, tokenizer, base_model, lm_model,
                      maze_data, train_idx, eval_idx)

    print('\n=== All MoERM experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
