"""
Creative Deliberation Experiments — testing novel training strategies.

Experiment 1: Multi-task controller (ONE controller, ALL SpatialEval tasks)
Experiment 2: Progressive round curriculum (1→2→3→5 rounds during training)
Experiment 3: Self-distillation (5-round teacher → 1-round student)
Experiment 4: Mid-layer injection (inject thoughts at layer 16 instead of input)
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/creative'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


class LowrankDeliberation(RecurrentDeliberation):
    """Deliberation with lowrank writer (proven best)."""
    def __init__(self, frozen_llm, rank=64, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        d_state = kwargs.get('d_state', 512)
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part


class MidLayerDeliberation(LowrankDeliberation):
    """Inject thought tokens at a mid-layer instead of input.
    Avoids early FFN corruption of thought signal."""

    def __init__(self, frozen_llm, inject_layer=16, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        self.inject_layer = inject_layer

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb):
        lm_model = self.frozen_llm.model

        # Run first N layers with just prompt + answer (no thoughts)
        dec_input = torch.cat([prompt_emb, answer_emb], dim=1)
        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=dec_input.device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_input, pos_ids)

        h = dec_input
        tapped_pools = []

        for i, layer in enumerate(lm_model.layers):
            if i == self.inject_layer:
                # Inject thought tokens at this layer
                # Thought tokens get position IDs after the answer
                t_len = thought_emb.shape[1]
                # Concatenate thoughts into the hidden states
                h = torch.cat([
                    h[:, :prompt_emb.shape[1]],  # prompt hidden states
                    thought_emb.to(h.dtype),       # thought tokens (fresh)
                    h[:, prompt_emb.shape[1]:]     # answer hidden states
                ], dim=1)
                # Recompute position embeddings for extended sequence
                T_new = h.shape[1]
                pos_ids_new = torch.arange(T_new, device=h.device).unsqueeze(0)
                pos_emb = lm_model.rotary_emb(h, pos_ids_new)

            h = layer(h, position_embeddings=pos_emb)
            if i in self.tapped_layers:
                tapped_pools.append(h.mean(dim=1))

        h = lm_model.norm(h)
        logits = self.frozen_llm.lm_head(h)

        # Think slot hidden states (if injected)
        if self.inject_layer < len(lm_model.layers):
            p_len = prompt_emb.shape[1]
            t_len = thought_emb.shape[1]
            think_h = h[:, p_len:p_len+t_len]
        else:
            think_h = h[:, :self.n_slots]  # fallback

        return logits, think_h, tapped_pools


def load_spatialeval(tasks):
    """Load multiple SpatialEval tasks."""
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    data = {}
    for task in tasks:
        data[task] = [s for s in ds if s['id'].startswith(task)]
        print(f'  {task}: {len(data[task])} samples', flush=True)
    return data


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def run_multitask(seed, total_steps, n_rounds, tokenizer, base_model, all_data, choice_ids):
    """Train ONE controller on ALL tasks simultaneously."""
    tag = f'multitask_r{n_rounds}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  MULTI-TASK: {list(all_data.keys())} | Rounds: {n_rounds} | Seed: {seed}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = LowrankDeliberation(
        frozen_llm=base_model, rank=64,
        d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model

    # Build mixed training pool
    tasks = list(all_data.keys())
    train_pools = {}
    eval_pools = {}
    for task in tasks:
        random.seed(0)
        indices = list(range(len(all_data[task])))
        random.shuffle(indices)
        split = min(1000, len(indices) * 2 // 3)
        train_pools[task] = indices[:split]
        eval_pools[task] = indices[split:]

    random.seed(seed)
    t0 = time.time()
    losses_hist = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        # Sample random task
        task = random.choice(tasks)
        data = all_data[task]
        idx = train_pools[task][step % len(train_pools[task])]
        sample = data[idx]

        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)
        total_loss = total_loss / 8  # grad accum 8
        total_loss.backward()

        if (step + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 500 == 0:
            avg = sum(losses_hist[-500:]) / len(losses_hist[-500:])
            print(f'  step {step+1} | ce={avg:.4f} | task={task} | {time.time()-t0:.0f}s', flush=True)

    # Eval per task
    print(f'\n  === Multi-Task Eval ===', flush=True)
    controller.eval()
    results = {}

    for task in tasks:
        data = all_data[task]
        eval_idx = eval_pools[task]
        correct = 0
        n_eval = len(eval_idx)

        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            answer_label_val = CHOICE_MAP.get(oracle[0], 0)

            prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                                   add_special_tokens=True).to(device)
            answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                   add_special_tokens=False).to(device)

            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
                pred = all_cl[-1].argmax(dim=-1).item()
            if pred == answer_label_val:
                correct += 1

        acc = correct / n_eval
        results[task] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  {task}: {acc:.4f} ({correct}/{n_eval})', flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'multitask_deliberation',
        'tasks': tasks, 'n_rounds': n_rounds, 'seed': seed,
        'total_steps': total_steps, 'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer
    torch.cuda.empty_cache()
    return results


def run_curriculum(seed, total_steps, tokenizer, base_model, maze_data, train_idx, eval_idx, choice_ids):
    """Progressive round curriculum: 1→2→3→5 rounds during training."""
    tag = f'curriculum_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  CURRICULUM: 1→2→3→5 rounds | Seed: {seed}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = LowrankDeliberation(
        frozen_llm=base_model, rank=64,
        d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model

    # Curriculum schedule: which round count at each phase
    # 0-750: 1 round, 750-1500: 2 rounds, 1500-2250: 3 rounds, 2250-3000: 5 rounds
    phase_size = total_steps // 4
    curriculum = [(1, phase_size), (2, phase_size), (3, phase_size), (5, total_steps - 3*phase_size)]

    t0 = time.time()
    losses_hist = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    for n_rounds, n_steps in curriculum:
        print(f'  --- Phase: {n_rounds} rounds for {n_steps} steps ---', flush=True)
        for step in range(n_steps):
            sample = maze_data[train_idx[global_step % len(train_idx)]]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            answer_label = CHOICE_MAP.get(oracle[0], 0)

            prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                                   add_special_tokens=True).to(device)
            answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                   add_special_tokens=False).to(device)

            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

            label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

            all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
            total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)
            total_loss = total_loss / 8
            total_loss.backward()

            if (global_step + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in controller.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses_hist.append(loss_parts['final_ce'])
            global_step += 1

            if (global_step) % 500 == 0:
                avg = sum(losses_hist[-500:]) / len(losses_hist[-500:])
                print(f'  step {global_step} | ce={avg:.4f} | rounds={n_rounds} | {time.time()-t0:.0f}s', flush=True)

    # Eval at different round counts
    print(f'\n  === Curriculum Eval ===', flush=True)
    controller.eval()
    results = {}

    for eval_rounds in [1, 2, 3, 5, 8]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            answer_label_val = CHOICE_MAP.get(oracle[0], 0)

            prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                                   add_special_tokens=True).to(device)
            answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                   add_special_tokens=False).to(device)
            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=eval_rounds)
                pred = all_cl[-1].argmax(dim=-1).item()
            if pred == answer_label_val:
                correct += 1

        acc = correct / n_eval
        results[f'rounds={eval_rounds}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  rounds={eval_rounds}: {acc:.4f} ({correct}/{n_eval})', flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'curriculum_deliberation',
        'curriculum': [(r, s) for r, s in curriculum],
        'seed': seed, 'total_steps': total_steps,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer
    torch.cuda.empty_cache()
    return results


def run_midlayer(seed, total_steps, n_rounds, inject_layer, tokenizer, base_model,
                 maze_data, train_idx, eval_idx, choice_ids):
    """Inject thought tokens at a mid-layer instead of input."""
    tag = f'midlayer_L{inject_layer}_r{n_rounds}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  MID-LAYER INJECTION: layer={inject_layer} | Rounds: {n_rounds} | Seed: {seed}', flush=True)
    print(f'{"="*60}', flush=True)

    controller = MidLayerDeliberation(
        frozen_llm=base_model, inject_layer=inject_layer, rank=64,
        d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model
    t0 = time.time()
    losses_hist = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)
        total_loss = total_loss / 8
        total_loss.backward()

        if (step + 1) % 8 == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 500 == 0:
            avg = sum(losses_hist[-500:]) / len(losses_hist[-500:])
            print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n  === Mid-Layer Eval ===', flush=True)
    controller.eval()
    correct = 0
    n_eval = len(eval_idx)
    for idx in eval_idx:
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label_val = CHOICE_MAP.get(oracle[0], 0)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
            all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
            pred = all_cl[-1].argmax(dim=-1).item()
        if pred == answer_label_val:
            correct += 1

    acc = correct / n_eval
    print(f'  midlayer L{inject_layer}: {acc:.4f} ({correct}/{n_eval})', flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'midlayer_deliberation',
        'inject_layer': inject_layer,
        'n_rounds': n_rounds, 'seed': seed, 'total_steps': total_steps,
        'accuracy': acc, 'correct': correct, 'total': n_eval,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['multitask', 'curriculum', 'midlayer', 'all'])
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
    print('Model loaded.', flush=True)

    choice_ids = get_choice_token_ids(tokenizer)

    experiments = [args.experiment] if args.experiment != 'all' else ['multitask', 'curriculum', 'midlayer']

    for exp in experiments:
        if exp == 'multitask':
            tasks = ['mazenav', 'spatialmap', 'spatialgrid']
            all_data = load_spatialeval(tasks)
            for seed in seeds:
                run_multitask(seed, args.steps, 3, tokenizer, base_model, all_data, choice_ids)

        elif exp == 'curriculum':
            from datasets import load_dataset
            ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
            maze_data = [s for s in ds if s['id'].startswith('mazenav')]
            random.seed(0)
            indices = list(range(len(maze_data)))
            random.shuffle(indices)
            train_idx, eval_idx = indices[:1000], indices[1000:]

            for seed in seeds:
                run_curriculum(seed, args.steps, tokenizer, base_model, maze_data,
                              train_idx, eval_idx, choice_ids)

        elif exp == 'midlayer':
            from datasets import load_dataset
            ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
            maze_data = [s for s in ds if s['id'].startswith('mazenav')]
            random.seed(0)
            indices = list(range(len(maze_data)))
            random.shuffle(indices)
            train_idx, eval_idx = indices[:1000], indices[1000:]

            # Test injection at different layers
            for inject_layer in [8, 16, 24]:
                for seed in seeds[:1]:  # Just seed 42 for sweep
                    run_midlayer(seed, args.steps, 3, inject_layer, tokenizer, base_model,
                                maze_data, train_idx, eval_idx, choice_ids)

    print('\n=== All creative experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
