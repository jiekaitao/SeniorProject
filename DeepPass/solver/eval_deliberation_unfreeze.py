"""
Partial Unfreezing Experiment: Can we break the 72% ceiling?

The 72% ceiling on SpatialEval mazenav has been proven fundamental to the
frozen decoder. What if we unfreeze the last K layers during deliberation
training? This allows the decoder to adapt its answer-producing layers
while keeping most of the model frozen.

Tests K = 0 (fully frozen, control), 2, 4, 8 unfrozen layers.
Uses lower LR for unfrozen layers to prevent catastrophic forgetting.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/unfreeze'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def unfreeze_last_k_layers(model, k):
    """Unfreeze the last K decoder layers + final norm + lm_head."""
    n_layers = len(model.model.layers)
    unfrozen_count = 0

    # Unfreeze last K layers
    for i in range(n_layers - k, n_layers):
        for p in model.model.layers[i].parameters():
            p.requires_grad = True
            unfrozen_count += p.numel()

    # Unfreeze final norm
    for p in model.model.norm.parameters():
        p.requires_grad = True
        unfrozen_count += p.numel()

    # Unfreeze lm_head
    for p in model.lm_head.parameters():
        p.requires_grad = True
        unfrozen_count += p.numel()

    print(f'  Unfroze last {k}/{n_layers} layers + norm + lm_head = {unfrozen_count:,} params', flush=True)
    return unfrozen_count


def refreeze_all(model):
    """Re-freeze all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def run_experiment(n_unfreeze, seed, total_steps, n_rounds, n_slots,
                   tokenizer, base_model, maze_data, train_idx, eval_idx, choice_ids):
    tag = f'unfreeze_k{n_unfreeze}_r{n_rounds}_s{n_slots}_seed{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'\n{"="*60}', flush=True)
    print(f'  Unfreeze last {n_unfreeze} layers | rounds={n_rounds} | seed={seed}', flush=True)
    print(f'{"="*60}', flush=True)

    # Re-freeze everything first (in case of previous experiment)
    refreeze_all(base_model)

    # Create controller
    controller = RecurrentDeliberation(
        frozen_llm=base_model, d_state=512, n_slots=n_slots,
        tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    controller_params = controller.count_trainable()
    print(f'  Controller params: {controller_params:,}', flush=True)

    # Unfreeze decoder layers
    unfrozen_params = 0
    if n_unfreeze > 0:
        unfrozen_params = unfreeze_last_k_layers(base_model, n_unfreeze)

    total_trainable = controller_params + unfrozen_params
    print(f'  Total trainable: {total_trainable:,} ({total_trainable/1e6:.1f}M)', flush=True)

    # Two param groups: controller-only (higher LR) + unfrozen decoder (lower LR)
    # controller.parameters() includes base_model since it's a submodule,
    # so we need to separate them explicitly
    base_param_ids = set(id(p) for p in base_model.parameters())
    controller_only_params = [p for p in controller.parameters()
                              if p.requires_grad and id(p) not in base_param_ids]
    decoder_params = [p for p in base_model.parameters() if p.requires_grad]

    param_groups = [{'params': controller_only_params, 'lr': 1e-4}]
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': 1e-5})  # 10x lower

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    print(f'  Optimizer: {len(controller_only_params)} controller params, {len(decoder_params)} decoder params', flush=True)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model
    t0 = time.time()
    losses_hist = []

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        answer_label = CHOICE_MAP.get(oracle[0], 0)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                               add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        # For unfrozen layers, we need gradients through embeddings
        if n_unfreeze > 0:
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
        else:
            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)

        if total_loss.requires_grad:
            total_loss.backward()
            # Clip all trainable params
            all_trainable = [p for p in controller.parameters() if p.requires_grad]
            if n_unfreeze > 0:
                all_trainable += [p for p in base_model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses_hist.append(loss_parts['final_ce'])

        if (step + 1) % 200 == 0:
            avg = sum(losses_hist[-200:]) / len(losses_hist[-200:])
            r1 = (all_cl[0].argmax(dim=-1) == label_tensor).float().item()
            rN = (all_cl[-1].argmax(dim=-1) == label_tensor).float().item()
            print(f'  step {step+1} | ce={avg:.4f} | r1={r1:.0f} rN={rN:.0f} | {time.time()-t0:.0f}s', flush=True)

    # ========== EVAL ==========
    print(f'\n  === Eval ({len(eval_idx)} samples) ===', flush=True)
    controller.eval()
    base_model.eval()
    results = {}

    for eval_rounds in [0, n_rounds]:
        correct = 0
        n_eval = len(eval_idx)
        per_round_correct = [0] * max(eval_rounds, 1)

        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option'].strip().upper()
            answer_label_val = CHOICE_MAP.get(oracle[0], 0)

            if eval_rounds == 0:
                prompt = text + "\nAnswer:"
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                if oracle in answer.upper()[:10]:
                    correct += 1
            else:
                prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900,
                                       add_special_tokens=True).to(device)
                answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                       add_special_tokens=False).to(device)
                with torch.no_grad():
                    prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                    answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                    all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=eval_rounds)

                    for r in range(eval_rounds):
                        if all_cl[r].argmax(dim=-1).item() == answer_label_val:
                            per_round_correct[r] += 1

                    pred = all_cl[-1].argmax(dim=-1).item()
                    if pred == answer_label_val:
                        correct += 1

        acc = correct / n_eval
        results[f'rounds={eval_rounds}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
        }
        if eval_rounds > 0:
            results[f'rounds={eval_rounds}']['per_round'] = [c / n_eval for c in per_round_correct]

        label = 'baseline (modified decoder)' if eval_rounds == 0 else f'rounds={eval_rounds}'
        print(f'  {label}: {acc:.4f} ({correct}/{n_eval})', flush=True)
        if eval_rounds > 0:
            per_r_str = ' '.join(f'r{i+1}={c/n_eval:.3f}' for i, c in enumerate(per_round_correct))
            print(f'    per-round: {per_r_str}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'partial_unfreeze_deliberation',
        'n_unfreeze': n_unfreeze,
        'n_rounds': n_rounds, 'n_slots': n_slots,
        'seed': seed, 'total_steps': total_steps,
        'controller_params': controller_params,
        'unfrozen_params': unfrozen_params,
        'total_trainable': total_trainable,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    # IMPORTANT: re-freeze everything for next experiment
    refreeze_all(base_model)
    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unfreeze', type=str, default='0,2,4,8')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()

    unfreeze_list = [int(x) for x in args.unfreeze.split(',')]

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
    maze_data = load_maze_nav()

    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    all_results = {}
    for n_unfreeze in unfreeze_list:
        results = run_experiment(
            n_unfreeze, args.seed, args.steps, args.rounds, args.slots,
            tokenizer, base_model, maze_data, train_idx, eval_idx, choice_ids
        )
        all_results[f'unfreeze={n_unfreeze}'] = results

    # Summary
    print(f'\n=== UNFREEZING SUMMARY ===', flush=True)
    print(f'  {"Layers":<12} {"Accuracy":<12} {"Delta vs frozen":<15}', flush=True)
    print(f'  {"-"*39}', flush=True)
    frozen_acc = None
    for n_unfreeze in unfreeze_list:
        key = f'unfreeze={n_unfreeze}'
        # Get the deliberation accuracy (not baseline)
        for rk, rv in all_results[key].items():
            if 'rounds=' in rk and rk != 'rounds=0':
                acc = rv['accuracy']
                if frozen_acc is None:
                    frozen_acc = acc
                delta = acc - frozen_acc if frozen_acc else 0
                print(f'  {n_unfreeze:<12} {acc:<12.4f} {delta:+.4f}', flush=True)

    print('\n=== All unfreeze experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
