"""
K-Scaling Experiment: Does more deliberation rounds at test time help?

Train deliberation controller with 3 rounds, then evaluate at 1-20 rounds.
This directly tests the core thesis claim: adaptive compute time helps LLMs reason.

If accuracy improves with more rounds, we have K-scaling.
If it plateaus or degrades, more rounds don't help beyond what was trained.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/kscaling'
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_rounds', type=int, default=3)
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--eval_rounds', type=str, default='1,2,3,5,8,10,15,20')
    args = parser.parse_args()

    eval_round_list = [int(x) for x in args.eval_rounds.split(',')]

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
    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model

    maze_data = load_maze_nav()
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # ========== TRAIN ==========
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tapped = (8, 16, 24)

    controller = RecurrentDeliberation(
        frozen_llm=base_model, d_state=512, n_slots=args.slots,
        tapped_layers=tapped, topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    n_params = controller.count_trainable()
    print(f'Trainable params: {n_params:,}', flush=True)
    print(f'Training with {args.train_rounds} rounds, {args.steps} steps...', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses_hist = []
    per_round_train_acc = {r: [] for r in range(args.train_rounds)}

    for step in range(args.steps):
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

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=args.train_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses_hist.append(loss_parts['final_ce'])

        # Track per-round accuracy
        for r in range(args.train_rounds):
            correct = (all_cl[r].argmax(dim=-1) == label_tensor).float().item()
            per_round_train_acc[r].append(correct)

        if (step + 1) % 200 == 0:
            avg = sum(losses_hist[-200:]) / len(losses_hist[-200:])
            r_accs = ' '.join(f'r{r+1}={sum(per_round_train_acc[r][-200:])/200:.3f}'
                              for r in range(args.train_rounds))
            print(f'  step {step+1} | ce={avg:.4f} | {r_accs} | {time.time()-t0:.0f}s', flush=True)

    train_time = time.time() - t0
    print(f'\nTraining done in {train_time:.0f}s', flush=True)

    # ========== EVAL AT DIFFERENT ROUND COUNTS ==========
    print(f'\n=== K-Scaling Eval ({len(eval_idx)} samples) ===', flush=True)
    controller.eval()
    results = {}

    # Baseline (no controller)
    print('Evaluating baseline...', flush=True)
    correct_base = 0
    for idx in eval_idx:
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()
        prompt = text + "\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if oracle in answer.upper()[:10]:
            correct_base += 1
    base_acc = correct_base / len(eval_idx)
    results['baseline'] = {'accuracy': base_acc, 'correct': correct_base, 'total': len(eval_idx)}
    print(f'  baseline: {base_acc:.4f} ({correct_base}/{len(eval_idx)})', flush=True)

    # Evaluate at each round count
    for n_eval_rounds in eval_round_list:
        correct = 0
        per_round_correct = [0] * n_eval_rounds
        verify_probs_sum = [0.0] * n_eval_rounds

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
                all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor,
                                           rounds=n_eval_rounds)

                for r in range(n_eval_rounds):
                    if all_cl[r].argmax(dim=-1).item() == answer_label_val:
                        per_round_correct[r] += 1
                    verify_probs_sum[r] += torch.sigmoid(all_v[r]).item()

                pred = all_cl[-1].argmax(dim=-1).item()
                if pred == answer_label_val:
                    correct += 1

        n_eval = len(eval_idx)
        acc = correct / n_eval
        per_r_accs = [c / n_eval for c in per_round_correct]
        avg_verify = [v / n_eval for v in verify_probs_sum]

        results[f'rounds={n_eval_rounds}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
            'per_round_accuracy': per_r_accs,
            'avg_verifier_confidence': avg_verify,
        }

        per_r_str = ' '.join(f'r{i+1}={a:.3f}' for i, a in enumerate(per_r_accs[-5:]))  # last 5
        trend = "UP" if len(per_r_accs) > 1 and per_r_accs[-1] > per_r_accs[0] else "DOWN/FLAT"
        print(f'  rounds={n_eval_rounds}: final={acc:.4f} | {per_r_str} | trend={trend}', flush=True)

    # Summary table
    print(f'\n=== K-SCALING SUMMARY ===', flush=True)
    print(f'  {"Rounds":<10} {"Accuracy":<12} {"Delta vs base":<15}', flush=True)
    print(f'  {"-"*37}', flush=True)
    print(f'  {"base":<10} {base_acc:<12.4f} {"—":<15}', flush=True)
    for n_eval_rounds in eval_round_list:
        key = f'rounds={n_eval_rounds}'
        acc = results[key]['accuracy']
        delta = acc - base_acc
        marker = " ***" if acc == max(r['accuracy'] for k, r in results.items() if k != 'baseline') else ""
        print(f'  {n_eval_rounds:<10} {acc:<12.4f} {delta:+.4f}{marker}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': f'kscaling_train{args.train_rounds}_seed{args.seed}',
        'method': 'kscaling_deliberation',
        'train_rounds': args.train_rounds, 'n_slots': args.slots,
        'seed': args.seed, 'total_steps': args.steps,
        'trainable_params': n_params,
        'train_time_s': train_time,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'results': results,
        'eval_round_counts': eval_round_list,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'kscaling_train{args.train_rounds}_seed{args.seed}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'\nSaved. Total time: {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
