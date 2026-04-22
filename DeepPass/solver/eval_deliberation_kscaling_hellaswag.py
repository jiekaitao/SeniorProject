"""
K-Scaling on HellaSwag: Does test-time scaling generalize beyond maze navigation?

Train deliberation controller on HellaSwag with 3 rounds, then evaluate at 1-20.
If we see K-scaling here too, the phenomenon is general, not task-specific.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/kscaling'


def load_hellaswag():
    from datasets import load_dataset
    ds = load_dataset('Rowan/hellaswag')
    train_data = [s for s in ds['train'] if len(s['endings']) == 4]
    val_data = [s for s in ds['validation'] if len(s['endings']) == 4]
    print(f'HellaSwag: {len(train_data)} train, {len(val_data)} val', flush=True)
    return train_data, val_data


def format_hellaswag_prompt(sample):
    ctx = sample['ctx']
    endings = sample['endings']
    prompt = f"{ctx}\n\nWhich ending is most likely?\n"
    for i, (label, ending) in enumerate(zip(['A', 'B', 'C', 'D'], endings)):
        prompt += f"{label}. {ending}\n"
    return prompt


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
    parser.add_argument('--eval_rounds', type=str, default='1,2,3,5,8,10,15')
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

    train_data, val_data = load_hellaswag()
    random.seed(0)
    random.shuffle(train_data)
    n_test = min(len(val_data), 500)
    test_subset = val_data[:n_test]

    # Train
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    controller = RecurrentDeliberation(
        frozen_llm=base_model, d_state=512, n_slots=args.slots,
        tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    print(f'Trainable params: {controller.count_trainable():,}', flush=True)
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

    for step in range(args.steps):
        sample = train_data[step % len(train_data)]
        prompt_text = format_hellaswag_prompt(sample)
        answer_label = int(sample['label'])

        prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                               max_length=480, add_special_tokens=True).to(device)
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

        if (step + 1) % 200 == 0:
            avg = sum(losses_hist[-200:]) / len(losses_hist[-200:])
            r_accs = ' '.join(f'r{r+1}={sum(1 for i in range(max(0,step-199),step+1) if losses_hist[i]<1.0)/200:.3f}'
                              for r in range(min(3, args.train_rounds)))
            print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    print(f'\nTraining done in {time.time()-t0:.0f}s', flush=True)

    # Baseline
    print('\n=== K-Scaling Eval ===', flush=True)
    controller.eval()
    results = {}

    print('Evaluating baseline...', flush=True)
    correct_base = 0
    for sample in test_subset:
        prompt = format_hellaswag_prompt(sample) + "Answer:"
        answer_label = int(sample['label'])
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = base_model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids]
            pred = logits.argmax(dim=-1).item()
        if pred == answer_label:
            correct_base += 1
    base_acc = correct_base / n_test
    results['baseline'] = {'accuracy': base_acc, 'correct': correct_base, 'total': n_test}
    print(f'  baseline: {base_acc:.4f}', flush=True)

    # Evaluate at each round count
    for n_eval_rounds in eval_round_list:
        correct = 0
        per_round_correct = [0] * n_eval_rounds

        for sample in test_subset:
            prompt_text = format_hellaswag_prompt(sample)
            answer_label_val = int(sample['label'])

            prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                                   max_length=480, add_special_tokens=True).to(device)
            answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                   add_special_tokens=False).to(device)
            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_eval_rounds)

                for r in range(n_eval_rounds):
                    if all_cl[r].argmax(dim=-1).item() == answer_label_val:
                        per_round_correct[r] += 1
                if all_cl[-1].argmax(dim=-1).item() == answer_label_val:
                    correct += 1

        acc = correct / n_test
        per_r = [c / n_test for c in per_round_correct]
        results[f'rounds={n_eval_rounds}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_test,
            'per_round_accuracy': per_r,
        }
        last_r = per_r[-1] if per_r else acc
        print(f'  rounds={n_eval_rounds}: {acc:.4f} (delta={acc-base_acc:+.4f})', flush=True)

    # Summary
    print(f'\n=== K-SCALING SUMMARY (HellaSwag) ===', flush=True)
    print(f'  {"Rounds":<10} {"Accuracy":<12} {"Delta":<12}', flush=True)
    print(f'  base       {base_acc:<12.4f} —', flush=True)
    for n in eval_round_list:
        acc = results[f'rounds={n}']['accuracy']
        delta = acc - base_acc
        marker = " ***" if acc == max(r['accuracy'] for k, r in results.items() if k != 'baseline') else ""
        print(f'  {n:<10} {acc:<12.4f} {delta:+.4f}{marker}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': f'kscaling_hellaswag_train{args.train_rounds}_seed{args.seed}',
        'method': 'kscaling_deliberation_hellaswag',
        'benchmark': 'HellaSwag',
        'train_rounds': args.train_rounds, 'n_slots': args.slots,
        'seed': args.seed, 'total_steps': args.steps,
        'baseline_accuracy': base_acc,
        'results': results,
        'eval_round_counts': eval_round_list,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    fname = f'kscaling_hellaswag_train{args.train_rounds}_seed{args.seed}.json'
    with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'\nSaved: {fname}. Total time: {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
