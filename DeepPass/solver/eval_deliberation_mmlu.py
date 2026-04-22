"""
Deliberation Controller on MMLU (STEM subset).

Tests whether iterative latent computation helps on knowledge+reasoning tasks.
Uses abstract_algebra, elementary_mathematics, high_school_mathematics,
college_mathematics, high_school_physics, college_physics.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/mmlu'
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

STEM_SUBJECTS = [
    'abstract_algebra', 'elementary_mathematics', 'high_school_mathematics',
    'college_mathematics', 'high_school_physics', 'college_physics',
    'high_school_chemistry', 'college_chemistry',
]


def load_mmlu_stem():
    from datasets import load_dataset
    all_data = []
    for subj in STEM_SUBJECTS:
        try:
            ds = load_dataset('cais/mmlu', subj)
            # Collect from all available splits
            for split_name in ['test', 'validation', 'dev']:
                if split_name in ds:
                    for s in ds[split_name]:
                        s['subject'] = subj
                        all_data.append(s)
            n = sum(len(ds[s]) for s in ds if s in ['test', 'validation', 'dev'])
            print(f'  {subj}: {n} samples', flush=True)
        except Exception as e:
            print(f'  {subj}: FAILED ({e})', flush=True)
    # Split 70/30 for train/test
    random.seed(0)
    random.shuffle(all_data)
    split = int(len(all_data) * 0.7)
    train_data = all_data[:split]
    test_data = all_data[split:]
    print(f'MMLU STEM total: {len(train_data)} train, {len(test_data)} test', flush=True)
    return train_data, test_data


def format_mmlu_prompt(sample):
    q = sample['question']
    choices = sample['choices']
    prompt = f"Question: {q}\n"
    for i, (label, text) in enumerate(zip(['A', 'B', 'C', 'D'], choices)):
        prompt += f"{label}. {text}\n"
    return prompt


def get_choice_token_ids(tokenizer):
    ids = []
    for c in ['A', 'B', 'C', 'D']:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=str, default='1,2,3')
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--seeds', type=str, default='42,7')
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()

    rounds_list = [int(x) for x in args.rounds.split(',')]
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
    choice_ids_tensor = torch.tensor(choice_ids, device=device)
    lm_model = base_model.model

    train_data, test_data = load_mmlu_stem()
    random.seed(0)
    random.shuffle(train_data)
    random.shuffle(test_data)
    n_test = min(len(test_data), 500)
    test_subset = test_data[:n_test]

    # Baseline
    print('\n=== Baseline ===', flush=True)
    correct_base = 0
    for ei, sample in enumerate(test_subset):
        prompt = format_mmlu_prompt(sample) + "Answer:"
        answer_label = sample['answer']  # 0-3

        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = base_model(enc['input_ids'])
            # Check logits at last position for A/B/C/D tokens
            logits = out.logits[:, -1, choice_ids]
            pred = logits.argmax(dim=-1).item()
        if pred == answer_label:
            correct_base += 1
        if (ei + 1) % 100 == 0:
            print(f'  {ei+1}/{n_test} | acc={correct_base/(ei+1):.3f}', flush=True)

    base_acc = correct_base / n_test
    print(f'  Baseline (logit): {base_acc:.4f} ({correct_base}/{n_test})', flush=True)

    for n_rounds in rounds_list:
        for seed in seeds:
            tag = f'mmlu_r{n_rounds}_s{args.slots}_seed{seed}'
            random.seed(seed)
            torch.manual_seed(seed)

            print(f'\n{"="*60}', flush=True)
            print(f'  MMLU STEM: rounds={n_rounds}, seed={seed}', flush=True)
            print(f'{"="*60}', flush=True)

            controller = RecurrentDeliberation(
                frozen_llm=base_model, d_state=512, n_slots=args.slots,
                tapped_layers=(8, 16, 24), topk_vocab=64,
            ).to(device=device, dtype=torch.bfloat16)
            print(f'  Params: {controller.count_trainable():,}', flush=True)

            optimizer = torch.optim.AdamW(
                [p for p in controller.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=0.05
            )
            warmup = 200
            def lr_sched(step, total=args.steps, w=warmup):
                if step < w: return step / w
                return 0.5 * (1 + math.cos(math.pi * (step - w) / (total - w)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

            t0 = time.time()
            losses_hist = []

            for step in range(args.steps):
                sample = train_data[step % len(train_data)]
                prompt_text = format_mmlu_prompt(sample)
                answer_label = sample['answer']  # 0-3

                prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                                       max_length=480, add_special_tokens=True).to(device)
                answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                       add_special_tokens=False).to(device)

                with torch.no_grad():
                    prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                    answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

                label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

                all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
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
                    print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

            # Eval
            print(f'\n  === Eval ({n_test} samples) ===', flush=True)
            controller.eval()
            correct = 0
            per_round_correct = [0] * n_rounds

            for sample in test_subset:
                prompt_text = format_mmlu_prompt(sample)
                answer_label_val = sample['answer']

                prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                                       max_length=480, add_special_tokens=True).to(device)
                answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                                       add_special_tokens=False).to(device)

                with torch.no_grad():
                    prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                    answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                    all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)

                    for r in range(n_rounds):
                        if all_cl[r].argmax(dim=-1).item() == answer_label_val:
                            per_round_correct[r] += 1
                    if all_cl[-1].argmax(dim=-1).item() == answer_label_val:
                        correct += 1

            acc = correct / n_test
            per_r = [c / n_test for c in per_round_correct]
            per_r_str = ' '.join(f'r{i+1}={a:.3f}' for i, a in enumerate(per_r))
            print(f'  FINAL: {acc:.4f} | {per_r_str} | delta={acc-base_acc:+.4f}', flush=True)

            os.makedirs(RESULTS_DIR, exist_ok=True)
            result_data = {
                'tag': tag, 'method': 'deliberation_mmlu',
                'benchmark': 'MMLU-STEM', 'subjects': STEM_SUBJECTS,
                'n_rounds': n_rounds, 'n_slots': args.slots,
                'seed': seed, 'total_steps': args.steps,
                'trainable_params': controller.count_trainable(),
                'baseline_accuracy': base_acc,
                'accuracy': acc, 'correct': correct, 'total': n_test,
                'per_round_accuracy': per_r,
                'delta_vs_baseline': acc - base_acc,
                'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f'  Saved: {tag}.json', flush=True)

            del controller, optimizer, scheduler
            torch.cuda.empty_cache()

    print('\n=== All MMLU experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
