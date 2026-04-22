"""
Adaptive Computation Time via Verifier-Guided Early Stopping.

The thesis claim: LLMs should think harder ONLY when needed.
This experiment tests it directly:
  - Run deliberation up to max_rounds
  - At each round, check verifier confidence
  - If verifier says "confident correct" (>threshold), stop early
  - Compare: fixed rounds vs adaptive stopping

Metrics:
  - accuracy at each strategy
  - average rounds used (compute saved)
  - accuracy-per-round (efficiency curve)
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/adaptive'
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
    parser.add_argument('--max_eval_rounds', type=int, default=10)
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--thresholds', type=str, default='0.5,0.6,0.7,0.8,0.9,0.95')
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',')]

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

    controller = RecurrentDeliberation(
        frozen_llm=base_model, d_state=512, n_slots=args.slots,
        tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

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

        if (step + 1) % 500 == 0:
            avg = sum(losses_hist[-500:]) / len(losses_hist[-500:])
            print(f'  step {step+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    print(f'\nTraining done in {time.time()-t0:.0f}s', flush=True)

    # ========== ADAPTIVE EVAL ==========
    print(f'\n=== Adaptive Computation Time Eval ({len(eval_idx)} samples) ===', flush=True)
    controller.eval()

    # For each sample, run max_eval_rounds and record per-round predictions + verifier
    sample_records = []

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
                                       rounds=args.max_eval_rounds)

        record = {
            'label': answer_label_val,
            'predictions': [cl.argmax(dim=-1).item() for cl in all_cl],
            'verifier_probs': [torch.sigmoid(v).item() for v in all_v],
            'answer_entropy': [],
        }
        for cl in all_cl:
            probs = cl.float().softmax(dim=-1)
            ent = -(probs * probs.clamp_min(1e-8).log()).sum().item()
            record['answer_entropy'].append(ent)

        sample_records.append(record)

    n_eval = len(sample_records)

    # ========== STRATEGY COMPARISON ==========
    results = {}

    # Fixed rounds strategies
    for n_rounds in [1, 2, 3, 5, 8, args.max_eval_rounds]:
        if n_rounds > args.max_eval_rounds:
            continue
        correct = sum(1 for r in sample_records if r['predictions'][n_rounds-1] == r['label'])
        acc = correct / n_eval
        results[f'fixed_{n_rounds}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
            'avg_rounds': float(n_rounds),
        }
        print(f'  Fixed {n_rounds} rounds: {acc:.4f}', flush=True)

    # Adaptive strategies (verifier threshold)
    print(f'\n  --- Adaptive Strategies ---', flush=True)
    for thresh in thresholds:
        correct = 0
        total_rounds_used = 0

        for rec in sample_records:
            stopped_at = args.max_eval_rounds - 1  # default: use all rounds
            for r in range(args.max_eval_rounds):
                if rec['verifier_probs'][r] > thresh:
                    stopped_at = r
                    break
            pred = rec['predictions'][stopped_at]
            if pred == rec['label']:
                correct += 1
            total_rounds_used += stopped_at + 1

        acc = correct / n_eval
        avg_rounds = total_rounds_used / n_eval
        results[f'adaptive_v{thresh}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
            'avg_rounds': avg_rounds,
            'threshold': thresh,
        }
        print(f'  Verifier >{thresh}: acc={acc:.4f} | avg_rounds={avg_rounds:.2f}', flush=True)

    # Entropy-based adaptive (stop when entropy < threshold)
    print(f'\n  --- Entropy-Based Adaptive ---', flush=True)
    for ent_thresh in [0.5, 0.8, 1.0, 1.2]:
        correct = 0
        total_rounds_used = 0

        for rec in sample_records:
            stopped_at = args.max_eval_rounds - 1
            for r in range(args.max_eval_rounds):
                if rec['answer_entropy'][r] < ent_thresh:
                    stopped_at = r
                    break
            pred = rec['predictions'][stopped_at]
            if pred == rec['label']:
                correct += 1
            total_rounds_used += stopped_at + 1

        acc = correct / n_eval
        avg_rounds = total_rounds_used / n_eval
        results[f'adaptive_ent{ent_thresh}'] = {
            'accuracy': acc, 'correct': correct, 'total': n_eval,
            'avg_rounds': avg_rounds,
            'threshold': ent_thresh,
        }
        print(f'  Entropy <{ent_thresh}: acc={acc:.4f} | avg_rounds={avg_rounds:.2f}', flush=True)

    # Summary table
    print(f'\n=== ADAPTIVE COMPUTATION SUMMARY ===', flush=True)
    print(f'  {"Strategy":<25} {"Accuracy":<12} {"Avg Rounds":<12} {"Efficiency":<12}', flush=True)
    print(f'  {"-"*61}', flush=True)
    for name, r in sorted(results.items()):
        eff = r['accuracy'] / r['avg_rounds'] if r['avg_rounds'] > 0 else 0
        print(f'  {name:<25} {r["accuracy"]:<12.4f} {r["avg_rounds"]:<12.2f} {eff:<12.4f}', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': f'adaptive_seed{args.seed}',
        'method': 'adaptive_deliberation',
        'train_rounds': args.train_rounds,
        'max_eval_rounds': args.max_eval_rounds,
        'seed': args.seed, 'total_steps': args.steps,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    fname = f'adaptive_seed{args.seed}.json'
    with open(os.path.join(RESULTS_DIR, fname), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'\nSaved: {fname}. Total time: {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
