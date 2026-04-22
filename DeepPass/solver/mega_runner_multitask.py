"""Multi-task controller training + zero-shot held-out evaluation.

Direct test of 'general intelligence' claim: if a single controller trained on
4 benchmarks transfers zero-shot to 4 held-out benchmarks, the controller is
learning general reasoning, not benchmark-specific shortcuts.

Train pool:     HellaSwag + WinoGrande + BoolQ + SpatialGrid (samples mixed)
Held-out pool:  OpenBookQA + CommonsenseQA + SpatialMap + Mazenav

Usage:
  python solver/mega_runner_multitask.py \
      --model models/full/Llama-3.1-8B-Instruct \
      --train_benchmarks hellaswag,winogrande,boolq,spatialgrid \
      --heldout_benchmarks openbookqa,commonsenseqa,spatialmap,mazenav \
      --total_steps 16000 --seed 42 --grad_accum 16

Each benchmark can have a different number of choices (2-5). The controller
handles this via a shared vocab-superposition + lowrank writer; for eval, we
use the subset of choice letters matching each benchmark's n_choices.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from mega_runner_benchmarks import (
    FlexMidLayerDeliberation, BENCHMARK_LOADERS,
)

device = torch.device('cuda')
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


def load_spatialeval(task):
    """Load one SpatialEval subtask in the same format as BENCHMARK_LOADERS."""
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    data = [s for s in ds if s['id'].startswith(task)]
    random.seed(0)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split = min(1000, len(indices) * 2 // 3)
    train, test = indices[:split], indices[split:]
    fmt = {'train': [], 'test': []}
    for name, idxs in [('train', train), ('test', test)]:
        for i in idxs:
            s = data[i]
            oracle = s['oracle_option'].strip().upper()[0]
            fmt[name].append({
                'text': s['text'], 'label': CHOICE_MAP.get(oracle, 0),
                'n_choices': 4,
            })
    return fmt, 4


def get_max_choice_tokens(tokenizer, max_n=5):
    """Get token IDs for A/B/C/D/E."""
    letters = ['A', 'B', 'C', 'D', 'E'][:max_n]
    return [tokenizer.encode(f' {c}', add_special_tokens=False)[0] for c in letters]


def eval_benchmark(controller, base_model, tokenizer, data, choice_ids_all,
                   max_n, eval_rounds=5, limit=500):
    controller.eval()
    test_data = data[:limit]
    correct = 0
    total = 0
    lm_model = base_model.model
    answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
        for sample in test_data:
            n = sample.get('n_choices', 4)
            choice_ids_t = torch.tensor(choice_ids_all[:n], device=device)
            text = sample['text'][:1500]
            prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            # Temporarily align controller.n_choices so feature shapes line up
            orig_n = controller.n_choices
            # The FlexMidLayerDeliberation we built uses n_choices for build_features;
            # we pad if necessary in the forward. Best: just pass the full 5-choice
            # ids and only look at first n.
            choice_ids_full = torch.tensor(choice_ids_all[:max_n], device=device)
            all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_full, rounds=eval_rounds)
            final = all_cl[-1]  # (B, max_n)
            pred = final[:, :n].argmax(dim=-1).item()
            if pred == sample['label']:
                correct += 1
            total += 1
    controller.train()
    return correct / max(total, 1), correct, total


def train_multitask(base_model, tokenizer, train_pools, total_steps, seed, grad_accum,
                    inject_layer, n_rounds, max_n, choice_ids_all, tag, results_dir):
    random.seed(seed)
    torch.manual_seed(seed)

    controller = FlexMidLayerDeliberation(
        frozen_llm=base_model, inject_layer=inject_layer, n_choices=max_n,
        rank=64, d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)
    print(f'  Controller: {sum(p.numel() for p in controller.parameters() if p.requires_grad)/1e6:.1f}M params, {max_n} choices (max)', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    lm_model = base_model.model
    answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
    with torch.no_grad():
        answer_emb_base = lm_model.embed_tokens(answer_enc['input_ids'])

    choice_ids_full = torch.tensor(choice_ids_all[:max_n], device=device)

    # Build round-robin indices into each benchmark's train set
    pools = [(name, pool, random.sample(range(len(pool)), len(pool))) for name, pool in train_pools.items()]
    counters = {name: 0 for name, _, _ in pools}

    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        # Round-robin: each step picks next task
        task_idx = step % len(pools)
        task_name, pool, order = pools[task_idx]
        idx = order[counters[task_name] % len(order)]
        counters[task_name] += 1
        sample = pool[idx]
        n = sample.get('n_choices', 4)
        text = sample['text'][:1500]
        label = sample['label']

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
        label_t = torch.tensor([label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb_base, choice_ids_full, rounds=n_rounds)
        # Only compute CE on the first n choices
        if max_n == n:
            loss, lp = controller.compute_loss(all_cl, all_v, label_t)
        else:
            # Mask out choices beyond n by slicing
            cl_sliced = [cl[:, :n] for cl in all_cl]
            # Use CE manually on sliced logits (skip verifier if dimensions mismatch)
            from torch.nn.functional import cross_entropy, relu
            final_ce = cross_entropy(cl_sliced[-1].float(), label_t)
            first_ce = cross_entropy(cl_sliced[0].float(), label_t) if len(cl_sliced) > 1 else final_ce
            progress = relu(final_ce - first_ce + 0.1)
            # verifier BCE against correctness
            verify_loss = torch.tensor(0., device=device, dtype=torch.float32)
            for r, v in enumerate(all_v):
                pred_correct = (cl_sliced[r].argmax(dim=-1) == label_t).float()
                verify_loss = verify_loss + F.binary_cross_entropy_with_logits(
                    v.float().squeeze(-1), pred_correct)
            verify_loss = verify_loss / max(len(all_v), 1)
            loss = final_ce + 0.5 * verify_loss + 0.1 * progress
            lp = {'final_ce': final_ce.item(), 'verify_loss': verify_loss.item(),
                  'progress_loss': float(progress.item())}
        loss = loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(lp['final_ce'])
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {task_name} | {time.time()-t0:.0f}s', flush=True)

    return controller, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--train_benchmarks', type=str,
                        default='hellaswag,winogrande,boolq,spatialgrid')
    parser.add_argument('--heldout_benchmarks', type=str,
                        default='openbookqa,commonsenseqa,spatialmap,mazenav')
    parser.add_argument('--inject_layer', type=int, default=12)
    parser.add_argument('--n_rounds', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str, default='results/data/multitask')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    for p in base_model.parameters(): p.requires_grad = False
    print('Model loaded.', flush=True)

    # Load all benchmarks
    def load_any(name):
        if name in BENCHMARK_LOADERS:
            return BENCHMARK_LOADERS[name]()
        # SpatialEval fallback
        if name in ('spatialgrid', 'mazenav', 'spatialmap', 'spatialreal'):
            return load_spatialeval(name)
        raise ValueError(f'Unknown benchmark: {name}')

    train_benchmarks = [b.strip() for b in args.train_benchmarks.split(',')]
    heldout_benchmarks = [b.strip() for b in args.heldout_benchmarks.split(',')]

    train_pools = {}
    n_choices_all = {}
    for b in train_benchmarks:
        data, n = load_any(b)
        train_pools[b] = data['train']
        n_choices_all[b] = n
    heldout_data = {}
    for b in heldout_benchmarks:
        data, n = load_any(b)
        heldout_data[b] = (data['test'], n)
        n_choices_all[b] = n

    max_n = max(n_choices_all.values())
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'
    choice_ids_all = get_max_choice_tokens(tokenizer, max_n=max_n)

    tag = f'multitask_{model_short}_train{"+".join(b[:3] for b in train_benchmarks)}_{args.total_steps//1000}k_s{args.seed}'
    print(f'Tag: {tag}', flush=True)
    print(f'Train benchmarks: {train_benchmarks}', flush=True)
    print(f'Held-out benchmarks: {heldout_benchmarks}', flush=True)
    print(f'Max choices: {max_n}', flush=True)
    for b, n in n_choices_all.items():
        print(f'  {b}: n_choices={n}, train={len(train_pools.get(b, []))}, test={len(heldout_data[b][0]) if b in heldout_data else "(train-only)"}', flush=True)

    # Train
    controller, train_time = train_multitask(
        base_model, tokenizer, train_pools,
        total_steps=args.total_steps, seed=args.seed, grad_accum=args.grad_accum,
        inject_layer=args.inject_layer, n_rounds=args.n_rounds,
        max_n=max_n, choice_ids_all=choice_ids_all, tag=tag, results_dir=args.results_dir,
    )
    print(f'Training done in {train_time:.0f}s', flush=True)

    # Eval on BOTH train benchmarks (should be OK) and held-out (the real test)
    results = {}
    lm_model = base_model.model
    answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)

    # Recompute test sets for train benchmarks too (sanity)
    train_test = {}
    for b in train_benchmarks:
        data, _ = load_any(b)
        train_test[b] = (data['test'], n_choices_all[b])

    all_eval = {**train_test, **heldout_data}
    for b, (test_data, n) in all_eval.items():
        is_heldout = b in heldout_benchmarks
        print(f'\n  === Eval on {b} ({"HELDOUT" if is_heldout else "train"}, n={n}) ===', flush=True)
        for er in [3, 5, 8]:
            acc, correct, total = eval_benchmark(
                controller, base_model, tokenizer, test_data,
                choice_ids_all=choice_ids_all, max_n=max_n, eval_rounds=er, limit=500)
            results.setdefault(b, {})[f'rounds={er}'] = {
                'accuracy': acc, 'correct': correct, 'total': total,
                'is_heldout': is_heldout,
            }
            print(f'    rounds={er}: {acc:.4f} ({correct}/{total})', flush=True)

    os.makedirs(args.results_dir, exist_ok=True)
    out = {
        'tag': tag,
        'model': args.model,
        'train_benchmarks': train_benchmarks,
        'heldout_benchmarks': heldout_benchmarks,
        'total_steps': args.total_steps,
        'seed': args.seed,
        'grad_accum': args.grad_accum,
        'inject_layer': args.inject_layer,
        'n_rounds': args.n_rounds,
        'train_time_s': train_time,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.results_dir, f'{tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {tag}.json', flush=True)


if __name__ == '__main__':
    main()
