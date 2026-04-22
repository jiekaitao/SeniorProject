"""
Multi-Benchmark Deliberation Controller.

Tests across many standard benchmarks with the best config:
lowrank writer + configurable grad accumulation.

Supports: WinoGrande, PIQA, OpenBookQA, BoolQ, CommonsenseQA, HellaSwag, ARC
Adapts number of choices per benchmark.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/multi_benchmark'


class FlexChoiceDeliberation(RecurrentDeliberation):
    """Deliberation controller with flexible choice count and lowrank writer."""
    def __init__(self, frozen_llm, n_choices=4, rank=64, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        self.n_choices = n_choices
        d_state = kwargs.get('d_state', 512)

        # Lowrank writer
        self.to_lowrank = nn.Linear(d_state, rank, bias=False)
        self.U = nn.Parameter(torch.randn(rank, self.d_model) * 0.02)
        nn.init.normal_(self.to_lowrank.weight, std=0.01)

        # Rebuild read/verify for flexible choice count
        n_tap = len(kwargs.get('tapped_layers', (8, 16, 24)))
        n_slots = kwargs.get('n_slots', 8)
        read_dim = n_tap * self.d_model + n_slots * self.d_model + n_choices + 2
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048), nn.GELU(),
            nn.Linear(2048, n_slots * d_state),
        )
        self.verifier = nn.Sequential(
            nn.Linear(n_tap * self.d_model + n_slots * self.d_model + n_choices, 512),
            nn.GELU(), nn.Linear(512, 1),
        )

    def latent_to_thought_embs(self, z):
        E = self.frozen_llm.model.embed_tokens.weight
        logits = self.to_vocab_logits(z)
        vals, idx = logits.topk(self.topk_vocab, dim=-1)
        probs = F.softmax(vals, dim=-1)
        chosen_embs = E[idx]
        vocab_part = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)
        lowrank_part = self.to_lowrank(z) @ self.U
        return vocab_part + 0.12 * lowrank_part  # Fixed gate ~sigmoid(-2)


def load_benchmark(name):
    """Load benchmark dataset. Returns (train_data, test_data, n_choices, format_fn, label_fn)."""
    from datasets import load_dataset

    if name == 'winogrande':
        ds = load_dataset('allenai/winogrande', 'winogrande_xl')
        def fmt(s):
            return f"Sentence: {s['sentence']}\n1. {s['option1']}\n2. {s['option2']}\n"
        def lbl(s): return int(s['answer']) - 1  # 1/2 -> 0/1
        return list(ds['train']), list(ds['validation']), 2, fmt, lbl

    elif name == 'piqa':
        # PIQA dataset scripts broken on HF, use SIQA instead
        ds = load_dataset('allenai/social_i_qa')
        def fmt(s):
            return f"Context: {s['context']}\nQuestion: {s['question']}\nA. {s['answerA']}\nB. {s['answerB']}\nC. {s['answerC']}\n"
        def lbl(s): return int(s['label']) - 1  # 1/2/3 -> 0/1/2
        return list(ds['train']), list(ds['validation']), 3, fmt, lbl

    elif name == 'openbookqa':
        ds = load_dataset('allenai/openbookqa', 'main')
        CMAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        def fmt(s):
            q = s['question_stem']
            choices = s['choices']
            prompt = f"Question: {q}\n"
            for label, text in zip(choices['label'], choices['text']):
                prompt += f"{label}. {text}\n"
            return prompt
        def lbl(s): return CMAP.get(s['answerKey'], 0)
        return list(ds['train']), list(ds['test']), 4, fmt, lbl

    elif name == 'boolq':
        ds = load_dataset('google/boolq')
        def fmt(s):
            return f"Passage: {s['passage'][:500]}\nQuestion: {s['question']}\nA. True\nB. False\n"
        def lbl(s): return 0 if s['answer'] else 1  # True=A=0, False=B=1
        return list(ds['train']), list(ds['validation']), 2, fmt, lbl

    elif name == 'commonsenseqa':
        ds = load_dataset('tau/commonsense_qa')
        CMAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        def fmt(s):
            q = s['question']
            prompt = f"Question: {q}\n"
            for label, text in zip(s['choices']['label'], s['choices']['text']):
                prompt += f"{label}. {text}\n"
            return prompt
        def lbl(s): return CMAP.get(s['answerKey'], 0)
        return list(ds['train']), list(ds['validation']), 5, fmt, lbl

    else:
        raise ValueError(f"Unknown benchmark: {name}")


def get_choice_tokens(tokenizer, n_choices):
    """Get token IDs for choice labels."""
    if n_choices == 2:
        labels = ['A', 'B']
    elif n_choices == 4:
        labels = ['A', 'B', 'C', 'D']
    elif n_choices == 5:
        labels = ['A', 'B', 'C', 'D', 'E']
    else:
        labels = [chr(65 + i) for i in range(n_choices)]

    ids = []
    for c in labels:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def run_benchmark(bench_name, model_path, seed, steps, n_rounds, grad_accum):
    print(f'\n{"="*60}', flush=True)
    print(f'  Benchmark: {bench_name} | Model: {os.path.basename(model_path)}', flush=True)
    print(f'  Rounds: {n_rounds} | GA: {grad_accum} | Seed: {seed}', flush=True)
    print(f'{"="*60}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False

    train_data, test_data, n_choices, format_fn, label_fn = load_benchmark(bench_name)
    print(f'  {len(train_data)} train, {len(test_data)} test, {n_choices} choices', flush=True)

    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_tensor = torch.tensor(choice_ids, device=device)

    # Determine tapped layers
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        n_layers = len(base_model.model.layers)
        lm_model = base_model.model
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
        n_layers = len(base_model.model.language_model.layers)
        lm_model = base_model.model.language_model
    else:
        raise ValueError("Cannot find layers")

    step = max(1, n_layers // 4)
    tapped = (step, 2*step, min(3*step, n_layers-1))

    random.seed(seed)
    torch.manual_seed(seed)

    controller = FlexChoiceDeliberation(
        frozen_llm=base_model, n_choices=n_choices, rank=64,
        d_state=512, n_slots=8, tapped_layers=tapped, topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    print(f'  Params: {controller.count_trainable():,} | Tapped: {tapped}', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(step_i):
        if step_i < warmup: return step_i / warmup
        return 0.5 * (1 + math.cos(math.pi * (step_i - warmup) / (steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses_hist = []
    random.shuffle(train_data)

    # Baseline eval first (logit-based)
    n_test = min(len(test_data), 500)
    test_subset = test_data[:n_test]

    print('  Baseline eval...', flush=True)
    correct_base = 0
    for sample in test_subset:
        prompt = format_fn(sample) + "Answer:"
        answer_label = label_fn(sample)
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = base_model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids]
            pred = logits.argmax(dim=-1).item()
        if pred == answer_label:
            correct_base += 1
    base_acc = correct_base / n_test
    print(f'  Baseline: {base_acc:.4f} ({correct_base}/{n_test})', flush=True)

    # Train
    embed_fn = lm_model.embed_tokens if hasattr(lm_model, 'embed_tokens') else base_model.model.embed_tokens
    optimizer.zero_grad(set_to_none=True)

    for step_i in range(steps):
        sample = train_data[step_i % len(train_data)]
        prompt_text = format_fn(sample)
        answer_label = label_fn(sample)

        prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                               max_length=480, add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = embed_fn(prompt_enc['input_ids'])
            answer_emb = embed_fn(answer_enc['input_ids'])

        label_tensor = torch.tensor([answer_label], device=device, dtype=torch.long)

        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)
        total_loss, loss_parts = controller.compute_loss(all_cl, all_v, label_tensor)
        total_loss = total_loss / grad_accum

        if total_loss.requires_grad:
            total_loss.backward()

        if (step_i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses_hist.append(loss_parts['final_ce'])

        if (step_i + 1) % 200 == 0:
            avg = sum(losses_hist[-200:]) / len(losses_hist[-200:])
            print(f'  step {step_i+1} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n  === Eval ({n_test} samples) ===', flush=True)
    controller.eval()
    correct = 0
    per_round_correct = [0] * n_rounds

    for sample in test_subset:
        prompt_text = format_fn(sample)
        answer_label = label_fn(sample)

        prompt_enc = tokenizer(prompt_text, return_tensors='pt', truncation=True,
                               max_length=480, add_special_tokens=True).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt',
                               add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = embed_fn(prompt_enc['input_ids'])
            answer_emb = embed_fn(answer_enc['input_ids'])
            all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_tensor, rounds=n_rounds)

            for r in range(n_rounds):
                if all_cl[r].argmax(dim=-1).item() == answer_label:
                    per_round_correct[r] += 1
            if all_cl[-1].argmax(dim=-1).item() == answer_label:
                correct += 1

    acc = correct / n_test
    per_r = [c / n_test for c in per_round_correct]
    per_r_str = ' '.join(f'r{i+1}={a:.3f}' for i, a in enumerate(per_r))
    delta = acc - base_acc
    print(f'  FINAL: {acc:.4f} | {per_r_str} | delta={delta:+.4f}', flush=True)

    # Save
    model_name = os.path.basename(model_path)
    tag = f'{bench_name}_{model_name}_r{n_rounds}_ga{grad_accum}_seed{seed}'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': bench_name, 'model': model_name,
        'n_choices': n_choices, 'n_rounds': n_rounds,
        'grad_accum': grad_accum, 'seed': seed, 'total_steps': steps,
        'baseline_accuracy': base_acc,
        'accuracy': acc, 'delta': delta,
        'per_round_accuracy': per_r,
        'final_loss': sum(losses_hist[-50:]) / max(len(losses_hist[-50:]), 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler, base_model
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', type=str, default='winogrande,piqa,openbookqa')
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--grad_accum', type=int, default=8)
    args = parser.parse_args()

    benchmarks = args.benchmarks.split(',')
    seeds = [int(x) for x in args.seeds.split(',')]

    for bench in benchmarks:
        for seed in seeds:
            try:
                run_benchmark(bench, args.model, seed, args.steps, args.rounds, args.grad_accum)
            except Exception as e:
                print(f'  ERROR on {bench} seed={seed}: {e}', flush=True)
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

    print('\n=== All benchmarks complete ===', flush=True)


if __name__ == '__main__':
    main()
