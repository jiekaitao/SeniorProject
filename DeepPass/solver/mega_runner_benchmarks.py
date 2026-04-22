"""
Mega runner for MAJOR reasoning benchmarks using our BEST architecture.

Previous benchmark runs failed because they used the old architecture:
  - No lowrank writer (high variance)
  - No gradient accumulation (noisy updates)
  - Only 2000 steps (undertrained)
  - Base RecurrentDeliberation (no mid-layer injection)

This script uses the PROVEN config:
  - MidLayerDeliberation + lowrank writer (rank=64)
  - GA=16 (std 12.8% → 0.3%)
  - 8000 steps (optimal for SpatialGrid)
  - Mid-layer injection at L12
  - 5 rounds training, K-scaling eval at 3/5/8

Supported benchmarks:
  - arc_challenge: 4-way science reasoning (AI2 ARC-Challenge)
  - hellaswag: 4-way commonsense completion
  - winogrande: 2-way coreference resolution
  - boolq: 2-way yes/no reasoning
  - commonsenseqa: 5-way commonsense reasoning
  - openbookqa: 4-way open-book science
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from recurrent_deliberation import RecurrentDeliberation, RMSNorm
from eval_deliberation_creative import MidLayerDeliberation, LowrankDeliberation

device = torch.device('cuda')


class FlexMidLayerDeliberation(MidLayerDeliberation):
    """MidLayerDeliberation that handles variable choice counts (2-5)."""

    def __init__(self, frozen_llm, n_choices=4, **kwargs):
        super().__init__(frozen_llm, **kwargs)
        self.n_choices = n_choices

        # Rebuild read_proj and verifier for correct dimensions
        n_tap = len(self.tapped_list)
        # read: tapped_pools (n_tap * d_model) + think_h (n_slots * d_model) + choices + entropy + margin
        read_dim = n_tap * self.d_model + self.n_slots * self.d_model + n_choices + 2
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, self.n_slots * (kwargs.get('d_state', 512))),
        )
        self.verifier = nn.Sequential(
            nn.Linear(n_tap * self.d_model + self.n_slots * self.d_model + n_choices, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, prompt_emb, answer_emb, choice_ids, rounds=2):
        B = prompt_emb.shape[0]
        z = self.z0.expand(B, -1, -1).clone()
        nc = choice_ids.shape[0]

        all_choice_logits = []
        all_verify = []

        for r in range(rounds):
            thought_emb = self.latent_to_thought_embs(z)
            logits, think_h, tapped_pools = self.forward_frozen_round(
                prompt_emb, thought_emb.to(prompt_emb.dtype), answer_emb
            )
            ans_logits = logits[:, -1, choice_ids]  # (B, nc)
            all_choice_logits.append(ans_logits)

            # Build features
            dtype = think_h.dtype
            probs = ans_logits.float().softmax(dim=-1).to(dtype)
            entropy = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1, keepdim=True).to(dtype)
            top2 = probs.float().topk(min(2, nc), dim=-1).values
            margin = (top2[:, :1] - top2[:, 1:2]).to(dtype) if top2.shape[-1] >= 2 else top2[:, :1].to(dtype)

            feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [probs, entropy, margin],
                dim=-1
            )

            verify_feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [probs],
                dim=-1
            )
            verify = self.verifier(verify_feat)
            all_verify.append(verify)

            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, self.n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        return all_choice_logits, all_verify


# ============================================================
# Benchmark loaders
# ============================================================

def load_arc_challenge():
    from datasets import load_dataset
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
    # Filter to 4-choice only
    train = [s for s in ds['train'] if len(s['choices']['text']) == 4]
    test = [s for s in ds['test'] if len(s['choices']['text']) == 4]
    formatted = {'train': [], 'test': []}
    for split_name, split_data in [('train', train), ('test', test)]:
        for s in split_data:
            q = s['question']
            choices = s['choices']['text']
            labels = s['choices']['label']
            text = q + '\n' + '\n'.join(f'{l}. {c}' for l, c in zip(labels, choices))
            answer = s['answerKey']
            label_idx = labels.index(answer) if answer in labels else 0
            formatted[split_name].append({'text': text, 'label': label_idx, 'n_choices': 4})
    print(f'  ARC-Challenge: {len(formatted["train"])} train, {len(formatted["test"])} test')
    return formatted, 4


def load_hellaswag():
    from datasets import load_dataset
    ds = load_dataset('Rowan/hellaswag')
    formatted = {'train': [], 'test': []}
    for split_name, hf_split in [('train', 'train'), ('test', 'validation')]:
        data = ds[hf_split]
        for s in data:
            endings = s['endings']
            if len(endings) != 4:
                continue
            ctx = s['ctx']
            text = ctx + '\n' + '\n'.join(f'{chr(65+i)}. {e}' for i, e in enumerate(endings))
            label = int(s['label'])
            formatted[split_name].append({'text': text, 'label': label, 'n_choices': 4})
    print(f'  HellaSwag: {len(formatted["train"])} train, {len(formatted["test"])} test')
    return formatted, 4


def load_winogrande():
    from datasets import load_dataset
    ds = load_dataset('allenai/winogrande', 'winogrande_xl')
    formatted = {'train': [], 'test': []}
    for split_name, hf_split in [('train', 'train'), ('test', 'validation')]:
        for s in ds[hf_split]:
            text = s['sentence'] + '\nA. ' + s['option1'] + '\nB. ' + s['option2']
            label = int(s['answer']) - 1  # 1-indexed → 0-indexed
            formatted[split_name].append({'text': text, 'label': label, 'n_choices': 2})
    print(f'  WinoGrande: {len(formatted["train"])} train, {len(formatted["test"])} test')
    return formatted, 2


def load_boolq():
    from datasets import load_dataset
    ds = load_dataset('google/boolq')
    formatted = {'train': [], 'test': []}
    for split_name, hf_split in [('train', 'train'), ('test', 'validation')]:
        for s in ds[hf_split]:
            text = s['passage'] + '\nQuestion: ' + s['question'] + '?\nA. True\nB. False'
            label = 0 if s['answer'] else 1  # True=A=0, False=B=1
            formatted[split_name].append({'text': text, 'label': label, 'n_choices': 2})
    print(f'  BoolQ: {len(formatted["train"])} train, {len(formatted["test"])} test')
    return formatted, 2


def load_commonsenseqa():
    from datasets import load_dataset
    ds = load_dataset('tau/commonsense_qa')
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    formatted = {'train': [], 'test': []}
    for split_name, hf_split in [('train', 'train'), ('test', 'validation')]:
        for s in ds[hf_split]:
            q = s['question']
            choices = s['choices']['text']
            labels = s['choices']['label']
            text = q + '\n' + '\n'.join(f'{l}. {c}' for l, c in zip(labels, choices))
            answer = s['answerKey']
            label_idx = label_map.get(answer, 0)
            formatted[split_name].append({'text': text, 'label': label_idx, 'n_choices': 5})
    print(f'  CommonsenseQA: {len(formatted["train"])} train, {len(formatted["test"])} test')
    return formatted, 5


def load_openbookqa():
    from datasets import load_dataset
    ds = load_dataset('allenai/openbookqa', 'main')
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    formatted = {'train': [], 'test': []}
    for split_name, hf_split in [('train', 'train'), ('test', 'test')]:
        for s in ds[hf_split]:
            q = s['question_stem']
            choices = s['choices']['text']
            labels = s['choices']['label']
            text = q + '\n' + '\n'.join(f'{l}. {c}' for l, c in zip(labels, choices))
            answer = s['answerKey']
            label_idx = label_map.get(answer, 0)
            formatted[split_name].append({'text': text, 'label': label_idx, 'n_choices': 4})
    print(f'  OpenBookQA: {len(formatted["train"])} train, {len(formatted["test"])} test')
    return formatted, 4


BENCHMARK_LOADERS = {
    'arc_challenge': load_arc_challenge,
    'hellaswag': load_hellaswag,
    'winogrande': load_winogrande,
    'boolq': load_boolq,
    'commonsenseqa': load_commonsenseqa,
    'openbookqa': load_openbookqa,
}


def get_choice_tokens(tokenizer, n_choices):
    """Get token IDs for choices: A, B, C, D, E (or True/False for 2-choice)."""
    choice_letters = ['A', 'B', 'C', 'D', 'E'][:n_choices]
    ids = []
    for c in choice_letters:
        toks = tokenizer.encode(f" {c}", add_special_tokens=False)
        ids.append(toks[0])
    return ids


def train_and_eval_benchmark(base_model, tokenizer, lm_model, data, n_choices,
                              inject_layer, n_rounds, total_steps, seed,
                              grad_accum, tag, results_dir, benchmark_name):
    """Train and evaluate on a reasoning benchmark using best architecture."""
    random.seed(seed)
    torch.manual_seed(seed)

    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    # Use our BEST architecture
    controller = FlexMidLayerDeliberation(
        frozen_llm=base_model, inject_layer=inject_layer, n_choices=n_choices,
        rank=64, d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    param_count = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    print(f'  Controller: {param_count/1e6:.1f}M params, {n_choices} choices', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    train_data = data['train']
    test_data = data['test'][:500]  # Cap eval at 500
    random.shuffle(train_data)

    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    # Baseline first
    print(f'  Computing baseline...', flush=True)
    baseline_correct = 0
    for sample in test_data:
        text = sample['text'][:1500] + "\nAnswer:"
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        with torch.no_grad():
            out = base_model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
        if pred == sample['label']:
            baseline_correct += 1
    baseline_acc = baseline_correct / len(test_data)
    print(f'  Baseline: {baseline_acc:.4f} ({baseline_correct}/{len(test_data)})', flush=True)

    # Train
    print(f'  Training {total_steps} steps (GA={grad_accum})...', flush=True)
    for step in range(total_steps):
        sample = train_data[step % len(train_data)]
        text = sample['text'][:1500]
        answer_label = sample['label']

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_t = torch.tensor([answer_label], device=device, dtype=torch.long)
        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t, rounds=n_rounds)
        loss, lp = controller.compute_loss(all_cl, all_v, label_t)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(lp['final_ce'])
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval with K-scaling
    controller.eval()
    results = {'baseline': {'accuracy': baseline_acc, 'correct': baseline_correct, 'total': len(test_data)}}

    for er in [3, 5, 8]:
        correct = 0
        for sample in test_data:
            text = sample['text'][:1500]
            prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)
            with torch.no_grad():
                prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
                answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])
                all_cl, _ = controller(prompt_emb, answer_emb, choice_ids_t, rounds=er)
                if all_cl[-1].argmax(dim=-1).item() == sample['label']:
                    correct += 1
        acc = correct / len(test_data)
        delta = acc - baseline_acc
        results[f'rounds={er}'] = {
            'accuracy': acc, 'correct': correct, 'total': len(test_data),
            'delta': delta,
        }
        print(f'  rounds={er}: {acc:.4f} (delta={delta:+.4f})', flush=True)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': benchmark_name, 'n_choices': n_choices,
        'inject_layer': inject_layer, 'n_rounds': n_rounds,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--benchmarks', type=str, required=True,
                        help='Comma-separated benchmark names')
    parser.add_argument('--inject_layer', type=int, default=12)
    parser.add_argument('--n_rounds', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str,
                        default='results/data/benchmarks')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model
    print(f'Model loaded.', flush=True)

    benchmarks = args.benchmarks.split(',')
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'

    overall_t0 = time.time()
    for i, bench_name in enumerate(benchmarks):
        bench_name = bench_name.strip()
        print(f'\n{"="*70}', flush=True)
        print(f'[{i+1}/{len(benchmarks)}] {bench_name}', flush=True)
        print(f'{"="*70}', flush=True)

        if bench_name not in BENCHMARK_LOADERS:
            print(f'  Unknown benchmark: {bench_name}', flush=True)
            continue

        try:
            data, n_choices = BENCHMARK_LOADERS[bench_name]()
            tag = f'bench_{model_short}_{bench_name}_L{args.inject_layer}_{args.total_steps//1000}k_s{args.seed}'

            train_and_eval_benchmark(
                base_model, tokenizer, lm_model, data, n_choices,
                inject_layer=args.inject_layer,
                n_rounds=args.n_rounds,
                total_steps=args.total_steps,
                seed=args.seed,
                grad_accum=args.grad_accum,
                tag=tag,
                results_dir=args.results_dir,
                benchmark_name=bench_name,
            )
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    print(f'\n=== All benchmarks done in {time.time()-overall_t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
