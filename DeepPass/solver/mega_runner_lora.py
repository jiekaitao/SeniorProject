"""
LoRA baseline for fair comparison with the deliberation controller.

Same parameter budget (~140M), same train/test splits, same benchmarks.
If LoRA beats our controller, we're just a worse fine-tuning method.
If our controller beats LoRA with the same param budget, iterative
computation is actually adding value.

Matching the controller setup:
  - Llama 3.1 8B (Base or Instruct)
  - 8000 training steps
  - Grad accumulation = 16
  - AdamW lr=1e-4, cosine schedule, warmup=200
  - bfloat16
  - Same benchmark loaders from mega_runner_benchmarks.py
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from mega_runner_benchmarks import BENCHMARK_LOADERS, get_choice_tokens

device = torch.device('cuda')


def train_and_eval_lora(base_model, tokenizer, data, n_choices,
                         lora_r, n_rounds_placeholder, total_steps, seed,
                         grad_accum, tag, results_dir, benchmark_name):
    """LoRA fine-tune + evaluate on a reasoning benchmark."""
    random.seed(seed)
    torch.manual_seed(seed)

    choice_ids = get_choice_tokens(tokenizer, n_choices)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    # Apply LoRA to all linear modules (attention + MLP)
    # target_modules matches Llama's linear layer names
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Wrap model with LoRA
    model = get_peft_model(base_model, lora_config)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  LoRA r={lora_r}: {trainable_params/1e6:.1f}M trainable / {total_params/1e6:.1f}M total ({100*trainable_params/total_params:.2f}%)', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05
    )
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    train_data = data['train']
    test_data = data['test'][:500]
    random.shuffle(train_data)

    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    # Baseline (LoRA not yet trained — but LoRA init is zero, so this = zero-shot)
    print(f'  Computing baseline...', flush=True)
    model.eval()
    baseline_correct = 0
    with torch.no_grad():
        for sample in test_data:
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
            if pred == sample['label']:
                baseline_correct += 1
    baseline_acc = baseline_correct / len(test_data)
    print(f'  Baseline: {baseline_acc:.4f} ({baseline_correct}/{len(test_data)})', flush=True)

    # Train
    model.train()
    print(f'  Training {total_steps} steps (GA={grad_accum})...', flush=True)
    for step in range(total_steps):
        sample = train_data[step % len(train_data)]
        text = sample['text'][:1500] + "\nAnswer:"
        answer_label = sample['label']

        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        out = model(enc['input_ids'])
        logits = out.logits[:, -1, choice_ids_t]  # (1, n_choices)

        label_t = torch.tensor([answer_label], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.float(), label_t)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * grad_accum)
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in test_data:
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            if logits.argmax(dim=-1).item() == sample['label']:
                correct += 1
    final_acc = correct / len(test_data)
    delta = final_acc - baseline_acc
    print(f'  LoRA: {final_acc:.4f} (delta={delta:+.4f})', flush=True)

    results = {
        'baseline': {'accuracy': baseline_acc, 'correct': baseline_correct, 'total': len(test_data)},
        'lora_final': {'accuracy': final_acc, 'correct': correct, 'total': len(test_data), 'delta': delta},
    }

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': benchmark_name, 'n_choices': n_choices,
        'method': 'lora', 'lora_r': lora_r,
        'trainable_params_M': trainable_params / 1e6,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {tag}.json ({time.time()-t0:.0f}s)', flush=True)

    # Unwrap LoRA — we need fresh base model for next benchmark
    model = model.unload()
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--benchmarks', type=str, required=True)
    parser.add_argument('--lora_r', type=int, default=64,
                        help='LoRA rank — r=64 gives ~140M trainable params (matches controller)')
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str,
                        default='results/data/lora_baseline')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    print(f'Model loaded.', flush=True)

    benchmarks = args.benchmarks.split(',')
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'

    overall_t0 = time.time()
    for i, bench_name in enumerate(benchmarks):
        bench_name = bench_name.strip()
        print(f'\n{"="*70}', flush=True)
        print(f'[{i+1}/{len(benchmarks)}] LoRA on {bench_name}', flush=True)
        print(f'{"="*70}', flush=True)

        if bench_name not in BENCHMARK_LOADERS:
            print(f'  Unknown benchmark: {bench_name}', flush=True)
            continue

        try:
            data, n_choices = BENCHMARK_LOADERS[bench_name]()
            tag = f'lora_{model_short}_{bench_name}_r{args.lora_r}_{args.total_steps//1000}k_s{args.seed}'

            train_and_eval_lora(
                base_model, tokenizer, data, n_choices,
                lora_r=args.lora_r,
                n_rounds_placeholder=5,
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

    print(f'\n=== All LoRA baselines done in {time.time()-overall_t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
