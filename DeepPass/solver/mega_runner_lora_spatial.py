"""
LoRA baseline on SpatialEval — the critical test.
Our controller gives +39pp on SpatialGrid. Does LoRA match that?
If controller beats LoRA on SpatialEval, we have the "iterative reasoning
helps specifically on spatial tasks" finding.
"""
import os, sys, torch, json, random, math, time, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))

device = torch.device('cuda')
CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def train_and_eval_lora_spatial(base_model, tokenizer, data, train_idx, eval_idx,
                                  lora_r, total_steps, seed, grad_accum,
                                  tag, results_dir, task_name):
    random.seed(seed)
    torch.manual_seed(seed)

    choice_letters = ['A', 'B', 'C', 'D']
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in choice_letters]
    choice_ids_t = torch.tensor(choice_ids, device=device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r,
        lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  LoRA r={lora_r}: {trainable_params/1e6:.1f}M trainable', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # Baseline
    model.eval()
    print(f'  Computing baseline...', flush=True)
    baseline_correct = 0
    with torch.no_grad():
        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
            oracle = sample['oracle_option'].strip().upper()
            al = CHOICE_MAP.get(oracle[0], 0)
            if pred == al: baseline_correct += 1
    baseline_acc = baseline_correct / len(eval_idx)
    print(f'  Baseline: {baseline_acc:.4f}', flush=True)

    # Train
    model.train()
    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(total_steps):
        sample = data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500] + "\nAnswer:"
        oracle = sample['oracle_option'].strip().upper()
        al = CHOICE_MAP.get(oracle[0], 0)
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        out = model(enc['input_ids'])
        logits = out.logits[:, -1, choice_ids_t]
        label_t = torch.tensor([al], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.float(), label_t) / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item() * grad_accum)
        if (step + 1) % 1000 == 0:
            print(f'  step {step+1} | ce={sum(losses[-1000:])/1000:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx in eval_idx:
            sample = data[idx]
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            logits = out.logits[:, -1, choice_ids_t]
            oracle = sample['oracle_option'].strip().upper()
            al = CHOICE_MAP.get(oracle[0], 0)
            if logits.argmax(dim=-1).item() == al: correct += 1
    final_acc = correct / len(eval_idx)
    delta = final_acc - baseline_acc
    print(f'  LoRA: {final_acc:.4f} (delta={delta:+.4f})', flush=True)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'benchmark': task_name, 'method': 'lora', 'lora_r': lora_r,
        'trainable_params_M': trainable_params / 1e6,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'results': {
            'baseline': {'accuracy': baseline_acc, 'correct': baseline_correct, 'total': len(eval_idx)},
            'lora_final': {'accuracy': final_acc, 'correct': correct, 'total': len(eval_idx), 'delta': delta},
        },
    }
    with open(os.path.join(results_dir, f'{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved {tag}.json ({time.time()-t0:.0f}s)', flush=True)
    model = model.unload()
    torch.cuda.empty_cache()
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/Llama-3.1-8B-Instruct')
    parser.add_argument('--tasks', type=str, default='spatialgrid,mazenav,spatialmap')
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str, default='results/data/lora_baseline')
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    print(f'Model loaded.', flush=True)

    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    model_short = 'inst' if 'instruct' in args.model.lower() else 'base'

    overall_t0 = time.time()
    for task in args.tasks.split(','):
        task = task.strip()
        print(f'\n=== LoRA on {task} ===', flush=True)
        data = [s for s in ds if s['id'].startswith(task)]
        random.seed(0)
        indices = list(range(len(data)))
        random.shuffle(indices)
        split = min(1000, len(indices) * 2 // 3)
        train_idx, eval_idx = indices[:split], indices[split:]
        print(f'  {task}: {len(train_idx)} train, {len(eval_idx)} eval', flush=True)

        tag = f'lora_{model_short}_{task}_r{args.lora_r}_{args.total_steps//1000}k_s{args.seed}'
        try:
            train_and_eval_lora_spatial(
                base_model, tokenizer, data, train_idx, eval_idx,
                lora_r=args.lora_r, total_steps=args.total_steps,
                seed=args.seed, grad_accum=args.grad_accum,
                tag=tag, results_dir=args.results_dir, task_name=task,
            )
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    print(f'\n=== Done in {time.time()-overall_t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
