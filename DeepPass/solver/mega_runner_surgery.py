"""
Mazenav surgery runner — falsify the "Mazenav is secretly pattern-matching"
hypothesis by removing / corrupting the X path markers and re-running the
same controller-vs-LoRA comparison.

This script combines:
  - Surgery dataset loaders from `spatialeval_surgery.py`
    ({no_x, distract, unsolved})
  - Controller training via FlexMidLayerDeliberation from
    `mega_runner_benchmarks.py` (proven 140M-param config: lowrank writer
    rank=64, d_state=512, n_slots=8, mid-layer injection at L12, 5 rounds
    train, K-scaling eval at 3/5/8)
  - LoRA training via the same setup as `mega_runner_lora_spatial.py`
    (r=64, target_modules = {q,k,v,o,gate,up,down}_proj, alpha=2r,
    dropout=0.05)

Hyperparameters match the original Mazenav runs: lr=1e-4, wd=0.05, cosine
schedule with 200-step warmup, GA=16, bf16, 8000 steps, eval cap 500.

Usage:
    python solver/mega_runner_surgery.py \
        --method {controller|lora} \
        --model models/full/Llama-3.1-8B[-Instruct] \
        --variant {no_x|distract|unsolved} \
        --inject_layer 12 \
        --n_rounds 5 \
        --total_steps 8000 \
        --seed 42 \
        --grad_accum 16 \
        --results_dir results/data/surgery

Tag format: `surgery_{method}_{model_short}_{variant}_s{seed}.json`
where model_short is 'inst' or 'base'.
"""
import os
import sys
import math
import json
import time
import random
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault('HF_HOME', '/blue/cis4914/jietao/hf_cache')
sys.path.insert(0, os.path.dirname(__file__))

from mega_runner_benchmarks import FlexMidLayerDeliberation
from spatialeval_surgery import SURGERY_LOADERS, CHOICE_MAP


device = torch.device('cuda')


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _choice_ids(tokenizer, n_choices=4):
    letters = ['A', 'B', 'C', 'D', 'E'][:n_choices]
    ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in letters]
    return torch.tensor(ids, device=device)


def _oracle_label(sample):
    """Pull the label index from either a 'label' field or 'oracle_option'."""
    if 'label' in sample:
        return int(sample['label'])
    oracle = sample['oracle_option'].strip().upper()
    return CHOICE_MAP.get(oracle[:1], 0)


def _cosine_lr(step, warmup, total_steps):
    if step < warmup:
        return step / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))


# --------------------------------------------------------------------------
# Controller training
# --------------------------------------------------------------------------

def train_and_eval_controller(base_model, tokenizer, lm_model, data, n_choices,
                              inject_layer, n_rounds, total_steps, seed,
                              grad_accum, tag, results_dir, variant):
    random.seed(seed)
    torch.manual_seed(seed)

    choice_ids_t = _choice_ids(tokenizer, n_choices)

    controller = FlexMidLayerDeliberation(
        frozen_llm=base_model, inject_layer=inject_layer, n_choices=n_choices,
        rank=64, d_state=512, n_slots=8, tapped_layers=(8, 16, 24), topk_vocab=64,
    ).to(device=device, dtype=torch.bfloat16)

    n_trainable = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    print(f'  Controller: {n_trainable/1e6:.1f}M trainable, {n_choices} choices, L{inject_layer}, {n_rounds} rounds', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05,
    )
    warmup = 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_lr(s, warmup, total_steps)
    )

    train_data = list(data['train'])
    test_data = data['test'][:500]
    random.shuffle(train_data)

    t0 = time.time()

    # Baseline
    print(f'  Computing baseline on {len(test_data)} eval samples...', flush=True)
    base_correct = 0
    for sample in test_data:
        text = sample['text'][:1500] + "\nAnswer:"
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        with torch.no_grad():
            out = base_model(enc['input_ids'])
            pred = out.logits[:, -1, choice_ids_t].argmax(dim=-1).item()
        if pred == _oracle_label(sample):
            base_correct += 1
    base_acc = base_correct / len(test_data)
    print(f'  Baseline: {base_acc:.4f} ({base_correct}/{len(test_data)})', flush=True)

    # Train
    print(f'  Training {total_steps} steps, GA={grad_accum}...', flush=True)
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        sample = train_data[step % len(train_data)]
        text = sample['text'][:1500]
        label = _oracle_label(sample)

        prompt_enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
        answer_enc = tokenizer("\nAnswer:", return_tensors='pt', add_special_tokens=False).to(device)

        with torch.no_grad():
            prompt_emb = lm_model.embed_tokens(prompt_enc['input_ids'])
            answer_emb = lm_model.embed_tokens(answer_enc['input_ids'])

        label_t = torch.tensor([label], device=device, dtype=torch.long)
        all_cl, all_v = controller(prompt_emb, answer_emb, choice_ids_t, rounds=n_rounds)
        loss, lp = controller.compute_loss(all_cl, all_v, label_t)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in controller.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(lp['final_ce'])
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval (K-scaling)
    controller.eval()
    results = {'baseline': {
        'accuracy': base_acc, 'correct': base_correct, 'total': len(test_data)
    }}

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
                if all_cl[-1].argmax(dim=-1).item() == _oracle_label(sample):
                    correct += 1
        acc = correct / len(test_data)
        delta = acc - base_acc
        results[f'rounds={er}'] = {
            'accuracy': acc, 'correct': correct, 'total': len(test_data),
            'delta': delta,
        }
        print(f'  rounds={er}: {acc:.4f} (delta={delta:+.4f})', flush=True)

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'controller', 'benchmark': 'mazenav', 'variant': variant,
        'n_choices': n_choices, 'inject_layer': inject_layer, 'n_rounds': n_rounds,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'trainable_params_M': n_trainable / 1e6,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(results_dir, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {out_path} ({time.time()-t0:.0f}s)', flush=True)

    del controller, optimizer, scheduler
    torch.cuda.empty_cache()
    return result_data


# --------------------------------------------------------------------------
# LoRA training
# --------------------------------------------------------------------------

def train_and_eval_lora(base_model, tokenizer, data, n_choices,
                        total_steps, seed, grad_accum, tag, results_dir, variant,
                        lora_r=64):
    from peft import LoraConfig, get_peft_model, TaskType

    random.seed(seed)
    torch.manual_seed(seed)

    choice_ids_t = _choice_ids(tokenizer, n_choices)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r,
        lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  LoRA r={lora_r}: {n_trainable/1e6:.1f}M trainable', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.05,
    )
    warmup = 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_lr(s, warmup, total_steps)
    )

    train_data = list(data['train'])
    test_data = data['test'][:500]
    random.shuffle(train_data)

    t0 = time.time()

    # Baseline
    model.eval()
    print(f'  Computing baseline on {len(test_data)} eval samples...', flush=True)
    base_correct = 0
    with torch.no_grad():
        for sample in test_data:
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            pred = out.logits[:, -1, choice_ids_t].argmax(dim=-1).item()
            if pred == _oracle_label(sample):
                base_correct += 1
    base_acc = base_correct / len(test_data)
    print(f'  Baseline: {base_acc:.4f} ({base_correct}/{len(test_data)})', flush=True)

    # Train
    model.train()
    print(f'  Training {total_steps} steps, GA={grad_accum}...', flush=True)
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        sample = train_data[step % len(train_data)]
        text = sample['text'][:1500] + "\nAnswer:"
        label = _oracle_label(sample)
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)

        out = model(enc['input_ids'])
        logits = out.logits[:, -1, choice_ids_t]
        label_t = torch.tensor([label], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits.float(), label_t) / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * grad_accum)
        if (step + 1) % 1000 == 0:
            avg = sum(losses[-1000:]) / 1000
            print(f'  step {step+1}/{total_steps} | ce={avg:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Final eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in test_data:
            text = sample['text'][:1500] + "\nAnswer:"
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=1900).to(device)
            out = model(enc['input_ids'])
            pred = out.logits[:, -1, choice_ids_t].argmax(dim=-1).item()
            if pred == _oracle_label(sample):
                correct += 1
    final_acc = correct / len(test_data)
    delta = final_acc - base_acc
    print(f'  LoRA: {final_acc:.4f} (delta={delta:+.4f})', flush=True)

    results = {
        'baseline': {'accuracy': base_acc, 'correct': base_correct, 'total': len(test_data)},
        'lora_final': {'accuracy': final_acc, 'correct': correct, 'total': len(test_data),
                       'delta': delta},
    }

    os.makedirs(results_dir, exist_ok=True)
    result_data = {
        'tag': tag, 'method': 'lora', 'benchmark': 'mazenav', 'variant': variant,
        'n_choices': n_choices, 'lora_r': lora_r,
        'total_steps': total_steps, 'seed': seed, 'grad_accum': grad_accum,
        'trainable_params_M': n_trainable / 1e6,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = os.path.join(results_dir, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'  Saved: {out_path} ({time.time()-t0:.0f}s)', flush=True)

    # Strip LoRA adapters so subsequent work (if any) sees the frozen base
    try:
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        return result_data, base
    except Exception:
        torch.cuda.empty_cache()
        return result_data, base_model


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _model_short(model_path):
    return 'inst' if 'instruct' in model_path.lower() else 'base'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['controller', 'lora'], required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--variant', choices=list(SURGERY_LOADERS), required=True)
    parser.add_argument('--inject_layer', type=int, default=12,
                        help='Controller mid-layer injection point (ignored for lora).')
    parser.add_argument('--n_rounds', type=int, default=5,
                        help='Controller training rounds (ignored for lora).')
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str, default='results/data/surgery')
    parser.add_argument('--lora_r', type=int, default=64)
    args = parser.parse_args()

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    # Freeze base; LoRA / controller will add trainable params on top.
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model
    print(f'Model loaded.', flush=True)

    print(f'Loading mazenav variant "{args.variant}"...', flush=True)
    data, n_choices = SURGERY_LOADERS[args.variant](seed=0)
    print(f'  {len(data["train"])} train, {len(data["test"])} eval, {n_choices} choices', flush=True)

    model_short = _model_short(args.model)
    tag = f'surgery_{args.method}_{model_short}_{args.variant}_s{args.seed}'

    print(f'\n=== {tag} ===', flush=True)

    try:
        if args.method == 'controller':
            train_and_eval_controller(
                base_model, tokenizer, lm_model, data, n_choices,
                inject_layer=args.inject_layer,
                n_rounds=args.n_rounds,
                total_steps=args.total_steps,
                seed=args.seed,
                grad_accum=args.grad_accum,
                tag=tag,
                results_dir=args.results_dir,
                variant=args.variant,
            )
        else:
            train_and_eval_lora(
                base_model, tokenizer, data, n_choices,
                total_steps=args.total_steps,
                seed=args.seed,
                grad_accum=args.grad_accum,
                tag=tag,
                results_dir=args.results_dir,
                variant=args.variant,
                lora_r=args.lora_r,
            )
    except Exception as e:
        print(f'  ERROR: {e}', flush=True)
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        raise

    print(f'\n=== Done ===', flush=True)


if __name__ == '__main__':
    main()
