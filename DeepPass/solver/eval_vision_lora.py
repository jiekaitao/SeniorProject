"""LoRA baseline on VSR (vision-language, PaLiGemma2).

Our controller gives +33pp on PaLiGemma VSR. If LoRA with matched params
also gives +33pp, the controller's vision advantage is just fine-tuning.
If LoRA gives less, the controller wins on vision too.
"""
import os, sys, torch, json, random, math, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))

device = torch.device('cuda')


def load_image_safe(url_or_path, max_retries=2):
    for attempt in range(max_retries):
        try:
            if url_or_path.startswith('http'):
                response = requests.get(url_or_path, timeout=10)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                img = Image.open(url_or_path).convert('RGB')
            return img
        except Exception:
            if attempt == max_retries - 1:
                return None
    return None


def get_binary_choice_ids(tokenizer):
    true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]
    return [true_id, false_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/full/paligemma2-10b')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--results_dir', type=str, default='results/data/vision_lora')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    print(f'Loading {args.model}...', flush=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    print('Model loaded.', flush=True)

    # LoRA config - target LM attention + MLP projections (Gemma2 naming)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r,
        lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'LoRA r={args.lora_r}: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.0f}M total', flush=True)

    from datasets import load_dataset
    ds = load_dataset('cambridgeltl/vsr_random')
    train_data = list(ds['train'])
    eval_data = list(ds['test'])
    random.shuffle(train_data)

    choice_ids = get_binary_choice_ids(tokenizer)
    choice_ids_t = torch.tensor(choice_ids, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(s):
        if s < warmup: return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / (args.total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # Baseline — 200 samples
    model.eval()
    baseline_correct = 0
    baseline_total = 0
    print('Baseline...', flush=True)
    with torch.no_grad():
        for sample in eval_data[:200]:
            img = load_image_safe(sample.get('image_link', sample.get('image', '')))
            if img is None: continue
            caption = sample['caption']; label = sample['label']
            prompt = f'Is this statement about the image true or false? "{caption}"\nAnswer:'
            inputs = processor(text=prompt, images=img, return_tensors='pt').to(device)
            out = model(**inputs)
            logits = out.logits[:, -1, choice_ids_t]
            pred = logits.argmax(dim=-1).item()
            if pred == (0 if label else 1): baseline_correct += 1
            baseline_total += 1
    baseline_acc = baseline_correct / max(baseline_total, 1)
    print(f'Baseline: {baseline_acc:.4f} ({baseline_correct}/{baseline_total})', flush=True)

    # Train
    model.train()
    t0 = time.time()
    losses = []
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.total_steps):
        sample = train_data[step % len(train_data)]
        img = load_image_safe(sample.get('image_link', sample.get('image', '')))
        if img is None:
            skipped += 1; continue
        caption = sample['caption']; label = sample['label']
        prompt = f'Is this statement about the image true or false? "{caption}"\nAnswer:'
        try:
            inputs = processor(text=prompt, images=img, return_tensors='pt').to(device)
            out = model(**inputs)
            logits = out.logits[:, -1, choice_ids_t]
            target = torch.tensor([0 if label else 1], device=device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), target) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item() * args.grad_accum)
        except Exception as e:
            skipped += 1
            if skipped < 5:
                print(f'skip {step}: {e}', flush=True)
        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / max(len(losses[-500:]), 1)
            print(f'step {step+1}/{args.total_steps} | ce={avg:.4f} | skip={skipped} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    model.eval()
    print('Eval...', flush=True)
    correct = 0; total = 0; eval_skip = 0
    with torch.no_grad():
        for sample in eval_data:
            img = load_image_safe(sample.get('image_link', sample.get('image', '')))
            if img is None: eval_skip += 1; continue
            caption = sample['caption']; label = sample['label']
            prompt = f'Is this statement about the image true or false? "{caption}"\nAnswer:'
            try:
                inputs = processor(text=prompt, images=img, return_tensors='pt').to(device)
                out = model(**inputs)
                logits = out.logits[:, -1, choice_ids_t]
                pred = logits.argmax(dim=-1).item()
                if pred == (0 if label else 1): correct += 1
                total += 1
            except Exception:
                eval_skip += 1
    final_acc = correct / max(total, 1)
    print(f'Final: {final_acc:.4f} ({correct}/{total}), skipped={eval_skip}', flush=True)

    tag = args.tag or f'vision_lora_paligemma_r{args.lora_r}_{args.total_steps//1000}k_s{args.seed}'
    os.makedirs(args.results_dir, exist_ok=True)
    out = {
        'tag': tag, 'method': 'lora', 'model': args.model,
        'lora_r': args.lora_r, 'trainable_params_M': trainable_params/1e6,
        'total_steps': args.total_steps, 'seed': args.seed, 'grad_accum': args.grad_accum,
        'train_skipped': skipped,
        'results': {
            'baseline': {'accuracy': baseline_acc, 'correct': baseline_correct, 'total': baseline_total},
            'lora_final': {'accuracy': final_acc, 'correct': correct, 'total': total,
                           'skipped': eval_skip, 'delta': final_acc - baseline_acc},
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(args.results_dir, f'{tag}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved {tag}.json', flush=True)


if __name__ == '__main__':
    main()
