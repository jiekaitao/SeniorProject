"""
Universal LoRA replay trainer — works with any HuggingFace causal LM.
Adapts the replay experiment to different model sizes and architectures.
"""

import os, sys, time, math, random, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def get_data_stream(tokenizer, seq_len, batch_size):
    general = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                                split='train', streaming=True))
    math_ds = iter(load_dataset('nvidia/OpenMathInstruct-2', split='train', streaming=True))
    token_buffer = []

    def get_text():
        r = random.random()
        try:
            if r < 0.5:
                ex = next(math_ds)
                return f"Problem: {ex.get('problem', '')}\nSolution: {ex.get('generated_solution', '')}"
            else:
                return next(general).get('text', '')
        except (StopIteration, Exception):
            return ''

    while True:
        text = get_text()
        if not text or len(text) < 50:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                  max_length=seq_len * 2)
        tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)
        while len(token_buffer) >= (seq_len + 1) * batch_size:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch.append(chunk)
            t = torch.tensor(batch, dtype=torch.long)
            yield t


def evaluate_ppl(model, tokenizer, device, n=15, seq_len=512, bs=2):
    model.eval()
    ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                           split='train', streaming=True))
    buf = []
    target = (seq_len + 1) * bs * n * 2 + 500
    while len(buf) < target:
        ex = next(ds)
        t = ex.get('text', '')
        if t and len(t) > 50:
            toks = tokenizer.encode(t, add_special_tokens=False, truncation=True,
                                    max_length=seq_len * 2)
            toks.append(tokenizer.eos_token_id)
            buf.extend(toks)

    results = {}
    for adapter_on in [False, True]:
        if adapter_on:
            model.enable_adapter_layers()
            label = 'K=2'
        else:
            model.disable_adapter_layers()
            label = 'K=1'

        total_loss = total_tok = 0
        buf_copy = list(buf)
        with torch.no_grad():
            for _ in range(n):
                batch = []
                for _ in range(bs):
                    batch.append(buf_copy[:seq_len + 1])
                    buf_copy = buf_copy[seq_len:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                out = model(t[:, :-1], labels=t[:, :-1])
                total_loss += out.loss.item() * t[:, 1:].numel()
                total_tok += t[:, 1:].numel()
        results[label] = math.exp(total_loss / max(total_tok, 1))

    model.enable_adapter_layers()
    model.train()
    return results


def get_layer_prefix(model_config):
    """Detect the layer module path for different model architectures."""
    arch = model_config.architectures[0] if model_config.architectures else ""
    if "Llama" in arch or "Mistral" in arch:
        return "model.layers"
    elif "Gemma" in arch:
        return "model.layers"
    elif "GPT" in arch:
        return "transformer.h"
    else:
        return "model.layers"  # default


def get_attn_proj_names(model_config):
    """Detect attention projection names for different architectures."""
    arch = model_config.architectures[0] if model_config.architectures else ""
    if "Llama" in arch or "Mistral" in arch or "Gemma" in arch:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "GPT" in arch:
        return ["c_attn", "c_proj"]
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]


def train(model_path, target_layer_ids=None, lora_rank=32, total_steps=10000,
          seq_len=1024, batch_size=2, lr=3e-4):
    device = torch.device('cuda')

    print(f'Loading {model_path}...', flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
    )
    print(f'  Loaded in {time.time()-t0:.0f}s', flush=True)
    print(f'  Layers: {model.config.num_hidden_layers}', flush=True)
    print(f'  Hidden: {model.config.hidden_size}', flush=True)

    n_layers = model.config.num_hidden_layers
    layer_prefix = get_layer_prefix(model.config)
    proj_names = get_attn_proj_names(model.config)

    # Default: middle 60% of layers
    if target_layer_ids is None:
        start = n_layers // 5
        end = n_layers * 4 // 5
        target_layer_ids = list(range(start, end))

    target_modules = []
    for lid in target_layer_ids:
        for proj in proj_names:
            target_modules.append(f"{layer_prefix}.{lid}.self_attn.{proj}")

    print(f'  Replay layers: {target_layer_ids[0]}-{target_layer_ids[-1]} ({len(target_layer_ids)} layers)', flush=True)
    print(f'  LoRA rank: {lora_rank}', flush=True)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank // 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    warmup = min(500, total_steps // 10)

    def lr_schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    save_dir = f'experiment_a/checkpoints_{os.path.basename(model_path)}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n=== LoRA Replay Training ===', flush=True)
    print(f'  Model: {model_path}', flush=True)
    print(f'  Steps: {total_steps}, LR: {lr}', flush=True)

    data_stream = get_data_stream(tokenizer, seq_len=seq_len, batch_size=batch_size)
    model.train()

    step = 0
    running_loss = 0
    t0 = time.time()
    best_ppl = float('inf')

    for batch in data_stream:
        if step >= total_steps:
            break
        batch = batch.to(device)
        input_ids = batch[:, :-1]

        model.enable_adapter_layers()
        out = model(input_ids, labels=input_ids)
        loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % 50 == 0:
            avg = running_loss / 50
            elapsed = time.time() - t0
            print(f'  step {step:5d} | loss={avg:.4f} | {elapsed:.0f}s', flush=True)
            running_loss = 0

        if step % 500 == 0:
            ppl = evaluate_ppl(model, tokenizer, device, n=15, seq_len=seq_len, bs=batch_size)
            delta = ppl['K=2'] - ppl['K=1']
            print(f'  --- EVAL step {step}: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} '
                  f'(delta={delta:+.2f}) ---', flush=True)
            if ppl['K=2'] < best_ppl:
                best_ppl = ppl['K=2']
                model.save_pretrained(f'{save_dir}/best')
                tokenizer.save_pretrained(f'{save_dir}/best')
                print(f'  --- SAVED best ---', flush=True)

    ppl = evaluate_ppl(model, tokenizer, device, n=30, seq_len=seq_len, bs=batch_size)
    delta = ppl['K=2'] - ppl['K=1']
    print(f'\n=== Complete: PPL K=1={ppl["K=1"]:.2f} K=2={ppl["K=2"]:.2f} (delta={delta:+.2f}) ===',
          flush=True)
    print(f'  Best PPL: {best_ppl:.2f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model path or HF name')
    parser.add_argument('--layers', default=None, help='Comma-separated layer IDs')
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]

    train(args.model, target_layer_ids=layers, lora_rank=args.rank,
          total_steps=args.steps, seq_len=args.seq_len,
          batch_size=args.batch_size, lr=args.lr)
