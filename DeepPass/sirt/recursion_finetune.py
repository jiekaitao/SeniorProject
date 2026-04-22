"""
Universal Recursion Fine-Tune: Convert any pre-trained model to SIRT-style

Takes a pre-trained model and teaches it recursion via:
1. Designate middle layers as recursive core
2. Add LayerIdxWrapper for KV cache
3. Fine-tune with Stage 2 curriculum (K=1-3) then Stage 3 adaptive halting
4. Only train the recursive core layers (freeze prelude + coda)

Usage:
    python recursion_finetune.py --model <hf_id_or_path> --name <name> \
        --core_start 10 --core_end 16 --max_steps 2000
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoModelForCausalLM, AutoTokenizer


class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None

    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'):
            self.layer.self_attn.layer_idx = self.new_layer_idx
        try:
            return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None:
                self.layer.self_attn.layer_idx = self.orig_attn

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


def build_recursive_model(model, core_start, core_end, K):
    """
    Convert model to recursive: duplicate core layers K times.
    Prelude: layers [0, core_start)
    Core: layers [core_start, core_end) × K repetitions
    Coda: layers [core_end, N)
    """
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model

    original_layers = list(inner.layers)
    N = len(original_layers)

    # Build order: prelude + core*K + coda
    order = list(range(core_start))
    for _ in range(K):
        order.extend(range(core_start, core_end))
    order.extend(range(core_end, N))

    # Track duplicates
    seen = set()
    is_dup = []
    for idx in order:
        is_dup.append(idx in seen)
        seen.add(idx)

    # Build new layer list
    new_layers = []
    for phys_idx, (orig_idx, dup) in enumerate(zip(order, is_dup)):
        layer = original_layers[orig_idx]
        if dup:
            new_layers.append(LayerIdxWrapper(layer, phys_idx))
        else:
            layer.layer_idx = phys_idx
            if hasattr(layer, 'self_attn'):
                layer.self_attn.layer_idx = phys_idx
            new_layers.append(layer)

    inner.layers = nn.ModuleList(new_layers)

    # Update configs
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None:
            continue
        if hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = len(new_layers)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            orig_types = cfg.layer_types
            if len(orig_types) >= N:
                cfg.layer_types = [orig_types[idx] for idx in order]
            else:
                cfg.layer_types = [orig_types[idx % len(orig_types)] for idx in order]
        if hasattr(cfg, 'use_cache'):
            cfg.use_cache = True
    # Also update inner model's config if separate
    inner_obj = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner_obj, 'language_model'):
        inner_obj = inner_obj.language_model
    if hasattr(inner_obj, 'config'):
        if hasattr(inner_obj.config, 'num_hidden_layers'):
            inner_obj.config.num_hidden_layers = len(new_layers)
        if hasattr(inner_obj.config, 'layer_types') and inner_obj.config.layer_types:
            orig_types = inner_obj.config.layer_types
            inner_obj.config.layer_types = [orig_types[idx] for idx in order]

    return original_layers, N, len(new_layers)


def restore_model(model, original_layers, N):
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    inner.layers = nn.ModuleList(original_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = N
    for i, l in enumerate(original_layers):
        l.layer_idx = i
        if hasattr(l, 'self_attn'):
            l.self_attn.layer_idx = i


def evaluate_model(model, tokenizer, device, K, tag=""):
    """Quick dual-probe evaluation."""
    eq_all = _load_questions()

    def gen(prompt):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        # Remove token_type_ids if present (Gemma3 requires it during training but not generation)
        gen_kwargs = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            try:
                out = model.generate(**gen_kwargs, max_new_tokens=64, do_sample=False, use_cache=True)
            except Exception:
                out = model.generate(**gen_kwargs, max_new_tokens=64, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gen_long(prompt):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        gen_kwargs = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            try:
                out = model.generate(**gen_kwargs, max_new_tokens=128, do_sample=False, use_cache=True)
            except Exception:
                out = model.generate(**gen_kwargs, max_new_tokens=128, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def finetune_recursion(model, tokenizer, device, data_dir, core_start, core_end,
                       max_steps=2000, lr=1e-5, batch_size=2, seq_len=2048):
    """
    Stage 2+3 fine-tune: teach the model to use recursion.
    Uses on-the-fly tokenization with the MODEL'S OWN tokenizer (fixes vocab mismatch).
    Only trains core layers with very low LR to avoid catastrophic forgetting.
    """
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    original_layers = list(inner.layers)
    N = len(original_layers)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the core layers
    for i in range(core_start, core_end):
        for param in original_layers[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)', flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Use streaming text data with the model's own tokenizer (fixes vocab mismatch)
    from datasets import load_dataset
    print(f'  Loading fineweb-edu stream...', flush=True)
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                      split='train', streaming=True)

    # Training loop
    model.train()
    step = 0
    running_loss = 0
    t0 = time.time()
    token_buffer = []

    for example in ds:
        if step >= max_steps:
            break

        text = example.get('text', '')
        if not text or len(text) < 100:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len * 4)
        token_buffer.extend(tokens)
        if tokenizer.eos_token_id is not None:
            token_buffer.append(tokenizer.eos_token_id)

        while len(token_buffer) >= (seq_len + 1) * batch_size and step < max_steps:
            # Build batch
            batch_tokens = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch_tokens.append(chunk)

            batch = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            # Random K for curriculum (mostly K=1-2 to be gentle)
            K = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]

            # Apply recursion
            orig_layers, orig_N, new_N = build_recursive_model(model, core_start, core_end, K)

            try:
                # Handle Gemma3 token_type_ids requirement
                fwd_kwargs = {'input_ids': input_ids, 'use_cache': False}
                if hasattr(model.config, 'model_type') and 'gemma' in str(model.config.model_type):
                    fwd_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
                outputs = model(**fwd_kwargs)
                logits = outputs.logits

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # tighter clipping
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()
            except Exception as e:
                print(f'  step {step} ERROR: {e}', flush=True)
                optimizer.zero_grad(set_to_none=True)

            # Restore
            restore_model(model, orig_layers, orig_N)

            step += 1

            if step % 100 == 0:
                avg_loss = running_loss / min(100, step)
                elapsed = time.time() - t0
                print(f'  step {step:5d} | loss {avg_loss:.4f} | K={K} | {elapsed:.0f}s', flush=True)
                running_loss = 0

    print(f'  Training done: {step} steps, {time.time()-t0:.0f}s', flush=True)

    # Freeze all for evaluation
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--data_dir', default='sirt/data')
    parser.add_argument('--core_start', type=int, required=True)
    parser.add_argument('--core_end', type=int, required=True)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=2048)
    args = parser.parse_args()

    SAVE_DIR = f'sirt/recursion_ft/{args.name}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cuda')

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )

    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    N = len(inner.layers)
    print(f'Loaded: {N} layers, core=[{args.core_start},{args.core_end})', flush=True)

    # ======================================================================
    # 1. Baseline (no recursion)
    # ======================================================================
    print('\n=== Baseline (K=1) ===', flush=True)
    baseline = evaluate_model(model, tokenizer, device, 1, 'baseline')

    # ======================================================================
    # 2. Evaluate with recursion K=2 (no fine-tune, just duplicate)
    # ======================================================================
    print('\n=== Pre-finetune Recursion (K=2) ===', flush=True)
    orig, orig_N, new_N = build_recursive_model(model, args.core_start, args.core_end, 2)
    pre_ft_k2 = evaluate_model(model, tokenizer, device, 2, 'pre-ft K=2')
    restore_model(model, orig, orig_N)

    # ======================================================================
    # 3. Fine-tune with recursion curriculum
    # ======================================================================
    print('\n=== Recursion Fine-Tune ===', flush=True)
    finetune_recursion(
        model, tokenizer, device, args.data_dir,
        args.core_start, args.core_end,
        max_steps=args.max_steps, lr=args.lr,
        batch_size=args.batch_size, seq_len=args.seq_len,
    )

    # ======================================================================
    # 4. Evaluate after fine-tune at K=1, K=2, K=3
    # ======================================================================
    print('\n=== Post-finetune Evaluation ===', flush=True)

    # K=1 (should be similar to baseline — did we hurt it?)
    post_k1 = evaluate_model(model, tokenizer, device, 1, 'post-ft K=1')

    # K=2
    orig, orig_N, new_N = build_recursive_model(model, args.core_start, args.core_end, 2)
    post_k2 = evaluate_model(model, tokenizer, device, 2, 'post-ft K=2')
    restore_model(model, orig, orig_N)

    # K=3
    orig, orig_N, new_N = build_recursive_model(model, args.core_start, args.core_end, 3)
    post_k3 = evaluate_model(model, tokenizer, device, 3, 'post-ft K=3')
    restore_model(model, orig, orig_N)

    # ======================================================================
    # 5. Summary
    # ======================================================================
    print(f'\n{"=" * 50}', flush=True)
    print(f'SUMMARY — {args.name}', flush=True)
    print(f'{"=" * 50}', flush=True)
    print(f'  Baseline (K=1):     {baseline["combined"]:.2f}', flush=True)
    print(f'  Pre-ft K=2:         {pre_ft_k2["combined"]:.2f} ({pre_ft_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft K=1:        {post_k1["combined"]:.2f} ({post_k1["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft K=2:        {post_k2["combined"]:.2f} ({post_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft K=3:        {post_k3["combined"]:.2f} ({post_k3["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Recursion benefit:  {post_k2["combined"]-post_k1["combined"]:+.2f} (K=2 vs K=1 post-ft)', flush=True)
    print('COMPLETE', flush=True)

    results = {
        'model': args.model, 'name': args.name, 'layers': N,
        'core': [args.core_start, args.core_end],
        'max_steps': args.max_steps,
        'baseline': baseline,
        'pre_ft_k2': pre_ft_k2,
        'post_ft_k1': post_k1,
        'post_ft_k2': post_k2,
        'post_ft_k3': post_k3,
    }
    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/results.json', flush=True)


if __name__ == '__main__':
    main()
