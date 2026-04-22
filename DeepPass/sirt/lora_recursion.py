"""
Pass-2-Only LoRA Recursion Fine-Tuning

Key insight from GPT-5.4 Pro: freeze ALL base weights, add rank-8 LoRA adapters
that ONLY activate during the second pass. K=1 is preserved BY CONSTRUCTION.

Combined with contrastive loss: only update when K=2 beats K=1.

Usage:
    python lora_recursion.py --model <path> --name <name> --core_start 10 --core_end 13
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


class LoRALayer(nn.Module):
    """Low-rank adapter that can be toggled on/off per pass."""
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = alpha / rank

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero so LoRA contribution starts at zero

        self.active = False  # Only active during pass 2

    def forward(self, x):
        y = self.base(x)
        if self.active:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            y = y + self.scale * lora_out
        return y


class LayerIdxWrapper(nn.Module):
    """Wraps a layer to give unique layer_idx for KV cache."""
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


def inject_lora(model, core_start, core_end, rank=8, alpha=16):
    """Replace linear layers in core with LoRA-wrapped versions."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model

    layers = list(inner.layers)
    lora_modules = []
    target_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    for i in range(core_start, core_end):
        layer = layers[i]
        for name in target_names:
            # Navigate to the module
            parts = name.split('.')
            parent = layer
            for part in ['self_attn'] if name in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else ['mlp']:
                if hasattr(parent, part):
                    parent = getattr(parent, part)

            if hasattr(parent, name.split('.')[-1]):
                base_linear = getattr(parent, name.split('.')[-1])
                if isinstance(base_linear, nn.Linear):
                    lora = LoRALayer(base_linear, rank=rank, alpha=alpha)
                    setattr(parent, name.split('.')[-1], lora)
                    lora_modules.append(lora)

    return lora_modules


def set_lora_active(lora_modules, active):
    """Toggle all LoRA modules on/off."""
    for m in lora_modules:
        m.active = active


def build_recursive_order(core_start, core_end, N, K):
    """Build layer execution order with K passes through core."""
    order = list(range(core_start))
    for _ in range(K):
        order.extend(range(core_start, core_end))
    order.extend(range(core_end, N))
    return order


def apply_recursion(model, core_start, core_end, K, lora_modules):
    """Apply K-pass recursion with LoRA active only on pass 2+."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model

    original_layers = list(inner.layers)
    N = len(original_layers)
    order = build_recursive_order(core_start, core_end, N, K)

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
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = len(new_layers)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[idx] for idx in order]
        if hasattr(cfg, 'use_cache'):
            cfg.use_cache = False  # Disable during training

    # LoRA: OFF for pass 1, ON for pass 2
    # This is handled by the forward hooks — we need a different approach
    # Use a pre-forward hook on each core layer to toggle LoRA based on fire count
    hooks = []
    for layer_idx in range(core_start, core_end):
        layer = original_layers[layer_idx]
        counter = [0]
        n_core = core_end - core_start

        def make_toggle_hook(ctr, n_core, lora_mods, layer_idx, core_start):
            def hook(module, args):
                ctr[0] += 1
                # First n_core fires = pass 1 (LoRA OFF)
                # Next n_core fires = pass 2 (LoRA ON)
                is_pass2 = ctr[0] > n_core
                # Find LoRA modules in this layer
                for m in lora_mods:
                    # Check if this LoRA belongs to this layer
                    if hasattr(m, 'active'):
                        m.active = is_pass2
            return hook

        # Actually, simpler: toggle ALL LoRAs based on whether we're in pass 2
        # Since layers fire sequentially, we can use the layer_idx wrapper detection

    # Simpler approach: use the LayerIdxWrapper as pass-2 indicator
    # When a wrapped layer fires, LoRA should be ON
    # When the original fires, LoRA should be OFF

    # Register pre-forward hooks on each core layer
    for layer_idx in range(core_start, core_end):
        layer = original_layers[layer_idx]
        fire_count = [0]

        def make_lora_hook(ctr, lora_mods):
            def hook(module, args):
                ctr[0] += 1
                # Odd fires = pass 1 (LoRA OFF), Even fires = pass 2 (LoRA ON)
                for m in lora_mods:
                    m.active = (ctr[0] % 2 == 0)
            return hook

        h = layer.register_forward_pre_hook(make_lora_hook(fire_count, lora_modules))
        hooks.append(h)

    return original_layers, N, hooks


def restore_model(model, original_layers, N, hooks):
    """Restore original layers and remove hooks."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model

    for h in hooks:
        h.remove()

    inner.layers = nn.ModuleList(original_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = N

    for i, l in enumerate(original_layers):
        l.layer_idx = i
        if hasattr(l, 'self_attn'):
            l.self_attn.layer_idx = i


def evaluate_model(model, tokenizer, device, tag=""):
    """Quick dual-probe evaluation."""
    eq_all = _load_questions()

    def gen(prompt):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        gen_kwargs = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            try:
                out = model.generate(**gen_kwargs, max_new_tokens=64, do_sample=False, use_cache=True)
            except:
                out = model.generate(**gen_kwargs, max_new_tokens=64, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gen_long(prompt):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        gen_kwargs = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            try:
                out = model.generate(**gen_kwargs, max_new_tokens=128, do_sample=False, use_cache=True)
            except:
                out = model.generate(**gen_kwargs, max_new_tokens=128, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def train_lora_recursion(model, tokenizer, device, core_start, core_end, lora_modules,
                         max_steps=300, lr=5e-5, batch_size=1, seq_len=1024, contrastive=True):
    """Train pass-2-only LoRA with optional contrastive objective."""
    from datasets import load_dataset

    # Only LoRA params are trainable
    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])

    trainable = sum(p.numel() for p in lora_params)
    print(f'  LoRA trainable params: {trainable:,}', flush=True)

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)

    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    original_layers = list(inner.layers)
    N = len(original_layers)

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    model.train()
    step = 0
    running_loss = 0
    pos_adv_count = 0
    t0 = time.time()
    token_buffer = []

    for example in ds:
        if step >= max_steps:
            break

        text = example.get('text', '')
        if not text or len(text) < 100:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_len * 4)
        if tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= (seq_len + 1) * batch_size and step < max_steps:
            batch_tokens = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_len + 1]
                token_buffer = token_buffer[seq_len:]
                batch_tokens.append(chunk)

            batch = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            if contrastive:
                # K=1 loss (LoRA OFF — baseline preserved)
                set_lora_active(lora_modules, False)
                with torch.no_grad():
                    fwd_kwargs = {'input_ids': input_ids, 'use_cache': False}
                    if hasattr(model.config, 'model_type') and 'gemma' in str(model.config.model_type):
                        fwd_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
                    out1 = model(**fwd_kwargs)
                    l1 = F.cross_entropy(out1.logits.reshape(-1, out1.logits.size(-1)),
                                         labels.reshape(-1)).item()

                # K=2 loss (LoRA ON during pass 2)
                orig, orig_N, hooks = apply_recursion(model, core_start, core_end, 2, lora_modules)

                fwd_kwargs = {'input_ids': input_ids, 'use_cache': False}
                if hasattr(model.config, 'model_type') and 'gemma' in str(model.config.model_type):
                    fwd_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
                out2 = model(**fwd_kwargs)
                l2 = F.cross_entropy(out2.logits.reshape(-1, out2.logits.size(-1)),
                                     labels.reshape(-1))

                # Contrastive weight: only train when K=2 is better
                margin = 0.05
                w = torch.sigmoid(torch.tensor((l1 - l2.item() - margin) / 0.30))

                if l2.item() < l1:
                    pos_adv_count += 1

                loss = w * l2
            else:
                # Simple K=2 training
                K = random.choices([1, 2], weights=[0.35, 0.65])[0]
                if K == 1:
                    set_lora_active(lora_modules, False)
                    fwd_kwargs = {'input_ids': input_ids, 'use_cache': False}
                    if hasattr(model.config, 'model_type') and 'gemma' in str(model.config.model_type):
                        fwd_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
                    out = model(**fwd_kwargs)
                    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)),
                                          labels.reshape(-1))
                else:
                    orig, orig_N, hooks = apply_recursion(model, core_start, core_end, 2, lora_modules)
                    fwd_kwargs = {'input_ids': input_ids, 'use_cache': False}
                    if hasattr(model.config, 'model_type') and 'gemma' in str(model.config.model_type):
                        fwd_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
                    out = model(**fwd_kwargs)
                    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)),
                                          labels.reshape(-1))

            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 0.3)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                running_loss += loss.item()
            except Exception as e:
                print(f'  step {step} ERROR: {e}', flush=True)
                optimizer.zero_grad(set_to_none=True)

            # Restore model after recursion
            if 'orig' in dir():
                restore_model(model, orig, orig_N, hooks)
            set_lora_active(lora_modules, False)

            step += 1
            if step % 50 == 0:
                avg_loss = running_loss / min(50, step)
                adv_rate = pos_adv_count / step if step > 0 else 0
                elapsed = time.time() - t0
                print(f'  step {step:4d} | loss {avg_loss:.4f} | adv_rate={adv_rate:.2f} | {elapsed:.0f}s', flush=True)
                running_loss = 0

    print(f'  Training done: {step} steps, {time.time()-t0:.0f}s', flush=True)
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, required=True)
    parser.add_argument('--core_end', type=int, required=True)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--contrastive', action='store_true', default=True)
    parser.add_argument('--no_contrastive', dest='contrastive', action='store_false')
    args = parser.parse_args()

    SAVE_DIR = f'sirt/recursion_ft/{args.name}_lora'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    # Load model
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
    print(f'Loaded: {N} layers', flush=True)

    # Freeze ALL base weights
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into core layers
    print(f'\nInjecting rank-{args.rank} LoRA into layers [{args.core_start},{args.core_end})...', flush=True)
    lora_modules = inject_lora(model, args.core_start, args.core_end, rank=args.rank)
    # Move LoRA params to same device AND dtype as model
    model_dtype = next(model.parameters()).dtype
    for m in lora_modules:
        m.lora_A = nn.Parameter(m.lora_A.to(device=device, dtype=model_dtype))
        m.lora_B = nn.Parameter(m.lora_B.to(device=device, dtype=model_dtype))
    print(f'  Injected {len(lora_modules)} LoRA modules (on {device})', flush=True)

    # ======================================================================
    # 1. Baseline (K=1, LoRA OFF — should be identical to original)
    # ======================================================================
    print('\n=== Baseline K=1 (LoRA OFF) ===', flush=True)
    set_lora_active(lora_modules, False)
    baseline = evaluate_model(model, tokenizer, device, 'baseline K=1')

    # ======================================================================
    # 2. Pre-ft K=2 (LoRA ON but untrained — should be same as raw dup)
    # ======================================================================
    print('\n=== Pre-ft K=2 (LoRA untrained) ===', flush=True)
    orig, orig_N, hooks = apply_recursion(model, args.core_start, args.core_end, 2, lora_modules)
    pre_k2 = evaluate_model(model, tokenizer, device, 'pre-ft K=2')
    restore_model(model, orig, orig_N, hooks)
    set_lora_active(lora_modules, False)

    # ======================================================================
    # 3. Train LoRA
    # ======================================================================
    print(f'\n=== Training (contrastive={args.contrastive}) ===', flush=True)
    train_lora_recursion(
        model, tokenizer, device, args.core_start, args.core_end, lora_modules,
        max_steps=args.max_steps, lr=args.lr,
        batch_size=args.batch_size, seq_len=args.seq_len,
        contrastive=args.contrastive,
    )

    # ======================================================================
    # 4. Post-ft K=1 (LoRA OFF — should be IDENTICAL to baseline)
    # ======================================================================
    print('\n=== Post-ft K=1 (LoRA OFF — must match baseline) ===', flush=True)
    set_lora_active(lora_modules, False)
    post_k1 = evaluate_model(model, tokenizer, device, 'post-ft K=1')

    # ======================================================================
    # 5. Post-ft K=2 (LoRA ON)
    # ======================================================================
    print('\n=== Post-ft K=2 (LoRA trained) ===', flush=True)
    orig, orig_N, hooks = apply_recursion(model, args.core_start, args.core_end, 2, lora_modules)
    post_k2 = evaluate_model(model, tokenizer, device, 'post-ft K=2')
    restore_model(model, orig, orig_N, hooks)

    # ======================================================================
    # Summary
    # ======================================================================
    print(f'\n{"=" * 50}', flush=True)
    print(f'SUMMARY — {args.name} (Pass-2 LoRA)', flush=True)
    print(f'{"=" * 50}', flush=True)
    print(f'  Baseline K=1:     {baseline["combined"]:.2f}', flush=True)
    print(f'  Pre-ft K=2:       {pre_k2["combined"]:.2f} ({pre_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft K=1:      {post_k1["combined"]:.2f} ({post_k1["combined"]-baseline["combined"]:+.2f}) {"✅" if abs(post_k1["combined"]-baseline["combined"]) < 0.5 else "❌"}', flush=True)
    print(f'  Post-ft K=2:      {post_k2["combined"]:.2f} ({post_k2["combined"]-baseline["combined"]:+.2f}) {"✅" if post_k2["combined"] > baseline["combined"] else "❌"}', flush=True)
    print(f'  K=1 preserved:    {abs(post_k1["combined"]-baseline["combined"]) < 0.5}', flush=True)
    print(f'  K=2 > baseline:   {post_k2["combined"] > baseline["combined"]}', flush=True)
    print('COMPLETE', flush=True)

    results = {
        'model': args.model, 'name': args.name,
        'core': [args.core_start, args.core_end],
        'rank': args.rank, 'steps': args.max_steps,
        'contrastive': args.contrastive,
        'baseline': baseline, 'pre_k2': pre_k2,
        'post_k1': post_k1, 'post_k2': post_k2,
        'k1_preserved': abs(post_k1['combined'] - baseline['combined']) < 0.5,
        'k2_improves': post_k2['combined'] > baseline['combined'],
    }
    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/results.json', flush=True)


if __name__ == '__main__':
    main()
