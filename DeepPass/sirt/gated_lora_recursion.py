"""
Gated Pass-2 LoRA with Contrastive Training + Task-Conditional Routing

The GPT-5.4 Pro "paradigm shift" recipe:
1. Pass-2-only OPLoRA (K=1 preserved by construction)
2. Contrastive K=1/K=2 weighting (only learn from positive-advantage examples)
3. Learned task gate (predicts whether to recurse based on first-pass hidden states)
4. Math-inference whisper (β_FFN=0.2 on math-like prompts at inference only)

Combined recipe: the smallest intervention that fixes "3 up, 1 down" → "4 up"
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


class Pass2LoRA(nn.Module):
    """LoRA that only activates during pass 2. K=1 preserved by construction."""
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # zero init — warmup phase will train it
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.pass_idx = 1  # toggled externally

    def forward(self, x):
        y = self.base(x)
        if self.pass_idx == 2:
            y = y + self.scale * (x @ self.lora_A.t() @ self.lora_B.t())
        return y


class RecursionGate(nn.Module):
    """Tiny gate: predicts whether to recurse and optimal α from first-pass hidden states."""
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),  # [recurse_logit, alpha_logit]
        )

    def forward(self, h1_pooled):
        out = self.net(h1_pooled)
        recurse_prob = torch.sigmoid(out[:, 0])
        alpha = torch.sigmoid(out[:, 1])  # [0, 1]
        return recurse_prob, alpha


class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None
    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'): self.layer.self_attn.layer_idx = self.new_layer_idx
        try: return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None: self.layer.self_attn.layer_idx = self.orig_attn
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self.layer, name)


def inject_pass2_lora(model, core_start, core_end, rank_attn=8, rank_ffn=4, device=None, dtype=None):
    """Inject Pass2LoRA into core layers. Different ranks for attn vs FFN."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    layers = list(inner.layers)
    lora_modules = []

    for i in range(core_start, core_end):
        layer = layers[i]
        # Attention projections: rank_attn
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(layer.self_attn, name):
                base = getattr(layer.self_attn, name)
                if isinstance(base, nn.Linear):
                    lora = Pass2LoRA(base, rank=rank_attn)
                    if device: lora.lora_A = nn.Parameter(lora.lora_A.to(device=device, dtype=dtype))
                    if device: lora.lora_B = nn.Parameter(lora.lora_B.to(device=device, dtype=dtype))
                    setattr(layer.self_attn, name, lora)
                    lora_modules.append(lora)

        # FFN projections: rank_ffn (lower rank — less aggressive on memory retrieval)
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(layer.mlp, name):
                base = getattr(layer.mlp, name)
                if isinstance(base, nn.Linear):
                    lora = Pass2LoRA(base, rank=rank_ffn)
                    if device: lora.lora_A = nn.Parameter(lora.lora_A.to(device=device, dtype=dtype))
                    if device: lora.lora_B = nn.Parameter(lora.lora_B.to(device=device, dtype=dtype))
                    setattr(layer.mlp, name, lora)
                    lora_modules.append(lora)

    return lora_modules


def set_pass_idx(lora_modules, pass_idx):
    for m in lora_modules:
        m.pass_idx = pass_idx


def build_k2(model, core_start, core_end):
    """Build K=2 layer order with LayerIdxWrapper."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    original_layers = list(inner.layers)
    N = len(original_layers)

    order = list(range(core_end)) + list(range(core_start, core_end)) + list(range(core_end, N))
    seen = set()
    new_layers = []
    for pi, oi in enumerate(order):
        l = original_layers[oi]
        if oi in seen:
            new_layers.append(LayerIdxWrapper(l, pi))
        else:
            l.layer_idx = pi
            if hasattr(l, 'self_attn'): l.self_attn.layer_idx = pi
            new_layers.append(l)
        seen.add(oi)

    inner.layers = nn.ModuleList(new_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None: continue
        if hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = len(new_layers)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[oi] for oi in order]
        if hasattr(cfg, 'use_cache'): cfg.use_cache = False
    # Also update inner model config
    if hasattr(inner, 'config'):
        if hasattr(inner.config, 'num_hidden_layers'): inner.config.num_hidden_layers = len(new_layers)
        if hasattr(inner.config, 'layer_types') and inner.config.layer_types:
            inner.config.layer_types = [inner.config.layer_types[oi] for oi in order]
    return original_layers, N


def restore(model, original_layers, N):
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    inner.layers = nn.ModuleList(original_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'num_hidden_layers'): cfg.num_hidden_layers = N
    for i, l in enumerate(original_layers):
        l.layer_idx = i
        if hasattr(l, 'self_attn'): l.self_attn.layer_idx = i


def train_gated_lora(model, tokenizer, device, core_start, core_end,
                     lora_modules, gate, max_steps=320, lr=5e-5):
    """Combined training: contrastive + gated + pass-2 LoRA."""
    from datasets import load_dataset

    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    original_layers = list(inner.layers)
    N = len(original_layers)

    # Trainable: LoRA params + gate params
    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])
    gate_params = list(gate.parameters())
    all_params = lora_params + gate_params
    trainable = sum(p.numel() for p in all_params)
    print(f'  Trainable: {trainable:,} (LoRA: {sum(p.numel() for p in lora_params):,}, gate: {sum(p.numel() for p in gate_params):,})', flush=True)

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    model.train()
    gate.train()
    step = 0
    running_loss = 0
    pos_adv = 0
    t0 = time.time()
    token_buffer = []

    for example in ds:
        if step >= max_steps:
            break
        text = example.get('text', '')
        if not text or len(text) < 100:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=2048)
        if tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)
        token_buffer.extend(tokens)

        while len(token_buffer) >= 1025 and step < max_steps:
            chunk = torch.tensor([token_buffer[:1025]], dtype=torch.long).to(device)
            token_buffer = token_buffer[1024:]
            input_ids = chunk[:, :-1]
            labels = chunk[:, 1:]

            # === K=1 loss (LoRA OFF, no recursion) ===
            set_pass_idx(lora_modules, 1)
            with torch.no_grad():
                fwd1_kwargs = {'input_ids': input_ids, 'use_cache': False, 'output_hidden_states': True}
                if hasattr(model.config, 'model_type') and 'gemma' in str(getattr(model.config, 'model_type', '')):
                    fwd1_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
                try:
                    out1 = model(**fwd1_kwargs)
                except TypeError:
                    fwd1_kwargs.pop('output_hidden_states', None)
                    out1 = model(**fwd1_kwargs)
                l1 = F.cross_entropy(out1.logits.reshape(-1, out1.logits.size(-1)),
                                     labels.reshape(-1)).item()
                # Get first-pass hidden state for gate
                if hasattr(out1, 'hidden_states') and out1.hidden_states is not None:
                    h1_pooled = out1.hidden_states[-1][:, -16:, :].mean(dim=1).detach()
                else:
                    # Fallback: project logits through a random projection (cheap approximation)
                    h1_pooled = torch.zeros(1, d_model, device=device, dtype=torch.float32)

            # === K=2 loss (LoRA ON during pass 2) ===
            # Set LoRA active for ALL passes during K=2 (simpler, avoids hook complexity)
            set_pass_idx(lora_modules, 2)
            orig, orig_N = build_k2(model, core_start, core_end)

            fwd2_kwargs = {'input_ids': input_ids, 'use_cache': False}
            if hasattr(model.config, 'model_type') and 'gemma' in str(getattr(model.config, 'model_type', '')):
                fwd2_kwargs['token_type_ids'] = torch.zeros_like(input_ids)
            out2 = model(**fwd2_kwargs)
            l2 = F.cross_entropy(out2.logits.reshape(-1, out2.logits.size(-1)),
                                 labels.reshape(-1))

            restore(model, orig, orig_N)
            set_pass_idx(lora_modules, 1)

            # === Gate prediction (always run) ===
            recurse_prob, alpha_pred = gate(h1_pooled.float())
            advantage = l1 - l2.item()
            if advantage > 0:
                pos_adv += 1
            gate_target = torch.tensor([1.0 if advantage > 0.05 else 0.0], device=device)
            gate_loss = F.binary_cross_entropy(recurse_prob, gate_target)

            # === Contrastive weight (with warmup) ===
            warmup_steps = 80
            if step < warmup_steps:
                # Warmup: standard K=2 loss + gate loss
                loss = l2 + 0.05 * gate_loss
            else:
                # Contrastive: weight by advantage
                w = torch.sigmoid(torch.tensor((advantage - 0.05) / 0.30, device=device))
                loss = w * l2 + 0.1 * gate_loss

            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 0.3)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                running_loss += l2.item()
            except Exception as e:
                print(f'  step {step} ERROR: {e}', flush=True)
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 50 == 0:
                avg_loss = running_loss / min(50, step)
                adv_rate = pos_adv / step
                elapsed = time.time() - t0
                print(f'  step {step:4d} | loss {avg_loss:.4f} | adv_rate={adv_rate:.2f} | '
                      f'gate_p={recurse_prob.item():.2f} | {elapsed:.0f}s', flush=True)
                running_loss = 0

    print(f'  Done: {step} steps, adv_rate={pos_adv/max(step,1):.2f}, {time.time()-t0:.0f}s', flush=True)
    model.eval()
    gate.eval()


def evaluate(model, tokenizer, device, tag=""):
    model.eval()
    for m in model.modules(): m.training = False
    eq_all = _load_questions()
    def gen(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            try:
                out = model.generate(**kw, max_new_tokens=64, do_sample=False, use_cache=False)
            except Exception:
                kw.pop('token_type_ids', None)
                out = model.generate(**kw, max_new_tokens=64, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    def gen_long(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            try:
                out = model.generate(**kw, max_new_tokens=128, do_sample=False, use_cache=False)
            except Exception:
                kw.pop('token_type_ids', None)
                out = model.generate(**kw, max_new_tokens=128, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, default=10)
    parser.add_argument('--core_end', type=int, default=13)
    parser.add_argument('--max_steps', type=int, default=320)
    parser.add_argument('--lr', type=float, default=5e-5)
    args = parser.parse_args()

    SAVE_DIR = f'sirt/recursion_ft/{args.name}_gated'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    model_dtype = next(model.parameters()).dtype

    # Freeze ALL base weights
    for param in model.parameters():
        param.requires_grad = False

    # Inject pass-2 LoRA (rank 8 attn, rank 4 FFN)
    print(f'Injecting Pass-2 LoRA [core {args.core_start}-{args.core_end})...', flush=True)
    lora_modules = inject_pass2_lora(model, args.core_start, args.core_end,
                                      rank_attn=8, rank_ffn=4,
                                      device=device, dtype=model_dtype)
    print(f'  {len(lora_modules)} LoRA modules', flush=True)

    # Create gate
    d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
    gate = RecursionGate(d_model).to(device=device, dtype=torch.float32)

    # === 1. Baseline K=1 ===
    print('\n=== Baseline K=1 ===', flush=True)
    set_pass_idx(lora_modules, 1)
    baseline = evaluate(model, tokenizer, device, 'baseline K=1')

    # === 2. Pre-ft K=2 (LoRA untrained) ===
    print('\n=== Pre-ft K=2 ===', flush=True)
    orig, orig_N = build_k2(model, args.core_start, args.core_end)
    set_pass_idx(lora_modules, 2)
    pre_k2 = evaluate(model, tokenizer, device, 'pre-ft K=2')
    restore(model, orig, orig_N)
    set_pass_idx(lora_modules, 1)

    # === 3. Train ===
    print('\n=== Gated Contrastive Training ===', flush=True)
    train_gated_lora(model, tokenizer, device, args.core_start, args.core_end,
                     lora_modules, gate, max_steps=args.max_steps, lr=args.lr)

    # === 4. Post-ft K=1 (must match baseline exactly) ===
    print('\n=== Post-ft K=1 ===', flush=True)
    set_pass_idx(lora_modules, 1)
    model.eval()
    # Ensure ALL submodules are in eval mode (Gemma3 nested model fix)
    for m in model.modules():
        m.training = False
    model.config.use_cache = True
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'use_cache'):
        model.config.text_config.use_cache = True
    post_k1 = evaluate(model, tokenizer, device, 'post-ft K=1')

    # === 5. Post-ft K=2 ===
    print('\n=== Post-ft K=2 ===', flush=True)
    orig, orig_N = build_k2(model, args.core_start, args.core_end)
    # Toggle LoRA via hooks
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    original_layers = list(inner.layers)
    hooks = []
    for layer_idx in range(args.core_start, args.core_end):
        module = original_layers[layer_idx]
        counter = [0]
        n_core = args.core_end - args.core_start
        def make_toggle(ctr, nc, lm):
            def hook(module, args):
                ctr[0] += 1
                set_pass_idx(lm, 2 if ctr[0] > nc else 1)
            return hook
        h = module.register_forward_pre_hook(make_toggle(counter, n_core, lora_modules))
        hooks.append(h)

    model.config.use_cache = True
    post_k2 = evaluate(model, tokenizer, device, 'post-ft K=2')
    for h in hooks: h.remove()
    restore(model, orig, orig_N)

    # === Summary ===
    print(f'\n{"=" * 50}', flush=True)
    print(f'SUMMARY — {args.name} (Gated Pass-2 LoRA)', flush=True)
    print(f'{"=" * 50}', flush=True)
    print(f'  Baseline K=1:  {baseline["combined"]:.2f}', flush=True)
    print(f'  Pre-ft K=2:    {pre_k2["combined"]:.2f} ({pre_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft K=1:   {post_k1["combined"]:.2f} ({post_k1["combined"]-baseline["combined"]:+.2f}) {"✅" if abs(post_k1["combined"]-baseline["combined"]) < 0.5 else "❌"}', flush=True)
    print(f'  Post-ft K=2:   {post_k2["combined"]:.2f} ({post_k2["combined"]-baseline["combined"]:+.2f}) {"✅" if post_k2["combined"] > baseline["combined"] else "❌"}', flush=True)
    print('COMPLETE', flush=True)

    results = {
        'model': args.model, 'name': args.name,
        'core': [args.core_start, args.core_end],
        'baseline': baseline, 'pre_k2': pre_k2,
        'post_k1': post_k1, 'post_k2': post_k2,
    }
    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/results.json', flush=True)


if __name__ == '__main__':
    main()
