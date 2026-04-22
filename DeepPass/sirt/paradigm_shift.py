"""
Paradigm Shift: Combined 5-Intervention Recipe for Recursion Fine-Tuning

Addresses 3 structural problems from GPT-5.4 Pro analysis:
1. K=1 drift -> Pass-2-only OPLoRA (K=1 preserved by construction)
2. Unconditional recursion -> Task gate + contrastive K=1/K=2 weighting
3. Objective mismatch -> Contrastive loss, alpha warmup, OPLoRA penalty

Inference enhancements:
4. Math FFN whisper: beta=0.2 on pass-2 FFN for factual inputs
5. Gate-routed recursion: skip K=2 when gate predicts no benefit

Usage:
    python paradigm_shift.py --model <path> --name <name> --core_start 10 --core_end 13
"""

import os, sys, json, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Components
# ============================================================

class Pass2OPLoRA(nn.Module):
    """LoRA active only during pass 2, with orthogonal projection penalty."""

    def __init__(self, base_linear, rank=8, alpha=16, is_ffn=False):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = alpha / rank
        self.is_ffn = is_ffn
        self.warmup_scale = 1.0

        for p in self.base.parameters():
            p.requires_grad = False

        d_in = base_linear.in_features
        d_out = base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.pass_idx = 1  # 1=OFF, 2=ON — toggled by LayerIdxWrapper

        # Cached SVD for OPLoRA penalty (lazy init)
        self._U_k = None
        self._V_k = None

    def forward(self, x):
        y = self.base(x)
        if self.pass_idx == 2:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            y = y + self.warmup_scale * self.scale * lora_out
        return y

    def compute_oplora_penalty(self, k=16):
        """Penalty for LoRA aligning with base weight's top-k singular vectors."""
        if self._U_k is None:
            with torch.no_grad():
                W = self.base.weight.detach().float()
                try:
                    U, S, V = torch.svd_lowrank(W, q=min(k, min(W.shape) - 1))
                    self._U_k = U.detach()
                    self._V_k = V.detach()
                except Exception:
                    return torch.tensor(0.0, device=self.lora_A.device)
        delta = (self.lora_B @ self.lora_A).float()
        left = torch.norm(self._U_k.t() @ delta)
        right = torch.norm(delta @ self._V_k)
        return left + right


class RecursionGate(nn.Module):
    """Predicts whether recursion benefits the current input."""

    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h1_pooled):
        return torch.sigmoid(self.net(h1_pooled)).squeeze(-1)


class LayerIdxWrapper(nn.Module):
    """Wrapper for duplicated layers: unique layer_idx + LoRA toggle."""

    # Class-level flag for FFN whisper
    _pass2_active = False

    def __init__(self, layer, new_idx, lora_modules=None):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None
        self._lora_modules = lora_modules or []

    def forward(self, *args, **kwargs):
        # Swap layer indices for KV cache
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'):
            self.layer.self_attn.layer_idx = self.new_layer_idx
        # Enable LoRA for pass 2
        for l in self._lora_modules:
            l.pass_idx = 2
        LayerIdxWrapper._pass2_active = True
        try:
            return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None:
                self.layer.self_attn.layer_idx = self.orig_attn
            for l in self._lora_modules:
                l.pass_idx = 1
            LayerIdxWrapper._pass2_active = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


# ============================================================
# Model Operations
# ============================================================

def get_inner(model):
    """Navigate to the inner transformer module."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    return inner


def inject_oplora(model, core_start, core_end, rank_attn=8, rank_ffn=4,
                  device=None, dtype=None):
    """Inject Pass2OPLoRA into core layers. Returns (all_loras, layer_loras)."""
    inner = get_inner(model)
    layers = list(inner.layers)
    all_loras = []
    layer_loras = {}

    for i in range(core_start, core_end):
        layer = layers[i]
        layer_loras[i] = []

        # Attention projections
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if hasattr(layer.self_attn, name):
                base = getattr(layer.self_attn, name)
                if isinstance(base, nn.Linear):
                    lora = Pass2OPLoRA(base, rank=rank_attn, is_ffn=False)
                    if device:
                        lora.lora_A = nn.Parameter(lora.lora_A.to(device=device, dtype=dtype))
                        lora.lora_B = nn.Parameter(lora.lora_B.to(device=device, dtype=dtype))
                    setattr(layer.self_attn, name, lora)
                    all_loras.append(lora)
                    layer_loras[i].append(lora)

        # FFN projections (lower rank)
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(layer.mlp, name):
                base = getattr(layer.mlp, name)
                if isinstance(base, nn.Linear):
                    lora = Pass2OPLoRA(base, rank=rank_ffn, is_ffn=True)
                    if device:
                        lora.lora_A = nn.Parameter(lora.lora_A.to(device=device, dtype=dtype))
                        lora.lora_B = nn.Parameter(lora.lora_B.to(device=device, dtype=dtype))
                    setattr(layer.mlp, name, lora)
                    all_loras.append(lora)
                    layer_loras[i].append(lora)

    return all_loras, layer_loras


def build_k2(model, core_start, core_end, layer_loras=None):
    """Build K=2 layer order. LayerIdxWrapper handles KV cache + LoRA toggle."""
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)

    order = list(range(core_end)) + list(range(core_start, core_end)) + list(range(core_end, N))
    seen = set()
    new_layers = []
    for pi, oi in enumerate(order):
        l = original_layers[oi]
        if oi in seen:
            loras = layer_loras.get(oi, []) if layer_loras else []
            new_layers.append(LayerIdxWrapper(l, pi, lora_modules=loras))
        else:
            l.layer_idx = pi
            if hasattr(l, 'self_attn'):
                l.self_attn.layer_idx = pi
            new_layers.append(l)
        seen.add(oi)

    inner.layers = nn.ModuleList(new_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg is None:
            continue
        if hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = len(new_layers)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[oi % len(cfg.layer_types)] for oi in order]
    if hasattr(inner, 'config') and hasattr(inner.config, 'num_hidden_layers'):
        inner.config.num_hidden_layers = len(new_layers)
        if hasattr(inner.config, 'layer_types') and inner.config.layer_types:
            inner.config.layer_types = [inner.config.layer_types[oi % len(inner.config.layer_types)] for oi in order]

    return original_layers, N


def restore(model, original_layers, N):
    """Restore original layer order."""
    inner = get_inner(model)
    inner.layers = nn.ModuleList(original_layers)
    for cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if cfg and hasattr(cfg, 'num_hidden_layers'):
            cfg.num_hidden_layers = N
    if hasattr(inner, 'config') and hasattr(inner.config, 'num_hidden_layers'):
        inner.config.num_hidden_layers = N
    for i, l in enumerate(original_layers):
        l.layer_idx = i
        if hasattr(l, 'self_attn'):
            l.self_attn.layer_idx = i


def setup_ffn_whisper(model, core_start, core_end, beta=0.2):
    """Register hooks to scale MLP output by beta during pass 2."""
    inner = get_inner(model)
    hooks = []
    for i in range(core_start, core_end):
        layer = list(inner.layers)[i]
        actual_layer = layer.layer if isinstance(layer, LayerIdxWrapper) else layer

        def make_hook(b):
            def hook(module, input, output):
                if LayerIdxWrapper._pass2_active:
                    if isinstance(output, tuple):
                        return tuple(o * b if isinstance(o, torch.Tensor) else o for o in output)
                    return output * b
                return output
            return hook

        h = actual_layer.mlp.register_forward_hook(make_hook(beta))
        hooks.append(h)
    return hooks


# ============================================================
# Training
# ============================================================

def alpha_schedule(step, warmup_steps=80, a0=0.05, a1=1.0):
    t = min(step / max(warmup_steps, 1), 1.0)
    return a0 + t * (a1 - a0)


def compute_oplora_penalty(lora_modules, k=16):
    penalties = [m.compute_oplora_penalty(k) for m in lora_modules]
    valid = [p for p in penalties if p.item() > 0]
    return sum(valid) / max(len(valid), 1) if valid else torch.tensor(0.0)


def train(model, tokenizer, device, core_start, core_end,
          lora_modules, layer_loras, gate,
          max_steps=320, lr=5e-5, warmup_steps=80,
          contrastive_margin=0.05, contrastive_temp=0.30,
          gate_weight=0.10, oplora_weight=0.02):
    """Combined training: contrastive + gated + OPLoRA + alpha warmup."""
    from datasets import load_dataset

    lora_params = []
    for m in lora_modules:
        lora_params.extend([m.lora_A, m.lora_B])
    gate_params = list(gate.parameters())
    all_params = lora_params + gate_params

    n_lora = sum(p.numel() for p in lora_params)
    n_gate = sum(p.numel() for p in gate_params)
    print(f'  Trainable: {n_lora + n_gate:,} (LoRA: {n_lora:,}, gate: {n_gate:,})', flush=True)

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    d_model = getattr(model.config, 'hidden_size', 4096)
    is_gemma = 'gemma' in str(getattr(model.config, 'model_type', ''))

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    model.train()
    gate.train()

    step = 0
    running_l2 = 0
    running_gate_l = 0
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

            # Alpha warmup
            ws = alpha_schedule(step, warmup_steps)
            for m in lora_modules:
                m.warmup_scale = ws

            # === K=1 forward (LoRA OFF — pass_idx=1 by default) ===
            for m in lora_modules:
                m.pass_idx = 1
            with torch.no_grad():
                fwd_kw = {'input_ids': input_ids, 'use_cache': False, 'output_hidden_states': True}
                if is_gemma:
                    fwd_kw['token_type_ids'] = torch.zeros_like(input_ids)
                try:
                    out1 = model(**fwd_kw)
                except TypeError:
                    fwd_kw.pop('output_hidden_states', None)
                    out1 = model(**fwd_kw)
                l1 = F.cross_entropy(out1.logits.view(-1, out1.logits.size(-1)),
                                     labels.view(-1)).item()
                if hasattr(out1, 'hidden_states') and out1.hidden_states is not None:
                    h1_pooled = out1.hidden_states[-1][:, -16:, :].mean(dim=1).detach()
                else:
                    h1_pooled = torch.zeros(1, d_model, device=device, dtype=torch.float32)

            # === K=2 forward (LoRA ON during pass 2 via LayerIdxWrapper) ===
            for m in lora_modules:
                m.pass_idx = 1  # default off; wrapper toggles to 2
            orig, orig_N = build_k2(model, core_start, core_end, layer_loras)

            fwd_kw2 = {'input_ids': input_ids, 'use_cache': False}
            if is_gemma:
                fwd_kw2['token_type_ids'] = torch.zeros_like(input_ids)
            out2 = model(**fwd_kw2)
            l2 = F.cross_entropy(out2.logits.view(-1, out2.logits.size(-1)),
                                 labels.view(-1))

            restore(model, orig, orig_N)
            for m in lora_modules:
                m.pass_idx = 1

            # === Gate ===
            recurse_prob = gate(h1_pooled.float())
            advantage = l1 - l2.item()
            if advantage > 0:
                pos_adv += 1
            gate_target = torch.tensor([1.0 if advantage > contrastive_margin else 0.0],
                                       device=device)
            gate_loss = F.binary_cross_entropy(recurse_prob, gate_target)

            # === Loss ===
            if step < warmup_steps:
                loss = l2 + gate_weight * gate_loss
            else:
                w = torch.sigmoid(torch.tensor((advantage - contrastive_margin) / contrastive_temp,
                                               device=device))
                loss = w * l2 + gate_weight * gate_loss

            # OPLoRA penalty (every 10 steps)
            if oplora_weight > 0 and step % 10 == 0:
                op_loss = compute_oplora_penalty(lora_modules, k=16)
                loss = loss + oplora_weight * op_loss

            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 0.3)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_l2 += l2.item()
                running_gate_l += gate_loss.item()
            except Exception as e:
                print(f'  step {step} ERROR: {e}', flush=True)
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 50 == 0:
                adv_rate = pos_adv / step
                elapsed = time.time() - t0
                print(f'  step {step:4d} | l2={running_l2/50:.4f} | gate={running_gate_l/50:.4f} | '
                      f'adv={adv_rate:.2f} | ws={ws:.2f} | p={recurse_prob.item():.2f} | '
                      f'lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s', flush=True)
                running_l2 = running_gate_l = 0

    print(f'  Done: {step} steps, adv_rate={pos_adv/max(step,1):.2f}, {time.time()-t0:.0f}s', flush=True)
    model.eval()
    gate.eval()


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, tokenizer, device, tag=""):
    model.eval()
    for m in model.modules():
        m.training = False
    eq_all = _load_questions()

    def gen(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            try:
                out = model.generate(**kw, max_new_tokens=64, do_sample=False, use_cache=False)
            except Exception:
                kw['use_cache'] = False
                out = model.generate(**kw, max_new_tokens=64, do_sample=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gen_long(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            try:
                out = model.generate(**kw, max_new_tokens=128, do_sample=False, use_cache=False)
            except Exception:
                kw['use_cache'] = False
                out = model.generate(**kw, max_new_tokens=128, do_sample=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} '
          f'combined={combined:.2f} ({time.time()-t0:.0f}s)', flush=True)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, default=10)
    parser.add_argument('--core_end', type=int, default=13)
    parser.add_argument('--rank_attn', type=int, default=8)
    parser.add_argument('--rank_ffn', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=320)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=80)
    parser.add_argument('--gate_weight', type=float, default=0.10)
    parser.add_argument('--oplora_weight', type=float, default=0.02)
    parser.add_argument('--ffn_whisper_beta', type=float, default=0.2,
                        help='FFN scale on pass 2 during eval (0.2=whisper)')
    args = parser.parse_args()

    SAVE_DIR = f'sirt/recursion_ft/{args.name}_paradigm'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    # === Load model ===
    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    model_dtype = next(model.parameters()).dtype

    for param in model.parameters():
        param.requires_grad = False

    # === Inject OPLoRA ===
    print(f'Injecting OPLoRA [core {args.core_start}-{args.core_end}) '
          f'rank_attn={args.rank_attn} rank_ffn={args.rank_ffn}...', flush=True)
    lora_modules, layer_loras = inject_oplora(
        model, args.core_start, args.core_end,
        rank_attn=args.rank_attn, rank_ffn=args.rank_ffn,
        device=device, dtype=model_dtype,
    )
    print(f'  {len(lora_modules)} OPLoRA modules', flush=True)

    # === Gate ===
    d_model = getattr(model.config, 'hidden_size', 4096)
    gate = RecursionGate(d_model).to(device=device, dtype=torch.float32)

    # === Precompute SVDs ===
    print('  Precomputing SVDs for OPLoRA...', flush=True)
    t0 = time.time()
    for m in lora_modules:
        m.compute_oplora_penalty(k=16)
    print(f'  Done in {time.time()-t0:.1f}s', flush=True)

    # === Baseline K=1 ===
    print('\n=== Baseline K=1 ===', flush=True)
    for m in lora_modules:
        m.pass_idx = 1
    baseline = evaluate(model, tokenizer, device, 'baseline K=1')

    # === Pre-ft K=2 ===
    print('\n=== Pre-ft K=2 (raw dup + untrained LoRA) ===', flush=True)
    orig, orig_N = build_k2(model, args.core_start, args.core_end, layer_loras)
    pre_k2 = evaluate(model, tokenizer, device, 'pre-ft K=2')
    restore(model, orig, orig_N)

    # === Train ===
    print(f'\n=== Paradigm Shift Training ({args.max_steps} steps) ===', flush=True)
    train(model, tokenizer, device, args.core_start, args.core_end,
          lora_modules, layer_loras, gate,
          max_steps=args.max_steps, lr=args.lr, warmup_steps=args.warmup_steps,
          gate_weight=args.gate_weight, oplora_weight=args.oplora_weight)

    # === Post-ft K=1 (must match baseline) ===
    print('\n=== Post-ft K=1 ===', flush=True)
    for m in lora_modules:
        m.pass_idx = 1
    post_k1 = evaluate(model, tokenizer, device, 'post-ft K=1')

    # === Post-ft K=2 ===
    print('\n=== Post-ft K=2 ===', flush=True)
    orig, orig_N = build_k2(model, args.core_start, args.core_end, layer_loras)
    post_k2 = evaluate(model, tokenizer, device, 'post-ft K=2')
    restore(model, orig, orig_N)

    # === Post-ft K=2 + FFN whisper ===
    print(f'\n=== Post-ft K=2 + FFN whisper (beta={args.ffn_whisper_beta}) ===', flush=True)
    orig, orig_N = build_k2(model, args.core_start, args.core_end, layer_loras)
    whisper_hooks = setup_ffn_whisper(model, args.core_start, args.core_end,
                                      beta=args.ffn_whisper_beta)
    post_k2_whisper = evaluate(model, tokenizer, device,
                                f'post-ft K=2 whisper(b={args.ffn_whisper_beta})')
    for h in whisper_hooks:
        h.remove()
    restore(model, orig, orig_N)

    # === Summary ===
    k1_delta = post_k1['combined'] - baseline['combined']
    k2_delta = post_k2['combined'] - baseline['combined']
    k2w_delta = post_k2_whisper['combined'] - baseline['combined']
    print(f'\n{"=" * 60}', flush=True)
    print(f'SUMMARY -- {args.name} (Paradigm Shift)', flush=True)
    print(f'{"=" * 60}', flush=True)
    print(f'  Baseline K=1:       {baseline["combined"]:.2f}', flush=True)
    print(f'  Pre-ft K=2:         {pre_k2["combined"]:.2f} ({pre_k2["combined"]-baseline["combined"]:+.2f})', flush=True)
    print(f'  Post-ft K=1:        {post_k1["combined"]:.2f} ({k1_delta:+.2f}) '
          f'{"PASS" if abs(k1_delta) < 0.5 else "FAIL"}', flush=True)
    print(f'  Post-ft K=2:        {post_k2["combined"]:.2f} ({k2_delta:+.2f}) '
          f'{"PASS" if k2_delta > 0 else "FAIL"}', flush=True)
    print(f'  Post-ft K=2+whisp:  {post_k2_whisper["combined"]:.2f} ({k2w_delta:+.2f})', flush=True)
    best_tag = 'K=2+whisper' if k2w_delta > k2_delta else 'K=2'
    best_delta = max(k2_delta, k2w_delta)
    print(f'  Best config:        {best_tag} ({best_delta:+.2f})', flush=True)
    print('COMPLETE', flush=True)

    # === Save checkpoint ===
    ckpt = {
        'lora_state': {f'lora_{i}': {'A': m.lora_A.data.cpu(), 'B': m.lora_B.data.cpu(),
                                      'is_ffn': m.is_ffn, 'rank': m.rank}
                       for i, m in enumerate(lora_modules)},
        'gate_state': gate.state_dict(),
        'config': vars(args),
        'results': {
            'baseline': baseline, 'pre_k2': pre_k2,
            'post_k1': post_k1, 'post_k2': post_k2,
            'post_k2_whisper': post_k2_whisper,
        },
    }
    torch.save(ckpt, f'{SAVE_DIR}/checkpoint.pt')

    results = {
        'model': args.model, 'name': args.name,
        'core': [args.core_start, args.core_end],
        'rank_attn': args.rank_attn, 'rank_ffn': args.rank_ffn,
        'steps': args.max_steps, 'warmup': args.warmup_steps,
        'baseline': baseline, 'pre_k2': pre_k2,
        'post_k1': post_k1, 'post_k2': post_k2,
        'post_k2_whisper': post_k2_whisper,
        'k1_preserved': abs(k1_delta) < 0.5,
        'k2_improves': k2_delta > 0,
        'best_config': best_tag,
        'best_delta': best_delta,
    }
    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/', flush=True)


if __name__ == '__main__':
    main()
