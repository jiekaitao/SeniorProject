"""
Step-Decayed FFN: Test monotone FFN schedules on duplicated models.

Key insight from K-sweep + GPT-5.4 Pro analysis:
  K=2 benefits from FFN (full > whisper > attn-only)
  K=3+ FFN becomes destructive
  => Front-load FFN, decay toward attention-only on later passes

Tests multiple beta schedules on Mistral [28,29):
  Schedule A: [1.0, 0.2, 0.0, 0.0]  — full first pass, whisper second, attn-only after
  Schedule B: [1.0, 0.25, 0.05, 0.0]
  Schedule C: [0.8, 0.2, 0.05, 0.0]
  Schedule D: [1.0, 0.5, 0.1, 0.0]  — slower decay

Each schedule tested at K=2, K=3, K=4.

Usage:
    python step_decayed_ffn.py --model <path> --name <name> --core_start 28 --core_end 29
"""

import os, sys, json, time, argparse
import torch
import torch.nn as nn

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


def get_inner(model):
    inner = model.model if hasattr(model, 'model') else model.transformer
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    return inner


def build_k_recursive(model, core_start, core_end, K):
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)
    order = list(range(core_start))
    for _ in range(K):
        order.extend(range(core_start, core_end))
    order.extend(range(core_end, N))

    seen = set()
    new_layers = []
    for pi, oi in enumerate(order):
        l = original_layers[oi]
        if oi in seen:
            new_layers.append(LayerIdxWrapper(l, pi))
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
    return original_layers, N


def restore(model, original_layers, N):
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


class StepDecayedFFNHook:
    """Scales MLP output based on which pass we're on.

    Uses a per-step beta schedule. Tracks which pass via fire count
    on the physical layer. Resets at the start of each model.generate() call.
    """

    def __init__(self, physical_layer, betas):
        self.betas = betas  # [beta_pass1, beta_pass2, beta_pass3, ...]
        self.fire_count = 0
        self.hook = physical_layer.mlp.register_forward_hook(self._hook_fn)
        self._active = True

    def _hook_fn(self, module, input, output):
        if not self._active:
            return output
        # Determine which pass this is (0-indexed)
        pass_idx = self.fire_count
        self.fire_count += 1

        if pass_idx < len(self.betas):
            beta = self.betas[pass_idx]
        else:
            beta = self.betas[-1]  # use last beta for any extra passes

        if beta != 1.0:
            if isinstance(output, tuple):
                return tuple(o * beta if isinstance(o, torch.Tensor) else o for o in output)
            return output * beta
        return output

    def reset(self):
        self.fire_count = 0

    def remove(self):
        self.hook.remove()


def setup_step_decay(model, core_start, core_end, betas):
    """Install step-decayed FFN hooks on core layers."""
    inner = get_inner(model)
    original_layers = list(inner.layers)
    hooks = []
    for i in range(core_start, core_end):
        layer = original_layers[i]
        h = StepDecayedFFNHook(layer, betas)
        hooks.append(h)
    return hooks


def evaluate(model, tokenizer, device, decay_hooks=None, tag=""):
    model.eval()
    for m in model.modules():
        m.training = False
    eq_all = _load_questions()

    # Wrap generate to reset fire counts per call
    original_generate = model.generate

    def patched_generate(*a, **kw):
        if decay_hooks:
            for h in decay_hooks:
                h.reset()
        kw['use_cache'] = False
        return original_generate(*a, **kw)

    model.generate = patched_generate

    def gen(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            out = model.generate(**kw, max_new_tokens=64, do_sample=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gen_long(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            out = model.generate(**kw, max_new_tokens=128, do_sample=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} '
          f'combined={combined:.2f} ({elapsed:.0f}s)', flush=True)

    model.generate = original_generate
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, required=True)
    parser.add_argument('--core_end', type=int, required=True)
    args = parser.parse_args()

    SAVE_DIR = f'results/data/step_decayed/{args.name}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    results = {}

    # Baseline K=1
    print('\n=== K=1 baseline ===', flush=True)
    results['K=1'] = evaluate(model, tokenizer, device, tag='K=1')
    baseline = results['K=1']['combined']

    # Beta schedules to test (indexed by pass: pass1, pass2, pass3, pass4)
    schedules = {
        'full':      [1.0, 1.0, 1.0, 1.0],
        'A':         [1.0, 0.2, 0.0, 0.0],
        'B':         [1.0, 0.25, 0.05, 0.0],
        'C':         [0.8, 0.2, 0.05, 0.0],
        'D':         [1.0, 0.5, 0.1, 0.0],
        'attn_only': [1.0, 0.0, 0.0, 0.0],
    }

    for K in [2, 3, 4]:
        for sched_name, betas in schedules.items():
            tag = f'K={K}_{sched_name}'
            print(f'\n=== {tag} beta={betas[:K]} ===', flush=True)

            # Build K-recursive model
            orig, N = build_k_recursive(model, args.core_start, args.core_end, K)

            # Install step-decayed hooks
            decay_hooks = setup_step_decay(model, args.core_start, args.core_end, betas)

            result = evaluate(model, tokenizer, device, decay_hooks=decay_hooks, tag=tag)
            results[tag] = {**result, 'K': K, 'schedule': sched_name, 'betas': betas[:K]}

            # Cleanup
            for h in decay_hooks:
                h.remove()
            restore(model, orig, N)

    # Summary
    print(f'\n{"=" * 80}', flush=True)
    print(f'STEP-DECAYED FFN SWEEP — {args.name}', flush=True)
    print(f'Core: [{args.core_start}, {args.core_end})', flush=True)
    print(f'{"=" * 80}', flush=True)
    print(f'{"Config":<25} {"Math":>8} {"EQ":>8} {"Combined":>10} {"Delta":>8}', flush=True)
    print(f'{"-" * 65}', flush=True)

    print(f'{"K=1 baseline":<25} {results["K=1"]["math"]:>8.4f} {results["K=1"]["eq"]:>8.1f} '
          f'{results["K=1"]["combined"]:>10.2f} {0:>+8.2f}', flush=True)

    for K in [2, 3, 4]:
        print(f'--- K={K} ---', flush=True)
        for sched_name in schedules:
            key = f'K={K}_{sched_name}'
            if key in results:
                r = results[key]
                delta = r['combined'] - baseline
                marker = ' ***' if delta > 0 and delta == max(
                    results.get(f'K={K}_{s}', {}).get('combined', 0) - baseline
                    for s in schedules
                ) else ''
                print(f'  {sched_name:<21} {r["math"]:>8.4f} {r["eq"]:>8.1f} '
                      f'{r["combined"]:>10.2f} {delta:>+8.2f}{marker}', flush=True)

    # Best overall
    best_key = max((k for k in results if k != 'K=1'),
                   key=lambda k: results[k]['combined'])
    best_delta = results[best_key]['combined'] - baseline
    print(f'\nBest: {best_key} ({best_delta:+.2f})', flush=True)
    print('COMPLETE', flush=True)

    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/', flush=True)


if __name__ == '__main__':
    main()
