"""
K-Degradation Sweep: Why does higher K hurt accuracy?

Tests K=1,2,3,4 with three modes at each K:
1. Full duplication (attention + FFN)
2. Attention-only (FFN scaled by beta=0.0 on pass 2+)
3. FFN-whisper (FFN scaled by beta=0.2 on pass 2+)

Also measures:
- Hidden state L2 norm at the seam (how much does pass 2 change the representation?)
- Logit entropy (does more recursion = more or less confident?)
- Per-component breakdown (math vs EQ-bench)

This directly tests the mechanistic hypothesis:
  "Attention re-computation refines reasoning, FFN re-computation corrupts memory"

If true, we should see:
  - Attention-only: monotonic improvement or at least stable with K
  - Full: peaks at K=2, degrades at K=3+
  - FFN-whisper: intermediate

Usage:
    python k_degradation_sweep.py --model <path> --name <name> --core_start 10 --core_end 13
"""

import os, sys, json, time, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Build layer order with K passes through [core_start, core_end)."""
    inner = get_inner(model)
    original_layers = list(inner.layers)
    N = len(original_layers)

    # Build order: prelude + core*K + coda
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


def setup_ffn_scaling(model, core_start, core_end, beta):
    """Hook MLP modules of core layers to scale output by beta on pass 2+.

    Detection: LayerIdxWrapper instances have new_layer_idx != orig_idx.
    We track fire counts per physical layer — first fire = pass 1, subsequent = pass 2+.
    """
    inner = get_inner(model)
    original_layers_list = list(inner.layers)

    # Identify physical core layer objects
    physical_core = {}
    for i in range(core_start, min(core_end, len(original_layers_list))):
        layer = original_layers_list[i]
        if isinstance(layer, LayerIdxWrapper):
            physical_core[id(layer.layer)] = layer.layer
        else:
            physical_core[id(layer)] = layer

    hooks = []
    fire_counts = {}

    for layer_id, layer in physical_core.items():
        fire_counts[layer_id] = [0]

        def make_mlp_hook(counter, b):
            def hook(module, input, output):
                counter[0] += 1
                if counter[0] > 1:  # pass 2+
                    if isinstance(output, tuple):
                        return tuple(o * b if isinstance(o, torch.Tensor) else o for o in output)
                    return output * b
                return output
            return hook

        h = layer.mlp.register_forward_hook(make_mlp_hook(fire_counts[layer_id], beta))
        hooks.append(h)

    return hooks, fire_counts


def reset_ffn_counts(fire_counts):
    for k in fire_counts:
        fire_counts[k][0] = 0


def measure_hidden_state_change(model, tokenizer, device, core_start, core_end, K):
    """Measure L2 norm of hidden state change between K=1 and K=K at the seam point."""
    inner = get_inner(model)

    # Use a few test prompts
    prompts = [
        "The capital of France is",
        "To solve this equation, we first need to",
        "The feeling of sadness when remembering",
        "If x squared equals 144, then x is",
    ]

    norms = []
    entropies_k1 = []
    entropies_kK = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(device)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            # K=1 logits
            out1 = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
            logits1 = out1.logits[:, -1, :]
            h1 = out1.hidden_states[core_end] if hasattr(out1, 'hidden_states') and out1.hidden_states else None

            # Entropy at K=1
            probs1 = F.softmax(logits1, dim=-1)
            ent1 = -(probs1 * (probs1 + 1e-10).log()).sum(-1).item()
            entropies_k1.append(ent1)

        if K > 1:
            orig, N = build_k_recursive(model, core_start, core_end, K)
            with torch.no_grad():
                outK = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
                logitsK = outK.logits[:, -1, :]

                probsK = F.softmax(logitsK, dim=-1)
                entK = -(probsK * (probsK + 1e-10).log()).sum(-1).item()
                entropies_kK.append(entK)

                # KL divergence
                kl = F.kl_div(F.log_softmax(logitsK, dim=-1),
                              F.softmax(logits1, dim=-1),
                              reduction='batchmean').item()
                norms.append(kl)

            restore(model, orig, N)
        else:
            entropies_kK = entropies_k1[:]
            norms.append(0.0)

    return {
        'avg_kl_divergence': sum(norms) / len(norms),
        'avg_entropy_k1': sum(entropies_k1) / len(entropies_k1),
        'avg_entropy_kK': sum(entropies_kK) / len(entropies_kK),
        'entropy_delta': sum(entropies_kK) / len(entropies_kK) - sum(entropies_k1) / len(entropies_k1),
    }


def evaluate(model, tokenizer, device, tag=""):
    model.eval()
    for m in model.modules():
        m.training = False
    eq_all = _load_questions()

    def gen(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            out = model.generate(**kw, max_new_tokens=64, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def gen_long(p):
        inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512).to(device)
        kw = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.no_grad():
            out = model.generate(**kw, max_new_tokens=128, do_sample=False, use_cache=False)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    elapsed = time.time() - t0
    print(f'  {tag}: math={math_r["score"]:.4f} eq={eq_r["score"]:.1f} '
          f'combined={combined:.2f} ({elapsed:.0f}s)', flush=True)
    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cache_dir', default='/blue/cis4914/jietao/hf_cache')
    parser.add_argument('--core_start', type=int, required=True)
    parser.add_argument('--core_end', type=int, required=True)
    parser.add_argument('--max_k', type=int, default=4)
    args = parser.parse_args()

    SAVE_DIR = f'results/data/k_degradation/{args.name}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda')

    print(f'Loading {args.model}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, device_map='auto',
        dtype=torch.bfloat16, trust_remote_code=True,
    )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    all_results = {}

    # === K=1 baseline (always first) ===
    print(f'\n=== K=1 (baseline) ===', flush=True)
    baseline = evaluate(model, tokenizer, device, 'K=1 baseline')
    hs_k1 = measure_hidden_state_change(model, tokenizer, device,
                                         args.core_start, args.core_end, 1)
    all_results['K=1'] = {**baseline, **hs_k1, 'mode': 'baseline'}

    # === Sweep K=2..max_k with three modes ===
    modes = [
        ('full', 1.0),      # Full duplication (FFN beta=1.0)
        ('attn_only', 0.0), # Attention-only (FFN beta=0.0 on pass 2+)
        ('whisper', 0.2),   # FFN whisper (FFN beta=0.2 on pass 2+)
    ]

    for K in range(2, args.max_k + 1):
        for mode_name, beta in modes:
            tag = f'K={K}_{mode_name}'
            print(f'\n=== {tag} (beta_ffn={beta}) ===', flush=True)

            orig, N = build_k_recursive(model, args.core_start, args.core_end, K)

            if beta < 1.0:
                ffn_hooks, ffn_counts = setup_ffn_scaling(
                    model, args.core_start, args.core_end, beta)
            else:
                ffn_hooks, ffn_counts = [], {}

            # Hidden state analysis (before generation, to avoid hook interference)
            # Do it without FFN hooks for clean comparison
            if ffn_hooks:
                for h in ffn_hooks:
                    h.remove()

            restore(model, orig, N)
            hs = measure_hidden_state_change(model, tokenizer, device,
                                              args.core_start, args.core_end, K)

            # Now evaluate with FFN scaling
            orig, N = build_k_recursive(model, args.core_start, args.core_end, K)
            if beta < 1.0:
                ffn_hooks, ffn_counts = setup_ffn_scaling(
                    model, args.core_start, args.core_end, beta)

            # Wrap generate to reset FFN counts per call
            original_generate = model.generate
            def make_patched_gen(fc, orig_gen):
                def patched(*a, **kw):
                    kw['use_cache'] = False
                    for k in fc:
                        fc[k][0] = 0
                    return orig_gen(*a, **kw)
                return patched

            if ffn_counts:
                model.generate = make_patched_gen(ffn_counts, original_generate)

            result = evaluate(model, tokenizer, device, tag)

            if ffn_counts:
                model.generate = original_generate

            for h in ffn_hooks:
                h.remove()
            restore(model, orig, N)

            all_results[tag] = {**result, **hs, 'mode': mode_name, 'K': K, 'beta_ffn': beta}

    # === Summary table ===
    print(f'\n{"=" * 80}', flush=True)
    print(f'K-DEGRADATION SWEEP — {args.name}', flush=True)
    print(f'Core: [{args.core_start}, {args.core_end})', flush=True)
    print(f'{"=" * 80}', flush=True)
    print(f'{"Config":<20} {"Math":>8} {"EQ":>8} {"Combined":>10} {"Delta":>8} '
          f'{"KL_div":>8} {"Ent_delta":>10}', flush=True)
    print(f'{"-" * 80}', flush=True)

    baseline_comb = all_results['K=1']['combined']
    for key in sorted(all_results.keys(), key=lambda x: (
        int(x.split('=')[1].split('_')[0]),
        {'baseline': 0, 'full': 1, 'attn_only': 2, 'whisper': 3}.get(
            all_results[x]['mode'], 4)
    )):
        r = all_results[key]
        delta = r['combined'] - baseline_comb
        kl = r.get('avg_kl_divergence', 0)
        ent_d = r.get('entropy_delta', 0)
        print(f'{key:<20} {r["math"]:>8.4f} {r["eq"]:>8.1f} {r["combined"]:>10.2f} '
              f'{delta:>+8.2f} {kl:>8.4f} {ent_d:>+10.4f}', flush=True)

    print(f'\nKey findings:', flush=True)

    # Check: does attention-only degrade less?
    for K in range(2, args.max_k + 1):
        full_key = f'K={K}_full'
        attn_key = f'K={K}_attn_only'
        if full_key in all_results and attn_key in all_results:
            full_d = all_results[full_key]['combined'] - baseline_comb
            attn_d = all_results[attn_key]['combined'] - baseline_comb
            print(f'  K={K}: full={full_d:+.2f}, attn_only={attn_d:+.2f}, '
                  f'FFN harm={full_d - attn_d:+.2f}', flush=True)

    print('COMPLETE', flush=True)

    with open(f'{SAVE_DIR}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Saved to {SAVE_DIR}/results.json', flush=True)


if __name__ == '__main__':
    main()
