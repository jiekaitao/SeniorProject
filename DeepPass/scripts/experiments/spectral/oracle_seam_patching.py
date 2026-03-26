"""
Oracle Seam Patching — The Decisive Adapter Experiment

For each duplicated block, tests what happens when we scale the second-pass
contribution at the exit seam:

    h_patched = h1 + alpha * (h2 - h1)

where:
    h1 = output after first pass through block (same as base model at that point)
    h2 = output after second pass (the actual duplicated output)
    alpha = 0.0 (rollback: erase second pass)
           0.25, 0.5, 0.75 (partial)
           1.0 (no intervention, full duplication)

This tells us:
    - alpha=1.0 best: second pass is genuinely useful and seam-compatible
    - 0 < alpha < 1 best: second pass has useful signal but overshoots
    - alpha=0 best: second pass is mostly destructive

Tests on both good configs ((45,52) on 72B) and the best stacking pair.
"""
import sys, os, json, torch, torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe


class SeamPatchingHook:
    """
    Hooks into the forward pass to intercept hidden states at block boundaries
    and apply residual scaling at the exit seam.

    For a duplicated block (i,j), the execution order is:
        ... layer j-1 (first pass) -> layer i (second pass) -> ... -> layer j-1 (second pass) -> layer j ...

    We need to:
    1. Cache h1 (output after first pass through the block = input to second pass)
    2. After second pass completes, replace output with h1 + alpha*(h2 - h1)
    """

    def __init__(self, model, block_start, block_end, alpha=1.0):
        self.model = model
        self.inner = model.model
        self.block_start = block_start
        self.block_end = block_end
        self.alpha = alpha
        self.h_after_first_pass = None
        self.hooks = []

    def _find_seam_indices(self, layer_order):
        """Find the indices in layer_order where the first and second passes end."""
        # First pass ends at the first occurrence of block_end-1
        # Second pass ends at the second occurrence of block_end-1
        last_layer = self.block_end - 1
        occurrences = [i for i, idx in enumerate(layer_order) if idx == last_layer]
        if len(occurrences) < 2:
            raise ValueError(f"Block ({self.block_start},{self.block_end}) not duplicated in layer order")
        return occurrences[0], occurrences[1]

    def install(self, layer_order):
        """Install forward hooks on the relevant layers."""
        first_pass_end, second_pass_end = self._find_seam_indices(layer_order)

        layers_module = self.inner.layers

        # Hook after first pass: cache h1
        def cache_hook(module, input, output, step_idx=first_pass_end):
            h = output[0] if isinstance(output, tuple) else output
            self.h_after_first_pass = h.detach().clone()
            return output

        # Hook after second pass: apply alpha scaling
        def patch_hook(module, input, output, step_idx=second_pass_end):
            if self.h_after_first_pass is None:
                return output

            h2 = output[0] if isinstance(output, tuple) else output
            h1 = self.h_after_first_pass.to(h2.device, h2.dtype)

            h_patched = h1 + self.alpha * (h2 - h1)

            self.h_after_first_pass = None  # Reset for next forward pass

            if isinstance(output, tuple):
                return (h_patched,) + output[1:]
            return h_patched

        hook1 = layers_module[first_pass_end].register_forward_hook(cache_hook)
        hook2 = layers_module[second_pass_end].register_forward_hook(patch_hook)
        self.hooks = [hook1, hook2]

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.h_after_first_pass = None


def build_layer_order(blocks, N):
    """Build execution order for multiple blocks."""
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))
    return order


def evaluate_with_alpha(model, tokenizer, inner, original_layers, N, blocks, alpha_configs):
    """
    For each block in blocks, test different alpha values at that block's exit seam.

    alpha_configs: list of alpha values to test
    """
    results = []

    # Build the duplicated model
    order = build_layer_order(blocks, N)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)

    def gen(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=64)

    def gen_long(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    # First: baseline (alpha=1.0 for all blocks, i.e., normal duplication)
    print(f"\n  Testing alpha=1.0 (normal duplication)...")
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    print(f"    alpha=1.0: math={math_r['score']:.4f} eq={eq_r['score']:.1f} combined={combined:.2f}")
    results.append({
        'block_patched': 'none',
        'alpha': 1.0,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined
    })

    # For each block, test different alphas
    for block in blocks:
        bi, bj = block
        print(f"\n  Patching block ({bi},{bj}):")

        for alpha in alpha_configs:
            if alpha == 1.0:
                continue  # Already tested above

            patcher = SeamPatchingHook(model, bi, bj, alpha=alpha)
            patcher.install(order)

            math_r = run_math_probe(gen, verbose=False)
            eq_r = run_eq_bench_probe(gen_long, verbose=False)
            combined = math_r['score'] * 50 + eq_r['score'] * 0.5
            print(f"    alpha={alpha:.2f} on ({bi},{bj}): math={math_r['score']:.4f} eq={eq_r['score']:.1f} combined={combined:.2f}")

            results.append({
                'block_patched': f"({bi},{bj})",
                'alpha': alpha,
                'math': math_r['score'],
                'eq': eq_r['score'],
                'combined': combined
            })

            patcher.remove()

    # Restore model
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    return results


def main():
    print("=== ORACLE SEAM PATCHING ON 72B ===")
    model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f"Loaded: {N} layers")

    alphas = [0.0, 0.25, 0.5, 0.75]  # 1.0 is tested as baseline

    all_results = {}

    # --- Test 1: Single block (45,52) — Ng's config ---
    print("\n" + "="*60)
    print("TEST 1: Single block (45,52) — Ng's config")
    print("="*60)
    r = evaluate_with_alpha(model, tokenizer, inner, original_layers, N,
                            [(45, 52)], alphas)
    all_results['single_45_52'] = r

    # --- Test 2: Single block (50,60) — our best single ---
    print("\n" + "="*60)
    print("TEST 2: Single block (50,60) — our best single")
    print("="*60)
    r = evaluate_with_alpha(model, tokenizer, inner, original_layers, N,
                            [(50, 60)], alphas)
    all_results['single_50_60'] = r

    # --- Test 3: Best pair (0,7)+(45,52) — our best combined ---
    print("\n" + "="*60)
    print("TEST 3: Best pair (0,7)+(45,52) — our best combined")
    print("="*60)
    # Test patching each block's seam independently
    r = evaluate_with_alpha(model, tokenizer, inner, original_layers, N,
                            [(0, 7), (45, 52)], alphas)
    all_results['pair_0_7_45_52'] = r

    # --- Test 4: Single block (0,7) alone — to understand its contribution ---
    print("\n" + "="*60)
    print("TEST 4: Single block (0,7) — early block alone")
    print("="*60)
    r = evaluate_with_alpha(model, tokenizer, inner, original_layers, N,
                            [(0, 7)], alphas)
    all_results['single_0_7'] = r

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY — Optimal Alpha per Config")
    print("="*60)

    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        # Group by block_patched
        by_block = {}
        for r in results:
            key = r['block_patched']
            if key not in by_block:
                by_block[key] = []
            by_block[key].append(r)

        for block, entries in by_block.items():
            best = max(entries, key=lambda x: x['combined'])
            for e in sorted(entries, key=lambda x: x['alpha']):
                marker = " <-- BEST" if e['alpha'] == best['alpha'] else ""
                print(f"  block={block:12s} alpha={e['alpha']:.2f}: "
                      f"math={e['math']:.4f} eq={e['eq']:.1f} combined={e['combined']:.2f}{marker}")

    # Save
    with open('results/72b_oracle_seam_patching.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/72b_oracle_seam_patching.json")


if __name__ == '__main__':
    main()
