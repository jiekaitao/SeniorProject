"""
Norm-Preserving Projection at Duplication Seam

Hypothesis: LayerNorm after the seam normalizes everything anyway, so the
DIRECTION of h2 matters but not its magnitude. We found alpha=1.0 is always
optimal for scalar blending — this suggests the second pass's direction is
what matters, not its norm.

Tests:
  1. h1_norm_h2_dir: ||h1|| * (h2 / ||h2||) — same direction as h2, same norm as h1
  2. h2_norm_h1_dir: ||h2|| * (h1 / ||h1||) — same direction as h1, same norm as h2
  3. geomean_h2_dir: sqrt(||h1||*||h2||) * (h2 / ||h2||) — geometric mean norm, h2 direction
  4. per_token variants of (1)-(3): norms computed per-token not per-tensor

Baselines:
  - alpha=0: h1 only (no duplication effect)
  - alpha=1: h2 only (full duplication, standard RYS)
"""

import sys, os, json, time, torch, torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/Qwen3.5-9B'
RESULTS_PATH = Path('results/data/norm_preserving_results.json')

# Calibration prompts for displacement rho screen
CAL_PROMPTS = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'Calculate 15! / 13!',
    'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What emotion would someone feel after losing a close friend?',
    'How would a parent feel seeing their child graduate?',
]


# =============================================================================
# Norm-preservation functions (operate on hidden state tensors)
# =============================================================================

def norm_preserve_h1_norm_h2_dir(h1, h2, per_token=False):
    """Direction of h2, norm of h1."""
    if per_token:
        # Per-token: norms computed along hidden dim (last dim), keepdim for broadcasting
        h1_norm = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h2_norm = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return h1_norm * (h2 / h2_norm)
    else:
        h1_norm = h1.norm().clamp(min=1e-8)
        h2_norm = h2.norm().clamp(min=1e-8)
        return h1_norm * (h2 / h2_norm)


def norm_preserve_h2_norm_h1_dir(h1, h2, per_token=False):
    """Direction of h1, norm of h2 (reverse)."""
    if per_token:
        h1_norm = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h2_norm = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return h2_norm * (h1 / h1_norm)
    else:
        h1_norm = h1.norm().clamp(min=1e-8)
        h2_norm = h2.norm().clamp(min=1e-8)
        return h2_norm * (h1 / h1_norm)


def norm_preserve_geomean_h2_dir(h1, h2, per_token=False):
    """Direction of h2, geometric mean of norms."""
    if per_token:
        h1_norm = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h2_norm = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        geo_norm = torch.sqrt(h1_norm * h2_norm)
        return geo_norm * (h2 / h2_norm)
    else:
        h1_norm = h1.norm().clamp(min=1e-8)
        h2_norm = h2.norm().clamp(min=1e-8)
        geo_norm = torch.sqrt(h1_norm * h2_norm)
        return geo_norm * (h2 / h2_norm)


# Registry of all variants to test
NORM_VARIANTS = {
    'h1_norm_h2_dir': lambda h1, h2: norm_preserve_h1_norm_h2_dir(h1, h2, per_token=False),
    'h2_norm_h1_dir': lambda h1, h2: norm_preserve_h2_norm_h1_dir(h1, h2, per_token=False),
    'geomean_h2_dir': lambda h1, h2: norm_preserve_geomean_h2_dir(h1, h2, per_token=False),
    'h1_norm_h2_dir_pertoken': lambda h1, h2: norm_preserve_h1_norm_h2_dir(h1, h2, per_token=True),
    'h2_norm_h1_dir_pertoken': lambda h1, h2: norm_preserve_h2_norm_h1_dir(h1, h2, per_token=True),
    'geomean_h2_dir_pertoken': lambda h1, h2: norm_preserve_geomean_h2_dir(h1, h2, per_token=True),
}


# =============================================================================
# Displacement rho screen (fast block selection)
# =============================================================================

def compute_displacement_rho(inner, original_layers, device, tokenizer, block, prompts):
    """
    Compute displacement rho for a block: ||F(F(h)) - F(h)|| / ||F(h) - h||
    where F is the block [i, j).
    """
    i, j = block
    rhos = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = inner.embed_tokens(ids['input_ids'])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Run layers before block
            for l in range(i):
                out = original_layers[l](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            h_in = h.clone()

            # First pass through block
            h1 = h_in.clone()
            for l in range(i, j):
                out = original_layers[l](h1, position_embeddings=pos_embeds, use_cache=False)
                h1 = out[0] if isinstance(out, tuple) else out

            # Second pass through block
            h2 = h1.clone()
            for l in range(i, j):
                out = original_layers[l](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out

            num = torch.norm(h2 - h1).item()
            den = torch.norm(h1 - h_in).item()
            if den > 1e-8:
                rhos.append(num / den)

    return float(np.mean(rhos)) if rhos else 1.0


def find_best_block(inner, original_layers, N, device, tokenizer, step=4):
    """
    Quick displacement rho screen to find the best single block for duplication.
    Lower rho = block is more contractive = better for duplication.
    Tests block sizes 1,3,5,7 at the given step.
    """
    print("\n" + "=" * 60)
    print("DISPLACEMENT RHO SCREEN (step=%d)" % step)
    print("=" * 60)

    candidates = []
    for start in range(0, N - 1, step):
        for size in [1, 3, 5, 7]:
            end = start + size
            if end <= N:
                candidates.append((start, end))

    print(f"Screening {len(candidates)} blocks...")
    block_rhos = {}
    for idx, block in enumerate(candidates):
        rho = compute_displacement_rho(inner, original_layers, device, tokenizer, block, CAL_PROMPTS)
        block_rhos[block] = rho
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(candidates)}] ({block[0]:2d},{block[1]:2d}) rho={rho:.4f}")

    sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
    print("\nTop 10 blocks by displacement rho (lower = better):")
    for b, r in sorted_blocks[:10]:
        print(f"  ({b[0]:2d},{b[1]:2d}): rho={r:.4f}")

    best_block = sorted_blocks[0][0]
    print(f"\nBest block: ({best_block[0]},{best_block[1]}) rho={sorted_blocks[0][1]:.4f}")
    return best_block, sorted_blocks


# =============================================================================
# Layer-by-layer forward with seam intervention
# =============================================================================

def build_layer_order(block, N):
    """Build layer execution order with one block duplicated."""
    i, j = block
    order = list(range(j)) + list(range(i, j)) + list(range(j, N))
    return order


def find_seam_positions(layer_order, block_start, block_end):
    """
    Find step indices where first and second pass of a block end.
    Returns (first_pass_end_step, second_pass_end_step).
    """
    last_layer = block_end - 1
    occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
    if len(occurrences) < 2:
        raise ValueError(f"Block ({block_start},{block_end}) not duplicated in layer order. "
                         f"Layer {last_layer} appears {len(occurrences)} time(s)")
    return occurrences[0], occurrences[1]


def generate_with_norm_intervention(model, inner, tokenizer, prompt, layer_order,
                                     original_layers, block, intervention_fn,
                                     max_new_tokens=64):
    """
    Generate text with manual layer-by-layer forward pass.
    At the seam, applies intervention_fn(h1, h2) -> h_patched.

    intervention_fn: callable(h1, h2) -> h_patched
        h1 = hidden state after first pass through block
        h2 = hidden state after second pass through block

    Special cases:
        intervention_fn = None means alpha=1.0 (use h2 as-is, standard duplication)
        intervention_fn = 'alpha0' means alpha=0.0 (use h1, skip second pass effect)
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    first_end, second_end = find_seam_positions(layer_order, block[0], block[1])

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Embedding
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Layer-by-layer forward
            h_after_first_pass = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                # Cache h after first pass ends
                if step_idx == first_end:
                    h_after_first_pass = h.clone()

                # Apply intervention after second pass ends
                if step_idx == second_end and h_after_first_pass is not None:
                    if intervention_fn == 'alpha0':
                        h = h_after_first_pass  # discard second pass
                    elif intervention_fn is not None:
                        h = intervention_fn(h_after_first_pass, h)
                    # else: intervention_fn is None -> keep h2 as-is (alpha=1.0)

            # Final norm + LM head
            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    generated = input_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_variant(model, inner, tokenizer, layer_order, original_layers,
                     block, intervention_fn, name, verbose=True):
    """Run math_probe + eq_bench_probe for a norm-preservation variant."""
    def gen(p):
        return generate_with_norm_intervention(
            model, inner, tokenizer, p, layer_order, original_layers,
            block, intervention_fn, max_new_tokens=64)

    def gen_long(p):
        return generate_with_norm_intervention(
            model, inner, tokenizer, p, layer_order, original_layers,
            block, intervention_fn, max_new_tokens=128)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    elapsed = time.time() - t0

    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    if verbose:
        print(f"    {name:35s}: math={math_r['score']:.4f} eq={eq_r['score']:.1f} "
              f"combined={combined:.2f} ({elapsed:.0f}s)")

    return {
        'name': name,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined,
        'elapsed': elapsed,
    }


def evaluate_baseline_no_dup(model, tokenizer, name="baseline (no dup)"):
    """Run probes on the unmodified model (no duplication at all)."""
    from layer_duplicator import generate_no_cache

    def gen(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=64)

    def gen_long(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    elapsed = time.time() - t0

    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    print(f"    {name:35s}: math={math_r['score']:.4f} eq={eq_r['score']:.1f} "
          f"combined={combined:.2f} ({elapsed:.0f}s)")

    return {
        'name': name,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined,
        'elapsed': elapsed,
    }


# =============================================================================
# Norm analysis: measure how h1/h2 norms actually differ
# =============================================================================

def analyze_norms_at_seam(inner, original_layers, device, tokenizer, block, prompts):
    """
    Collect statistics on h1/h2 norms at the seam to understand the norm landscape.
    """
    i, j = block
    layer_order = build_layer_order(block, len(original_layers))
    first_end, second_end = find_seam_positions(layer_order, i, j)

    stats = {
        'h1_norms': [], 'h2_norms': [],
        'h1_token_norms': [], 'h2_token_norms': [],
        'cosine_sim': [], 'norm_ratio': [],
    }

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = inner.embed_tokens(ids['input_ids'])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_after_first = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                if step_idx == first_end:
                    h_after_first = h.clone()

                if step_idx == second_end and h_after_first is not None:
                    h1 = h_after_first.float()
                    h2 = h.float()

                    # Global norms
                    n1 = h1.norm().item()
                    n2 = h2.norm().item()
                    stats['h1_norms'].append(n1)
                    stats['h2_norms'].append(n2)
                    stats['norm_ratio'].append(n2 / max(n1, 1e-8))

                    # Per-token norms (mean across tokens)
                    tn1 = h1.norm(dim=-1).mean().item()
                    tn2 = h2.norm(dim=-1).mean().item()
                    stats['h1_token_norms'].append(tn1)
                    stats['h2_token_norms'].append(tn2)

                    # Cosine similarity (flatten)
                    cos = torch.nn.functional.cosine_similarity(
                        h1.reshape(1, -1), h2.reshape(1, -1)).item()
                    stats['cosine_sim'].append(cos)

    # Summarize
    summary = {}
    for k, v in stats.items():
        if v:
            summary[k] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'min': float(np.min(v)),
                'max': float(np.max(v)),
            }
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("NORM-PRESERVING PROJECTION TEST")
    print(f"Model: {MODEL_PATH}")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_original_model(MODEL_PATH)
    inner = model.model  # Qwen3_5TextModel with .layers, .embed_tokens, .norm, .rotary_emb
    original_layers = list(inner.layers)
    N = len(original_layers)
    device = next(model.parameters()).device
    print(f"Loaded: {N} layers on {device}")

    # Restore layers to original (in case load_original_model modified anything)
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N

    # =========================================================================
    # Step 1: Quick displacement rho screen (step=4)
    # =========================================================================
    best_block, sorted_blocks = find_best_block(
        inner, original_layers, N, device, tokenizer, step=4)

    # =========================================================================
    # Step 2: Analyze norms at the seam for the best block
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"NORM ANALYSIS AT SEAM — block ({best_block[0]},{best_block[1]})")
    print("=" * 60)

    norm_stats = analyze_norms_at_seam(
        inner, original_layers, device, tokenizer, best_block, CAL_PROMPTS)

    for k, v in norm_stats.items():
        print(f"  {k:20s}: mean={v['mean']:.4f} std={v['std']:.4f} "
              f"[{v['min']:.4f}, {v['max']:.4f}]")

    # =========================================================================
    # Step 3: Evaluate all variants
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"EVALUATING NORM-PRESERVATION VARIANTS — block ({best_block[0]},{best_block[1]})")
    print("=" * 60)

    layer_order = build_layer_order(best_block, N)
    all_results = []

    # --- Baseline: no duplication ---
    print("\n  [Baseline — no duplication]")
    # Restore original layers for baseline
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N
    r = evaluate_baseline_no_dup(model, tokenizer)
    all_results.append(r)
    baseline_combined = r['combined']

    # --- alpha=0 (use h1, discard second pass) ---
    print("\n  [alpha=0 — first pass only, discard second pass]")
    r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers,
                         best_block, 'alpha0', 'alpha=0 (h1 only)')
    all_results.append(r)

    # --- alpha=1 (standard duplication, use h2 as-is) ---
    print("\n  [alpha=1 — standard duplication]")
    r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers,
                         best_block, None, 'alpha=1 (h2 only, standard)')
    all_results.append(r)

    # --- Norm-preservation variants ---
    print("\n  [Norm-preservation variants]")
    for variant_name, variant_fn in NORM_VARIANTS.items():
        r = evaluate_variant(model, inner, tokenizer, layer_order, original_layers,
                             best_block, variant_fn, variant_name)
        all_results.append(r)

    # =========================================================================
    # Step 4: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — Norm-Preserving Projection Test")
    print("=" * 70)
    print(f"Model: {MODEL_PATH} ({N} layers)")
    print(f"Best block: ({best_block[0]},{best_block[1]})")
    print(f"Baseline combined: {baseline_combined:.2f}")
    print()
    print(f"{'Variant':40s} {'Math':>8s} {'EQ':>8s} {'Combined':>10s} {'Delta':>10s}")
    print("-" * 80)

    sorted_results = sorted(all_results, key=lambda x: x['combined'], reverse=True)
    for r in sorted_results:
        delta = r['combined'] - baseline_combined
        marker = " <-- BEST" if r == sorted_results[0] else ""
        print(f"{r['name']:40s} {r['math']:8.4f} {r['eq']:8.1f} "
              f"{r['combined']:10.2f} {delta:+10.2f}{marker}")

    # Key insight: does any norm variant beat alpha=1?
    alpha1_result = next(r for r in all_results if r['name'] == 'alpha=1 (h2 only, standard)')
    print("\n--- Comparison vs alpha=1 (standard duplication) ---")
    for r in sorted_results:
        if r['name'] != 'alpha=1 (h2 only, standard)':
            delta = r['combined'] - alpha1_result['combined']
            print(f"  {r['name']:40s}: {delta:+.2f}")

    # =========================================================================
    # Save results
    # =========================================================================
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        'metadata': {
            'model': MODEL_PATH,
            'num_layers': N,
            'best_block': list(best_block),
            'date': datetime.now().isoformat(),
            'step': 4,
        },
        'norm_analysis': norm_stats,
        'spectral_screen': [
            {'block': list(b), 'rho': r}
            for b, r in sorted_blocks[:20]
        ],
        'results': all_results,
        'summary': {
            'baseline_combined': baseline_combined,
            'best_variant': sorted_results[0]['name'],
            'best_combined': sorted_results[0]['combined'],
            'alpha1_combined': alpha1_result['combined'],
        },
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
