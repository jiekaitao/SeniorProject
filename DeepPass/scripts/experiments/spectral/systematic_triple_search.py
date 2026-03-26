"""
Systematic Triple Search Pipeline for 72B

The approach that found (0,7)+(45,52) in 20 evaluations instead of 3,241,
now applied recursively to find the best triple.

Pipeline:
  Step 1: Spectral screen ALL candidate blocks → top 20
  Step 2: For all ~190 pairs from top 20, compute conditional rho → rank pairs
  Step 3: Dual-probe top 10 pairs → find best pair
  Step 4: For best 3 pairs, spectral screen all third blocks → rank triples
  Step 5: Dual-probe top 5 triples → find best triple

Estimated time: ~2.5 hours on 1 GPU
"""
import sys, os, json, torch, torch.nn as nn, numpy as np
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe


def build_layer_order(blocks, N):
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))
    return order


def compute_displacement_rho(inner, original_layers, N, block, tokenizer, device, prompts, n=8):
    """Compute displacement rho for a block on the current model state."""
    i, j = block
    rhos = []
    for prompt in prompts[:n]:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            h = inner.embed_tokens(ids["input_ids"])
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # Run to block start using CURRENT model layers
            current_N = len(inner.layers)
            for layer_idx in range(min(i, current_N)):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            h_input = h.clone()

            # First pass through block using ORIGINAL layers
            h1 = h_input.clone()
            for layer_idx in range(i, j):
                out = original_layers[layer_idx](h1, position_embeddings=pos_embeds, use_cache=False)
                h1 = out[0] if isinstance(out, tuple) else out

            # Second pass
            h2 = h1.clone()
            for layer_idx in range(i, j):
                out = original_layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                h2 = out[0] if isinstance(out, tuple) else out

            num = torch.norm(h2 - h1).item()
            den = torch.norm(h1 - h_input).item()
            if den > 1e-8:
                rhos.append(num / den)
    return float(np.mean(rhos)) if rhos else 1.0


def dual_probe(model, tokenizer, inner, original_layers, N, blocks):
    """Run both math probe and EQ-bench on a given block configuration."""
    if blocks:
        order = build_layer_order(blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)

    def gen(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=64)
    def gen_long(p): return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    if blocks:
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

    return {'math': math_r['score'], 'eq': eq_r['score'], 'combined': combined}


def non_overlapping(blocks):
    """Check if a set of blocks has no overlaps."""
    sorted_b = sorted(blocks)
    for idx in range(len(sorted_b) - 1):
        if sorted_b[idx][1] > sorted_b[idx + 1][0]:
            return False
    return True


# Calibration prompts for spectral screening
CAL_PROMPTS = [
    "What is 127 * 348?",
    "What is 99999 * 99999?",
    "Calculate 15! / 13!",
    "What is 2^16?",
    "What is the sum of all integers from 1 to 100?",
    "If f(x) = 3x^2 - 2x + 1, what is f(5)?",
    "What emotion would someone feel after losing a close friend?",
    "How would a parent feel seeing their child graduate?",
]


def main():
    print("=" * 70)
    print("SYSTEMATIC TRIPLE SEARCH PIPELINE — 72B")
    print("=" * 70)

    model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    device = next(model.parameters()).device
    print(f"Loaded: {N} layers\n")

    all_results = {
        'step1_spectral': [],
        'step2_pair_screen': [],
        'step3_pair_probe': [],
        'step4_triple_screen': [],
        'step5_triple_probe': [],
    }

    # ===========================================================
    # STEP 1: Spectral screen all candidate blocks
    # ===========================================================
    print("=" * 50)
    print("STEP 1: Spectral screening (all candidate blocks)")
    print("=" * 50)

    candidates = []
    for start in range(0, 65, 5):
        for size in [5, 7, 10]:
            end = start + size
            if end <= N:
                candidates.append((start, end))

    # Also add known good blocks
    for b in [(0, 7), (10, 15), (15, 20), (45, 52), (50, 60), (55, 60), (35, 40)]:
        if b not in candidates:
            candidates.append(b)

    print(f"  Screening {len(candidates)} candidate blocks...")
    block_rhos = {}
    for block in candidates:
        rho = compute_displacement_rho(inner, original_layers, N, block, tokenizer, device, CAL_PROMPTS)
        block_rhos[block] = rho

    # Rank by rho (lower = better for duplication)
    sorted_blocks = sorted(block_rhos.items(), key=lambda x: x[1])
    top_20 = [b for b, r in sorted_blocks[:20]]

    print(f"\n  Top 20 blocks by displacement rho:")
    for b, r in sorted_blocks[:20]:
        print(f"    ({b[0]:2d},{b[1]:2d}): rho={r:.4f}")
    all_results['step1_spectral'] = [{'block': list(b), 'rho': r} for b, r in sorted_blocks]

    # ===========================================================
    # STEP 2: Screen all pairs with conditional rho
    # ===========================================================
    print(f"\n{'=' * 50}")
    print("STEP 2: Pair screening (conditional rho)")
    print("=" * 50)

    # Generate all non-overlapping pairs from top 20
    pairs = [(a, b) for a, b in combinations(top_20, 2) if non_overlapping([a, b])]
    print(f"  {len(pairs)} non-overlapping pairs from top-20 blocks")

    pair_scores = []
    for a, b in pairs:
        # Apply block a
        order_a = build_layer_order([a], N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_a])
        model.config.num_hidden_layers = len(order_a)

        # Compute rho of b on modified model
        rho_b_cond = compute_displacement_rho(inner, original_layers, N, b, tokenizer, device, CAL_PROMPTS, n=4)

        # Restore
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

        # Also compute in reverse
        order_b = build_layer_order([b], N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_b])
        model.config.num_hidden_layers = len(order_b)

        rho_a_cond = compute_displacement_rho(inner, original_layers, N, a, tokenizer, device, CAL_PROMPTS, n=4)

        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

        # Pair score: lower combined conditional rho = better
        pair_rho = (rho_a_cond + rho_b_cond) / 2
        pair_scores.append({
            'pair': [list(a), list(b)],
            'rho_a': block_rhos[a],
            'rho_b': block_rhos[b],
            'rho_a_cond': rho_a_cond,
            'rho_b_cond': rho_b_cond,
            'pair_rho': pair_rho,
        })

    pair_scores.sort(key=lambda x: x['pair_rho'])

    print(f"\n  Top 15 pairs by conditional rho:")
    for ps in pair_scores[:15]:
        a, b = tuple(ps['pair'][0]), tuple(ps['pair'][1])
        print(f"    ({a[0]:2d},{a[1]:2d})+({b[0]:2d},{b[1]:2d}): pair_rho={ps['pair_rho']:.4f}")
    all_results['step2_pair_screen'] = pair_scores

    # ===========================================================
    # STEP 3: Dual-probe top 10 pairs
    # ===========================================================
    print(f"\n{'=' * 50}")
    print("STEP 3: Dual-probe evaluation (top 10 pairs)")
    print("=" * 50)

    # Always include known good pairs for comparison
    must_eval = [
        [(0, 7), (45, 52)],    # our best
        [(15, 20), (50, 60)],  # second best
    ]

    eval_pairs = []
    for ps in pair_scores[:10]:
        pair = [tuple(ps['pair'][0]), tuple(ps['pair'][1])]
        if pair not in eval_pairs:
            eval_pairs.append(pair)
    for mp in must_eval:
        if mp not in eval_pairs:
            eval_pairs.append(mp)

    pair_probe_results = []
    for blocks in eval_pairs:
        name = "+".join(f"({b[0]},{b[1]})" for b in blocks)
        print(f"  Probing {name}...")
        r = dual_probe(model, tokenizer, inner, original_layers, N, blocks)
        print(f"    math={r['math']:.4f} eq={r['eq']:.1f} combined={r['combined']:.2f}")
        pair_probe_results.append({
            'blocks': [list(b) for b in blocks],
            'name': name,
            **r
        })

    pair_probe_results.sort(key=lambda x: -x['combined'])
    all_results['step3_pair_probe'] = pair_probe_results

    print(f"\n  PAIR RANKING (by combined score):")
    for r in pair_probe_results:
        print(f"    {r['name']:30s}: combined={r['combined']:.2f}")

    # ===========================================================
    # STEP 4: Screen third blocks for top 3 pairs
    # ===========================================================
    print(f"\n{'=' * 50}")
    print("STEP 4: Triple screening (conditional rho for third block)")
    print("=" * 50)

    top_3_pairs = pair_probe_results[:3]
    triple_candidates = []

    for pair_result in top_3_pairs:
        pair_blocks = [tuple(b) for b in pair_result['blocks']]
        pair_name = pair_result['name']
        print(f"\n  Extending {pair_name} (combined={pair_result['combined']:.2f}):")

        # Apply the pair
        order_pair = build_layer_order(pair_blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order_pair])
        model.config.num_hidden_layers = len(order_pair)

        # Screen all candidate third blocks
        third_scores = []
        for block in candidates:
            if not non_overlapping(pair_blocks + [block]):
                continue
            rho_c = compute_displacement_rho(inner, original_layers, N, block, tokenizer, device, CAL_PROMPTS, n=4)
            third_scores.append({'block': list(block), 'rho': rho_c})

        # Restore
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

        third_scores.sort(key=lambda x: x['rho'])
        print(f"    Top 5 third blocks:")
        for ts in third_scores[:5]:
            b = ts['block']
            print(f"      ({b[0]:2d},{b[1]:2d}): rho={ts['rho']:.4f}")

        # Save top 5 candidates for each pair
        for ts in third_scores[:5]:
            triple_candidates.append({
                'pair': pair_result,
                'third': ts,
                'triple_blocks': [list(b) for b in pair_blocks] + [ts['block']],
            })

    all_results['step4_triple_screen'] = triple_candidates

    # ===========================================================
    # STEP 5: Dual-probe top triples
    # ===========================================================
    print(f"\n{'=' * 50}")
    print("STEP 5: Dual-probe evaluation (top triples)")
    print("=" * 50)

    # Deduplicate triples (same set of blocks)
    seen = set()
    unique_triples = []
    for tc in triple_candidates:
        key = tuple(sorted(tuple(b) for b in tc['triple_blocks']))
        if key not in seen:
            seen.add(key)
            unique_triples.append(tc)

    # Limit to top 8
    unique_triples = unique_triples[:8]

    triple_probe_results = []
    for tc in unique_triples:
        blocks = [tuple(b) for b in tc['triple_blocks']]
        name = "+".join(f"({b[0]},{b[1]})" for b in sorted(blocks))
        print(f"  Probing {name}...")
        r = dual_probe(model, tokenizer, inner, original_layers, N, sorted(blocks))
        print(f"    math={r['math']:.4f} eq={r['eq']:.1f} combined={r['combined']:.2f}")
        triple_probe_results.append({
            'blocks': [list(b) for b in sorted(blocks)],
            'name': name,
            **r
        })

    triple_probe_results.sort(key=lambda x: -x['combined'])
    all_results['step5_triple_probe'] = triple_probe_results

    # ===========================================================
    # FINAL SUMMARY
    # ===========================================================
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS — SYSTEMATIC TRIPLE SEARCH")
    print("=" * 70)

    # Known references
    print(f"\n  REFERENCE SCORES:")
    print(f"    Baseline:              combined=70.52")
    print(f"    Ng (45,52):            combined≈76.76")
    print(f"    Our (50,60) single:    combined≈79.66")
    print(f"    Our (0,7)+(45,52):     combined=79.91")

    print(f"\n  BEST PAIRS (this run):")
    for r in pair_probe_results[:5]:
        delta = r['combined'] - 79.91
        print(f"    {r['name']:35s}: combined={r['combined']:.2f} (vs best pair: {delta:+.2f})")

    print(f"\n  BEST TRIPLES (this run):")
    for r in triple_probe_results[:5]:
        delta = r['combined'] - 79.91
        print(f"    {r['name']:45s}: combined={r['combined']:.2f} (vs best pair: {delta:+.2f})")

    best_triple = triple_probe_results[0] if triple_probe_results else None
    if best_triple and best_triple['combined'] > 79.91:
        print(f"\n  *** TRIPLE BEATS BEST PAIR! ***")
        print(f"  {best_triple['name']}: combined={best_triple['combined']:.2f}")
    else:
        print(f"\n  No triple beats best pair (0,7)+(45,52) = 79.91")

    # Save
    os.makedirs('results/data/72b/triples', exist_ok=True)
    with open('results/data/72b/triples/systematic_search.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/data/72b/triples/systematic_search.json")


if __name__ == '__main__':
    main()
