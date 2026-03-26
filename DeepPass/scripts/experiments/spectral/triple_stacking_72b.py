"""
Triple-block stacking on 72B.

Takes our best pair (0,7)+(45,52) (combined=79.91) and tests adding a third
block from various regions to see if 3-block stacking can push higher.

Also tests adding a third block to (15,20)+(50,60) (combined=76.90).
"""
import sys, os, json, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe


def build_multi_block_order(blocks, N):
    """Build execution order for multiple non-overlapping blocks."""
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(list(range(prev, j)))
        order.extend(list(range(i, j)))
        prev = j
    order.extend(list(range(prev, N)))
    return order


def evaluate_config(model, tokenizer, inner, original_layers, N, blocks, name):
    """Apply blocks, run dual probe, restore model."""
    if blocks:
        order = build_multi_block_order(blocks, N)
        inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
        model.config.num_hidden_layers = len(order)

    def gen(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=64)

    def gen_long(p):
        return generate_no_cache(model, tokenizer, p, max_new_tokens=128)

    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    print(f"  {name:40s}: math={math_r['score']:.4f} eq={eq_r['score']:.1f} combined={combined:.2f}")

    if blocks:
        inner.layers = nn.ModuleList(original_layers)
        model.config.num_hidden_layers = N

    return {
        'name': name,
        'blocks': [list(b) for b in blocks],
        'math': math_r['score'],
        'eq': eq_r['score'],
        'combined': combined
    }


def main():
    print("=== TRIPLE-BLOCK STACKING ON 72B ===")
    model, tokenizer = load_original_model('models/full/calme-2.1-qwen2-72b')
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f"Loaded: {N} layers")

    results = []

    # --- Best pair: (0,7)+(45,52) = 79.91 combined ---
    print("\n--- Adding third block to (0,7)+(45,52) [combined=79.91] ---")

    # Candidate third blocks from different regions
    # Avoid overlap with (0,7) and (45,52)
    third_block_candidates = [
        (15, 20),   # mid-early
        (20, 25),   # mid
        (25, 30),   # mid
        (30, 35),   # mid-deep
        (35, 40),   # pre-Ng region
        (55, 60),   # post-Ng region
        (60, 65),   # deep
        (10, 15),   # early-mid
        (50, 60),   # our best single (overlaps with 45,52 at 50-52, skip if issue)
        (7, 12),    # right after first block
    ]

    base_pair = [(0, 7), (45, 52)]

    for (ci, cj) in third_block_candidates:
        # Check no overlap with existing blocks
        overlaps = False
        for (bi, bj) in base_pair:
            if ci < bj and cj > bi:
                overlaps = True
                break
        if overlaps:
            print(f"  Skipping ({ci},{cj}) — overlaps with base pair")
            continue

        triple = base_pair + [(ci, cj)]
        name = f"(0,7)+(45,52)+({ci},{cj})"
        r = evaluate_config(model, tokenizer, inner, original_layers, N, triple, name)
        results.append(r)

    # --- Second best pair: (15,20)+(50,60) = 76.90 combined ---
    print("\n--- Adding third block to (15,20)+(50,60) [combined=76.90] ---")

    base_pair_2 = [(15, 20), (50, 60)]
    third_candidates_2 = [
        (0, 7),     # early (same as winning pair's first block)
        (7, 12),    # early-mid
        (25, 30),   # mid
        (30, 35),   # mid-deep
        (35, 40),   # pre-deep
        (40, 45),   # pre-Ng
        (45, 50),   # Ng region (no overlap with 50,60)
        (60, 65),   # deep
    ]

    for (ci, cj) in third_candidates_2:
        overlaps = False
        for (bi, bj) in base_pair_2:
            if ci < bj and cj > bi:
                overlaps = True
                break
        if overlaps:
            print(f"  Skipping ({ci},{cj}) — overlaps with base pair")
            continue

        triple = base_pair_2 + [(ci, cj)]
        name = f"(15,20)+(50,60)+({ci},{cj})"
        r = evaluate_config(model, tokenizer, inner, original_layers, N, triple, name)
        results.append(r)

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"Reference: (0,7)+(45,52) = 79.91 combined")
    print(f"Reference: (15,20)+(50,60) = 76.90 combined")
    print()

    for r in sorted(results, key=lambda x: -x['combined']):
        delta_best = r['combined'] - 79.91
        print(f"  {r['name']:45s}: combined={r['combined']:.2f} (vs best pair: {delta_best:+.2f})")

    # Save
    with open('results/72b_triple_stacking.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/72b_triple_stacking.json")


if __name__ == '__main__':
    main()
