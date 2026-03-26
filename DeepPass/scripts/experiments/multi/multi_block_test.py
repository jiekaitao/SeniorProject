"""
DeepPass Multi-Block Duplication Test

Tests simultaneous duplication of TWO non-overlapping circuit blocks.
This goes beyond Ng's single-block RYS approach — can stacking two
duplicated regions beat the best single-block config?

For single-block (a,b): layers become [0..b-1, a..b-1, b..N-1]
For dual-block (a1,b1) + (a2,b2) where b1 <= a2:
  [0..b1-1, a1..b1-1, b1..b2-1, a2..b2-1, b2..N-1]
  i.e., first block duplicated in place, then second block duplicated in place.

Top single-block configs on Qwen2-7B-Instruct (28 layers):
  (10,11) +25.7%
  (18,21) +23.5%
  (16,21) +21.8%
  (14,27) +19.0%
"""

import json
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import generate_no_cache
from math_probe import run_math_probe
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
N_LAYERS = 28

# Best single-block configs: (i, j) and their delta from baseline
SINGLE_BLOCK_CONFIGS = [
    ((10, 11), 0.257),   # +25.7%
    ((18, 21), 0.235),   # +23.5%
    ((16, 21), 0.218),   # +21.8%
    ((14, 27), 0.190),   # +19.0%
]

BEST_SINGLE_DELTA = max(d for _, d in SINGLE_BLOCK_CONFIGS)  # 0.257


def blocks_overlap(block1, block2):
    """
    Check if two blocks overlap. Blocks are (i, j) meaning layers [i, j).
    They overlap if one block's range intersects the other's.
    """
    a1, b1 = block1
    a2, b2 = block2
    # Non-overlapping if one ends before the other starts
    return not (b1 <= a2 or b2 <= a1)


def build_dual_block_layer_order(block1, block2, N):
    """
    Build the layer execution order for two simultaneous duplications.

    Given block1 = (a1, b1) and block2 = (a2, b2), with b1 <= a2:
      [0..b1-1, a1..b1-1, b1..b2-1, a2..b2-1, b2..N-1]

    If block2 comes first (b2 <= a1), we swap them.
    """
    a1, b1 = block1
    a2, b2 = block2

    # Ensure block1 comes first (lower indices)
    if a2 < a1:
        a1, b1, a2, b2 = a2, b2, a1, b1

    # Build the order:
    # Original layers up to end of first block
    order = list(range(b1))
    # Duplicate first block
    order += list(range(a1, b1))
    # Original layers between the two blocks
    order += list(range(b1, b2))
    # Duplicate second block
    order += list(range(a2, b2))
    # Original layers after second block
    order += list(range(b2, N))

    return order


def main():
    output_dir = Path("/blue/cis4914/jietao/DeepPass/results") / \
        f"multi_block_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DeepPass Multi-Block Duplication Test")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Layers: {N_LAYERS}")
    print(f"Best single-block delta: {BEST_SINGLE_DELTA:+.4f}")
    print(f"Output: {output_dir}")
    print()

    # ---- Load model ----
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr_name))
    original_num_layers = model.config.num_hidden_layers
    N = len(original_layers)
    assert N == N_LAYERS, f"Expected {N_LAYERS} layers, got {N}"
    print(f"Model loaded: {N} layers")

    def make_gen_fn():
        def fn(prompt):
            return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
        return fn

    # ---- Baseline ----
    print("\n--- Baseline (no duplication) ---")
    gen_fn = make_gen_fn()
    baseline = run_math_probe(gen_fn, verbose=True)
    baseline_score = baseline["score"]
    print(f"\nBaseline score: {baseline_score:.4f}")

    # ---- Generate all non-overlapping pairs ----
    single_blocks = [cfg for cfg, _ in SINGLE_BLOCK_CONFIGS]
    pairs = []
    for b1, b2 in combinations(single_blocks, 2):
        if not blocks_overlap(b1, b2):
            pairs.append((b1, b2))

    print(f"\n--- Testing {len(pairs)} non-overlapping dual-block configs ---")
    for b1, b2 in pairs:
        print(f"  {b1} + {b2}")
    print()

    # ---- Also re-test single blocks for comparison ----
    print("--- Re-testing single-block configs for comparison ---")
    single_results = {}
    for (i, j), reported_delta in SINGLE_BLOCK_CONFIGS:
        t0 = time.time()
        try:
            layer_order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            new_layers = [original_layers[k] for k in layer_order]
            setattr(inner, attr_name, nn.ModuleList(new_layers))
            model.config.num_hidden_layers = len(new_layers)

            gen_fn = make_gen_fn()
            result = run_math_probe(gen_fn, verbose=False)
            score = result["score"]
            delta = score - baseline_score

            single_results[(i, j)] = {
                "score": score,
                "delta": delta,
                "reported_delta": reported_delta,
                "total_layers": len(new_layers),
            }
            elapsed = time.time() - t0
            print(f"  ({i:2d},{j:2d}) score={score:.4f} delta={delta:+.4f} "
                  f"(reported: {reported_delta:+.4f}) [{elapsed:.1f}s]")
        except Exception as e:
            print(f"  ({i:2d},{j:2d}) ERROR: {e}")
            single_results[(i, j)] = {"error": str(e)}
        finally:
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            model.config.num_hidden_layers = original_num_layers

    # ---- Dual-block tests ----
    print("\n--- Dual-block duplication results ---")
    dual_results = {}
    for idx, (block1, block2) in enumerate(pairs):
        a1, b1 = block1
        a2, b2 = block2
        t0 = time.time()

        try:
            layer_order = build_dual_block_layer_order(block1, block2, N)
            new_layers = [original_layers[k] for k in layer_order]
            setattr(inner, attr_name, nn.ModuleList(new_layers))
            model.config.num_hidden_layers = len(new_layers)

            gen_fn = make_gen_fn()
            result = run_math_probe(gen_fn, verbose=True)
            score = result["score"]
            delta = score - baseline_score

            key = f"({a1},{b1})+({a2},{b2})"
            dual_results[key] = {
                "block1": [a1, b1],
                "block2": [a2, b2],
                "layer_order": layer_order,
                "total_layers": len(new_layers),
                "score": score,
                "delta": delta,
                "beats_best_single": delta > BEST_SINGLE_DELTA,
            }
            elapsed = time.time() - t0

            indicator = "***BEATS BEST***" if delta > BEST_SINGLE_DELTA else ""
            print(f"\n  [{idx+1}/{len(pairs)}] ({a1},{b1})+({a2},{b2}) "
                  f"total_layers={len(new_layers)} "
                  f"score={score:.4f} delta={delta:+.4f} {indicator} [{elapsed:.1f}s]\n")

        except Exception as e:
            key = f"({a1},{b1})+({a2},{b2})"
            print(f"\n  [{idx+1}/{len(pairs)}] ({a1},{b1})+({a2},{b2}) ERROR: {e}\n")
            dual_results[key] = {"error": str(e)}

        finally:
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            model.config.num_hidden_layers = original_num_layers

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline score: {baseline_score:.4f}")

    print(f"\nBest single-block configs (re-measured):")
    best_single_measured = -999
    for (i, j), v in sorted(single_results.items(), key=lambda x: x[1].get("delta", -999), reverse=True):
        if "error" not in v:
            print(f"  ({i:2d},{j:2d}) delta={v['delta']:+.4f}  ({v['total_layers']} layers)")
            best_single_measured = max(best_single_measured, v["delta"])

    print(f"\nDual-block configs:")
    best_dual_key = None
    best_dual_delta = -999
    for key, v in sorted(dual_results.items(), key=lambda x: x[1].get("delta", -999), reverse=True):
        if "error" not in v:
            flag = " *** BEATS BEST SINGLE ***" if v["beats_best_single"] else ""
            print(f"  {key} delta={v['delta']:+.4f}  ({v['total_layers']} layers){flag}")
            if v["delta"] > best_dual_delta:
                best_dual_delta = v["delta"]
                best_dual_key = key

    print(f"\n{'=' * 70}")
    print(f"Best single-block delta (measured): {best_single_measured:+.4f}")
    print(f"Best dual-block delta:              {best_dual_delta:+.4f}")
    if best_dual_delta > best_single_measured:
        print(f"RESULT: Dual-block BEATS single-block by {best_dual_delta - best_single_measured:+.4f}!")
        print(f"Best dual config: {best_dual_key}")
    else:
        print(f"RESULT: Single-block still wins (by {best_single_measured - best_dual_delta:+.4f})")
    print("=" * 70)

    # ---- Save results ----
    all_results = {
        "baseline_score": baseline_score,
        "best_single_reported_delta": BEST_SINGLE_DELTA,
        "best_single_measured_delta": best_single_measured,
        "best_dual_delta": best_dual_delta,
        "best_dual_config": best_dual_key,
        "dual_beats_single": best_dual_delta > best_single_measured,
        "single_results": {f"{i},{j}": v for (i, j), v in single_results.items()},
        "dual_results": dual_results,
    }
    results_path = output_dir / "multi_block_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
