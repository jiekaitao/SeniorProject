"""
DeepPass Even/Odd Parity Test

Tests the hypothesis that when duplicating a block, adding an ODD number of
extra layers causes interference (like "reversing" weights), while adding an
EVEN number does not.

The idea: each pass through a block applies a transformation T.
- 1 pass (baseline): T^1
- 2 passes (1 extra): T^2 -- odd extra layers
- 3 passes (2 extra): T^3 -- even extra layers
- etc.

If T has eigenvalues with negative real parts or rotational structure,
odd vs even powers could behave very differently.

Experiments:
1. Single-layer block (10,11): 1-6 passes
2. 3-layer block (18,21): 1-4 passes
3. Multi-block (10,11)+(14,27): asymmetric pass counts
"""

import json
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import generate_no_cache
from math_probe import run_math_probe
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
N_LAYERS = 28
RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/even_odd_test")


def apply_n_pass(model, inner, attr_name, original_layers, i, j, n_passes):
    """
    Apply block [i, j) a total of n_passes times.

    Layer order: [0..j-1] + [i..j-1] * (n_passes - 1) + [j..N-1]

    For n_passes=1 this is just the original model.
    """
    N = len(original_layers)
    new_layers = list(original_layers[:j])
    for _ in range(n_passes - 1):
        new_layers.extend(original_layers[i:j])
    new_layers.extend(original_layers[j:])

    setattr(inner, attr_name, nn.ModuleList(new_layers))
    model.config.num_hidden_layers = len(new_layers)
    return len(new_layers)


def apply_multi_block_n_pass(model, inner, attr_name, original_layers,
                              block1, n1, block2, n2):
    """
    Apply two non-overlapping blocks with independent pass counts.

    block1 = (a1, b1) with n1 passes, block2 = (a2, b2) with n2 passes.
    Assumes b1 <= a2 (block1 comes first).

    Layer order:
      [0..b1-1] + [a1..b1-1]*(n1-1) + [b1..b2-1] + [a2..b2-1]*(n2-1) + [b2..N-1]
    """
    a1, b1 = block1
    a2, b2 = block2
    N = len(original_layers)

    # Ensure block1 comes first
    if a2 < a1:
        a1, b1, a2, b2 = a2, b2, a1, b1
        n1, n2 = n2, n1

    new_layers = list(original_layers[:b1])
    for _ in range(n1 - 1):
        new_layers.extend(original_layers[a1:b1])
    new_layers.extend(original_layers[b1:b2])
    for _ in range(n2 - 1):
        new_layers.extend(original_layers[a2:b2])
    new_layers.extend(original_layers[b2:])

    setattr(inner, attr_name, nn.ModuleList(new_layers))
    model.config.num_hidden_layers = len(new_layers)
    return len(new_layers)


def restore(model, inner, attr_name, original_layers, original_num):
    """Restore original layer configuration."""
    setattr(inner, attr_name, nn.ModuleList(original_layers))
    model.config.num_hidden_layers = original_num


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("DeepPass Even/Odd Parity Test")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {RESULTS_DIR}")
    print(f"Timestamp: {timestamp}")
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
    original_num = model.config.num_hidden_layers
    N = len(original_layers)
    assert N == N_LAYERS, f"Expected {N_LAYERS} layers, got {N}"
    print(f"Model loaded: {N} layers\n")

    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)

    all_results = {}

    # ==================================================================
    # EXPERIMENT 1: Single-layer block (10, 11) — 1 to 6 passes
    # ==================================================================
    print("=" * 70)
    print("EXPERIMENT 1: Block (10,11) — single layer, 1-6 passes")
    print("=" * 70)

    exp1_results = {}
    block_i, block_j = 10, 11
    block_size = block_j - block_i  # 1

    for n_passes in range(1, 7):
        restore(model, inner, attr_name, original_layers, original_num)
        total = apply_n_pass(model, inner, attr_name, original_layers,
                             block_i, block_j, n_passes)
        extra = block_size * (n_passes - 1)
        extra_parity = "EVEN" if extra % 2 == 0 else "ODD"
        total_parity = "EVEN" if total % 2 == 0 else "ODD"

        label = f"{n_passes} passes"
        if n_passes == 1:
            label += " (baseline)"

        print(f"\n--- {label}: {total} total layers, "
              f"+{extra} extra ({extra_parity} extra, {total_parity} total) ---")

        t0 = time.time()
        result = run_math_probe(gen_fn, verbose=True)
        elapsed = time.time() - t0

        exp1_results[n_passes] = {
            "n_passes": n_passes,
            "total_layers": total,
            "extra_layers": extra,
            "extra_parity": extra_parity,
            "total_parity": total_parity,
            "score": result["score"],
            "scores": result["scores"],
            "elapsed": elapsed,
        }
        print(f"  Score: {result['score']:.4f} [{elapsed:.1f}s]")

    restore(model, inner, attr_name, original_layers, original_num)

    # Summary for exp1
    print(f"\n{'='*70}")
    print("EXP1 SUMMARY: Block (10,11)")
    print(f"{'Passes':>8} {'Total':>6} {'Extra':>6} {'Parity':>8} {'Score':>8} {'Delta':>10}")
    baseline_score_1 = exp1_results[1]["score"]
    for n in sorted(exp1_results.keys()):
        r = exp1_results[n]
        delta = r["score"] - baseline_score_1
        print(f"{n:8d} {r['total_layers']:6d} {r['extra_layers']:6d} "
              f"{r['extra_parity']:>8s} {r['score']:8.4f} {delta:+10.4f}")
    print()

    all_results["exp1_block_10_11"] = {
        "block": [block_i, block_j],
        "block_size": block_size,
        "baseline_score": baseline_score_1,
        "results": exp1_results,
    }

    # ==================================================================
    # EXPERIMENT 2: 3-layer block (18, 21) — 1 to 4 passes
    # ==================================================================
    print("=" * 70)
    print("EXPERIMENT 2: Block (18,21) — 3 layers, 1-4 passes")
    print("=" * 70)

    exp2_results = {}
    block_i, block_j = 18, 21
    block_size = block_j - block_i  # 3

    for n_passes in range(1, 5):
        restore(model, inner, attr_name, original_layers, original_num)
        total = apply_n_pass(model, inner, attr_name, original_layers,
                             block_i, block_j, n_passes)
        extra = block_size * (n_passes - 1)
        extra_parity = "EVEN" if extra % 2 == 0 else "ODD"
        total_parity = "EVEN" if total % 2 == 0 else "ODD"

        label = f"{n_passes} passes"
        if n_passes == 1:
            label += " (baseline)"

        print(f"\n--- {label}: {total} total layers, "
              f"+{extra} extra ({extra_parity} extra, {total_parity} total) ---")

        t0 = time.time()
        result = run_math_probe(gen_fn, verbose=True)
        elapsed = time.time() - t0

        exp2_results[n_passes] = {
            "n_passes": n_passes,
            "total_layers": total,
            "extra_layers": extra,
            "extra_parity": extra_parity,
            "total_parity": total_parity,
            "score": result["score"],
            "scores": result["scores"],
            "elapsed": elapsed,
        }
        print(f"  Score: {result['score']:.4f} [{elapsed:.1f}s]")

    restore(model, inner, attr_name, original_layers, original_num)

    # Summary for exp2
    print(f"\n{'='*70}")
    print("EXP2 SUMMARY: Block (18,21)")
    print(f"{'Passes':>8} {'Total':>6} {'Extra':>6} {'Parity':>8} {'Score':>8} {'Delta':>10}")
    baseline_score_2 = exp2_results[1]["score"]
    for n in sorted(exp2_results.keys()):
        r = exp2_results[n]
        delta = r["score"] - baseline_score_2
        print(f"{n:8d} {r['total_layers']:6d} {r['extra_layers']:6d} "
              f"{r['extra_parity']:>8s} {r['score']:8.4f} {delta:+10.4f}")
    print()

    all_results["exp2_block_18_21"] = {
        "block": [block_i, block_j],
        "block_size": block_size,
        "baseline_score": baseline_score_2,
        "results": exp2_results,
    }

    # ==================================================================
    # EXPERIMENT 3: Multi-block (10,11) + (14,27) with asymmetric passes
    # ==================================================================
    print("=" * 70)
    print("EXPERIMENT 3: Multi-block (10,11)+(14,27) — asymmetric passes")
    print("=" * 70)

    exp3_results = {}
    b1 = (10, 11)  # 1-layer block
    b2 = (14, 27)  # 13-layer block
    b1_size = b1[1] - b1[0]  # 1
    b2_size = b2[1] - b2[0]  # 13

    configs = [
        ("standard_2x2", 2, 2, "Each block 2 passes"),
        ("b1_3x_b2_2x", 3, 2, "Block1 at 3 passes (even extra for b1) + Block2 at 2 passes"),
        ("b1_2x_b2_3x", 2, 3, "Block1 at 2 passes + Block2 at 3 passes (even extra for b2)"),
    ]

    for label, n1, n2, description in configs:
        restore(model, inner, attr_name, original_layers, original_num)
        total = apply_multi_block_n_pass(
            model, inner, attr_name, original_layers, b1, n1, b2, n2
        )
        extra_b1 = b1_size * (n1 - 1)
        extra_b2 = b2_size * (n2 - 1)
        total_extra = extra_b1 + extra_b2

        print(f"\n--- {label}: {description} ---")
        print(f"    Block1 (10,11): {n1} passes, +{extra_b1} extra layers")
        print(f"    Block2 (14,27): {n2} passes, +{extra_b2} extra layers")
        print(f"    Total: {total} layers (+{total_extra} extra)")

        t0 = time.time()
        result = run_math_probe(gen_fn, verbose=True)
        elapsed = time.time() - t0

        exp3_results[label] = {
            "block1": list(b1),
            "block1_passes": n1,
            "block2": list(b2),
            "block2_passes": n2,
            "total_layers": total,
            "extra_b1": extra_b1,
            "extra_b2": extra_b2,
            "total_extra": total_extra,
            "description": description,
            "score": result["score"],
            "scores": result["scores"],
            "elapsed": elapsed,
        }
        print(f"  Score: {result['score']:.4f} [{elapsed:.1f}s]")

    restore(model, inner, attr_name, original_layers, original_num)

    # Summary for exp3
    print(f"\n{'='*70}")
    print("EXP3 SUMMARY: Multi-block (10,11)+(14,27)")
    print(f"{'Config':<16} {'B1 passes':>10} {'B2 passes':>10} {'Total':>6} "
          f"{'Score':>8} {'Delta':>10}")
    # Use baseline from exp1 (same model, no duplication)
    for label in ["standard_2x2", "b1_3x_b2_2x", "b1_2x_b2_3x"]:
        r = exp3_results[label]
        delta = r["score"] - baseline_score_1
        print(f"{label:<16} {r['block1_passes']:10d} {r['block2_passes']:10d} "
              f"{r['total_layers']:6d} {r['score']:8.4f} {delta:+10.4f}")
    print()

    all_results["exp3_multi_block"] = {
        "block1": list(b1),
        "block2": list(b2),
        "baseline_score": baseline_score_1,
        "results": exp3_results,
    }

    # ==================================================================
    # FINAL ANALYSIS: Does parity matter?
    # ==================================================================
    print("=" * 70)
    print("PARITY ANALYSIS")
    print("=" * 70)

    # Exp 1: single layer block — extra layers are 0, 1, 2, 3, 4, 5
    print("\nBlock (10,11) — 1-layer block:")
    print("  Extra layers: 0=baseline, 1=odd, 2=even, 3=odd, 4=even, 5=odd")
    even_scores_1 = []
    odd_scores_1 = []
    for n in range(2, 7):
        r = exp1_results[n]
        delta = r["score"] - baseline_score_1
        if r["extra_layers"] % 2 == 0:
            even_scores_1.append(delta)
        else:
            odd_scores_1.append(delta)

    avg_even_1 = sum(even_scores_1) / len(even_scores_1) if even_scores_1 else 0
    avg_odd_1 = sum(odd_scores_1) / len(odd_scores_1) if odd_scores_1 else 0
    print(f"  Avg delta (EVEN extra): {avg_even_1:+.4f} (n={len(even_scores_1)})")
    print(f"  Avg delta (ODD extra):  {avg_odd_1:+.4f} (n={len(odd_scores_1)})")

    # Exp 2: 3-layer block — extra layers are 0, 3, 6, 9
    print("\nBlock (18,21) — 3-layer block:")
    print("  Extra layers: 0=baseline, 3=odd, 6=even, 9=odd")
    even_scores_2 = []
    odd_scores_2 = []
    for n in range(2, 5):
        r = exp2_results[n]
        delta = r["score"] - baseline_score_2
        if r["extra_layers"] % 2 == 0:
            even_scores_2.append(delta)
        else:
            odd_scores_2.append(delta)

    avg_even_2 = sum(even_scores_2) / len(even_scores_2) if even_scores_2 else 0
    avg_odd_2 = sum(odd_scores_2) / len(odd_scores_2) if odd_scores_2 else 0
    print(f"  Avg delta (EVEN extra): {avg_even_2:+.4f} (n={len(even_scores_2)})")
    print(f"  Avg delta (ODD extra):  {avg_odd_2:+.4f} (n={len(odd_scores_2)})")

    # Combined verdict
    all_even = even_scores_1 + even_scores_2
    all_odd = odd_scores_1 + odd_scores_2
    avg_all_even = sum(all_even) / len(all_even) if all_even else 0
    avg_all_odd = sum(all_odd) / len(all_odd) if all_odd else 0

    print(f"\nCOMBINED:")
    print(f"  Avg delta (EVEN extra): {avg_all_even:+.4f} (n={len(all_even)})")
    print(f"  Avg delta (ODD extra):  {avg_all_odd:+.4f} (n={len(all_odd)})")
    diff = avg_all_even - avg_all_odd
    print(f"  Even - Odd difference:  {diff:+.4f}")

    if abs(diff) < 0.02:
        verdict = "NO significant parity effect detected"
    elif diff > 0:
        verdict = "EVEN extra layers perform BETTER (supports hypothesis)"
    else:
        verdict = "ODD extra layers perform BETTER (contradicts hypothesis)"
    print(f"\n  VERDICT: {verdict}")

    all_results["parity_analysis"] = {
        "exp1_avg_even_delta": avg_even_1,
        "exp1_avg_odd_delta": avg_odd_1,
        "exp2_avg_even_delta": avg_even_2,
        "exp2_avg_odd_delta": avg_odd_2,
        "combined_avg_even_delta": avg_all_even,
        "combined_avg_odd_delta": avg_all_odd,
        "even_minus_odd": diff,
        "verdict": verdict,
    }

    # Save
    results_path = RESULTS_DIR / "even_odd_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Also save a human-readable summary
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Even/Odd Parity Test Summary\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Model: {MODEL_PATH}\n\n")

        f.write("EXPERIMENT 1: Block (10,11) — single layer\n")
        f.write(f"{'Passes':>8} {'Total':>6} {'Extra':>6} {'Parity':>8} "
                f"{'Score':>8} {'Delta':>10}\n")
        for n in sorted(exp1_results.keys()):
            r = exp1_results[n]
            delta = r["score"] - baseline_score_1
            f.write(f"{n:8d} {r['total_layers']:6d} {r['extra_layers']:6d} "
                    f"{r['extra_parity']:>8s} {r['score']:8.4f} {delta:+10.4f}\n")

        f.write(f"\nEXPERIMENT 2: Block (18,21) — 3 layers\n")
        f.write(f"{'Passes':>8} {'Total':>6} {'Extra':>6} {'Parity':>8} "
                f"{'Score':>8} {'Delta':>10}\n")
        for n in sorted(exp2_results.keys()):
            r = exp2_results[n]
            delta = r["score"] - baseline_score_2
            f.write(f"{n:8d} {r['total_layers']:6d} {r['extra_layers']:6d} "
                    f"{r['extra_parity']:>8s} {r['score']:8.4f} {delta:+10.4f}\n")

        f.write(f"\nEXPERIMENT 3: Multi-block (10,11)+(14,27)\n")
        f.write(f"{'Config':<16} {'B1x':>5} {'B2x':>5} {'Total':>6} "
                f"{'Score':>8} {'Delta':>10}\n")
        for label in ["standard_2x2", "b1_3x_b2_2x", "b1_2x_b2_3x"]:
            r = exp3_results[label]
            delta = r["score"] - baseline_score_1
            f.write(f"{label:<16} {r['block1_passes']:5d} {r['block2_passes']:5d} "
                    f"{r['total_layers']:6d} {r['score']:8.4f} {delta:+10.4f}\n")

        f.write(f"\nPARITY ANALYSIS:\n")
        f.write(f"  Exp1 avg even delta: {avg_even_1:+.4f}\n")
        f.write(f"  Exp1 avg odd delta:  {avg_odd_1:+.4f}\n")
        f.write(f"  Exp2 avg even delta: {avg_even_2:+.4f}\n")
        f.write(f"  Exp2 avg odd delta:  {avg_odd_2:+.4f}\n")
        f.write(f"  Combined even:       {avg_all_even:+.4f}\n")
        f.write(f"  Combined odd:        {avg_all_odd:+.4f}\n")
        f.write(f"  Even - Odd:          {diff:+.4f}\n")
        f.write(f"\n  VERDICT: {verdict}\n")

    print(f"Summary saved to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
