"""
DeepPass Multi-Pass Test

Tests what happens when you duplicate the same block MORE than once.
Ng only tested 2 passes. Does 3 or 4 passes help? TRM theory predicts:
- If displacement rho < 1: more passes = closer to fixed point = better
- If displacement rho ≈ 1: diminishing returns
- If displacement rho > 1: each pass makes things worse

This is exactly the question TRM's Raju-Netrapalli framework answers.
"""

import sys, os, time, json, torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def apply_n_pass_duplication(model, i, j, n_passes):
    """Apply a block n times instead of just 2."""
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr_name))
    N = len(original_layers)

    # Build layer sequence: [0..j-1] + [i..j-1] * (n_passes - 1) + [j..N-1]
    new_layers = original_layers[:j]
    for _ in range(n_passes - 1):
        new_layers.extend(original_layers[i:j])
    new_layers.extend(original_layers[j:])

    setattr(inner, attr_name, nn.ModuleList(new_layers))
    model.config.num_hidden_layers = len(new_layers)
    return original_layers, N


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / f"multi_pass_{Path(args.model).name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_original_model(args.model)
    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr_name))
    N = len(original_layers)

    # For 28-layer model, test proportional to Ng's optimal on 80-layer
    i = int(N * 0.5625)  # 15 for 28-layer
    j = int(N * 0.65)    # 18 for 28-layer
    if j <= i:
        j = i + 2

    def make_gen_fn():
        return lambda prompt: generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)

    results = {}

    # Test 1 to 5 passes
    for n_passes in range(1, 6):
        # Restore original
        setattr(inner, attr_name, nn.ModuleList(original_layers))
        model.config.num_hidden_layers = N

        if n_passes > 1:
            apply_n_pass_duplication(model, i, j, n_passes)

        total_layers = model.config.num_hidden_layers
        dup_count = (j - i) * (n_passes - 1)

        print(f"\n{'='*60}")
        print(f"TESTING: {n_passes} passes through ({i},{j})")
        print(f"Total layers: {total_layers} ({N} original + {dup_count} duplicated)")
        print(f"{'='*60}")

        gen_fn = make_gen_fn()
        result = run_math_probe(gen_fn, verbose=True)

        results[n_passes] = {
            "score": result["score"],
            "n_passes": n_passes,
            "total_layers": total_layers,
            "dup_count": dup_count,
            "config": f"({i},{j})",
        }

    # Summary
    print(f"\n{'='*60}")
    print("MULTI-PASS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Passes':>8} {'Layers':>8} {'Score':>8} {'Delta':>10}")
    baseline_score = results[1]["score"]
    for n in sorted(results.keys()):
        r = results[n]
        delta = r["score"] - baseline_score
        print(f"{n:8d} {r['total_layers']:8d} {r['score']:8.4f} {delta:+10.4f}")

    with open(run_dir / "multi_pass_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Restore original
    setattr(inner, attr_name, nn.ModuleList(original_layers))
    model.config.num_hidden_layers = N


if __name__ == "__main__":
    main()
