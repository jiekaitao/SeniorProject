"""
DeepPass Benchmark Runner

Benchmarks a model before and after layer duplication using:
1. Math Probe (Ng's hard math guesstimate)
2. Optionally lm-evaluation-harness for leaderboard benchmarks

Usage:
    python benchmark.py --model /path/to/model
    python benchmark.py --model /path/to/model --i 45 --j 52
    python benchmark.py --model /path/to/model --baseline-only
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import (
    load_original_model, apply_layer_duplication, restore_layers, generate_no_cache
)
from math_probe import run_math_probe


RESULTS_DIR = Path(__file__).parent.parent / "results"


def make_generate_fn(model, tokenizer, use_cache=True):
    if use_cache:
        def fn(prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
        return fn
    else:
        def fn(prompt):
            return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
        return fn


def benchmark_model(model_path: str, i: int = None, j: int = None,
                    tag: str = ""):
    """Run math probe benchmark, optionally with layer duplication."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name
    config_str = f"dup_{i}_{j}" if (i is not None and j is not None) else "baseline"
    run_name = f"{model_name}_{config_str}_{timestamp}"
    if tag:
        run_name = f"{tag}_{run_name}"

    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_path": model_path, "i": i, "j": j,
        "tag": tag, "timestamp": timestamp, "config": config_str,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nLoading model ({config_str})...")
    model, tokenizer = load_original_model(model_path)

    if i is not None and j is not None:
        apply_layer_duplication(model, i, j)
        gen_fn = make_generate_fn(model, tokenizer, use_cache=False)
    else:
        gen_fn = make_generate_fn(model, tokenizer, use_cache=True)

    print("\n" + "=" * 60)
    print(f"MATH PROBE — {config_str}")
    print("=" * 60)
    results = run_math_probe(gen_fn, verbose=True)

    with open(run_dir / "math_probe.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {run_dir}")

    del model
    torch.cuda.empty_cache()
    return results, run_dir


def main():
    parser = argparse.ArgumentParser(description="DeepPass Benchmark Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--i", type=int, default=None)
    parser.add_argument("--j", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    if args.baseline_only or (args.i is None and args.j is None):
        print("Running BASELINE...")
        baseline, baseline_dir = benchmark_model(args.model, tag=args.tag or "baseline")

        if not args.baseline_only:
            # Also run Ng's optimal config
            print("\nRunning with RYS duplication (i=45, j=52)...")
            dup, dup_dir = benchmark_model(args.model, i=45, j=52, tag=args.tag or "rys")

            delta = dup["score"] - baseline["score"]
            pct = (delta / baseline["score"] * 100) if baseline["score"] > 0 else 0
            print(f"\n{'=' * 60}")
            print(f"COMPARISON")
            print(f"  Baseline:   {baseline['score']:.4f}")
            print(f"  Duplicated: {dup['score']:.4f}")
            print(f"  Delta:      {delta:+.4f} ({pct:+.2f}%)")
            print(f"{'=' * 60}")
    else:
        benchmark_model(args.model, i=args.i, j=args.j, tag=args.tag)


if __name__ == "__main__":
    main()
