"""
DeepPass Brain Scanner

Full (i, j) sweep — generates heatmaps showing which layer-duplication
configurations improve model performance. Each pixel = one config evaluated.

Usage:
    python brain_scanner.py --model /path/to/model --step 2 --max-dup 20
"""

import json
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import generate_no_cache
from math_probe import run_math_probe
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_sweep(
    model_path: str,
    output_dir: str,
    step: int = 1,
    i_range: tuple = None,
    j_range: tuple = None,
    max_dup_layers: int = None,
    device_map: str = "auto",
    verbose: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    inner = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner, 'layers') else 'h'
    original_layers = list(getattr(inner, attr_name))
    N = len(original_layers)
    original_num_layers = model.config.num_hidden_layers
    print(f"Model has {N} layers")

    def make_gen_fn():
        def fn(prompt):
            return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
        return fn

    # Run baseline
    print("\nRunning baseline evaluation...")
    gen_fn = make_gen_fn()
    baseline = run_math_probe(gen_fn, verbose=verbose)
    baseline_score = baseline["score"]
    print(f"Baseline math score: {baseline_score:.4f}")

    with open(output_dir / "baseline.json", "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    # Determine sweep configs
    i_min, i_max = i_range if i_range else (0, N)
    j_min, j_max = j_range if j_range else (1, N + 1)

    configs = []
    for i in range(i_min, i_max, step):
        for j in range(max(i + 1, j_min), j_max, step):
            dup_count = j - i
            if max_dup_layers and dup_count > max_dup_layers:
                continue
            configs.append((i, j))

    total = len(configs)
    print(f"\nSweeping {total} configurations (step={step})...")

    results = {}

    for idx, (i, j) in enumerate(configs):
        dup_count = j - i
        t0 = time.time()

        try:
            # Build duplicated layer sequence using references
            layer_order = list(range(j)) + list(range(i, j)) + list(range(j, N))
            new_layers = [original_layers[k] for k in layer_order]
            setattr(inner, attr_name, nn.ModuleList(new_layers))
            model.config.num_hidden_layers = len(new_layers)

            gen_fn = make_gen_fn()
            result = run_math_probe(gen_fn, verbose=False)
            score = result["score"]
            delta = score - baseline_score

            results[(i, j)] = {"score": score, "delta": delta, "dup_count": dup_count}

            elapsed = time.time() - t0
            if verbose:
                indicator = "+" if delta > 0 else "-" if delta < 0 else "="
                print(f"  [{idx+1:4d}/{total}] ({i:3d},{j:3d}) dup={dup_count:2d} "
                      f"score={score:.4f} delta={delta:+.4f} {indicator} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{idx+1:4d}/{total}] ({i:3d},{j:3d}) ERROR: {e}")
            results[(i, j)] = {"score": 0, "delta": -baseline_score, "dup_count": dup_count, "error": str(e)}

        finally:
            # Restore original layers
            setattr(inner, attr_name, nn.ModuleList(original_layers))
            model.config.num_hidden_layers = original_num_layers

        if (idx + 1) % 10 == 0:
            save_results(results, baseline_score, N, output_dir)

    save_results(results, baseline_score, N, output_dir)
    generate_heatmap(results, baseline_score, N, output_dir)
    print(f"\nSweep complete! Results saved to {output_dir}")
    return results


def save_results(results, baseline_score, N, output_dir):
    serializable = {f"{i},{j}": v for (i, j), v in results.items()}
    data = {"baseline_score": baseline_score, "num_layers": N, "results": serializable}
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(data, f, indent=2)


def generate_heatmap(results, baseline_score, N, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping heatmap")
        return

    heatmap = np.full((N, N + 1), np.nan)
    for (i, j), v in results.items():
        if "error" not in v:
            heatmap[i, j] = v["delta"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    valid = heatmap[~np.isnan(heatmap)]
    vmax = np.nanmax(np.abs(valid)) if len(valid) > 0 else 0.1

    im = ax.imshow(heatmap, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   origin='upper', aspect='auto')
    ax.set_xlabel('End layer j', fontsize=12)
    ax.set_ylabel('Start layer i', fontsize=12)
    ax.set_title(f'DeepPass Brain Scan — Math Probe Delta\n'
                 f'Baseline: {baseline_score:.4f} | N={N} layers', fontsize=14)
    plt.colorbar(im, ax=ax, label='Score delta from baseline')

    if results:
        best = max(results.items(), key=lambda x: x[1].get("delta", -999))
        bi, bj = best[0]
        bd = best[1]["delta"]
        ax.plot(bj, bi, 'o', color='lime', markersize=12,
                markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'Best: ({bi},{bj})\n+{bd:.4f}',
                    xy=(bj, bi), xytext=(bj + 2, bi - 3),
                    fontsize=9, color='green',
                    arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_math.png", dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_dir / 'heatmap_math.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepPass Brain Scanner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--i-min", type=int, default=0)
    parser.add_argument("--i-max", type=int, default=None)
    parser.add_argument("--j-min", type=int, default=1)
    parser.add_argument("--j-max", type=int, default=None)
    parser.add_argument("--max-dup", type=int, default=None)
    args = parser.parse_args()

    if args.output is None:
        model_name = Path(args.model).name
        args.output = str(Path(__file__).parent.parent / "results" /
                         f"sweep_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    i_range = (args.i_min, args.i_max) if args.i_max else None
    j_range = (args.j_min, args.j_max) if args.j_max else None

    run_sweep(args.model, args.output, step=args.step,
              i_range=i_range, j_range=j_range, max_dup_layers=args.max_dup)
