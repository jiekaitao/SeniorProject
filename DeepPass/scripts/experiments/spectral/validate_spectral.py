"""
DeepPass Spectral Validation

The key test: does spectral analysis predict the brain scanner heatmap?
Compare spectral metrics (cheap: 2-3 forward passes per block) against
actual benchmark results (expensive: full eval per block).

If displacement_rho < 1 correlates with improved benchmark scores,
we can replace Ng's 3,241-config sweep with O(N) spectral analysis.
"""

import json, sys, argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr


def load_results(path):
    with open(path) as f:
        return json.load(f)


def validate(spectral_path, sweep_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral = load_results(spectral_path)
    sweep = load_results(sweep_path)

    # Find overlapping configs
    common_configs = []
    for key, sweep_val in sweep["results"].items():
        if key in spectral["results"]:
            s = spectral["results"][key]
            if "error" not in s and "error" not in sweep_val:
                i, j = map(int, key.split(","))
                common_configs.append({
                    "config": key,
                    "i": i, "j": j,
                    "sweep_delta": sweep_val["delta"],
                    "sweep_score": sweep_val["score"],
                    "pert_rho": s["perturbation_rho"],
                    "disp_rho": s["displacement_rho"],
                    "fp_dist": s.get("fixed_point_dist", s.get("residual", 0)),
                })

    if not common_configs:
        print("No overlapping configs found!")
        return

    print(f"Found {len(common_configs)} overlapping configs")

    # Extract arrays
    deltas = np.array([c["sweep_delta"] for c in common_configs])
    pert_rhos = np.array([c["pert_rho"] for c in common_configs])
    disp_rhos = np.array([c["disp_rho"] for c in common_configs])
    fp_dists = np.array([c["fp_dist"] for c in common_configs])

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: Spectral Metrics vs Benchmark Improvement")
    print("=" * 70)

    metrics = {
        "perturbation_rho": pert_rhos,
        "displacement_rho": disp_rhos,
        "fixed_point_dist": fp_dists,
        "1 - displacement_rho (contraction)": 1 - disp_rhos,
        "1 / (1 + fp_dist)": 1 / (1 + fp_dists),
    }

    correlations = {}
    for name, values in metrics.items():
        if np.std(values) < 1e-10:
            print(f"  {name:40s} — constant, skipping")
            continue
        spear_r, spear_p = spearmanr(values, deltas)
        pears_r, pears_p = pearsonr(values, deltas)
        correlations[name] = {
            "spearman_r": float(spear_r), "spearman_p": float(spear_p),
            "pearson_r": float(pears_r), "pearson_p": float(pears_p),
        }
        sig = "***" if spear_p < 0.001 else "**" if spear_p < 0.01 else "*" if spear_p < 0.05 else ""
        print(f"  {name:40s} Spearman r={spear_r:+.4f} (p={spear_p:.4f}){sig}  "
              f"Pearson r={pears_r:+.4f}")

    # Check if top spectral predictions match top sweep results
    print("\n" + "=" * 70)
    print("TOP-K PREDICTION ACCURACY")
    print("=" * 70)

    # Sort by each metric and check if top-K configs also have positive deltas
    for k in [5, 10, 20]:
        if k > len(common_configs):
            continue

        # Top-K by displacement rho (lower = better = more contractive)
        sorted_by_disp = sorted(common_configs, key=lambda c: c["disp_rho"])
        top_k_disp = sorted_by_disp[:k]
        hit_rate = sum(1 for c in top_k_disp if c["sweep_delta"] > 0) / k

        # Top-K by actual benchmark delta (ground truth)
        sorted_by_delta = sorted(common_configs, key=lambda c: c["sweep_delta"], reverse=True)
        top_k_truth = set(c["config"] for c in sorted_by_delta[:k])
        overlap = sum(1 for c in top_k_disp if c["config"] in top_k_truth)

        print(f"  Top-{k:2d} by displacement_rho: {hit_rate:.0%} have positive delta, "
              f"{overlap}/{k} overlap with best configs")

    # Save results
    validation = {
        "num_configs": len(common_configs),
        "correlations": correlations,
        "configs": common_configs,
    }
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    # Generate scatter plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (metric, values, label) in zip(axes, [
            ("disp_rho", disp_rhos, "Displacement ρ"),
            ("pert_rho", pert_rhos, "Perturbation ρ"),
            ("fp_dist", fp_dists, "Fixed-point dist"),
        ]):
            colors = ['red' if d > 0 else 'blue' for d in deltas]
            ax.scatter(values, deltas, c=colors, alpha=0.5, s=20)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(label)
            ax.set_ylabel('Benchmark delta')
            r = correlations.get(metric, {}).get("spearman_r", 0)
            ax.set_title(f'{label} vs delta (r={r:.3f})')

        plt.tight_layout()
        plt.savefig(output_dir / "validation_scatter.png", dpi=150)
        plt.close()
        print(f"\nScatter plot saved to {output_dir / 'validation_scatter.png'}")
    except ImportError:
        pass

    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral", type=str, required=True)
    parser.add_argument("--sweep", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/validation")
    args = parser.parse_args()
    validate(args.spectral, args.sweep, args.output)
