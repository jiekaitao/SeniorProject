#!/usr/bin/env python3
"""
Compare PCA latent subspace analysis results between BCE and BrierHalting.
Generates side-by-side comparison plots.

Usage:
    python experiments/compare_pca.py \
        --bce results/phase_a/pca_bce/pca_results.json \
        --brier results/phase_a/pca_brier/pca_results.json \
        --output_dir results/phase_a/pca_comparison
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bce", required=True)
    parser.add_argument("--brier", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bce = load_results(args.bce)
    brier = load_results(args.brier)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("PCA Latent Subspace Comparison: BCE vs BrierHalting", fontsize=14, fontweight="bold")

    T = bce["max_steps"]
    steps = list(range(1, T + 1))

    # Colors
    c_bce = "#2196F3"
    c_brier = "#FF5722"

    # (0,0) Eigenvalue spectrum comparison
    ax = axes[0, 0]
    n_eig = min(len(bce["eigenvalue_spectrum"]), len(brier["eigenvalue_spectrum"]), 50)
    ax.semilogy(range(1, n_eig + 1), bce["eigenvalue_spectrum"][:n_eig],
                f"-o", color=c_bce, ms=3, label=f"BCE ({bce['explained_variance_topk']:.1%} in top-{bce['pca_dims']})")
    ax.semilogy(range(1, n_eig + 1), brier["eigenvalue_spectrum"][:n_eig],
                f"-s", color=c_brier, ms=3, label=f"BrierHalting ({brier['explained_variance_topk']:.1%} in top-{brier['pca_dims']})")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Full-space displacement rho comparison
    ax = axes[0, 1]
    ax.plot(steps, bce["full_rho"], "-o", color=c_bce, ms=3, label="BCE full")
    ax.plot(steps, brier["full_rho"], "-s", color=c_brier, ms=3, label="BrierHalting full")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="rho = 1")
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Displacement Ratio")
    ax.set_title("Full-Space Displacement Rho")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.0)

    # (0,2) Projected displacement rho comparison
    ax = axes[0, 2]
    ax.plot(steps, bce["projected_rho"], "-o", color=c_bce, ms=3, label="BCE projected")
    ax.plot(steps, brier["projected_rho"], "-s", color=c_brier, ms=3, label="BrierHalting projected")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="rho = 1")
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Displacement Ratio")
    ax.set_title(f"Projected Displacement Rho (top-{bce['pca_dims']} PCs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.0)

    # (1,0) Participation ratio comparison
    ax = axes[1, 0]
    ax.plot(steps, bce["participation_ratio"], "-o", color=c_bce, ms=3, label="BCE")
    ax.plot(steps, brier["participation_ratio"], "-s", color=c_brier, ms=3, label="BrierHalting")
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Effective Dimensionality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Per-step accuracy comparison
    ax = axes[1, 1]
    ax.plot(steps, bce["per_step_accuracy"], "-o", color=c_bce, ms=3, label="BCE")
    ax.plot(steps, brier["per_step_accuracy"], "-s", color=c_brier, ms=3, label="BrierHalting")
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Exact Accuracy")
    ax.set_title("Accuracy vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Correct vs incorrect separation comparison
    ax = axes[1, 2]
    t_range = list(range(T + 1))
    ax.plot(t_range, bce["correct_incorrect_separation"], "-o", color=c_bce, ms=3, label="BCE")
    ax.plot(t_range, brier["correct_incorrect_separation"], "-s", color=c_brier, ms=3, label="BrierHalting")
    ax.set_xlabel("Refinement Step (0 = init)")
    ax.set_ylabel("L2 Distance")
    ax.set_title("Correct vs Incorrect Centroid Separation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(args.output_dir, "pca_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {plot_path}")

    # Print summary stats
    bce_full_rho = [r for r in bce["full_rho"][2:] if r == r]  # skip NaN
    brier_full_rho = [r for r in brier["full_rho"][2:] if r == r]
    bce_proj_rho = [r for r in bce["projected_rho"][2:] if r == r]
    brier_proj_rho = [r for r in brier["projected_rho"][2:] if r == r]

    print(f"\n{'Metric':<40} {'BCE':>10} {'BrierHalting':>12}")
    print("-" * 65)
    print(f"{'Top-20 PCs explained variance':<40} {bce['explained_variance_topk']:>9.1%} {brier['explained_variance_topk']:>11.1%}")
    print(f"{'PC1 eigenvalue':<40} {bce['eigenvalue_spectrum'][0]:>10.4f} {brier['eigenvalue_spectrum'][0]:>12.4f}")
    print(f"{'PC2 eigenvalue':<40} {bce['eigenvalue_spectrum'][1]:>10.4f} {brier['eigenvalue_spectrum'][1]:>12.4f}")
    print(f"{'Mean full-space rho (steps 3+)':<40} {np.mean(bce_full_rho):>10.4f} {np.mean(brier_full_rho):>12.4f}")
    print(f"{'Mean projected rho (steps 3+)':<40} {np.mean(bce_proj_rho):>10.4f} {np.mean(brier_proj_rho):>12.4f}")
    print(f"{'PR at step 1':<40} {bce['participation_ratio'][0]:>10.2f} {brier['participation_ratio'][0]:>12.2f}")
    print(f"{'PR steady-state (steps 10+)':<40} {np.mean(bce['participation_ratio'][9:]):>10.2f} {np.mean(brier['participation_ratio'][9:]):>12.2f}")
    print(f"{'Max separation':<40} {max(bce['correct_incorrect_separation']):>10.3f} {max(brier['correct_incorrect_separation']):>12.3f}")
    print(f"{'Accuracy at step 2':<40} {bce['per_step_accuracy'][1]:>9.1%} {brier['per_step_accuracy'][1]:>11.1%}")
    print(f"{'Final accuracy':<40} {bce['per_step_accuracy'][-1]:>9.1%} {brier['per_step_accuracy'][-1]:>11.1%}")


if __name__ == "__main__":
    main()
