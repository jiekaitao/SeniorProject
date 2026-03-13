#!/usr/bin/env python3
"""
Phase A — Theory Advocate: Latent Subspace Contraction Analysis

Resolves the rho > 1 paradox by measuring contraction in the task-relevant subspace.

1. Collect hidden state trajectories for correct vs incorrect mazes
2. PCA to find the "solution-relevant subspace"
3. Compute projected contraction rate within that subspace
4. Track effective dimensionality (participation ratio) across steps
5. Compare between BCE and BrierHalting

Usage:
    python experiments/latent_pca_analysis.py \
        --checkpoint checkpoints/SeniorProjectTRM/BASE_MODEL/step_168696 \
        --data_path data/maze-30x30-hard-1k \
        --max_steps 32 \
        --output_dir results/phase_a/pca_bce \
        --label "BCE (best@ep21600)"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_raju_netrapalli import (
    load_model_for_analysis, load_test_batches, one_h_cycle, IGNORE_LABEL_ID
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def collect_trajectories(inner, batches, max_steps=32, max_samples=512):
    """
    Run refinement and collect z_H trajectories + correctness labels per sample.

    Returns:
        trajectories: (max_steps+1, N, D) — z_H pooled over sequence dim at each step
        is_correct: (N,) bool — whether the sample is fully correct at final step
        per_step_correct: (max_steps, N) bool — correctness at each step
    """
    config = inner.config
    plen = inner.puzzle_emb_len

    all_z_H = []  # list of (max_steps+1, B, D) tensors
    all_correct_final = []
    all_correct_per_step = []
    n_collected = 0

    for batch in batches:
        if n_collected >= max_samples:
            break

        labels = batch["labels"]
        B = labels.shape[0]
        take = min(B, max_samples - n_collected)

        input_emb = inner._input_embeddings(batch["inputs"][:take], batch["puzzle_identifiers"][:take])
        seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
        labs = labels[:take]

        z_H = inner.H_init.view(1, 1, -1).expand(take, config.seq_len + plen, -1).clone()
        z_L = inner.L_init.view(1, 1, -1).expand(take, config.seq_len + plen, -1).clone()

        # Pool z_H: mean over non-puzzle-emb positions
        z_H_pooled = [z_H[:, plen:, :].mean(dim=1).cpu()]  # (B, D)
        step_correct = []

        for t in range(1, max_steps + 1):
            z_H, z_L = one_h_cycle(inner, z_H, z_L, input_emb, seq_info)
            z_H_pooled.append(z_H[:, plen:, :].mean(dim=1).cpu())

            # Correctness at this step
            logits = inner.lm_head(z_H)[:, plen:]
            preds = torch.argmax(logits, dim=-1)
            mask = labs != IGNORE_LABEL_ID
            correct = (mask & (preds == labs)).sum(-1) == mask.sum(-1)
            step_correct.append(correct.cpu())

        traj = torch.stack(z_H_pooled, dim=0)  # (max_steps+1, B, D)
        all_z_H.append(traj)
        all_correct_final.append(step_correct[-1])
        all_correct_per_step.append(torch.stack(step_correct, dim=0))  # (max_steps, B)
        n_collected += take
        print(f"  Collected {n_collected}/{max_samples} samples, "
              f"correct at final step: {step_correct[-1].sum().item()}/{take}")

    trajectories = torch.cat(all_z_H, dim=1)  # (T+1, N, D)
    is_correct = torch.cat(all_correct_final, dim=0)  # (N,)
    per_step_correct = torch.cat(all_correct_per_step, dim=1)  # (T, N)

    return trajectories.numpy(), is_correct.numpy(), per_step_correct.numpy()


def compute_participation_ratio(cov_matrix):
    """Participation ratio = (sum eigenvalues)^2 / sum(eigenvalues^2)"""
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total < 1e-15:
        return 1.0
    return total ** 2 / (eigenvalues ** 2).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--pca_dims", type=int, default=20)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model, inner, config, metadata = load_model_for_analysis(args.checkpoint, args.data_path)
    print(f"  hidden={config.hidden_size}, H_cycles={config.H_cycles}")

    print(f"\nLoading test data from {args.data_path} ...")
    batches = load_test_batches(args.data_path, metadata, batch_size=64)
    print(f"  {len(batches)} batches")

    print(f"\nCollecting trajectories (max_steps={args.max_steps}, max_samples={args.max_samples}) ...")
    t0 = time.time()
    trajectories, is_correct, per_step_correct = collect_trajectories(
        inner, batches, max_steps=args.max_steps, max_samples=args.max_samples
    )
    print(f"  Done in {time.time() - t0:.0f}s")
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Correct at final step: {is_correct.sum()}/{len(is_correct)}")

    T_plus_1, N, D = trajectories.shape
    T = T_plus_1 - 1

    # ---- 1. PCA of decision boundary ----
    print("\nComputing solution-relevant subspace via PCA...")
    final_states = trajectories[-1]  # (N, D)
    correct_mean = final_states[is_correct].mean(axis=0)
    incorrect_mean = final_states[~is_correct].mean(axis=0) if (~is_correct).sum() > 0 else correct_mean

    # PCA on final states
    centered = final_states - final_states.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top-k principal components define the solution subspace
    k = min(args.pca_dims, D)
    V_k = eigenvectors[:, :k]  # (D, k)
    explained_variance = eigenvalues[:k].sum() / eigenvalues.sum()
    print(f"  Top-{k} PCs explain {explained_variance:.2%} of variance")

    # ---- 2. Projected contraction rate ----
    print("\nComputing projected contraction rates...")
    projected_rho = []
    full_rho = []
    participation_ratios = []

    for t in range(1, T + 1):
        # Full-space displacement
        delta = trajectories[t] - trajectories[t - 1]  # (N, D)
        full_norm = np.linalg.norm(delta, axis=1).mean()

        # Projected displacement
        delta_proj = delta @ V_k  # (N, k)
        proj_norm = np.linalg.norm(delta_proj, axis=1).mean()

        if t >= 2:
            prev_delta = trajectories[t - 1] - trajectories[t - 2]
            prev_full = np.linalg.norm(prev_delta, axis=1).mean()
            prev_proj = np.linalg.norm(prev_delta @ V_k, axis=1).mean()

            full_rho.append(full_norm / max(prev_full, 1e-15))
            projected_rho.append(proj_norm / max(prev_proj, 1e-15))
        else:
            full_rho.append(float("nan"))
            projected_rho.append(float("nan"))

        # Participation ratio of state at this step
        step_states = trajectories[t]
        step_centered = step_states - step_states.mean(axis=0)
        step_cov = np.cov(step_centered.T)
        pr = compute_participation_ratio(step_cov)
        participation_ratios.append(pr)

    # ---- 3. Per-step accuracy (for correlation with geometry) ----
    per_step_acc = per_step_correct.mean(axis=1).tolist()  # (T,)

    # ---- 4. Correct vs incorrect trajectory distance ----
    print("\nComputing correct vs incorrect trajectory separation...")
    separation = []
    for t in range(T + 1):
        states_t = trajectories[t]
        if is_correct.sum() > 0 and (~is_correct).sum() > 0:
            c_mean = states_t[is_correct].mean(axis=0)
            i_mean = states_t[~is_correct].mean(axis=0)
            sep = np.linalg.norm(c_mean - i_mean)
        else:
            sep = 0.0
        separation.append(float(sep))

    # ---- Results ----
    def to_native(x):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(x, (np.floating, np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, list):
            return [to_native(v) for v in x]
        return x

    results = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "max_steps": T,
        "num_samples": int(N),
        "num_correct": int(is_correct.sum()),
        "hidden_dim": int(D),
        "pca_dims": k,
        "explained_variance_topk": float(explained_variance),
        "eigenvalue_spectrum": [float(v) for v in eigenvalues[:50]],
        "full_rho": to_native(full_rho),
        "projected_rho": to_native(projected_rho),
        "participation_ratio": to_native(participation_ratios),
        "per_step_accuracy": to_native(per_step_acc),
        "correct_incorrect_separation": to_native(separation),
    }

    json_path = os.path.join(args.output_dir, "pca_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Latent Subspace Analysis: {args.label}", fontsize=14, fontweight="bold")
    steps = list(range(1, T + 1))

    # (0,0) Eigenvalue spectrum
    ax = axes[0, 0]
    ax.semilogy(range(1, min(51, len(eigenvalues) + 1)), eigenvalues[:50], "b-o", ms=3)
    ax.axvline(k, color="red", ls="--", label=f"k={k}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalue Spectrum (top-{k} = {explained_variance:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Full vs projected contraction rate
    ax = axes[0, 1]
    ax.plot(steps, full_rho, "r-o", ms=3, label="Full-space rho")
    ax.plot(steps, projected_rho, "b-o", ms=3, label=f"Projected rho (top-{k} PCs)")
    ax.axhline(1.0, color="gray", ls="--", label="rho = 1")
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Contraction Rate")
    ax.set_title("Full vs Subspace Contraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) Participation ratio
    ax = axes[0, 2]
    ax.plot(steps, participation_ratios, "g-o", ms=3)
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Effective Dimensionality")
    ax.grid(True, alpha=0.3)

    # (1,0) Per-step accuracy
    ax = axes[1, 0]
    ax.plot(steps, per_step_acc, "b-o", ms=3)
    ax.set_xlabel("Refinement Step")
    ax.set_ylabel("Exact Accuracy")
    ax.set_title("Accuracy vs Step")
    ax.grid(True, alpha=0.3)

    # (1,1) Correct vs incorrect separation
    ax = axes[1, 1]
    ax.plot(range(T + 1), separation, "m-o", ms=3)
    ax.set_xlabel("Refinement Step (0 = init)")
    ax.set_ylabel("L2 Distance")
    ax.set_title("Correct vs Incorrect Centroid Separation")
    ax.grid(True, alpha=0.3)

    # (1,2) Summary
    ax = axes[1, 2]
    ax.axis("off")
    mean_full = np.nanmean(full_rho[2:])
    mean_proj = np.nanmean(projected_rho[2:])
    lines = [
        f"Samples: {N} ({is_correct.sum()} correct, {(~is_correct).sum()} incorrect)",
        f"Hidden dim: {D}",
        f"Top-{k} PCs explain: {explained_variance:.1%} variance",
        f"",
        f"Mean full-space rho (steps 3+): {mean_full:.4f}",
        f"Mean projected rho (steps 3+):  {mean_proj:.4f}",
        f"Ratio (projected/full):         {mean_proj/max(mean_full, 1e-10):.4f}",
        f"",
        f"Participation ratio at step 1:  {participation_ratios[0]:.1f}",
        f"Participation ratio at step {T}: {participation_ratios[-1]:.1f}",
        f"",
        f"Max separation: {max(separation):.2f} at step {np.argmax(separation)}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    ax.set_title("Summary")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(args.output_dir, "pca_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
