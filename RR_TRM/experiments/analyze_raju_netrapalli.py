#!/usr/bin/env python3
"""
Raju-Netrapalli Error Model Analysis for TRM
=============================================

Tests four predictions from the error model applied to iterative refinement:

  1. Spectral norm (rho) estimation at each refinement step via perturbation
     analysis and power iteration.

  2. Accuracy vs. refinement depth T — run T H-cycles and measure exact-match
     and token-level accuracy at each step.

  3. Three-parameter accuracy law fit:
         a_TRM(T) = gamma(q/2, q / (2 * r_step * g(T, rho))) / Gamma(q/2)
     where g(T, rho) = (1 - rho^(2T)) / (1 - rho^2).

  4. Argmax snap-back error correction — after each H-cycle, decode z_H via
     argmax and re-embed, then continue refinement.

Usage
-----
# 1) Train two models (BCE baseline and BrierHalting):
    cd RR_TRM
    DISABLE_COMPILE=1 python pretrain.py --config-name=cfg_maze_bce
    DISABLE_COMPILE=1 python pretrain.py --config-name=cfg_maze_brier

# 2) Analyze each checkpoint:
    python experiments/analyze_raju_netrapalli.py analyze \
        --checkpoint checkpoints/<project>/<run>/step_XXXX \
        --data_path data/maze-30x30-hard-1k \
        --max_steps 64 \
        --output_dir results/rn_analysis/bce

    python experiments/analyze_raju_netrapalli.py analyze \
        --checkpoint checkpoints/<project>/<run>/step_XXXX \
        --data_path data/maze-30x30-hard-1k \
        --max_steps 64 \
        --output_dir results/rn_analysis/brier

# 3) Compare BCE vs BrierHalting:
    python experiments/analyze_raju_netrapalli.py compare \
        results/rn_analysis/bce/results.json \
        results/rn_analysis/brier/results.json \
        --labels BCE BrierHalting \
        --output_dir results/rn_analysis/comparison
"""

import argparse
import json
import os
import sys
import time
import yaml

import numpy as np
import torch
import torch.nn.functional as F

# Add parent dir so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.losses import IGNORE_LABEL_ID
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from dataset.common import PuzzleDatasetMetadata as _DSMeta  # noqa: for type alias
from utils.functions import load_model_class

# Lazy-import scipy + matplotlib (not needed at module load time)
_scipy_imported = False
_plt = None


def _import_scipy():
    global _scipy_imported
    if not _scipy_imported:
        from scipy.special import gammainc as _gi  # noqa
        from scipy.optimize import curve_fit as _cf  # noqa
        _scipy_imported = True
    from scipy.special import gammainc
    from scipy.optimize import curve_fit
    return gammainc, curve_fit


def _import_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_analysis(checkpoint_path: str, data_path: str):
    """Load a trained TRM checkpoint (without torch.compile) for analysis."""

    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Expected all_config.yaml in the checkpoint directory."
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Dataset metadata
    with open(os.path.join(data_path, "test", "dataset.json"), "r") as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))

    # Reconstruct model config
    arch = config_dict["arch"]
    arch_extra = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg = dict(
        **arch_extra,
        batch_size=256,  # arbitrary; only matters for sparse-emb training buffer
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    loss_extra = {k: v for k, v in arch["loss"].items() if k != "name"}

    model_cls = load_model_class(arch["name"])
    loss_head_cls = load_model_class(arch["loss"]["name"])

    with torch.device("cuda"):
        model = loss_head_cls(model_cls(model_cfg), **loss_extra)

    # Load state dict — strip _orig_mod. prefix left by torch.compile
    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.float()  # Cast to float32 for analysis (bfloat16 kills perturbation precision)
    # Also fix CastedEmbedding.cast_to and forward_dtype which .float() doesn't touch
    for m in model.modules():
        if hasattr(m, "cast_to"):
            m.cast_to = torch.float32
        if hasattr(m, "forward_dtype") and not isinstance(m.forward_dtype, str):
            m.forward_dtype = torch.float32
    model.eval()

    inner = model.model.inner  # ACTLossHead -> TRM -> Inner
    return model, inner, inner.config, metadata


def load_test_batches(data_path: str, metadata, batch_size: int = 256):
    """Load all test-set batches into GPU memory."""
    ds = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[data_path],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )
    batches = []
    for _set_name, batch, _gbs in ds:
        batches.append({k: v.cuda() for k, v in batch.items()})
    return batches


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def one_h_cycle(inner, z_H, z_L, input_emb, seq_info):
    """One H-cycle: L_cycles updates of z_L, then one update of z_H."""
    for _ in range(inner.config.L_cycles):
        z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
    z_H = inner.L_level(z_H, z_L, **seq_info)
    return z_H, z_L


@torch.no_grad()
def compute_accuracy(inner, z_H, labels, puzzle_emb_len):
    """Token-level and exact-match accuracy."""
    logits = inner.lm_head(z_H)[:, puzzle_emb_len:]
    preds = torch.argmax(logits, dim=-1)
    mask = labels != IGNORE_LABEL_ID
    counts = mask.sum(-1)
    correct = mask & (preds == labels)
    token_acc = correct.sum().float() / mask.sum().float()
    exact_acc = (correct.sum(-1) == counts).float().mean()
    return token_acc.item(), exact_acc.item()


@torch.no_grad()
def argmax_snapback(inner, z_H, puzzle_emb_len):
    """Decode z_H -> tokens via argmax, re-embed, replace token positions."""
    logits = inner.lm_head(z_H)[:, puzzle_emb_len:]
    tokens = torch.argmax(logits, dim=-1).to(torch.int32)
    clean_emb = inner.embed_tokens(tokens) * inner.embed_scale
    z_H_snap = z_H.clone()
    z_H_snap[:, puzzle_emb_len:] = clean_emb
    return z_H_snap


# ---------------------------------------------------------------------------
# Spectral norm estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_rho_perturbation(inner, z_H, z_L, input_emb, seq_info,
                              num_perturbations: int = 20, epsilon: float = 1e-3):
    """Estimate rho via random perturbations. Returns max, mean, std of ratios."""
    z_H_ref, z_L_ref = one_h_cycle(inner, z_H, z_L, input_emb, seq_info)

    ratios = []
    for _ in range(num_perturbations):
        dH = torch.randn_like(z_H)
        dL = torch.randn_like(z_L)
        d_norm = (dH.flatten().norm() ** 2 + dL.flatten().norm() ** 2).sqrt()
        dH = epsilon * dH / d_norm
        dL = epsilon * dL / d_norm

        z_H_p, z_L_p = one_h_cycle(inner, z_H + dH, z_L + dL, input_emb, seq_info)
        out_norm = ((z_H_p - z_H_ref).flatten().norm() ** 2 +
                    (z_L_p - z_L_ref).flatten().norm() ** 2).sqrt()
        ratios.append((out_norm / epsilon).item())

    return {"max": max(ratios), "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios))}


@torch.no_grad()
def estimate_spectral_norm_power_iter(inner, z_H, z_L, input_emb, seq_info,
                                      num_iters: int = 30, epsilon: float = 1e-3):
    """Power iteration for top singular value of the Jacobian."""
    z_H_base, z_L_base = one_h_cycle(inner, z_H, z_L, input_emb, seq_info)

    v_H = torch.randn_like(z_H)
    v_L = torch.randn_like(z_L)

    sigmas = []
    for _ in range(num_iters):
        v_norm = (v_H.flatten().norm() ** 2 + v_L.flatten().norm() ** 2).sqrt()
        v_H, v_L = v_H / v_norm, v_L / v_norm

        z_H_p, z_L_p = one_h_cycle(inner, z_H + epsilon * v_H,
                                    z_L + epsilon * v_L, input_emb, seq_info)
        u_H = (z_H_p - z_H_base) / epsilon
        u_L = (z_L_p - z_L_base) / epsilon
        sigma = (u_H.flatten().norm() ** 2 + u_L.flatten().norm() ** 2).sqrt()
        sigmas.append(sigma.item())

        v_H, v_L = u_H, u_L  # next direction

    return sigmas[-1], sigmas


# ---------------------------------------------------------------------------
# Three-parameter accuracy law
# ---------------------------------------------------------------------------

def accuracy_law_trm(T, q, r_step, rho):
    """
    a_TRM(T) = P(q/2, q / (2 * r_step * g(T, rho)))

    P is the regularised lower incomplete gamma (scipy.special.gammainc).
    g(T, rho) = (1 - rho^(2T)) / (1 - rho^2)  when rho != 1, else T.
    """
    gammainc, _ = _import_scipy()
    T = np.asarray(T, dtype=np.float64)
    rho = float(rho)
    g_T = T.copy() if abs(rho - 1.0) < 1e-8 else (1 - rho ** (2 * T)) / (1 - rho ** 2)
    a = q / 2.0
    x = np.clip(q / (2 * r_step * g_T), 1e-30, None)
    return gammainc(a, x)


def fit_accuracy_law(T_values, acc_values):
    """Fit (q, r_step, rho) to accuracy-vs-T data via least squares."""
    _, curve_fit = _import_scipy()
    T_arr = np.array(T_values, dtype=np.float64)
    acc_arr = np.array(acc_values, dtype=np.float64)

    best, best_res = None, np.inf
    for q0 in [2, 5, 10, 20, 50]:
        for r0 in [0.01, 0.1, 0.5, 1.0]:
            for rho0 in [0.5, 0.8, 0.95, 1.0, 1.05]:
                try:
                    popt, pcov = curve_fit(
                        accuracy_law_trm, T_arr, acc_arr,
                        p0=[q0, r0, rho0],
                        bounds=([0.5, 1e-8, 0.01], [200, 100, 3.0]),
                        maxfev=20000, method="trf",
                    )
                    res = float(np.sum((accuracy_law_trm(T_arr, *popt) - acc_arr) ** 2))
                    if res < best_res:
                        best_res = res
                        best = (popt, pcov, res)
                except Exception:
                    continue

    if best is None:
        print("  WARNING: curve fitting failed for all initial conditions")
        return None

    popt, pcov, residual = best
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 3
    return {
        "q": float(popt[0]),
        "r_step": float(popt[1]),
        "rho_fit": float(popt[2]),
        "q_err": float(perr[0]),
        "r_step_err": float(perr[1]),
        "rho_fit_err": float(perr[2]),
        "residual": float(residual),
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_analysis(inner, batches, metadata, *,
                 max_steps: int = 64,
                 num_perturbations: int = 20,
                 power_iter_steps: int = 30,
                 power_iter_interval: int = 5):
    """
    Run the complete analysis over test batches.

    For each batch, initialise (z_H, z_L), then run up to *max_steps* H-cycles,
    collecting accuracy, rho, and snap-back accuracy at every step.
    """
    config = inner.config
    plen = inner.puzzle_emb_len
    H_cycles = config.H_cycles  # steps per ACT iteration (for annotation only)

    # Accumulators  (step -> list-of-values)
    acc_tok = {t: [] for t in range(1, max_steps + 1)}
    acc_ex  = {t: [] for t in range(1, max_steps + 1)}
    rho_pert = {t: [] for t in range(1, max_steps + 1)}
    rho_pi   = {t: [] for t in range(1, max_steps + 1)}
    snap_tok = {t: [] for t in range(1, max_steps + 1)}
    snap_ex  = {t: [] for t in range(1, max_steps + 1)}

    wall_start = time.time()

    for bi, batch in enumerate(batches):
        labels = batch["labels"]
        B = labels.shape[0]

        # Input embeddings (fixed for this batch)
        input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        seq_info = dict(
            cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
        )

        # Initialise latent state
        z_H = inner.H_init.view(1, 1, -1).expand(B, config.seq_len + plen, -1).clone()
        z_L = inner.L_init.view(1, 1, -1).expand(B, config.seq_len + plen, -1).clone()
        z_H_snap = z_H.clone()
        z_L_snap = z_L.clone()

        for t in range(1, max_steps + 1):
            # --- Standard refinement ---
            z_H, z_L = one_h_cycle(inner, z_H, z_L, input_emb, seq_info)
            ta, ea = compute_accuracy(inner, z_H, labels, plen)
            acc_tok[t].append(ta)
            acc_ex[t].append(ea)

            # --- Rho (perturbation) ---
            rho_result = estimate_rho_perturbation(
                inner, z_H, z_L, input_emb, seq_info,
                num_perturbations=num_perturbations,
            )
            rho_pert[t].append(rho_result)

            # --- Rho (power iteration) — skip some steps to save time ---
            if t <= 10 or t % power_iter_interval == 0:
                sigma, _ = estimate_spectral_norm_power_iter(
                    inner, z_H, z_L, input_emb, seq_info,
                    num_iters=power_iter_steps,
                )
                rho_pi[t].append(sigma)

            # --- Snap-back refinement ---
            z_H_snap, z_L_snap = one_h_cycle(
                inner, z_H_snap, z_L_snap, input_emb, seq_info
            )
            z_H_snap = argmax_snapback(inner, z_H_snap, plen)
            sta, sea = compute_accuracy(inner, z_H_snap, labels, plen)
            snap_tok[t].append(sta)
            snap_ex[t].append(sea)

            # Progress
            if t <= 5 or t % 10 == 0 or t == max_steps:
                elapsed = time.time() - wall_start
                print(
                    f"  batch {bi+1}/{len(batches)} step {t:3d}: "
                    f"exact={ea:.4f}  tok={ta:.4f}  "
                    f"rho_max={rho_result['max']:.4f}  rho_mean={rho_result['mean']:.4f}  "
                    f"snap_exact={sea:.4f}  [{elapsed:.0f}s]"
                )

    # --- Aggregate across batches ---
    T_values = list(range(1, max_steps + 1))
    results = {
        "T": T_values,
        "H_cycles_per_ACT_step": H_cycles,
        "token_accuracy":     [float(np.mean(acc_tok[t])) for t in T_values],
        "exact_accuracy":     [float(np.mean(acc_ex[t]))  for t in T_values],
        "token_accuracy_std": [float(np.std(acc_tok[t]))  for t in T_values],
        "exact_accuracy_std": [float(np.std(acc_ex[t]))   for t in T_values],
        "rho_max":  [float(np.mean([r["max"]  for r in rho_pert[t]])) for t in T_values],
        "rho_mean": [float(np.mean([r["mean"] for r in rho_pert[t]])) for t in T_values],
        "rho_std":  [float(np.mean([r["std"]  for r in rho_pert[t]])) for t in T_values],
        "rho_power_iter": {
            str(t): float(np.mean(rho_pi[t])) for t in T_values if rho_pi[t]
        },
        "snap_token_accuracy": [float(np.mean(snap_tok[t])) for t in T_values],
        "snap_exact_accuracy": [float(np.mean(snap_ex[t]))  for t in T_values],
    }

    # --- Fit accuracy law ---
    print("\nFitting three-parameter accuracy law...")
    fit = fit_accuracy_law(T_values, results["exact_accuracy"])
    if fit:
        results["fit"] = fit
        print(f"  q       = {fit['q']:.2f} +/- {fit['q_err']:.2f}")
        print(f"  r_step  = {fit['r_step']:.4f} +/- {fit['r_step_err']:.4f}")
        print(f"  rho_fit = {fit['rho_fit']:.4f} +/- {fit['rho_fit_err']:.4f}")
        print(f"  residual = {fit['residual']:.6f}")

    total_time = time.time() - wall_start
    results["wall_time_seconds"] = total_time
    print(f"\nDone in {total_time:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, output_dir, label=""):
    plt = _import_plt()
    os.makedirs(output_dir, exist_ok=True)
    T = results["T"]
    H = results["H_cycles_per_ACT_step"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Raju-Netrapalli Analysis: {label}", fontsize=14, fontweight="bold")

    # (0,0) Accuracy vs T
    ax = axes[0, 0]
    ax.plot(T, results["exact_accuracy"], "b-o", ms=2, label="Exact accuracy")
    ax.plot(T, results["token_accuracy"], "g-o", ms=2, label="Token accuracy")
    # Mark multiples of H_cycles (the "designed" operating points)
    act_steps = [t for t in T if t % H == 0]
    act_acc   = [results["exact_accuracy"][t - 1] for t in act_steps]
    ax.plot(act_steps, act_acc, "rv", ms=5, label=f"ACT boundaries (H_cycles={H})")
    if "fit" in results:
        T_d = np.linspace(1, max(T), 300)
        f = results["fit"]
        ax.plot(T_d, accuracy_law_trm(T_d, f["q"], f["r_step"], f["rho_fit"]),
                "r--", lw=2,
                label=f"Fit: q={f['q']:.1f}, r={f['r_step']:.3f}, rho={f['rho_fit']:.3f}")
    ax.set_xlabel("Refinement Steps T (H-cycles)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Refinement Depth")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,1) Rho vs T
    ax = axes[0, 1]
    ax.plot(T, results["rho_max"], "r-o", ms=2, label="rho_max (perturbation)")
    ax.plot(T, results["rho_mean"], "b-o", ms=2, label="rho_mean (perturbation)")
    pi = results["rho_power_iter"]
    if pi:
        pi_T = [int(k) for k in pi.keys()]
        ax.plot(pi_T, list(pi.values()), "k^", ms=6, label="rho (power iter)")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="rho = 1")
    ax.set_xlabel("Refinement Step t")
    ax.set_ylabel("Spectral Norm rho")
    ax.set_title("Jacobian Spectral Norm")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Snap-back comparison
    ax = axes[0, 2]
    ax.plot(T, results["exact_accuracy"], "b-o", ms=2, label="Standard")
    ax.plot(T, results["snap_exact_accuracy"], "r-o", ms=2, label="Argmax snap-back")
    ax.set_xlabel("Refinement Steps T")
    ax.set_ylabel("Exact Accuracy")
    ax.set_title("Snap-back Error Correction (Exp 4)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) Rho bar chart
    ax = axes[1, 0]
    colors = ["steelblue" if r < 1 else "salmon" for r in results["rho_mean"]]
    ax.bar(T, results["rho_mean"], color=colors, alpha=0.7, width=0.8)
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_xlabel("Refinement Step t")
    ax.set_ylabel("Mean rho")
    ax.set_title("Contraction (blue) / Expansion (red) per Step")
    ax.grid(True, alpha=0.3)

    # (1,1) g(T, rho) curves
    ax = axes[1, 1]
    T_d = np.arange(1, max(T) + 1, dtype=float)
    for rho_val, c, ls, lbl in [
        (0.9, "green", "-", "rho=0.9 (contractive)"),
        (1.0, "gray", "--", "rho=1.0 (random walk)"),
        (1.1, "red", ":", "rho=1.1 (expansive)"),
    ]:
        g = T_d if abs(rho_val - 1) < 1e-8 else (1 - rho_val ** (2 * T_d)) / (1 - rho_val ** 2)
        ax.plot(T_d, g, color=c, ls=ls, lw=2, label=lbl)
    if "fit" in results:
        rho_f = results["fit"]["rho_fit"]
        if abs(rho_f - 1) > 1e-8:
            g = (1 - rho_f ** (2 * T_d)) / (1 - rho_f ** 2)
        else:
            g = T_d
        ax.plot(T_d, g, "b-", lw=2, label=f"rho_fit={rho_f:.3f}")
    ax.set_xlabel("Refinement Steps T")
    ax.set_ylabel("g(T, rho)")
    ax.set_title("Noise Accumulation Function")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # (1,2) Summary
    ax = axes[1, 2]
    ax.axis("off")
    peak = max(results["exact_accuracy"])
    peak_T = T[np.argmax(results["exact_accuracy"])]
    mean_rho = float(np.mean(results["rho_mean"]))
    lines = [
        f"Peak exact accuracy: {peak:.4f} @ T={peak_T}",
        f"Mean rho (perturbation): {mean_rho:.4f}",
        f"rho < 1 (contractive): {'YES' if mean_rho < 1 else 'NO'}",
        "",
    ]
    if "fit" in results:
        f = results["fit"]
        lines += [
            "Fitted parameters:",
            f"  q       = {f['q']:.2f} +/- {f['q_err']:.2f}",
            f"  r_step  = {f['r_step']:.4f} +/- {f['r_step_err']:.4f}",
            f"  rho_fit = {f['rho_fit']:.4f} +/- {f['rho_fit_err']:.4f}",
            f"  residual = {f['residual']:.6f}",
        ]
    snap_peak = max(results["snap_exact_accuracy"])
    snap_peak_T = T[np.argmax(results["snap_exact_accuracy"])]
    lines += [
        "",
        f"Snap-back peak: {snap_peak:.4f} @ T={snap_peak_T}",
        f"Snap-back helps: {'YES' if snap_peak > peak else 'NO'}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    ax.set_title("Summary")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_comparison(results_list, labels, output_dir):
    plt = _import_plt()
    os.makedirs(output_dir, exist_ok=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for i, (res, lbl) in enumerate(zip(results_list, labels)):
        T = res["T"]
        c = colors[i % len(colors)]

        # Accuracy
        axes[0].plot(T, res["exact_accuracy"], "-o", color=c, ms=2, label=lbl)
        # Rho
        axes[1].plot(T, res["rho_mean"], "-o", color=c, ms=2, label=lbl)
        # Snap-back delta
        delta = np.array(res["snap_exact_accuracy"]) - np.array(res["exact_accuracy"])
        axes[2].plot(T, delta, "-o", color=c, ms=2, label=lbl)
        # Rho power iter
        pi = res.get("rho_power_iter", {})
        if pi:
            axes[3].plot([int(k) for k in pi], list(pi.values()), "o-", color=c, ms=3, label=lbl)

    axes[0].set_title("Exact Accuracy vs T")
    axes[0].set_xlabel("T")
    axes[0].set_ylabel("Exact Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Mean rho (perturbation)")
    axes[1].axhline(1.0, color="gray", ls="--")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("rho")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Snap-back Improvement")
    axes[2].axhline(0, color="gray", ls="--")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel("snap_acc - standard_acc")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_title("Spectral Norm (power iter)")
    axes[3].axhline(1.0, color="gray", ls="--")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("rho")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("BCE vs BrierHalting: Raju-Netrapalli Analysis", fontsize=14,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(output_dir, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Print table
    print("\n=== Comparison Summary ===")
    for res, lbl in zip(results_list, labels):
        pk = max(res["exact_accuracy"])
        pk_T = res["T"][np.argmax(res["exact_accuracy"])]
        mr = float(np.mean(res["rho_mean"]))
        fit = res.get("fit", {})
        print(f"  {lbl:20s}  peak={pk:.4f}@T={pk_T}  mean_rho={mr:.4f}"
              f"  rho_fit={fit.get('rho_fit', float('nan')):.4f}")
    print(f"\nSaved comparison plot to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Raju-Netrapalli Error Model Analysis for TRM"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- analyze ---
    p = sub.add_parser("analyze", help="Analyze a single checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--max_steps", type=int, default=64)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_perturbations", type=int, default=20)
    p.add_argument("--power_iter_steps", type=int, default=30)
    p.add_argument("--power_iter_interval", type=int, default=5,
                   help="Compute power-iter rho every N steps (always first 10)")

    # --- compare ---
    p = sub.add_parser("compare", help="Compare results from multiple runs")
    p.add_argument("result_files", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.command == "analyze":
        print(f"Loading model from {args.checkpoint} ...")
        model, inner, config, metadata = load_model_for_analysis(
            args.checkpoint, args.data_path
        )
        print(f"  Model config: hidden={config.hidden_size}, "
              f"H_cycles={config.H_cycles}, L_cycles={config.L_cycles}, "
              f"L_layers={config.L_layers}, halt_max={config.halt_max_steps}")
        print(f"  Dataset: seq_len={metadata.seq_len}, vocab={metadata.vocab_size}")

        print(f"\nLoading test data from {args.data_path} ...")
        batches = load_test_batches(args.data_path, metadata, args.batch_size)
        print(f"  {len(batches)} batch(es)")

        print(f"\nRunning analysis (max_steps={args.max_steps}) ...")
        results = run_analysis(
            inner, batches, metadata,
            max_steps=args.max_steps,
            num_perturbations=args.num_perturbations,
            power_iter_steps=args.power_iter_steps,
            power_iter_interval=args.power_iter_interval,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output_dir}/results.json")

        plot_results(results, args.output_dir, label=args.label or args.checkpoint)

    elif args.command == "compare":
        results_list = []
        for path in args.result_files:
            with open(path) as f:
                results_list.append(json.load(f))
        labels = args.labels or [
            os.path.basename(os.path.dirname(p)) for p in args.result_files
        ]
        plot_comparison(results_list, labels, args.output_dir)


if __name__ == "__main__":
    main()
