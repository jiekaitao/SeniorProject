#!/usr/bin/env python3
"""
Phase A — Modeling Advocate: Fit alternative accuracy curves to TRM step-accuracy data.

Tests logistic sigmoid, log-normal CDF, and exponential saturation models against
the failed Raju-Netrapalli incomplete gamma CDF.

Usage:
    python experiments/fit_accuracy_curves.py \
        results/rn_analysis/bce_best/results.json \
        results/rn_analysis/brier_best/results.json \
        --labels "BCE (best@ep21600)" "BrierHalting (best@ep5800)" \
        --output_dir results/phase_a/curve_fits
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammainc
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Models ----

def logistic(t, A_max, k, t0):
    """Logistic sigmoid: A(t) = A_max / (1 + exp(-k*(t - t0)))"""
    return A_max / (1.0 + np.exp(-k * (t - t0)))


def lognormal_cdf(t, A_max, mu, sigma):
    """Log-normal CDF: A(t) = A_max * Phi((ln(t) - mu) / sigma)"""
    t = np.maximum(t, 1e-10)
    return A_max * norm.cdf((np.log(t) - mu) / sigma)


def exp_saturation(t, A_max, rate, offset):
    """Exponential saturation: A(t) = A_max * (1 - exp(-rate * (t - offset)))"""
    return A_max * np.maximum(0, 1.0 - np.exp(-rate * np.maximum(t - offset, 0)))


def raju_netrapalli(t, q, r_step, rho):
    """Raju-Netrapalli: incomplete gamma CDF"""
    t = np.asarray(t, dtype=np.float64)
    if abs(rho - 1.0) < 1e-8:
        g_t = t.copy()
    else:
        g_t = (1 - rho ** (2 * t)) / (1 - rho ** 2)
    a = q / 2.0
    x = np.clip(q / (2 * r_step * g_t), 1e-30, None)
    return gammainc(a, x)


MODELS = {
    "Logistic": {
        "func": logistic,
        "p0_list": [(0.87, 2.0, 2.5), (0.87, 5.0, 2.0), (0.87, 1.0, 3.0)],
        "bounds": ([0.5, 0.01, 0.0], [1.0, 50.0, 20.0]),
        "param_names": ["A_max", "k", "t0"],
    },
    "Log-Normal CDF": {
        "func": lognormal_cdf,
        "p0_list": [(0.87, 0.5, 0.5), (0.87, 0.8, 0.3), (0.87, 0.3, 1.0)],
        "bounds": ([0.5, -2.0, 0.01], [1.0, 5.0, 5.0]),
        "param_names": ["A_max", "mu", "sigma"],
    },
    "Exp Saturation": {
        "func": exp_saturation,
        "p0_list": [(0.87, 1.0, 0.5), (0.87, 2.0, 1.0), (0.87, 0.5, 0.0)],
        "bounds": ([0.5, 0.001, -5.0], [1.0, 50.0, 10.0]),
        "param_names": ["A_max", "rate", "offset"],
    },
    "Raju-Netrapalli": {
        "func": raju_netrapalli,
        "p0_list": [(2, 0.1, 0.5), (5, 0.5, 0.8), (10, 1.0, 0.95), (20, 0.01, 1.05)],
        "bounds": ([0.5, 1e-8, 0.01], [200, 100, 3.0]),
        "param_names": ["q", "r_step", "rho"],
    },
}


def fit_model(model_name, t_data, acc_data):
    """Fit a model, return best params and metrics."""
    spec = MODELS[model_name]
    func = spec["func"]
    best, best_res = None, np.inf

    for p0 in spec["p0_list"]:
        try:
            popt, pcov = curve_fit(
                func, t_data, acc_data, p0=p0,
                bounds=spec["bounds"], maxfev=50000, method="trf",
            )
            pred = func(t_data, *popt)
            residual = float(np.sum((pred - acc_data) ** 2))
            if residual < best_res:
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * len(p0)
                best_res = residual
                best = {
                    "params": {n: float(v) for n, v in zip(spec["param_names"], popt)},
                    "errors": {n: float(v) for n, v in zip(spec["param_names"], perr)},
                    "residual_SSE": residual,
                    "RMSE": float(np.sqrt(residual / len(t_data))),
                    "popt": list(popt),
                }
        except Exception:
            continue

    if best is None:
        return {"params": {}, "residual_SSE": float("inf"), "RMSE": float("inf"), "popt": []}

    # AIC (assuming Gaussian errors)
    n = len(t_data)
    k = len(spec["param_names"])
    if best["residual_SSE"] > 0:
        best["AIC"] = n * np.log(best["residual_SSE"] / n) + 2 * k
        best["BIC"] = n * np.log(best["residual_SSE"] / n) + k * np.log(n)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.result_files]

    all_fits = {}
    all_data = []

    for path, label in zip(args.result_files, labels):
        with open(path) as f:
            data = json.load(f)
        t_data = np.array(data["T"], dtype=np.float64)
        acc_data = np.array(data["exact_accuracy"], dtype=np.float64)
        all_data.append((t_data, acc_data, label))

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        fits = {}
        for model_name in MODELS:
            result = fit_model(model_name, t_data, acc_data)
            fits[model_name] = result
            if result["params"]:
                params_str = ", ".join(f"{k}={v:.4f}" for k, v in result["params"].items())
                print(f"  {model_name:20s}  RMSE={result['RMSE']:.6f}  AIC={result.get('AIC', float('nan')):.2f}  {params_str}")
            else:
                print(f"  {model_name:20s}  FAILED")

        # Rank by AIC
        ranked = sorted(
            [(name, f) for name, f in fits.items() if f["params"]],
            key=lambda x: x[1].get("AIC", float("inf"))
        )
        if ranked:
            print(f"\n  Best model (AIC): {ranked[0][0]}")
        all_fits[label] = fits

    # ---- Plot ----
    n_datasets = len(all_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 6), squeeze=False)

    colors = {"Logistic": "#e41a1c", "Log-Normal CDF": "#377eb8",
              "Exp Saturation": "#4daf4a", "Raju-Netrapalli": "#984ea3"}

    for col, (t_data, acc_data, label) in enumerate(all_data):
        ax = axes[0, col]
        ax.plot(t_data, acc_data, "ko", ms=3, alpha=0.5, label="Data")

        t_smooth = np.linspace(0.5, max(t_data), 500)
        fits = all_fits[label]
        ranked = sorted(
            [(name, f) for name, f in fits.items() if f["params"]],
            key=lambda x: x[1].get("AIC", float("inf"))
        )
        for name, fit in ranked:
            func = MODELS[name]["func"]
            pred = func(t_smooth, *fit["popt"])
            ax.plot(t_smooth, pred, color=colors.get(name, "gray"), lw=2,
                    label=f"{name} (RMSE={fit['RMSE']:.5f})")

        ax.set_xlabel("Refinement Step T")
        ax.set_ylabel("Exact Accuracy")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Phase A: Alternative Accuracy Curve Fits", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(args.output_dir, "curve_fits.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {plot_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, "curve_fit_results.json")
    with open(json_path, "w") as f:
        json.dump(all_fits, f, indent=2, default=str)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
