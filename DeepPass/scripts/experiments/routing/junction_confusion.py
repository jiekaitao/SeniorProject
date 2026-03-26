"""
Junction Confusion Analysis — Measuring "layer confusion" when layers are duplicated.

When a transformer block [i, j) is duplicated via RYS, the junction point — where
the second pass feeds into layer j — receives hidden states that were never seen
during training. This script quantifies that confusion using three metrics:

1. BLOOD (Between-Layer Transformation Smoothness)
   From Jelenic et al., ICLR 2024. Measures the Frobenius norm of the Jacobian
   of each layer's transformation using Hutchinson's trace estimator. In-distribution
   inputs produce smooth (low-norm) transformations; OOD inputs produce sharp (high-norm)
   ones. Spikes in BLOOD score at junctions indicate layers receiving unexpected input.

2. Per-Layer Mahalanobis Distance
   First, collect hidden-state statistics (mean, variance) at each layer boundary on
   the base model. Then on the duplicated model, measure how far each layer's input
   deviates from the base-model distribution. Uses diagonal covariance approximation
   for efficiency.

3. Angular Distance
   Cosine similarity between consecutive layer inputs and outputs. Measures how much
   each layer rotates its input. Abnormally large angular jumps at junctions indicate
   the layer is trying to compensate for unexpected input.

The script runs both base and duplicated models on the same prompts and produces
side-by-side comparison plots showing where confusion spikes.
"""

import sys
import os
import json
import argparse
import time
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts')
from layer_duplicator import load_original_model, apply_layer_duplication

# Results directory
RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/junction_confusion")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Diverse prompts spanning different tasks
PROMPTS = [
    "What is the capital of France?",
    "Explain how a transformer neural network processes text step by step.",
    "Write a Python function to compute the Fibonacci sequence.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "What is 78313 multiplied by 88537?",
    "Describe a sunset on Mars in vivid detail.",
    "The square root of 152399025 is",
    "List the planets in our solar system in order from the Sun.",
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "Translate 'hello world' into French, Spanish, and German.",
]


def collect_hidden_states(model, tokenizer, prompts, layer_order=None):
    """
    Run prompts through the model and collect hidden states at every layer boundary.

    Returns:
        hidden_states: list of dicts, one per prompt.
            Each dict maps layer_boundary_index -> tensor of shape (seq_len, hidden_dim).
            Boundary 0 = embedding output (input to layer 0).
            Boundary k = output of layer k-1 / input to layer k.
    """
    device = next(model.parameters()).device
    inner = model.model

    if layer_order is None:
        layer_order = list(range(len(inner.layers)))

    all_hidden = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            boundaries = {}
            # Boundary 0: embedding output
            boundaries[0] = h.detach().squeeze(0).float().cpu()

            for step_idx, layer_idx in enumerate(layer_order):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                # Boundary (step_idx + 1): output of this layer / input to next
                boundaries[step_idx + 1] = h.detach().squeeze(0).float().cpu()

        all_hidden.append(boundaries)

    return all_hidden


def compute_blood_scores(model, tokenizer, prompts, layer_order=None, n_hutchinson=5):
    """
    Compute BLOOD scores at every layer boundary.

    Uses Hutchinson's trace estimator: ||J_f||^2_F ~ E[||J_f z||^2] where z ~ N(0, I).
    Average over n_hutchinson random projections for stability.

    Returns:
        blood_scores: dict mapping step_index -> list of scores (one per prompt).
    """
    device = next(model.parameters()).device
    inner = model.model

    if layer_order is None:
        layer_order = list(range(len(inner.layers)))

    num_steps = len(layer_order)
    blood_scores = defaultdict(list)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_ids = inputs["input_ids"]

        # Forward pass to get hidden states at each boundary (no grad)
        with torch.no_grad():
            h_embed = inner.embed_tokens(input_ids)
            seq_len = h_embed.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h_embed, pos_ids)

        # Collect all boundary hidden states first (no grad)
        boundary_h = [h_embed.detach()]
        h = h_embed
        with torch.no_grad():
            for step_idx, layer_idx in enumerate(layer_order):
                out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
                boundary_h.append(h.detach())

        # Now compute Jacobian norms via Hutchinson's estimator (needs grad for one layer at a time)
        for step_idx, layer_idx in enumerate(layer_order):
            h_in = boundary_h[step_idx].detach().clone()

            jac_norm_estimates = []
            for _ in range(n_hutchinson):
                z = torch.randn_like(h_in)
                h_in_grad = h_in.detach().clone().requires_grad_(True)

                # Run just this one layer with grad enabled
                out = inner.layers[layer_idx](
                    h_in_grad, position_embeddings=pos_embeds, use_cache=False
                )
                h_out = out[0] if isinstance(out, tuple) else out

                # Compute J^T z via autograd
                Jz = torch.autograd.grad(
                    h_out, h_in_grad, grad_outputs=z, create_graph=False
                )[0]
                jac_norm_sq = (Jz ** 2).sum().item()
                jac_norm_estimates.append(jac_norm_sq)

            blood_score = float(np.mean(jac_norm_estimates))
            blood_scores[step_idx].append(blood_score)

        # Free memory
        del boundary_h, h_embed, h
        torch.cuda.empty_cache()

    return dict(blood_scores)


def compute_mahalanobis_stats(hidden_states_list):
    """
    Compute per-boundary mean and diagonal variance from base model hidden states.

    Args:
        hidden_states_list: list of dicts from collect_hidden_states.

    Returns:
        stats: dict mapping boundary_index -> (mean, var) where each is (hidden_dim,).
    """
    # Aggregate all hidden states at each boundary across prompts and tokens
    boundary_data = defaultdict(list)
    for hs_dict in hidden_states_list:
        for boundary_idx, tensor in hs_dict.items():
            # tensor shape: (seq_len, hidden_dim)
            boundary_data[boundary_idx].append(tensor)

    stats = {}
    for boundary_idx, tensors in boundary_data.items():
        # Concatenate all tokens across all prompts: (total_tokens, hidden_dim)
        all_tokens = torch.cat(tensors, dim=0)
        mu = all_tokens.mean(dim=0)
        var = all_tokens.var(dim=0) + 1e-6  # diagonal covariance with floor
        stats[boundary_idx] = (mu, var)

    return stats


def compute_mahalanobis_distances(hidden_states_list, base_stats):
    """
    Compute per-boundary Mahalanobis distance of hidden states from base distribution.

    Uses diagonal approximation: d = sqrt(sum((h - mu)^2 / var)).

    Args:
        hidden_states_list: hidden states from the model being evaluated.
        base_stats: stats from compute_mahalanobis_stats on the base model.

    Returns:
        distances: dict mapping boundary_index -> list of distances (one per prompt).
    """
    distances = defaultdict(list)

    for hs_dict in hidden_states_list:
        for boundary_idx, tensor in hs_dict.items():
            if boundary_idx not in base_stats:
                continue
            mu, var = base_stats[boundary_idx]
            # tensor: (seq_len, hidden_dim), compute per-token then average
            diff = tensor - mu.unsqueeze(0)
            d_sq = (diff ** 2 / var.unsqueeze(0)).sum(dim=-1)  # (seq_len,)
            d = d_sq.sqrt().mean().item()
            distances[boundary_idx].append(d)

    return dict(distances)


def compute_angular_distances(hidden_states_list):
    """
    Compute angular distance (1 - cosine_similarity) between consecutive layer boundaries.

    Returns:
        angular: dict mapping step_index -> list of angular distances (one per prompt).
            step_index k = angular distance between boundary k and boundary k+1.
    """
    angular = defaultdict(list)

    for hs_dict in hidden_states_list:
        max_boundary = max(hs_dict.keys())
        for k in range(max_boundary):
            if k not in hs_dict or (k + 1) not in hs_dict:
                continue
            h_in = hs_dict[k]      # (seq_len, hidden_dim)
            h_out = hs_dict[k + 1]  # (seq_len, hidden_dim)

            # Flatten to single vectors (mean over sequence)
            v_in = h_in.mean(dim=0)
            v_out = h_out.mean(dim=0)

            cos_sim = F.cosine_similarity(v_in.unsqueeze(0), v_out.unsqueeze(0)).item()
            angular_dist = 1.0 - cos_sim
            angular[k].append(angular_dist)

    return dict(angular)


def build_layer_order(num_layers, i, j):
    """
    Build the duplicated layer execution order.
    Normal:     [0, 1, ..., j-1, j, ..., N-1]
    Duplicated: [0, ..., j-1, i, ..., j-1, j, ..., N-1]
    """
    return list(range(j)) + list(range(i, j)) + list(range(j, num_layers))


def build_layer_labels(num_layers, layer_order, i, j):
    """
    Build human-readable labels for each step in the layer order.
    For the base model: ["L0", "L1", ..., "L{N-1}"]
    For duplicated: labels showing which layer index, with (dup) suffix for second pass.
    """
    labels = []
    # Track how many times each layer has appeared
    seen_count = defaultdict(int)
    for idx in layer_order:
        seen_count[idx] += 1
        if seen_count[idx] > 1:
            labels.append(f"L{idx}*")
        else:
            labels.append(f"L{idx}")
    return labels


def plot_comparison(base_metrics, dup_metrics, base_labels, dup_labels,
                    metric_name, title, ylabel, output_path,
                    i=None, j=None):
    """
    Plot side-by-side comparison of a metric for base vs duplicated model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    # Compute mean and std across prompts
    def get_mean_std(metrics_dict):
        steps = sorted(metrics_dict.keys())
        means = [np.mean(metrics_dict[s]) for s in steps]
        stds = [np.std(metrics_dict[s]) for s in steps]
        return steps, means, stds

    # Base model
    ax = axes[0]
    steps_b, means_b, stds_b = get_mean_std(base_metrics)
    ax.bar(range(len(steps_b)), means_b, yerr=stds_b, alpha=0.7,
           color="steelblue", capsize=2, label="Base model")
    ax.set_title(f"Base Model — {title}", fontsize=12)
    ax.set_ylabel(ylabel)
    if base_labels and len(base_labels) == len(steps_b):
        ax.set_xticks(range(len(steps_b)))
        ax.set_xticklabels(base_labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Duplicated model
    ax = axes[1]
    steps_d, means_d, stds_d = get_mean_std(dup_metrics)

    # Color bars: junction layers get red
    colors = []
    if i is not None and j is not None:
        dup_order = build_layer_order(len(base_labels), i, j)
        # The junction is at position j (in the duplicated order), which is the
        # first layer after the duplicated block ends its second pass.
        # In the layer_order, the second pass of layer i starts at index j.
        junction_start = j  # start of second pass
        junction_end = j + (j - i)  # end of second pass
        for idx in range(len(steps_d)):
            if junction_start <= idx < junction_end:
                colors.append("salmon")
            elif idx == junction_end and idx < len(steps_d):
                colors.append("red")  # the actual junction point
            else:
                colors.append("steelblue")
    else:
        colors = ["steelblue"] * len(steps_d)

    ax.bar(range(len(steps_d)), means_d, yerr=stds_d, alpha=0.7,
           color=colors, capsize=2, label="Duplicated model")
    ax.set_title(f"Duplicated Model (i={i}, j={j}) — {title}", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer Step")
    if dup_labels and len(dup_labels) == len(steps_d):
        ax.set_xticks(range(len(steps_d)))
        ax.set_xticklabels(dup_labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="Normal layers"),
        Patch(facecolor="salmon", alpha=0.7, label="Duplicated layers (2nd pass)"),
        Patch(facecolor="red", alpha=0.7, label="Junction point"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def compute_confusion_delta(base_metrics, dup_metrics, dup_layer_order, i, j):
    """
    Compute the confusion delta: how much each step's metric differs between
    base and duplicated models.

    For the duplicated model, we need to map steps back to the corresponding
    base-model step for comparison. The first j steps are identical; the duplicated
    block adds (j-i) extra steps; then the remaining steps offset by (j-i).

    Returns:
        deltas: list of (step_idx_in_dup, layer_label, delta, base_val, dup_val)
    """
    num_base = max(base_metrics.keys()) + 1
    block_size = j - i

    deltas = []
    for dup_step in sorted(dup_metrics.keys()):
        dup_val = np.mean(dup_metrics[dup_step])

        # Map dup_step to base_step
        if dup_step < j:
            # Before duplication: same step
            base_step = dup_step
        elif dup_step < j + block_size:
            # During second pass: map to the corresponding layer in base
            # Second pass layer index = i + (dup_step - j)
            # In base, this layer's output is at step i + (dup_step - j) + 1
            # But this is the SECOND time through — no direct base equivalent
            base_step = None
        else:
            # After duplication: offset by block_size
            base_step = dup_step - block_size

        if base_step is not None and base_step in base_metrics:
            base_val = np.mean(base_metrics[base_step])
            delta = dup_val - base_val
        else:
            base_val = None
            delta = None

        layer_idx = dup_layer_order[dup_step] if dup_step < len(dup_layer_order) else "?"
        is_second_pass = j <= dup_step < j + block_size
        label = f"L{layer_idx}" + ("*" if is_second_pass else "")

        deltas.append({
            "dup_step": dup_step,
            "label": label,
            "delta": delta,
            "base_val": base_val,
            "dup_val": dup_val,
            "is_junction": dup_step == j + block_size,
            "is_second_pass": is_second_pass,
        })

    return deltas


def run_analysis(model_path, i, j, n_hutchinson=5):
    """Run the full junction confusion analysis."""
    print(f"\n{'=' * 70}")
    print(f"JUNCTION CONFUSION ANALYSIS")
    print(f"Model: {model_path}")
    print(f"Duplication config: (i={i}, j={j})")
    print(f"{'=' * 70}")

    t0 = time.time()

    # Load model
    model, tokenizer = load_original_model(model_path)
    device = next(model.parameters()).device
    inner = model.model
    num_layers = len(inner.layers)

    print(f"\nModel has {num_layers} layers.")
    print(f"Layer duplication: layers [{i}, {j}) will be repeated.")
    print(f"Duplicated execution order has {num_layers + (j - i)} steps.")

    base_order = list(range(num_layers))
    dup_order = build_layer_order(num_layers, i, j)
    base_labels = build_layer_labels(num_layers, base_order, i, j)
    dup_labels = build_layer_labels(num_layers, dup_order, i, j)

    # =========================================================================
    # Phase 1: Base model — collect hidden states and statistics
    # =========================================================================
    print(f"\n--- Phase 1: Base Model Hidden States ({len(PROMPTS)} prompts) ---")
    base_hidden = collect_hidden_states(model, tokenizer, PROMPTS, layer_order=base_order)
    print(f"  Collected hidden states at {len(base_hidden[0])} boundaries.")

    # Compute Mahalanobis reference statistics from base model
    print("  Computing Mahalanobis reference statistics...")
    base_mahal_stats = compute_mahalanobis_stats(base_hidden)

    # Base model metrics
    print("  Computing base angular distances...")
    base_angular = compute_angular_distances(base_hidden)

    print("  Computing base Mahalanobis distances (self-check)...")
    base_mahal = compute_mahalanobis_distances(base_hidden, base_mahal_stats)

    print(f"  Computing base BLOOD scores (Hutchinson, n={n_hutchinson})...")
    base_blood = compute_blood_scores(model, tokenizer, PROMPTS,
                                       layer_order=base_order, n_hutchinson=n_hutchinson)

    # =========================================================================
    # Phase 2: Duplicated model — run the same prompts through dup order
    # =========================================================================
    print(f"\n--- Phase 2: Duplicated Model Hidden States ---")
    dup_hidden = collect_hidden_states(model, tokenizer, PROMPTS, layer_order=dup_order)
    print(f"  Collected hidden states at {len(dup_hidden[0])} boundaries.")

    # For Mahalanobis on the duplicated model, we need to map boundaries to
    # the base model's reference. Boundary k in duplicated -> same layer output
    # as in base for the first j steps, then the second pass steps have no
    # direct base equivalent, then after the block the layers resume with offset.
    #
    # We build a mapping: dup_boundary -> base_boundary for Mahalanobis lookup.
    block_size = j - i
    dup_mahal_stats_mapped = {}
    for dup_boundary in range(len(dup_order) + 1):
        if dup_boundary <= j:
            # Before or at the end of first pass: same as base
            base_boundary = dup_boundary
        elif dup_boundary <= j + block_size:
            # During second pass: these boundaries don't exist in base
            # Use the base boundary for the corresponding layer output in first pass
            # Second pass starts at layer i. dup_boundary = j + offset means
            # we're at the output of layer i + offset - 1 (on second pass).
            # In base, that output is at boundary i + (dup_boundary - j).
            base_boundary = i + (dup_boundary - j)
        else:
            # After the block: offset by block_size
            base_boundary = dup_boundary - block_size

        if base_boundary in base_mahal_stats:
            dup_mahal_stats_mapped[dup_boundary] = base_mahal_stats[base_boundary]

    # Compute Mahalanobis distances for duplicated model using mapped stats
    print("  Computing duplicated Mahalanobis distances...")
    dup_mahal = defaultdict(list)
    for hs_dict in dup_hidden:
        for boundary_idx, tensor in hs_dict.items():
            if boundary_idx not in dup_mahal_stats_mapped:
                continue
            mu, var = dup_mahal_stats_mapped[boundary_idx]
            diff = tensor - mu.unsqueeze(0)
            d_sq = (diff ** 2 / var.unsqueeze(0)).sum(dim=-1)
            d = d_sq.sqrt().mean().item()
            dup_mahal[boundary_idx].append(d)
    dup_mahal = dict(dup_mahal)

    print("  Computing duplicated angular distances...")
    dup_angular = compute_angular_distances(dup_hidden)

    print("  Computing duplicated BLOOD scores...")
    dup_blood = compute_blood_scores(model, tokenizer, PROMPTS,
                                      layer_order=dup_order, n_hutchinson=n_hutchinson)

    # =========================================================================
    # Phase 3: Plots
    # =========================================================================
    print(f"\n--- Phase 3: Generating Plots ---")

    tag = f"i{i}_j{j}"

    # BLOOD plot
    plot_comparison(
        base_blood, dup_blood, base_labels, dup_labels,
        metric_name="blood",
        title="BLOOD Score (Jacobian Frobenius Norm)",
        ylabel="||J||^2_F (Hutchinson estimate)",
        output_path=RESULTS_DIR / f"blood_{tag}.png",
        i=i, j=j
    )

    # Mahalanobis plot
    # For base, use boundary indices; for dup, use boundary indices.
    # Since the number of boundaries differs, we plot them separately with labels.
    plot_comparison(
        base_mahal, dup_mahal,
        [f"B{k}" for k in sorted(base_mahal.keys())],
        [f"B{k}" for k in sorted(dup_mahal.keys())],
        metric_name="mahalanobis",
        title="Mahalanobis Distance from Base Distribution",
        ylabel="Mahalanobis Distance (diagonal approx)",
        output_path=RESULTS_DIR / f"mahalanobis_{tag}.png",
        i=i, j=j
    )

    # Angular distance plot
    plot_comparison(
        base_angular, dup_angular, base_labels, dup_labels,
        metric_name="angular",
        title="Angular Distance (1 - cosine similarity)",
        ylabel="Angular Distance",
        output_path=RESULTS_DIR / f"angular_{tag}.png",
        i=i, j=j
    )

    # =========================================================================
    # Phase 4: Confusion deltas and summary
    # =========================================================================
    print(f"\n--- Phase 4: Confusion Delta Analysis ---")

    blood_deltas = compute_confusion_delta(base_blood, dup_blood, dup_order, i, j)
    angular_deltas = compute_confusion_delta(base_angular, dup_angular, dup_order, i, j)
    mahal_deltas = compute_confusion_delta(base_mahal, dup_mahal, dup_order, i, j)

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'Step':>5} {'Label':>6} {'BLOOD delta':>14} {'Angular delta':>14} "
          f"{'Mahal delta':>14} {'Flags':>15}")
    print(f"{'-' * 90}")

    summary_rows = []
    for idx in range(len(blood_deltas)):
        bd = blood_deltas[idx]
        ad = angular_deltas[idx] if idx < len(angular_deltas) else {"delta": None}
        md = mahal_deltas[idx] if idx < len(mahal_deltas) else {"delta": None}

        flags = []
        if bd.get("is_junction"):
            flags.append("JUNCTION")
        if bd.get("is_second_pass"):
            flags.append("2ND_PASS")

        def fmt(v):
            return f"{v:+.4f}" if v is not None else "  N/A"

        row = {
            "step": bd["dup_step"],
            "label": bd["label"],
            "blood_delta": bd["delta"],
            "angular_delta": ad["delta"],
            "mahal_delta": md["delta"],
            "blood_dup": bd["dup_val"],
            "angular_dup": ad.get("dup_val"),
            "mahal_dup": md.get("dup_val"),
            "is_junction": bd.get("is_junction", False),
            "is_second_pass": bd.get("is_second_pass", False),
        }
        summary_rows.append(row)

        flag_str = ", ".join(flags) if flags else ""
        print(f"{bd['dup_step']:>5} {bd['label']:>6} {fmt(bd['delta']):>14} "
              f"{fmt(ad['delta']):>14} {fmt(md['delta']):>14} {flag_str:>15}")

    print(f"{'-' * 90}")

    # Highlight top-5 layers by absolute confusion delta for each metric
    print(f"\n--- Top 5 Layers by |Confusion Delta| ---")
    for name, deltas_list in [("BLOOD", blood_deltas), ("Angular", angular_deltas),
                               ("Mahalanobis", mahal_deltas)]:
        valid = [(d["dup_step"], d["label"], abs(d["delta"]), d["delta"])
                 for d in deltas_list if d["delta"] is not None]
        valid.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  {name}:")
        for step, label, abs_delta, delta in valid[:5]:
            print(f"    Step {step:3d} ({label:>5s}): delta = {delta:+.4f}")

    # =========================================================================
    # Phase 5: Save results as JSON
    # =========================================================================
    print(f"\n--- Phase 5: Saving Results ---")

    def metrics_to_serializable(metrics_dict):
        return {str(k): v for k, v in metrics_dict.items()}

    results = {
        "config": {
            "model_path": model_path,
            "i": i,
            "j": j,
            "num_layers": num_layers,
            "num_prompts": len(PROMPTS),
            "prompts": PROMPTS,
            "base_order": base_order,
            "dup_order": dup_order,
        },
        "base_model": {
            "blood": metrics_to_serializable(base_blood),
            "angular": metrics_to_serializable(base_angular),
            "mahalanobis": metrics_to_serializable(base_mahal),
        },
        "duplicated_model": {
            "blood": metrics_to_serializable(dup_blood),
            "angular": metrics_to_serializable(dup_angular),
            "mahalanobis": metrics_to_serializable(dup_mahal),
        },
        "confusion_deltas": {
            "blood": [
                {k: v for k, v in d.items()} for d in blood_deltas
            ],
            "angular": [
                {k: v for k, v in d.items()} for d in angular_deltas
            ],
            "mahalanobis": [
                {k: v for k, v in d.items()} for d in mahal_deltas
            ],
        },
        "summary": summary_rows,
    }

    json_path = RESULTS_DIR / f"metrics_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved metrics: {json_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Results in: {RESULTS_DIR}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Junction Confusion Analysis — Measures layer confusion "
                    "at junction points in duplicated transformer models."
    )
    parser.add_argument(
        "--model", type=str,
        default="/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct",
        help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--i", type=int, required=True,
        help="Start of duplicated block (inclusive)"
    )
    parser.add_argument(
        "--j", type=int, required=True,
        help="End of duplicated block (exclusive)"
    )
    parser.add_argument(
        "--hutchinson-samples", type=int, default=5,
        help="Number of Hutchinson samples for BLOOD estimation (default: 5)"
    )
    args = parser.parse_args()

    run_analysis(args.model, args.i, args.j, n_hutchinson=args.hutchinson_samples)


if __name__ == "__main__":
    main()
