"""
DeepPass Junction Diagnosis

Detailed analysis of WHAT ACTUALLY HAPPENS at the junction point when
layer 10 is duplicated in the 7B model (config (10,11) = single-layer dup).

Instead of blindly trying to fix the junction, this script measures:
1. Hidden states at EVERY layer for normal vs duplicated model
2. Cosine similarity, L2 distance, activation statistics at the junction
3. Final hidden state comparison (before LM head)
4. Per-layer "representation drift" between normal and duplicated

This tells us exactly what junction FT needs to correct.
"""

import sys, os, json, torch, copy
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer_duplicator import load_original_model
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path(__file__).parent.parent / "results" / "junction_diagnosis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
DUP_I, DUP_J = 10, 11  # Best config: single-layer duplication of layer 10

TEST_PROMPTS = [
    "What is 15 multiplied by 23?",
    "Explain the concept of recursion in programming.",
    "The capital of France is",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
]


def capture_all_hidden_states(model, tokenizer, prompt, label="model"):
    """
    Run forward pass and capture hidden states at EVERY layer using hooks.

    Returns:
        dict with keys:
            'embedding': output of embedding layer (before any transformer block)
            'layer_inputs': list of inputs to each layer
            'layer_outputs': list of outputs from each layer
            'final_hidden': final hidden state (after last layer + norm)
            'logits': model output logits
    """
    inner = model.model
    layers = inner.layers
    N = len(layers)

    layer_inputs = [None] * N
    layer_outputs = [None] * N
    embedding_output = [None]
    final_hidden = [None]

    hooks = []

    # Hook to capture input and output of each layer
    for layer_idx in range(N):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                # Input is a tuple; first element is the hidden state
                if isinstance(inp, tuple):
                    layer_inputs[idx] = inp[0].detach().cpu().float()
                else:
                    layer_inputs[idx] = inp.detach().cpu().float()

                # Output can be a tuple (hidden_state, ...) or just hidden_state
                if isinstance(out, tuple):
                    layer_outputs[idx] = out[0].detach().cpu().float()
                else:
                    layer_outputs[idx] = out.detach().cpu().float()
            return hook_fn
        hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    # Hook on embed_tokens to capture embedding output
    def embed_hook(module, inp, out):
        embedding_output[0] = out.detach().cpu().float()
    hooks.append(inner.embed_tokens.register_forward_hook(embed_hook))

    # Hook on final layer norm to capture final hidden state
    def norm_hook(module, inp, out):
        final_hidden[0] = out.detach().cpu().float()
    hooks.append(inner.norm.register_forward_hook(norm_hook))

    # Forward pass
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits.detach().cpu().float()

    # Clean up hooks
    for h in hooks:
        h.remove()

    return {
        'embedding': embedding_output[0],
        'layer_inputs': layer_inputs,
        'layer_outputs': layer_outputs,
        'final_hidden': final_hidden[0],
        'logits': logits,
        'num_layers': N,
    }


def compute_tensor_stats(tensor):
    """Compute comprehensive statistics on a tensor."""
    t = tensor.float()
    return {
        'mean': t.mean().item(),
        'std': t.std().item(),
        'min': t.min().item(),
        'max': t.max().item(),
        'abs_mean': t.abs().mean().item(),
        'abs_max': t.abs().max().item(),
        'l2_norm': t.norm(2).item(),
        'num_elements': t.numel(),
    }


def compute_comparison_metrics(a, b):
    """
    Compare two tensors comprehensively.
    Returns dict of metrics.
    """
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()

    # L2 distance
    diff = a_flat - b_flat
    l2_dist = diff.norm(2).item()

    # Relative L2 (normalized by the norm of a)
    a_norm = a_flat.norm(2).item()
    rel_l2 = l2_dist / (a_norm + 1e-8)

    # Per-dimension analysis (last dim = hidden dim)
    # Reshape to (seq_len, hidden_dim) for per-dimension analysis
    if a.dim() == 3:
        a_2d = a.squeeze(0)  # (seq_len, hidden_dim)
        b_2d = b.squeeze(0)
    elif a.dim() == 2:
        a_2d = a
        b_2d = b
    else:
        a_2d = a_flat.unsqueeze(0)
        b_2d = b_flat.unsqueeze(0)

    dim_diff = (a_2d - b_2d).abs()  # (seq_len, hidden_dim)

    # Per hidden-dim: mean absolute difference across sequence
    per_dim_mean_diff = dim_diff.mean(dim=0)  # (hidden_dim,)

    # Threshold for "significantly different"
    a_scale = a_2d.abs().mean().item()
    threshold = a_scale * 0.01  # 1% of mean activation magnitude
    sig_different = (per_dim_mean_diff > threshold).sum().item()
    total_dims = per_dim_mean_diff.shape[0]

    # Mean shift analysis
    a_mean_per_dim = a_2d.mean(dim=0)  # (hidden_dim,)
    b_mean_per_dim = b_2d.mean(dim=0)
    mean_shift = (b_mean_per_dim - a_mean_per_dim)
    mean_shift_magnitude = mean_shift.abs().mean().item()
    mean_shift_direction_cos = torch.nn.functional.cosine_similarity(
        a_mean_per_dim.unsqueeze(0), b_mean_per_dim.unsqueeze(0)
    ).item()

    # Variance comparison
    a_var = a_2d.var(dim=0)  # (hidden_dim,)
    b_var = b_2d.var(dim=0)
    var_ratio = (b_var / (a_var + 1e-8)).mean().item()

    # Angular distance
    angular_dist = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi

    # Statistics of the difference itself
    diff_stats = compute_tensor_stats(diff)

    return {
        'cosine_similarity': cos_sim,
        'angular_distance_degrees': angular_dist,
        'l2_distance': l2_dist,
        'relative_l2': rel_l2,
        'significantly_different_dims': sig_different,
        'total_dims': total_dims,
        'fraction_dims_different': sig_different / (total_dims + 1e-8),
        'mean_shift_magnitude': mean_shift_magnitude,
        'mean_direction_cosine': mean_shift_direction_cos,
        'variance_ratio_mean': var_ratio,
        'diff_stats': diff_stats,
    }


def analyze_junction(normal_states, dup_states, i, j):
    """
    Specifically analyze what happens at the junction.

    For config (10, 11) single-layer dup:
    Normal model:  layers 0-27 (28 layers)
    Dup model:     layers 0-9, 10(first), 10(second), 11-27 (29 layers)

    In the dup model's layer list:
    - Index 0-10: layers 0-10 (same as normal 0-10, first pass through 10)
    - Index 11: layer 10 again (second pass = the duplicated layer)
    - Index 12-28: layers 11-27 (same as normal 11-27)

    The "junction" is between:
    - dup layer index 10's output → dup layer index 11's input
    - (= output of layer 10 first pass → input of layer 10 second pass)
    """
    results = {}

    # === JUNCTION ANALYSIS ===
    # In dup model: layer index 10 output goes to layer index 11 input
    # Layer index 10 output = first pass through layer 10
    # Layer index 11 input = input to second pass through layer 10

    dup_first_pass_output = dup_states['layer_outputs'][j - 1]  # layer 10 first output (index 10)
    dup_second_pass_input = dup_states['layer_inputs'][j]       # layer 10 second input (index 11)

    # In normal model, what does layer 10 see as input?
    normal_layer10_input = normal_states['layer_inputs'][i]
    normal_layer10_output = normal_states['layer_outputs'][i]

    print("\n" + "="*80)
    print("JUNCTION ANALYSIS: Layer 10 first pass output → Layer 10 second pass input")
    print("="*80)

    # 1. Compare input to layer 10 (first time in normal) vs input to layer 10 (first time in dup)
    # These SHOULD be identical since layers 0-9 are the same
    print("\n--- Sanity Check: Normal layer 10 input vs Dup layer 10 first input ---")
    sanity = compute_comparison_metrics(normal_layer10_input, dup_states['layer_inputs'][i])
    print(f"  Cosine similarity: {sanity['cosine_similarity']:.8f}")
    print(f"  L2 distance: {sanity['l2_distance']:.8e}")
    results['sanity_check_pre_junction'] = sanity

    # 2. The key question: What does the second pass of layer 10 receive vs what layer 10 normally receives?
    # Normal layer 10 input = embedding → layers 0-9 output
    # Dup second pass input = embedding → layers 0-9 → layer 10 output (= processed one more time)
    print("\n--- KEY: Normal layer 10 input vs Dup layer 10 SECOND input ---")
    print("  (What layer 10 normally sees vs what it sees the second time)")
    key_comparison = compute_comparison_metrics(normal_layer10_input, dup_second_pass_input)
    for k, v in key_comparison.items():
        if k != 'diff_stats':
            print(f"  {k}: {v}")
    results['normal_input_vs_dup_second_input'] = key_comparison

    # 3. Compare layer 10's first output vs its second output
    # (How much does layer 10 change its own output on second application?)
    dup_second_pass_output = dup_states['layer_outputs'][j]  # layer index 11 = second pass output
    print("\n--- Layer 10 first output vs second output (in dup model) ---")
    first_vs_second_output = compute_comparison_metrics(dup_first_pass_output, dup_second_pass_output)
    for k, v in first_vs_second_output.items():
        if k != 'diff_stats':
            print(f"  {k}: {v}")
    results['first_output_vs_second_output'] = first_vs_second_output

    # 4. Is the second pass input just the first pass output? (sanity check about residual connections)
    print("\n--- Is second pass input == first pass output? ---")
    io_check = compute_comparison_metrics(dup_first_pass_output, dup_second_pass_input)
    print(f"  Cosine similarity: {io_check['cosine_similarity']:.8f}")
    print(f"  L2 distance: {io_check['l2_distance']:.8e}")
    results['output_to_input_continuity'] = io_check

    # 5. Activation magnitude distributions
    print("\n--- Activation Magnitude Distributions ---")
    activations = {
        'normal_layer10_input': normal_layer10_input,
        'normal_layer10_output': normal_layer10_output,
        'dup_first_pass_input': dup_states['layer_inputs'][i],
        'dup_first_pass_output': dup_first_pass_output,
        'dup_second_pass_input': dup_second_pass_input,
        'dup_second_pass_output': dup_second_pass_output,
    }

    act_stats = {}
    for name, tensor in activations.items():
        stats = compute_tensor_stats(tensor)
        act_stats[name] = stats
        print(f"  {name}:")
        print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
              f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
              f"abs_mean={stats['abs_mean']:.6f}")
    results['activation_stats'] = act_stats

    # 6. What does layer 10 DO to its input? (transformation analysis)
    print("\n--- Layer 10 Transformation Analysis ---")
    # Normal: what layer 10 adds (residual contribution)
    normal_residual = normal_layer10_output - normal_layer10_input
    normal_res_stats = compute_tensor_stats(normal_residual)
    print(f"  Normal residual contribution: mean={normal_res_stats['mean']:.6f}, "
          f"std={normal_res_stats['std']:.6f}, l2={normal_res_stats['l2_norm']:.4f}")

    # First pass in dup: what layer 10 adds
    dup_first_residual = dup_first_pass_output - dup_states['layer_inputs'][i]
    dup_first_res_stats = compute_tensor_stats(dup_first_residual)
    print(f"  Dup 1st pass residual:        mean={dup_first_res_stats['mean']:.6f}, "
          f"std={dup_first_res_stats['std']:.6f}, l2={dup_first_res_stats['l2_norm']:.4f}")

    # Second pass in dup: what layer 10 adds
    dup_second_residual = dup_second_pass_output - dup_second_pass_input
    dup_second_res_stats = compute_tensor_stats(dup_second_residual)
    print(f"  Dup 2nd pass residual:        mean={dup_second_res_stats['mean']:.6f}, "
          f"std={dup_second_res_stats['std']:.6f}, l2={dup_second_res_stats['l2_norm']:.4f}")

    # Compare residuals: is the layer doing the same thing both times?
    residual_comparison = compute_comparison_metrics(normal_residual, dup_second_residual)
    print(f"\n  Normal residual vs Dup 2nd pass residual:")
    print(f"    Cosine similarity: {residual_comparison['cosine_similarity']:.6f}")
    print(f"    L2 distance: {residual_comparison['l2_distance']:.4f}")

    results['residual_analysis'] = {
        'normal': normal_res_stats,
        'dup_first': dup_first_res_stats,
        'dup_second': dup_second_res_stats,
        'residual_cos_normal_vs_dup2': residual_comparison['cosine_similarity'],
    }

    return results


def analyze_representation_drift(normal_states, dup_states, i, j):
    """
    For each layer, compare normal vs dup model hidden states.
    This shows how the "damage" from duplication propagates through the network.

    Alignment:
    - Normal layers 0..27 (28 total)
    - Dup layers: 0..10, [10 again], 11..27 → indices 0..28 (29 total)
    - Dup index 0-10 correspond to normal 0-10
    - Dup index 11 is the extra (second pass of 10) — no normal counterpart
    - Dup index 12-28 correspond to normal 11-27
    """
    normal_N = normal_states['num_layers']
    dup_N = dup_states['num_layers']
    block_size = j - i  # = 1 for (10, 11)

    print("\n" + "="*80)
    print("REPRESENTATION DRIFT: Per-layer comparison (normal vs duplicated)")
    print("="*80)

    drift_data = []

    # Pre-junction layers (should be identical)
    print("\n--- Pre-junction layers (0 to {}) ---".format(j - 1))
    for normal_idx in range(j):
        dup_idx = normal_idx
        metrics = compute_comparison_metrics(
            normal_states['layer_outputs'][normal_idx],
            dup_states['layer_outputs'][dup_idx]
        )
        drift_data.append({
            'normal_layer': normal_idx,
            'dup_layer': dup_idx,
            'region': 'pre_junction',
            'cosine_similarity': metrics['cosine_similarity'],
            'l2_distance': metrics['l2_distance'],
            'relative_l2': metrics['relative_l2'],
            'angular_distance': metrics['angular_distance_degrees'],
        })
        cos = metrics['cosine_similarity']
        l2 = metrics['l2_distance']
        status = "IDENTICAL" if cos > 0.9999 else "DIFFERS!"
        print(f"  Normal layer {normal_idx:2d} vs Dup layer {dup_idx:2d}: "
              f"cos={cos:.8f}, L2={l2:.6e} [{status}]")

    # Junction: compare normal layer 10 output vs dup layer 11 output (second pass)
    print(f"\n--- Junction layer (dup extra layer) ---")
    dup_extra_idx = j  # = 11, the duplicated layer
    # Compare against normal layer 10 output (closest "equivalent")
    metrics = compute_comparison_metrics(
        normal_states['layer_outputs'][i],  # normal layer 10 output
        dup_states['layer_outputs'][dup_extra_idx]  # dup second pass output
    )
    drift_data.append({
        'normal_layer': f'{i}(output)',
        'dup_layer': dup_extra_idx,
        'region': 'junction',
        'cosine_similarity': metrics['cosine_similarity'],
        'l2_distance': metrics['l2_distance'],
        'relative_l2': metrics['relative_l2'],
        'angular_distance': metrics['angular_distance_degrees'],
    })
    print(f"  Normal layer {i:2d} output vs Dup layer {dup_extra_idx} (2nd pass) output: "
          f"cos={metrics['cosine_similarity']:.8f}, L2={metrics['l2_distance']:.6e}")

    # Post-junction layers
    print(f"\n--- Post-junction layers ({j} to {normal_N - 1}) ---")
    for normal_idx in range(j, normal_N):
        dup_idx = normal_idx + block_size  # offset by the number of duplicated layers
        metrics = compute_comparison_metrics(
            normal_states['layer_outputs'][normal_idx],
            dup_states['layer_outputs'][dup_idx]
        )
        drift_data.append({
            'normal_layer': normal_idx,
            'dup_layer': dup_idx,
            'region': 'post_junction',
            'cosine_similarity': metrics['cosine_similarity'],
            'l2_distance': metrics['l2_distance'],
            'relative_l2': metrics['relative_l2'],
            'angular_distance': metrics['angular_distance_degrees'],
        })
        cos = metrics['cosine_similarity']
        l2 = metrics['l2_distance']
        print(f"  Normal layer {normal_idx:2d} vs Dup layer {dup_idx:2d}: "
              f"cos={cos:.8f}, L2={l2:.6e}, ang={metrics['angular_distance_degrees']:.4f}deg")

    # Final hidden states
    print(f"\n--- Final hidden states (after norm) ---")
    final_metrics = compute_comparison_metrics(
        normal_states['final_hidden'],
        dup_states['final_hidden']
    )
    print(f"  Cosine similarity: {final_metrics['cosine_similarity']:.8f}")
    print(f"  L2 distance: {final_metrics['l2_distance']:.6e}")
    print(f"  Relative L2: {final_metrics['relative_l2']:.6e}")
    print(f"  Angular distance: {final_metrics['angular_distance_degrees']:.4f} degrees")
    print(f"  Significantly different dims: {final_metrics['significantly_different_dims']}/{final_metrics['total_dims']}")
    print(f"  Mean shift magnitude: {final_metrics['mean_shift_magnitude']:.6e}")
    print(f"  Variance ratio (dup/normal): {final_metrics['variance_ratio_mean']:.6f}")

    # Logit comparison
    print(f"\n--- Logit comparison ---")
    normal_logits = normal_states['logits'][0, -1, :]  # last token logits
    dup_logits = dup_states['logits'][0, -1, :]
    logit_metrics = compute_comparison_metrics(
        normal_logits.unsqueeze(0), dup_logits.unsqueeze(0)
    )
    print(f"  Cosine similarity: {logit_metrics['cosine_similarity']:.8f}")
    print(f"  L2 distance: {logit_metrics['l2_distance']:.6e}")

    # Top-5 token predictions
    normal_top5 = torch.topk(normal_logits, 5)
    dup_top5 = torch.topk(dup_logits, 5)

    return {
        'drift_data': drift_data,
        'final_comparison': final_metrics,
        'logit_comparison': logit_metrics,
        'normal_top5_indices': normal_top5.indices.tolist(),
        'dup_top5_indices': dup_top5.indices.tolist(),
    }


def analyze_dimensional_structure(normal_states, dup_states, i, j):
    """
    Deep dive into the dimensional structure of the difference.
    Is the difference a rotation? a scaling? a shift? structured or random?
    """
    print("\n" + "="*80)
    print("DIMENSIONAL STRUCTURE ANALYSIS")
    print("="*80)

    # Focus on the junction: normal layer 10 output vs dup second pass output
    dup_extra_idx = j  # index 11
    normal_out = normal_states['layer_outputs'][i].squeeze(0).float()  # (seq, hidden)
    dup_out = dup_states['layer_outputs'][dup_extra_idx].squeeze(0).float()  # (seq, hidden)

    diff = dup_out - normal_out  # (seq, hidden)

    # 1. Is the difference low-rank? (SVD analysis)
    print("\n--- SVD of the difference matrix ---")
    U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
    total_energy = (S ** 2).sum().item()

    # How many singular values capture 90%, 95%, 99% of energy?
    cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_components = (cumulative < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% energy captured by {n_components}/{len(S)} components")

    print(f"  Top 10 singular values: {S[:10].tolist()}")
    print(f"  Ratio S[0]/S[1]: {S[0].item()/(S[1].item()+1e-8):.4f}")

    # 2. Is it a mean shift or per-token different?
    print("\n--- Mean shift vs per-token variation ---")
    mean_diff = diff.mean(dim=0)  # (hidden,)
    mean_diff_norm = mean_diff.norm().item()
    per_token_diffs = diff - mean_diff.unsqueeze(0)  # residual after removing mean
    per_token_norm = per_token_diffs.norm().item()
    print(f"  Mean difference vector norm: {mean_diff_norm:.6f}")
    print(f"  Residual (per-token variation) norm: {per_token_norm:.6f}")
    print(f"  Ratio (mean / total): {mean_diff_norm / (mean_diff_norm + per_token_norm + 1e-8):.4f}")

    # 3. Per-dimension analysis
    print("\n--- Per-dimension difference distribution ---")
    dim_diffs = diff.abs().mean(dim=0)  # (hidden,)
    top_dims = torch.topk(dim_diffs, 20)
    print(f"  Top 20 most-changed dimensions:")
    for idx, val in zip(top_dims.indices.tolist(), top_dims.values.tolist()):
        print(f"    dim {idx}: mean_abs_diff = {val:.6f}")

    # 4. Is it correlated with activation magnitude?
    print("\n--- Correlation: difference vs activation magnitude ---")
    normal_mag = normal_out.abs().mean(dim=0)  # (hidden,)
    correlation = torch.corrcoef(torch.stack([dim_diffs, normal_mag]))[0, 1].item()
    print(f"  Pearson correlation: {correlation:.6f}")

    # 5. Distribution of the difference
    print("\n--- Distribution of difference values ---")
    diff_flat = diff.flatten()
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pvals = np.percentile(diff_flat.numpy(), percentiles)
    for p, v in zip(percentiles, pvals):
        print(f"  {p}th percentile: {v:.6f}")
    print(f"  Skewness: {((diff_flat - diff_flat.mean()) ** 3).mean().item() / (diff_flat.std().item() ** 3 + 1e-8):.4f}")
    print(f"  Kurtosis: {((diff_flat - diff_flat.mean()) ** 4).mean().item() / (diff_flat.std().item() ** 4 + 1e-8):.4f}")

    # 6. Compare FINAL outputs too
    print("\n--- SVD of final hidden state difference ---")
    normal_final = normal_states['final_hidden'].squeeze(0).float()
    dup_final = dup_states['final_hidden'].squeeze(0).float()
    final_diff = dup_final - normal_final

    U_f, S_f, Vh_f = torch.linalg.svd(final_diff, full_matrices=False)
    total_energy_f = (S_f ** 2).sum().item()
    cumulative_f = torch.cumsum(S_f ** 2, dim=0) / total_energy_f
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_components = (cumulative_f < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% energy captured by {n_components}/{len(S_f)} components")

    return {
        'svd_junction': {
            'singular_values_top20': S[:20].tolist(),
            'cumulative_energy': cumulative.tolist(),
            'rank_for_90pct': int((cumulative < 0.9).sum().item() + 1),
            'rank_for_99pct': int((cumulative < 0.99).sum().item() + 1),
        },
        'mean_shift': {
            'mean_diff_norm': mean_diff_norm,
            'residual_norm': per_token_norm,
            'ratio': mean_diff_norm / (mean_diff_norm + per_token_norm + 1e-8),
        },
        'top_changed_dims': list(zip(top_dims.indices.tolist(), top_dims.values.tolist())),
        'correlation_diff_vs_magnitude': correlation,
        'svd_final': {
            'singular_values_top20': S_f[:20].tolist(),
            'rank_for_90pct': int((cumulative_f < 0.9).sum().item() + 1),
            'rank_for_99pct': int((cumulative_f < 0.99).sum().item() + 1),
        },
    }


def generate_summary(all_results):
    """Generate a human-readable summary of findings."""
    summary = []
    summary.append("=" * 80)
    summary.append("JUNCTION DIAGNOSIS SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nModel: Qwen2-7B-Instruct (28 layers)")
    summary.append(f"Duplication config: ({DUP_I}, {DUP_J}) — layer 10 duplicated once")
    summary.append(f"Number of test prompts: {len(TEST_PROMPTS)}")
    summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary.append("\n" + "-" * 80)
    summary.append("KEY FINDINGS (averaged across prompts)")
    summary.append("-" * 80)

    # Aggregate across prompts
    pre_junction_cos = []
    junction_cos = []
    post_junction_cos = []
    post_junction_angular = []
    final_cos = []
    final_l2 = []
    junction_input_cos = []
    first_vs_second_cos = []
    mean_shift_ratios = []
    svd_rank_90 = []
    svd_rank_99 = []
    residual_cos = []

    for prompt_results in all_results:
        drift = prompt_results['drift']
        junction = prompt_results['junction']
        dimensional = prompt_results['dimensional']

        for d in drift['drift_data']:
            if d['region'] == 'pre_junction':
                pre_junction_cos.append(d['cosine_similarity'])
            elif d['region'] == 'post_junction':
                post_junction_cos.append(d['cosine_similarity'])
                post_junction_angular.append(d['angular_distance'])

        final_cos.append(drift['final_comparison']['cosine_similarity'])
        final_l2.append(drift['final_comparison']['l2_distance'])

        junction_input_cos.append(junction['normal_input_vs_dup_second_input']['cosine_similarity'])
        first_vs_second_cos.append(junction['first_output_vs_second_output']['cosine_similarity'])

        if 'residual_analysis' in junction:
            residual_cos.append(junction['residual_analysis']['residual_cos_normal_vs_dup2'])

        mean_shift_ratios.append(dimensional['mean_shift']['ratio'])
        svd_rank_90.append(dimensional['svd_junction']['rank_for_90pct'])
        svd_rank_99.append(dimensional['svd_junction']['rank_for_99pct'])

    summary.append(f"\n1. SANITY CHECK: Pre-junction layers (0-10)")
    summary.append(f"   Mean cosine similarity: {np.mean(pre_junction_cos):.8f}")
    summary.append(f"   → {'PASS: layers before junction are identical' if np.mean(pre_junction_cos) > 0.9999 else 'FAIL: unexpected difference!'}")

    summary.append(f"\n2. JUNCTION IMPACT: Normal layer 10 input vs Dup second pass input")
    summary.append(f"   Mean cosine similarity: {np.mean(junction_input_cos):.6f}")
    summary.append(f"   → The second pass sees {'very similar' if np.mean(junction_input_cos) > 0.99 else 'significantly different' if np.mean(junction_input_cos) < 0.9 else 'moderately different'} input")

    summary.append(f"\n3. LAYER 10 SELF-APPLICATION: First output vs Second output")
    summary.append(f"   Mean cosine similarity: {np.mean(first_vs_second_cos):.6f}")
    summary.append(f"   → Layer 10 {'barely changes' if np.mean(first_vs_second_cos) > 0.99 else 'significantly modifies' if np.mean(first_vs_second_cos) < 0.9 else 'moderately modifies'} its output on second application")

    if residual_cos:
        summary.append(f"\n4. RESIDUAL COMPARISON: Normal vs Dup 2nd pass")
        summary.append(f"   Mean cosine similarity: {np.mean(residual_cos):.6f}")
        summary.append(f"   → Layer 10's residual contribution on 2nd pass is {'similar to' if np.mean(residual_cos) > 0.5 else 'very different from'} its normal contribution")

    summary.append(f"\n5. POST-JUNCTION DRIFT:")
    # Group by layer
    n_prompts = len(all_results)
    n_post_layers = len(post_junction_cos) // n_prompts if n_prompts > 0 else 0
    if n_post_layers > 0:
        for layer_offset in range(min(5, n_post_layers)):
            layer_cos = [post_junction_cos[p * n_post_layers + layer_offset] for p in range(n_prompts)]
            layer_ang = [post_junction_angular[p * n_post_layers + layer_offset] for p in range(n_prompts)]
            normal_idx = DUP_J + layer_offset
            summary.append(f"   Layer {normal_idx}: cos={np.mean(layer_cos):.6f}, angular={np.mean(layer_ang):.2f}deg")
        if n_post_layers > 5:
            # Show last few layers
            summary.append(f"   ...")
            for layer_offset in range(max(5, n_post_layers - 3), n_post_layers):
                layer_cos = [post_junction_cos[p * n_post_layers + layer_offset] for p in range(n_prompts)]
                layer_ang = [post_junction_angular[p * n_post_layers + layer_offset] for p in range(n_prompts)]
                normal_idx = DUP_J + layer_offset
                summary.append(f"   Layer {normal_idx}: cos={np.mean(layer_cos):.6f}, angular={np.mean(layer_ang):.2f}deg")

    summary.append(f"\n6. FINAL OUTPUT IMPACT:")
    summary.append(f"   Final hidden state cosine sim: {np.mean(final_cos):.6f}")
    summary.append(f"   Final hidden state L2 dist: {np.mean(final_l2):.4f}")

    drift_direction = "RECOVERING" if (len(post_junction_cos) > 2 and
                                        post_junction_cos[-1] > post_junction_cos[0]) else \
                      "AMPLIFYING" if (len(post_junction_cos) > 2 and
                                       post_junction_cos[-1] < post_junction_cos[0]) else "UNCLEAR"
    summary.append(f"   Drift trend: {drift_direction}")

    summary.append(f"\n7. STRUCTURE OF THE DIFFERENCE:")
    summary.append(f"   Mean shift ratio: {np.mean(mean_shift_ratios):.4f}")
    summary.append(f"   → {'Mostly mean shift' if np.mean(mean_shift_ratios) > 0.5 else 'Mostly per-token variation'}")
    summary.append(f"   SVD rank for 90% energy: {np.mean(svd_rank_90):.1f}")
    summary.append(f"   SVD rank for 99% energy: {np.mean(svd_rank_99):.1f}")
    low_rank = np.mean(svd_rank_90) < 5
    summary.append(f"   → Difference is {'LOW RANK (structured)' if low_rank else 'HIGH RANK (distributed)'}")

    summary.append(f"\n8. IMPLICATIONS FOR FINE-TUNING:")
    if np.mean(mean_shift_ratios) > 0.5 and low_rank:
        summary.append("   → RECOMMENDATION: A simple bias correction or low-rank adapter (LoRA)")
        summary.append("     could fix the junction. MSE loss on hidden states should work.")
    elif np.mean(mean_shift_ratios) > 0.5:
        summary.append("   → RECOMMENDATION: Mean shift correction + MSE loss")
        summary.append("     The difference is mainly a shift, MSE should capture this.")
    elif low_rank:
        summary.append("   → RECOMMENDATION: Low-rank correction (LoRA/spectral)")
        summary.append("     Structured difference can be corrected with few parameters.")
    else:
        summary.append("   → RECOMMENDATION: Full fine-tuning of junction layers needed.")
        summary.append("     Consider KL-div on output distributions + MSE on hidden states.")

    cos_val = np.mean(first_vs_second_cos)
    if cos_val > 0.99:
        summary.append("   NOTE: Layer 10 is nearly idempotent (second pass barely changes output).")
        summary.append("   The performance drop may come from downstream layers seeing")
        summary.append("   slightly out-of-distribution activations that amplify through the network.")
    elif cos_val > 0.9:
        summary.append("   NOTE: Layer 10 makes moderate changes on second pass.")
        summary.append("   Junction FT should teach it to produce useful second-pass output.")
    else:
        summary.append("   NOTE: Layer 10 significantly changes its output on second pass.")
        summary.append("   This is a large distributional shift that needs correction.")

    return "\n".join(summary)


def main():
    print("=" * 80)
    print("DeepPass Junction Diagnosis")
    print(f"Model: {MODEL_PATH}")
    print(f"Config: ({DUP_I}, {DUP_J}) — single layer duplication of layer {DUP_I}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_original_model(MODEL_PATH)
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    print(f"Original model: {N} layers")

    all_results = []

    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'#' * 80}")
        print(f"PROMPT {prompt_idx + 1}/{len(TEST_PROMPTS)}: {prompt}")
        print(f"{'#' * 80}")

        # === 1. Normal model forward pass ===
        print("\n>>> Running normal model (28 layers)...")
        # Restore original layers
        inner.layers = nn.ModuleList(list(original_layers))
        model.config.num_hidden_layers = N

        normal_states = capture_all_hidden_states(model, tokenizer, prompt, "normal")
        print(f"  Captured {normal_states['num_layers']} layer states")

        # Get normal model prediction
        normal_logits = normal_states['logits'][0, -1, :]
        normal_top5 = torch.topk(normal_logits, 5)
        normal_tokens = [tokenizer.decode([idx]) for idx in normal_top5.indices.tolist()]
        print(f"  Top-5 predictions: {list(zip(normal_tokens, normal_top5.values.tolist()))}")

        # === 2. Duplicated model forward pass ===
        print("\n>>> Running duplicated model (29 layers, layer 10 repeated)...")
        # Build duplicated layer sequence
        layer_order = list(range(DUP_J)) + list(range(DUP_I, DUP_J)) + list(range(DUP_J, N))
        new_layers = [original_layers[idx] for idx in layer_order]
        inner.layers = nn.ModuleList(new_layers)
        model.config.num_hidden_layers = len(new_layers)

        dup_states = capture_all_hidden_states(model, tokenizer, prompt, "duplicated")
        print(f"  Captured {dup_states['num_layers']} layer states")

        # Get dup model prediction
        dup_logits = dup_states['logits'][0, -1, :]
        dup_top5 = torch.topk(dup_logits, 5)
        dup_tokens = [tokenizer.decode([idx]) for idx in dup_top5.indices.tolist()]
        print(f"  Top-5 predictions: {list(zip(dup_tokens, dup_top5.values.tolist()))}")

        # === 3. Junction analysis ===
        junction_results = analyze_junction(normal_states, dup_states, DUP_I, DUP_J)

        # === 4. Representation drift ===
        drift_results = analyze_representation_drift(normal_states, dup_states, DUP_I, DUP_J)

        # === 5. Dimensional structure ===
        dimensional_results = analyze_dimensional_structure(normal_states, dup_states, DUP_I, DUP_J)

        prompt_results = {
            'prompt': prompt,
            'prompt_idx': prompt_idx,
            'junction': junction_results,
            'drift': drift_results,
            'dimensional': dimensional_results,
            'normal_top5_tokens': normal_tokens,
            'dup_top5_tokens': dup_tokens,
        }
        all_results.append(prompt_results)

        # Free GPU memory between prompts
        del normal_states, dup_states
        torch.cuda.empty_cache()

    # Restore original model
    inner.layers = nn.ModuleList(list(original_layers))
    model.config.num_hidden_layers = N

    # Generate summary
    summary = generate_summary(all_results)
    print("\n\n" + summary)

    # Save results
    # Convert results to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, tuple):
            return [make_serializable(x) for x in obj]
        return obj

    serializable_results = make_serializable(all_results)

    results_path = RESULTS_DIR / "junction_diagnosis_results.json"
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    summary_path = RESULTS_DIR / "junction_diagnosis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
