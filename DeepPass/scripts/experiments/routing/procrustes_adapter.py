"""
Pure Orthogonal Procrustes Adapter for Layer Duplication Junctions

The minimal possible intervention: a fixed orthogonal rotation that preserves
norms exactly. No training, no gradients, no hyperparameters to tune.

The problem: when layers [i,j) are duplicated, the junction output (what comes
out of the second pass through layer j-1) is distributionally shifted relative
to what layer j expects. Every trainable adapter we tried (KL, BLOOD, task
utility, ReFT gated) hurts good configs because training introduces distortion
even when the shift is small.

The insight: the shift may be primarily a ROTATION of the representation space,
not a scaling or projection. An orthogonal matrix R that solves:

    min_R ||X_dup @ R - X_base||^2_F   subject to R^T R = I

has a closed-form solution via SVD (the Orthogonal Procrustes problem):

    X_base^T @ X_dup = U S V^T
    R = V @ U^T

Properties:
  - Zero information loss: ||x||_2 = ||R @ x||_2 for all x
  - For good configs where the shift is small, R ~ I (near-identity)
  - For bad configs where the shift is large, R corrects it
  - No training whatsoever: purely analytical, no gradients, no NaN, no overfitting
  - Closed-form: computed once from ~50 prompts, applied forever

Procedure:
  1. Load the base model, collect hidden states at the junction point
     (output of layer j-1 in the unduplicated model = what layer j expects)
  2. Build the duplicated model, collect hidden states at the same junction
     (output of the second pass through layer j-1 = what layer j actually gets)
  3. Solve the Orthogonal Procrustes problem in float32
  4. Apply R as a fixed (non-trainable) linear adapter after the junction exit
  5. Evaluate: does R preserve good configs and fix bad ones?
"""

import sys
import os
import json
import copy
import time
import gc

import torch
import torch.nn as nn
from pathlib import Path

# Import project modules
sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/procrustes_adapter")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"


# =============================================================================
# Calibration prompts (diverse, ~50 for robust statistics)
# =============================================================================

CALIBRATION_PROMPTS = [
    # Math-heavy (target domain for math probe)
    "What is 78313 multiplied by 88537?",
    "The cube root of 74088 is approximately",
    "What is 9999 multiplied by 9999?",
    "The square root of 152399025 is",
    "What is 123456789 multiplied by 987654321?",
    "What is 31415 divided by 271?",
    "What is 2 to the power of 17?",
    "What is 456789 squared?",
    "What is 7777777 multiplied by 3333333?",
    "What is 11111111 multiplied by 11111111?",
    "What is 2 raised to the power of 48?",
    "What is 54321 multiplied by 12345?",
    # General knowledge
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "To solve a quadratic equation, you can use",
    "Machine learning models are trained by",
    "The derivative of sin(x) is",
    "A linked list is a data structure where",
    "The speed of light in a vacuum is approximately",
    "The Pythagorean theorem states that",
    "To implement quicksort, you first choose a pivot",
    "The integral of e^x dx equals",
    "In economics, inflation is defined as",
    "The chemical formula for water is H2O because",
    "A recursive function calls itself until",
    "Photosynthesis converts sunlight into",
    "The Fibonacci sequence starts with 0, 1, and then",
    "In linear algebra, the determinant of a matrix",
    "The human genome contains approximately",
    "Gradient descent works by computing the derivative",
    # Reasoning
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines?",
    "All roses are flowers. Some flowers fade quickly. Can we conclude some roses fade quickly?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "There are 3 boxes. One has only apples, one only oranges, one both. All labels are wrong.",
    # Instruction following
    "List five fruits in alphabetical order, separated by semicolons.",
    "Write exactly three sentences about the moon.",
    "Explain quantum entanglement in one paragraph.",
    "Describe a sunset on Mars in vivid detail.",
    "Translate 'hello world' into French, Spanish, and German.",
    # Code
    "Write a Python function to compute the nth Fibonacci number.",
    "Implement binary search in Python.",
    "How do you reverse a linked list in place?",
    "Write a function to check if a string is a palindrome.",
    # Science
    "The second law of thermodynamics states that",
    "DNA replication begins when the enzyme helicase",
    "The Heisenberg uncertainty principle means that",
    "Black holes form when massive stars collapse because",
    "The periodic table is organized by atomic number because",
    # History/Culture
    "The French Revolution began in 1789 when",
    "The Renaissance was a cultural movement that",
]


# =============================================================================
# Helper: Run a range of layers manually
# =============================================================================

def run_layer_range(inner_model, h, start, end, pos_embeds):
    """Run layers [start, end) on hidden state h. Qwen2 needs position_embeddings."""
    for idx in range(start, end):
        out = inner_model.layers[idx](
            h, position_embeddings=pos_embeds, use_cache=False
        )
        h = out[0] if isinstance(out, tuple) else out
    return h


# =============================================================================
# Collect hidden states at the junction point
# =============================================================================

def collect_junction_hidden_states(model, tokenizer, prompts, layer_idx, device):
    """
    Collect hidden states at a specific layer boundary using hooks.

    We hook the OUTPUT of layer `layer_idx`. This gives us what the
    next layer (layer_idx + 1) receives as input.

    Returns:
        all_tokens: tensor of shape [total_tokens, hidden_dim] (float32)
                    All token hidden states concatenated across prompts.
    """
    inner = model.model
    all_states = []

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # Flatten batch and sequence: [B, S, D] -> [B*S, D]
        all_states.append(h.detach().float().cpu().reshape(-1, h.shape[-1]))

    hook = inner.layers[layer_idx].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=64).to(device)
            model(**inp, use_cache=False)

    hook.remove()

    # Concatenate all token hidden states: [total_tokens, hidden_dim]
    all_tokens = torch.cat(all_states, dim=0)
    return all_tokens


# =============================================================================
# Orthogonal Procrustes Solution
# =============================================================================

def solve_procrustes(X_base, X_dup):
    """
    Solve the Orthogonal Procrustes problem:

        min_R ||X_dup @ R - X_base||^2_F   subject to R^T R = I

    Solution: SVD of M = X_base^T @ X_dup, then R = V @ U^T.

    This finds the rotation R such that X_dup @ R is as close as possible
    to X_base in the least-squares sense, while preserving all norms.

    Args:
        X_base: [N, D] hidden states from the base model (what the layer expects)
        X_dup:  [N, D] hidden states from the duplicated model (what it gets)

    Returns:
        R:          [D, D] orthogonal matrix
        diagnostics: dict with SVD details and alignment quality
    """
    assert X_base.shape == X_dup.shape, \
        f"Shape mismatch: base={X_base.shape}, dup={X_dup.shape}"
    N, D = X_base.shape

    print(f"    Procrustes: {N} tokens x {D} dimensions")
    print(f"    X_base norm: {X_base.norm():.4f}, X_dup norm: {X_dup.norm():.4f}")

    # Compute cross-covariance matrix M = X_base^T @ X_dup  [D, D]
    # All in float32 for numerical stability
    X_base_f32 = X_base.float()
    X_dup_f32 = X_dup.float()

    M = X_base_f32.T @ X_dup_f32  # [D, D]

    # SVD of M
    print(f"    Computing SVD of {D}x{D} matrix...")
    t0 = time.time()
    U, S, Vt = torch.linalg.svd(M, full_matrices=True)
    svd_time = time.time() - t0
    print(f"    SVD completed in {svd_time:.2f}s")

    # Optimal rotation: R = V @ U^T
    R = Vt.T @ U.T  # [D, D]

    # Verify orthogonality: R^T @ R should be identity
    RtR = R.T @ R
    I = torch.eye(D, dtype=torch.float32)
    orth_error = (RtR - I).norm().item()

    # Ensure det(R) = +1 (proper rotation, not reflection)
    det_R = torch.linalg.det(R).item()
    if det_R < 0:
        print(f"    WARNING: det(R) = {det_R:.6f}, correcting reflection...")
        # Flip the sign of the last column of V (or equivalently, last row of Vt)
        Vt_corrected = Vt.clone()
        Vt_corrected[-1, :] *= -1
        R = Vt_corrected.T @ U.T
        det_R = torch.linalg.det(R).item()

    # Measure alignment quality
    # Before: ||X_dup - X_base||^2
    residual_before = (X_dup_f32 - X_base_f32).pow(2).sum().item()
    # After:  ||X_dup @ R - X_base||^2
    residual_after = (X_dup_f32 @ R - X_base_f32).pow(2).sum().item()
    # Per-token average
    per_token_before = residual_before / N
    per_token_after = residual_after / N

    # How close is R to identity?
    R_minus_I_norm = (R - I).norm().item()
    R_minus_I_fro_normalized = R_minus_I_norm / (D ** 0.5)  # normalized by sqrt(D)

    # Singular value distribution (tells us about the alignment structure)
    s_min, s_max, s_mean = S.min().item(), S.max().item(), S.mean().item()
    s_std = S.std().item()

    # Angular distance from identity (Frobenius norm of log(R))
    # For orthogonal R, this is related to the rotation angle
    # Simple proxy: average absolute angle of rotation in each eigenvector direction
    # cos(theta_i) = S_i / (||x_base_i|| * ||x_dup_i||)
    # More directly: ||R - I||_F = 2 * sum(sin^2(theta_i/2))

    diagnostics = {
        "N_tokens": N,
        "D_hidden": D,
        "svd_time_s": svd_time,
        "orth_error": orth_error,
        "det_R": det_R,
        "R_minus_I_norm": R_minus_I_norm,
        "R_minus_I_fro_normalized": R_minus_I_fro_normalized,
        "residual_before": per_token_before,
        "residual_after": per_token_after,
        "reduction_pct": (1 - residual_after / max(residual_before, 1e-10)) * 100,
        "singular_values": {
            "min": s_min, "max": s_max, "mean": s_mean, "std": s_std,
            "top5": S[:5].tolist(),
            "bottom5": S[-5:].tolist(),
        },
    }

    print(f"    Orthogonality error:  {orth_error:.2e}")
    print(f"    det(R):              {det_R:.6f}")
    print(f"    ||R - I||_F:         {R_minus_I_norm:.4f}")
    print(f"    ||R - I||_F / sqrt(D): {R_minus_I_fro_normalized:.6f}")
    print(f"    Residual before:     {per_token_before:.4f}")
    print(f"    Residual after:      {per_token_after:.4f}")
    print(f"    Reduction:           {diagnostics['reduction_pct']:.1f}%")
    print(f"    Singular values:     min={s_min:.4f} max={s_max:.4f} "
          f"mean={s_mean:.4f} std={s_std:.4f}")

    return R, diagnostics


# =============================================================================
# Procrustes Adapter Module (wraps a layer, applies R after it)
# =============================================================================

class ProcrustesAdapterLayer(nn.Module):
    """
    Wraps a transformer layer and applies the fixed orthogonal rotation R
    to its output hidden states.

    The rotation R is NOT trainable. It is computed once via SVD and frozen.
    This module simply does: output = layer(input); output[0] = output[0] @ R

    All adapter math is done in float32 for precision, then cast back to
    the model's dtype (bfloat16).
    """
    def __init__(self, original_layer, R):
        """
        Args:
            original_layer: the original transformer layer module
            R: [D, D] orthogonal matrix (float32), will be stored as a buffer
        """
        super().__init__()
        self.layer = original_layer
        # Register R as a non-trainable buffer so it moves with the model
        self.register_buffer('R', R.clone())

    def __getattr__(self, name):
        """Delegate attribute access to the original layer for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, *args, **kwargs):
        """Run the original layer, then apply R to the hidden state output."""
        output = self.layer(*args, **kwargs)
        if isinstance(output, tuple):
            h = output[0]
            # Apply rotation in float32 for precision
            h_f32 = h.float()
            h_rotated = h_f32 @ self.R
            h_rotated = h_rotated.to(h.dtype)
            return (h_rotated,) + output[1:]
        else:
            h_f32 = output.float()
            h_rotated = h_f32 @ self.R
            return h_rotated.to(output.dtype)


# =============================================================================
# Main: Full Procrustes Adapter Pipeline
# =============================================================================

def run_procrustes_adapter(
    model_path=MODEL_PATH,
    i=10,
    j=11,
    num_calibration_prompts=50,
    tag=None,
):
    """
    Full pipeline for the Orthogonal Procrustes adapter.

    Steps:
      1. Load model, evaluate baseline (no duplication)
      2. Collect X_base: hidden states at layer j-1 output in the base model
      3. Build duplicated model, evaluate pre-adapter
      4. Collect X_dup: hidden states at the junction exit in the duplicated model
      5. Solve Procrustes for R
      6. Apply R as a fixed adapter
      7. Evaluate post-adapter

    Args:
        model_path: path to HuggingFace model
        i, j: duplication config (layers [i, j) are duplicated)
        num_calibration_prompts: number of prompts for SVD calibration
        tag: experiment identifier
    """
    if tag is None:
        tag = f"procrustes_{i}_{j}"

    dup_count = j - i

    print(f"\n{'='*70}")
    print(f"ORTHOGONAL PROCRUSTES ADAPTER: config ({i},{j})")
    print(f"{'='*70}")
    print(f"  Model:                  {model_path}")
    print(f"  Duplication:            layers [{i},{j})")
    print(f"  Dup count:              {dup_count}")
    print(f"  Calibration prompts:    {num_calibration_prompts}")
    print(f"  Trainable parameters:   0 (purely analytical)")

    calibration_prompts = CALIBRATION_PROMPTS[:num_calibration_prompts]

    # =========================================================================
    # Step 1: Load model, baseline evaluation
    # =========================================================================
    print(f"\n--- Step 1: Loading model and baseline ---")
    model, tokenizer = load_original_model(model_path)
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    print(f"  Layers:     {N}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Device:     {device}")

    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)

    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline score: {baseline['score']:.4f}")

    # =========================================================================
    # Step 2: Collect X_base (what layer j normally sees as input)
    # =========================================================================
    # In the base model, layer j receives the output of layer j-1.
    # So we hook layer j-1 and collect its output.
    base_hook_layer = j - 1

    print(f"\n--- Step 2: Collecting base hidden states at layer {base_hook_layer} ---")
    X_base = collect_junction_hidden_states(
        model, tokenizer, calibration_prompts, base_hook_layer, device
    )
    print(f"  Collected X_base: {X_base.shape}")
    print(f"  X_base mean norm per token: {X_base.norm(dim=-1).mean():.4f}")

    # =========================================================================
    # Step 3: Build duplicated model, pre-adapter evaluation
    # =========================================================================
    print(f"\n--- Step 3: Building duplicated model ({i},{j}) ---")

    # Deep copy the layers to be duplicated (avoid shared parameter issues
    # when wrapping with adapter later)
    new_layers = list(original_layers[:j])
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(original_layers[idx]))
    new_layers.extend(original_layers[j:])

    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)

    # In the duplicated model, the junction exit is the last layer of the
    # duplicated block. The layer order is:
    #   [0, ..., j-1, i', ..., (j-1)', j, ..., N-1]
    # where i'...(j-1)' are the deep copies.
    # The junction exit = j + dup_count - 1 (last duplicated layer)
    junction_exit = j + dup_count - 1
    target_layer = j + dup_count  # first original layer after junction

    print(f"  New layer count:     {new_N}")
    print(f"  Junction exit layer: {junction_exit}")
    print(f"  Target layer:        {target_layer}")

    # Pre-adapter evaluation
    print("\n  Pre-adapter evaluation (dup, no adapter):")
    pre_result = run_math_probe(gen_fn, verbose=False)
    pre_score = pre_result['score']
    pre_delta = pre_score - baseline['score']
    is_good_config = pre_delta > 0
    print(f"  Pre-adapter score:   {pre_score:.4f} (delta: {pre_delta:+.4f})")
    print(f"  Config type:         {'GOOD' if is_good_config else 'BAD'}")

    # =========================================================================
    # Step 4: Collect X_dup (what the junction actually produces)
    # =========================================================================
    print(f"\n--- Step 4: Collecting dup hidden states at junction exit {junction_exit} ---")
    X_dup = collect_junction_hidden_states(
        model, tokenizer, calibration_prompts, junction_exit, device
    )
    print(f"  Collected X_dup:     {X_dup.shape}")
    print(f"  X_dup mean norm per token: {X_dup.norm(dim=-1).mean():.4f}")

    # Match token counts (should be identical since same prompts, but be safe)
    min_tokens = min(X_base.shape[0], X_dup.shape[0])
    X_base = X_base[:min_tokens]
    X_dup = X_dup[:min_tokens]
    print(f"  Matched to {min_tokens} tokens")

    # =========================================================================
    # Step 5: Solve Orthogonal Procrustes
    # =========================================================================
    print(f"\n--- Step 5: Solving Orthogonal Procrustes ---")
    R, diagnostics = solve_procrustes(X_base, X_dup)

    # =========================================================================
    # Step 6: Apply R as a fixed adapter
    # =========================================================================
    print(f"\n--- Step 6: Applying Procrustes adapter ---")

    # Move R to the model's device (keep float32 for precision)
    R_device = R.to(device)

    # Wrap the junction exit layer with the Procrustes adapter
    inner.layers[junction_exit] = ProcrustesAdapterLayer(
        inner.layers[junction_exit], R_device
    )
    print(f"  Wrapped layer {junction_exit} with Procrustes rotation")

    # =========================================================================
    # Step 7: Post-adapter evaluation
    # =========================================================================
    print(f"\n--- Step 7: Post-adapter evaluation ---")
    model.eval()
    post_result = run_math_probe(gen_fn, verbose=True)
    post_score = post_result['score']
    post_delta = post_score - baseline['score']
    adapter_gain = post_score - pre_score

    # =========================================================================
    # Step 8: Also test WITHOUT the rotation (identity baseline)
    # =========================================================================
    # We already have pre_score for this, but let's be explicit
    identity_score = pre_score
    identity_delta = pre_delta

    # =========================================================================
    # Results Summary
    # =========================================================================
    print(f"\n  {'='*60}")
    print(f"  PROCRUSTES ADAPTER RESULTS: {tag}")
    print(f"  {'='*60}")
    print(f"  Original (no dup):       {baseline['score']:.4f}")
    print(f"  Identity (dup, no R):    {identity_score:.4f} ({identity_delta:+.4f})")
    print(f"  Procrustes (dup + R):    {post_score:.4f} ({post_delta:+.4f})")
    print(f"  Procrustes gain over identity: {adapter_gain:+.4f}")
    print(f"  Config type:             {'GOOD' if is_good_config else 'BAD'}")
    print(f"  ||R - I||_F:             {diagnostics['R_minus_I_norm']:.4f}")
    print(f"  ||R - I||_F / sqrt(D):   {diagnostics['R_minus_I_fro_normalized']:.6f}")
    print(f"  Residual reduction:      {diagnostics['reduction_pct']:.1f}%")
    print(f"  Trainable parameters:    0")

    if is_good_config and abs(identity_delta) > 1e-6:
        # Good config: did we preserve the improvement?
        preserved_identity = identity_delta / baseline['score'] * 100
        preserved_procrustes = post_delta / baseline['score'] * 100
        print(f"\n  GOOD CONFIG ANALYSIS:")
        print(f"    Identity preserves improvement:    {identity_delta:+.4f}")
        print(f"    Procrustes preserves improvement:  {post_delta:+.4f}")
        if abs(identity_delta) > 1e-6:
            pct = post_delta / identity_delta * 100
            print(f"    Procrustes retains {pct:.1f}% of identity's advantage")
    elif not is_good_config and abs(identity_delta) > 1e-6:
        # Bad config: did we recover quality?
        recovery = adapter_gain / abs(identity_delta) * 100
        print(f"\n  BAD CONFIG ANALYSIS:")
        print(f"    Quality lost by duplication: {identity_delta:+.4f}")
        print(f"    Procrustes recovers:         {adapter_gain:+.4f}")
        print(f"    Recovery:                    {recovery:.1f}%")

    # =========================================================================
    # Save results
    # =========================================================================
    result = {
        "tag": tag,
        "config": [i, j],
        "model": model_path,
        "baseline": baseline['score'],
        "identity_score": identity_score,
        "identity_delta": identity_delta,
        "procrustes_score": post_score,
        "procrustes_delta": post_delta,
        "adapter_gain": adapter_gain,
        "is_good_config": is_good_config,
        "trainable_params": 0,
        "diagnostics": diagnostics,
        "baseline_details": baseline['scores'],
        "identity_details": pre_result['scores'],
        "procrustes_details": post_result['scores'],
        "num_calibration_prompts": num_calibration_prompts,
        "num_calibration_tokens": min_tokens,
    }

    results_path = RESULTS_DIR / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save the rotation matrix
    R_path = RESULTS_DIR / f"R_{tag}.pt"
    torch.save({
        'R': R.cpu(),
        'config': {'i': i, 'j': j},
        'diagnostics': diagnostics,
    }, R_path)
    print(f"  Rotation matrix saved to {R_path}")

    # Cleanup
    del model, tokenizer, X_base, X_dup, R, R_device
    gc.collect()
    torch.cuda.empty_cache()

    return result


# =============================================================================
# Sweep: Test across good and bad configs
# =============================================================================

def run_sweep(model_path=MODEL_PATH, num_calibration_prompts=50):
    """
    Run the Procrustes adapter on both good and bad configs.

    Good configs (duplication helps): should preserve improvement (R ~ I).
    Bad configs (duplication hurts): should recover quality (R corrects shift).
    """
    configs = [
        # Good config: duplication helps on math probe
        (10, 11, "good_10_11"),
        # Bad config: duplication hurts on math probe
        (4, 9, "bad_4_9"),
    ]

    results = []
    for ci, cj, label in configs:
        print(f"\n{'#'*70}")
        print(f"# Config ({ci},{cj}) -- {label}")
        print(f"{'#'*70}")

        result = run_procrustes_adapter(
            model_path=model_path,
            i=ci, j=cj,
            num_calibration_prompts=num_calibration_prompts,
            tag=f"procrustes_{label}",
        )
        results.append(result)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("ORTHOGONAL PROCRUSTES ADAPTER -- SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Tag':>25} {'Type':>5} {'Base':>7} {'Ident':>7} "
          f"{'Procr':>7} {'Gain':>7} {'||R-I||':>8}")
    print(f"  {'-'*72}")
    for r in results:
        t = "GOOD" if r['is_good_config'] else "BAD"
        ri_norm = r['diagnostics']['R_minus_I_norm']
        print(
            f"  {r['tag']:>25} {t:>5} "
            f"{r['baseline']:7.4f} {r['identity_score']:7.4f} "
            f"{r['procrustes_score']:7.4f} {r['adapter_gain']:+7.4f} "
            f"{ri_norm:8.4f}"
        )

    # Key analysis
    print(f"\n  KEY QUESTION: Does Procrustes satisfy BOTH requirements?")
    for r in results:
        tag = r['tag']
        if r['is_good_config']:
            # Good config: Procrustes should preserve the improvement
            # (i.e., post_delta >= identity_delta * 0.9)
            if abs(r['identity_delta']) > 1e-6:
                pct = r['procrustes_delta'] / r['identity_delta'] * 100
                verdict = "YES" if pct >= 90 else "PARTIALLY" if pct > 50 else "NO"
                print(f"  [{tag}] GOOD: preserves {pct:.1f}% of improvement -- {verdict}")
            else:
                print(f"  [{tag}] GOOD: identity delta too small to measure")
        else:
            # Bad config: Procrustes should recover quality
            if abs(r['identity_delta']) > 1e-6:
                recovery = r['adapter_gain'] / abs(r['identity_delta']) * 100
                verdict = "YES" if recovery >= 50 else "PARTIALLY" if recovery > 10 else "NO"
                print(f"  [{tag}] BAD:  recovers {recovery:.1f}% of lost quality -- {verdict}")
            else:
                print(f"  [{tag}] BAD:  identity delta too small to measure")

    # Save sweep results
    sweep_path = RESULTS_DIR / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Sweep results saved to {sweep_path}")

    return results


# =============================================================================
# Entry point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Orthogonal Procrustes Adapter for layer duplication junctions"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help="Path to HuggingFace model"
    )
    parser.add_argument("--i", type=int, default=10, help="Dup block start (inclusive)")
    parser.add_argument("--j", type=int, default=11, help="Dup block end (exclusive)")
    parser.add_argument(
        "--calibration-prompts", type=int, default=50,
        help="Number of prompts for Procrustes calibration"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run sweep over good (10,11) and bad (4,9) configs"
    )
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag")

    args = parser.parse_args()

    t0 = time.time()

    if args.sweep:
        run_sweep(
            model_path=args.model,
            num_calibration_prompts=args.calibration_prompts,
        )
    else:
        run_procrustes_adapter(
            model_path=args.model,
            i=args.i, j=args.j,
            num_calibration_prompts=args.calibration_prompts,
            tag=args.tag,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
