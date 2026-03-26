"""
Junction Adapter Training with BLOOD Smoothness Loss

Instead of KL divergence with the base model (which FIGHTS iterative refinement
by trying to match baseline output), this trains the adapter to minimize BLOOD
(Between-Layer Transformation Smoothness) at the junction.

Key insight from the BLOOD paper (Ahn et al.):
  - In-distribution inputs produce smooth transformations at each layer
  - OOD inputs produce jagged/high-norm Jacobians
  - After layer duplication, the junction output is "OOD" relative to the
    downstream layers — they've never seen input shaped like this
  - The adapter should make the junction SMOOTH, not match baseline output

BLOOD score at a layer:
  h_in  = input hidden state to the layer
  h_out = layer(h_in)
  z     = randn_like(h_in)
  Jz    = autograd.grad(h_out, h_in, z)   # Jacobian-vector product
  blood = (Jz ** 2).sum() / h_in.numel()  # Frobenius norm estimate

Low BLOOD = smooth transformation = in-distribution input.
The adapter learns to project junction output into the "smooth" manifold
that downstream layers expect, WITHOUT requiring teacher logits.

Architecture:
  Same JunctionAdapter as V4 (residual bottleneck MLP), but:
  - Adapter kept in float32 for stable gradients
  - BLOOD computation in float32 for autograd
  - Model stays in bfloat16

Training:
  loss = blood_score + lambda * ||adapter(h) - h||^2
  Only adapter parameters get gradients. Model is frozen.
"""

import sys
import os
import json
import copy
import time
import gc
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import project modules
sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/adapter_blood")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"


# =============================================================================
# Adapter Architecture (matches V4 but uses raw Parameters for dtype control)
# =============================================================================

class JunctionAdapter(nn.Module):
    """
    Residual bottleneck adapter, kept in float32 for stable BLOOD gradients.

    Forward:
      1. Cast input to float32
      2. Compute residual correction: h + Up(GELU(Down(h)))
      3. Cast output back to bfloat16 for downstream layers

    Initialization: Up-projection starts near-zero, so adapter(x) ~ x.
    """
    def __init__(self, hidden_dim, bottleneck=256):
        super().__init__()
        # Store as raw Parameters (float32)
        self.down_weight = nn.Parameter(
            torch.randn(bottleneck, hidden_dim) * 0.02
        )
        self.up_weight = nn.Parameter(
            torch.randn(hidden_dim, bottleneck) * 0.001
        )

    def forward(self, x):
        dtype = x.dtype
        # Upcast to float32 for stable gradients
        h = x.float()
        d = self.down_weight  # already float32
        u = self.up_weight    # already float32
        correction = F.linear(F.gelu(F.linear(h, d)), u)
        out = h + correction
        # Cast back to model dtype
        return out.to(dtype)


# =============================================================================
# Helper: run a range of layers manually
# =============================================================================

def run_layer_range(inner_model, h, start, end, pos_embeds):
    """Run layers [start, end) on hidden state h."""
    for idx in range(start, end):
        out = inner_model.layers[idx](
            h, position_embeddings=pos_embeds, use_cache=False
        )
        h = out[0] if isinstance(out, tuple) else out
    return h


# =============================================================================
# BLOOD Score Computation
# =============================================================================

def compute_blood_score(layer, h_in, pos_embeds, num_probes=1):
    """
    Compute BLOOD (Between-Layer Transformation Smoothness) for a single layer.

    BLOOD = E_z[ ||J(h_in) z||^2 ] / h_in.numel()

    where J is the Jacobian of the layer's transformation and z ~ N(0, I).
    We estimate this with Jacobian-vector products via autograd.

    Args:
        layer: transformer layer module
        h_in: input hidden state [B, seq_len, hidden_dim], will be detached
        pos_embeds: position embeddings for the layer
        num_probes: number of random probes to average (more = lower variance)

    Returns:
        blood_score: scalar tensor with grad through h_in's upstream
    """
    # Detach h_in from the main graph and enable grad for Jacobian computation
    h = h_in.detach().float().requires_grad_(True)

    # Forward through the target layer
    out = layer(h.to(h_in.dtype), position_embeddings=pos_embeds, use_cache=False)
    h_out = out[0] if isinstance(out, tuple) else out
    h_out = h_out.float()  # float32 for stable autograd

    total_blood = 0.0
    for _ in range(num_probes):
        # Random probe vector
        z = torch.randn_like(h)

        # Jacobian-vector product: J(h) @ z
        Jz = torch.autograd.grad(
            outputs=h_out,
            inputs=h,
            grad_outputs=z,
            create_graph=False,  # No second-order needed
            retain_graph=True,
        )[0]

        # ||Jz||^2 / numel  (Frobenius norm estimate)
        total_blood += (Jz ** 2).sum() / h.numel()

    blood_score = total_blood / num_probes
    return blood_score


def compute_blood_score_differentiable(layer, h_in, pos_embeds, num_probes=1):
    """
    Compute BLOOD score with gradients flowing back through h_in.

    This version keeps the computation graph alive so that gradients
    can flow back to the adapter that produced h_in.

    The trick: we don't detach h_in. Instead, we run the layer,
    compute ||Jz||^2, and let autograd handle the rest.
    """
    # h_in should already have grads from the adapter
    h = h_in.float()
    h.requires_grad_(True)

    # Forward through target layer with SDPA disabled (needed for autograd.grad)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = layer(h.to(h_in.dtype), position_embeddings=pos_embeds, use_cache=False)
    h_out = out[0] if isinstance(out, tuple) else out
    h_out = h_out.float()

    total_blood = torch.tensor(0.0, device=h.device)
    for _ in range(num_probes):
        z = torch.randn_like(h)

        # Jacobian-vector product
        Jz = torch.autograd.grad(
            outputs=h_out,
            inputs=h,
            grad_outputs=z,
            create_graph=True,   # Need this for backprop through adapter
            retain_graph=True,
        )[0]

        total_blood = total_blood + (Jz ** 2).sum() / h.numel()

    return total_blood / num_probes


# =============================================================================
# Training Prompts
# =============================================================================

TRAINING_PROMPTS = [
    # Math-heavy (target domain)
    "What is 78313 multiplied by 88537?",
    "The cube root of 74088 is approximately",
    "What is 9999 multiplied by 9999?",
    "The square root of 152399025 is",
    "What is 123456789 multiplied by 987654321?",
    "What is 31415 divided by 271?",
    "What is 2 to the power of 17?",
    # General knowledge
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "To solve a quadratic equation, you can use",
    "Machine learning models are trained by",
    "The derivative of sin(x) is",
    # Reasoning
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines?",
    # Instruction following
    "List five fruits in alphabetical order, separated by semicolons.",
    "Write exactly three sentences about the moon.",
]


# =============================================================================
# Main Training Function
# =============================================================================

def train_blood_adapter(
    model_path=MODEL_PATH,
    i=10,
    j=11,
    num_steps=200,
    bottleneck=256,
    lr=1e-3,
    reg_lambda=0.01,
    num_probes=1,
    tag=None,
):
    """
    Train a junction adapter using BLOOD smoothness as the loss.

    Args:
        model_path: path to HuggingFace model
        i, j: layer duplication config (duplicate layers [i, j))
        num_steps: training iterations
        bottleneck: adapter bottleneck dimension
        lr: learning rate
        reg_lambda: weight for regularization term ||adapter(h) - h||^2
        num_probes: number of random probes per BLOOD estimate
        tag: experiment tag for saving results
    """
    if tag is None:
        tag = f"blood_{i}_{j}"

    print(f"\n{'='*70}")
    print(f"ADAPTER BLOOD TRAINING: config ({i},{j})")
    print(f"{'='*70}")
    print(f"  Model:       {model_path}")
    print(f"  Bottleneck:  {bottleneck}")
    print(f"  LR:          {lr}")
    print(f"  Reg lambda:  {reg_lambda}")
    print(f"  Num probes:  {num_probes}")
    print(f"  Steps:       {num_steps}")

    # =========================================================================
    # Step 1: Load model
    # =========================================================================
    model, tokenizer = load_original_model(model_path)
    inner = model.model
    original_layers = list(inner.layers)
    N = len(original_layers)
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    dup_count = j - i

    print(f"  Layers:      {N}")
    print(f"  Hidden dim:  {hidden_dim}")
    print(f"  Dup count:   {dup_count}")

    # =========================================================================
    # Step 2: Baseline evaluation (no duplication)
    # =========================================================================
    print("\n--- Step 1: Baseline (no duplication) ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline score: {baseline['score']:.4f}")

    # =========================================================================
    # Step 3: Build duplicated model
    # =========================================================================
    print(f"\n--- Step 2: Building duplicated model ({i},{j}) ---")
    new_layers = list(original_layers[:j])
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(original_layers[idx]))
    new_layers.extend(original_layers[j:])
    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)
    print(f"  New layer count: {new_N}")

    # =========================================================================
    # Step 4: Pre-adapter evaluation (duplication but no adapter)
    # =========================================================================
    print("\n--- Step 3: Pre-adapter score (dup, no adapter) ---")
    pre_result = run_math_probe(gen_fn, verbose=False)
    pre_score = pre_result['score']
    pre_delta = pre_score - baseline['score']
    print(f"  Pre-adapter score: {pre_score:.4f} (delta: {pre_delta:+.4f})")

    # =========================================================================
    # Step 5: Insert adapter at junction
    # =========================================================================
    # For config (i, j), duplication creates:
    #   0..j-1 | j..j+dup-1 (dup block) | j+dup..new_N-1 (original suffix)
    #
    # The junction is at the exit of the duplicated block.
    # The adapter goes BETWEEN the last duplicated layer and the first
    # original suffix layer.
    #
    # junction_layer_idx = j + dup_count - 1  (last layer of dup block)
    # target_layer_idx   = j + dup_count      (first layer after junction)
    #
    # We apply the adapter after junction_layer_idx. But for BLOOD, we
    # measure smoothness at target_layer_idx (the layer that receives
    # the adapter's output).

    junction_exit = j + dup_count - 1   # Last layer of duplicated block
    target_layer_idx = j + dup_count    # First layer after the junction

    print(f"\n--- Step 4: Inserting adapter ---")
    print(f"  Junction exit layer:  {junction_exit}")
    print(f"  BLOOD target layer:   {target_layer_idx}")
    print(f"  Adapter: {hidden_dim} -> {bottleneck} -> {hidden_dim} (float32)")

    adapter = JunctionAdapter(hidden_dim, bottleneck).to(device)
    # Adapter stays in float32, model stays in bfloat16

    adapter_params = list(adapter.parameters())
    trainable = sum(p.numel() for p in adapter_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Adapter params: {trainable:,} ({100*trainable/total_params:.4f}% of model)")

    # =========================================================================
    # Step 6: Freeze model, enable adapter grads
    # =========================================================================
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter_params:
        param.requires_grad = True

    # =========================================================================
    # Step 7: Tokenize training prompts
    # =========================================================================
    print(f"\n--- Step 5: Tokenizing {len(TRAINING_PROMPTS)} prompts ---")
    tokenized_prompts = []
    for p in TRAINING_PROMPTS:
        inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
        tokenized_prompts.append(inp['input_ids'].to(device))

    # =========================================================================
    # Step 8: Training loop
    # =========================================================================
    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=num_steps,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=100,
    )

    print(f"\n--- Step 6: Training ({num_steps} steps, BLOOD loss) ---")
    losses_blood = []
    losses_reg = []
    losses_total = []

    target_layer = inner.layers[target_layer_idx]

    for step in range(num_steps):
        optimizer.zero_grad()

        step_blood = 0.0
        step_reg = 0.0
        step_total = 0.0
        n_prompts = 0

        for input_ids in tokenized_prompts:
            # --- Forward pass up to and through the junction ---
            with torch.no_grad():
                # Embed
                h = inner.embed_tokens(input_ids)
                seq_len = h.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_embeds = inner.rotary_emb(h, position_ids)

                # Run all layers up to and including junction_exit
                h = run_layer_range(inner, h, 0, junction_exit + 1, pos_embeds)

            # h is now the output of the junction exit layer (bfloat16, no grad)
            # Apply adapter (adapter is float32, handles dtype internally)
            h_pre_adapter = h.detach()  # Input to adapter
            h_post_adapter = adapter(h_pre_adapter)  # Output of adapter

            # --- Regularization: penalize adapter for changing too much ---
            # ||adapter(h) - h||^2, computed in float32
            reg_loss = (
                (h_post_adapter.float() - h_pre_adapter.float()) ** 2
            ).mean()

            # --- BLOOD score at the target layer ---
            # The target layer receives h_post_adapter.
            # We need gradients to flow back through h_post_adapter to the adapter.
            blood_loss = compute_blood_score_differentiable(
                target_layer, h_post_adapter, pos_embeds,
                num_probes=num_probes,
            )

            # --- Total loss ---
            loss = blood_loss + reg_lambda * reg_loss
            loss.backward()

            step_blood += blood_loss.item()
            step_reg += reg_loss.item()
            step_total += loss.item()
            n_prompts += 1

            # Clean up
            del h, h_pre_adapter, h_post_adapter, blood_loss, reg_loss, loss

        # Average over prompts
        avg_blood = step_blood / n_prompts
        avg_reg = step_reg / n_prompts
        avg_total = step_total / n_prompts
        losses_blood.append(avg_blood)
        losses_reg.append(avg_reg)
        losses_total.append(avg_total)

        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 25 == 0 or step == 0:
            current_lr = scheduler.get_last_lr()[0]
            # Compute adapter norm (how much it's changing the signal)
            with torch.no_grad():
                adapter_norm = sum(
                    (p ** 2).sum().item() for p in adapter_params
                )
            print(
                f"    Step {step+1:4d}/{num_steps}  "
                f"blood={avg_blood:.6f}  "
                f"reg={avg_reg:.6f}  "
                f"total={avg_total:.6f}  "
                f"lr={current_lr:.2e}  "
                f"adapter_norm={adapter_norm:.4f}"
            )

        # Periodic VRAM cleanup
        if (step + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # =========================================================================
    # Step 9: Post-adapter evaluation
    # =========================================================================
    print("\n--- Step 7: Post-adapter evaluation ---")
    model.eval()

    # We need to insert the adapter into the model for generation.
    # Wrap the junction exit layer so the adapter runs after it.
    class AdapterWrappedLayer(nn.Module):
        """Wraps a transformer layer, applying an adapter AFTER its forward pass."""
        def __init__(self, original_layer, adapter_module):
            super().__init__()
            self.layer = original_layer
            self.adapter = adapter_module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.layer, name)

        def forward(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)
            if isinstance(output, tuple):
                h = output[0]
                h = self.adapter(h)
                return (h,) + output[1:]
            else:
                return self.adapter(output)

    # Wrap the junction exit layer with the trained adapter
    inner.layers[junction_exit] = AdapterWrappedLayer(
        inner.layers[junction_exit], adapter
    )

    post_result = run_math_probe(gen_fn, verbose=False)
    post_score = post_result['score']
    post_delta = post_score - baseline['score']
    adapter_gain = post_score - pre_score

    # =========================================================================
    # Step 10: Results
    # =========================================================================
    print(f"\n  {'='*60}")
    print(f"  BLOOD ADAPTER RESULTS: {tag}")
    print(f"  {'='*60}")
    print(f"  Baseline (no dup):     {baseline['score']:.4f}")
    print(f"  Pre-adapter (dup):     {pre_score:.4f} ({pre_delta:+.4f})")
    print(f"  Post-adapter (dup+ad): {post_score:.4f} ({post_delta:+.4f})")
    print(f"  Adapter gain:          {adapter_gain:+.4f}")
    print(f"  Adapter params:        {trainable:,} ({100*trainable/total_params:.4f}%)")
    print(f"  Final BLOOD loss:      {losses_blood[-1]:.6f}")
    print(f"  Final reg loss:        {losses_reg[-1]:.6f}")
    if pre_delta > 0:
        preserved = post_delta / pre_delta * 100
        print(f"  Improvement preserved: {preserved:.1f}%")
    elif abs(pre_delta) > 1e-6:
        recovery = adapter_gain / abs(pre_delta) * 100
        print(f"  Quality recovery:      {recovery:.1f}%")

    # =========================================================================
    # Step 11: Save results
    # =========================================================================
    result = {
        "tag": tag,
        "config": [i, j],
        "model": model_path,
        "bottleneck": bottleneck,
        "reg_lambda": reg_lambda,
        "num_probes": num_probes,
        "baseline": baseline['score'],
        "pre_adapter": pre_score,
        "pre_delta": pre_delta,
        "post_adapter": post_score,
        "post_delta": post_delta,
        "adapter_gain": adapter_gain,
        "adapter_params": trainable,
        "adapter_pct": 100 * trainable / total_params,
        "steps": num_steps,
        "lr": lr,
        "final_blood_loss": losses_blood[-1],
        "final_reg_loss": losses_reg[-1],
        "final_total_loss": losses_total[-1],
        "losses_blood": losses_blood,
        "losses_reg": losses_reg,
        "losses_total": losses_total,
        "baseline_details": baseline['scores'],
        "pre_details": pre_result['scores'],
        "post_details": post_result['scores'],
    }

    # Save adapter weights
    adapter_save = {
        'adapter': adapter.state_dict(),
        'config': {
            'i': i, 'j': j,
            'hidden_dim': hidden_dim,
            'bottleneck': bottleneck,
            'junction_exit': junction_exit,
            'target_layer': target_layer_idx,
        },
    }
    torch.save(adapter_save, RESULTS_DIR / f"adapter_weights_{tag}.pt")
    print(f"\n  Adapter weights saved to {RESULTS_DIR / f'adapter_weights_{tag}.pt'}")

    with open(RESULTS_DIR / f"results_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {RESULTS_DIR / f'results_{tag}.json'}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# =============================================================================
# Comparison: BLOOD vs KL adapter training
# =============================================================================

def run_comparison(model_path=MODEL_PATH, i=10, j=11, num_steps=200,
                   bottleneck=256, lr=1e-3):
    """
    Run both BLOOD and KL adapter training on the same config for comparison.
    """
    print(f"\n{'='*70}")
    print(f"BLOOD vs KL ADAPTER COMPARISON: config ({i},{j})")
    print(f"{'='*70}")

    # Run BLOOD training
    blood_result = train_blood_adapter(
        model_path=model_path, i=i, j=j,
        num_steps=num_steps, bottleneck=bottleneck, lr=lr,
        reg_lambda=0.01, num_probes=1,
        tag=f"blood_{i}_{j}",
    )

    # Run KL training (import V4 adapter)
    sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts/experiments/junction_ft')
    try:
        from junction_ft_v4_adapter import run_adapter_ft
        kl_result = run_adapter_ft(
            model_path, i, j,
            num_steps=num_steps, bottleneck_dim=bottleneck, lr=lr,
            tag=f"kl_{i}_{j}",
        )
    except ImportError:
        print("  WARNING: Could not import V4 adapter for KL comparison.")
        print("  Running BLOOD-only.")
        kl_result = None

    # Summary
    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS: config ({i},{j})")
    print(f"{'='*70}")
    print(f"  {'Method':>20} {'Baseline':>9} {'Pre':>9} {'Post':>9} {'Gain':>9}")
    print(f"  {'-'*58}")
    print(f"  {'BLOOD':>20} {blood_result['baseline']:9.4f} "
          f"{blood_result['pre_adapter']:9.4f} {blood_result['post_adapter']:9.4f} "
          f"{blood_result['adapter_gain']:+9.4f}")
    if kl_result:
        print(f"  {'KL (V4)':>20} {kl_result['baseline']:9.4f} "
              f"{kl_result['pre_adapter']:9.4f} {kl_result['post_adapter']:9.4f} "
              f"{kl_result['adapter_gain']:+9.4f}")

    comparison = {
        "config": [i, j],
        "blood": blood_result,
        "kl": kl_result,
    }
    with open(RESULTS_DIR / f"comparison_{i}_{j}.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train junction adapter with BLOOD smoothness loss"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help="Path to HuggingFace model"
    )
    parser.add_argument(
        "--i", type=int, default=10,
        help="Start of duplicated block (inclusive)"
    )
    parser.add_argument(
        "--j", type=int, default=11,
        help="End of duplicated block (exclusive)"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of training steps"
    )
    parser.add_argument(
        "--bottleneck", type=int, default=256,
        help="Adapter bottleneck dimension"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=0.01,
        help="Regularization weight for ||adapter(h) - h||^2"
    )
    parser.add_argument(
        "--num-probes", type=int, default=1,
        help="Number of random probes for BLOOD estimate"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run KL adapter for comparison"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Experiment tag"
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.compare:
        run_comparison(
            model_path=args.model, i=args.i, j=args.j,
            num_steps=args.steps, bottleneck=args.bottleneck, lr=args.lr,
        )
    else:
        train_blood_adapter(
            model_path=args.model,
            i=args.i,
            j=args.j,
            num_steps=args.steps,
            bottleneck=args.bottleneck,
            lr=args.lr,
            reg_lambda=args.reg_lambda,
            num_probes=args.num_probes,
            tag=args.tag,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
