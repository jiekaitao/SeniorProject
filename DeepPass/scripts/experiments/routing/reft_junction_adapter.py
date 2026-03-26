"""
ReFT Junction Adapter — Low-Rank Orthogonal Gated Correction

Combines three ideas to solve the junction adapter problem:

1. ReFT (Representation Finetuning): Operate only in a low-rank subspace.
   Instead of modifying all 3584 dimensions, identify the r dimensions where
   the distributional shift lives and correct only those.

2. Procrustes Alignment: The correction within that subspace is an orthogonal
   rotation — it preserves norms (information), just re-aligns directions.
   Parameterized via Cayley map to stay exactly on SO(r).

3. Input-Conditional Gating: A gate g(x) in [0, 1] that measures how OOD
   the input looks (via learned Mahalanobis distance). When the junction is
   smooth (in-distribution), g -> 0 and the adapter is identity. When the
   junction is broken (OOD), g -> 1 and the full correction applies.

Architecture:
   R       = learned [d, r] projection matrix (what subspace to correct)
   Q       = Cayley(A) where A is skew-symmetric [r, r] (orthogonal rotation)
   b       = learned [r] bias (shift within subspace)
   gate    = sigma(linear(mahalanobis_features(x)))

   Forward:
     z = x @ R                          # project to subspace [*, r]
     z_corrected = z @ Q + b            # rotate + shift in subspace
     delta = (z_corrected - z) @ R^T    # lift correction back to full space
     g = gate(x)                        # scalar gate per token
     output = x + g * delta             # gated residual

Loss: Mahalanobis distance of adapter output w.r.t. the expected hidden-state
distribution at that layer. This targets the DISTRIBUTION of hidden states
(not the model's output logits), so it doesn't fight the improvement from
duplication — it just asks "does the hidden state look normal to downstream
layers?"

Why this works where others fail:
  - KL loss fights improvement: Our loss targets hidden-state distribution,
    not output logits. A config that improves reasoning can do so by changing
    logits while keeping hidden states in-distribution.
  - BLOOD collapses: Our gate naturally prevents collapse — the gate goes to
    zero, making the adapter identity, which is the trivial minimum for both
    our Mahalanobis loss AND task performance.
  - Identity adapter can't fix bad configs: Our gate goes to 1 for OOD inputs,
    allowing the orthogonal correction to re-align the distribution.
  - NaN in bfloat16: All adapter math runs in float32 with careful casting.
"""

import sys
import os
import json
import copy
import time
import gc
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import project modules
sys.path.insert(0, '/blue/cis4914/jietao/DeepPass/scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe

RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results/reft_junction")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/small/Qwen2-7B-Instruct"
HIDDEN_DIM = 3584


# =============================================================================
# Core: Cayley Orthogonal Parameterization
# =============================================================================

def cayley_map(A):
    """
    Cayley map: maps a skew-symmetric matrix A to an orthogonal matrix Q.

    Q = (I - A)(I + A)^{-1}

    This parameterization guarantees Q is in SO(r) for any A.
    When A = 0, Q = I (identity rotation = adapter does nothing).

    All computation in float32 for numerical stability.
    """
    r = A.shape[0]
    I = torch.eye(r, device=A.device, dtype=torch.float32)
    # A should already be skew-symmetric, but enforce it
    A_skew = (A - A.T) / 2.0
    # (I + A) must be invertible — guaranteed for skew-symmetric A
    # Use solve instead of inverse for numerical stability
    Q = torch.linalg.solve(I + A_skew, I - A_skew)
    return Q


# =============================================================================
# Core: Running Statistics Tracker (for Mahalanobis distance)
# =============================================================================

class RunningStats(nn.Module):
    """
    Tracks running mean and covariance of hidden states in a low-rank subspace.

    Used to compute Mahalanobis distance: how far is this hidden state from
    what the layer normally sees?

    We track statistics in the projected r-dimensional subspace (not full d),
    making the covariance matrix r x r instead of d x d.

    All statistics maintained in float32 for numerical precision.
    """
    def __init__(self, dim, momentum=0.01):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        # Running statistics (not trainable, but need to be on the right device)
        self.register_buffer('running_mean', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('running_cov', torch.eye(dim, dtype=torch.float32))
        self.register_buffer('running_cov_inv', torch.eye(dim, dtype=torch.float32))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        # Flag: have we collected enough samples to trust the statistics?
        self.register_buffer('warmed_up', torch.tensor(False, dtype=torch.bool))
        self.warmup_threshold = 50  # batches before we trust the statistics

    @torch.no_grad()
    def update(self, z):
        """
        Update running statistics with a batch of projected hidden states.

        Args:
            z: [B, seq_len, r] projected hidden states (float32)
        """
        z = z.float().detach()
        # Flatten to [N, r] where N = B * seq_len
        z_flat = z.reshape(-1, self.dim)
        n = z_flat.shape[0]
        if n == 0:
            return

        batch_mean = z_flat.mean(dim=0)
        # Centered data for covariance
        z_centered = z_flat - batch_mean
        batch_cov = (z_centered.T @ z_centered) / max(n - 1, 1)

        # Exponential moving average
        alpha = self.momentum
        self.running_mean.mul_(1 - alpha).add_(batch_mean, alpha=alpha)
        self.running_cov.mul_(1 - alpha).add_(batch_cov, alpha=alpha)

        self.count.add_(1)
        if self.count.item() >= self.warmup_threshold:
            self.warmed_up.fill_(True)
            # Update inverse (regularized for stability)
            reg = 1e-4 * torch.eye(self.dim, device=z.device, dtype=torch.float32)
            self.running_cov_inv = torch.linalg.inv(self.running_cov + reg)

    def mahalanobis_sq(self, z):
        """
        Compute squared Mahalanobis distance of z from the running distribution.

        Args:
            z: [B, seq_len, r] projected hidden states

        Returns:
            dist_sq: [B, seq_len] squared Mahalanobis distance per token
        """
        z = z.float()
        # Center
        delta = z - self.running_mean.unsqueeze(0).unsqueeze(0)  # [B, S, r]
        # Mahalanobis: delta @ Sigma^{-1} @ delta^T (per token)
        # [B, S, r] @ [r, r] -> [B, S, r], then sum over r
        transformed = delta @ self.running_cov_inv  # [B, S, r]
        dist_sq = (transformed * delta).sum(dim=-1)  # [B, S]
        return dist_sq


# =============================================================================
# The ReFT Junction Adapter
# =============================================================================

class ReFTJunctionAdapter(nn.Module):
    """
    Low-Rank Orthogonal Gated Junction Adapter.

    Components:
      1. Subspace projection R: [d, r] — identifies which r dimensions to fix
      2. Orthogonal rotation Q: Cayley(A) where A is [r, r] skew-symmetric
      3. Subspace bias b: [r] — translational correction in the subspace
      4. Input-conditional gate: uses Mahalanobis distance features

    The adapter is identity when:
      - A = 0 (no rotation)
      - b = 0 (no shift)
      - gate = 0 (no correction applied)

    Initialization ensures all three conditions hold approximately, so the
    adapter starts as identity and only learns corrections where needed.

    Parameters:
      hidden_dim: model hidden dimension (3584 for Qwen2-7B)
      rank: dimension of the correction subspace (default 32)
      gate_hidden: hidden dim of the gating network (default 64)
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, rank=32, gate_hidden=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # --- Subspace Projection R ---
        # Initialize with random orthogonal columns (Gram-Schmidt)
        R_init = torch.randn(hidden_dim, rank, dtype=torch.float32)
        R_init, _ = torch.linalg.qr(R_init)  # orthonormal columns
        self.R = nn.Parameter(R_init)

        # --- Skew-symmetric generator A for Cayley map ---
        # A = 0 -> Q = I (identity rotation, adapter = identity)
        # Initialize near zero with small noise for symmetry breaking
        self.A = nn.Parameter(torch.zeros(rank, rank, dtype=torch.float32))
        # Small random init to break symmetry
        with torch.no_grad():
            self.A.add_(torch.randn_like(self.A) * 0.001)

        # --- Subspace bias ---
        # Starts at zero (no shift)
        self.b = nn.Parameter(torch.zeros(rank, dtype=torch.float32))

        # --- Input-conditional gate ---
        # Gate network: takes Mahalanobis distance features and outputs gate in [0, 1]
        # Features: [mahal_dist_mean, mahal_dist_max, norm_ratio, r projected features]
        gate_input_dim = 3 + rank  # mahal stats + projected mean
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(gate_hidden, 1, dtype=torch.float32),
        )
        # Initialize gate output bias negative so sigmoid(output) ~ 0
        # This makes the gate start near-closed (adapter = identity)
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(-3.0)  # sigmoid(-3) = 0.047
            self.gate_net[-1].weight.mul_(0.01)

        # --- Running stats for Mahalanobis distance ---
        self.stats = RunningStats(dim=rank, momentum=0.01)

    def _compute_gate_features(self, x, z):
        """
        Compute features for the gating network.

        Args:
            x: [B, S, d] input hidden state (float32)
            z: [B, S, r] projected hidden state (float32)

        Returns:
            features: [B, 1, gate_input_dim] features for gating (averaged over S)
        """
        # Mahalanobis distance features (how OOD is this input?)
        if self.stats.warmed_up:
            mahal_sq = self.stats.mahalanobis_sq(z)  # [B, S]
            mahal_mean = mahal_sq.mean(dim=-1, keepdim=True)  # [B, 1]
            mahal_max = mahal_sq.max(dim=-1, keepdim=True).values  # [B, 1]
        else:
            # Before warmup, use simple norm-based proxy
            mahal_mean = z.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True)
            mahal_max = z.pow(2).sum(dim=-1).max(dim=-1, keepdim=True).values

        # Norm ratio: ||z|| / expected (proxy for distribution shift)
        z_norm = z.pow(2).sum(dim=-1).mean(dim=-1, keepdim=True).sqrt()  # [B, 1]
        expected_norm = math.sqrt(self.rank)  # expected norm of standard normal in r dims
        norm_ratio = z_norm / (expected_norm + 1e-6)  # [B, 1]

        # Mean projected features (captures systematic shifts)
        z_mean = z.mean(dim=1)  # [B, r]

        # Concatenate: [B, 3 + r]
        features = torch.cat([
            mahal_mean,   # [B, 1]
            mahal_max,    # [B, 1]
            norm_ratio,   # [B, 1]
            z_mean,       # [B, r]
        ], dim=-1)

        return features.unsqueeze(1)  # [B, 1, 3+r] for broadcasting over seq

    def forward(self, x):
        """
        Forward pass of the ReFT junction adapter.

        Args:
            x: [B, S, d] hidden state (bfloat16 from model)

        Returns:
            output: [B, S, d] corrected hidden state (bfloat16)
        """
        input_dtype = x.dtype

        # Upcast to float32 for all adapter math
        x_f32 = x.float()

        # 1. Project to subspace
        # R is [d, r], columns are approximately orthonormal
        z = x_f32 @ self.R  # [B, S, r]

        # 2. Compute orthogonal rotation via Cayley map
        A_skew = (self.A - self.A.T) / 2.0  # enforce skew-symmetry
        Q = cayley_map(A_skew)  # [r, r], orthogonal

        # 3. Apply rotation and bias in subspace
        z_corrected = z @ Q + self.b  # [B, S, r]

        # 4. Compute correction in full space
        # delta = (z_corrected - z) @ R^T lifts the subspace correction
        # back to the full d-dimensional space
        delta_sub = z_corrected - z  # [B, S, r]
        delta = delta_sub @ self.R.T  # [B, S, d]

        # 5. Compute input-conditional gate
        gate_features = self._compute_gate_features(x_f32, z)  # [B, 1, gate_input_dim]
        gate = torch.sigmoid(self.gate_net(gate_features))  # [B, 1, 1]

        # 6. Apply gated correction
        output = x_f32 + gate * delta  # [B, S, d]

        # 7. Update running statistics (only during training)
        if self.training:
            self.stats.update(z)

        # Cast back to model dtype
        return output.to(input_dtype)

    def get_gate_value(self):
        """Return the last computed gate value for monitoring."""
        # This is approximate; for exact value, store during forward
        return None

    def rotation_magnitude(self):
        """How far Q is from identity. 0 = no rotation."""
        with torch.no_grad():
            A_skew = (self.A - self.A.T) / 2.0
            return A_skew.norm().item()

    def bias_magnitude(self):
        """Magnitude of the subspace bias."""
        with torch.no_grad():
            return self.b.norm().item()

    def param_count(self):
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self):
        return (
            f"hidden_dim={self.hidden_dim}, rank={self.rank}, "
            f"params={self.param_count():,}"
        )


# =============================================================================
# Loss: Mahalanobis Distribution Matching
# =============================================================================

class MahalanobisDistributionLoss(nn.Module):
    """
    Loss that minimizes the Mahalanobis distance of the adapter output
    w.r.t. the expected hidden-state distribution at the target layer.

    Key difference from KL loss:
      - KL targets the model's OUTPUT distribution (logits) -> fights improvement
      - This targets the HIDDEN STATE distribution -> just asks "does this look
        normal to downstream layers?"

    The loss is:
      L = E_tokens[ max(0, mahal_sq(h_adapted) - chi2_threshold) ]

    The chi2_threshold is the expected squared Mahalanobis distance for
    in-distribution data in r dimensions (= r, since chi-squared(r) has mean r).
    This creates a "dead zone" — if the adapted hidden state is within the
    expected distribution, the loss is ZERO. The adapter is only penalized
    for being OOD.

    Additionally, we add a soft orthogonality penalty on R to keep the
    projection well-conditioned.
    """
    def __init__(self, rank, orth_weight=0.01):
        super().__init__()
        self.rank = rank
        self.orth_weight = orth_weight
        # chi2(r) has mean r, std sqrt(2r)
        # Use mean + 1*std as threshold: generous dead zone
        self.chi2_threshold = rank + math.sqrt(2 * rank)

    def forward(self, adapter, h_adapted, h_pre_adapter):
        """
        Compute the loss.

        Args:
            adapter: ReFTJunctionAdapter instance (for accessing stats and R)
            h_adapted: [B, S, d] adapter output (bfloat16)
            h_pre_adapter: [B, S, d] adapter input (bfloat16)

        Returns:
            loss: scalar tensor with gradients
            loss_dict: dict of component losses for logging
        """
        h = h_adapted.float()

        # 1. Project adapted output to subspace
        z_adapted = h @ adapter.R  # [B, S, r]

        # 2. Mahalanobis distance in the subspace
        if adapter.stats.warmed_up:
            mahal_sq = adapter.stats.mahalanobis_sq(z_adapted)  # [B, S]
        else:
            # Before warmup, use L2 distance from origin as proxy
            # (stats haven't converged yet, so just use rough approximation)
            mahal_sq = z_adapted.pow(2).sum(dim=-1)  # [B, S]

        # 3. Hinge-style: only penalize beyond the dead zone
        # This is crucial: if the junction is already smooth, loss = 0,
        # so the adapter has no incentive to change anything.
        excess = F.relu(mahal_sq - self.chi2_threshold)  # [B, S]
        mahal_loss = excess.mean()

        # 4. Soft orthogonality penalty on R
        # R^T R should be close to I_r
        # This keeps the projection well-conditioned so Mahalanobis distance
        # is meaningful in the subspace
        RtR = adapter.R.T @ adapter.R  # [r, r]
        I_r = torch.eye(adapter.rank, device=RtR.device, dtype=torch.float32)
        orth_loss = (RtR - I_r).pow(2).mean()

        # 5. Regularize gate to prefer staying closed
        # Small penalty on gate output to encourage identity behavior
        # (This is implicit from the bias init, but a gentle push helps)
        # We don't add this directly — the gate's negative bias init handles it.

        # 6. Total loss
        total = mahal_loss + self.orth_weight * orth_loss

        loss_dict = {
            'mahal': mahal_loss.item(),
            'orth': orth_loss.item(),
            'total': total.item(),
            'mean_mahal_sq': mahal_sq.mean().item(),
            'max_mahal_sq': mahal_sq.max().item(),
            'threshold': self.chi2_threshold,
        }

        return total, loss_dict


# =============================================================================
# Adapter Layer Wrapper (for injection into model)
# =============================================================================

class AdapterWrappedLayer(nn.Module):
    """
    Wraps a transformer layer to apply the ReFT adapter AFTER its forward pass.
    Handles tuple outputs from Qwen2 layers.
    """
    def __init__(self, original_layer, adapter):
        super().__init__()
        self.layer = original_layer
        self.adapter = adapter

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
# Calibration: Collect reference distribution from the unduplicated model
# =============================================================================

def collect_reference_stats(model, tokenizer, adapter, target_layer_idx,
                            prompts, device, num_batches=50):
    """
    Collect reference hidden-state statistics at the target layer
    from the UNDUPLICATED model. These statistics define "what does
    normal input look like for this layer?"

    We project through the adapter's R matrix and update the RunningStats.

    Args:
        model: unduplicated model
        tokenizer: tokenizer
        adapter: ReFTJunctionAdapter (to use its R projection and stats)
        target_layer_idx: which layer to collect stats at
        prompts: calibration prompts
        device: torch device
        num_batches: number of collection passes
    """
    inner = model.model
    all_hidden = []

    def hook_fn(module, input, output):
        h = input[0] if isinstance(input, tuple) else input
        # We want the INPUT to this layer (what it normally receives)
        all_hidden.append(h.detach())

    hook = inner.layers[target_layer_idx].register_forward_hook(
        lambda m, inp, out: all_hidden.append(
            (inp[0] if isinstance(inp, tuple) else inp).detach()
        )
    )

    # Run prompts through the unduplicated model
    adapter.eval()
    with torch.no_grad():
        for batch_idx in range(min(num_batches, len(prompts))):
            prompt = prompts[batch_idx % len(prompts)]
            inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=64)
            inp = {k: v.to(device) for k, v in inp.items()}
            model(**inp, use_cache=False)

    hook.remove()

    # Update adapter's running stats with reference distribution
    adapter.train()
    for h in all_hidden:
        z = h.float() @ adapter.R  # project to subspace
        adapter.stats.update(z)

    print(f"  Calibrated running stats from {len(all_hidden)} forward passes")
    print(f"  Stats warmed up: {adapter.stats.warmed_up.item()}")
    if adapter.stats.warmed_up:
        print(f"  Running mean norm: {adapter.stats.running_mean.norm().item():.4f}")
        cov_diag = adapter.stats.running_cov.diag()
        print(f"  Running cov diag: min={cov_diag.min().item():.4f}, "
              f"max={cov_diag.max().item():.4f}, "
              f"mean={cov_diag.mean().item():.4f}")


# =============================================================================
# Training Prompts
# =============================================================================

TRAINING_PROMPTS = [
    # Math-heavy (target domain for math probe)
    "What is 78313 multiplied by 88537?",
    "The cube root of 74088 is approximately",
    "What is 9999 multiplied by 9999?",
    "The square root of 152399025 is",
    "What is 123456789 multiplied by 987654321?",
    "What is 31415 divided by 271?",
    "What is 2 to the power of 17?",
    "What is 456789 squared?",
    # General knowledge (distribution diversity)
    "The theory of general relativity states that",
    "In Python, a decorator is a function that",
    "To solve a quadratic equation, you can use",
    "Machine learning models are trained by",
    "The derivative of sin(x) is",
    "A linked list is a data structure where",
    "The speed of light in a vacuum is approximately",
    # Reasoning
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines?",
    # Instruction following
    "List five fruits in alphabetical order, separated by semicolons.",
    "Write exactly three sentences about the moon.",
    # Additional diversity
    "The Pythagorean theorem states that",
    "To implement quicksort, you first choose a pivot",
    "The integral of e^x dx equals",
    "The Fibonacci sequence starts with 0, 1, and then",
    "Gradient descent works by computing the derivative",
]


# =============================================================================
# Main Training Procedure
# =============================================================================

def train_reft_adapter(
    model_path=MODEL_PATH,
    i=10,
    j=11,
    rank=32,
    gate_hidden=64,
    num_steps=300,
    lr=3e-4,
    orth_weight=0.01,
    calibration_batches=100,
    tag=None,
):
    """
    Train a ReFT junction adapter using Mahalanobis distribution loss.

    The procedure:
      1. Load model, run baseline evaluation
      2. Calibrate: collect reference hidden-state distribution from unduplicated model
      3. Build duplicated model
      4. Train adapter with Mahalanobis + orthogonality loss
      5. Evaluate with adapter

    Args:
        model_path: path to HuggingFace model
        i, j: duplication config — duplicate layers [i, j)
        rank: subspace dimension for ReFT correction
        gate_hidden: hidden dim of gating network
        num_steps: training iterations
        lr: learning rate
        orth_weight: weight for R orthogonality penalty
        calibration_batches: number of passes for stats calibration
        tag: experiment identifier
    """
    if tag is None:
        tag = f"reft_{i}_{j}_r{rank}"

    print(f"\n{'='*70}")
    print(f"ReFT JUNCTION ADAPTER: config ({i},{j}), rank={rank}")
    print(f"{'='*70}")
    print(f"  Model:        {model_path}")
    print(f"  Rank:         {rank}")
    print(f"  Gate hidden:  {gate_hidden}")
    print(f"  LR:           {lr}")
    print(f"  Orth weight:  {orth_weight}")
    print(f"  Steps:        {num_steps}")

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

    print(f"  Layers:       {N}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Dup count:    {dup_count}")

    # =========================================================================
    # Step 2: Baseline evaluation (no duplication)
    # =========================================================================
    print("\n--- Step 1: Baseline (no duplication) ---")
    def gen_fn(prompt):
        return generate_no_cache(model, tokenizer, prompt, max_new_tokens=64)
    baseline = run_math_probe(gen_fn, verbose=False)
    print(f"  Baseline score: {baseline['score']:.4f}")

    # =========================================================================
    # Step 3: Create adapter and calibrate reference distribution
    # =========================================================================
    # The adapter needs to know what "normal" input looks like for the layer
    # right after the junction. In the unduplicated model, this is layer j.
    # (In the duplicated model, this becomes layer j + dup_count.)

    print(f"\n--- Step 2: Calibrating reference distribution at layer {j} ---")
    adapter = ReFTJunctionAdapter(
        hidden_dim=hidden_dim, rank=rank, gate_hidden=gate_hidden
    ).to(device)

    print(f"  Adapter params: {adapter.param_count():,}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Adapter fraction: {100 * adapter.param_count() / total_params:.4f}%")

    # Calibrate with unduplicated model
    collect_reference_stats(
        model, tokenizer, adapter, j,
        TRAINING_PROMPTS, device,
        num_batches=calibration_batches,
    )

    # =========================================================================
    # Step 4: Build duplicated model
    # =========================================================================
    print(f"\n--- Step 3: Building duplicated model ({i},{j}) ---")
    new_layers = list(original_layers[:j])
    for idx in range(i, j):
        new_layers.append(copy.deepcopy(original_layers[idx]))
    new_layers.extend(original_layers[j:])
    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    new_N = len(new_layers)

    junction_exit = j + dup_count - 1   # last layer of duplicated block
    target_layer_idx = j + dup_count     # first layer after junction

    print(f"  New layer count:    {new_N}")
    print(f"  Junction exit:      {junction_exit}")
    print(f"  Target layer:       {target_layer_idx}")

    # =========================================================================
    # Step 5: Pre-adapter evaluation
    # =========================================================================
    print("\n--- Step 4: Pre-adapter score (dup, no adapter) ---")
    pre_result = run_math_probe(gen_fn, verbose=False)
    pre_score = pre_result['score']
    pre_delta = pre_score - baseline['score']
    is_good_config = pre_delta > 0
    print(f"  Pre-adapter score: {pre_score:.4f} (delta: {pre_delta:+.4f})")
    print(f"  Config type: {'GOOD' if is_good_config else 'BAD'}")

    # =========================================================================
    # Step 6: Prepare training
    # =========================================================================
    print(f"\n--- Step 5: Setting up training ---")

    # Freeze ALL model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Only adapter parameters are trainable
    adapter.train()
    for param in adapter.parameters():
        param.requires_grad = True
    # RunningStats buffers should NOT have gradients
    for buf_name, buf in adapter.stats.named_buffers():
        if isinstance(buf, torch.Tensor):
            buf.requires_grad_(False)

    # Loss function
    loss_fn = MahalanobisDistributionLoss(rank=rank, orth_weight=orth_weight)

    # Optimizer: different LR for different components
    optimizer = torch.optim.AdamW([
        {'params': [adapter.R], 'lr': lr * 0.3},           # Subspace moves slowly
        {'params': [adapter.A], 'lr': lr},                  # Rotation moves normally
        {'params': [adapter.b], 'lr': lr},                  # Bias moves normally
        {'params': adapter.gate_net.parameters(), 'lr': lr * 0.5},  # Gate is careful
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr * 0.3, lr, lr, lr * 0.5],
        total_steps=num_steps,
        pct_start=0.15,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )

    # Tokenize training prompts
    tokenized_prompts = []
    for p in TRAINING_PROMPTS:
        inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
        tokenized_prompts.append(inp['input_ids'].to(device))

    # =========================================================================
    # Step 7: Training loop
    # =========================================================================
    print(f"\n--- Step 6: Training ({num_steps} steps) ---")

    history = {
        'mahal': [], 'orth': [], 'total': [],
        'gate_mean': [], 'rotation_mag': [], 'bias_mag': [],
    }

    for step in range(num_steps):
        optimizer.zero_grad()

        step_loss = 0.0
        step_mahal = 0.0
        step_orth = 0.0
        step_gate = 0.0
        n_prompts = 0

        for input_ids in tokenized_prompts:
            # --- Forward pass up to the junction ---
            with torch.no_grad():
                # Embed
                h = inner.embed_tokens(input_ids)
                seq_len = h.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_embeds = inner.rotary_emb(h, position_ids)

                # Run all layers up to and including junction_exit
                h = run_layer_range(inner, h, 0, junction_exit + 1, pos_embeds)

            # h is the junction output (bfloat16, no grad)
            h_pre = h.detach()

            # --- Apply adapter (float32 internally) ---
            h_post = adapter(h_pre)

            # --- Compute loss ---
            loss, loss_dict = loss_fn(adapter, h_post, h_pre)

            # --- Backward ---
            loss.backward()

            step_loss += loss_dict['total']
            step_mahal += loss_dict['mahal']
            step_orth += loss_dict['orth']
            n_prompts += 1

            # Track gate value
            with torch.no_grad():
                z_check = h_pre.float() @ adapter.R
                gate_feats = adapter._compute_gate_features(h_pre.float(), z_check)
                gate_val = torch.sigmoid(adapter.gate_net(gate_feats)).mean().item()
                step_gate += gate_val

            del h, h_pre, h_post, loss

        # Average and step
        avg_loss = step_loss / n_prompts
        avg_mahal = step_mahal / n_prompts
        avg_orth = step_orth / n_prompts
        avg_gate = step_gate / n_prompts

        history['total'].append(avg_loss)
        history['mahal'].append(avg_mahal)
        history['orth'].append(avg_orth)
        history['gate_mean'].append(avg_gate)
        history['rotation_mag'].append(adapter.rotation_magnitude())
        history['bias_mag'].append(adapter.bias_magnitude())

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if (step + 1) % 25 == 0 or step == 0:
            current_lr = scheduler.get_last_lr()[1]  # main LR group
            print(
                f"    Step {step+1:4d}/{num_steps}  "
                f"loss={avg_loss:.6f}  "
                f"mahal={avg_mahal:.6f}  "
                f"orth={avg_orth:.6f}  "
                f"gate={avg_gate:.4f}  "
                f"rot={adapter.rotation_magnitude():.4f}  "
                f"bias={adapter.bias_magnitude():.4f}  "
                f"lr={current_lr:.2e}"
            )

        # Periodic cleanup
        if (step + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # =========================================================================
    # Step 8: Post-adapter evaluation
    # =========================================================================
    print("\n--- Step 7: Post-adapter evaluation ---")
    model.eval()
    adapter.eval()

    # Wrap the junction exit layer with the trained adapter
    inner.layers[junction_exit] = AdapterWrappedLayer(
        inner.layers[junction_exit], adapter
    )

    post_result = run_math_probe(gen_fn, verbose=True)
    post_score = post_result['score']
    post_delta = post_score - baseline['score']
    adapter_gain = post_score - pre_score

    # =========================================================================
    # Step 9: Results
    # =========================================================================
    print(f"\n  {'='*60}")
    print(f"  ReFT JUNCTION ADAPTER RESULTS: {tag}")
    print(f"  {'='*60}")
    print(f"  Baseline (no dup):     {baseline['score']:.4f}")
    print(f"  Pre-adapter (dup):     {pre_score:.4f} ({pre_delta:+.4f})")
    print(f"  Post-adapter (dup+ad): {post_score:.4f} ({post_delta:+.4f})")
    print(f"  Adapter gain:          {adapter_gain:+.4f}")
    print(f"  Config type:           {'GOOD' if is_good_config else 'BAD'}")
    print(f"  Final gate mean:       {history['gate_mean'][-1]:.4f}")
    print(f"  Final rotation mag:    {history['rotation_mag'][-1]:.4f}")
    print(f"  Final bias mag:        {history['bias_mag'][-1]:.4f}")
    print(f"  Adapter params:        {adapter.param_count():,}")

    if is_good_config and abs(pre_delta) > 1e-6:
        preserved = post_delta / pre_delta * 100
        print(f"  Improvement preserved: {preserved:.1f}%")
    elif not is_good_config and abs(pre_delta) > 1e-6:
        recovery = adapter_gain / abs(pre_delta) * 100
        print(f"  Quality recovery:      {recovery:.1f}%")

    # =========================================================================
    # Step 10: Save
    # =========================================================================
    result = {
        "tag": tag,
        "config": [i, j],
        "model": model_path,
        "rank": rank,
        "gate_hidden": gate_hidden,
        "baseline": baseline['score'],
        "pre_adapter": pre_score,
        "pre_delta": pre_delta,
        "post_adapter": post_score,
        "post_delta": post_delta,
        "adapter_gain": adapter_gain,
        "is_good_config": is_good_config,
        "adapter_params": adapter.param_count(),
        "adapter_pct": 100 * adapter.param_count() / total_params,
        "steps": num_steps,
        "lr": lr,
        "final_gate_mean": history['gate_mean'][-1],
        "final_rotation_mag": history['rotation_mag'][-1],
        "final_bias_mag": history['bias_mag'][-1],
        "history": history,
        "baseline_details": baseline['scores'],
        "pre_details": pre_result['scores'],
        "post_details": post_result['scores'],
    }

    # Save adapter weights
    adapter_save = {
        'adapter_state_dict': adapter.state_dict(),
        'config': {
            'i': i, 'j': j,
            'hidden_dim': hidden_dim,
            'rank': rank,
            'gate_hidden': gate_hidden,
            'junction_exit': junction_exit,
            'target_layer': target_layer_idx,
        },
    }
    weights_path = RESULTS_DIR / f"adapter_weights_{tag}.pt"
    torch.save(adapter_save, weights_path)
    print(f"\n  Adapter weights saved to {weights_path}")

    results_path = RESULTS_DIR / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# =============================================================================
# Sweep: Test across good and bad configs
# =============================================================================

def run_sweep(model_path=MODEL_PATH, rank=32, num_steps=300):
    """
    Run the ReFT adapter on both good and bad duplication configs.

    Good configs: should PRESERVE the improvement from duplication.
    Bad configs: should RECOVER quality lost from duplication.
    """
    configs = [
        # Good configs (duplication helps)
        (10, 11, "good_10_11"),
        (18, 21, "good_18_21"),
        # Bad configs (duplication hurts)
        (4, 9, "bad_4_9"),
        (15, 18, "bad_15_18"),
    ]

    results = []
    for ci, cj, label in configs:
        print(f"\n{'#'*70}")
        print(f"# Config ({ci},{cj}) — {label}")
        print(f"{'#'*70}")

        result = train_reft_adapter(
            model_path=model_path,
            i=ci, j=cj,
            rank=rank,
            num_steps=num_steps,
            tag=f"reft_{label}_r{rank}",
        )
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("ReFT JUNCTION ADAPTER — SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Tag':>25} {'Type':>5} {'Base':>7} {'Pre':>7} "
          f"{'Post':>7} {'Gain':>7} {'Gate':>6} {'Rot':>6}")
    print(f"  {'-'*75}")
    for r in results:
        t = "GOOD" if r['is_good_config'] else "BAD"
        print(
            f"  {r['tag']:>25} {t:>5} "
            f"{r['baseline']:7.4f} {r['pre_adapter']:7.4f} "
            f"{r['post_adapter']:7.4f} {r['adapter_gain']:+7.4f} "
            f"{r['final_gate_mean']:6.4f} {r['final_rotation_mag']:6.4f}"
        )

    # Save sweep results
    sweep_path = RESULTS_DIR / f"sweep_r{rank}.json"
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {sweep_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ReFT Junction Adapter: Low-rank orthogonal gated correction"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help="Path to HuggingFace model"
    )
    parser.add_argument("--i", type=int, default=10, help="Dup block start")
    parser.add_argument("--j", type=int, default=11, help="Dup block end")
    parser.add_argument("--rank", type=int, default=32, help="Subspace rank")
    parser.add_argument("--gate-hidden", type=int, default=64, help="Gate hidden dim")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--orth-weight", type=float, default=0.01,
                        help="Orthogonality regularization weight")
    parser.add_argument("--calibration-batches", type=int, default=100,
                        help="Calibration passes for reference stats")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep across good+bad configs")
    parser.add_argument("--tag", type=str, default=None, help="Experiment tag")

    args = parser.parse_args()

    t0 = time.time()

    if args.sweep:
        run_sweep(
            model_path=args.model,
            rank=args.rank,
            num_steps=args.steps,
        )
    else:
        train_reft_adapter(
            model_path=args.model,
            i=args.i,
            j=args.j,
            rank=args.rank,
            gate_hidden=args.gate_hidden,
            num_steps=args.steps,
            lr=args.lr,
            orth_weight=args.orth_weight,
            calibration_batches=args.calibration_batches,
            tag=args.tag,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
