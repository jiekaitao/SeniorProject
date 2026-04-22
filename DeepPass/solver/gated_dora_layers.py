"""Gated DoRA-style linear layers for hybrid controller.

Wraps selected ``nn.Linear`` modules in a frozen LLM with a DoRA-style
trainable delta whose contribution is gated per-batch by an external scalar
``alpha`` produced by the controller. When ``alpha == 0`` the module is a
pure passthrough of the frozen base linear; when ``alpha == 1`` the full
DoRA delta is applied.

DoRA (decoupled weight decomposition) decomposes the effective weight into
a magnitude and a direction:
    W' = m * (W0 + B @ A) / ||W0 + B @ A||_row
We use a memory-efficient reformulation that never materialises the full
weight matrix. Given per-row magnitudes ``mag`` and the frozen row norms
``||W0||``, the effective output is:
    y_eff = W0 x + (mag/||W0|| - 1) * (W0 x) + B (A x)
(approximation that keeps the direction of W0 and layers the low-rank
correction on top — this matches the standard DoRA approximation used in
other implementations like peft). Mag is initialised to ||W0|| so at
``alpha == 0`` the layer is *exactly* the frozen base linear.
"""
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F


# Thread-local storage isn't actually needed for single-GPU training (each
# forward pass runs serially inside the Python GIL), but we keep the alpha
# on the module instance so the LM's standard forward (which cannot accept
# extra args) still works.


class GatedDoRALinear(nn.Module):
    """A frozen base linear plus a DoRA-style trainable delta whose
    contribution is gated per-batch by an external scalar ``alpha``.

    The controller sets ``self.current_alpha`` before each model forward
    via :func:`set_gate_alphas`. When ``current_alpha`` is ``None`` the
    module is a pure passthrough of the base linear (useful for the
    baseline forward that computes the unmodified logits).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base_linear
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        out_dim, in_dim = base_linear.weight.shape
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        # Low-rank trainable params (B @ A applied to x).
        # A: (rank, in_dim), B: (out_dim, rank). Init B = 0 so at t=0 delta = 0
        # beyond magnitude rescaling.
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

        # Magnitude per output row, initialised to base row norms so that
        # mag/||W|| == 1 and the scaling term vanishes at t=0.
        with torch.no_grad():
            row_norms = base_linear.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
            # Cache frozen row norms as a buffer so we don't recompute every forward
            # (and so we don't accidentally track the base weight in the graph).
            self.register_buffer('base_row_norms', row_norms.detach().clone())
        self.mag = nn.Parameter(row_norms.detach().clone())

        # Current alpha is set externally by the controller before each forward.
        # None means "skip DoRA delta entirely and act as pure base linear".
        self.current_alpha = None

    def forward(self, x):
        """Standard forward that threads `self.current_alpha` through.

        ``x`` may be ``(*, in_dim)`` (sequence-style Llama linear inputs are
        ``(B, T, in_dim)``).
        """
        y_base = F.linear(x, self.base.weight, self.base.bias)
        alpha = self.current_alpha
        if alpha is None:
            return y_base

        # Bring alpha into the right dtype (module may hold fp32 alpha but
        # y_base is bf16).
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(y_base.dtype)
        else:
            alpha = torch.as_tensor(alpha, device=y_base.device, dtype=y_base.dtype)

        # Magnitude rescaling: (mag/||W0|| - 1) * (W0 x).
        # base_row_norms is (out_dim, 1); squeeze to (out_dim,) for broadcast.
        scale = (self.mag / self.base_row_norms).to(y_base.dtype).squeeze(-1) - 1.0
        y_scale = scale * y_base  # broadcast over leading dims

        # Low-rank correction: B @ A x.
        # F.linear(x, A) -> (*, rank); F.linear(., B) -> (*, out_dim).
        y_lora = F.linear(F.linear(x, self.A.to(y_base.dtype)),
                          self.B.to(y_base.dtype))

        delta_y = y_scale + y_lora

        # Broadcast alpha over non-batch dims if it's per-batch.
        if isinstance(alpha, torch.Tensor) and alpha.dim() > 0:
            # delta_y: (B, T, out_dim) or (B, out_dim). alpha: (B,)
            while alpha.dim() < delta_y.dim():
                alpha = alpha.unsqueeze(-1)
        return y_base + alpha * delta_y

    def trainable_param_count(self):
        return self.A.numel() + self.B.numel() + self.mag.numel()


def wrap_model_with_gated_dora(model,
                               target_modules=('q_proj', 'o_proj',
                                                'gate_proj', 'up_proj', 'down_proj'),
                               target_layers=None,
                               rank=16):
    """Wrap selected linear modules in ``model`` with :class:`GatedDoRALinear`.

    Args:
        model: A frozen HuggingFace causal LM (Llama-style supported).
        target_modules: Module names to wrap within each transformer block.
        target_layers: Tuple of layer indices. Defaults to the last half.
        rank: DoRA low-rank dimension.

    Returns:
        (model, sites) where ``sites`` is a list of tuples
        ``(layer_idx, module_name, GatedDoRALinear)`` in the order the
        controller's gate head will index them.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        layers = model.language_model.layers
    else:
        raise ValueError(
            'wrap_model_with_gated_dora: unsupported model structure '
            '(expected `model.model.layers` or `model.language_model.layers`).')

    n_layers = len(layers)
    if target_layers is None:
        target_layers = tuple(range(n_layers // 2, n_layers))

    sites = []
    for li in target_layers:
        if li < 0 or li >= n_layers:
            continue
        layer = layers[li]
        for name in target_modules:
            if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                parent = getattr(layer, 'self_attn', None)
            elif name in ('gate_proj', 'up_proj', 'down_proj'):
                parent = getattr(layer, 'mlp', None)
            else:
                parent = None
            if parent is None:
                continue
            if not hasattr(parent, name):
                continue
            base_lin = getattr(parent, name)
            if not isinstance(base_lin, nn.Linear):
                continue
            gated = GatedDoRALinear(base_lin, rank=rank)
            # Preserve dtype/device of base linear
            gated = gated.to(device=base_lin.weight.device,
                             dtype=base_lin.weight.dtype)
            setattr(parent, name, gated)
            sites.append((li, name, gated))
    return model, sites


def set_gate_alphas(sites, alpha_per_site):
    """Set per-batch gate alpha on each wrapped site.

    Args:
        sites: list of (layer_idx, module_name, GatedDoRALinear) tuples
            returned by :func:`wrap_model_with_gated_dora`.
        alpha_per_site: tensor of shape ``(B, n_sites)`` with values in
            ``[0, 1]``. Column ``i`` is the gate for ``sites[i]``.
    """
    assert alpha_per_site.dim() == 2, \
        f'alpha_per_site must be (B, n_sites), got {alpha_per_site.shape}'
    assert alpha_per_site.shape[1] == len(sites), \
        f'alpha_per_site has {alpha_per_site.shape[1]} columns but there are {len(sites)} sites'
    for i, (_li, _name, gated) in enumerate(sites):
        gated.current_alpha = alpha_per_site[:, i]


def clear_gate_alphas(sites):
    """Reset all gate alphas to ``None`` (pure passthrough)."""
    for _li, _name, g in sites:
        g.current_alpha = None


def count_dora_params(sites):
    """Total trainable DoRA parameters across all wrapped sites."""
    return sum(s[2].trainable_param_count() for s in sites)
