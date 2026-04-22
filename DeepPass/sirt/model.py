"""
SIRT-170M: Sublayer-Iterated Recursive Transformer

3-zone architecture:
  - Prelude: 3 standard dense blocks (early processing)
  - Recursive Core: 3 shared blocks, repeated 1-4 times adaptively
    - Attention always iterates
    - FFN contribution gated by token-wise β (margin + stability routing)
  - Coda: 4 standard dense blocks (factual tail, never recursive)

~170M parameters with d=1024, tied embeddings, GQA (16 heads, 4 KV heads).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SIRTConfig:
    vocab_size: int = 128256  # LLaMA 3 tokenizer
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    d_head: int = 64  # d_model // n_heads
    ffn_dim: int = 3072  # ~3x d_model for SwiGLU
    context_len: int = 4096
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # Architecture zones
    n_prelude: int = 3
    n_recursive_blocks: int = 3
    n_coda: int = 4
    max_recursions: int = 4

    # FFN gating
    beta_max: float = 0.25
    gate_margin_quantile: float = 0.1

    # Halting
    halt_mode: str = "sequence_act"  # "fixed", "sequence_act"
    halt_epsilon: float = 0.05

    # Training
    dropout: float = 0.0

    @property
    def n_unique_layers(self):
        return self.n_prelude + self.n_recursive_blocks + self.n_coda

    @property
    def total_params_approx(self):
        """Approximate parameter count."""
        d, dff, dkv = self.d_model, self.ffn_dim, self.d_head * self.n_kv_heads
        # Embedding (shared with LM head if tied)
        emb = self.vocab_size * d
        # Per standard block: attn (QKV + O) + FFN (gate + up + down) + 2 norms
        attn = d * d + d * dkv + d * dkv + d * d  # Q, K, V, O
        ffn = d * dff * 3  # gate, up, down (SwiGLU)
        norms = d * 2
        block = attn + ffn + norms
        # Recursive blocks also have beta router
        router = (d + 2) * (d // 4) + (d // 4) * 1 + 1  # tiny
        n_blocks = self.n_prelude + self.n_recursive_blocks + self.n_coda
        # Halt head
        halt = d + 1
        total = emb + n_blocks * block + self.n_recursive_blocks * router + halt
        return total


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class DepthScaledRMSNorm(RMSNorm):
    """RMSNorm with depth-aware scaling to prevent gradient issues in deep recursion."""
    def forward(self, x, virtual_depth: int = 1):
        scale = 1.0 / math.sqrt(max(virtual_depth, 1))
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight * scale


def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[:x.shape[-2], :d//2].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :d//2].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GQAAttention(nn.Module):
    """Grouped-Query Attention with RoPE."""
    def __init__(self, cfg: SIRTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.n_groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin, mask=None, kv_cache=None, cache_slot=None):
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # KV cache
        if kv_cache is not None and cache_slot is not None:
            if cache_slot in kv_cache:
                prev_k, prev_v = kv_cache[cache_slot]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            kv_cache[cache_slot] = (k, v)

        # Expand KV heads for GQA
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, -1, self.d_head)
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, -1, self.d_head)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and kv_cache is None))

        out = attn.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class DenseBlock(nn.Module):
    """Standard transformer block (prelude/coda layers)."""
    def __init__(self, cfg: SIRTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.wg = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wd = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, h, rope_cos, rope_sin, mask=None, kv_cache=None, cache_slot=None):
        h = h + self.attn(self.norm1(h), rope_cos, rope_sin, mask, kv_cache, cache_slot)
        z = self.norm2(h)
        h = h + self.wd(F.silu(self.wg(z)) * self.wu(z))
        return h


class RecursiveCoreBlock(nn.Module):
    """
    Recursive block with token-wise FFN gating.

    Attention always iterates. FFN contribution is gated by β ∈ [0, β_max],
    computed from gate margin (stability of FFN activation) and a tiny router.
    """
    def __init__(self, cfg: SIRTConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = DepthScaledRMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.norm2 = DepthScaledRMSNorm(cfg.d_model, cfg.norm_eps)
        self.wg = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wd = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

        # β router: takes hidden state + margin + stability → [0, β_max]
        router_in_dim = cfg.d_model + 2
        router_hidden = cfg.d_model // 4
        self.beta_router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(router_hidden, 1, bias=True),
        )
        self.block_bias = nn.Parameter(torch.tensor(-2.0))  # init biased toward low β

    def forward(self, h, rope_cos, rope_sin, prev_gate=None, virtual_depth=1,
                mask=None, kv_cache=None, cache_slot=None):
        # Attention (always applied)
        u = h + self.attn(self.norm1(h, virtual_depth), rope_cos, rope_sin,
                         mask, kv_cache, cache_slot)

        # FFN with gated β
        z = self.norm2(u, virtual_depth)
        g = self.wg(z)  # gate values [B, L, d_ff]
        v = self.wu(z)  # up values

        # Gate margin: approximate 10th percentile via sorted index
        g_abs = g.abs().float()
        k = max(1, int(g_abs.shape[-1] * self.cfg.gate_margin_quantile))
        margin = g_abs.kthvalue(k, dim=-1, keepdim=True).values.to(g.dtype)  # [B, L, 1]

        # Stability: how similar are gate activations to previous recursion?
        if prev_gate is not None:
            stability = 1.0 - (
                torch.sigmoid(2.0 * g) - torch.sigmoid(2.0 * prev_gate)
            ).abs().mean(dim=-1, keepdim=True)
        else:
            stability = torch.ones(g.shape[0], g.shape[1], 1,
                                   device=g.device, dtype=g.dtype)

        # Compute β ∈ [0, β_max]
        # Pool hidden state to avoid d_model-sized input per token
        z_pooled = z.mean(dim=-1, keepdim=True).expand_as(margin)  # [B, L, 1] — simplified
        router_in = torch.cat([z, margin, stability], dim=-1)  # [B, L, d+2]
        beta = self.cfg.beta_max * torch.sigmoid(
            self.block_bias + self.beta_router(router_in)
        )  # [B, L, 1]

        # FFN output scaled by β
        ffn_out = self.wd(F.silu(g) * v)
        h = u + beta * ffn_out

        return h, g.detach(), beta


class SIRTLM(nn.Module):
    """
    SIRT-170M: Sublayer-Iterated Recursive Transformer.

    Architecture:
        Embedding → Prelude (3 blocks) → Recursive Core (3 blocks × K iterations)
        → Coda (4 blocks) → LM Head

    The recursive core uses shared weights with adaptive halting and
    token-wise FFN gating informed by gate margin stability.
    """
    def __init__(self, cfg: SIRTConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Prelude: standard blocks
        self.prelude = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_prelude)])

        # Recursive core: shared blocks
        self.core = nn.ModuleList([RecursiveCoreBlock(cfg) for _ in range(cfg.n_recursive_blocks)])

        # Coda: standard blocks (factual tail)
        self.coda = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_coda)])

        # Halting mechanism
        self.halt_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.halt_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.constant_(self.halt_head.bias, 1.0)  # bias toward continuing

        # Output
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie embeddings
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE
        cos, sin = precompute_rope_freqs(cfg.d_head, cfg.context_len, cfg.rope_base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None, kv_cache=None,
                fixed_recursions=None, train_halting=False):
        """
        Args:
            input_ids: [B, L]
            labels: [B, L] for training (shifted internally)
            kv_cache: dict for inference caching
            fixed_recursions: int, override adaptive halting with fixed K
            train_halting: bool, enable ACT loss computation
        """
        B, L = input_ids.shape
        h = self.embed(input_ids)
        slot = 0

        # === Prelude ===
        for blk in self.prelude:
            h = blk(h, self.rope_cos, self.rope_sin, kv_cache=kv_cache, cache_slot=slot)
            slot += 1

        # === Recursive Core ===
        K = fixed_recursions or self.cfg.max_recursions
        states = []
        halt_probs = []
        prev_gates = [None] * len(self.core)
        all_betas = []

        for t in range(K):
            virtual_depth = self.cfg.n_prelude + t * len(self.core)
            for j, blk in enumerate(self.core):
                cache_slot = self.cfg.n_prelude + t * len(self.core) + j
                h, prev_gates[j], beta = blk(
                    h, self.rope_cos, self.rope_sin,
                    prev_gate=prev_gates[j],
                    virtual_depth=virtual_depth + j + 1,
                    kv_cache=kv_cache, cache_slot=cache_slot,
                )
                all_betas.append(beta)

            states.append(h)

            # Halting probability
            if train_halting or self.cfg.halt_mode == "sequence_act":
                pooled = self.halt_norm(h).mean(dim=1).float()  # [B, d] in float32
                p = torch.sigmoid(self.halt_head(pooled.to(self.halt_head.weight.dtype)))  # [B, 1]
                halt_probs.append(p)

                # Early stopping during inference
                if not self.training and fixed_recursions is None:
                    cumulative = sum(halt_probs)
                    if (cumulative > 1.0 - self.cfg.halt_epsilon).all():
                        break

        # ACT mixture or use last state
        if train_halting and len(halt_probs) > 0:
            # Compute geometric halt distribution
            remainders = []
            cum_halt = torch.zeros(B, 1, device=h.device)
            for t, p in enumerate(halt_probs):
                if t < len(halt_probs) - 1:
                    pi = p * (1.0 - cum_halt)
                    cum_halt = cum_halt + pi
                else:
                    pi = 1.0 - cum_halt  # last step gets remainder
                remainders.append(pi)

            # Weighted mixture of states
            h = torch.zeros_like(states[0])
            for state, pi in zip(states, remainders):
                h = h + pi.unsqueeze(-1) * state

            expected_steps = sum((t + 1) * pi.mean() for t, pi in enumerate(remainders))
            h = h.to(states[0].dtype)  # back to bf16 after float32 mixture
        else:
            expected_steps = torch.tensor(float(len(states)), device=h.device)

        # === Coda ===
        coda_start_slot = self.cfg.n_prelude + K * len(self.core)
        for i, blk in enumerate(self.coda):
            h = blk(h, self.rope_cos, self.rope_sin,
                    kv_cache=kv_cache, cache_slot=coda_start_slot + i)

        # === Output ===
        logits = self.lm_head(self.final_norm(h))

        # Compute losses
        loss = None
        aux = {}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Auxiliary losses
        avg_beta = torch.stack(all_betas).mean() if all_betas else torch.tensor(0.0)
        aux['expected_steps'] = expected_steps
        aux['beta_mean'] = avg_beta
        aux['n_recursions'] = len(states)

        return logits, loss, aux

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(cfg: Optional[SIRTConfig] = None) -> SIRTLM:
    """Create SIRT model with default or custom config."""
    if cfg is None:
        cfg = SIRTConfig()
    model = SIRTLM(cfg)
    print(f"SIRT-{model.count_parameters()/1e6:.0f}M created")
    print(f"  Config: d={cfg.d_model}, heads={cfg.n_heads}/{cfg.n_kv_heads}, "
          f"ffn={cfg.ffn_dim}, ctx={cfg.context_len}")
    print(f"  Zones: {cfg.n_prelude} prelude + {cfg.n_recursive_blocks} recursive "
          f"(max {cfg.max_recursions}x) + {cfg.n_coda} coda")
    print(f"  Effective depth: {cfg.n_prelude + cfg.n_recursive_blocks + cfg.n_coda} "
          f"to {cfg.n_prelude + cfg.n_recursive_blocks * cfg.max_recursions + cfg.n_coda} layers")
    print(f"  Parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    cfg = SIRTConfig()
    model = create_model(cfg)

    # Smoke test
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss, aux = model(x, labels=x, fixed_recursions=2)
    print(f"\nSmoke test:")
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Avg β: {aux['beta_mean'].item():.4f}")
    print(f"  Expected steps: {aux['expected_steps'].item():.2f}")
    print(f"  Recursions: {aux['n_recursions']}")
