"""
PSRT-MoR-lite: Mixture-of-Recursions with Split-State Design

Architecture per GPT-5.4 Pro recommendation:
- Shared attention across all experts (stable backbone)
- Expert-specific FFN branches with monotone step-decaying beta
- Sequence-level top-2 router
- Fixed memory channel (m_0), iterated reasoning channel (r)
- Geometric halting with acceleration-based exit

Three experts:
  Expert 1 (reason-refine):     beta = [0.25, 0.10, 0.02, 0.00]
  Expert 2 (math-single-refresh): beta = [0.80, 0.20, 0.05, 0.00]
  Expert 3 (safe-fluency):      beta = [0.10, 0.02, 0.00, 0.00]

Usage:
    python mor_lite.py  # smoke test
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoRLiteConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    d_head: int = 64
    ffn_dim: int = 3072
    context_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    n_prelude: int = 2
    n_core: int = 3
    n_coda: int = 5
    max_recursion: int = 4
    n_experts: int = 3
    top_k: int = 2

    # Expert FFN beta schedules (per step, monotone decreasing)
    expert_betas: List[List[float]] = field(default_factory=lambda: [
        [0.25, 0.10, 0.02, 0.00],  # reason-refine
        [0.80, 0.20, 0.05, 0.00],  # math-single-refresh
        [0.10, 0.02, 0.00, 0.00],  # safe-fluency
    ])

    halt_epsilon: float = 0.05
    accel_threshold: float = 0.1
    dropout: float = 0.0


# ============================================================
# Building Blocks (reused from PSRT)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope(dim, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    c = cos[:x.shape[-2], :d // 2].unsqueeze(0).unsqueeze(0)
    s = sin[:x.shape[-2], :d // 2].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class SharedAttention(nn.Module):
    """Shared GQA attention — same across all experts."""
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv, self.d_head).transpose(1, 2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        if self.groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(B, self.n_heads, -1, self.d_head)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(B, self.n_heads, -1, self.d_head)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


class ExpertFFN(nn.Module):
    """Expert-specific SwiGLU FFN."""
    def __init__(self, cfg):
        super().__init__()
        self.wg = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wd = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.wd(F.silu(self.wg(x)) * self.wu(x))


class SharedCoreBlock(nn.Module):
    """One core block: shared attention + expert-specific FFN branches."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SharedAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.expert_ffns = nn.ModuleList([ExpertFFN(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x, rope_cos, rope_sin, expert_weights, beta):
        """
        Args:
            x: (B, L, D) input
            expert_weights: (B, n_experts) router weights
            beta: (n_experts,) FFN scale for this step
        """
        # Shared attention
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)

        # Expert-weighted FFN
        z = self.norm2(x)
        ffn_out = torch.zeros_like(x)
        for e, ffn in enumerate(self.expert_ffns):
            w = expert_weights[:, e].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            ffn_out = ffn_out + w * beta[e] * ffn(z)

        x = x + ffn_out
        return x


class SequenceRouter(nn.Module):
    """Sequence-level top-k router."""
    def __init__(self, cfg):
        super().__init__()
        # Router input: pooled reasoning state + scalar features
        router_dim = cfg.d_model + 3  # +step_frac, +hidden_norm, +confidence
        self.net = nn.Sequential(
            nn.Linear(router_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cfg.n_experts),
        )
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts

    def forward(self, r_pooled, step_frac, hidden_norm, confidence):
        """
        Args:
            r_pooled: (B, D) pooled reasoning state
            step_frac: scalar, current_step / max_steps
            hidden_norm: (B,) L2 norm of reasoning state
            confidence: (B,) max softmax probability from last logits
        Returns:
            weights: (B, n_experts) sparse top-k weights
            raw_logits: (B, n_experts) for aux losses
        """
        features = torch.cat([
            r_pooled,
            torch.full((r_pooled.shape[0], 1), step_frac, device=r_pooled.device),
            hidden_norm.unsqueeze(-1),
            confidence.unsqueeze(-1),
        ], dim=-1)

        logits = self.net(features)

        # Top-k selection
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Sparse weight tensor
        weights = torch.zeros_like(logits)
        weights.scatter_(1, topk_idx, topk_weights)

        return weights, logits


class TransformerBlock(nn.Module):
    """Standard block for prelude/coda."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SharedAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.wg = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wd = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, h, rope_cos, rope_sin):
        h = h + self.attn(self.norm1(h), rope_cos, rope_sin)
        z = self.norm2(h)
        h = h + self.wd(F.silu(self.wg(z)) * self.wu(z))
        return h


# ============================================================
# PSRT-MoR-lite Model
# ============================================================

class PSRTMoRLite(nn.Module):
    def __init__(self, cfg: MoRLiteConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Prelude
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_prelude)])

        # Memory/Reasoning projections
        self.proj_m = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_r = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Recursive core (shared attention + expert FFNs)
        self.core = nn.ModuleList([SharedCoreBlock(cfg) for _ in range(cfg.n_core)])

        # Mixing parameter
        self.alpha_logit = nn.Parameter(torch.zeros(1))

        # Router
        self.router = SequenceRouter(cfg)

        # Halting
        self.halt_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.halt_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.constant_(self.halt_head.bias, 1.0)

        # Combine
        self.combine = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)

        # Coda
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_coda)])

        # Output
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE
        cos, sin = precompute_rope(cfg.d_head, cfg.context_len, cfg.rope_base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        # Expert beta schedules as buffer
        betas = torch.tensor(cfg.expert_betas)  # (n_experts, max_recursion)
        self.register_buffer('expert_betas', betas, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None, fixed_K=None, train_halting=False,
                uniform_routing=False):
        B, L = input_ids.shape
        h = self.embed(input_ids)

        # Prelude
        for blk in self.prelude:
            h = blk(h, self.rope_cos, self.rope_sin)

        # Split
        m_0 = self.proj_m(h)
        r = self.proj_r(h)
        alpha = torch.sigmoid(self.alpha_logit)

        K = fixed_K or self.cfg.max_recursion
        r_states = [r]
        halt_probs = []
        route_logits_all = []
        prev_delta_norm = None

        for t in range(K):
            # Router input
            r_pooled = r.mean(dim=1)  # (B, D)
            h_norm = r.float().norm(dim=-1).mean(dim=-1)  # (B,)
            confidence = torch.ones(B, device=r.device) * 0.5  # placeholder

            step_frac = t / max(K - 1, 1)

            if uniform_routing:
                expert_weights = torch.ones(B, self.cfg.n_experts, device=r.device) / self.cfg.n_experts
                route_logits = expert_weights
            else:
                expert_weights, route_logits = self.router(r_pooled, step_frac, h_norm, confidence)
            route_logits_all.append(route_logits)

            # Get betas for this step
            beta_t = self.expert_betas[:, min(t, self.expert_betas.shape[1] - 1)]  # (n_experts,)

            # Core blocks
            h_core = r + m_0
            for blk in self.core:
                h_core = blk(h_core, self.rope_cos, self.rope_sin, expert_weights, beta_t)
            r_new = h_core - m_0

            # Mix
            r = (1.0 - alpha) * r + alpha * r_new
            r_states.append(r)

            # Halting
            if train_halting or (not self.training and fixed_K is None):
                pooled = self.halt_norm(r).mean(dim=1).float()
                p = torch.sigmoid(self.halt_head(pooled.to(self.halt_head.weight.dtype)))
                halt_probs.append(p)

                # Acceleration-based exit
                delta_norm = (r - r_states[-2]).float().norm(dim=-1).mean(dim=-1)  # (B,)
                if prev_delta_norm is not None and not self.training:
                    accel = (delta_norm - prev_delta_norm).abs() / (prev_delta_norm + 1e-6)
                    if (p.squeeze(-1) > 0.5).all() and (accel < self.cfg.accel_threshold).all():
                        break
                prev_delta_norm = delta_norm

        # ACT mixture or last state
        if train_halting and halt_probs:
            remainders = []
            cum_halt = torch.zeros(B, 1, device=r.device)
            for t, p in enumerate(halt_probs):
                if t < len(halt_probs) - 1:
                    pi = p * (1.0 - cum_halt)
                    cum_halt = cum_halt + pi
                else:
                    pi = 1.0 - cum_halt
                remainders.append(pi)
            r = torch.zeros_like(r_states[0])
            for state, pi in zip(r_states[1:], remainders):
                r = r + pi.unsqueeze(-1) * state
            r = r.to(r_states[0].dtype)
            expected_steps = sum((t + 1) * pi.mean() for t, pi in enumerate(remainders))
        else:
            expected_steps = torch.tensor(float(len(r_states) - 1), device=r.device)

        # Combine
        h = self.combine(torch.cat([m_0, r], dim=-1))

        # Coda
        for blk in self.coda:
            h = blk(h, self.rope_cos, self.rope_sin)

        logits = self.lm_head(self.final_norm(h))

        # Loss
        loss = None
        aux = {
            'expected_steps': expected_steps,
            'alpha': alpha.item(),
            'n_recursions': len(r_states) - 1,
        }

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

            # Route load balancing
            if route_logits_all:
                avg_route = torch.stack([F.softmax(rl, dim=-1) for rl in route_logits_all]).mean(0)
                uniform = torch.ones_like(avg_route) / self.cfg.n_experts
                bal_loss = F.kl_div(avg_route.log(), uniform, reduction='batchmean')
            else:
                bal_loss = torch.tensor(0.0, device=logits.device)

            # Route entropy bonus (encourage exploration)
            if route_logits_all:
                ent = -torch.stack([
                    (F.softmax(rl, dim=-1) * F.log_softmax(rl, dim=-1)).sum(-1).mean()
                    for rl in route_logits_all
                ]).mean()
                ent_bonus = ent  # negative = low entropy = penalize
            else:
                ent_bonus = torch.tensor(0.0, device=logits.device)

            loss = lm_loss + 0.01 * bal_loss + 0.001 * ent_bonus
            aux['lm_loss'] = lm_loss.item()
            aux['bal_loss'] = bal_loss.item()

            if route_logits_all:
                avg_probs = F.softmax(route_logits_all[-1], dim=-1).mean(0)
                aux['route_dist'] = avg_probs.detach().cpu().tolist()

        return logits, loss, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_mor_lite(size='172m'):
    if size == '172m':
        cfg = MoRLiteConfig()
    else:
        cfg = MoRLiteConfig(
            d_model=2048, n_heads=32, n_kv_heads=8, d_head=64,
            ffn_dim=5632, context_len=2048,
            n_prelude=6, n_core=6, n_coda=12,
        )
    model = PSRTMoRLite(cfg)
    n = model.count_params()
    print(f'PSRT-MoR-lite-{n/1e6:.0f}M created')
    print(f'  d={cfg.d_model} heads={cfg.n_heads}/{cfg.n_kv_heads} ffn={cfg.ffn_dim}')
    print(f'  Experts: {cfg.n_experts} (top-{cfg.top_k})')
    print(f'  Zones: {cfg.n_prelude}+{cfg.n_core}+{cfg.n_coda}')
    print(f'  Beta schedules:')
    for i, b in enumerate(cfg.expert_betas):
        print(f'    Expert {i}: {b}')
    print(f'  Parameters: {n:,}')
    return model, cfg


if __name__ == '__main__':
    model, cfg = create_mor_lite('172m')
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss, aux = model(x, labels=x, fixed_K=2, uniform_routing=True)
    print(f'\nSmoke test: logits={logits.shape} loss={loss.item():.4f} '
          f'alpha={aux["alpha"]:.3f} K={aux["n_recursions"]}')
    print(f'Route dist: {aux.get("route_dist", "N/A")}')
