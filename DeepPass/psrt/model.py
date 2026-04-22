"""
PSRT: Projected Split-State Recurrent Transformer

Core insight from DeepPass: attention re-computation helps reasoning, but
FFN re-computation corrupts factual memory. PSRT makes this separation
architectural: memory is frozen inside the recurrent loop, only reasoning iterates.

Architecture:
    Embedding -> Prelude (standard blocks)
    -> Project to memory (m) and reasoning (r)
    -> Recursive Core: r_{k+1} = (1-alpha)*r_k + alpha*(Core(r_k + m_0) - m_0)
       - Memory m_0 provides persistent factual context
       - Only reasoning r iterates toward a fixed point
       - Halting head controls adaptive depth
    -> Combine(m_0, r_K) -> Coda (standard blocks) -> LM Head

The convergence guarantee: if Core is rho-Lipschitz with rho < 1, then
r_k converges geometrically to a unique fixed point r*, while m_0 drift
is bounded by L_m * B * M (which is zero when beta=0).

Sizes:
    PSRT-172M: d=1024, 10 blocks (2+3+5), vocab=50257 (GPT-2), tied embeddings
    PSRT-1.1B: d=2048, 24 blocks (6+6+12), vocab=50257, tied embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class PSRTConfig:
    vocab_size: int = 50257  # GPT-2 tokenizer
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    d_head: int = 64
    ffn_dim: int = 3072
    context_len: int = 2048
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # Architecture zones
    n_prelude: int = 2
    n_core: int = 3
    n_coda: int = 5
    max_recursion: int = 4

    # Halting
    halt_epsilon: float = 0.05

    # Training
    dropout: float = 0.0

    @property
    def n_total_blocks(self):
        return self.n_prelude + self.n_core + self.n_coda


# ============================================================
# Building Blocks
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


class GQAttention(nn.Module):
    def __init__(self, cfg: PSRTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv, self.d_head).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if self.groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(B, self.n_heads, -1, self.d_head)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(B, self.n_heads, -1, self.d_head)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with SwiGLU FFN."""
    def __init__(self, cfg: PSRTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.wg = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wd = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, h, rope_cos, rope_sin, mask=None):
        h = h + self.attn(self.norm1(h), rope_cos, rope_sin, mask)
        z = self.norm2(h)
        h = h + self.wd(F.silu(self.wg(z)) * self.wu(z))
        return h


# ============================================================
# PSRT Model
# ============================================================

class PSRT(nn.Module):
    """
    Projected Split-State Recurrent Transformer.

    Memory channel (m) is frozen inside the recurrent loop.
    Reasoning channel (r) iterates toward a fixed point.
    The key property: memory retrieval never corrupts, reasoning refines.
    """

    def __init__(self, cfg: PSRTConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Prelude
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_prelude)])

        # Memory/Reasoning projections
        self.proj_m = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_r = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Recursive core (shared weights across iterations)
        self.core = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_core)])

        # Per-iteration mixing parameter (learned)
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid -> alpha in [0,1]

        # Halting head
        self.halt_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.halt_head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.constant_(self.halt_head.bias, 1.0)  # bias toward continuing

        # Combination: merge memory + reasoning
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None, fixed_K=None, train_halting=False):
        """
        Forward pass with split-state recursion.

        Args:
            input_ids: (B, L) token ids
            labels: (B, L) target ids (loss computed on shifted predictions)
            fixed_K: override max recursion depth
            train_halting: enable ACT loss
        """
        B, L = input_ids.shape
        h = self.embed(input_ids)

        # === Prelude ===
        for blk in self.prelude:
            h = blk(h, self.rope_cos, self.rope_sin)

        # === Split into memory and reasoning ===
        m_0 = self.proj_m(h)  # (B, L, D) — frozen during recursion
        r = self.proj_r(h)    # (B, L, D) — iterates

        # === Recursive Core ===
        K = fixed_K or self.cfg.max_recursion
        alpha = torch.sigmoid(self.alpha_logit)

        r_states = [r]
        halt_probs = []

        for k in range(K):
            # Core input: r + m_0 (memory provides persistent context)
            h_core = r + m_0

            for blk in self.core:
                h_core = blk(h_core, self.rope_cos, self.rope_sin)

            # Extract reasoning update: subtract memory contribution
            r_new = h_core - m_0

            # Mixing: r_{k+1} = (1-alpha)*r_k + alpha*r_new
            r = (1.0 - alpha) * r + alpha * r_new

            r_states.append(r)

            # Halting
            if train_halting or (not self.training and fixed_K is None):
                pooled = self.halt_norm(r).mean(dim=1).float()
                p = torch.sigmoid(self.halt_head(pooled.to(self.halt_head.weight.dtype)))
                halt_probs.append(p)

                if not self.training and fixed_K is None:
                    cum = sum(halt_probs)
                    if (cum > 1.0 - self.cfg.halt_epsilon).all():
                        break

        # === ACT mixture or use last state ===
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

        # === Combine memory + reasoning ===
        h = self.combine(torch.cat([m_0, r], dim=-1))

        # === Coda ===
        for blk in self.coda:
            h = blk(h, self.rope_cos, self.rope_sin)

        # === Output ===
        logits = self.lm_head(self.final_norm(h))

        # === Loss ===
        loss = None
        aux = {
            'expected_steps': expected_steps,
            'alpha': alpha.item(),
            'n_recursions': len(r_states) - 1,
        }

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Memory preservation loss (should be near zero by construction)
            # This only matters if we later add optional memory updates
            mem_loss = torch.tensor(0.0, device=logits.device)

            # Halt regularization: penalize expected computation
            halt_loss = 0.01 * expected_steps if train_halting else torch.tensor(0.0, device=logits.device)

            loss = lm_loss + halt_loss + mem_loss
            aux['lm_loss'] = lm_loss.item()

        return logits, loss, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Configs
# ============================================================

def psrt_172m():
    return PSRTConfig(
        d_model=1024, n_heads=16, n_kv_heads=4, d_head=64,
        ffn_dim=3072, context_len=2048,
        n_prelude=2, n_core=3, n_coda=5,
        max_recursion=4, vocab_size=50257,
    )

def psrt_1b():
    return PSRTConfig(
        d_model=2048, n_heads=32, n_kv_heads=8, d_head=64,
        ffn_dim=5632, context_len=2048,
        n_prelude=6, n_core=6, n_coda=12,
        max_recursion=4, vocab_size=50257,
    )


def create_model(size='172m'):
    cfg = psrt_172m() if size == '172m' else psrt_1b()
    model = PSRT(cfg)
    n = model.count_params()
    print(f'PSRT-{n/1e6:.0f}M created')
    print(f'  d={cfg.d_model} heads={cfg.n_heads}/{cfg.n_kv_heads} ffn={cfg.ffn_dim}')
    print(f'  Zones: {cfg.n_prelude} prelude + {cfg.n_core} core (max {cfg.max_recursion}x) + {cfg.n_coda} coda')
    print(f'  Effective depth: {cfg.n_total_blocks} to '
          f'{cfg.n_prelude + cfg.n_core * cfg.max_recursion + cfg.n_coda}')
    print(f'  Parameters: {n:,}')
    return model, cfg


if __name__ == '__main__':
    model, cfg = create_model('172m')
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss, aux = model(x, labels=x, fixed_K=2)
    print(f'\nSmoke test: logits={logits.shape} loss={loss.item():.4f} '
          f'alpha={aux["alpha"]:.3f} K={aux["n_recursions"]}')
