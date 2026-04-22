"""
ARR-PSRT: Adaptive Re-Reading Projected Split-State Recurrent Transformer

The continuous-space analogue of chain-of-thought: each recursion step
re-reads the input with a changed reasoning state, getting genuinely
new context each iteration.

Architecture:
  Embedding → Prelude → Compress to prompt_bank (16 tokens)
  → proj_m(h)=m₀, proj_r(h)=r₀, init scratchpad S₀
  → Loop (up to K times):
      1. Re-read: c_t = CrossAttend(Q=r_t, KV=[prompt_bank; S_t])
      2. Shared attention: u_t = SelfAttend(r_t + c_t + m₀)
      3. Expert FFN: r̃_e = u_t + β_{e,t} · FFN_e(u_t)
      4. Route: r_{t+1} = (1-α)r_t + α · Σ_e π_e · r̃_e
      5. Scratchpad write: S_{t+1} = S_t + W_down(r_{t+1} - r_t)
      6. Halt check
  → Combine([m₀, r_T]) → Coda → LM Head

New vs MoR-lite:
  + Prompt bank (compressed input for re-reading)
  + Cross-attention re-reader (reasoning queries prompt differently each step)
  + Scratchpad (persistent notebook across recursion steps)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List


@dataclass
class ARRConfig:
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

    # Zones
    n_prelude: int = 2
    n_core: int = 3
    n_coda: int = 5
    max_recursion: int = 4

    # Experts
    n_experts: int = 3
    top_k: int = 2
    expert_betas: List[List[float]] = field(default_factory=lambda: [
        [0.25, 0.10, 0.02, 0.00],  # reason-refine
        [0.80, 0.20, 0.05, 0.00],  # math-single-refresh
        [0.10, 0.02, 0.00, 0.00],  # safe-fluency
    ])

    # Re-reading
    prompt_bank_size: int = 16   # compressed prompt tokens
    scratchpad_size: int = 8     # persistent memory slots

    # Halting
    halt_epsilon: float = 0.05
    accel_threshold: float = 0.1
    dropout: float = 0.0


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
    """Standard GQA with RoPE for self-attention."""
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


class CrossAttention(nn.Module):
    """Cross-attention: Q from reasoning state, KV from prompt_bank + scratchpad.
    No RoPE (bank/scratchpad positions are abstract, not sequential)."""
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, q_input, kv_input):
        """q_input: (B, L, D), kv_input: (B, M, D) where M = bank + scratchpad size"""
        B, L, _ = q_input.shape
        M = kv_input.shape[1]

        q = self.wq(q_input).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(kv_input).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(kv_input).view(B, M, self.n_heads, self.d_head).transpose(1, 2)

        # No causal mask — reasoning can attend to all bank + scratchpad slots
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


class ExpertFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wg = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wu = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.wd = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.wd(F.silu(self.wg(x)) * self.wu(x))


class PromptCompressor(nn.Module):
    """Compress sequence hidden states into fixed-size prompt bank via learned queries."""
    def __init__(self, cfg):
        super().__init__()
        self.bank_queries = nn.Parameter(torch.randn(1, cfg.prompt_bank_size, cfg.d_model) * 0.02)
        self.cross_attn = CrossAttention(cfg)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, h):
        """h: (B, L, D) → bank: (B, bank_size, D)"""
        B = h.shape[0]
        queries = self.bank_queries.expand(B, -1, -1)
        bank = self.cross_attn(queries, self.norm(h))
        return bank


class ScratchpadWriter(nn.Module):
    """Write reasoning deltas to scratchpad slots.

    Stabilized: normalize delta_r before projection, add decay to prevent
    unbounded accumulation, clamp write vector, and normalize output.
    """
    def __init__(self, cfg):
        super().__init__()
        self.delta_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.scratch_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj_down = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.scratchpad_size, bias=True)
        nn.init.zeros_(self.gate.bias)
        self.decay = 0.95
        self.write_scale = 0.1
        self.write_clip = 5.0

    def forward(self, scratchpad, delta_r):
        delta = self.delta_norm(delta_r.mean(dim=1))
        write_vec = self.proj_down(delta)
        write_vec = torch.tanh(write_vec / self.write_clip) * self.write_clip
        gate_logits = self.gate(write_vec)
        gate_weights = torch.sigmoid(gate_logits).unsqueeze(-1)
        scratchpad = self.decay * scratchpad + self.write_scale * gate_weights * write_vec.unsqueeze(1)
        return self.scratch_norm(scratchpad)


class CoreBlock(nn.Module):
    """One core block: re-read + shared attention + expert FFN."""
    def __init__(self, cfg):
        super().__init__()
        # Re-reader: cross-attend from reasoning to [bank; scratchpad]
        self.reread_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.reread_attn = CrossAttention(cfg)

        # Shared self-attention
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = GQAttention(cfg)

        # Expert FFNs
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.expert_ffns = nn.ModuleList([ExpertFFN(cfg) for _ in range(cfg.n_experts)])

    def forward(self, r, m_0, bank_scratch, rope_cos, rope_sin,
                expert_weights, beta):
        # 1. Re-read: reasoning queries the prompt bank + scratchpad
        c = self.reread_attn(self.reread_norm(r), bank_scratch)
        r = r + c

        # 2. Shared self-attention on (r + m_0)
        h = r + m_0
        r = r + self.self_attn(self.norm1(h), rope_cos, rope_sin)

        # 3. Expert-weighted FFN with step-specific beta
        # Skip experts with zero beta to avoid 0.0 * inf = NaN tripwire
        z = self.norm2(r)
        ffn_out = torch.zeros_like(r)
        for e, ffn in enumerate(self.expert_ffns):
            be = float(beta[e])
            if be == 0.0:
                continue
            w = expert_weights[:, e].view(-1, 1, 1)
            ffn_out = ffn_out + (w * be) * ffn(z)
        r = r + ffn_out

        return r


class SequenceRouter(nn.Module):
    """Sequence-level top-k router with step/norm/confidence features."""
    def __init__(self, cfg):
        super().__init__()
        router_dim = cfg.d_model + 3
        self.net = nn.Sequential(
            nn.Linear(router_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cfg.n_experts),
        )
        # Zero-init final layer: router starts producing uniform outputs
        # Prevents one-hot collapse at Phase B start (random init → concentrated logits → NaN)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.top_k = cfg.top_k

    def forward(self, r_pooled, step_frac, h_norm, confidence, soft=False, temperature=1.0):
        features = torch.cat([
            r_pooled,
            torch.full((r_pooled.shape[0], 1), step_frac, device=r_pooled.device),
            h_norm.unsqueeze(-1),
            confidence.unsqueeze(-1),
        ], dim=-1)
        logits = self.net(features).clamp(-10, 10)
        if soft:
            # Soft routing with temperature: prevents one-hot collapse
            weights = F.softmax(logits / temperature, dim=-1)
        else:
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, topk_idx, topk_weights)
        # Entropy of routing distribution (for regularization)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        return weights, logits, entropy


class TransformerBlock(nn.Module):
    """Standard block for prelude/coda."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg)
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
# ARR-PSRT Model
# ============================================================

class ARRPSRT(nn.Module):
    def __init__(self, cfg: ARRConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Prelude
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_prelude)])

        # Prompt compressor: prelude output → fixed bank tokens
        self.compressor = PromptCompressor(cfg)

        # Memory/Reasoning projections
        self.proj_m = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj_r = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Initial scratchpad
        self.scratch_init = nn.Parameter(
            torch.randn(1, cfg.scratchpad_size, cfg.d_model) * 0.02)

        # Recursive core
        self.core = nn.ModuleList([CoreBlock(cfg) for _ in range(cfg.n_core)])

        # Scratchpad writer
        self.scratch_writer = ScratchpadWriter(cfg)

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

        # Expert beta schedules
        betas = torch.tensor(cfg.expert_betas)
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
                uniform_routing=False, soft_routing=False, router_temp=1.0):
        B, L = input_ids.shape
        h = self.embed(input_ids)

        # === Prelude ===
        for blk in self.prelude:
            h = blk(h, self.rope_cos, self.rope_sin)

        # === Compress to prompt bank ===
        bank = self.compressor(h)  # (B, bank_size, D)

        # === Split ===
        m_0 = self.proj_m(h)
        r = self.proj_r(h)
        alpha = torch.sigmoid(self.alpha_logit.clamp(-10, 10))

        # === Init scratchpad ===
        scratchpad = self.scratch_init.expand(B, -1, -1).clone()  # (B, scratch_size, D)

        K = fixed_K or self.cfg.max_recursion
        r_states = [r]
        halt_probs = []
        route_logits_all = []
        prev_delta_norm = None

        for t in range(K):
            # Concatenate bank + scratchpad for re-reading
            bank_scratch = torch.cat([bank, scratchpad], dim=1)  # (B, bank+scratch, D)

            # Finite check: catch the first bad tensor, not the last bad loss
            if self.training and not torch.isfinite(bank_scratch).all():
                print(f'  [NaN-TRACE] t={t} bank_scratch nonfinite (scratchpad overflow?)', flush=True)
            if self.training and not torch.isfinite(r).all():
                print(f'  [NaN-TRACE] t={t} r nonfinite before core blocks', flush=True)

            # Router
            r_pooled = r.mean(dim=1)
            # Use mean-abs instead of norm to avoid overflow on large r
            h_norm = r.float().abs().mean(dim=-1).mean(dim=-1).clamp_max(1e4)
            confidence = torch.ones(B, device=r.device) * 0.5
            step_frac = t / max(K - 1, 1)

            if uniform_routing:
                expert_weights = torch.ones(B, self.cfg.n_experts, device=r.device) / self.cfg.n_experts
                route_logits = expert_weights
                route_entropy = torch.tensor(0.0, device=r.device)
            else:
                expert_weights, route_logits, route_entropy = self.router(
                    r_pooled, step_frac, h_norm, confidence, soft=soft_routing, temperature=router_temp)
            route_logits_all.append(route_logits)

            # Get betas for this step
            beta_t = self.expert_betas[:, min(t, self.expert_betas.shape[1] - 1)]

            # Core blocks (re-read + shared attn + expert FFN)
            r_prev = r
            for blk in self.core:
                r = blk(r, m_0, bank_scratch, self.rope_cos, self.rope_sin,
                        expert_weights, beta_t)

            if self.training and not torch.isfinite(r).all():
                print(f'  [NaN-TRACE] t={t} r nonfinite AFTER core blocks', flush=True)

            # Mix with previous
            r = (1.0 - alpha) * r_prev + alpha * r

            # Write to scratchpad
            delta_r = r - r_prev
            scratchpad = self.scratch_writer(scratchpad, delta_r)

            if self.training and not torch.isfinite(scratchpad).all():
                print(f'  [NaN-TRACE] t={t} scratchpad nonfinite after write', flush=True)

            r_states.append(r)

            # Halting
            if train_halting or (not self.training and fixed_K is None):
                pooled = self.halt_norm(r).mean(dim=1).float()
                p = torch.sigmoid(self.halt_head(pooled.to(self.halt_head.weight.dtype)))
                halt_probs.append(p)

                delta_norm = delta_r.float().norm(dim=-1).mean(dim=-1)
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

        # === Combine ===
        h = self.combine(torch.cat([m_0, r], dim=-1))

        # === Coda ===
        for blk in self.coda:
            h = blk(h, self.rope_cos, self.rope_sin)

        logits = self.lm_head(self.final_norm(h))

        # === Loss ===
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

            # Route balancing — compute in fp32 with clamps for numerical safety
            if route_logits_all:
                probs = [F.softmax(rl.float(), dim=-1) for rl in route_logits_all]
                avg_route = torch.stack(probs).mean(0).clamp_min(1e-8)
                uniform = torch.full_like(avg_route, 1.0 / self.cfg.n_experts)
                bal_loss = F.kl_div(avg_route.log(), uniform, reduction='batchmean')
                p_last = probs[-1].clamp_min(1e-8)
                ent = -(p_last * p_last.log()).sum(-1).mean()
            else:
                bal_loss = torch.tensor(0.0, device=logits.device)
                ent = torch.tensor(0.0, device=logits.device)

            # Subtract entropy to ENCOURAGE diversity (maximize entropy = spread weight)
            # 0.1 coefficient: 0.5 was too strong for the v17 architecture (loss went negative)
            loss = lm_loss + 0.01 * bal_loss - 0.1 * ent
            aux['lm_loss'] = lm_loss.item()
            aux['bal_loss'] = bal_loss.item()

            if route_logits_all:
                aux['route_dist'] = F.softmax(route_logits_all[-1], dim=-1).mean(0).detach().cpu().tolist()

        return logits, loss, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_arr_psrt(size='172m'):
    if size == '172m':
        cfg = ARRConfig()
    else:
        cfg = ARRConfig(
            d_model=2048, n_heads=32, n_kv_heads=8, d_head=64,
            ffn_dim=5632, n_prelude=6, n_core=6, n_coda=12,
        )
    model = ARRPSRT(cfg)
    n = model.count_params()
    print(f'ARR-PSRT-{n/1e6:.0f}M created')
    print(f'  d={cfg.d_model} heads={cfg.n_heads}/{cfg.n_kv_heads}')
    print(f'  Zones: {cfg.n_prelude}+{cfg.n_core}+{cfg.n_coda}')
    print(f'  Experts: {cfg.n_experts} (top-{cfg.top_k})')
    print(f'  Prompt bank: {cfg.prompt_bank_size} tokens')
    print(f'  Scratchpad: {cfg.scratchpad_size} slots')
    print(f'  Parameters: {n:,}')
    return model, cfg


if __name__ == '__main__':
    model, cfg = create_arr_psrt('172m')
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss, aux = model(x, labels=x, fixed_K=2, uniform_routing=True)
    print(f'\nSmoke test: logits={logits.shape} loss={loss.item():.4f} '
          f'alpha={aux["alpha"]:.3f} K={aux["n_recursions"]}')
