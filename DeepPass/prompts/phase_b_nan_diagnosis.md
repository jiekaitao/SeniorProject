# ARR-PSRT Phase B NaN Crisis — Full Context for Diagnosis

## What I Need From You

My ARR-PSRT model (1.1B params, from-scratch) trains fine in Phase A (25K steps, frozen backbone, K=2, uniform routing) but consistently NaN's within 200-2000 steps of Phase B, which introduces **soft learned routing** and **K=3**. I've tried 13 versions over multiple days. I need you to identify the root cause and propose a concrete fix.

## The Architecture

ARR-PSRT = Adaptive Re-Reading Projected Split-State Recurrent Transformer. It's a from-scratch transformer (GPT-2 tokenizer, 1.7B params at 1B size) that performs **multiple reasoning passes** over the input:

```
Embedding → Prelude (6 blocks) → Compress to prompt_bank (16 tokens)
→ proj_m(h)=m₀ (frozen memory), proj_r(h)=r₀ (reasoning state)
→ Init scratchpad S₀ (8 slots, learned parameter)
→ Loop K times:
    1. Re-read: c = CrossAttend(Q=norm(r), KV=[prompt_bank; scratchpad])
    2. r = r + c
    3. h = r + m₀
    4. r = r + SelfAttend(norm(h))
    5. Expert FFN: ffn_out = Σ_e (weight_e * beta_e * FFN_e(norm(r)))
    6. r = r + ffn_out
    7. r = (1-α)·r_prev + α·r     [α is a learned sigmoid scalar, ~0.5]
    8. scratchpad = ScratchpadWriter(scratchpad, r - r_prev)
→ Combine([m₀, r]) → Coda (12 blocks) → LM Head
```

Key details:
- **3 experts** with **top-2 routing** (Phase C) or **soft routing** (Phase B)
- **Prompt bank**: 16 tokens compressed from prelude output via learned cross-attention queries
- **Scratchpad**: 8 slots, updated each iteration via gated write: `scratchpad += sigmoid(gate) * proj(mean(delta_r))`
- **Beta schedule**: each expert has per-iteration scaling `[0.25,0.10,0.02,0.00]`, `[0.80,0.20,0.05,0.00]`, `[0.10,0.02,0.00,0.00]`
- Model trains in **bfloat16** on B200 GPU (192GB)

## Training Phases

- **Phase A (steps 0-25000)**: K=2, uniform routing (router never called), only expert/reread/scratch/compressor params train. Shared backbone is frozen. **This phase works perfectly.** Loss converges from ~10 to ~8.1.
- **Phase B (steps 25000-67000)**: K={2,3} (60%/40% random), soft routing with temperature annealing 3.0→1.0, backbone unfreezes with scaled LR. **This phase always NaN's.**
- **Phase C (steps 67000-100000)**: K={1,2,3,4}, top-2 hard routing, halting enabled. Never reached.

## The Phase B Transition — What Changes

At step 25000, simultaneously:
1. `uniform_routing` → `False` (router is now called)
2. K can now be 3 (was always 2)
3. `soft_routing` → `True` (softmax with temp 3.0→1.0)
4. Backbone unfreezes with warmup LR (0→target over 2000 steps)
5. Gradient clip tightens from 1.0 to 0.1

## Every Crash — Complete Data

| Version | Shared LR | Backbone | Router Init | Entropy Coeff | Died at Step | Steps in Phase B | Key Observation |
|---------|-----------|----------|-------------|---------------|-------------|-----------------|-----------------|
| v2-v9 | various | unfrozen | random | +0.001 (WRONG SIGN) | 25100-25300 | 100-300 | Entropy was PENALIZING diversity |
| v10 | 0.001x | unfrozen | random | -0.1 (fixed) | 27,019 | 2,019 | Best ever. route=[0.33,0.33,0.33]. Loss rising 8.17→8.25 before NaN |
| v11 | 0.0001x | unfrozen | random | -0.1 | 26,691 | 1,691 | Lower LR → died SOONER. Loss rising 8.22→8.71 before NaN |
| v12 | FROZEN | frozen | random | -0.1 | 25,373 | 373 | route=[1.00,0.00,0.00]. Backbone NOT the cause. |
| v13 | FROZEN | frozen | ZEROED | -0.5 | 25,238 | 238 | route=[0.33,0.33,0.33] uniform! Router NOT the cause either. Loss was DECREASING (7.69→7.64→NaN) |

### Critical Observations:
1. **v10 vs v11**: Lower backbone LR made it worse → backbone is NOT the primary cause
2. **v12**: Frozen backbone, died at 373 steps → backbone confirmed NOT the cause
3. **v13**: Frozen backbone + zeroed router + 5x entropy → STILL died at 238 steps, even with perfectly uniform routing. **Router collapse is NOT the cause either.**
4. **v13 gradient norms at step 25000**: expert=17.53 | shared=60.44 | router=0.07 | reread_attn=6.37 | scratch=16.06
5. **v12 gradient norms at step 25000**: expert=0.81 | shared=18.56 | router=2.33 | reread_attn=0.32 | scratch=0.09
6. v13 had 20-178x larger gradients than v12 in expert/reread/scratch — but clip_val=0.1 should handle this
7. The NaN appears suddenly (step 25200 loss=7.64, step 25238 NaN) with no gradual spike

### What's Different Between Phase A and Phase B?

Phase A (works):
- K=2 always
- uniform_routing=True → router never called, expert_weights = [0.33, 0.33, 0.33]
- All 3 experts contribute equally with beta_t weights
- soft_routing=False (doesn't matter since uniform)
- Backbone frozen

Phase B (NaN):
- K=2 or 3 (random per step)
- uniform_routing=False → router IS called (even if weights turn out uniform)
- soft_routing=True with temperature=3.0
- Even with zeroed router producing uniform weights, it still NaN's

**The ONLY changes that are always present in all Phase B crashes:**
1. K can be 3 (40% of steps)
2. The router forward pass runs (even if output is uniform)
3. The loss now includes entropy and balance terms

## Hypotheses I Haven't Tested

1. **K=3 with beta schedule**: At K=3 (iteration t=2), beta values are `[0.02, 0.05, 0.00]`. Expert 2 has beta=0.00 — its FFN contribution is multiplied by zero. But gradients still flow through it (0.0 * ffn(z) still computes ffn(z)). Could this create gradient issues?

2. **Scratchpad accumulation with K=3**: With K=3, scratchpad gets written 3 times. The write is additive: `scratchpad = scratchpad + gate * write_vec`. If gate weights are ~0.5 and write_vec magnitude grows, scratchpad could overflow bfloat16 after 3 writes. bfloat16 max is ~3.4e38 but precision is only 7 decimal digits.

3. **Cross-attention to growing scratchpad**: At iteration 3, the reasoning state cross-attends to [bank; scratchpad] where scratchpad has been written to twice already. If scratchpad values are large, the attention scores could overflow.

4. **The balance loss KL divergence**: `F.kl_div(avg_route.log(), uniform)` — if avg_route has very small values, log() approaches -inf. With soft routing at temp 3.0, min prob ≈ 0.001. log(0.001) = -6.9. This shouldn't NaN but could create large gradients.

5. **The entropy loss computation**: `-(softmax * log_softmax).sum(-1).mean()` is subtracted from loss with coeff 0.5. The gradient of this w.r.t. router params propagates through the entire expert FFN computation (since route_logits come from the router which depends on r_pooled). Could this create a problematic gradient path?

6. **ScratchpadWriter proj_down has no normalization**: `write_vec = proj_down(delta_r.mean(dim=1))`. If delta_r is large (no norm), write_vec is large, and sigmoid(gate) * large_vec accumulates in scratchpad.

7. **`r + m_0` in CoreBlock line 237**: If r has grown large through multiple iterations, r + m_0 could have large values going into self-attention. The pre-norm (norm1) should handle this, but RMSNorm preserves magnitude direction.

## The Full Model Code

```python
"""
ARR-PSRT: Adaptive Re-Reading Projected Split-State Recurrent Transformer
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
    n_prelude: int = 2
    n_core: int = 3
    n_coda: int = 5
    max_recursion: int = 4
    n_experts: int = 3
    top_k: int = 2
    expert_betas: List[List[float]] = field(default_factory=lambda: [
        [0.25, 0.10, 0.02, 0.00],
        [0.80, 0.20, 0.05, 0.00],
        [0.10, 0.02, 0.00, 0.00],
    ])
    prompt_bank_size: int = 16
    scratchpad_size: int = 8
    halt_epsilon: float = 0.05
    accel_threshold: float = 0.1
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, q_input, kv_input):
        B, L, _ = q_input.shape
        M = kv_input.shape[1]
        q = self.wq(q_input).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(kv_input).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(kv_input).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
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


class ScratchpadWriter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_down = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.scratchpad_size, bias=True)
        nn.init.zeros_(self.gate.bias)

    def forward(self, scratchpad, delta_r):
        write_vec = self.proj_down(delta_r.mean(dim=1))  # (B, D)
        gate_logits = self.gate(write_vec)  # (B, S)
        gate_weights = torch.sigmoid(gate_logits).unsqueeze(-1)  # (B, S, 1)
        scratchpad = scratchpad + gate_weights * write_vec.unsqueeze(1)
        return scratchpad


class CoreBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.reread_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.reread_attn = CrossAttention(cfg)
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.expert_ffns = nn.ModuleList([ExpertFFN(cfg) for _ in range(cfg.n_experts)])

    def forward(self, r, m_0, bank_scratch, rope_cos, rope_sin, expert_weights, beta):
        c = self.reread_attn(self.reread_norm(r), bank_scratch)
        r = r + c
        h = r + m_0
        r = r + self.self_attn(self.norm1(h), rope_cos, rope_sin)
        z = self.norm2(r)
        ffn_out = torch.zeros_like(r)
        for e, ffn in enumerate(self.expert_ffns):
            w = expert_weights[:, e].unsqueeze(-1).unsqueeze(-1)
            ffn_out = ffn_out + w * beta[e] * ffn(z)
        r = r + ffn_out
        return r
```

## The Forward Pass (Main Loop)

```python
for t in range(K):
    bank_scratch = torch.cat([bank, scratchpad], dim=1)
    
    r_pooled = r.mean(dim=1)
    h_norm = r.float().norm(dim=-1).mean(dim=-1)
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
    
    beta_t = self.expert_betas[:, min(t, self.expert_betas.shape[1] - 1)]
    
    r_prev = r
    for blk in self.core:
        r = blk(r, m_0, bank_scratch, self.rope_cos, self.rope_sin, expert_weights, beta_t)
    
    r = (1.0 - alpha) * r_prev + alpha * r
    
    delta_r = r - r_prev
    scratchpad = self.scratch_writer(scratchpad, delta_r)
    r_states.append(r)
```

## The Loss

```python
lm_loss = F.cross_entropy(logits[:, :-1].view(-1, vocab_size), labels[:, 1:].view(-1), ignore_index=-100)

avg_route = torch.stack([F.softmax(rl, dim=-1) for rl in route_logits_all]).mean(0)
uniform = torch.ones_like(avg_route) / n_experts
bal_loss = F.kl_div(avg_route.log(), uniform, reduction='batchmean')
ent = -(F.softmax(route_logits_all[-1], dim=-1) * F.log_softmax(route_logits_all[-1], dim=-1)).sum(-1).mean()

loss = lm_loss + 0.01 * bal_loss - 0.5 * ent
```

## The Training Loop Phase B Section

```python
# Phase B settings per step:
fixed_K = random.choices([2, 3], weights=[0.60, 0.40])[0]
uniform_routing = False
soft_routing = True
router_temp = 3.0 - 2.0 * pb_frac  # 3.0 → 1.0

# After loss.backward():
clip_val = 0.1  # (was 1.0 in Phase A)
torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
```

## The 1B Config

```python
cfg = ARRConfig(
    d_model=2048, n_heads=32, n_kv_heads=8, d_head=64,
    ffn_dim=5632, n_prelude=6, n_core=6, n_coda=12,
)
# Total: 1,739,152,269 params
# 6 core blocks, each with: CrossAttention + GQAttention + 3 ExpertFFNs
```

## What the Uniform Control Tells Us

There's a control run (`--uniform_only`) that keeps K=2 and uniform routing forever (no Phase B transition). It's at step 13,700 and **completely stable**. This proves the model architecture CAN train stably — it's the Phase B changes that break it.

## Dense Baseline Comparison

Two dense (no ARR) runs of similar size are at step 65K with PPL ~76-78 and completely stable. These are what ARR needs to beat.

## Questions for You

1. **What is actually causing the NaN?** It's not backbone unfreezing, not router collapse. The model goes from loss=7.64 to NaN in 38 steps with frozen backbone and uniform routing.

2. **Is K=3 the culprit?** Phase A never uses K=3. With K=3, the beta values at t=2 are [0.02, 0.05, 0.00] — expert 2 contributes ZERO. Also, the scratchpad gets a 3rd write. Also, the core blocks run 3×6=18 sequential operations (vs 2×6=12 for K=2).

3. **Is there a numerical stability issue in the recursion loop?** With 6 core blocks × K iterations, the computation graph is very deep. Each core block has cross-attention + self-attention + expert FFN, all with residual connections. At K=3 with 6 core blocks, that's 18 blocks of gradient flow.

4. **Is the problem in the loss terms?** The balance loss uses `avg_route.log()` which could produce very negative values. The entropy term creates a gradient path from the loss through the router through r_pooled back through the entire recursion.

5. **Could this be a bfloat16 precision issue?** The model is trained in bfloat16 (7-bit mantissa). Deep recursion with 18 sequential operations could accumulate enough error to produce NaN. The RMSNorm converts to float32 internally but the activations between blocks are bfloat16.

6. **Proposed fix**: Should I (a) force K=2 only for first N steps of Phase B, (b) add activation clamping inside the recursion loop, (c) run in float32, (d) add LayerNorm to the scratchpad, (e) something else entirely?
