# Why Does Re-Reading (K=2) Produce Worse Perplexity Than Single-Pass (K=1)?

## What I Need From You

I have a 1.1B-parameter transformer that performs K re-reading passes over its input before producing output. The architecture is designed so that the second pass should refine understanding — cross-attending back to compressed input tokens and updating a scratchpad memory. **But K=2 consistently produces worse perplexity than K=1 during evaluation, despite the model being trained at K=2.**

I need a formal mathematical analysis of why the second pass hurts rather than helps, and concrete architectural changes that would fix this. I want proof-level rigor: bounding arguments for information flow, analysis of the gradient dynamics, and information-theoretic arguments for when re-reading should vs shouldn't help.

## Research Background

### The Core Thesis
We discovered that duplicating transformer layers at inference time improves reasoning but hurts factual recall. On a 72B model:
- IFEval (instruction following): **+2.3%** from layer duplication
- MuSR (multi-step reasoning): **+1.3%**
- MATH (math problem solving): **-6.4%**
- Custom probes (arithmetic + EQ-bench): **+7.31** combined score improvement

Sublayer analysis showed: **attention repetition helps, FFN/MLP repetition hurts.** The FFN stores factual associations; re-running it with shifted hidden states "overshoots" the correct retrieval basin — like a lookup table queried with a slightly wrong key.

### The ARR Architecture
To capture the benefit of repeated attention while avoiding FFN corruption, we designed ARR-PSRT (Adaptive Re-Reading Projected Split-State Recurrent Transformer). Key design:
- **Separate attention from FFN in the recursion loop**: cross-attention re-reads input each pass, self-attention refines reasoning, but expert FFN contribution is **faded via beta schedule** across passes: `[0.80, 0.20, 0.05, 0.00]` for the strongest expert
- **Prompt bank**: input compressed to 16 fixed tokens via learned cross-attention queries, providing stable KV for re-reading
- **Scratchpad**: 8 persistent memory slots updated each pass via gated, bounded writes
- **Split state**: input projected into m₀ (frozen memory) and r (evolving reasoning state)

### Dense Baseline
A standard transformer of the same size (1.7B params with same d_model, heads, layers distributed as 6+6+12) trained on identical data:
```
Step  2K: PPL 567
Step  4K: PPL 393
Step  6K: PPL 306
Step 10K: PPL 192
Step 20K: PPL 123
Step 50K: PPL  79
Step 100K: PPL 63 (best: 60.7)
```

### ARR v16 (Current Best Run — Joint Training)
All params train together from step 0 (no curriculum/phase split). K=2 with uniform routing.
```
Step  2K: PPL K=1=1167, K=2=1212  (delta=+46 — K=2 WORSE)
Step  4K: PPL K=1=929,  K=2=942   (delta=+13 — K=2 WORSE)
Step 5.8K: loss=5.15 (still training, next eval at 6K)
```

Training loss trajectory: 6.83 → 6.33 → 5.70 → 5.41 → 5.15 (steps 500 → 1K → 3K → 4K → 5.8K)

**The problem: K=2 PPL is consistently 1-4% worse than K=1.** The model is training with K=2, but when evaluated at K=1 (single pass, no re-reading), it performs better. This means the second pass is actively hurting.

## The Architecture in Detail

### 1B Config
```python
d_model=2048, n_heads=32, n_kv_heads=8, d_head=64, ffn_dim=5632
n_prelude=6, n_core=6, n_coda=12
n_experts=3, top_k=2
prompt_bank_size=16, scratchpad_size=8
Total: 1,739,156,365 parameters
```

### Forward Pass (Pseudocode)
```
h = Embed(input_ids)                    # (B, L, 2048)

# Prelude: 6 standard transformer blocks
for blk in prelude:
    h = blk(h)                          # standard attention + FFN

# Compress input to prompt bank
bank = CrossAttend(Q=learned_queries, KV=norm(h))  # (B, 16, 2048)

# Split into frozen memory and evolving reasoning
m₀ = proj_m(h)                         # (B, L, 2048) — NEVER changes
r  = proj_r(h)                         # (B, L, 2048) — updated each pass
α  = sigmoid(learned_logit)            # ~0.50, mixing parameter
scratchpad = learned_init              # (B, 8, 2048)

# Recursion loop (K=2 during training)
for t in 0..K-1:
    bank_scratch = cat([bank, scratchpad])  # (B, 24, 2048)
    
    # === 6 Core Blocks, each doing: ===
    for each core block:
        # 1. Re-read: reasoning cross-attends to bank+scratchpad
        c = CrossAttend(Q=norm(r), KV=bank_scratch)
        r = r + c
        
        # 2. Self-attention on reasoning + memory
        r = r + SelfAttend(norm(r + m₀))
        
        # 3. Expert FFN with beta decay
        #    t=0: betas = [0.25, 0.80, 0.10]
        #    t=1: betas = [0.10, 0.20, 0.02]
        z = norm(r)
        ffn_out = Σ_e (weight_e * beta_{e,t} * FFN_e(z))
        r = r + ffn_out
    
    # Mix with previous state
    r = (1-α) * r_prev + α * r         # α ≈ 0.50
    
    # Write to scratchpad (stabilized)
    delta_r = r - r_prev
    write_vec = tanh(proj(norm(delta_r.mean(dim=1))) / 5.0) * 5.0
    scratchpad = 0.95 * scratchpad + 0.1 * sigmoid(gate) * write_vec

# Combine and output
h = Linear(cat([m₀, r]))              # (B, L, 2048)
for blk in coda:                       # 12 standard transformer blocks
    h = blk(h)
logits = lm_head(norm(h))
```

### How PPL K=1 vs K=2 Is Evaluated
```python
for K in [1, 2]:
    total_loss = 0
    for batch in eval_data:
        _, loss, _ = model(batch, labels=batch, fixed_K=K)
        total_loss += loss * num_tokens
    PPL[K] = exp(total_loss / total_tokens)
```
The model is called with `fixed_K=1` (one pass through core blocks) or `fixed_K=2` (two passes). Same weights, same input. The only difference is whether the recursion loop runs once or twice.

## The Full Model Code

```python
class ScratchpadWriter(nn.Module):
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
    def __init__(self, cfg):
        super().__init__()
        self.reread_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.reread_attn = CrossAttention(cfg)
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.expert_ffns = nn.ModuleList([ExpertFFN(cfg) for _ in range(cfg.n_experts)])

    def forward(self, r, m_0, bank_scratch, rope_cos, rope_sin, expert_weights, beta):
        # 1. Re-read
        c = self.reread_attn(self.reread_norm(r), bank_scratch)
        r = r + c
        # 2. Self-attention on (r + m_0)
        h = r + m_0
        r = r + self.self_attn(self.norm1(h), rope_cos, rope_sin)
        # 3. Expert FFN (skip zero-beta)
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
```

### The Recursion Loop (from ARRPSRT.forward)
```python
for t in range(K):
    bank_scratch = torch.cat([bank, scratchpad], dim=1)
    
    # Router (currently uniform: all experts get 1/3 weight)
    expert_weights = torch.ones(B, 3, device=r.device) / 3
    
    # Beta for this iteration step
    beta_t = self.expert_betas[:, min(t, 3)]
    # t=0: [0.25, 0.80, 0.10]
    # t=1: [0.10, 0.20, 0.02]
    
    r_prev = r
    for blk in self.core:  # 6 core blocks
        r = blk(r, m_0, bank_scratch, rope_cos, rope_sin, expert_weights, beta_t)
    
    # Alpha mixing
    r = (1.0 - alpha) * r_prev + alpha * r  # alpha ≈ 0.50
    
    # Scratchpad write
    delta_r = r - r_prev
    scratchpad = self.scratch_writer(scratchpad, delta_r)

# After loop: combine m₀ and final r
h = self.combine(torch.cat([m_0, r], dim=-1))  # Linear(4096 → 2048)
```

### The Combine + Coda
```python
h = self.combine(torch.cat([m_0, r], dim=-1))  # (B, L, 2048)
for blk in self.coda:  # 12 standard transformer blocks
    h = blk(h, rope_cos, rope_sin)
logits = self.lm_head(self.final_norm(h))
```

### Key Observation About K=1 vs K=2 Evaluation

When `K=1`:
- The core blocks run once with beta_t = [0.25, 0.80, 0.10]
- r gets one round of cross-attention + self-attention + expert FFN
- alpha mixing: r = 0.5 * r_init + 0.5 * r_after_core
- combine: h = Linear(cat([m₀, r]))
- 12 coda blocks process h

When `K=2`:
- Pass 1: same as K=1 but r is now updated
- Pass 2: core blocks run again with beta_t = [0.10, 0.20, 0.02] (much weaker FFN)
  - Cross-attention now re-reads bank + UPDATED scratchpad
  - Self-attention now uses updated r + m₀
- alpha mixing again: r = 0.5 * r_after_pass1 + 0.5 * r_after_pass2
- combine and coda are the same

**The model trains at K=2 but is worse at K=2 than K=1. This means the second pass is adding noise/corruption to r, not useful refinement.**

## Complete Experiment History

### Phase B NaN Crisis (v2-v14) — Now Solved
The original architecture had Phase A (frozen backbone, train experts only) then Phase B (unfreeze backbone, add K=3 and routing). Phase B always NaN'd:
- v2-v9: entropy regularization sign bug → fixed
- v10-v11: backbone LR too high/low → not the cause
- v12: frozen backbone → still NaN'd → not backbone
- v13: zeroed router → still NaN'd → not router
- **v14: stabilized scratchpad (RMSNorm, decay, bounded writes) + skip zero-beta → SURVIVED 11K+ steps**
- Root cause: scratchpad was unbounded additive integrator overflowing bfloat16, zero beta created 0×∞=NaN

### Phase A/B Curriculum Failure (v15 series)
- v15 (backbone 0.001x): PPL oscillated 850-1037, never converged
- v15b (backbone 0.01x): diverged, gradient explosion
- v15c (backbone 0.003x): slow

### Joint Training Breakthrough (v16 — current)
- v16: all params from step 0, no phase split → 8x more efficient than Phase A/B
- Loss at step 5.8K: 5.15 (competitive with dense at same step)
- **BUT: K=2 PPL consistently worse than K=1 (delta +13 to +46)**

## Hypotheses for Why K=2 Hurts

1. **The alpha mixing destroys information.** At alpha=0.5, the output is `0.5 * r_prev + 0.5 * r_new`. After pass 2, the final r is a 50/50 mix of pass-1 output and pass-2 output. But pass-1 output was already a 50/50 mix of initial r and pass-1 core output. So the final r contains 25% initial, 25% pass-1, 50% pass-2. The initial representation (which the prelude/coda were optimized for) is diluted.

2. **The prompt bank is too small (16 tokens).** The compressed input has only 16 tokens of context. Cross-attending back to 16 tokens may not contain enough information to improve the reasoning state. The re-reading might be "re-reading" noise rather than useful signal.

3. **The core blocks are shared across passes.** The same 6 core blocks run for both t=0 and t=1. They receive different beta values (0.80 vs 0.20 for the strongest expert) but the attention weights and FFN weights are identical. The blocks may have learned to be optimal for pass 1 (where most training gradient comes from) and suboptimal for pass 2.

4. **The coda doesn't know about K.** The 12 coda blocks receive the combined output regardless of how many passes occurred. With K=1, the coda processes r from one pass. With K=2, r has been through two passes and alpha-mixed twice. The coda may have learned to undo/correct the first pass's representation, but the second pass shifts it further from what the coda expects.

5. **Training at K=2 but evaluating at K=1 reveals that the model has learned to rely on the coda to fix pass-2 corruption.** The model might be learning: "pass 2 corrupts, coda fixes." When you evaluate at K=1, there's no corruption for the coda to fix, so it works better.

6. **The scratchpad stabilization (decay=0.95, write_scale=0.1) may be too aggressive.** The scratchpad barely changes between passes (0.1 scale, then normalized). If the scratchpad doesn't carry meaningful information from pass 1 to pass 2, the re-reading in pass 2 is essentially the same as pass 1 — no new information.

7. **Information bottleneck in the combine layer.** `combine = Linear(2*d_model → d_model)` projects the concatenation of [m₀, r] down to d_model. With K=2, r has been refined but m₀ is unchanged. The combine layer may have learned to mostly pass through m₀ (the stable representation) and ignore r (which changes with K), making the recursion irrelevant.

## Mathematical Questions

1. **Bound the information gain from the second pass.** Given that the prompt bank has 16 tokens of dimension 2048, and the reasoning state r has L tokens of dimension 2048, what is the maximum mutual information that cross-attention can extract from the bank on the second pass that wasn't available on the first pass? Under what conditions is this positive?

2. **Analyze the alpha mixing as a contraction.** With alpha=0.5, each pass contracts the representation toward a fixed point. Show that after 2 passes, the effective "distance traveled" from the initial state is bounded, and compute the convergence rate. Is alpha=0.5 too conservative?

3. **Characterize when shared weights help vs hurt across passes.** The core blocks use identical weights for t=0 and t=1 but receive different beta schedules. Under what conditions on the weight matrices does a second pass through the same blocks improve the output? When does it hurt? This relates to the spectral properties of the iteration map.

4. **Is the architecture fundamentally unable to benefit from K>1?** Given the combine layer's bottleneck and the coda's fixed processing, is there a theoretical argument that the optimal strategy is always to put all computation in pass 1 and make pass 2 an identity? If so, what architectural change would break this degeneracy?

5. **What is the correct way to fade FFN contribution across passes?** The current beta schedule [0.80, 0.20, 0.05, 0.00] was hand-designed. Is there a principled derivation (e.g., minimizing the expected change in the nearest-neighbor retrieval basin for memorized facts) for the optimal decay curve?

6. **Compare the effective depth of K=1 vs K=2.** K=1 gives: 6 prelude + 6 core + 12 coda = 24 block evaluations. K=2 gives: 6 prelude + 12 core + 12 coda = 30 block evaluations. A 30-layer standard transformer should have lower PPL than a 24-layer one. Does our architecture's K=2 mode effectively waste those 6 extra block evaluations?
