# Why Does Our Re-Reading Architecture Lose to a Standard Transformer on Perplexity, and Can It Be Fixed?

## What I Need From You

We built a 1.74B-parameter transformer that performs K re-reading passes over its input before producing output. **Re-reading provably helps within the architecture**: K=2 gives up to 12% lower perplexity than K=1 (same weights, same input, just one more pass). But the architecture as a whole is **4x worse than a standard dense transformer of the same size** on absolute perplexity.

I need you to think extensively and deeply about:
1. **Why** the architecture loses despite re-reading helping — what is the fundamental bottleneck?
2. **Whether** it is even theoretically possible for a re-reading architecture to beat dense at the same parameter count, or if this is a provable impossibility
3. **How** to redesign the architecture so re-reading provides a net benefit over a standard transformer — not just "K=2 beats K=1 within ARR" but "ARR beats dense"
4. **What the optimal allocation of parameters** should be between the standard backbone and the re-reading mechanism

I want mathematical rigor: information-theoretic arguments, capacity analysis, formal comparisons of effective depth, and principled architectural proposals. Think for a long time before answering. Consider this from multiple angles — optimization, information theory, effective depth, parameter efficiency, the lottery ticket hypothesis, and any other frameworks you think are relevant.

## Research Background

### The Core Discovery
We found that duplicating specific transformer layers at inference time on a pretrained 72B model improves reasoning but hurts factual recall:
- IFEval (instruction following): **+2.3%**
- MuSR (multi-step reasoning): **+1.3%**
- MATH (math problem solving): **-6.4%**
- Custom probes: **+7.31 combined score** (with per-layer alpha tuning)

Sublayer analysis revealed: **attention repetition helps, FFN/MLP repetition hurts.** The FFN stores factual associations; re-running it with a shifted hidden state "overshoots" the correct retrieval basin. This motivated ARR: an architecture that repeats attention while fading FFN across passes.

### Key Distinction: Runtime Duplication vs From-Scratch Training
- **Runtime duplication on pretrained models WORKS**: +7.31 on 72B, no extra params, just re-running existing layers
- **Training a re-reading model from scratch DOESN'T WORK**: 4x worse PPL than dense at same param count

This asymmetry is the central puzzle. Why does re-reading help when you bolt it onto a pretrained model but hurt when you train from scratch?

## The Two Models Being Compared

### Dense Baseline (1.74B params)
A standard transformer: embedding → 24 transformer blocks → LM head. Each block has GQA self-attention + SiLU-gated FFN. Trained on identical data (45% FineWeb-Edu + 20% OpenMathInstruct + 35% science/reasoning) with identical optimizer (AdamW, cosine LR, 0.01 weight decay).

**Config**: d_model=2048, n_heads=32, n_kv=8, d_head=64, ffn_dim=5632, 24 layers, context_len=2048, batch_size=4, LR=1.5e-4, 100K steps.

### ARR-PSRT (1.74B params, same budget)
```
Embedding → 6 Prelude blocks → Compress to prompt_bank (16 tokens)
→ proj_m(h)=m₀ (frozen memory), proj_r(h)=r (reasoning state)
→ Init scratchpad S₀ (8 slots)
→ Loop K times:
    for each of 6 core blocks:
        1. CrossAttend(Q=norm(r), KV=[bank; scratchpad])  → r += c
        2. SelfAttend(norm(r + m₀))                        → r += attn_out
        3. ExpertFFN: Σ_e (w_e * β_{e,t} * FFN_e(norm(r))) → r += ffn_out
    r = (1-α) * r_prev + α * r        [α ≈ 0.5]
    scratchpad = decay * S + write(delta_r)
→ Combine([m₀, r]) via Linear(2D→D)
→ 12 Coda blocks → LM Head
```

**Config**: d_model=2048, n_heads=32, n_kv=8, d_head=64, ffn_dim=5632, n_prelude=6, n_core=6, n_coda=12, 3 experts, bank=16, scratchpad=8, context_len=2048, batch_size=4, LR=1e-4, 100K steps.

**Joint training** (no Phase A/B curriculum — all params train from step 0).

## Complete PPL Trajectories

### Dense Baseline (two seeds)
```
Step   2K: PPL  567 / 535
Step   4K: PPL  393 / 397
Step   6K: PPL  306 / 260
Step   8K: PPL  206 / 211
Step  10K: PPL  192 / 179
Step  12K: PPL  164 / 162
Step  14K: PPL  164 / 133
Step  16K: PPL  141 / 148
Step  20K: PPL  123 / 133
Step  30K: PPL  100 / —
Step  50K: PPL   79 / —
Step  80K: PPL   71 / —
Step 100K: PPL   63 / 68
Best:      PPL   60.7 / 59.2
```

### ARR v16b (LR=1e-4, joint training, K=2, uniform routing)
```
Step   2K: PPL K=1=1238  K=2=1269  delta=+31  (K=2 worse)
Step   4K: PPL K=1= 918  K=2= 925  delta= +7  (K=2 worse)
Step   6K: PPL K=1= 893  K=2= 869  delta=-24  (K=2 better!)
Step   8K: PPL K=1= 849  K=2= 804  delta=-45
Step  10K: PPL K=1= 823  K=2= 781  delta=-42
Step  12K: PPL K=1= 804  K=2= 790  delta=-14
Step  14K: PPL K=1= 765  K=2= 703  delta=-62
Step  16K: PPL K=1= 737  K=2= 651  delta=-86  (12% improvement from re-reading!)
```

### ARR v16 (LR=1.5e-4, joint training, K=2, uniform routing)
```
Step   2K: PPL K=1=1167  K=2=1212  delta=+46
Step   4K: PPL K=1= 929  K=2= 942  delta=+13
Step   6K: PPL K=1= 959  K=2= 953  delta= -6
Step   8K: PPL K=1= 980  K=2= 979  delta= -1
Step  10K: PPL K=1= 936  K=2= 905  delta=-31
Step  12K: PPL K=1= 948  K=2= 970  delta=+22  (oscillating)
Step  14K: PPL K=1= 935  K=2= 936  delta= +1
Step  16K: PPL K=1= 940  K=2= 941  delta= +1
Step  20K: PPL K=1= 967  K=2= 897  delta=-69
```

### The Gap at Step 14K
| Model | PPL |
|-------|-----|
| Dense (seed 1) | 164 |
| Dense (seed 2) | 133 |
| ARR v16b K=2 | 703 |
| ARR v16b K=1 | 765 |

**ARR is 4.3x worse.** Re-reading closes 62 PPL points within ARR, but the starting point is 700+ vs 164.

## Parameter Allocation Analysis

### Dense (1.74B)
- 24 identical blocks × (attention + FFN) = 24 unique transformations
- Every parameter contributes to language modeling
- 24 × (self_attn + FFN) ≈ 24 × 71M ≈ 1.70B in blocks + embeddings

### ARR (1.74B, same total)
- 6 prelude blocks: 6 × 71M ≈ 426M (standard transformer)
- 12 coda blocks: 12 × 71M ≈ 852M (standard transformer)
- 6 core blocks, each with:
  - cross-attention (reread): ~34M per block
  - self-attention: ~34M per block  
  - **3 expert FFNs**: ~37M × 3 = 111M per block (3x dense FFN cost!)
  - Total per core block: ~179M
  - 6 core blocks: ~1,074M
- Overhead: proj_m, proj_r, combine, compressor, scratchpad, router ≈ ~50M
- Total: 426M + 852M + 1074M + 50M ≈ 2,402M

**Wait — ARR is actually larger than dense** because core blocks have 3 expert FFNs each. The 1.74B figure is for the 172m config. At the 1B config:
- Dense 1B: ~1.74B params, 24 blocks
- ARR 1B: ~1.74B params, but 18 standard-equivalent blocks (6 prelude + 12 coda) + 6 core blocks with 3 experts each

The core blocks are **much more expensive** than standard blocks. This means the prelude and coda have fewer/smaller blocks to stay within budget, or the experts are smaller. Either way, ARR has fewer effective standard-transformer layers than dense.

## My Hypothesis for Why ARR Loses

### 1. The Capacity Tax
ARR dedicates ~30-40% of its parameter budget to re-reading infrastructure (cross-attention, 3× expert FFNs, scratchpad, projections) instead of standard language modeling capacity. If re-reading improves PPL by 12% (K=2 vs K=1 delta) but the capacity tax costs 4x PPL, the net effect is hugely negative.

### 2. Shared Weights Across Passes Waste Depth
At K=2, ARR runs 6+12+12=30 block evaluations. Dense runs 24. But 12 of ARR's 30 are shared-weight repeats of 6 core blocks. A 30-layer dense transformer would have 30 UNIQUE transformations. ARR's K=2 is not "30 layers of depth" — it's "24 unique layers plus 6 repeated with different beta." The effective new information from pass 2 is much less than 6 fresh layers would provide.

### 3. The Prelude/Coda Bottleneck
Dense has 24 equally-capable blocks. ARR has 6 prelude + 12 coda = 18 standard blocks, but they must support the entire language modeling task PLUS interface with the re-reading core. The prelude is especially thin (6 blocks) — it must produce representations good enough for both language modeling AND prompt bank compression AND the m₀/r split.

### 4. The Prompt Bank Bottleneck (16 tokens)
The prompt bank compresses the full input sequence down to 16 tokens. This is the ONLY input-conditioned memory that the re-reading loop can access. With d_head=64 and 32 heads, the maximum rank of the reread channel is 16×32=512 dimensions per token. You previously showed this is a hard cap on what re-reading can contribute.

### 5. Why Runtime Duplication Works But From-Scratch Doesn't
Runtime duplication on a pretrained model has zero capacity tax: the 72B model already has 80 well-trained layers, and duplicating 7 of them costs nothing in parameters. The re-reading benefit is pure upside.

From-scratch training must ALLOCATE parameters to the re-reading mechanism, stealing from the baseline language modeling capacity. The re-reading benefit must exceed this stolen capacity — and at 1.74B params, it doesn't.

## Specific Questions

Think deeply about each of these. I want you to reason from first principles, not pattern-match to known architectures.

1. **Is there a fundamental information-theoretic argument that re-reading can't help enough to overcome the capacity tax at this scale?** Is there a model size threshold above which the re-reading benefit exceeds the capacity cost? Can you estimate that threshold?

2. **What is the optimal parameter allocation?** If you have a fixed budget of N parameters and want to maximize the benefit of K re-reading passes, how should you split between "backbone capacity" and "re-reading mechanism capacity"? Is there a closed-form or scaling-law argument?

3. **Would a simpler re-reading mechanism work better?** The current design has 3 expert FFNs, a prompt bank compressor, a scratchpad, and a router. What if re-reading was just "run the same standard transformer block twice with a learned gate on the second pass"? That has near-zero capacity tax but still provides repeated attention.

4. **Is the problem actually about training, not architecture?** The model trains at K=2 but evaluates at both K=1 and K=2. Maybe the issue is that training at K=2 forces the model to compromise between being good at K=1 and K=2, and a standard transformer trained for the same FLOPs would be better at K=1? In other words: is the training signal from pass 2 hurting the optimization of pass 1?

5. **Could re-reading work better on specific tasks rather than general language modeling?** Our 72B results showed it helps reasoning (+2.3% IFEval, +1.3% MuSR) but hurts knowledge (-6.4% MATH). Maybe training a re-reading model on reasoning-heavy data would show a benefit, while general LM (next-token prediction on FineWeb) is exactly the wrong evaluation?

6. **What would a provably-better-than-dense re-reading architecture look like?** Not incremental fixes to ARR, but from first principles: given K re-reading passes and N parameters, what architecture maximizes the information extracted per pass while minimizing the capacity tax? Consider approaches from universal transformers, DEQ (deep equilibrium models), and mixture-of-depths.

7. **Is the comparison even fair?** ARR at K=2 uses ~25% more FLOPs per token than dense. If we compared ARR to a dense model with 25% more layers (30 instead of 24), would the gap be even larger? What's the FLOP-normalized comparison?

8. **The alpha mixing problem.** With α=0.5, the final state is 25% initial + 25% pass-1 + 50% pass-2. Your prior analysis showed this gives pass 2 too much leverage relative to its information content. But the model is learning α — it's at 0.56 now (moved from 0.50 toward favoring pass 2). Does this tell us that the model WANTS more pass-2 influence, and the problem is elsewhere?

9. **Would adaptive K help?** Instead of always K=2, what if K=1 for "easy" tokens and K=2 only for tokens where the model is uncertain? This is Mixture of Depths / adaptive computation. Does the re-reading benefit concentrate on specific tokens, and if so, using K=2 everywhere wastes capacity on tokens that don't need it?

10. **The deepest question: does next-token prediction actually benefit from re-reading?** Language modeling is fundamentally a pattern completion task. Does reading the pattern twice genuinely help, or is the 72B improvement we saw just "more compute on the same input" which any form of extra depth would provide?

## The Code

### ARR Config (1B)
```python
ARRConfig(
    d_model=2048, n_heads=32, n_kv_heads=8, d_head=64,
    ffn_dim=5632, n_prelude=6, n_core=6, n_coda=12,
    n_experts=3, top_k=2,
    expert_betas=[[0.25,0.10,0.02,0.00], [0.80,0.20,0.05,0.00], [0.10,0.02,0.00,0.00]],
    prompt_bank_size=16, scratchpad_size=8,
)
# Total: 1,739,156,365 params
```

### CoreBlock
```python
class CoreBlock(nn.Module):
    def __init__(self, cfg):
        self.reread_norm = RMSNorm(cfg.d_model)
        self.reread_attn = CrossAttention(cfg)  # cross-attend to bank+scratchpad
        self.norm1 = RMSNorm(cfg.d_model)
        self.self_attn = GQAttention(cfg)       # self-attend on r+m₀
        self.norm2 = RMSNorm(cfg.d_model)
        self.expert_ffns = nn.ModuleList([ExpertFFN(cfg) for _ in range(3)])

    def forward(self, r, m_0, bank_scratch, rope_cos, rope_sin, expert_weights, beta):
        c = self.reread_attn(self.reread_norm(r), bank_scratch)
        r = r + c
        h = r + m_0
        r = r + self.self_attn(self.norm1(h), rope_cos, rope_sin)
        z = self.norm2(r)
        ffn_out = sum(w_e * beta_e * ffn_e(z) for e, w_e, beta_e, ffn_e 
                      if beta_e > 0)
        r = r + ffn_out
        return r
```

### Recursion Loop
```python
for t in range(K):
    bank_scratch = cat([bank, scratchpad])       # (B, 24, D)
    expert_weights = uniform(1/3, 1/3, 1/3)     # Phase A: no learned routing yet
    beta_t = expert_betas[:, t]                   # t=0: [0.25,0.80,0.10], t=1: [0.10,0.20,0.02]
    
    r_prev = r
    for blk in core_blocks:                       # 6 core blocks
        r = blk(r, m_0, bank_scratch, ..., expert_weights, beta_t)
    
    r = (1-alpha) * r_prev + alpha * r            # alpha ≈ 0.5
    scratchpad = 0.95 * scratchpad + 0.1 * gated_write(delta_r)

h = Linear(cat([m_0, r]))                        # combine → coda → logits
```

### Dense Baseline
```python
# Standard transformer, same d_model/heads/ffn_dim
# 24 blocks, each: self_attn + FFN
# No experts, no bank, no scratchpad, no split state
# Same data, same optimizer, same LR schedule
```

## What Has Been Tried and Failed

### Previous Architecture Attempts (v2-v13)
Trained with Phase A (freeze backbone, train experts) then Phase B (unfreeze). Phase B always crashed with NaN due to:
- Unbounded scratchpad accumulation (fixed with decay+norm)
- Zero-beta expert tripwire: 0.0 × inf = NaN (fixed by skipping)
- Router one-hot collapse (fixed with zero-init + entropy regularization)

### Phase A/B Curriculum (v15 series)
After fixing NaN, the Phase A→B transition caused PPL to oscillate rather than converge. The backbone weights shifted, destabilizing experts trained on frozen features. Three backbone LR scales tried (0.001x, 0.003x, 0.01x) — all either stalled or diverged.

### Joint Training (v16 series — current)
Training all params from step 0 eliminated the Phase A/B problem. PPL converges smoothly. K=2 advantage appears after ~6K steps and grows to -86 at step 16K. But absolute PPL is still 4x worse than dense.

### GPT-5.4 Pro Architectural Redesign (v17)
You previously analyzed why K=2 hurts: pass 2 has too much leverage (50% via alpha=0.5) but too little new information (rank-32 scratchpad channel). Proposed fixes: split-state core, bank=64, slot-attention scratchpad, per-pass eta gates, gated combine. Implementation failed due to entropy term exploiting the new architecture (loss went negative). The fixes may be theoretically correct but were not practically testable.

## Think Extensively About This

Don't just address my hypotheses — develop your own. Consider:
- Scaling laws for recurrent-in-depth architectures
- Comparison to Universal Transformers, DEQ, PonderNet, and other adaptive-depth models
- Whether the split-state design (m₀ vs r) is fundamentally flawed
- Whether re-reading at the architecture level can ever match re-reading at the inference level
- What experiments would definitively resolve whether this approach can work
- Whether the entire from-scratch approach is misguided and we should focus on fine-tuning pretrained models with re-reading capabilities

I would rather you tell me "this approach is fundamentally doomed and here's the proof" than give me false hope. Be brutally honest.
