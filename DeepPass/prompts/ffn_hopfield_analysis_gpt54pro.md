# PROMPT FOR GPT-5.4 PRO: Deep Analysis of FFN Re-Retrieval Corruption in Layer Duplication

## Instructions

You are being consulted on a hard, partially-solved research problem in transformer mechanistic interpretability. We have extensive experimental data and a hypothesis that needs rigorous mathematical formalization, deeper analysis, and targeted experiments to prove or disprove.

**Think extremely deeply.** Use Modern Hopfield Network theory, information geometry, superposition theory, and any other relevant mathematical frameworks. Research online for relevant recent work on FFN factual storage, attention-FFN interaction, and energy landscape distortion in transformers.

Produce:
1. **Mathematical formalization** of why FFN duplication corrupts factual recall
2. **Predictions** that we can test experimentally
3. **Experiment designs** (5-10) to prove the mechanism deeper, ranked by impact
4. **Strategies** for selective sublayer duplication that preserves facts while boosting reasoning
5. **Mathematical proof or derivation** of when FFN duplication helps vs hurts

---

## The Setup

We duplicate contiguous blocks of transformer layers [i, j) in a 72B parameter LLM (Qwen2, 80 layers). The second pass uses the **same weights** — no new parameters. Each layer computes:

```
h_{l+1} = h_l + Attn(LN(h_l)) + FFN(LN(h_l + Attn(LN(h_l))))
```

where:
- `Attn` = grouped-query attention with rotary position embeddings (64 heads, 8 KV heads)
- `FFN` = SwiGLU MLP with hidden_dim=8192, intermediate_dim=29568
- `LN` = RMSNorm

## The Core Observation

**Duplication helps reasoning but hurts factual knowledge.**

lm-eval benchmark results (15% subsample, 72B model):

| Task | Type | Baseline | Pair dup | PLA single | Delta |
|------|------|----------|----------|------------|-------|
| IFEval | Instruction following | 0.545 | 0.567 | **0.605** | **+6.0%** |
| MuSR | Multi-step reasoning | 0.461 | **0.504** | 0.496 | +4.3% |
| BBH | Mixed reasoning | 0.659 | **0.666** | 0.661 | +0.7% |
| MATH Hard | Factual/procedural math | **0.381** | 0.332 | 0.332 | **-5.0%** |
| MMLU-PRO | Factual knowledge | **0.482** | 0.463 | 0.472 | **-1.9%** |

**Reasoning tasks improve. Knowledge tasks degrade.**

## The Sublayer Decomposition

We decomposed the second pass into attention-only and FFN-only components. For block (45,52) on dual probe:

### Per-layer sublayer sensitivity (which sublayer carries the benefit?):

| Layer | Attn-only | FFN-only | Both disabled | Full dup | Dominant |
|-------|-----------|----------|---------------|----------|----------|
| L0 (45) | **77.48** | 75.54 | 74.42 | 77.45 | Attention |
| L1 (46) | 75.72 | **76.89** | 74.91 | 77.45 | FFN |
| L2 (47) | **80.35** | 74.45 | 80.20 | 77.45 | **Attention (FFN destructive!)** |
| L3 (48) | 74.16 | **75.60** | 71.73 | 77.45 | FFN |
| L4 (49) | 71.92 | **74.43** | 72.04 | 77.45 | FFN |
| L5 (50) | **78.76** | 74.02 | 72.91 | 77.45 | Attention |
| L6 (51) | **78.28** | 74.48 | 77.97 | 77.45 | Attention |

**Attention dominates on 4/7 layers (L0, L2, L5, L6). FFN dominates on 3/7 (L1, L3, L4).**

### Critical observation on L2 (global layer 47):
- Attention-only: **80.35** (beats full dup by +2.90!)
- FFN-only: 74.45 (below baseline)
- Both disabled: 80.20 (nearly as good as attn-only — the FFN barely contributes positively)
- **The FFN on L2 is actively destructive during the second pass.**

### Block-level sublayer results:

| Config | Math | EQ | Combined | Delta |
|--------|------|-----|----------|-------|
| (45,52) full dup | 0.718 | 83.1 | 77.45 | +6.93 |
| (45,52) attn-only | 0.695 | 71.6 | 70.54 | +0.01 |
| (45,52) ffn-only | 0.531 | 75.9 | 64.52 | **-6.00** |
| (50,60) attn-only | 0.638 | 79.6 | 71.68 | +1.16 |
| (50,60) ffn-only | 0.629 | 74.9 | 68.90 | -1.62 |
| pair full dup | 0.743 | 85.5 | 79.91 | +9.38 |
| pair attn-only | 0.707 | 79.4 | 75.04 | +4.51 |
| pair ffn-only | 0.565 | 75.1 | 65.79 | **-4.73** |

**FFN-only is consistently harmful. Attention-only is positive or neutral.** But the full duplication (attn+FFN together) scores HIGHER than attention-only on EQ-bench (85.5 vs 79.4 for the pair), suggesting the FFN does contribute to emotional/creative reasoning when combined with attention.

### Sublayer-optimized coordinate descent result:
Best single block with per-sublayer alphas: **82.90** (vs 77.45 full dup, vs 82.77 per-layer alpha)
Optimal sublayer alphas show: attention alphas generally high (0.3-1.15), FFN alphas mixed (some high like L1 FFN=1.5, others very low)

## Our Hypothesis (Needs Formalization)

### The FFN Re-Retrieval Corruption Hypothesis

**Claim:** FFN layers store factual associations as key-value pairs in their weight matrices. During the second pass:

1. The input to the FFN has been perturbed by the first pass's residual connection
2. This perturbed input causes the FFN's key-matching to be slightly off
3. The FFN retrieves a **nearby but incorrect** fact, or a blurred mixture of facts
4. This corrupts the clean factual signal that was correctly retrieved on the first pass

**Meanwhile, attention benefits from repetition because:**
1. Attention performs **computation** (re-weighting which tokens to attend to), not **retrieval**
2. On the second pass, attention sees a **refined** representation (first pass output)
3. Re-computing attention on this refined input genuinely helps reasoning by allowing the model to "reconsider" its attention weights
4. This is analogous to iterative refinement / fixed-point iteration — the attention mechanism converges toward a better solution

### The Hopfield Network Connection

In the Modern Hopfield Network interpretation of transformers (Ramsauer et al., 2021):
- The FFN can be viewed as an associative memory with energy wells around stored patterns
- Each factual association (key → value) corresponds to an energy minimum
- On the first pass, the input lands in the correct energy well → correct fact retrieved
- On the second pass, the perturbed input lands in a **different** (nearby) energy well → wrong fact
- This is analogous to **spurious minima** in Hopfield networks — the energy landscape has many local minima, and a slightly perturbed input can fall into the wrong one

**Key question:** Can we formalize this as an energy landscape distortion? Specifically:
- Let `E(h)` be the FFN's effective energy function
- First pass: `h₁ = FFN(h₀)` retrieves from well at `h₀` → correct
- Second pass: `h₂ = FFN(h₁)` retrieves from well at `h₁ = h₀ + δ` → potentially incorrect if `δ` crosses a basin boundary

### Why EQ-bench Still Benefits

The FFN helps EQ-bench even on the second pass because emotional reasoning may NOT be stored as discrete factual associations. Instead, emotional understanding involves:
- **Distributed representations** that are robust to small perturbations
- **Soft interpolation** between emotional states (not hard key-matching)
- The FFN's second-pass contribution is a **soft refinement** of the emotional signal, not a discrete lookup

This predicts that the FFN damage is specifically to **discrete, precise factual retrieval** (MATH answers, factual knowledge) while **continuous, distributed representations** (emotions, reasoning patterns) are robust.

## The Data We Have

### Cross-layer duplication confirms early weights are "safer"
Using weights from layers 20-27 as the second pass for block (45,52) gives 78.92 — better than self-duplication (66.85 with our generate function). Early-layer FFNs have simpler, more general transformations that are less likely to cause spurious retrieval errors.

### Entropy gating shows knowledge inputs have low entropy
Math prompts: mean entropy 1.72, mean benefit +0.19 (duplication helps)
Knowledge prompts: mean entropy 1.08, mean benefit -0.02 (duplication hurts)
**The model is already confident on knowledge inputs** — the FFN has already retrieved the correct fact, and re-running it can only hurt.

### Norm analysis at the seam
On 72B: norm_ratio = 1.04, cosine = 0.997 between first and second pass outputs.
The perturbation is tiny and almost purely directional — but even this tiny shift can cross FFN basin boundaries for discrete facts.

## What We Need From You

### 1. Mathematical Formalization
Formalize the FFN re-retrieval corruption using:
- Modern Hopfield Network energy landscape theory
- The superposition hypothesis (Elhage et al., 2022) — how polysemantic FFN neurons store multiple facts
- Information-theoretic framework — mutual information between FFN input perturbation and output error

Provide:
- A formal theorem or proposition with proof sketch
- Conditions under which FFN duplication helps (continuous/distributed representations) vs hurts (discrete factual retrieval)
- Predicted relationship between the number of facts stored per neuron (superposition density) and vulnerability to re-retrieval corruption

### 2. Testable Predictions
Generate 5-10 specific, quantitative predictions from the theory that we can verify experimentally on our 72B model. Examples of the type we want:
- "FFN layers with higher superposition (measured by feature density) should show greater performance degradation when duplicated"
- "The corruption should be proportional to the norm of the perturbation δ divided by the minimum basin radius"

### 3. Experiment Designs (ranked by impact)
Design 5-10 GPU experiments we can run on our B200 cluster. For each:
- What to measure
- What the prediction is
- How it proves/disproves the mechanism
- Estimated GPU cost

Focus on experiments that:
- Distinguish our hypothesis from alternatives
- Lead to a **practical fix** (selective sublayer duplication strategy)
- Can be done in 2-8 GPU-hours each

### 4. Optimal Selective Duplication Strategy
Given that we can control `attn_alpha` and `ffn_alpha` independently per layer, and we know:
- Some layers benefit from attention duplication (L0, L2, L5, L6)
- Some layers benefit from FFN duplication (L1, L3, L4)
- The optimal configuration is layer-specific and NOT uniform

Design a **principled strategy** (not just grid search) for choosing sublayer alphas:
- Can we use the FFN's weight spectrum / singular values to predict which layers have "safe" FFN duplication?
- Can we use the attention pattern change between passes to predict which layers benefit from attention duplication?
- Can we design a **hybrid config** where:
  - Attention is fully duplicated on all layers (alpha_attn = 1.0-1.15)
  - FFN is duplicated only on layers where it's safe (alpha_ffn = 1.0 for safe layers, 0.0 for dangerous layers)
  - The "safety" is predicted by a cheap metric (not requiring full evaluation)

### 5. Connection to Greedy Stacking
Our greedy stacking (2-4 blocks with whisper alphas) achieved 82.58. Per-sublayer optimization on a single block achieved 82.90. Can the sublayer insight improve stacking?

Specifically:
- If we do greedy stacking with attention-only on all blocks (FFN skipped on all second passes), do triples work at alpha=1.0?
- Can we design a "sublayer-aware greedy stacking" algorithm that:
  1. Screens blocks using sublayer-decomposed rho (attn_rho vs ffn_rho)
  2. For each block, uses attn-only or full dup based on the screening signal
  3. Stacks multiple blocks with this adaptive sublayer control

### 6. The Bigger Picture
How does this connect to:
- Adaptive computation time (the DeepPass ultimate goal)
- The emerging understanding of transformer circuits
- Practical deployment (can we get speed benefits from skipping FFN on the second pass — it saves ~2/3 of the compute since FFN is the expensive part)

---

## Additional Critical Context

### The Alpha Tuning Breakthrough
We can control the "strength" of duplication per-layer and per-sublayer via alpha blending:
```
h_out = h_before + alpha * (layer(h_before) - h_before)
```
- alpha=0: skip the second pass entirely
- alpha=1: standard duplication
- alpha=1.15: mild overshoot (the block's correction is amplified by 15%)

**Per-layer alpha optimization** on block (45,52) found the optimal alphas for each layer:
`[L0=1.1, L1=1.0, L2=0.5, L3=1.3, L4=1.0, L5=0.9, L6=1.1]`

Notice: **L2 should be dampened to 0.5** — this is the SAME layer where our sublayer analysis shows the FFN is destructive. The per-layer alpha of 0.5 is a blunt compromise: it dampens both attention AND FFN, when ideally we'd keep attention at 1.0 and set FFN to 0.0.

The per-sublayer coordinate descent (still running) is at **82.90** with alphas like:
- L0: attn=1.0, ffn=0.5
- L1: attn=1.15, ffn=1.5
- L2: attn=0.3, ffn=(low)
- L6: attn=0.7, ffn=(TBD)

**The optimal FFN alpha VARIES per layer** — some layers benefit from FFN repetition (L1 ffn=1.5 — boosted!), others are destroyed by it (L2 ffn should be ~0). This is NOT simply "FFN bad" — it's layer-specific.

### Scale-Dependent Mechanism (9B vs 72B)
| Property | 9B (Qwen3.5) | 72B (Qwen2) |
|----------|-------------|-------------|
| Norm ratio (h2/h1) | **1.42** | **1.04** |
| Cosine sim (h1, h2) | **0.975** | **0.997** |
| Norm-preserving helps? | YES (+13.6) | NO (-2.3) |
| Deeper stacking? | NO (triples fail) | YES (quads work) |

On 9B, the perturbation is LARGE (42% norm inflation) and noisy → clamping norms helps. On 72B, the perturbation is TINY (4% norm, 0.3% direction) but sufficient to cross FFN basin boundaries for discrete facts.

**Prediction from the Hopfield theory:** 72B has more facts stored per FFN layer → denser energy landscape → smaller basins → even tiny perturbations cross boundaries. 9B has fewer facts → larger basins → perturbations stay within the correct basin, but the large norm inflation causes a different problem (representational noise).

### Cross-Layer Finding Supports the Hypothesis
Using early-layer FFN weights (layers 20-27) as the second pass for block (45,52) gives 78.92 — much better than self-duplication. Early FFNs:
- Store fewer/simpler facts (early layers handle syntax/low-level features)
- Have larger energy basins (less dense fact storage)
- Are less susceptible to re-retrieval corruption

Deep-layer FFNs (layers 60+) as second pass are catastrophic (combined < 10) because deep FFNs are highly specialized and expect very specific input distributions.

### Screening Metrics Don't Decompose Sublayers
Our best screening metric **SBUID = BLOOD_impact - 6000 * rho** (Spearman r=0.515, p=0.008 on 72B) measures the WHOLE layer — it cannot distinguish whether displacement comes from attention (useful) or FFN (potentially harmful). This is why:
- Rho alone fails (p=0.50) — it mixes good and bad displacement
- BLOOD partially works (Pearson p=0.004) — downstream Jacobian changes may correlate more with FFN corruption than attention re-computation

**A sublayer-decomposed screening metric** (rho_attn, rho_ffn, BLOOD_attn, BLOOD_ffn) could be dramatically more predictive. This has never been attempted.

### Practical Compute Savings
In a transformer layer, the FFN/MLP accounts for approximately **2/3 of the FLOPs** (SwiGLU with intermediate_dim=29568 vs attention with dim=8192). If we can skip FFN on the second pass:
- Each duplicated layer costs only ~1/3 of normal (just attention)
- A 7-layer block duplication goes from 7 extra layers of compute to ~2.3 equivalent layers
- This makes duplication **3x cheaper** per block
- Combined with the quality improvement from avoiding FFN corruption, this is a strict win

### The Full Experimental History
We have run >100 GPU experiments over 4 days, including:
- Greedy spectral stacking (2-5 blocks, with per-block and per-layer alpha)
- Bayesian optimization (60 evals → 83.97, vs 300 grid search → 84.07)
- Cross-layer duplication (G(F(h)) using different block weights)
- Entropy-gated duplication (per-input adaptive gating, weak but significant r=0.34)
- SBUID screening validated across 4 model scales
- Quantization survival (4-bit NF4)
- Inference speed benchmarks (4-20% slowdown, zero VRAM overhead)
- MoE layer duplication (first ever, works on Qwen3-30B-A3B)
- Prompt sensitivity validation (Kendall W=0.808)
- Alpha cross-validation (improvement holds on unseen questions)
- Full 171-question EQ-bench (rankings confirmed, absolute scores lower than 20q subset)

All data is available in structured JSON files. The paper has comprehensive documentation in HISTORY.md (~1800 lines) and PAPER.md (~200 lines).

---

**Think very carefully. Use rigorous mathematics. Reference relevant literature (Modern Hopfield Networks: Ramsauer et al. 2021, Superposition: Elhage et al. 2022, FFN as key-value memories: Geva et al. 2021, Transformer circuits: Anthropic 2023). This is a hard problem at the intersection of mechanistic interpretability, associative memory theory, and practical LLM optimization.**
