# PROMPT FOR GPT-5.4 PRO: Novel Screening Metrics for Layer Duplication Quality

## Instructions

You are being asked to solve a hard open research problem. **Think extremely hard and deeply.** Research online for relevant literature. Produce up to 16 distinct candidate solutions, ranked by likelihood of success. Each must include:
1. Full mathematical specification
2. Rigorous justification for why it should work
3. Computational cost analysis
4. Predicted failure modes
5. How to validate it

Do NOT produce garbage filler ideas. If you only have 8 good ones, stop at 8. Quality over quantity.

Consider techniques from: topological data analysis, spectral graph theory, information geometry, random matrix theory, persistent homology, Fisher information, optimal transport, neural tangent kernels, representation similarity analysis (CKA/CCA), Hessian analysis, loss landscape geometry, singular learning theory, mean field theory, feature attribution, circuit analysis, and any other mathematical framework you can find that might apply.

---

## Problem Statement

### Setup

We have a pre-trained transformer language model with N layers (N=80 for our main model, a 72B parameter Qwen2). Each layer `l` computes:

```
h_{l+1} = h_l + Attn(LN(h_l)) + FFN(LN(h_l + Attn(LN(h_l))))
```

**Layer duplication** takes a contiguous block of layers [i, j) and runs them twice in sequence:

```
Original: h_0 → L_0 → L_1 → ... → L_{i-1} → L_i → ... → L_{j-1} → L_j → ... → L_{N-1} → output
Duplicated: h_0 → L_0 → ... → L_{i-1} → [L_i → ... → L_{j-1}] → [L_i → ... → L_{j-1}] → L_j → ... → L_{N-1} → output
```

The second pass uses the **same weights** — no new parameters. This gives a model with N + (j - i) effective layers.

### The Problem

**Some blocks dramatically improve the model when duplicated. Others destroy it.** On our 72B model:

- Duplicating block [42,49) improves our combined quality score by +9.70 points (+13.8%)
- Duplicating block [45,52) improves by +6.93 (+9.8%)
- Duplicating block [28,33) **degrades** by -7.05 (-10.0%)
- Duplicating block [50,55) **degrades** by -5.89 (-8.4%)

We need a **cheap screening metric** — something computable in seconds or minutes per candidate block, without running expensive evaluation (which takes ~5 minutes per block on our 72B model, and there are O(N²) possible blocks).

### What We Tried and Why It Failed

#### 1. Displacement Rho (Spectral Norm of Logit Perturbation)

**Definition:**
```
ρ(i,j) = E_x[ ||logits(dup(x)) - logits(base(x))|| / ||logits(base(x))|| ]
```

where `dup(x)` is the model with block [i,j) duplicated and `base(x)` is the original model.

**Hypothesis:** Lower ρ means the block is more "contractive" — the second pass doesn't move the representation much, so duplication is safe.

**Result on 7B model (28 layers, n=14):** Spearman r = -0.582, p = 0.029. **Significant.** Low rho predicts good blocks.

**Result on 72B model (80 layers, n=25, fresh contemporaneous measurements):** Spearman r = +0.143, p = 0.495. **Not significant. Wrong sign.** The best blocks actually have HIGH rho (0.33-0.40), while low-rho blocks (0.16-0.19) have mixed or negative performance.

**Why it fails on 72B:** The best duplicable blocks are in the mid-deep region (layers 40-60) which have intrinsically higher logit displacement. Early blocks (layers 0-20) have low displacement but aren't the best for duplication. Rho captures perturbation magnitude, not perturbation *quality*.

#### 2. BLOOD Impact (Downstream Jacobian Smoothness Change)

**Definition:**
```
BLOOD_l = ||J_l||²_F  (Frobenius norm of the Jacobian at layer l, estimated via Hutchinson)
BLOOD_impact(i,j) = Σ_{l=j}^{N-1} [BLOOD_base_l - BLOOD_dup_l]
```

Measures how much downstream layers' Jacobian smoothness changes after duplication.

**Result on 7B (n=20):** Spearman r = -0.492, p = 0.028. **Significant.**

**Result on 72B (n=25):** Spearman r = +0.371, p = 0.068 (borderline). **Pearson r = +0.550, p = 0.004 (significant).** There IS a linear relationship but it's not monotonic — the ranking is imperfect.

**Partial success but insufficient:** BLOOD captures some signal but can't reliably rank the top-5 blocks.

#### 3. Residual Stability (Pairwise Block Independence)

For predicting which PAIRS of blocks work together:

**Definition:** Measure cosine similarity of block B's "residual" (logit delta) with and without block A also duplicated:
```
stability(A,B) = cos(residual_B_alone, residual_B_with_A)
```

**Result on 72B:** Spearman r = 0.117, p = 0.62 against actual pair quality. **Not significant.**

#### 4. DICE Epistasis Features (Multi-Feature Pair Prediction)

Combined 6 features: region_distance, effect_orthogonality, territory_orthogonality, rho_lift, blood_lift, out-of-distribution safety.

**Result on 72B (23 labeled pairs):** Spearman r = 0.191, p = 0.39. AUROC = 0.537. **Not significant.** Only region_distance had individual significance (r = 0.504, p = 0.017).

### The Complete Data

Here are all 25 blocks tested on the 72B model with fresh, contemporaneous measurements (deterministic — 0.00 std across 5 runs):

```
Block   | Midpoint | Size | Rho    | BLOOD_impact | Delta   | Combined
--------|----------|------|--------|-------------|---------|--------
(42,49) |  45.5    |  7   | 0.3307 |    +3377.3  |  +9.70  | 80.22
(50,60) |  55.0    | 10   | 0.3475 |    +1402.9  |  +8.32  | 78.84
(45,52) |  48.5    |  7   | 0.3982 |    +3496.7  |  +6.93  | 77.45
(45,50) |  47.5    |  5   | 0.3676 |    +3336.7  |  +5.48  | 76.00
(20,27) |  23.5    |  7   | 0.2158 |     +247.9  |  +4.47  | 75.00
(55,60) |  57.5    |  5   | 0.3141 |     +150.2  |  +4.08  | 74.61
(55,62) |  58.5    |  7   | 0.2890 |     +395.2  |  +2.87  | 73.40
(22,27) |  24.5    |  5   | 0.1938 |     +372.6  |  +2.57  | 73.10
( 0, 7) |   3.5    |  7   | 0.2047 |     +110.1  |  +2.39  | 72.92
( 4, 9) |   6.5    |  5   | 0.1650 |      -74.8  |  +1.19  | 71.71
(15,20) |  17.5    |  5   | 0.1847 |      -50.5  |  +1.02  | 71.55
(30,37) |  33.5    |  7   | 0.3231 |     +326.3  |  +0.89  | 71.41
(40,47) |  43.5    |  7   | 0.3211 |    +2941.3  |  +0.62  | 71.15
(40,45) |  42.5    |  5   | 0.2111 |     +870.7  |  +0.09  | 70.61
( 8,13) |  10.5    |  5   | 0.1796 |     -196.9  |  -0.96  | 69.57
(10,17) |  13.5    |  7   | 0.1701 |      -53.5  |  -1.78  | 68.74
(60,65) |  62.5    |  5   | 0.4246 |      +59.9  |  -1.98  | 68.54
( 5,12) |   8.5    |  7   | 0.1866 |     -111.6  |  -3.33  | 67.20
(70,77) |  73.5    |  7   | 0.3127 |    -1838.2  |  -4.57  | 65.95
( 0, 3) |   1.5    |  3   | 0.2679 |     +102.0  |  -5.01  | 65.51
(35,40) |  37.5    |  5   | 0.2949 |      +63.4  |  -5.06  | 65.46
(25,32) |  28.5    |  7   | 0.3241 |     +246.7  |  -5.30  | 65.22
(50,55) |  52.5    |  5   | 0.2167 |    +1351.0  |  -5.89  | 64.63
(65,72) |  68.5    |  7   | 0.3383 |     +723.9  |  -6.11  | 64.41
(28,33) |  30.5    |  5   | 0.3044 |     +533.6  |  -7.05  | 63.47
```

### Key Observations About the Data

1. **The best blocks cluster in layers 42-60** (midpoints 45-58). This is the "mid-deep" region.
2. **Rho doesn't separate good from bad** — the best block (42,49) has rho=0.33, while a terrible block (35,40) has rho=0.29. They're indistinguishable.
3. **BLOOD impact is high for good blocks** (3000-3500) but also high for mediocre ones (40,47 = 2941 but delta=+0.62). It's necessary but not sufficient.
4. **Block size matters somewhat** — larger blocks (7-10 layers) tend to be better, but (50,55) size=5 is terrible while (45,50) size=5 is good.
5. **Position is the strongest crude predictor** — mid-deep blocks are better — but within that region, ranking is hard.
6. **The relationship is NON-MONOTONIC** — no single feature increases/decreases linearly with quality.
7. **Negative BLOOD blocks can still be good** — (4,9) has blood=-75 but delta=+1.19. (50,55) has blood=+1351 but delta=-5.89.

### Model Architecture Details

- **Qwen2-72B:** 80 transformer layers, hidden_dim=8192, 64 attention heads, GQA with 8 KV heads
- Each layer: pre-norm (RMSNorm), grouped-query attention with rotary position embeddings, SwiGLU FFN
- Residual connections around both attention and FFN sublayers
- The second pass through a block sees its own output as input — the residual connection means `h_{out} = h_{in} + f(h_{in})`, so the second pass computes `h_{out2} = h_{out} + f(h_{out})`

### Correlation Summary (what works, what doesn't)

```
Metric           | Spearman r | p-value | Pearson r | p-value | Verdict
-----------------|-----------|---------|-----------|---------|--------
rho (displacement) | +0.143   | 0.495   | +0.164    | 0.434   | FAIL
blood_impact       | +0.371   | 0.068   | +0.550    | 0.004   | PARTIAL
block_size         | +0.275   | 0.184   | +0.381    | 0.061   | WEAK
midpoint           | +0.065   | 0.756   | +0.090    | 0.670   | FAIL
-rho (flipped)     | -0.143   | 0.495   | -0.164    | 0.434   | FAIL
```

### Computational Budget

- **"Cheap" means:** < 1 minute per candidate block on a single B200 GPU (179GB VRAM)
- **"Moderate" means:** < 5 minutes per block (same cost as one evaluation, but you only need to rank, not measure quality)
- **Baseline cost:** Full evaluation takes ~5 minutes per block. There are ~120 candidate blocks (varying start positions and sizes). Brute force = ~10 hours.
- **Target:** A metric that, when computed for all ~120 candidates, correctly identifies the top-5 blocks with at least 60% precision (3/5 overlap). Ideally achieves p < 0.05 on Spearman rank correlation.

### What a Solution Looks Like

A mathematical function `score(model, block)` that:
1. Takes the model weights and a block specification [i, j)
2. Computes a scalar score using only forward passes, weight inspection, or gradient computation (no evaluation data required beyond a few calibration prompts)
3. Produces rankings where higher-scored blocks tend to have higher duplication quality
4. Achieves Spearman r > 0.4 with p < 0.05 on our 25-block validation set
5. Costs < 5 minutes per block (ideally < 1 minute)

### Important Constraints

- **No training allowed** — the metric must be computable from the pre-trained weights alone (plus a few calibration prompts for forward passes)
- **Must work on 72B** — metrics that only work on small models are insufficient
- **Must handle varying block sizes** — blocks range from 3 to 10 layers
- **The metric doesn't need to be perfect** — even a coarse filter that reliably identifies the right *region* (layers 40-60) would be valuable, as long as it has statistical backing

### What We Think Is Going On (Hypotheses to Test)

1. **The good blocks are in a "representation transition zone"** where the model shifts from syntactic to semantic processing. Duplication gives extra compute where the representation is most actively being transformed.

2. **The second pass acts as an implicit denoiser** — the block already learned to map its input distribution to a refined output, and seeing its own output as input is like a fixed-point iteration. But this only works if the block's function is *mildly contractive* in the right subspace.

3. **The quality might depend on the INTERACTION between the duplicated block and the downstream layers** — not just the block's own properties. A block that produces representations the downstream layers can leverage is good; one that pushes representations off the downstream layers' manifold is bad.

4. **There may be a spectral property of the weight matrices** (e.g., the singular value distribution of the attention/FFN weights, or the effective rank) that predicts duplication quality, even though the simple rho metric doesn't capture it.

---

## Your Task

Propose up to 16 distinct screening metrics, ranked from most likely to succeed. For each:

1. **Name and one-line description**
2. **Full mathematical definition** (be precise — define all notation)
3. **Why it should work** — theoretical justification connecting the metric to duplication quality, with mathematical reasoning
4. **Computational cost** — how many forward passes, what operations, estimated wall time on a 72B model
5. **Predicted failure modes** — when would this metric break?
6. **Validation plan** — how to test it against our 25-block dataset

Research online for relevant literature. Consider work on:
- Neural network pruning metrics (which layers are important?)
- Knowledge distillation (which layers carry the most information?)
- Layer similarity / redundancy analysis (CKA, centered kernel alignment)
- Singular learning theory (learning coefficient / RLCT)
- Fisher information and natural gradient geometry
- Topological data analysis of neural network representations
- Loss landscape sharpness / flatness measures
- Optimal transport between layer representations
- Random matrix theory applied to weight matrices
- Feature attribution methods adapted to layer-level analysis
- The "residual stream" view of transformers (Elhage et al.)

**Think very carefully. This is a hard unsolved problem.**
