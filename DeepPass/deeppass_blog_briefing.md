# DeepPass Technical Blog Briefing

> **Instructions for the website Claude:** This document contains the full technical story, intuitive explanations, key data, and external references for writing the DeepPass blog post. Please write in a natural, human-like, easy-to-read style with beautiful story-like flow. Be humble with slight excitement — this is a research project by a student, not a corporate product launch. Weave the links in naturally (not as a references section). Use analogies heavily. The audience is ML-curious engineers and researchers who may not know transformer internals deeply. Create sections that flow like a narrative, not a dry paper. Think of it like a 3Blue1Brown video script meets a Distill.pub article.

---

## Part 1: The Core Intuition — LLMs Don't Pay Enough Attention

### The Viral Example

There's a now-famous example that broke the internet: *"I live 20 feet from the car wash. I want to wash my car. Should I walk or drive?"* Most LLMs confidently say "drive" — because they pattern-match "car wash" → "drive car there." They don't stop to think: wait, it's 20 feet away. You could walk.

This isn't a knowledge problem. The model *knows* what 20 feet means. It's an **attention** problem — the model didn't look hard enough at the "20 feet" part before jumping to a conclusion.

A fascinating paper from December 2024 ([Patel et al., "Repeat the Prompt Twice"](https://arxiv.org/pdf/2512.14982)) showed something remarkable: simply repeating the user's prompt twice in the input dramatically improves accuracy on these kinds of questions for non-reasoning LLMs. The model doesn't need new knowledge — it just needs another chance to *look more carefully* at what's already there.

**This is the seed of our entire project.** What if instead of repeating the *prompt*, we repeat the *thinking* — by running certain transformer layers a second time?

### What Layer Duplication Actually Does

A transformer model processes your input through a stack of layers — typically 28 to 80 layers depending on the model. Each layer has two parts:

1. **Attention** — the layer looks across all tokens and decides what's relevant to what ("20 feet" is relevant to "walk or drive")
2. **FFN (Feed-Forward Network)** — the layer retrieves stored knowledge and transforms the signal ("car wash" → facts about car washes)

Layer duplication takes a block of consecutive layers — say layers 45 through 52 — and runs them **twice**. Same weights, no training, no changes. The execution becomes:

```
[Layer 0, 1, 2, ..., 44, 45, 46, 47, 48, 49, 50, 51, 45, 46, 47, 48, 49, 50, 51, 52, 53, ..., 79]
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                        Second pass (same weights)
```

The second pass sees a **slightly refined** input — the output of the first pass. It's like re-reading a paragraph after you've already gotten the gist. You notice things you missed the first time.

**Credit where due:** This technique was discovered by David Ng ([RYS — Repeat Yourself](https://huggingface.co/dnhkng/RYS-XLarge)), who found that duplicating layers 45-52 on a 72B-parameter model improved benchmark scores noticeably — with zero training cost.

---

## Part 2: The Search — How We Found the Best Blocks to Duplicate

### Why Not Just Duplicate Everything?

If one duplicated block helps, why not duplicate all 80 layers? Because the benefit is **block-specific**. Some blocks improve the model. Others destroy it. And duplicating too many layers at once causes interference — the improvements cancel out or compound into noise.

We needed a way to **search** for which blocks help, and then **stack** multiple good blocks without them fighting each other.

### Phase 1: Spectral Screening (Minutes, Not Hours)

Evaluating every possible block configuration takes hours of GPU time per candidate. We needed a cheap filter.

**The idea:** Before running expensive benchmarks, measure how much each candidate block *changes* the model's internal representations when duplicated. We developed **SBUID** (Spectral Block Utility via Impact and Displacement):

```
SBUID = BLOOD_impact - λ × displacement_rho
```

- **BLOOD_impact** measures how much the duplication changes downstream layer behavior (good — means the block is doing something)
- **displacement_rho** measures how much the representations move indiscriminately (bad — means the block is just adding noise)
- Subtracting the noise isolates the "useful signal"

On the 72B model, SBUID achieves a Spearman correlation of r=0.515 (p=0.008) with actual benchmark performance. It's not perfect, but it narrows thousands of candidates down to ~25 worth testing.

**Runtime:** ~20 minutes to screen all candidates on a 72B model. Versus ~5 hours to evaluate each one.

### Phase 2: Dual-Probe Evaluation (Minutes Per Candidate)

For the top ~25 candidates from spectral screening, we evaluate each with a **dual probe**:

1. **Math probe** — 16 hard arithmetic questions with partial-credit scoring (~5 min per config on 72B, ~90 sec on 27B)
2. **EQ-bench probe** — 20 emotional intelligence questions (~60 sec per config)
3. **Combined score** = math × 50 + eq × 0.5 (each contributes ~50 points max, range 0-100)

The dual probe catches both reasoning improvement (math) and generation quality preservation (EQ-bench). A block that helps math but destroys coherent text generation would score poorly overall.

### Phase 3: Greedy Stacking — The Breakthrough

Here's where it gets interesting. Previous work (including Ng's) only tested **single blocks**. We asked: can we duplicate multiple blocks simultaneously?

**First attempt (simultaneous selection):** Pick the two best individual blocks and duplicate both. **Result: interference.** The blocks were chosen independently and fought each other.

**Our approach (greedy iterative):**

1. Screen the original model → find the best single block → apply it
2. Screen the **modified** model (with block 1 already applied) → find the best **complementary** block → apply it
3. Repeat: screen the doubly-modified model → find the best third block
4. Stop when adding another block no longer improves the score

The key insight: **the second block is chosen AFTER the first is applied**, so the screening sees the modified dynamics. It finds blocks that are complementary, not just individually strong.

### The Gemma3-27B Search (Specific Results)

**Model:** Google's Gemma 3 27B Instruct (62 transformer layers)

**Baseline combined score: 80.54**

**Step 1 — Comprehensive single-block search:**
- Screened all block sizes 1-4 across 62 layers using spectral analysis
- Evaluated top candidates with dual probe
- **Best single: block (20,21) = 83.76** (+3.22 over baseline)

**Step 2 — Pair search:**
- Applied (20,21), re-screened for complementary blocks
- Also tested alternative anchors: (0,2), (4,5), (12,13)
- **Best pair: (0,2)+(12,13) = 85.92** (+5.38)

**Step 3 — Triple search (the breakthrough):**
- Applied (0,2)+(12,13), swept late-layer candidates (40-60 range)
- Evaluated ~20 candidates at ~3 min each
- **Best triple: (0,2)+(12,13)+(47,48) = 87.80** (+7.27 over baseline)

The pattern that emerged: **early (0,2) + mid (12,13) + late (47,48)** — three blocks from completely different regions of the network, each contributing something different:
- **Early block (0,2):** Refines the initial embedding representation
- **Mid block (12,13):** Strengthens mid-level feature extraction
- **Late block (47,48):** Polishes the final reasoning steps

### Qwen2-72B Results (The Main Model)

**Model:** CaLMe 2.1 Qwen2 72B (80 transformer layers)

**Baseline: 70.52**
**Ng's single block (45,52): 76.76** (+6.24)

**Our greedy stacking progression:**
| Iteration | Config | Combined | Delta vs Ng |
|-----------|--------|----------|-------------|
| Single | (45,52) | 76.76 | — |
| Pair | (0,7)+(45,52) | 79.91 | +3.15 |
| Whisper quad | 4 blocks @0.02-0.15α | 82.58 | +5.82 |
| Per-layer triple | 3 blocks, 21 tuned alphas | 84.07 | +7.31 |

---

## Part 3: Alpha Tuning — Controlling the Volume Knob

### The Problem with Full Duplication

Running a block twice at full strength is like turning the volume to 11. Sometimes that's great. Sometimes it distorts.

### The Alpha Equation

We introduced a **blending parameter α** (alpha) at the seam between the first and second pass:

```
h_out = h₁ + α × (h₂ - h₁)
```

Where:
- **h₁** = the representation after the first pass through the block
- **h₂** = the representation after the second pass
- **α = 1.0** = standard duplication (use the second pass output entirely)
- **α = 0.0** = no duplication (ignore the second pass completely)
- **α = 0.5** = blend 50/50 between first and second pass
- **α = 0.1** = "whisper" mode — just a tiny nudge from the second pass

Think of it like this: the second pass produces a "correction" `(h₂ - h₁)`. Alpha controls **how much of that correction to apply**. Full correction? Half? Just a whisper?

### Why Alpha Matters: The Stacking Problem

When we stack multiple duplicated blocks, each one perturbs the signal. The first block's perturbation gets amplified by the second block, and so on. Without alpha control, stacking more than 2 blocks destroys the model.

**Whisper alphas** (α = 0.02 to 0.15) on additional blocks solved this. The first block runs at full strength. The second at 0.15. The third at 0.05. Each adds a gentle refinement without destabilizing the signal.

### Per-Layer Alpha: Each Layer Is Different

The real breakthrough came from realizing that **each layer within a duplicated block has a different optimal alpha**. For the 72B model's block (45,52) — 7 layers — the optimal per-layer alphas were:

| Layer | Optimal α | Interpretation |
|-------|-----------|----------------|
| 45 (L0) | 1.1 | Slight boost — this layer benefits from full re-computation |
| 46 (L1) | 1.0 | Standard — works fine as-is |
| 47 (L2) | **0.5** | **Dampen** — this layer's FFN is destructive (see Part 4) |
| 48 (L3) | **1.3** | **Strong boost** — this layer's correction is highly valuable |
| 49 (L4) | 1.0 | Standard |
| 50 (L5) | 0.9 | Slight dampen |
| 51 (L6) | 1.1 | Slight boost |

**Result:** Single block with 7 per-layer alphas → **82.77** (vs 76.76 with uniform α=1.0). That's a +6.01 improvement just from tuning 7 numbers.

### How We Tune: Bayesian Optimization

Tuning 7-21 alphas by grid search takes 300+ evaluations. We used **Bayesian optimization** (Optuna's Tree-structured Parzen Estimator):

- Models which alpha values are promising based on past evaluations
- Intelligently explores the search space
- **60 evaluations → 83.97** vs **300 evaluations → 84.07** (grid search)
- 5x more efficient, within 0.1 points of the optimum

---

## Part 4: The FFN Memory Wells — Why Duplication Helps Reasoning But Can Hurt Knowledge

### FFNs as Associative Memory

This is where the story gets really interesting, and where we owe a debt to 3Blue1Brown's beautiful explanation of [how LLMs store facts in MLP layers](https://www.youtube.com/watch?v=wjZofJX0v4M).

Each FFN (also called MLP) layer in a transformer acts like an **associative memory** — a lookup table of sorts. When a representation comes in, the FFN's gate neurons activate in a specific pattern, and the output is a weighted combination of stored "value vectors." Each stored pattern is like a **memory well** (think of a marble rolling on a landscape of hills and valleys — each valley is a stored fact).

In the mathematical formulation (SwiGLU architecture):

```
FFN(u) = W_out · (silu(W_gate · u) ⊙ (W_up · u))
```

Each intermediate channel is a memory cell: a query-sensitive gate (which facts are relevant?) multiplied by a value vector (what does that fact say?).

### The Basin-Crossing Problem

When the second pass runs, the input to each FFN is **slightly different** from the first pass — the attention mechanism has refined it. Usually this is great for attention (it gets to look again, more carefully). But for the FFN, this slight perturbation can be **catastrophic**.

**The analogy:** Imagine a landscape of memory wells. Each well stores a different fact. The marble (your representation) sits in the correct well after the first pass. The second pass gives the marble a tiny push. If the well is deep and wide, the marble stays put — same fact retrieved. But if the well is shallow, or there's a competing well very close by, the push knocks the marble into the **wrong well**. The model retrieves a *nearby but incorrect fact*.

This is the **FFN re-retrieval hypothesis**: the second pass corrupts factual recall by crossing basin boundaries in the FFN's energy landscape.

### The Evidence

We decomposed duplication into attention-only and FFN-only components on the 72B model. The results were striking:

**On layer 47 specifically:**
- Full duplication (attention + FFN): 77.45 combined
- Attention-only duplication (skip FFN second pass): 80.35 combined
- **The FFN is actively destructive** — removing it improves the score by 2.90 points

**Jaccard instability** (how much the FFN gate pattern changes between passes):

| Layer | Gate Stability | Interpretation |
|-------|---------------|----------------|
| 45 | 0.354 | Very unstable — different neurons fire |
| 46 | 0.393 | Unstable |
| 47 | 0.466 | Moderate (but most destructive) |
| 48 | 0.507 | Moderate |
| 49 | 0.584 | More stable |
| 50 | 0.606 | Stable |
| 51 | 0.612 | Most stable |

**Correlation between instability and FFN harm: Pearson r = -0.89.** The more the gates change between passes, the more the FFN hurts. This is exactly what the basin-crossing theory predicts — changed gates mean the marble is landing in different wells.

**Benchmark-level evidence (lm-eval on full datasets):**

| Task Type | Example | Duplication Effect |
|-----------|---------|-------------------|
| Reasoning | IFEval (+2.3%), MuSR (+1.3%), BBH (+0.97%) | **Improves** |
| Factual knowledge | MMLU-PRO (-0.80%), MATH Hard (+0.38%) | **Hurts or flat** |

Reasoning tasks improve because attention re-computation helps. Factual tasks degrade because FFN re-retrieval corrupts stored knowledge.

### The Scale Effect

This mechanism is **scale-dependent**:

| Property | 9B Model | 72B Model |
|----------|----------|-----------|
| Second-pass norm inflation | 42% | 4% |
| Cosine similarity (h₁ vs h₂) | 0.975 | 0.997 |
| Memory wells (facts per neuron) | Fewer, wider | More, narrower |

Larger models store more facts in superposition (more marbles crammed into the same landscape). The wells are **narrower and closer together** — so even a tiny push from the second pass can knock a marble into the wrong well. This explains why per-layer alpha tuning is so critical on large models, and why the FFN on layer 47 specifically needs to be dampened.

### The Attention Benefit

While the FFN story is about potential harm, the **attention** story is purely positive. The second pass gives the attention mechanism another chance to:

1. Notice relationships it missed (like "20 feet" being relevant to transportation choice)
2. Refine which tokens are attended to, given the slightly updated representations
3. Sharpen the signal for downstream layers

This connects directly back to the ["Repeat the Prompt Twice"](https://arxiv.org/pdf/2512.14982) finding — repetition helps because it gives the model more opportunities to attend to what matters.

---

## Part 5: Cross-Architecture Generalization

This isn't a one-model trick. We tested across 5 architectures:

| Model | Params | Layers | Baseline | Best Config | Improvement |
|-------|--------|--------|----------|-------------|-------------|
| Qwen2-72B | 72B | 80 | 70.52 | Triple + per-layer α | +13.55 |
| Gemma3-27B | 27B | 62 | 80.54 | Triple (0,2)+(12,13)+(47,48) | +7.27 |
| Qwen3.5-27B | 27B | 64 | 42.86 | Triple | +37.19 |
| Qwen3-30B MoE | 30B | 48 | 27.76 | Single best | +12.66 |
| Qwen3.5-9B | 9B | 32 | — | Limited benefit | Small |

The technique works across model families (Qwen, Gemma), sizes (9B to 72B), and even architectures (dense vs Mixture-of-Experts). Larger models benefit more, and stacking more blocks helps more on larger models.

---

## Part 6: Practical Implications

### What This Means

- **Zero training cost:** No gradient updates, no data needed. Just rearrange the execution order.
- **Zero extra memory:** The duplicated layers share weights with the originals. VRAM usage is identical.
- **Minimal speed cost:** 4-20% slower depending on how many blocks are duplicated and how large they are.
- **Works with quantization:** 4-bit NF4 quantization preserves the duplication benefit. A 72B model runs in 59GB with duplication enabled.

### The Bigger Picture

This is a step toward **adaptive computation** — the idea that models should think harder on difficult problems and breeze through easy ones. Layer duplication is a crude but effective form of this: repeat certain computations to refine the answer.

The FFN re-retrieval finding also opens a new direction: **sublayer-selective duplication**, where you repeat the attention mechanism (beneficial) but skip or dampen the FFN (potentially harmful) on a per-layer basis. Our hybrid configurations already show this works, achieving strong results with 35-65% less additional compute than full-block duplication.

---

## Key Numbers for the Blog

### Headline Results
- **+7.31** improvement on 72B model over previous state-of-the-art (Ng's RYS)
- **+7.27** improvement on Gemma3-27B (87.80 combined, from 80.54 baseline)
- **84.07** best combined score on 72B (triple with 21 per-layer alphas)
- **87.80** best combined score on Gemma3 (triple with greedy stacking)
- **5x** more efficient alpha search with Bayesian optimization
- **0** extra VRAM required
- **0** training needed

### Key Correlations
- SBUID screening: Spearman r=0.515, p=0.008
- Jaccard instability ↔ FFN harm: Pearson r=-0.89
- Cross-validation holds: +2.49 on unseen questions vs +2.83 on training set

---

## External Links to Include

1. **3Blue1Brown — How LLMs Store Facts in MLPs:** https://www.youtube.com/watch?v=wjZofJX0v4M
   - *Context:* This video explains how MLP/FFN layers act as key-value memory stores. Our FFN re-retrieval hypothesis builds directly on this understanding. Weave this in when explaining why duplicating FFN layers can hurt — the memory wells analogy.

2. **"Repeat the Prompt Twice" paper (Patel et al., Dec 2024):** https://arxiv.org/pdf/2512.14982
   - *Context:* This paper showed that simply repeating the prompt improves non-reasoning LLMs. It's the intuitive seed — LLMs don't "pay enough attention." Our work takes this from prompt-level to architecture-level: instead of repeating the input, repeat the computation.

3. **David Ng's RYS (Repeat Yourself):** https://huggingface.co/dnhkng/RYS-XLarge
   - *Context:* The original discovery that duplicating layers 45-52 improves a 72B model. Our work builds on Ng's finding with screening, stacking, and alpha optimization.

---

## Tone & Style Instructions for Website Claude

- **Humble but excited.** This is a student research project that found something surprising. Don't oversell. Say things like "we were surprised to find" and "this suggests" rather than "we proved" or "this breakthrough."
- **Story-like flow.** Start with the "LLMs don't pay attention" intuition. Build to the technical details. End with implications. Each section should feel like a natural next chapter.
- **Analogies are king.** The memory wells analogy, the "volume knob" for alpha, the "re-reading a paragraph" metaphor — use these heavily. The 3Blue1Brown video link should feel like a natural aside ("for a beautiful visual explanation of how this works, see...").
- **Show the journey.** Include failures and dead ends briefly — "we first tried X, which didn't work because Y, which led us to Z." This makes the narrative more authentic and relatable.
- **Visual-friendly.** Suggest diagrams for: (1) the layer execution order with duplicated blocks highlighted, (2) the alpha blending equation with a slider visual, (3) the memory wells landscape with the marble being pushed. These don't need to be in the document — just indicate where the website designer should add visuals.
- **Accessible but not dumbed down.** Assume the reader knows what a transformer is (roughly) but may not know what SwiGLU or Jaccard instability means. Explain jargon inline with parentheticals.
- **Credit generously.** Ng, 3Blue1Brown, the Repeat-the-Prompt paper. This builds on others' work and we should say so clearly.
- **End with questions, not answers.** The FFN re-retrieval hypothesis is still relatively untested beyond our correlation data. Per-input adaptive gating is an open problem. Frame the ending as "here's what we think is happening and here's what we want to explore next."
