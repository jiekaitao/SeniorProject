# GPT-5.4 Pro: The Transfer Gap — Why Attention Re-Computation Helps Probes But Not Benchmarks

**INSTRUCTIONS: Think with the depth of a Fields Medal mathematician and the breadth of a Turing Award computer scientist. Search the web extensively for 2025-2026 papers. Explore ideas from dynamical systems, information theory, topology, statistical mechanics, optimal transport, neuroscience, game theory, and any other field that might offer insight. Propose at least 10 approaches before selecting the top 5. Provide proofs or proof sketches, not hand-waving. Include PyTorch pseudocode for every approach.**

---

## The Problem (Precisely Stated)

We have a technique — running certain transformer layers twice with shared weights — that **consistently improves custom reasoning probes** but **does NOT improve standardized benchmarks** when measured on the full evaluation suite.

### The Data (Full 100% lm-eval, definitive numbers)

**Mistral 7B with whisper FFN (β=0.2) on block (28,29):**

| Benchmark | Baseline | Duplicated | Delta |
|-----------|----------|------------|-------|
| BBH | 46.90% | 46.73% | -0.17% |
| MATH Hard | 3.93% | 4.61% | **+0.68%** |
| MMLU-PRO | 30.24% | 30.39% | **+0.15%** |
| MuSR | 44.97% | 45.24% | **+0.26%** |

**LLaMA 3 8B with recursion fine-tune K=2, core [10,13]:**

| Benchmark | Baseline | FT K=2 | Delta |
|-----------|----------|--------|-------|
| BBH | 49.09% | 48.86% | -0.23% |
| MATH Hard | 9.06% | 6.95% | **-2.11%** |
| MMLU-PRO | 35.45% | 35.35% | -0.10% |
| MuSR | 37.96% | 38.49% | **+0.53%** |

**On our custom dual probe (math guesstimate + EQ-bench):**

| Model | Baseline | Best Dup | Delta |
|-------|----------|----------|-------|
| Gemma3 27B (recursion FT K=2) | 76.96 | 81.08 | **+4.11** |
| LLaMA 3 8B (recursion FT K=2) | 63.76 | 66.98 | **+3.22** |
| LLaMA 3 70B (raw dup pair) | 76.73 | 83.28 | **+6.55** |
| Mistral 7B (raw dup) | 39.03 | 41.99 | **+2.96** |

### The Gap

The dual probe shows +2 to +6 point improvements. Full lm-eval shows -0.2% to +0.7%. The 15% subsample inflated apparent gains by 3-5x — BBH appeared +0.9% but was actually -0.23%, MuSR appeared +3.5% but was actually +0.53%.

**The dual probe does not predict standardized benchmark performance.** This is the central finding.

### What We Know Mechanistically

1. **Attention benefits from re-computation** — iterative refinement of context aggregation, confirmed across 8 models

2. **FFN re-computation hurts factual recall** — basin-crossing in associative memory. Gate flip rates: Gemma3 L0=37% (unstable), L12=0.28% (stable), L47=0.65% (stable but harmful, causal: -2.06)

3. **Whisper FFN (β=0.2) is the best compromise** — enough FFN processing to interpret attention output, not enough to corrupt memory. Mistral shows 3/4 benchmarks positive with this.

4. **SBUID screening works on full-attention models** — r=0.668 on LLaMA 3 70B, fails on sliding window (Gemma3)

5. **Recursion fine-tuning (300 steps) teaches models to use K=2** — but the improvement is on the proxy metric, not the target

6. **Pass-2-only LoRA preserves K=1 perfectly** — but K=2 benefit doesn't survive to lm-eval

7. **The effect is real but small** — Mistral full eval: MATH +0.68%, MuSR +0.26%. Not zero, but not the +2.5% the subsample suggested.

---

## Everything We Tried (Complete List)

### Raw duplication approaches:
- Single block, pair, triple, quad configurations on 8 models
- Attention-only duplication (β_FFN=0)
- Whisper FFN (β_FFN=0.2, 0.3, 0.5)
- Per-layer alpha optimization (Bayesian, 40-60 trials)
- Per-sublayer alpha (attention + FFN separately)
- Cross-layer duplication (second pass uses different block's weights)
- Dropping harmful layers (removing L47 from Gemma3 triple)

### Fine-tuning approaches:
- Direct fine-tuning of core layers (v1-v4, various LR/steps)
- Pass-2-only LoRA (rank 8, contrastive + gated)
- Gated contrastive training with task-conditional routing
- Whisper FFN during training (destructive)
- SIRT architecture from scratch (172M, 3-stage training)

### Screening/analysis:
- SBUID spectral screening
- Direct Logit Attribution
- GEM eigenmask
- Gate margin / Jaccard instability
- Causal mediation patching
- HCES cross-entropy search

### What worked partially:
- Mistral whisper FFN: MATH +0.68%, MMLU +0.15%, MuSR +0.26% (full eval)
- LLaMA 3 8B FT: MuSR +0.53% (full eval)
- Pass-2 LoRA: K=1 preservation perfect (0.00 drift on all models)

### What failed:
- Everything on Gemma3 lm-eval (always regresses despite +4-7 on dual probe)
- Gated LoRA (adv_rate=0 with zero init, collapse with random init)
- Wide cores, dual zones, >300 steps training
- SmolLM-1.7B, Mistral recursion fine-tune (collapse)
- SIRT from scratch (insufficient data for benchmark-quality model)

---

## The Deep Question

**Why does attention re-computation help on open-ended reasoning probes but not on standardized multiple-choice and generation benchmarks?**

Our math probe asks "What is 78313086360375 × 88537453126609?" and scores partial credit. EQ-bench asks "Rate the emotions in this dialogue." These are open-ended generation tasks where the model has to PRODUCE an answer.

lm-eval benchmarks are mostly **loglikelihood scoring** (pick the option with highest probability) or **constrained generation** (produce a specific format). The model isn't "reasoning more" — it's computing conditional probabilities.

**Hypothesis: Layer duplication improves generation quality (longer, more coherent reasoning chains) but doesn't improve probability calibration (which option has highest logit).**

If this is true, then:
- The dual probe captures improved generation (real effect)
- lm-eval captures probability calibration (different skill)
- These are different capabilities and duplication only helps one

---

## What We Need

### Part 1: Theoretical Analysis of the Transfer Gap

Formalize mathematically:
- When does improved generation quality imply improved loglikelihood ranking?
- Under what conditions do they diverge?
- Is there a bound on how much generation improvement can exist without loglikelihood improvement?
- What does information theory say about the relationship between iterative refinement and conditional probability estimation?

### Part 2: Is There ANY Way to Make This Work?

Given our comprehensive negative results on full lm-eval, is there a theoretically grounded reason to believe that layer duplication CAN improve standardized benchmarks by more than ~1%? Or have we hit a fundamental ceiling?

Consider:
- **Test-time compute scaling** — recent work (e.g., s1, o1-style) shows that more compute at inference helps. But those use full reasoning chains, not layer repetition. Is layer repetition a fundamentally weaker form of test-time compute?
- **The relationship between depth and calibration** — does adding effective depth improve probability calibration? Or only generation coherence?
- **Optimal transport perspective** — is the Wasserstein distance between K=1 and K=2 hidden state distributions bounded in a way that limits benchmark improvement?

### Part 3: Five Novel Directions

If straightforward layer duplication + fine-tuning has hit its ceiling at ~0.5% improvement, what FUNDAMENTALLY DIFFERENT approaches could leverage the attention re-computation phenomenon for larger gains?

Think beyond "better fine-tuning" — we tried that exhaustively. Think about:
- **Different evaluation paradigms** where re-computation actually matters (chain-of-thought? multi-turn?)
- **Different model regimes** (maybe it only works at 70B+ scale?)
- **Architectural changes** that make re-computation more productive
- **Training from scratch** with recursion built in (not post-hoc)
- **Combining with other test-time compute methods** (tree search, verification, etc.)
- **Applications outside standard benchmarks** where the +4 dual probe improvement IS the relevant metric (coding, creative writing, emotional intelligence, etc.)

For each direction:
1. **Mathematical formulation** with theorems/proofs
2. **PyTorch pseudocode**
3. **Predicted improvement** (be honest — if you think the ceiling is 0.5%, say so)
4. **Compute cost** on B200 GPUs
5. **What would success/failure tell us** about the fundamental nature of attention re-computation

### Part 4: The Honest Assessment

Given everything we've done, give us a mathematically rigorous answer to:

**Is layer duplication a fundamentally limited technique (ceiling ~1% on standardized benchmarks), or is there a path to >5% improvement that we haven't found?**

Support your answer with:
- Theoretical bounds (information-theoretic, dynamical systems, etc.)
- Comparison to other test-time compute methods and their improvement curves
- What the existing literature says about depth vs calibration
- Whether the ~0.5% improvements we see are statistically significant or within noise

### Part 5: Design a SOTA Novel Architecture

Our SIRT-172M proof-of-concept showed the right instincts (3-zone design, token-wise β gate, adaptive halting) but was trained on too little data to matter. GPT-5.4 Pro previously recommended a 170M "deep thin" recursive decoder, but we need to think much deeper.

**Design an architecture from first principles that is mathematically guaranteed (or at least strongly motivated) to benefit from re-computation.** Don't just stack existing ideas — derive the architecture from the theory.

Consider these mathematical foundations:

**From Dynamical Systems:**
- The forward pass is a discrete map h_{t+1} = F_t(h_t). Under what conditions does iterating a subset of these maps improve the fixed-point quality?
- Theorem (desired): For maps F with spectral radius ρ(J_F) < 1 in a reasoning subspace and ρ(J_F) > 1 in a memory subspace, selective iteration improves reasoning convergence while bounded perturbation of memory. Derive the architecture that makes this spectral structure LEARNABLE.

**From Information Geometry:**
- Hidden states live on a statistical manifold. Attention computes a natural gradient step on this manifold. FFN computes a Euclidean step. Iterating the natural gradient (attention) is well-motivated. Iterating the Euclidean step (FFN) is not. What architecture makes this distinction explicit?
- Is there an architecture where the recursive core is constrained to lie on a specific submanifold (e.g., constant Fisher information), guaranteeing stable iteration?

**From Optimal Transport:**
- The second pass moves the hidden state distribution. Can we design an architecture where the second-pass map is the OPTIMAL TRANSPORT map from "first-pass output distribution" to "target output distribution"? This would make recursion provably beneficial.

**From Category Theory / Algebraic Structure:**
- Is there a monoidal structure on transformer layers where iteration corresponds to a well-defined algebraic operation (e.g., a monoid action)? If so, what constraints does this place on the weight matrices?

**From Statistical Mechanics:**
- Treat the FFN as a Hopfield network with energy landscape E(h). The first pass finds a local minimum. The second pass, after attention perturbation, may find a different minimum. Design an architecture where the energy landscape has a hierarchical structure: broad basins for reasoning (stable under iteration) and narrow basins for facts (unstable under iteration, but gated).

**Concrete requirements for the architecture:**
1. **Mathematically motivated** — every design choice follows from a theorem or proof sketch
2. **0.5B-2B parameters** — trainable on 4 B200 GPUs in a week
3. **Must beat a dense baseline of the same size** on at least 3/4 standard benchmarks
4. **Adaptive computation** — variable depth per input
5. **Production-ready** — KV cache compatible, <20% inference overhead
6. **Novel** — genuinely different from Universal Transformer, PonderNet, MoD, TRM, SIRT

Provide:
- Full mathematical derivation of the architecture
- Proof that it has desirable convergence properties under iteration
- Complete PyTorch model definition (~500 lines)
- Training recipe (data mix, stages, hyperparameters)
- Predicted benchmark performance with confidence intervals
- Comparison to every existing recursive/adaptive architecture

**Think like you're writing a NeurIPS best paper.** The architecture should be elegant, theoretically motivated, and empirically validated (at least in expectation based on our data).

### Part 6: What Should the Paper Say?

Given our results, what's the most impactful and honest paper we can write?

Options:
1. **"Layer duplication doesn't work on benchmarks"** — negative result paper
2. **"Layer duplication reveals the attention-FFN asymmetry"** — mechanistic finding
3. **"Proxy metrics don't predict benchmark performance"** — methodological contribution
4. **"SBUID screening enables efficient architecture search"** — tool paper
5. **"Toward adaptive computation in LLMs"** — future-looking vision paper
6. Something else entirely

For each option: what venue would accept it, what's the strength of the contribution, and what additional experiments (if any) would strengthen it?

---

## Papers to Reference and Search For

### Our references:
- Ng's RYS: https://huggingface.co/dnhkng/RYS-XLarge
- TRM: arXiv 2510.04871
- Universal Transformers: arXiv 1807.03819
- PonderNet: arXiv 2107.05407
- Mixture-of-Depths: arXiv 2404.02258
- Relaxed Recursive Transformers: arXiv 2410.20672
- Geva et al. FFN as Memory: arXiv 2012.14913
- ROME: arXiv 2202.05262
- Hopfield Networks: arXiv 2008.02217
- Repeat the Prompt Twice: arXiv 2512.14982
- MobileLLM: arXiv 2402.14905
- Curse of Depth: arXiv 2502.05795
- SOLAR 10.7B: arXiv 2312.15166
- Mixture-of-Recursion: arXiv 2507.10524
- LayerSkip: arXiv 2404.16710

### Search extensively for:
- 2025-2026 papers on test-time compute scaling (s1, o1, reasoning models)
- Papers on the disconnect between proxy metrics and downstream performance
- Papers on loglikelihood scoring vs generation quality in LLMs
- Papers on the fundamental limits of iterative/recursive computation
- Information-theoretic bounds on depth vs performance
- Papers on when more computation helps vs doesn't help in neural networks
- Neuroscience literature on when recurrence helps cognition
- Any negative result papers about layer duplication or depth upscaling

---

## Output Format

For each section, provide mathematical rigor:
- Explicit assumptions stated
- Theorem statements with proofs or proof sketches
- Connection between theory and our specific empirical observations
- Concrete predictions that could be tested

**Be brutally honest.** If layer duplication is fundamentally limited to <1% improvement on standardized benchmarks, say so and explain why. If there's a path to larger gains, show the math. We'd rather know the truth than chase diminishing returns.
