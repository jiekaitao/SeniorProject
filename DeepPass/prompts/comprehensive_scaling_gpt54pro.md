# GPT-5.4 Pro: Comprehensive Prompt — Scaling Recursion Fine-Tuning to SOTA

**IMPORTANT: This is a fresh context window. All information needed is in this prompt. Search the web extensively for recent papers and implementations. Think with mathematical rigor — provide proofs or proof sketches, not hand-waving.**

---

## Who We Are

Senior project research team at University of Florida. Access to NVIDIA B200 GPUs (192GB HBM3e). Two weeks of intensive experimentation on layer duplication and recursive computation in LLMs. We want to push our recursion fine-tuning technique to achieve SOTA or near-SOTA on the 8B model class.

---

## The Technique: Layer Duplication + Recursion Fine-Tuning

### How it works

Given a pre-trained LLM with N layers, we:
1. **Designate a "recursive core"** — a contiguous block of layers [core_start, core_end)
2. **Duplicate the core K times** — the execution order becomes: [0..core_start, core×K, core_end..N]
3. **Fine-tune with K=1-3 curriculum** — freeze all except core layers, train 300-500 steps at lr=5e-7

The duplicated layers **share weights** (zero extra VRAM). A `LayerIdxWrapper` gives each physical position a unique `layer_idx` for KV cache compatibility.

### The key: after 300 steps of gentle fine-tuning, K=2 (running core twice) beats K=1 (baseline) on benchmarks.

---

## ALL Our Experimental Results

### 1. Cross-Architecture Layer Duplication (Raw, No Fine-Tuning)

Dual probe (math guesstimate + EQ-bench, combined score ~0-100):

| Model | Params | Layers | Attn Type | Baseline | Best Dup Config | Best Δ |
|-------|--------|--------|-----------|----------|----------------|--------|
| LLaMA 3 8B | 8B | 32 | Full | 65.12 | (4,5) | +0.25 |
| LLaMA 3.1 8B | 8B | 32 | Full | 45.53 | (21,22) | +1.21 |
| Mistral 7B | 7B | 32 | Sliding window | 39.03 | (28,29) | +2.96 |
| Gemma 2 9B | 9B | 42 | Full | 41.17 | (5,6) | +6.11 |
| Gemma 3 27B | 27B | 62 | Sliding window | 80.54 | Triple (0,2)+(12,13)+(47,48) | +7.26 |
| LLaMA 3 70B | 70B | 80 | Full | 76.73 | Pair (10,11)+(61,62) | +6.55 |

### 2. lm-eval Standardized Benchmarks (15% subsample, BBH+MATH+MMLU-PRO+MuSR)

**Raw duplication (no fine-tuning) on lm-eval:**

| Model | Config | BBH Δ | MATH Δ | MMLU Δ | MuSR Δ |
|-------|--------|-------|--------|--------|--------|
| **Mistral 7B** | **(28,29) full** | **+0.7%** | **+1.0%** | **-0.5%** | **+2.6%** |
| Mistral 7B | (28,29) whisper β=0.2 | +0.3% | **+2.5%** | -0.1% | **+2.6%** |
| Mistral 7B | (28,29) attn-only | +0.3% | +0.5% | **+0.1%** | +1.7% |
| LLaMA 3 8B | (4,5) | +0.1% | -5.4% | -4.1% | -0.9% |
| LLaMA 3.1 8B | (21,22) | -1.6% | -5.0% | -1.0% | -0.9% |
| LLaMA 3 70B | (10,11) | -0.2% | -2.0% | -1.9% | -0.9% |
| Gemma 3 27B | Triple @1.0 | -3.4% | -1.0% | -4.8% | 0.0% |
| Gemma 3 27B | Attn-only triple | -1.5% | -1.0% | -3.2% | 0.0% |
| Gemma 3 27B | Pair (no L47) | -2.4% | -1.0% | -3.3% | +0.9% |
| Gemma 3 27B | Single (12,13) + whisper FFN | -2.7% | -1.0% | -1.7% | +1.7% |

**Key observation:** Mistral 7B is the ONLY model where raw duplication improves lm-eval. Whisper FFN (β=0.2) is the best variant: MATH +2.5%, MuSR +2.6%, MMLU essentially flat.

### 3. Recursion Fine-Tuning Results (Dual Probe)

| Model | Core Layers | Steps | LR | Baseline K=1 | Post-FT K=1 | Post-FT K=2 | K=2 Δ |
|-------|-------------|-------|-----|-------------|-------------|-------------|-------|
| **Gemma 3 27B** | [11,14] | 300 | 1e-6 | 76.96 | **76.96 (0.00)** | **81.08** | **+4.11** |
| **LLaMA 3 8B v3** | [10,13] | 300 | 5e-7 | 63.76 | 61.16 (-2.60) | **66.98** | **+3.22** |
| LLaMA 3 8B v2 | [10,13] | 500 | 1e-6 | 63.76 | 63.91 (+0.14) | 62.96 | -0.80 |
| LLaMA 3 8B v4 | [10,13] | 300 | 5e-7 | 63.76 | 61.54 (-2.23) | 65.58 | +1.81 |
| LLaMA 3 8B whisper | [10,13] | 500 | 5e-7 | 63.76 | 60.71 (-3.05) | 64.88 (full) | +1.12 |
| SmolLM2-360M | [10,16] | 1000 | 5e-6 | 1.97 | 2.99 (+1.01) | 20.29 | +18.31 |
| SmolLM2-1.7B | [8,14] | 500 | 5e-7 | 22.41 | 22.09 (-0.32) | 3.21 | -19.20 |
| Mistral 7B | [27,30] | 500 | 1e-6 | 60.86 | 61.35 (+0.49) | 0.00 | -60.86 |

**Key patterns:**
- Gemma 3 27B: K=1 perfectly preserved AND K=2 +4.11. The ideal result.
- LLaMA 3 8B v3: K=2 +3.22 but K=1 drops -2.60. Trade-off.
- v2 (lr=1e-6) preserved K=1 but K=2 didn't improve. v3 (lr=5e-7) improved K=2 but hurt K=1.
- Whisper FFN during fine-tuning killed K=2 performance (13.65). Don't scale FFN during training.
- Larger SmolLM (1.7B) completely broke. Mistral (sliding window) broke at K=2.

### 4. Recursion Fine-Tuning on lm-eval (The Key Result)

**LLaMA 3 8B (core [10,13], 300 steps, lr=5e-7, K=2):**

| Benchmark | Baseline | FT K=2 | Delta |
|-----------|----------|--------|-------|
| BBH | 47.5% | **48.4%** | **+0.9%** |
| MATH Hard | 9.9% | 8.9% | -1.0% |
| MMLU-PRO | 27.7% | **27.8%** | **+0.1%** |
| MuSR | 34.8% | **38.3%** | **+3.5%** |

**3 of 4 benchmarks improved. MuSR +3.5% is substantial.**

### 5. SBUID Screening Metric

`SBUID = BLOOD_impact - λ × displacement_rho`

| Model | Best λ | Spearman r | p-value | Significant? |
|-------|--------|-----------|---------|-------------|
| LLaMA 3 70B | 10k | **0.668** | **0.001** | Yes |
| Qwen2-72B | 6k | 0.52 | 0.008 | Yes |
| Gemma 3 27B | any | -0.075 | 0.567 | No (sliding window) |
| Small models | any | ~0.0 | ~1.0 | No |

162x speedup over brute-force on compatible architectures.

### 6. Neuron-Level Analysis (Gemma 3 27B)

Four independent methods converged:

| Method | L0 | L1 | L12 | L47 |
|--------|----|----|-----|-----|
| DLA (logit attribution) | Neutral (0.00) | Keep (+1.62) | Keep (+0.99) | Remove (-0.05) |
| GEM eigenmask | Off | On | On | Off |
| Gate flip rate | 37% (unstable) | 3% (stable) | 0.3% (stable) | 0.7% (stable but harmful) |
| Causal patching | +0.96 | — | **+1.09** | **-2.06** |

**Layer 47 is unanimously harmful. Layer 12 is unanimously helpful.**

### 7. Attention-FFN Asymmetry

| Model | FFN Impact (attn_only - full_dup) | Interpretation |
|-------|----------------------------------|----------------|
| Gemma 3 27B L12 | -1.33 | FFN helps |
| Gemma 3 27B L47 | -0.94 | FFN helps (but layer itself is harmful) |
| LLaMA 3 70B (10,11) | -1.70 | FFN helps |
| Mistral 7B (28,29) | -2.85 | FFN strongly helps |
| SmolLM-135M L6 | -2.82 | FFN strongly helps |

FFN helps on most models. The "FFN bad" hypothesis is wrong for most architectures. The correct framing: **some FFN is needed to process attention output, but full FFN re-retrieval can corrupt factual memory. β=0.2 is the sweet spot on Mistral.**

### 8. SIRT-172M Architecture (Novel, Trained From Scratch)

3-zone design: Prelude (3 blocks) → Recursive Core (3 shared blocks, K=1-4) → Coda (4 blocks).

Token-wise β gate: `β = β_max × σ(bias + router(hidden_state, margin, stability))`

Training results (705M tokens, 3 stages):
- Stage 1 (K=1): loss 11→5.4, β learned ~0.13
- Stage 2 (K=1-3 curriculum): loss 5.4→5.0
- Stage 3 (adaptive halting): E[K]=1.11, loss 5.0→4.99
- SIRT K=1 beats dense baseline: 9.60 vs 8.60 on dual probe
- But K>1 doesn't help (insufficient training data)

### 9. KV Cache Fix

`LayerIdxWrapper` temporarily swaps `layer_idx` during forward pass. Shared-weight layers get unique cache slots. Zero extra VRAM. Tested on Gemma 3 27B: correct output, 1.3x speedup.

---

## Scaling Experiments (COMPLETED)

### LLaMA 3 8B: Wider core, longer training, dual zones

| Config | Core | Steps | K=2 Combined | Delta vs Baseline |
|--------|------|-------|-------------|-------------------|
| **v3 (BEST)** | **[10,13] (3 layers)** | **300** | **66.98** | **+3.22** |
| Long training | [10,13] | 1000 | 65.84 | +2.08 |
| Wide core | [8,16] (8 layers) | 500 | 50.05 | -13.71 |
| Dual zones | [4,7]+[10,13] | 500 | 61.91 | -1.85 |
| Whisper FFN (β=0.2 during FT) | [10,13] | 500 | 13.65 | -50.11 |
| Whisper FFN (full FFN at eval) | [10,13] | 500 | 64.88 | +1.12 |

**Critical findings:**
1. **300 steps beats 1000 steps** — more training hurts. The model overfits to recursion and damages K=1.
2. **Narrow core (3 layers) beats wide core (8 layers)** — wider core = too many parameters changing = catastrophic forgetting.
3. **Dual zones hurt** — running two separate recursive blocks causes interference.
4. **Whisper FFN during training kills everything** — don't scale FFN during fine-tuning. Only use whisper at inference.
5. **The sweet spot is extremely narrow: [10,13], 300 steps, lr=5e-7, K=1-3 curriculum.**

### LLaMA 3 8B Whisper FFN lm-eval (full FFN K=2 post-ft)

Running lm-eval on the whisper-trained model evaluated with full FFN at K=2. Results pending.

### Gemma 3 27B ft K=2 lm-eval

Running at 93/202 generate_until. This is our cleanest recursion result (+4.11 dual probe, K=1 preserved). lm-eval will determine if it translates to standardized benchmarks.

---

## The Specific Problem

We want to achieve **universal improvement** (all 4 benchmarks up, or 3 up + 1 flat) on the **8B model class**.

Current best on LLaMA 3 8B: BBH +0.9%, MuSR +3.5%, MMLU +0.1%, but MATH -1.0%.

The MATH regression needs to be fixed. It's the one benchmark that consistently drops.

### Why MATH drops

MATH Hard requires precise numerical computation. Layer duplication perturbs the FFN's memory retrieval, which stores arithmetic patterns. The second pass through the FFN re-retrieves with slightly different gate activations (our Jaccard analysis shows 6-37% gate flip rates), leading to different (wrong) arithmetic retrievals.

### Why MuSR improves

MuSR tests multi-step reasoning (murder mysteries, object placement, team allocation). These require iterative attention refinement — exactly what layer repetition provides. The model gets another chance to aggregate information across the context.

---

## What We Need

### Part 1: Mathematical Analysis

**1a. Formalize the recursion fine-tuning objective.** Given a pre-trained model M with parameters θ, recursive core layers [c_s, c_e), and recursion count K:

- Define the optimal fine-tuning objective that MAXIMIZES K=2 performance while CONSTRAINING K=1 to stay within ε of baseline
- Prove (or show conditions for) when this constrained optimization has a non-trivial solution
- Show how the Jacobian spectral radius of the core block relates to recursion benefit

**1b. Why does LR matter so much?** lr=1e-6 preserves K=1 but K=2 doesn't improve. lr=5e-7 improves K=2 but K=1 drops. Formally characterize this trade-off.

**1c. Why does Gemma 3 27B preserve K=1 perfectly but LLaMA 3 8B doesn't?** What architectural property (layer count, width, attention mechanism) determines whether fine-tuning damages K=1 performance?

### Part 2: Five Novel Approaches to Fix MATH Regression

Each approach must:
- Have a mathematical justification (theorem or proof sketch)
- Include PyTorch pseudocode
- Estimate compute cost (in GPU-hours on a B200)
- Predict expected BBH/MATH/MMLU/MuSR deltas
- Identify failure modes and how to detect them

**Approaches should span these paradigms:**

1. **Spectral regularization** — constrain the Jacobian of core layers during fine-tuning to prevent arithmetic basin-crossing. Mathematical foundation: Lyapunov stability theory applied to discrete dynamical systems (the forward pass as a map).

2. **Contrastive recursion** — compute L(K=1) and L(K=2) on each batch. Only update when K=2 has lower loss. This prevents gradients from K=2 failures from corrupting the weights. Mathematical foundation: natural policy gradient / REINFORCE with baseline.

3. **LoRA per recursion depth** — don't fine-tune base weights at all. Add rank-8 LoRA adapters that activate ONLY during the second pass. The first pass is unmodified (K=1 preserved by construction). Mathematical foundation: Relaxed Recursive Transformers (arXiv 2410.20672).

4. **Task-conditional recursion** — train a tiny classifier (2-layer MLP on first-pass hidden states) that predicts "should we recurse on this input?" Based on our finding that MuSR benefits but MATH doesn't, route inputs accordingly.

5. **Residual recursion with α warmup** — instead of full layer duplication, add only α×(F(h) - h) on the second pass, where α starts at 0.01 and warms up during fine-tuning. This is our "whisper" approach but applied at the FULL LAYER level during training, not just FFN.

### Part 3: MATH-Specific Improvements

MATH Hard requires:
- Arithmetic precision (FFN-dependent)
- Multi-step computation (attention-dependent)
- Format compliance (instruction following)

Propose 3 approaches specifically for improving MATH without hurting other benchmarks:

1. Should we use math-specific training data for fine-tuning?
2. Should we recurse different layers for MATH vs other tasks?
3. Should we apply different β for MATH-like inputs at inference time?

### Part 4: Scaling Beyond 8B

If recursion fine-tuning works at 8B, how do we scale to 70B?
- Our raw duplication on LLaMA 3 70B gives +6.55 on dual probe but -1.3% avg on lm-eval
- The 70B model's VRAM is tight (140GB) — recursion adds layers
- How many tokens of fine-tuning are needed at 70B scale?

### Part 5: Concrete Implementation Plan for LLaMA 3 8B

Give us the **exact recipe** to maximize all 4 benchmarks:

```python
config = {
    'model': 'Meta-Llama-3-8B-Instruct',
    'core_start': ???,
    'core_end': ???,
    'K': ???,
    'lr': ???,
    'steps': ???,
    'warmup_steps': ???,
    'grad_clip': ???,
    'freeze_strategy': ???,  # "all_except_core", "lora", "adapter"
    'beta_ffn': ???,  # None, 0.2, learned
    'curriculum': ???,  # weights for K=1,2,3
    'training_data': ???,  # "general", "math_heavy", "mixed"
    'regularization': ???,  # None, "spectral", "distillation"
}
```

---

## Papers to Reference (and Search For More)

### Layer Duplication / Depth
- David Ng, RYS: https://huggingface.co/dnhkng/RYS-XLarge, https://dnhkng.github.io/posts/rys/
- SOLAR 10.7B (depth upscaling + fine-tuning)
- alainnothere/llm-circuit-finder: https://github.com/alainnothere/llm-circuit-finder

### Recursive / Adaptive Computation
- Universal Transformers (arXiv 1807.03819)
- PonderNet (arXiv 2107.05407)
- Relaxed Recursive Transformers (arXiv 2410.20672) — LoRA per depth
- Mixture-of-Recursion (arXiv 2507.10524)
- LayerSkip (arXiv 2404.16710)
- Mixture-of-Depths (arXiv 2404.02258)
- Adaptive Computation Time (Graves, arXiv 1603.08983)
- TRM (arXiv 2510.04871) — recursive reasoning 7M params

### FFN / Interpretability
- Geva et al. FFN as Key-Value Memories (arXiv 2012.14913)
- ROME / Locating Factual Associations (arXiv 2202.05262)
- Hopfield Networks (arXiv 2008.02217)
- 3Blue1Brown "How LLMs Store Facts": https://youtube.com/watch?v=wjZofJX0v4M

### Small Model Architecture
- MobileLLM (arXiv 2402.14905) — deep thin sub-1B
- SmolLM2 (arXiv 2502.02737)
- The Curse of Depth (arXiv 2502.05795)

### Efficient Fine-Tuning
- LoRA (arXiv 2106.09685)
- QLoRA (arXiv 2305.14314)

**SEARCH THE WEB for any 2025-2026 papers on:**
- Fine-tuning for recursive/iterative computation
- Depth upscaling results on LLaMA 3
- Benchmark improvements on 8B models through architectural tricks
- Preventing catastrophic forgetting during architecture modification
- MATH benchmark-specific improvements
- Contrastive training objectives for LLMs

---

## Output Format

For each of the 5 approaches in Part 2:

```
## Approach N: [Name]

### Theorem / Mathematical Foundation
[Formal statement with proof or proof sketch. Include explicit assumptions.]

### Algorithm (PyTorch pseudocode)
[Precise enough to implement directly. Include training loop, loss computation, hooks.]

### Computational Cost
[GPU-hours on B200. Number of forward passes. Memory overhead.]

### Predicted Benchmark Deltas
[BBH: +X%, MATH: +Y%, MMLU: +Z%, MuSR: +W%. With confidence intervals.]

### Failure Modes
[What could go wrong. How to detect it within 50 steps. What to do if it fails.]

### Connection to Literature
[Which papers support this approach. What's novel vs existing.]
```

**Think deeply. Explore breadth-first — consider 10+ approaches before selecting the top 5. Show your mathematical reasoning. We prefer rigorous analysis with explicit assumptions over hand-waving. Include proofs where possible.**
