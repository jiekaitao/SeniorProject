# GPT-5.4 Pro Prompt: PSRT-MoR Architecture Design + Novel Directions

## Context: What We've Built and Proven

We are building **PSRT (Projected Split-State Recurrent Transformer)** — a novel architecture where the hidden state is split into a **memory channel** (frozen during recursion) and a **reasoning channel** (iterates toward a fixed point). This directly addresses our empirical finding that attention re-computation helps reasoning but FFN re-computation corrupts factual memory.

### Key Experimental Results (all verified on real hardware, March 2026)

**1. The K-sweep decomposition (Mistral 7B, block [28,29)):**
| K | Full Duplication | Attention-Only | FFN Harm |
|---|-----------------|---------------|----------|
| 1 | 61.76 (baseline) | — | — |
| 2 | 65.26 (+3.50) | 61.43 (-0.33) | +3.83 |
| 3 | 48.10 (-13.66) | 61.23 (-0.52) | -13.13 |
| 4 | 9.62 (-52.13) | 62.85 (+1.09) | -53.23 |

**Finding:** Attention-only duplication is stable at K=4 (+1.09), while full duplication crashes exponentially (-52.13). FFN causes 53 points of damage at K=4. Nobody has published this decomposition.

**2. PSRT-172M training (from scratch):**
- Architecture: Prelude (2 blocks) → Proj_m + Proj_r → Recursive Core (3 shared blocks × K) → Combine → Coda (5 blocks)
- The recurrence: r_{k+1} = (1-α)r_k + α(Core(r_k + m_0) - m_0), memory m_0 frozen
- v1 (fineweb-edu only): K=2 crossed K=1 at step 12000 (delta=-0.64 PPL). But E[K] collapsed to 1.0 on easy text — model learned "never recurse."
- v2 (50% general + 25% math + 25% science): K=2 crossed at step 10000 with delta=-2.17 (3x larger benefit). Harder data makes the model learn recursion faster and use it more.
- At step 12000 v2: delta=-2.13 (stable, holding)

**3. Post-hoc duplication ceiling:**
- Paradigm Shift recipe (OPLoRA + contrastive + gate + α warmup) preserves K=1 perfectly (0.00 delta) but LoRA training never improves beyond raw duplication
- On lm-eval (BBH/MATH/MMLU-Pro/MuSR), gains are 0-1% — confirmed across LLaMA 3 8B and Mistral 7B
- The transfer gap is real: free-form probe improvements don't translate to multiple-choice accuracy (margin-shell bound)

**4. LLM-to-TRM conversion (surgical projection grafting):**
- Add proj_m, proj_r, combine layers around existing core blocks
- Freeze all base weights, train only projections (~67M params on 8B model)
- LLaMA 3 [10,13): failed (block doesn't help even raw)
- Mistral [28,29): results pending but pre-ft K=2 showed math↓ EQ↑ (interesting asymmetry)

**5. Reasoning probe (trick questions like "car wash" problem):**
- Neither prompt duplication nor layer duplication helped 7B models on trick questions
- At 7B scale, the knowledge gap is the bottleneck, not attention — re-computation can't create information the model doesn't have (data-processing inequality)

### Architecture Details

**PSRT model.py** (implemented and training):
```python
class PSRT(nn.Module):
    # Embedding → Prelude (standard blocks)
    # → proj_m(h) = m_0, proj_r(h) = r_0
    # → Recursive Core × K:
    #     h_core = r + m_0
    #     h_core = CoreBlocks(h_core)  # standard transformer blocks
    #     r_new = h_core - m_0
    #     r = (1-α)r + α·r_new
    #     [Halting head: should I keep recursing?]
    # → combine([m_0, r_final]) → Coda (standard blocks) → LM Head
```

**Key property:** Memory m_0 is never modified. The core blocks see (r + m_0) as input, so they have access to factual context, but the subtraction isolates the reasoning update. If Core is ρ-Lipschitz with ρ<1, reasoning converges geometrically to a fixed point.

### Compute Environment
- NVIDIA B200 GPUs (192GB HBM3e each), SLURM cluster
- Can use up to 5 GPUs simultaneously
- Training 172M model takes ~1 hour on 1 GPU
- Can scale to 1.1B (d=2048, 24 blocks) on 1 GPU

---

## What We Want From You

### 1. PSRT-MoR (Mixture of Recursions) Architecture Design

We want to extend PSRT with **multiple recursive core experts** and a learned router:

```
Prelude → split into m + r →
    ┌─ Core-A (expert 1) ─┐
r → ├─ Core-B (expert 2)  ├→ router picks which core(s) to iterate
    └─ Core-C (expert 3) ─┘
→ Combine(m, r) → Coda
```

Design questions:
- How should the router work? Token-level or sequence-level? Top-1 or top-2?
- Should different experts share some blocks (partial sharing) or be fully independent?
- How do we train the router without mode collapse (all inputs going to one expert)?
- Should the number of recursion steps K be per-expert or global?
- What's the right balance of experts × blocks per expert × total params?
- How does this interact with the memory/reasoning split? Should m be expert-specific too?

### 2. Novel Architectural Ideas Beyond MoR

Think deeply about what OTHER novel architectures could exploit the attention-FFN asymmetry. Our key empirical insight is:

> "Attention re-computation refines reasoning (stable at K=4). FFN re-computation corrupts factual memory (exponential degradation)."

What architectures would you propose that are:
- Genuinely novel (not just combining existing ideas)
- Theoretically grounded (with convergence guarantees or information-theoretic arguments)
- Practically trainable at 172M-1.1B scale
- Likely to produce measurable gains on standardized benchmarks (BBH, MATH, MMLU-Pro, MuSR)

We are specifically interested in ideas that go BEYOND what Universal Transformers, PonderNet, Mixture-of-Depths, and Huginn have explored.

### 3. Training Data and Objectives

Our v2 result (delta=-2.17 on math+science data vs -0.64 on general text) proves that training data matters enormously for adaptive computation. What training data mix and objectives would you recommend to:
- Maximize the benefit of recursion (make E[K]>1 on hard inputs)
- Maintain fluency on general text (don't sacrifice K=1 quality)
- Enable the halting head to make good decisions (when to recurse vs not)

### 4. Scaling Strategy

If PSRT-172M works, what's the path to a competitive model?
- How many tokens for 172M? For 1.1B?
- What's the minimum scale where PSRT could beat a dense baseline on 3/4 benchmarks?
- Should we use knowledge distillation from a larger model?

### 5. Paper Framing

Our strongest results are:
1. The K-sweep attention-FFN decomposition (genuinely novel empirical finding)
2. PSRT learning recursion from scratch (K=2 beats K=1)
3. The transfer gap theory (why probes work but benchmarks don't)

How should we frame the paper to maximize impact? What venue (NeurIPS, ICLR, EMNLP)? What title? What should the main claim be?

---

## Constraints
- American models only (no Qwen, DeepSeek, or other Chinese models)
- Max 5 B200 GPUs simultaneously
- Prefer architectures that can show results at 172M-1.1B scale (proof of concept)
- We want NOVEL ideas, not just combining existing techniques
- Include mathematical formalism where it adds clarity
- Be honest about what's likely to work vs speculative

## Please provide:
1. Detailed PSRT-MoR architecture specification with math
2. 2-3 genuinely novel alternative architectures with theoretical grounding
3. Recommended training data mix with specific HuggingFace dataset names
4. A training recipe (phases, hyperparameters, curriculum)
5. Paper framing recommendation
6. Your honest assessment of what's most likely to produce a strong result
