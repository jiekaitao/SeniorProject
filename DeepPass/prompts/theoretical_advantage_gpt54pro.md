# GPT-5.4 Pro Prompt: Tapping the Theoretical Advantage of Continuous-Space Recursion

## The Core Question

Chain-of-thought reasoning in token space (o1, R1, etc.) forces the model to compress intermediate thoughts into discrete tokens from a fixed vocabulary, losing information at every step. Continuous hidden-state recursion operates in the full d-dimensional vector space where the model can maintain nuanced, high-bandwidth intermediate representations impossible to express in tokens.

**Yet token-space reasoning models are winning.** Why? Because each generated token is fed BACK as new input — genuinely new information for attention to process. Our recursion re-processes the same input with the same weights.

**The question:** How do we tap the theoretical bandwidth advantage of continuous-space recursion while also getting the "new information per step" benefit that makes token-space reasoning work? How do we build a model that ACTUALLY thinks harder in vector space and produces measurably better outputs?

Think extra hard about this. We want genuinely novel architectural ideas.

---

## Complete Experimental Record (March 30, 2026)

Everything below is verified on NVIDIA B200 GPUs. No cherry-picking.

### 1. Project Background

**DeepPass** explores layer duplication in LLMs — running transformer blocks multiple times at inference. We started by replicating David Ng's RYS (Repeat Your Self) discovery, then developed spectral screening (SBUID), per-layer alpha tuning, and sublayer analysis across 5 model families (LLaMA 3 8B, Mistral 7B, Gemma3 27B, Qwen2 72B, Qwen3 30B MoE). ~200 GPU experiments total.

**Key constraint:** American models only (no Qwen, DeepSeek). Max 5 B200 GPUs simultaneously.

### 2. The K-Sweep Decomposition (STRONGEST FINDING)

Tested K=1,2,3,4 with three modes on Mistral 7B, block [28,29):

| K | Full Duplication | Attention-Only (FFN=0 on pass 2+) | FFN Harm |
|---|-----------------|-----------------------------------|----------|
| 1 | 61.76 (baseline) | — | — |
| 2 | **65.26 (+3.50)** | 61.43 (-0.33) | +3.83 |
| 3 | 48.10 (-13.66) | 61.23 (-0.52) | -13.13 |
| 4 | **9.62 (-52.13)** | **62.85 (+1.09)** | **-53.23** |

**Finding:** Attention-only duplication is STABLE at K=4 (+1.09). Full duplication crashes exponentially (-52.13). The FFN (memory retrieval) causes 53 points of damage at K=4. This has not been published before.

KL divergence at each K: 0.64, 3.99, 5.80 — representation diverges exponentially.

### 3. Step-Decayed FFN Schedule (TODAY'S RESULT)

Tested monotone-decreasing FFN beta schedules on Mistral [28,29):

| Schedule | Beta per pass | K=2 | K=3 | K=4 |
|----------|--------------|-----|-----|-----|
| full | [1.0, 1.0, 1.0, 1.0] | **+3.50** | -13.66 | -52.13 |
| B | [1.0, 0.25, 0.05, 0.0] | +2.01 | -0.52 | **+1.09** |
| C | [0.8, 0.2, 0.05, 0.0] | +2.01 | -0.52 | **+1.09** |
| D | [1.0, 0.5, 0.1, 0.0] | +2.01 | -0.52 | **+1.09** |
| attn_only | [1.0, 0.0, 0.0, 0.0] | -0.33 | -0.52 | **+1.09** |

**Finding:** Any schedule that decays FFN to zero on later passes completely prevents the K=3+ crash while maintaining K=2 benefits. Front-load FFN (helps at K=2), then decay toward attention-only. All decay schedules converge to the same K=4 result (+1.09) because the last passes are effectively attention-only regardless.

### 4. PSRT Architecture (Projected Split-State Recurrent Transformer)

Novel from-scratch architecture: memory frozen, reasoning iterates.

```
Embedding → Prelude → proj_m(h)=m₀, proj_r(h)=r₀
→ Core × K: r = (1-α)r + α(Core(r+m₀) - m₀)   [memory m₀ FROZEN]
→ Combine([m₀, r]) → Coda → LM Head
```

**PSRT v1 (fineweb-edu only, 172M):**
| Step | Phase | PPL K=1 | PPL K=2 | Delta |
|------|-------|---------|---------|-------|
| 10000 | P1 (K=1 only) | 357.90 | 379.44 | +21.54 |
| 12000 | P2 (K={1,2,3}) | 322.50 | 321.86 | **-0.64** ← crossover |
| 14000 | P2 | 296.75 | 295.71 | **-1.05** |
| Phase 3 | Adaptive halt | — | — | E[K] collapsed to 1.0 (data too easy) |

**PSRT v2 (50% general + 25% math + 25% science, 172M):**
| Step | Phase | PPL K=1 | PPL K=2 | Delta |
|------|-------|---------|---------|-------|
| 8000 | P1 | 422.47 | 440.97 | +18.50 |
| 10000 | P2 | 380.06 | 377.89 | **-2.17** ← faster crossover |
| 12000 | P2 | 348.79 | 346.66 | **-2.13** |
| 14000 | P2 | 318.96 | 316.99 | **-1.96** |
| Phase 3 | Adaptive halt | — | — | E[K] drifted to 1.03 (better than v1 but still low) |

**Key result:** Harder data (math+science) made K=2 cross over earlier (step 10K vs 12K) with 3x larger benefit (-2.17 vs -0.64). But adaptive halting still trends toward K=1.

### 5. PSRT-MoR-lite (TRAINING RIGHT NOW)

Shared attention + 3 expert-specific FFN branches with monotone decay schedules:
- Expert 1 (reason-refine): beta = [0.25, 0.10, 0.02, 0.00]
- Expert 2 (math-single-refresh): beta = [0.80, 0.20, 0.05, 0.00]
- Expert 3 (safe-fluency): beta = [0.10, 0.02, 0.00, 0.00]

Sequence-level top-2 router. Training in 3 phases: A (uniform routing) → B (soft routing) → C (top-2 + halting).

**Current results (step 6600, Phase B):**
- Loss: 10.9 → 5.70
- Eval at step 4000: PPL K=1=1780, K=2=1877 (delta=+97, K=2 still worse)
- Eval at step 6000: PPL K=1=1468, K=2=1522 (delta=+54, gap narrowing)

**CRITICAL FINDING — Expert specialization is emerging:**
```
K=2 inputs → Expert 2 dominates (route ~[0.16, 0.81, 0.02])
K=3 inputs → Experts 1+3 share  (route ~[0.45, 0.04, 0.50])
```
The router learned ON ITS OWN that K=2 should use the high-FFN expert while K=3 should use the low-FFN experts. This validates the attention-FFN asymmetry finding architecturally.

### 6. LLM-to-TRM Conversion (Surgical Projection Grafting)

Add proj_m, proj_r, combine layers (~67M trainable params) to frozen pre-trained models:

| Config | K=2 | K=3 | Notes |
|--------|-----|-----|-------|
| Mistral [28,29) 500 steps fineweb | **+0.62** | **+0.68** | Best — first positive trained K=2 on pre-trained model |
| Mistral [28,29) 500 steps (run 2) | +0.33 | +0.68 | Reproducible K=3 result |
| Mistral [28,29) 1500 steps | +0.09 | -0.32 | More training overshoots |
| Mistral [27,30) wide | -1.17 | -1.16 | Wrong block |
| Mistral [28,29) 500 steps hard mix | -2.38 | -0.40 | Hard data hurts TRM projections |
| LLaMA 3 [10,13) 500 steps | -14.61 | -11.06 | Block doesn't help even raw |

**Key insight:** TRM projections train best on general text (learning the memory/reasoning split), not task-specific data. The sweet spot is exactly 500 steps — more overshoots.

### 7. Paradigm Shift (5-Intervention Recipe)

Pass-2-only OPLoRA + contrastive + gate + alpha warmup + FFN whisper:
- K=1 preserved with **0.00 delta** on every model, every time
- But LoRA training NEVER improved beyond raw duplication
- CE loss optimization doesn't translate to generation quality

### 8. lm-eval Baseline (Mistral 7B, full benchmarks)

Baseline completed, K=2 pass still running:
- BBH: 46.90%
- MATH-Hard: 3.93%
- MMLU-Pro: 30.24%
- MuSR: 44.97%

K=2 results pending (~30 min).

### 9. Reasoning Probe (Trick Questions)

12 questions (car wash, bat+ball, lily pad, surgeon riddle):
- LLaMA K=2: **+4.2%** (improved)
- Mistral K=2: -12.5% (hurt)
- Car wash: wrong on ALL configs across ALL models
- At 7B scale, knowledge gaps are the bottleneck, not attention

### 10. LLaMA Step-Decay (RUNNING NOW)

Cross-model validation on LLaMA 3 8B [10,13). K=2 results so far show the FFN scaling hooks have a bug on multi-layer cores (fire counts don't reset between generation tokens). Mistral [28,29) results are clean because it's a single-layer core.

---

## What We Know Is True

1. **Attention re-computation is stable at any depth.** K=4 attention-only = +1.09 on Mistral.
2. **FFN re-computation corrupts exponentially.** K=4 full = -52.13. This is memory retrieval amplifying errors.
3. **Step-decayed FFN prevents the crash.** Any decay schedule that reaches zero prevents K=3+ degradation.
4. **PSRT's memory/reasoning split works.** K=2 beats K=1 when trained from scratch.
5. **Expert specialization emerges naturally.** MoR-lite router learns to use high-FFN experts for K=2 and low-FFN experts for K=3.
6. **CE loss optimization ≠ generation quality.** Every LoRA/fine-tuning attempt that reduced CE failed to improve generation.
7. **Post-hoc duplication has a 0-1% ceiling on lm-eval.** Margin-shell bound and data-processing inequality.
8. **Surgical TRM conversion can give +0.68** on the right model/block in 14 seconds.

## The Theoretical Advantage We Haven't Tapped

Chain-of-thought in token space: each step compresses to discrete tokens (lossy), but each token is genuinely new input (new information).

Continuous recursion in hidden-state space: each step operates in full d-dimensional space (lossless), but re-processes the same input with the same weights (no new information — data-processing inequality).

**We want architectures that combine:**
- The high bandwidth of continuous hidden-state space
- The "new information per step" property of token-space reasoning
- The memory protection of PSRT's split-state design
- The expert specialization of MoR-lite's routing

## What We Want From You

### 1. Novel Architectures That Tap the Continuous Bandwidth Advantage

Think deeply. The key constraint is the data-processing inequality: deterministic re-computation of the same input can't create new information. How do we break this? Ideas to explore:

- **Stochastic recursion:** Add learned noise/sampling to the reasoning channel so each iteration explores a different region of representation space
- **Cross-attention to external memory banks** that provide genuinely new information per step
- **Feedback connections** where the output of iteration k modifies the INPUT for iteration k+1 (like token-space CoT but in continuous space)
- **Multi-resolution reasoning** where different recursion steps operate at different granularities
- **Reasoning-conditioned re-reading** where the refined reasoning state changes HOW attention operates on the original input (essentially letting the model "re-read" the input with new eyes)

### 2. How to Make Adaptive Halting Actually Work

Both PSRT v1 and v2 had E[K] collapse toward 1.0 in Phase 3. The model always chooses "don't recurse." Why? And how do we fix it without artificial penalties?

### 3. How to Bridge the CE-to-Generation Gap

Every time we optimize CE loss, generation quality doesn't improve (or gets worse). This is the fundamental objective mismatch. What training objective would actually make recursion improve downstream generation?

### 4. Scaling Strategy

If we had unlimited compute, what's the architecture and training recipe that would produce a PSRT-MoR model that beats a dense baseline of the same size on BBH, MATH, MMLU-Pro, and MuSR? What's the minimum scale where this is realistic?

### 5. Paper Framing

Our strongest contributions:
1. K-sweep attention-FFN decomposition (novel empirical law)
2. Step-decayed FFN schedule (practical fix, zero cost)
3. PSRT architecture with memory/reasoning split (learns recursion)
4. MoR-lite expert specialization (experts learn the asymmetry on their own)

How should we frame this for maximum impact? What venue? What title?

## Constraints
- American models only
- 5 B200 GPUs, 192GB each
- Prefer 172M-1.1B scale proofs-of-concept
- We want GENUINELY NOVEL ideas, not just combining existing techniques
- Mathematical formalism where it adds clarity
- Be honest about what's speculative vs well-grounded

## Please provide:
1. 3-5 genuinely novel architectural ideas for continuous-space reasoning that break the data-processing inequality bottleneck
2. Mathematical analysis of why each idea could work
3. Concrete implementation specifications we can build this week
4. A training recipe for each
5. Your honest ranking of which is most likely to produce a breakthrough
6. Paper framing recommendation
