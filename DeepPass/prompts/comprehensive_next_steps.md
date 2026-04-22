# Comprehensive Research Synthesis: Design the Optimal Recursive LLM Architecture

## What I Need From You

We have spent two weeks and ~200 GPU-hours systematically investigating how to make LLMs benefit from recursive internal computation — re-reading their input, refining their hidden states across multiple passes before producing output. We have accumulated a rich and sometimes contradictory body of evidence across four major experimental tracks. **We now need you to synthesize everything, identify what we're missing, and design the optimal architecture that fully exploits our findings.**

Specifically, I need:

1. **A complete synthesis** of all our evidence — what works, what doesn't, and why
2. **A single unified architecture design** that maximizes the benefit of recursive computation in LLMs, with proof-level mathematical justification for every design choice
3. **A concrete training recipe** — data, loss, schedule, what to freeze, what to train
4. **Formal analysis** of the expected improvement bounds and under what conditions the architecture provably helps
5. **The minimal decisive experiment** to validate the design before scaling up
6. **Creative thinking** — we may be framing the problem wrong. Challenge our assumptions.

We have 4 NVIDIA B200 GPUs (192GB each) available for parallel experiments. Think deeply and verify your reasoning rigorously before responding.

## Complete Research Timeline and Evidence

### Track 1: Runtime Layer Duplication on Pretrained 72B Model

**Setup:** Take a pretrained Qwen2-72B (80 layers), duplicate specific layer blocks at inference time. No training, no weight changes. Just run layers [i,j) twice.

**Key results:**
- Naive duplication of layers 45-52: +6.24 combined score (math + EQ-bench)
- Per-layer alpha tuning (alpha blending at the seam): **+7.31** combined score
- Greedy multi-block stacking with alpha: up to +5.82
- Bayesian optimization over alpha: +7.21 (matching grid search in 60 vs 300 evals)

**Sublayer analysis (the foundational finding):**
- IFEval (instruction following): **+2.3%** from duplication
- MuSR (multi-step reasoning): **+1.3%**
- MATH (math problem solving): **-6.4%**
- BBH, MMLU-Pro: ~flat

**Root cause identified:** Attention repetition helps (iterative message passing refines understanding). FFN/MLP repetition hurts (shifts hidden states away from correct factual retrieval basins — "overshooting" the stored associations).

**Gemma3-27B cross-architecture validation:**
- Quad block duplication: combined score 85.58
- Confirms the effect generalizes across model families

### Track 2: TRM — Tiny Recursive Model (from-scratch, 7M params)

**Setup:** A ~7M parameter model trained from scratch on hard puzzle tasks (30×30 maze solving, ARC pattern induction). Genuine recursion with shared weights.

**Architecture:**
```python
# Two recurrent states: z_H (high-level), z_L (low-level)
# Same L_level blocks run repeatedly with input injection every step
for H_step in range(H_cycles):          # typically 3
    for L_step in range(L_cycles):       # typically 6
        z_L = L_level(z_L, z_H + input_embeddings)  # L refines, gets raw input every time
    z_H = L_level(z_H, z_L)             # H refines using L

output = lm_head(z_H)
```

**Config:** hidden_size=512, 2 L-layers, H_cycles=3, L_cycles=6, 8 heads, non-causal attention, ACT with Q-learning for halting.

**Key results:**
- Baseline: 9.10% exact maze accuracy, 97.94% token accuracy
- Full training modifications: **30.80% exact accuracy** (3.4× baseline)
- Reduced-MLP ablation (attention-only in early L-level blocks): **works well** — validates "repeat attention, reduce MLP"
- Displacement-based contraction rate: ~0.84-0.86 (converging on attracting manifold despite globally expansive Jacobian with ρ > 1)
- Spectral analysis: the model finds an attracting submanifold; convergence is NOT global but trajectory-specific

**Why TRM works (our analysis):**
1. Full input injection every step (no information bottleneck)
2. Shared weights = true fixed-point iteration
3. Non-causal (bidirectional) attention
4. Hard tasks where recursion genuinely helps (mazes, pattern induction)
5. Two-level hierarchical refinement (z_H, z_L)
6. Gradient truncation: only last H-cycle gets gradients

### Track 3: ARR-PSRT — From-Scratch Recursive Transformer (1.7B params)

**Setup:** Train a 1.7B parameter recursive transformer from scratch for next-token prediction. Split state (m₀ frozen memory + r evolving reasoning), prompt bank (16 compressed tokens), 3 expert FFNs with beta decay, scratchpad memory.

**Complete version history (v2-v16):**

| Phase | Versions | Key Finding |
|-------|----------|-------------|
| NaN crisis | v2-v13 | Scratchpad was unbounded additive integrator → overflow bfloat16. Fixed with RMSNorm + decay + bounded writes. Zero-beta expert created 0×∞=NaN tripwire. |
| Phase A/B curriculum | v14-v15c | Phase A (freeze backbone, train experts) then Phase B (unfreeze) always failed. Experts trained on frozen features couldn't adapt when backbone shifted. |
| Joint training | v16-v16b | All params from step 0. K=2 advantage appeared (delta up to -86 PPL) but oscillated. |

**The fatal flaw (formally proven by prior GPT analysis):**
- ARR's prompt bank (16 tokens) creates a **hard rank bottleneck**: max 512 dimensions of fresh info per reread (16 tokens × 32 heads), vs 2048 full-rank from dense self-attention
- The scratchpad carries zero new prompt information: `I(S_t; X | r_0, m_0, B) = 0` (provable by induction since S_0 is input-independent)
- 3 expert FFNs consume 3× FFN parameter budget in the replay core — spending parameters on the exact thing that hurts
- **Result: 4.3× worse PPL than dense baseline** despite K=2 helping within ARR

### Track 4: DAR — Dense Attention Replay on Pretrained Llama 3.1 8B

**Setup:** Freeze a pretrained Llama 3.1 8B. Add trainable replay components on middle layers. Compare K=1 (base model) vs K=2 (with replay).

#### Phase 1: Gate-only replay (20K trainable params)
Re-run the exact same frozen layer with a learned gate on the delta.

| Layers | Delta PPL | Notes |
|--------|-----------|-------|
| L12-15 | -0.07 | Best single band |
| L8-11 | -0.05 | Works |
| L16-19 | -0.05 | Works |
| L20-27 | -0.03 | Weakest |
| L8-19 (12 layers) | -0.05 | More layers ≠ better with gates |
| All 32 | 0.00 | Full model replay = no benefit |
| K=3 L12-15 | -0.06 | 3rd pass helps slightly |

**Dense containment confirmed:** K=1 PPL = 8.74 unchanged across ALL experiments. Zero cost when replay is off.

#### Phase 2: LoRA replay (~2-10M trainable params)
Apply PEFT LoRA (rank 32) to Q/K/V/O projections on replay layers. Train with adapters ON, eval with adapters ON (K=2) vs OFF (K=1).

| Layers | Rank | Delta PPL | Notes |
|--------|------|-----------|-------|
| L12-15 | 32 | -0.41 | 6× better than gate-only |
| L12-15 | 64 | -0.41 | More rank doesn't help at 4 layers |
| L8-19 | 32 | -0.45 | Wider band helps with LoRA |
| **L0-19** | **32** | **-0.50** | **Best 20-layer config** |
| L4-23 | 32 | -0.48 | Skip edges slightly worse |
| **L0-23** | **32** | **-0.51** | **Best overall (tied)** |
| **All 32** | **32** | **-0.51** | **Best overall (tied)** |
| L0-19 | 64 | -0.47 | Higher rank HURTS (overfitting) |
| L8-19 seed 2 | 32 | -0.45 | Reproducible |
| L8-19 seed 3 | 32 | -0.45 | Reproducible |
| L12-15 attn+FFN | 32 | -0.42 | FFN LoRA neutral (not harmful at this scale) |
| L12-15 math-only | 32 | -0.41 | Same delta on math vs mixed data |

**Ceiling: -0.51 PPL (5.8% improvement). Reached at 20-32 layers, rank 32.**

#### Phase 3: LM-Eval Benchmarks (L12-15 r32 checkpoint, 200 samples/task)

| Task | Baseline | LoRA Replay | Delta |
|------|----------|------------|-------|
| ARC-Challenge | 0.540 | 0.510 | **-0.030** |
| HellaSwag | 0.535 | 0.540 | **+0.005** |
| WinoGrande | 0.740 | 0.745 | **+0.005** |
| GSM8K | 0.000 | 0.000 | 0.000 |
| MMLU | 0.664 | 0.656 | **-0.009** |
| MMLU Formal Logic | 0.421 | 0.452 | **+0.032** |
| MMLU Jurisprudence | 0.732 | 0.769 | **+0.037** |

**Pattern: reasoning tasks improve (formal logic +3.2%, jurisprudence +3.7%), knowledge tasks regress (ARC -3.0%, MMLU -0.9%). Same as 72B.**

Note: This was evaluated with the L12-15 checkpoint (delta=-0.41), not the best L0-19 config (delta=-0.51). Better checkpoint should show stronger benchmark gains.

## The Consolidated Findings

### What We Know Works
1. **Attention repetition improves reasoning** — confirmed at 72B (runtime duplication), 7M (TRM), and 8B (LoRA replay)
2. **FFN/MLP repetition hurts factual recall** — confirmed at 72B (sublayer analysis), partially at 8B (attn-only LoRA ≈ attn+FFN LoRA, neither helps factual benchmarks)
3. **Full input injection every cycle** — TRM's key design: no lossy compression
4. **Shared-weight iteration converges on an attracting manifold** — TRM spectral analysis: ρ_displacement < 1 despite ρ_perturbation > 1
5. **Dense containment is essential** — gates=0 must give exact base model. Proven critical by ARR failure.
6. **LoRA differentiation is 6× more effective than same-weight replay** — gate-only: -0.07, LoRA: -0.41. The replay pass needs different attention weights.
7. **20-24 layers is the optimal replay band** for Llama 8B — wider is slightly better or equal, narrower is worse
8. **Rank 32 is the sweet spot** — rank 64 overfits

### What We Know Doesn't Work
1. **Training recursive LLMs from scratch** to beat dense on average PPL — proven by ARR (4.3× worse than dense)
2. **Compressed prompt banks** — hard rank bottleneck (16 tokens = 512 dims max), formally proven insufficient
3. **Split state (m₀/r) architectures** — ARR's split state was provably degenerate (combine layer learns to ignore r)
4. **Expert FFN routing in the replay core** — spends 3× parameters on the wrong thing
5. **Uniform replay (all tokens)** — same delta as hard-token-only routing. Most tokens are too easy for replay to help.
6. **Phase A/B curriculum** — experts trained on frozen features can't adapt to shifting backbone

### What We Don't Know
1. **Can replay beat dense on reasoning benchmarks (not just PPL)?** Our lm-eval was only on L12-15 with 200 samples. Need full eval on best config.
2. **Does the benefit scale with model size?** 72B duplication gave +7.31 combined. 8B LoRA gives -0.51 PPL. How would this work at 27B or 70B?
3. **Can we combine TRM-style from-scratch recursion with pretrained model augmentation?** Train a small recursive reasoning module, inject it into a pretrained LLM.
4. **What's the right task mix for training?** Math-only didn't help more than mixed data, but we only tested on PPL.
5. **Can bidirectional prompt prefill (à la Bitune) give bigger gains than causal replay?**
6. **Is the benefit from LoRA actually "replay" or just "more parameters on middle layers"?** A control experiment with LoRA but NO replay (just adapters always on) would distinguish these.

## The Central Puzzle

We have a clear signal: **re-running attention through a pretrained LLM with slightly different weights improves reasoning.** But we can't figure out how to make this into a large, reliable improvement rather than a small one. The delta plateaus at -0.51 PPL (5.8%) no matter how many layers or how we configure the replay.

Meanwhile, TRM achieves 3.4× improvement on mazes with genuine recursion. And our 72B runtime duplication gives +7.31 on probes. Why is the trained LoRA replay on Llama only giving 5.8% PPL?

Possible explanations:
1. **PPL is the wrong metric.** Average next-token prediction averages over easy and hard tokens. The benefit concentrates on hard tokens but is diluted.
2. **We're not actually doing recursion.** LoRA replay runs slightly different attention once more — it's not iterative refinement toward a fixed point. TRM runs 18 cycles. We run 1 extra pass.
3. **The LoRA is just learning to be a better single-pass model**, not learning to replay. It's equivalent to adding 4 more layers with slightly different weights.
4. **We need to train on reasoning tasks specifically**, not general LM. The 72B improvement was measured on reasoning probes. Our LoRA was trained on web text.
5. **We need the two-level TRM structure (z_H/z_L)** to get genuine iterative refinement, not just a single-stream replay.

## Questions for You

### Architecture Design
1. **Design a unified architecture** that combines: (a) TRM's proven recursion with input injection and shared weights, (b) pretrained LLM augmentation with dense containment, (c) attention-dominant replay with minimal FFN. Give the full forward pass, loss function, and training recipe. Justify every design choice with formal arguments.

2. **Should the recursive module be trained from scratch or fine-tuned from pretrained layers?** TRM trains from scratch successfully but on small scale. Our LoRA fine-tunes a pretrained model. What's the right hybrid?

3. **How do we get TRM-level iterative refinement (18 cycles, contraction to fixed point) inside a pretrained decoder?** The LoRA replay is one extra pass with different weights — that's not iterative refinement. Can we do 6+ cycles of shared-weight replay without gradient explosion or diminishing returns?

4. **Should the recursion be causal or bidirectional?** TRM uses non-causal. LLMs are causal. Bitune showed bidirectional prompt prefill works. What's the right choice and where?

### Mathematical Analysis
5. **Prove or disprove:** For a pretrained LLM with L layers and hidden dim d, there exists a replay configuration with O(d²) additional parameters such that the replayed model's expected loss on reasoning tasks is strictly lower than the base model's, with the gap growing with task complexity.

6. **Characterize the fixed-point structure:** Under what conditions on the pretrained weights does repeated application of a middle-band layer converge to a fixed point? What is the contraction rate? How does LoRA adaptation change the fixed-point landscape?

7. **Information-theoretic analysis:** How much additional mutual information about the correct answer can K replay cycles extract, as a function of K, the replay band width, and the LoRA rank? Derive the scaling law.

8. **Why does the delta plateau at -0.51?** Is this a fundamental limit of the architecture, or a training limitation? Derive a bound on the maximum improvement achievable by attention-only replay on a frozen base model.

### Experimental Design
9. **What is the single most important control experiment we haven't run?** (Candidate: LoRA with adapters always ON, no replay — tests whether the gain is from "replay" or just "more parameters.")

10. **Design the decisive experiment** that either confirms recursive computation helps LLMs or proves it doesn't. What metric, what task, what architecture, what baseline? Budget: 4 B200 GPUs for 1 week.

### Creative Challenges
11. **Are we framing the problem wrong?** Maybe "replay" isn't the right framing. TRM doesn't "replay" — it iteratively refines toward a fixed point. What if we trained the LLM to do iterative refinement in its hidden state, not "re-reading" the input?

12. **What would a neuroscience-informed design look like?** The brain uses recurrence for hard tasks (working memory, multi-step planning). How does biological recurrence differ from what we've tried, and what design principles transfer?

13. **Could we get bigger gains by changing WHAT is replayed rather than HOW MUCH?** Instead of replaying whole layers, what if we replayed only the attention patterns (keys/values) from specific heads that are most relevant to reasoning?

## Compute Budget
- 4 × NVIDIA B200 (192GB VRAM each)
- Available 24/7 for ~1 week
- Can run 4 experiments in parallel
- Llama 3.1 8B fits on 1 GPU in bfloat16 (~16GB)
- Gemma 3 27B available locally
- All infrastructure for LoRA training, lm-eval, custom training loops already built

## What I Want From Your Response

1. A **single, concrete architecture** with full forward-pass pseudocode — not 5 options, one best design
2. **Mathematical proofs** for why each design choice is optimal
3. A **training recipe** specific enough to implement in one afternoon
4. **Expected results** with quantitative predictions (e.g., "this should achieve delta=-X on PPL and +Y% on ARC-Challenge")
5. **The first experiment to run** that validates or kills the design within 24 hours on 4 GPUs
6. **Brutal honesty** if you think we should stop pursuing this direction entirely
