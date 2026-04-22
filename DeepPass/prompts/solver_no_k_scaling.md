# The Solver Beats the Base Model by 24% — But Won't Iterate. Why?

## Context

You've been consulting on this project across 5 prior sessions. Your last recommendation was to build a **separate bidirectional prompt solver** that feeds memory tokens to a frozen LLM decoder. We built it. It works — but only as a one-shot encoder, not as an iterative reasoner.

## What We Built

A separate trainable solver module (~50M params) that:
- Takes raw prompt embeddings from frozen Llama 3.1 8B
- Processes them bidirectionally with a two-level hierarchy (z_H global memory + z_L token workspace), inspired by TRM
- Re-injects raw prompt embeddings every inner cycle
- Uses shared weights across K outer cycles (true iteration)
- Gradient truncation: only last outer cycle gets gradients
- Outputs 16 memory tokens prepended to prompt for frozen decoder
- Trained on 80% math + 20% general with answer-only loss

```python
# The solver's core loop
for s in range(K_outer):          # 3 outer cycles
    for _ in range(K_inner):       # 6 inner steps each
        z_L = z_L + prompt_embs   # raw input re-injected
        z_L = z_L + cross_attn(z_L, z_H)
        z_L = bidirectional_self_attn(z_L)
    z_H = z_H + cross_attn(z_H, z_L)
    z_H = bidirectional_self_attn(z_H)

memory_tokens = project(z_H)  # → prepend to decoder
```

## The Result

**K=0 (no solver) PPL = 2.49. K≥1 (with solver) PPL = 1.89. 24% improvement.**

But:

| K_outer | PPL (consistent across ALL variants) |
|---------|--------------------------------------|
| 0 (base) | 2.49 |
| 1 | 1.89 |
| 2 | 1.89 |
| 3 | 1.89 |
| 4 | 1.89 |
| 8 | 1.89 |

**K=1 = K=2 = K=3 = K=4 = K=8 = 1.89. Zero K-scaling.**

## What We Varied (8 Experiments, All Same Result)

| Variant | K_outer | Gradient Truncation | Result |
|---------|---------|--------------------| -------|
| Full solver | 3 | Yes | K≥1 = 1.89 |
| No truncation | 3 | No | K≥1 = 1.89 |
| Shallow | 1 | Yes | K≥1 = 1.89 |
| Deep | 4 | Yes | K≥1 = 1.89 |
| Full v2 (repeat) | 3 | Yes | K≥1 = 1.79 |
| Multiround supervision | 1,2,3 weighted | Yes | K≥1 = 1.89 |
| Large (d=2048) | 3 | Yes | K≥1 = 1.89 |
| Sentinel backfill | 3 | Yes | K≥1 = 1.89 |

**Multiround supervision** (supervise at K=1, K=2, K=3 with increasing weight per batch — your recommended approach from the prior session) — **same result.** No K-scaling.

## What This Means

The solver is collapsing to a **one-step fixed point**. After 1 outer cycle, the memory tokens are already as good as they'll get. Additional cycles produce identical memory tokens because the solver has learned to solve the problem in one shot.

This is exactly what you predicted in consultation #5:

> "If T_θ learns to map h_0 close to a fixed point in one step on the train distribution, then W·T_θ(h_0,u) ≈ W·T_θ²(h_0,u) ≈ ... and the loss cannot distinguish that from genuine multi-step reasoning."

The solver took the path of least resistance: become a good one-step encoder.

## Why This Is Still Different From Our Earlier Failures

| Metric | Replay (CIRRA etc.) | Solver |
|--------|-------------------|--------|
| K=1 vs base | Same or worse | **24% better** |
| K=2 vs K=1 | Worse (hurts) | Same (neutral) |
| Extra compute | Wasted | Neutral |
| Architecture | Decoder replaying itself | Separate thinker |

The solver is the first architecture where the recursive module helps at all. But it helps as a **bidirectional prompt encoder**, not as an iterative reasoner. Functionally it's similar to Bitune (bidirectional prompt processing for decoder-only LLMs).

## The Fundamental Problem

The solver has no reason to iterate. Consider:

1. **The loss is the same regardless of K.** Whether K=1 or K=4, the answer-only CE loss on the decoder output is identical — so the optimizer makes K=1 perfect and K>1 redundant.

2. **The problems are too easy for 1 cycle.** With a 50M-param bidirectional solver on 512-token math prompts, one cycle of 6 inner steps (12 bidirectional attention passes) is already enough to extract all useful information.

3. **There's no explicit reward for deeper thinking.** The loss doesn't know or care how many cycles ran. It only sees the final decoder output.

4. **The solver's output space is low-dimensional.** 16 memory tokens × 4096 dims = 65K dimensions. Once these are saturated with useful information from cycle 1, there's nothing left for cycle 2 to improve.

## What TRM Does Differently (That We Still Haven't Captured)

TRM achieves K-scaling on mazes because:

1. **The problem genuinely requires iterative computation.** Finding a path through a 30×30 maze cannot be solved in one forward pass of 2 tiny layers. The model MUST iterate — there's no shortcut.

2. **The output is graded on global correctness.** Getting 90% of maze tokens right but missing one critical turn = wrong path = bad score. This creates pressure for the solver to keep refining.

3. **The carry state persists across sequences.** TRM's z_H and z_L carry over between batches — the model builds up a running representation across multiple problems.

4. **The task is small-model-bottlenecked.** TRM is 7M params solving 900-token mazes. It literally doesn't have enough single-pass capacity. Our solver is 50M params on 512-token prompts — overkill for one cycle.

## What I Think Is Missing

### Theory: K-scaling requires capacity bottleneck

If the solver can extract all useful information in one cycle, there's no pressure to iterate. K-scaling requires tasks where:
- The solver's per-cycle capacity is LESS than what the task demands
- Additional cycles genuinely expose new useful information
- The loss is sensitive to the incremental improvement from deeper thinking

### Possible Fixes

1. **Make the solver tiny.** If it can't solve the problem in one cycle, it HAS to iterate. TRM is 7M params. Our solver is 50M. What if we used a 1M-param solver on hard tasks?

2. **Use harder tasks.** 512-token math problems might be too easy. What about multi-hop reasoning, long-document QA, or actual graph problems encoded as text?

3. **Penalize early stopping.** Add a loss term that explicitly rewards improvement from K=1 to K=2: `loss += max(0, CE(K=1) - CE(K=2))` — the model is rewarded when deeper thinking helps.

4. **Make the output space progressive.** Instead of outputting all 16 memory tokens at once, output 4 tokens per cycle across 4 cycles. Cycle 1 gives a rough summary, cycle 2 refines, cycle 3 adds detail, cycle 4 finalizes.

5. **Use a verifier/critic.** Train a separate head that predicts "will the decoder get this right?" after each cycle. If the verifier says "not confident," force another cycle. This creates endogenous pressure to iterate.

6. **Detached multi-round supervision with DIFFERENT data.** Each outer round gets a different batch. The solver must generalize across rounds, not just memorize one problem in one step.

7. **RL reward: sequence-level correctness, not token-level CE.** The maze solver gets a binary reward (correct path or not). Token CE doesn't create pressure for global coherence that requires multiple passes.

## The Complete Experimental Record

### Track 1: 72B Runtime Duplication
- +7.31 combined score (no training)
- Attention helps reasoning, FFN hurts factual recall

### Track 2: TRM (7M params, mazes)
- 3.4× improvement with 18 cycles of shared-weight iteration
- Contraction rate ρ ≈ 0.85 on learned attracting manifold

### Track 3: ARR from scratch (1.7B)
- 4.3× worse PPL than dense (rank bottleneck, split-state degeneracy)

### Track 4: DAR/LoRA replay on Llama 8B
- Gate-only: -0.07 PPL (real but tiny)
- LoRA: -0.51 PPL (fake — control matched, was measuring LoRA quality)
- True replay (layers 2×): K=2 HURTS (PPL 7.89 → 9.06)

### Track 5: CIRRA on Llama 8B
- Separate trainable core, shared weights, dense reinjection
- K=1 always best. No K-scaling. Always-on control wins.

### Track 6: Prompt Solver on Llama 8B
- 24% PPL improvement (K=0=2.49 → K=1=1.89) ← **first real gain**
- No K-scaling (K=1=K=2=K=3=K=4=1.89)
- 8 variants all show identical pattern

## Critical Update: The Ceiling Is the Decoder, Not the Solver

After the initial 8 experiments, we ran 2 more targeted tests:

### Tiny Solver (d=256, 1 layer, ~2M params)
**Hypothesis:** If the solver is too small to solve the problem in one cycle, it MUST iterate.
**Result:** K≥1 = 1.89. Same ceiling. Even a 2M-param solver one-shots to 1.89.

### Iteration Penalty (explicit reward for K=3 beating K=1)
**Hypothesis:** Loss += penalty when K=3 doesn't beat K=1. Forces the optimizer to make iteration useful.
**Result:** K≥1 = 1.89. The penalty goes to zero because K=1 and K=3 produce identical output.

### What This Proves

**The bottleneck is NOT the solver.** A 2M-param solver and a 50M-param solver both hit 1.89. The bottleneck is the **interface between solver and decoder**: 16 prepended memory tokens have a fixed capacity to influence the frozen decoder's output. Once saturated at K=1, no solver improvement can push more information through.

This is a bandwidth-limited channel: 16 tokens × 4096 dims = 65K dimensions. At K=1, those 65K dims are already optimally used. At K=2+, the solver produces marginally different memory tokens that the decoder can't distinguish from K=1's output.

### Complete Results Table (10 experiments, all identical)

| Variant | Solver Size | K=0 | K=1 | K=2 | K=3 | K=4 |
|---------|------------|-----|-----|-----|-----|-----|
| Full | 50M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| No grad truncation | 50M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| Shallow | 50M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| Deep (K=4) | 50M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| Multiround supervision | 50M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| Large (d=2048) | 200M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| Repeat (seed 2) | 50M | 2.49 | 1.79 | 1.80 | 1.79 | 1.79 |
| Sentinel backfill | 50M | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| **Tiny (d=256)** | **2M** | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |
| **Iteration penalty** | **50M** | 2.49 | 1.89 | 1.89 | 1.89 | 1.89 |

### Implications

The solver→decoder interface needs to change. The fundamental issue: **we separated the thinker from the talker (good), but gave them a walkie-talkie when they need a fiber optic cable.**

Possible directions:
1. **More memory tokens** (64 or 128 instead of 16) — wider channel
2. **Inject memory into decoder hidden states** (not just prepend) — bypass the embedding bottleneck
3. **Progressive memory** — K=1 outputs 4 tokens, K=2 outputs 8, K=3 outputs 16
4. **Unfreeze decoder attention to memory tokens** — let decoder learn to read solver output better
5. **Cross-attention from decoder to solver state** — higher-bandwidth interface
6. **Solver modifies decoder's KV cache directly** — inject keys/values into specific decoder layers

## Questions

1. **Is the 1.89 ceiling a proven information-theoretic limit of 16 prepended memory tokens?** Formalize: given a frozen autoregressive decoder and M prepended soft tokens, what is the maximum mutual information between the soft tokens and the decoder's output? Is 16 tokens enough to transmit the information that additional solver cycles would compute?

2. **Is the bottleneck the solver capacity, the task difficulty, or the loss function?** Design an experiment that distinguishes these three hypotheses.

3. **How does TRM actually achieve K-scaling?** Formalize the conditions under which a shared-weight recurrent module improves monotonically with K. What properties of maze-solving create those conditions? Can any text task satisfy them?

4. **Design the smallest experiment that would achieve K=2 < K=1.** Not at scale — just prove the principle. Even on a toy task. What task, what architecture, what loss? I'll run it tonight.

5. **Should we give up on K-scaling for LLMs and accept that the bidirectional solver (24% gain, one-shot) is the best we can do?** If so, how do we make that result as strong as possible for a paper?

6. **Think wildly.** Is there something completely different we should try? We've been thinking "how to make the solver iterate." Maybe the question should be "how to make the LLM NEED the iteration." What if we deliberately degraded the frozen decoder (drop some layers, quantize aggressively) so it can't solve hard problems alone and NEEDS the solver to iterate?

## Compute
- 4 × B200 GPUs available
- Llama 3.1 8B loaded and ready
- Solver codebase at `/blue/cis4914/jietao/DeepPass/solver/`
- Can run experiments within minutes
