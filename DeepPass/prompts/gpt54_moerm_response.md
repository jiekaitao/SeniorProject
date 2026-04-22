# GPT-5.4 Pro Response: MoERM Architecture Evaluation

**Date:** April 7, 2026
**Context:** Asked GPT-5.4 Pro to evaluate our Mixture of External Reasoning Modules (MoERM) proposal with mathematical rigor and honest assessment.

## Verdict

Do **not** throw away the idea, but do throw away the **full** version for now. Build **MoERM-Lite**: 4 experts at most, sequence-level soft routing, fixed output prefix budget, fixed K at first, and strong equal-compute baselines.

## Critical Assessment

Our **current evidence** says memory capacity matters much more than iteration depth on SpatialEval, while K-scaling on multi-hop tasks is noisy and monotone in only 35% of runs. The first-order bottleneck is probably **representational bandwidth / workspace allocation**, not "which expert should think how many times."

**The right question:**
> Does routed conditional latent computation beat a monolithic latent solver under equal active compute and equal prefix length?

If we don't answer that, the whole project looks like architecture tourism.

## Build Order

1. **Single Solver scaling control** — increase solver width/slots, keep prefix=32
2. **Shared-core multi-expert baseline** — one SolverCore, N expert-specific initial states + FiLM/LoRA modulations
3. **MoERM-Lite** — N=4 experts, sequence-level soft routing, fixed K, 32-slot fusion
4. **Adaptive depth** — only if step 3 beats steps 1 and 2

## Recommended Architecture

- **Router**: sequence-level, NOT token-level. Attention pooling over prompt embeddings → 2-layer MLP → expert logits → softmax (soft during training, top-2 at inference)
- **Fusion**: Perceiver-style fixed-slot cross-attention with expert-ID embeddings and gate scaling into 32 output slots
- **Experts**: 4, not 8 or 16. Fixed K initially
- **REJECT** L1 sparsity on softmax gates (always sums to 1, useless). Use entropy or entmax instead

## Key Mathematical Results

### Why specialization is NOT guaranteed (Symmetric Collapse Proposition)

If all experts share same architecture, loss is permutation-invariant, experts initialized identically, and router initialized symmetrically, then equal experts remain equal under gradient flow. Need symmetry breakers: domain warm-start, expert-specific initial states, FiLM/LoRA modulations, or explicit domain labels.

### When routing beats a single solver

The conditional risk gap: R_mix = E[min_i r_i(x)] ≤ min_j E[r_j(x)] = R_single. Strict inequality iff different experts optimal on different subsets with nonzero measure.

### Capacity separation

Under fixed **active** rank budget r < Σ r_z, a routed mixture can represent mappings that a single monolithic solver cannot. The mixture advantage is **conditional allocation of active capacity**.

### Router gradient is advantage-based

∂ℓ/∂z_i = α_i(⟨g, M̄_i⟩ - Σ_j α_j⟨g, M̄_j⟩)

Router increases weight on experts whose memory gives lower downstream loss than the mixture average. Dead experts (α_i ≈ 0) get no gradient. Collapse (all M̄_i equal) gives zero router gradient.

## Practical Hyperparameters (Llama 3.1 8B on 4× B200)

- bf16, FlashAttention-2, gradient checkpointing
- seq_len 1024, microbatch 4-8, grad accum to 64-128
- AdamW, betas=(0.9, 0.95), wd=0.01
- Expert LR: 2e-4, Router/fusion LR: 5e-4 to 1e-3
- Warmup 3%, cosine decay, grad clip 1.0
- Router temp: start 2.0, anneal to 1.0 over 30%
- λ_lb=0.02 early then 0.005
- λ_ent=0 for first 20-30%, then 0.002-0.01
- λ_div ≤ 1e-4 if used at all

## Training Curriculum (Branch-Train-Route)

**Stage A: branch-train** — GPU 0: spatial, GPU 1: symbolic multi-hop, GPU 2: arithmetic, GPU 3: language. Short warm-start.
**Stage B: freeze experts** — Train only router + fusion on mixed data.
**Stage C: joint** — Lower expert LR, higher router/fusion LR, anneal routing temperature.

## Experiments to Run

1. **Memory vs mixture**: SingleSolver-12M/24M/48M all with prefix=32. Also 12M with 64/96 internal slots but only 32 exported.
2. **Shared-core specialization baseline**: One SolverCore + expert-specific (H_init, L_init) + FiLM
3. **MoERM-Lite**: N=4, soft routing, fixed K, 32-slot fusion. Compare at equal active params AND equal total params.
4. **Symmetry breaking**: Scratch joint vs domain warm-start. Measure MI(expert; domain), ablation heatmap, gate entropy.
5. **Adaptive depth**: Only if experiment 3 wins. Soft checkpoint mixing over {1,2,4,8}.

## Benchmarks for Specialization

4 domains: Spatial (SpatialEval, graph reachability), Symbolic multi-hop (pointer chasing, variable sub), Arithmetic (GSM8K, synthetic algebra), Language (StrategyQA, ARC-Challenge, BBH subsets)

Specialization diagnostics: ablation matrix A_{i,d} and mutual information I(E;D).

## Strongest Alternatives

1. **Shared-core expert codes**: One SolverCore, expert-specific states + FiLM. Cleanest if it works.
2. **Banked Memory Solver**: Single solver with larger internal bank + sparse readout. Tests "memory > routing" directly.
3. **TokMem/memory-token codebook**: Bank of reusable procedure tokens, router selects 1-2. Far simpler modular-memory story.

## Target Thesis Claims

**If MoERM wins:** "Conditional latent computation via routed external bidirectional solvers improves heterogeneous reasoning under a fixed decoder-prefix budget, and the improvement is explained by specialization under compute constraints rather than by raw parameter count alone."

**If controls win:** "The main benefit of external latent reasoning comes from memory/workspace capacity, not mixture routing." (Still a very good thesis.)

## Related Work Corrections

- CALM: ICLR 2024 "LLM Augmented LLMs"
- MoLE: ICLR 2024 (real)
- BTX: COLM 2024 (real)
- Proxy Tuning: COLM 2024 (real)
- COCONUT: COLM 2025 (not 2024)
- MoR: NeurIPS 2025 (real)
- TokMem: ICLR 2026 (not routing-heavy)
- MiCRo: ICLR 2026 poster (not 2025 EPFL)
- UniR: ICLR 2026 submission (not yet accepted as of April 7, 2026)

## Our Novelty

External, bidirectional, iterative latent experts that compress into a fixed prefix budget for a frozen causal decoder. Enough for a thesis if experiments are clean, but not enough if gains disappear against a larger single solver.
