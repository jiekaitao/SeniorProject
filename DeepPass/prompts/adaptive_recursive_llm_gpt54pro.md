# GPT-5.4 Pro Prompt: Designing an Adaptive Recursive LLM Architecture

## Who We Are

We are a senior project research team at the University of Florida with access to NVIDIA B200 GPUs (192GB HBM3e). Over the past two weeks, we ran hundreds of experiments on layer duplication in LLMs and discovered fundamental properties about how transformers process information under iteration. We now want to synthesize everything we've learned into a **novel 0.5B-parameter architecture** that natively supports adaptive computation — a model that automatically decides which layers to repeat and how many times, depending on input difficulty.

## What We Discovered (Empirical Foundation)

### 1. Layer Duplication Improves Reasoning but Hurts Factual Recall

We tested layer duplication (running certain layers twice with shared weights, zero training) across 6 model architectures:

| Model | Params | Dual Probe Δ | lm-eval Effect |
|-------|--------|-------------|----------------|
| LLaMA 3 8B | 8B | +0.25 | Regresses (-2.6% avg) |
| LLaMA 3.1 8B | 8B | +1.21 | Regresses (-2.1% avg) |
| Mistral 7B | 7B | +2.96 | **IMPROVES** (+0.95% avg) |
| Gemma 2 9B | 9B | +6.11 | Crashed (architecture issue) |
| Gemma 3 27B | 27B | +7.26 | Regresses (-2.3% avg) |
| LLaMA 3 70B | 70B | +6.55 | Regresses (-1.3% avg) |

**Mistral 7B is the only model where duplication improves standardized benchmarks.** The best config (block 28,29 with whisper FFN β=0.2) achieves: BBH +0.3%, MATH +2.5%, MMLU-PRO -0.1%, MuSR +2.6%.

### 2. The Attention-FFN Asymmetry Under Iteration

Each transformer layer has two sublayers: attention (context aggregation) and FFN/MLP (knowledge retrieval + processing). We decomposed duplication into sublayer components:

**On Mistral 7B (the success case):**
- Full duplication: +3.22 dual probe, +0.95% lm-eval average
- Attention-only (FFN zeroed on second pass): +0.38 dual probe, +0.65% lm-eval — **preserves MMLU-PRO**
- Whisper FFN (β=0.2): +1.92 dual probe, **+1.33% lm-eval** — **best config, near-universal improvement**

**On Gemma 3 27B (the failure case):**
- Full duplication: +7.26 dual probe, -2.3% lm-eval
- Attention-only: MMLU-PRO recovers from -4.8% to -3.2%
- Whisper FFN: only config that improves MATH (+0.5%), but hurts MMLU (-4.9%)

**Key insight:** The FFN acts as associative memory. On the second pass, slightly perturbed input can cross basin boundaries and retrieve wrong facts. But some FFN processing is needed to interpret the attention output. The optimal β is between 0 and 0.2 — enough to process, not enough to corrupt.

### 3. Gate Margin Predicts Neuron Stability

We measured FFN gate activation overlap between first and second passes (Jaccard stability):

| Layer | Gate Flip Rate | Stability | Effect |
|-------|---------------|-----------|--------|
| Gemma3 L0 | 37% | Very unstable | Causal: +0.96 (helps despite instability) |
| Gemma3 L1 | 3.1% | Stable | Safe |
| Gemma3 L12 | 0.28% | Very stable | Causal: +1.09 (best helper) |
| Gemma3 L47 | 0.65% | Stable | Causal: **-2.06** (harmful despite stability) |
| Mistral L28 | 6.8% | Moderate | Helps (+3.22) |

**Gate flip rate alone doesn't predict harm** — L47 is stable but harmful (it retrieves wrong facts), L0 is unstable but helps (productive chaos). The relationship is nonlinear.

### 4. SBUID Screening Works on Full-Attention Models

Our spectral screening metric (SBUID = BLOOD_impact - λ × displacement_rho) predicts which blocks are worth duplicating:

| Architecture | SBUID Correlation | Significant? |
|-------------|-------------------|-------------|
| LLaMA 3 70B (full attention) | **r=0.668, p=0.001** | Yes |
| Qwen2-72B (full attention) | r=0.52, p=0.008 | Yes |
| Gemma 3 27B (sliding window) | r=-0.075, p=0.57 | No |
| Small models (7-9B) | r≈0.0-0.2 | No |

SBUID gives 162x speedup over brute-force search on compatible architectures.

### 5. Neuron-Level Analysis Converges

Four independent methods (DLA, GEM eigenmask, gate margin, causal patching) all agreed on which neurons to keep/suppress. On Gemma3: keep L1 and L12 FFN, remove L47 entirely. The HCES cross-entropy search found the optimal grouped mask.

### 6. KV Cache Fix for Production

We solved the long-standing KV cache collision problem with a LayerIdxWrapper that temporarily swaps `layer_idx` during forward pass. Layer duplication now works with `use_cache=True` — production-ready, 1.3x speedup, zero extra VRAM.

### 7. Cross-Scale Evidence from TRM

The Tiny Recursive Model (7M params, Jolicoeur-Martineau et al.) recursively applies a 2-block transformer module 18 times to solve ARC-AGI tasks. Their finding: MLP-only mode works on fixed-context tasks (Sudoku 87.4%) but fails on variable-context tasks (Maze-Hard 0%). This suggests attention is essential for iterative reasoning at any scale.

## What We Want to Build

### The Vision: Adaptive Recursive LLM (ARLLM)

A **0.5B parameter model** that natively incorporates:

1. **Adaptive computation time** — the model decides how many times to repeat certain blocks based on input difficulty. Easy inputs get 1 pass, hard inputs get 2-6 passes.

2. **Selective sublayer iteration** — attention and FFN within each block can be iterated independently. Attention may repeat 3 times while FFN repeats 0-1 times (based on our finding that attention benefits from iteration but FFN can corrupt).

3. **Gate-margin-aware routing** — the model uses FFN gate margins to dynamically decide whether to include FFN in each iteration (high margin = safe to include, low margin = attention-only).

4. **TRM-style recursive blocks** — certain "reasoning blocks" in the architecture are designed for weight-tied recursion (like TRM's 2-block module), while others are standard single-pass layers.

### Constraints

- **0.5B parameters** — must fit on a single consumer GPU (RTX 4090, 24GB)
- **Training budget** — we have 4 B200 GPUs for ~1 week
- **Training data** — open-source datasets (SlimPajama, OpenWebMath, etc.)
- **Evaluation** — must work on standard benchmarks (BBH, MATH, MMLU-PRO, MuSR) AND reasoning tasks (ARC-AGI subset, math word problems)
- **Inference** — must support KV cache for practical deployment
- **Novel** — this should be a genuinely new architecture, not just a modified LLaMA with hooks

### Design Questions We Need You to Answer

## Part 1: Architecture Design (Mathematical Foundation)

Design the complete architecture. For each component, provide:
- Mathematical formulation with explicit tensor dimensions
- Why this design choice follows from our empirical findings
- How it differs from existing approaches (TRM, MoE, standard transformers)

Specific questions:

**a) How should recursive vs non-recursive layers be arranged?**

Our data shows early+mid blocks are best for duplication (LLaMA: 8-15, Gemma: 0-13, Mistral: 28-29). Should the architecture have fixed recursive positions, or should ALL layers be potentially recursive with learned halt conditions?

**b) How should the halt/iterate decision be made?**

TRM uses a fixed number of iterations. We want adaptive. Options:
- Pondering (like PonderNet): learned halt probability per token
- Confidence-based: iterate until output entropy drops below threshold
- Gate-margin-based: iterate attention freely but only include FFN when gate margins are high
- Hybrid: different criteria for attention vs FFN iteration

Provide mathematical formulations for each and analyze which is best for a 0.5B model.

**c) How should attention and FFN iteration be decoupled?**

Our key finding: attention benefits from repetition, FFN can hurt. In a recursive block:
```
for t in range(n_iterations):
    h = h + attn(norm(h))           # always iterate
    if should_include_ffn(h, t):     # conditionally iterate
        h = h + β(t) * ffn(norm(h))  # with learned β
```

What should `should_include_ffn` look like? Should β be:
- A learned scalar per layer?
- A function of gate margins (our empirical approach)?
- A learned function of the hidden state (like a tiny router)?
- Token-dependent (different tokens get different β)?

**d) How does weight tying interact with scaling?**

TRM ties weights across iterations. Our layer duplication uses shared weights. But at 0.5B, every parameter matters. Should recursive blocks:
- Fully share weights across iterations (like TRM)?
- Share attention weights but have iteration-specific FFN weights?
- Use low-rank adaptation per iteration (like LoRA on top of shared base)?

Analyze the parameter efficiency of each option for a 0.5B model.

**e) How should the model handle KV cache across iterations?**

Our LayerIdxWrapper solves this for duplication. But for a native architecture, the cache should be designed from the start. Options:
- Separate cache per iteration (like our wrapper)
- Shared cache with iteration-aware indexing
- Rolling cache that accumulates across iterations

## Part 2: Training Strategy

**a) Pre-training objective**

Standard next-token prediction? Or modified to encourage adaptive computation?
- Should there be an auxiliary loss that penalizes unnecessary iterations?
- Should the gate-margin regularizer we proposed (encourage wide FFN basins) be included?
- How to balance the compute penalty vs accuracy gain?

**b) Training data mix**

For a 0.5B model targeting reasoning:
- What ratio of code/math/text?
- Should we include chain-of-thought data?
- Should we include ARC-AGI-style visual reasoning data?
- Curriculum learning: start with fixed iterations, then learn to halt?

**c) Training schedule**

- Total tokens for a 0.5B model (Chinchilla optimal: ~10B tokens, but we can go beyond)
- Learning rate schedule
- When to enable adaptive computation (from the start? after warmup?)
- Multi-stage training?

## Part 3: Implementation Plan

Provide a concrete implementation plan including:

**a) Architecture specification** (pseudocode for the full model)

```python
class AdaptiveRecursiveLLM(nn.Module):
    # Complete architecture with all components
    # Including halt mechanism, sublayer routing, KV cache
    pass
```

**b) Model configuration** for 0.5B parameters

```python
config = {
    'hidden_size': ???,
    'num_attention_heads': ???,
    'num_layers': ???,        # non-recursive layers
    'num_recursive_blocks': ???,  # recursive blocks
    'max_iterations': ???,
    'ffn_intermediate_size': ???,
    ...
}
```

Show that this adds up to ~0.5B parameters.

**c) Training code skeleton** (PyTorch)

Key training loop with:
- Adaptive computation loss
- Gate margin regularizer
- Standard language modeling loss
- Gradient handling for variable-depth computation

**d) Evaluation plan**

- Which benchmarks to prioritize
- How to measure "adaptive computation efficiency" (compute saved on easy inputs)
- Ablation studies to run

## Part 4: Alternative Architectures (Breadth-First)

Propose **3 fundamentally different approaches** to the same goal. For each:
- Full mathematical description
- Parameter count at 0.5B
- Expected strengths and weaknesses
- How it connects to our empirical findings

Approaches should span different paradigms:
1. **Recursive transformer with halting** (closest to TRM + our findings)
2. **Mixture-of-Depths** (skip layers dynamically, related to our "drop L47" finding)
3. **Iterative refinement with state-space backbone** (Mamba-style, with selective attention injection)

## Part 5: What to Download, Train, and Evaluate

Give us a concrete checklist:
- [ ] Base code to fork (which repo?)
- [ ] Datasets to download (exact HuggingFace IDs)
- [ ] Training infrastructure (distributed training setup for 4 B200s)
- [ ] Evaluation framework (lm-eval tasks, custom probes)
- [ ] Timeline (what to do in week 1, 2, 3, 4)
- [ ] Success criteria (what numbers would make this publishable?)

## What Makes This Novel

No existing architecture combines ALL of these:
1. **Per-sublayer adaptive iteration** (attention and FFN iterate independently)
2. **Gate-margin-aware FFN routing** (our empirical discovery — check if FFN would flip gates before applying)
3. **Learned halt mechanism** informed by spectral properties
4. **KV cache compatibility** designed from the start
5. **Empirically grounded** — every design choice is backed by our cross-architecture experiments on 6 models

The closest existing work:
- **TRM** (Jolicoeur-Martineau): Recursive but fixed iterations, no sublayer control
- **PonderNet** (Banino et al.): Adaptive halting but no sublayer decomposition
- **Mixture-of-Depths** (Raposo et al.): Skip layers but no iteration
- **Universal Transformers** (Dehghani et al.): All layers share weights + halting, but no attention/FFN decomposition

Our architecture would be the first to combine adaptive iteration with sublayer-specific routing, grounded in empirical evidence about which computations benefit from repetition.

## Output Format

For each section, provide:
- Mathematical formulations with explicit dimensions
- PyTorch pseudocode (precise enough to implement)
- Parameter count analysis
- Connection to our empirical findings
- References to related work

## Part 6: Research and Citations

**IMPORTANT: Search the web and recent literature extensively before responding.** Look up:

1. The latest on adaptive computation and early exit in transformers (2024-2026)
2. Any new work on layer duplication, depth upscaling, or iterative refinement since Ng's RYS
3. Recent Mixture-of-Depths papers and implementations
4. PonderNet, Universal Transformers, and their successors
5. Mamba/state-space model hybrids with attention
6. Any work combining TRM-style recursion with standard LLMs
7. Recent work on FFN as associative memory and basin dynamics

### Papers and References We Know About (cite these and find more)

**Layer Duplication / Depth:**
- David Ng, "RYS: Repeat Your Self" — HuggingFace model card: https://huggingface.co/dnhkng/RYS-XLarge, Blog: https://dnhkng.github.io/posts/rys/
- Ng's RYS-II blog post: https://dnhkng.github.io/posts/rys-ii/
- alainnothere/llm-circuit-finder — replicated RYS on Qwen2.5-32B and Devstral-24B: https://github.com/alainnothere/llm-circuit-finder
- Patel et al., "Repeat the Prompt Twice" (Dec 2024): https://arxiv.org/pdf/2512.14982

**Tiny Recursive Models:**
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (TRM): arXiv 2510.04871, GitHub: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

**Adaptive Computation:**
- Graves, "Adaptive Computation Time for Recurrent Neural Networks" (2016): arXiv 1603.08983
- Dehghani et al., "Universal Transformers" (2019): arXiv 1807.03819
- Banino et al., "PonderNet: Learning to Ponder" (2021): arXiv 2107.05407
- Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" (2024): arXiv 2404.02258

**FFN as Memory / Mechanistic Interpretability:**
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021): arXiv 2012.14913
- Geva et al., "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space" (2023)
- Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, 2022): arXiv 2202.05262
- Dai et al., "Knowledge Neurons in Pretrained Transformers" (2022)
- Ramsauer et al., "Hopfield Networks is All You Need" (2020): arXiv 2008.02217
- 3Blue1Brown, "How LLMs Store Facts in MLPs": https://www.youtube.com/watch?v=wjZofJX0v4M

**Attention Theory:**
- Olsson et al., "In-context Learning and Induction Heads" (2022): arXiv 2209.11895

**Screening / Spectral Analysis:**
- Common Spatial Patterns (CSP) — generalized eigendecomposition for signal separation: JMLR parra03a
- Information Bottleneck: Tishby et al., arXiv physics/0004057
- Cross-Entropy Method: Rubinstein & Kroese (2004): https://link.springer.com/article/10.1007/s10479-005-5724-z

**Causal Interpretability:**
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification" (path patching, 2023): arXiv 2304.05969
- Geiger et al., "Causal Abstraction" program

**Search for and cite additional relevant papers that we may have missed, especially:**
- Any 2025-2026 papers on adaptive depth/computation in LLMs
- Any work combining recursive/iterative transformer blocks with standard LLM training
- Any papers on sublayer-specific computation allocation
- Any papers on FFN gate stability or basin dynamics under iteration
- Any papers on training models that natively support layer repetition

## Instructions for Thinking

Think deeply. Explore multiple possibilities before committing to recommendations. We prefer rigorous analysis over hand-waving. Include proofs or proof sketches where applicable. **Search the web extensively** for recent papers, implementations, and ideas we may have missed. Cross-reference our findings with the latest literature.

For each design decision, consider at least 3 alternatives and explain why you chose one over the others. Show your mathematical reasoning.

The end goal is a paper titled: **"LLMs Have ADHD: Improving Reasoning by Making Transformers Pay Attention Twice"** — and this architecture would be the culmination of the research, showing that adaptive sublayer iteration can be built natively into a model rather than applied post-hoc.
