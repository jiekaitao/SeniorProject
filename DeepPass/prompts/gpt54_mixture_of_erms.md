# GPT-5.4 Pro Prompt: Mixture of External Reasoning Modules (MoERM)

## Preamble

You are consulting on a senior thesis / research project titled **"LLMs Have ADHD: Iterative Latent Computation for Language Models."** The PI is a senior CS student at UF with access to **4× NVIDIA B200 GPUs (192GB VRAM each, 768GB total)** on a SLURM cluster. The codebase is in PyTorch. The frozen base LLMs available are **Llama 3.1 8B** and **Gemma 4 31B-IT**.

We need you to do three things:
1. **Critically evaluate** the proposed architecture below — is it worth pursuing, or should we discard it entirely? Be brutally honest. If it's a bad idea, say so and explain why, and suggest what we should do instead.
2. **If it has merit, develop it with proof-level mathematical rigor** — formal definitions, objective functions, convergence arguments, gradient flow analysis, capacity bounds, and computational complexity.
3. **Think outside the box.** Our initial framing may not be the best one. Propose alternatives, simplifications, or entirely different architectures that achieve the same goal more elegantly. We are not always correct.

**Before you begin, please search the web thoroughly** for related work published in 2024-2026, especially anything involving modular reasoning, mixture-of-experts for reasoning, compositional inference-time compute, or latent reasoning augmentation for frozen LLMs. Cite what you find and position our idea relative to the landscape.

---

## Background: What We've Built So Far

We have a working **Prompt Solver** architecture — a small (~12M param) bidirectional recursive transformer that "thinks" about a prompt and produces memory tokens that are prepended to a frozen LLM's input. The architecture:

```
Input prompt → Frozen LLM embeds → Solver processes bidirectionally → Memory tokens (16-32)
                                                                          ↓
                        Frozen LLM decoder sees: [memory tokens | full prompt] → answer
```

**The Solver (SolverCore):**
- Two-level hierarchy inspired by Alexia's TRM (Tiny Recursive Model):
  - z_L: token-aligned workspace (bidirectional self-attention + cross-attention from z_H)
  - z_H: global memory slots (self-attention + cross-attention from z_L)
- Shared weights across K iterations (true weight-tied recursion)
- Projects from LLM embedding space (4096) → solver space (512) → back to LLM space (4096)
- K_inner iterations within a level, K_outer full refinement rounds

**Results so far:**
- **SpatialEval (NeurIPS 2024 maze navigation benchmark):** Solver doubles Llama 3.1 8B accuracy from 33.4% → 70.6% with 32 memory slots
- **Multi-hop reasoning (pointer chasing, variable substitution):** K-scaling (more iterations = better) works but is noisy — only 35% of runs show monotone improvement at depth-8
- **TRM interpretability:** Probes reveal the model internally represents 95.3% reachability but the output head only expresses 83.2% — a 12% gap where useful computation is wasted
- **Memory capacity > iteration depth** for spatial tasks: mem32 (70.6%) >> mem16 (39.0%), but K=1 ≈ K=8

---

## The Proposed Idea: Mixture of External Reasoning Modules (MoERM)

### Core Concept

Instead of one solver for all tasks, maintain a **bank of N lightweight bidirectional reasoning modules (ERMs)**, each of which can specialize in a different reasoning modality. A learned **router** examines the input and selects which ERMs to activate, how many iterations each should run, and how to merge their memory token outputs before feeding them to the frozen LLM decoder.

### Proposed Architecture

```
Input prompt → Frozen LLM embeds → Router(prompt_emb) → selects top-k of N ERMs
                                        ↓
                    ERM_1(prompt_emb, K_1) → mem_1    ─┐
                    ERM_3(prompt_emb, K_3) → mem_3    ─┼→ Merge → [combined memory | prompt] → Frozen LLM → answer
                    ERM_7(prompt_emb, K_7) → mem_7    ─┘
```

**Key design questions we need your help with:**

1. **Router design:** Should routing be per-token, per-sequence, or hierarchical? Soft (weighted combination of all ERMs) or hard (top-k selection)? The router must be differentiable for end-to-end training.

2. **Memory merging:** How should we combine memory tokens from multiple ERMs? Options include:
   - Concatenation: [mem_1 | mem_3 | mem_7 | prompt] — simple but grows linearly
   - Attention-based fusion: cross-attention over all ERM outputs into a fixed-size memory
   - Weighted sum: router weights applied to ERM outputs
   - Something else entirely?

3. **Unsupervised specialization:** Can N blank-slate ERMs trained jointly on diverse data naturally differentiate? We hypothesize that with load-balancing loss + diversity regularization, ERMs will specialize (one for spatial, one for symbolic, one for linguistic, etc.) because specialization minimizes the joint loss. Is this theoretically grounded? Under what conditions does this work vs. degenerate (all ERMs converge to same function)?

4. **Adaptive K per ERM:** Each ERM should be able to run a different number of iterations. The router should also output K_i for each selected ERM (or the ERM should learn to halt via ACT). How does this interact with the routing decision?

5. **Optimal number of ERMs:** Can the system learn that it only needs M < N active ERMs? A sparsity prior on the router could achieve this — start with N=16 ERMs but the system discovers it only needs 4-5.

6. **Training recipe:** What's the right curriculum? Our hypothesis:
   - Phase 1: Train each ERM independently on a different data domain (spatial, math, code, language)
   - Phase 2: Freeze ERMs, train only the router on mixed data
   - Phase 3: End-to-end fine-tuning with load balancing
   - Or: skip Phase 1 entirely and let everything differentiate from scratch?

### My (Claude's) Implementation Sketch

Here's my initial thinking, but I want you to critique and improve it:

**Router:** A small transformer encoder (2 layers, 256-dim) that takes mean-pooled prompt embeddings and outputs:
- Gate logits g ∈ R^N (softmax → expert weights, top-k selection via straight-through Gumbel)
- Per-expert iteration counts K_i ∈ {1, 2, 4, 8} (categorical output)

**Each ERM:** Identical to our current SolverCore (~12M params each), but with separate learned H_init and L_init. N=8 ERMs = ~96M trainable params total + router.

**Merge:** Attention-based fusion — a learned cross-attention layer that takes queries from a fixed set of M=32 "output slots" and keys/values from all activated ERM outputs. This produces a fixed-size memory regardless of how many ERMs are active.

**Loss:** Standard next-token prediction on the answer, plus:
- Load balancing: each ERM should be selected roughly equally across the training distribution
- Diversity: cosine similarity penalty between ERM memory outputs (encourage different representations)
- Sparsity: L1 on router gate values (encourage using fewer ERMs)

**Compute estimate:** 8 ERMs × 12M params = 96M params. On B200 with Llama 3.1 8B frozen (16GB), we have ~170GB free for solver computation. Training should be feasible — each forward pass runs only top-2 ERMs (24M active params) plus the frozen decoder.

---

## What We Need From You

### Part 1: Critical Evaluation
- Is this idea fundamentally sound, or are there fatal flaws?
- Is the bidirectional ERM approach actually better than simpler alternatives (e.g., multiple LoRA adapters with routing, or just a single larger solver)?
- Given our results showing memory capacity matters more than iteration depth for spatial tasks, does the "mixture" aspect add value over just making one bigger solver?
- What's the minimum viable version we should build first?

### Part 2: Mathematical Development (if the idea has merit)
- Formal definition of the MoERM objective function
- Gradient flow analysis through the router → ERM → merge → frozen decoder pipeline
- Conditions for specialization to emerge (when does the load-balanced mixture avoid degenerate solutions?)
- Capacity analysis: how many ERMs are needed for K reasoning modalities?
- Convergence guarantees or bounds for the joint training
- Information-theoretic argument for why routing to specialized bidirectional modules should outperform a single large module

### Part 3: Practical Architecture Decisions
- Router architecture (your recommendation, not just ours)
- Memory merging strategy (mathematical analysis of each option)
- Training recipe with specific hyperparameters for our hardware (4× B200, Llama 3.1 8B base)
- What benchmarks should we evaluate on to demonstrate specialization?
- Expected failure modes and how to diagnose/fix them

### Part 4: Alternative Approaches
- What if we're wrong about the whole approach? What else could achieve "modular reasoning augmentation for frozen LLMs"?
- Could we achieve the same thing with simpler methods (e.g., learned prefix ensembles, multi-head memory tokens, or a single solver with internal routing)?
- How does this compare to just scaling up the single solver (more params, more memory slots)?
- Are there approaches from the MoE, neuroscience, or cognitive science literature that we're missing?

---

## Related Work We've Found (verify and expand on these)

Please search the web to verify these and find additional relevant work:

1. **UniR (Universal Reasoner, 2025)** — Composable plug-and-play reasoning modules attached to frozen LLMs via logit addition. Multiple modules can be composed by summing logits.

2. **MiCRo (Mixture of Cognitive Reasoners, 2025, EPFL)** — Partitions LM layers into 4 brain-inspired expert modules (language, logic, social, world) with learned routing. 3-stage curriculum training.

3. **CALM (Composition to Augment Language Models, 2024, DeepMind, ICLR)** — Augments frozen LLM with specialized smaller model via learned cross-attention. 13% improvement on low-resource tasks.

4. **Mixture-of-Recursions (MoR, 2025, NeurIPS)** — Per-token adaptive recursion depth routing in recursive transformers. New Pareto frontier at 135M-1.7B scale.

5. **X-LoRA / MoLE (2024)** — Multiple task-specific LoRA adapters on frozen backbone with learned gating. Dense routing with all adapters frozen.

6. **COCONUT (Chain of Continuous Thought, 2024)** — Latent reasoning in continuous space by feeding hidden states back as inputs, avoiding token-level reasoning overhead.

7. **Proxy Tuning (2024, COLM)** — Adding logit differences from small finetuned models to frozen large models. Closes 88% of the gap.

8. **Toolformer (2023, Meta, NeurIPS)** — LLM learns self-supervised tool routing. Our approach is a differentiable internalization of this.

9. **Branch-Train-MiX (BTX, 2024)** — Branch from pretrained seed, train domain experts independently, combine via MoE routing.

10. **TokMem (2025)** — Tokenized procedural memory with routing mechanism, outperforms prefix tuning.

11. **SMoA (Sparser Mixture-of-Adapters, 2025, NAACL)** — Frozen backbone + sparse-routed small adapter experts.

---

## Constraints and Preferences

- **Hardware:** 4× NVIDIA B200 (192GB each), SLURM cluster, PyTorch
- **Base LLMs:** Llama 3.1 8B (proven), Gemma 4 31B-IT (being tested)
- **Timeline:** Results needed within 2-3 weeks for senior thesis
- **Only American LLMs** (Meta Llama, Google Gemma — no Chinese models)
- **Existing codebase:** Working SolverCore, train_multihop.py, eval_spatialeval_v3.py in PyTorch
- **Our thesis claim:** "Iterative latent computation helps LLMs reason better, especially on tasks requiring spatial or multi-hop reasoning"
- **We value honest assessment over hype.** If this is a bad idea, tell us. If it's a 6-month project that can't fit in 3 weeks, tell us that too.
