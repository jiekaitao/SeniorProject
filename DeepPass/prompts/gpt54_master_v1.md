# MASTER BRIEFING — "LLMs Have ADHD" Senior Thesis: Full Research Context & Open Problem

## Instructions to You (GPT-5.4 Pro)

You are the technical director of this research direction. I need **proof-level mathematical rigor** in your arguments — not hand-waving. For every claim:
- Provide the mechanistic reasoning, ideally with equations or linear-algebra arguments.
- Cite relevant literature when the claim connects to known results (COCONUT, Recurrent-Depth, MoR, Dr.LLM, Quiet-STaR, TTT, BLOOD, RYS, TRM, ACT, universal transformers, DEQs).
- Propose at least one falsifiable experiment that would verify or disprove it.
- Where possible, **derive** behavior from first principles (loss landscape geometry, information bottleneck, representation theory of transformers, etc.).

I want **high creativity**. If the best answer requires inventing a new architecture, a new training objective, a new inference procedure, or a new mathematical framework for analyzing it — propose it. Don't settle for incremental tweaks if a rethink is warranted.

I also want you to **challenge my framing**. If you think the research direction has a fundamental ceiling, say so and justify.

Output budget: as long as needed. Aim for ≥20 dense pages if the problem warrants it.

---

## 0. TL;DR — What I Need From You

After ~500 GPU experiments on 4×B200, I have a learned "deliberation controller" (~140M params) that bolts onto a frozen Llama 3.1 8B and writes iterative thought tokens into the residual stream. It **decisively beats parameter-matched LoRA on SpatialGrid** (+37 percentage points) but **loses to LoRA on most semantic benchmarks** (WinoGrande, HellaSwag, CSQA, Mazenav). I need you to tell me:

1. **Why** does iterative latent computation win on structured reasoning and lose on semantic tasks? Derive it mathematically.
2. **How** to redesign the controller / training / inference so it wins everywhere — ideally zero-shot across benchmarks.
3. **Which 3 concrete experiments** should I run this week (I have 4×B200 for 2 weeks until thesis deadline).
4. **What's the strongest paper framing** given the evidence, and what's the honest ceiling.
5. **Are there deeper theoretical connections** (dynamical systems, information geometry, representation stability) that unify our findings?

---

## 1. Project Genealogy — Two Parallel Tracks That Converge

The thesis sits at the intersection of two research threads I've pursued this semester. Both aim at the same target: **adaptive computation time for frozen LLMs**. Understanding both is critical because their findings cross-validate each other.

### Track A: DeepPass — Zero-Shot Layer Duplication (no training)

Starting from David Ng's RYS ("Repeat Your Self") discovery — duplicating a mid-to-late transformer block at inference improves reasoning — we built a principled pipeline:

**Spectral Screening (our contribution):**
Given a candidate block `[i, j)`, compute displacement rho:
$$\rho(i,j) \;=\; \frac{\|F_{i:j}(F_{i:j}(h)) - F_{i:j}(h)\|}{\|F_{i:j}(h) - h\|}$$
where `F_{i:j}` is the block's transformation. Low rho = block is close to a fixed point = safe to iterate. Reduces search cost 162× vs brute force; 80% top-5 hit rate on 7B.

**SBUID screening metric (on 72B):**
$$\text{SBUID}_0 \;=\; \text{BLOOD-impact} - 6000\cdot\rho$$
First metric statistically significant on 72B (Spearman r=0.515, p=0.008). BLOOD-impact (Jacobian smoothness) and rho trade off — linear combination works.

**Greedy Stacking:**
Find best block → apply to modified model → spectrally screen again → find best second block → etc. Cross-region pairs (early + deep) stack complementarily:
- 72B pair `(0,7)+(45,52)`: combined 79.91 (baseline 70.52, Ng's best 76.76, **+3.15 over Ng**).

**Per-Layer Alpha (our biggest win):**
Each layer in a duplicated block gets its own alpha:
$$h_{\text{seam}} \;=\; h_1 + \sum_{l\in\text{block}} \alpha_l \cdot (h_2^{(l)} - h_1^{(l)})$$
Bayesian-optimized 21 alphas on a triple: **84.07, +7.31 over Ng's 76.76**. Zero training. Zero new weights. Just better alpha allocation.

**Key mechanistic finding — Sublayer Analysis:**
Duplication helps reasoning (IFEval +2.3%, MuSR +1.3%) but hurts knowledge (MATH −6.4%, MMLU-PRO −4.8%). Decomposing: attention benefits from repetition (it's a re-weighting operation), FFN hurts (it's key-value retrieval — on a perturbed query, FFN retrieves the wrong facts). **Attention-only duplication** resolves the trade-off. This is the "FFN re-retrieval corruption" hypothesis.

**Cross-architecture generalization:**
Works on Qwen2-72B, Qwen3.5-27B/9B, Gemma3-27B, Qwen3-30B MoE, Llama 3.1 8B. 4-bit NF4 quantization preserves the gain (+8.06 delta on 72B at 59GB). Inference overhead: 11% for a pair, 20% for a quad. Zero extra VRAM (shared weights).

### Track B: Solver → Recurrent Deliberation Controller (trainable, frozen backbone)

Motivated by TRM (Alexia Jolicoeur-Martineau's Tiny Recursive Model, 7M params, solves 30×30 mazes to 83% accuracy) and ACT (adaptive computation time), I built a trainable module that bolts onto a frozen LLM.

**V1: SolverCore (bidirectional).** Two-level z_L / z_H hierarchy, shared weights across K iterations, projects 4096↔512. Tested on multi-hop pointer chasing and variable substitution.

**V2: SpatialEval benchmark (NeurIPS 2024).** Mazenav, spatialmap, spatialgrid, spatialreal — 4-way MCQA. Llama 8B baseline on mazenav: 33.4%. Solver v1 + bypass decoder: **70.6%.** But wild variance (std ~15%).

**Hit the 72% ceiling.** Plateau at ~72% across 50+ runs despite architectural variants (width, depth, fusion, MoERM). Diagnosed by you (GPT-5.4 Pro, prior consultation) as a "frozen-decoder readout bottleneck" — the solver's latent memory could not cross the discrete token readout without information loss.

**V3: Recurrent Deliberation Controller (this thesis's centerpiece).** You recommended replacing the solver's decoder-bypass with a controller that writes "thought tokens" natively in the LM's embedding manifold. I implemented it exactly as you specified, then iterated.

---

## 2. The Controller — Current Architecture

Frozen backbone: Llama 3.1 8B (Base or Instruct), bf16.
Trainable: 140M params (lowrank residual writer variant, our best).

### 2.1 State & Heads

- **Latent state:** $z \in \mathbb{R}^{1 \times 8 \times 512}$. Learnable init $z_0 \sim \mathcal{N}(0, 0.02^2 I)$.
- **Tapped layers:** $\mathcal{T} = \{8, 16, 24\}$. At each tapped layer, take the residual stream $h^{(\ell)} \in \mathbb{R}^{T \times d}$ and mean-pool over sequence to $\bar h^{(\ell)} \in \mathbb{R}^{d}$.
- **Read head:**
$$\text{feat} = \text{concat}\left[\text{flatten}(\tilde z_{\text{prev}}),\ \bar h^{(8)},\ \bar h^{(16)},\ \bar h^{(24)},\ \text{softmax}(\text{logits}_{ABCD}),\ H[\text{softmax}],\ p_1 - p_2\right]$$
where $\tilde z_{\text{prev}}$ are the post-round hidden states at the thought-slot positions (8 × 4096), $H$ is entropy, $p_1 - p_2$ is top-1–top-2 margin.
- **Read projection:** $\Delta = W_2\, \text{GELU}(W_1\, \text{feat})$ with $W_1 \in \mathbb{R}^{2048 \times \cdot}$, $W_2 \in \mathbb{R}^{(8 \cdot 512) \times 2048}$.
- **State update:** $z \leftarrow \text{RMSNorm}(z + \sigma\, \Delta)$ with learned scalar gate $\sigma$ (init 0.1).

### 2.2 Write Head (lowrank + sparse-vocab-superposition)

$$\text{thought}(z) \;=\; \underbrace{\sum_{k=1}^{64} \text{softmax}(\text{logits}_k(z)) \cdot E[\text{topk}_k(z)]}_{\text{sparse vocab superposition}} \;+\; 0.12\cdot \underbrace{z\, W_\downarrow W_\uparrow}_{\text{lowrank residual}}$$
where $E \in \mathbb{R}^{|\mathcal{V}| \times d}$ is the frozen LM's embedding matrix, $W_\downarrow \in \mathbb{R}^{d_{\text{state}} \times 64}$, $W_\uparrow \in \mathbb{R}^{64 \times d}$. The 0.12 coefficient was swept (tested 0.05, 0.1, 0.12, 0.2, 0.5); 0.12 is the Pareto-optimal trade-off between expressiveness and manifold stability. Lowrank-only is worse; vocab-only hits a ceiling. Their combination **eliminates seed variance**: std drops from 12.8% → 0.1%.

### 2.3 Mid-Layer Injection

Thoughts are **not** prepended to the input. Instead:
- Layers $0..11$ run on `[prompt | answer_prefix]`.
- At layer 12, insert thought tokens: `h ← [h[:, :|prompt|], thought_emb, h[:, |prompt|:]]`.
- Recompute positional embeddings.
- Layers $12..31$ run on extended sequence.

Empirically: L0–L16 injection all give ~83% on SpatialGrid Instruct (layer-invariant). L18–L20 catastrophically fails on mazenav only (42% vs 70% at L17). **Task-specific dead zones exist.** On SpatialGrid, invariance; on mazenav, sharp 30pp cliff at L18–20. No cliff on spatialmap or spatialgrid.

### 2.4 Iteration & Loss

For 5 training rounds:
```
for r = 0 .. 4:
    thoughts = write(z)
    logits, think_h, taps = LM_frozen(prompt, thoughts, answer_prefix)
    ans_logits = logits[:, -1, [A_id, B_id, C_id, D_id]]
    verify = verifier(think_h, taps, softmax(ans_logits))
    if r < 4: z = RMSNorm(z + sigma * read_proj(features))
```

Loss:
$$\mathcal{L} \;=\; \underbrace{\text{CE}(\text{logits}_{\text{final}}, y)}_{\text{final answer}} \;+\; 0.5\cdot \underbrace{\tfrac{1}{R}\sum_r \text{BCE}(v_r, \mathbb{1}[\hat y_r = y])}_{\text{per-round verifier}} \;+\; 0.1\cdot \underbrace{\max(0, \text{CE}_{\text{final}} - \text{CE}_{\text{first}} + 0.1)}_{\text{progress loss}}$$

### 2.5 Training Recipe

- AdamW lr=1e-4, wd=0.05, cosine with 200-step warmup.
- **Gradient accumulation = 16** (critical: std from 12.8% → 0.3%).
- **Total steps = 8000** (critical: 2000 steps fails, 12000+ overfits on small benchmarks).
- Grad clip 1.0.
- Batch size 1 per step (GA=16 simulates batch 16).
- bfloat16 throughout.

---

## 3. The Benchmarks We Test On

### 3.1 SpatialEval (MilaWang/SpatialEval, "tqa" split, NeurIPS 2024)

**SpatialGrid** — counting in tabular grids:
```
Consider a 5x5 grid with animals from ['cat','dog','elephant','giraffe','rabbit']:
elephant | rabbit   | rabbit   | dog     | giraffe
cat      | rabbit   | elephant | dog     | cat
elephant | elephant | giraffe  | giraffe | rabbit
rabbit   | elephant | elephant | rabbit  | rabbit
elephant | cat      | elephant | cat     | cat
How many blocks contain dog?
A. 5  B. 6  C. 2  D. 7
```
Baseline (Llama 8B Inst zero-shot logit-based): ~48–77% depending on sampling. **This is the task where iterative computation uniquely helps.**

**Mazenav** — count direction changes on a *pre-marked* path:
```
#######
#E# # #
#X# # #
#X#  S#
#X###X#
#XXXXX#
How many right turns in the X-path from S to E?
A. 4  B. 8  C. 2  D. 7
```
The X's already mark the path. The task is "trace and count direction changes" — but the path is given. Baseline: ~16% (below random — model systematically mispredicts).

**SpatialMap** — spatial relations among objects.
**SpatialReal** — real-scene spatial questions.

### 3.2 Standard reasoning benchmarks

| Benchmark | Format | Train size | Notes |
|-----------|--------|------------|-------|
| HellaSwag | 4-way SC | ~40k | Commonsense completion |
| WinoGrande | 2-way | ~40k | Coreference resolution |
| BoolQ | 2-way | ~9.4k | Passage → yes/no |
| CommonsenseQA | 5-way | ~9.7k | Commonsense reasoning |
| OpenBookQA | 4-way | ~5k | Science with "book" |
| ARC-Challenge | 4-way | ~1.1k | Science reasoning |

### 3.3 Vision benchmarks (vision-LLM + controller)

- **VSR (Visual Spatial Reasoning, cambridgeltl/vsr_random):** binary True/False on spatial claims about images. Train ~7.7k, test ~2.2k. Images from COCO URLs.
- **PaLiGemma2-10B** (Google, Gemma2 backbone, SigLIP vision): used as frozen VLM.
- **LLaVA-NeXT 7B** (UW-Madison + Microsoft, Mistral backbone, CLIP vision): used as frozen VLM.
- **Llama 3.2 11B Vision** (Meta, downloaded but access still pending).

---

## 4. The Head-to-Head Results — Controller vs LoRA (~500 paired runs)

### 4.1 The LoRA baseline (apples-to-apples)

- **LoRA rank r = 64** on all 7 linear projections per transformer block: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` across all 32 Llama layers.
- Trainable params: **~140M** (matches controller exactly).
- Same train/test splits, same AdamW lr=1e-4 + cosine + GA=16, same 8000 steps, same bfloat16.
- **Single-pass inference** (no K-scaling analog).

### 4.2 Full results table (Δ = accuracy improvement over zero-shot baseline)

#### Benchmarks where LoRA wins

| Benchmark | Model | Controller Δ | LoRA Δ | Winner |
|-----------|-------|--------------|--------|--------|
| HellaSwag | Inst | +11.3pp (3 seeds) | +13.5pp (3 seeds) | LoRA +2.2pp |
| HellaSwag | Base | +12.6pp (2 seeds) | +18.6pp (1 seed) | LoRA +6.0pp |
| WinoGrande | Inst | +7.0pp (3 seeds, range 7.2–9.0) | +18.3pp (3 seeds, identical across replicates) | **LoRA +11.3pp** |
| WinoGrande | Base | +9.1pp (2 seeds) | +25.4pp (2 replicates identical) | **LoRA +16.3pp** |
| CommonsenseQA | Inst | +1.7pp (2 seeds, range 0.4–3.0) | +8.8pp (1 seed) | LoRA +7.1pp |
| OpenBookQA | Inst | +1.9pp (2 seeds) | +2.6pp (1 seed) | ~tie |
| Mazenav | Inst | +40pp (peak 74%) | +73pp (peak 89%) | **LoRA +33pp** |

#### Benchmarks where Controller wins decisively

| Benchmark | Model | Controller Δ | LoRA Δ | Winner |
|-----------|-------|--------------|--------|--------|
| SpatialGrid | Inst | **+39pp** (87.0% ± 0.8% across 5 seeds, all converge) | +1.7pp mean, range **[−52pp at 16k steps, +8.2pp at 8k]** across 6 seeds | **Controller +37pp, stable** |
| SpatialGrid | Base | **+34pp** (~79% stable across all seeds) | **Bimodal**: seeds 100/300 → 90% (+45pp), seeds 42/200 → 20% (−25pp, below random) | **Controller stable; LoRA failure rate 50%** |

#### BoolQ — roughly tied

| Model | Controller | LoRA |
|-------|-----------|------|
| Inst | +6.3pp | +7.2pp |
| Base | +6.2pp | +11.0pp |

### 4.3 Controller's unique properties LoRA cannot replicate

**(a) K-scaling (test-time compute).** Train with 5 rounds, evaluate at 3/5/8. Monotonic-plus-plateau: 8 rounds > 5 rounds > 3 rounds on most benchmarks. LoRA has no such knob.

**(b) Stability.** Across 400+ controller runs, zero catastrophic failures. LoRA breaks 3/4 times at 16k steps on SpatialGrid Inst (deterministic collapse to 25% = always-A). Bimodal on SpatialGrid Base.

**(c) Training compute scales positively.** Controller 8k→16k on HellaSwag: +11.3 → +14.8pp (monotonic improvement). LoRA 8k→16k on SpatialGrid Inst: +1.7 → **−52pp** (catastrophic).

**(d) Frozen backbone preserved.** Multiple controllers could be hot-swapped or combined on a single shared LM. LoRA adapters modify the LM weights (effectively).

### 4.4 Vision — PaLiGemma's +33pp result

- **PaLiGemma2-10B + controller on VSR:**
  - Baseline: 37.0% (below random! PaLiGemma has a calibration bias)
  - L8 (8k steps): 46.7% (+9.7pp)
  - L16 (5k steps): 52.3% (+15.3pp)
  - L24 (8k steps): 69.3% (+32.3pp)
  - **L32 (8k steps, deepest tested): 70.6% (+33.6pp)**
- **LLaVA-NeXT 7B + controller on VSR:** baseline 43.5% → +2.7pp. Poor result. Same architecture pattern (vision encoder + projector + LM) but different LM (Mistral instead of Gemma2). The task works dramatically better on PaLiGemma.
- **LoRA baseline not yet tested on VSR.** Open question.

---

## 5. What's Been Tried (Ablations Completed)

Across ~500 experiments on the controller:

1. **Injection layer sweep** (Inst SpatialGrid): L0 85.4%, L2 83.2%, L4 85.6%, L6 86.8%, L8 85.8%, L12 85.8%, L14 85.0%, L16 85.6%, L18 79.2%, L20 77.8%. **Layer-invariant L0–L16** on SpatialGrid; mild drop after.
2. **Injection layer sweep on mazenav:** sharp dead zone at L18–L20 (42% vs 70% at L17). **Task-specific.**
3. **Base vs Instruct:** Instruct +18pp on SpatialGrid, only +1.5pp on mazenav. Task-dependent benefit of instruction tuning.
4. **Round sweep (training rounds):** 1, 2, 3, 5, 8 rounds. **5 is sweet spot.** 1 is too few (controller can't develop thoughts); 8 causes overfitting.
5. **Round sweep (eval):** at test time 3, 5, 8 all give similar results; beyond 8 plateaus.
6. **Grad accumulation:** 1, 4, 8, 16, 32, 64. GA=16 best. Std 12.8% → 0.3%.
7. **Total steps:** 1500, 3000, 5000, 8000, 12000, 16000. **8000 optimal** for SpatialGrid. 16000 on HellaSwag continues to improve (+11.3 → +14.8pp). 16000 on SpatialGrid plateaus.
8. **Writer architecture:**
   - Vocab-only superposition: hits accuracy ceiling (~78% on SpatialGrid).
   - Lowrank-only: worse (thoughts drift off the LM manifold).
   - **Vocab + 0.12·lowrank: best** (eliminates variance AND hits 87%).
9. **Thought decoding:** decoded nearest vocab neighbors to learned thought tokens — opaque gibberish ("EEPROM", "avatel", etc.). No interpretable concepts. The controller communicates in learned latent codes, not human-readable tokens.
10. **Tapped layers:** (8, 16, 24) — tested others (4, 16, 28), similar. Not sensitive to exact choice as long as evenly spaced.
11. **Controller scale:** 70M, 140M, 280M, 560M, 1.1B. Marginal returns past 140M. Scaling the controller size doesn't close the semantic-task gap.
12. **Multi-task training:** one controller trained on mazenav+spatialgrid+spatialmap jointly. OK but per-task is better (~2–5pp worse than specialist).
13. **Curriculum (1→2→3→5 rounds):** no significant gain over fixed 5.
14. **Self-distillation (5-round teacher → 1-round student):** didn't help.
15. **Per-layer learnable gates** (multi-layer injection): gates got stuck at init (0.119), gradients didn't flow. Not fully debugged; might be fixable.
16. **Router (gumbel-softmax per-input layer selection):** didn't converge cleanly.
17. **Verifier head:** helps training stability, doesn't help inference accuracy.
18. **Attention-only vs FFN-only injection:** both variants tested; full works best. No clean story like we had on DeepPass layer duplication.
19. **Vision layer sweep on PaLiGemma:** L8 → L16 → L24 → L32. Monotonic improvement. **Deeper = better for vision spatial reasoning.** Suggests vision tokens need more LM processing before thoughts help.
20. **Seed variance:** across 10+ seeds at GA=16 + lowrank, std < 1% on converged runs. Training is essentially deterministic.

### What has NOT been tried (or tried and inconclusive)

- **Cross-benchmark zero-shot transfer.** Every benchmark requires its own controller training. Never tested whether a controller trained on HellaSwag works on WinoGrande out of the box.
- **Cross-LLM transfer.** Controller trained on Llama 8B — can it work on Gemma 3 27B with no retraining? Only Llama tested thoroughly.
- **Test-time learning (TTT).** Online gradient updates per-instance on controller params only. Not tried.
- **Retrieval-augmented thoughts.** Letting controller retrieve info from an external store. Not tried.
- **Distillation FROM LoRA TO controller.** Train controller to mimic LoRA's output distribution — would combine benefits? Not tried.
- **Hybrid controller + small LoRA** (joint training). Controller handles reasoning, LoRA handles memorization. Not tried.
- **Very long training** (50k+ steps) on controller to see if LoRA gap closes. Not tried (budget limits).

---

## 6. Mechanistic Hypotheses (to Evaluate)

Below are my current hypotheses for *why* the bifurcation exists. I want you to either validate them with rigorous argument or propose better explanations.

### H1. SpatialGrid requires *per-instance operation execution*; LoRA can't learn operations as static weight updates.

**Argument.** The SpatialGrid answer is $y = \sum_{c \in \text{cells}} \mathbb{1}[c = \text{target}]$. Different instances have different grids, different targets, different counts. No fixed input-to-output mapping exists that LoRA can learn. LoRA's gradient signal is incoherent (same visual tokens sometimes map to A, sometimes D). The best LoRA can do is learn the *prior* over answers, which degenerates to "always predict the modal letter" → 25% at 16k steps (we observe exactly this).

The controller, by contrast, has iterative rounds where each round can attend to a subset of cells, accumulate evidence in latent state, and refine. This is a *program*, not a mapping.

**Falsification:** If this is right, LoRA should also collapse on any counting/tallying task (TallyQA, GQA count, ...). If LoRA works well on other counting tasks, the hypothesis is wrong.

### H2. Mazenav is "secretly pattern-matching" because the path is pre-marked.

**Argument.** X-marked paths have only finitely many ASCII patterns (bounded maze size). Train set covers most patterns, test samples resemble train samples. LoRA memorizes "this X-configuration → 3 right turns" via gradient descent on the association. Controller has no such advantage — its iterative rounds don't help when the answer can be retrieved from surface patterns.

**Falsification:** Redesign mazenav so the path is NOT pre-marked (model must plan). Controller should win, LoRA should fail. Alternatively: on original mazenav, train a "LoRA with position-randomized data augmentation" to break memorization. If LoRA drops, memorization hypothesis confirmed.

### H3. WinoGrande/HellaSwag require updating stored factual associations.

**Argument.** WinoGrande asks "In sentence X, does pronoun refer to option 1 or option 2?" This requires world knowledge: "trophies don't fit in suitcases because suitcases are small." LoRA can directly update the FFN (which is a key-value factual store per Transformer Feed-Forward Layers Are Key-Value Memories, Geva et al. 2021) to sharpen such associations. Our controller's thought tokens cannot modify the FFN's stored factual associations — they can only reshape activations. Hence LoRA wins.

**Falsification:** Probe the controller's internal representation for factual knowledge modification. If we see the controller simply inducing attention patterns that surface already-present facts, the hypothesis is correct. If we see novel facts emerging, it's wrong.

### H4. The controller is fundamentally a *computation* primitive, not a *memorization* primitive.

**Argument (unifying H1–H3).** Define two idealized capabilities:
- **Computation:** compute $f(x)$ where $f$ is an operation over the input structure (counting, path tracing, logical deduction).
- **Memorization:** recall $g(c)$ where $g$ is a cached association (coreference resolution, commonsense lookup).

The controller's iterative architecture is a natural fit for computation (fixed-point semantics, Turing-completeness in the limit of infinite iterations). LoRA's weight-adjustment architecture is a natural fit for memorization (adjust stored key-value pairs).

If this is correct, the optimal system is **hybrid**: controller handles per-instance computation, LoRA (or equivalent) handles memorization.

### H5. LoRA's catastrophic collapse on SpatialGrid is a "degenerate minimum of high-entropy label distributions."

**Observation.** SpatialGrid train labels are ~uniform over A/B/C/D (entropy ≈ log 4). WinoGrande train labels are binary and roughly balanced. In both cases, if LoRA fails to learn the task, it can degenerate to "predict the marginal label distribution." On WinoGrande, this gives ~50% accuracy (non-catastrophic). On SpatialGrid, ~25% (catastrophic-looking). Thus: same failure mode, different appearance.

**Falsification:** Check if LoRA's entropy on SpatialGrid outputs is close to $\log 4$ at end of training. If yes, it's predicting uniform → degenerate solution confirmed. Also check if LoRA's final logits are independent of input grid content (ablate the grid).

### H6. K-scaling is equivalent to implicit depth scaling, bounded by the LM's intrinsic fixed-point convergence.

**Argument.** Each round of the controller effectively re-runs the LM with updated input. If the round-to-round dynamics $z_{r+1} = \Phi(z_r)$ converge to a fixed point $z^*$, then K-scaling approaches $z^*$ asymptotically. The rate of convergence is bounded by the spectral radius of the Jacobian $\|\nabla_z \Phi\| < 1$. If $\|\nabla_z \Phi\| \approx 1$, convergence is slow and more rounds help. If already converged, more rounds don't help.

This predicts: K-scaling gains should correlate with how "unconverged" the 1-round solution is, which correlates with task difficulty (distance from the model's zero-shot capability).

**Falsification:** Measure $\|z_{r+1} - z_r\|$ across rounds. Correlate with accuracy. Spectral radius of the learned update should predict K-scaling gain.

---

## 7. Deep Open Questions For You

### Q1. Derive the LoRA-collapse threshold on SpatialGrid.

Given Llama 3.1 8B's FFN dimension (14336) and attention head dim (128), and the SpatialGrid label entropy (≈ log 4), at what training step should we expect LoRA rank-64 to collapse to the degenerate minimum? Derive a formula or bound. This would turn the empirical "collapses at 16k steps" into a principled statement.

### Q2. Is there a unifying theorem for iterative-computation vs static-memorization?

Consider a learned function class $\mathcal{F}$ parameterized by $\theta$ with $|\theta| = P$ (fixed budget). Define:
- **Static expressivity:** $E_s(\mathcal{F}) = $ VC dim or Rademacher complexity of $f_\theta$.
- **Iterative expressivity:** $E_i(\mathcal{F}, K) = $ expressivity of $f_\theta^K$ (K-fold composition).

Is there a theorem like "$E_i(\mathcal{F}, K) / E_s(\mathcal{F}) \geq g(K)$ for task class $\mathcal{T}$"? If so, what conditions on $\mathcal{T}$ are required?

Related literature: universal transformers (Dehghani 2018), ALBERT layer sharing, DEQs (Bai et al. 2019), COCONUT latent CoT (Hao 2024), MoR (Bae 2024). What theoretical framework best fits our data?

### Q3. How to add memorization capability to the controller WITHOUT losing its stability?

Several possibilities — rank them:

(a) **Thought-gated LoRA.** Learnable mask $m_t(z) \in [0,1]$ modulates a small LoRA delta: $\Delta W = m_t(z) \cdot BA$. Thoughts decide when/how much to edit weights.

(b) **Soft prompt extension.** Controller writes both thought tokens and a "memory token" that gets added to the FFN's key matrix at inference.

(c) **Retrieval head.** Thoughts query an external key-value store (e.g., vocabulary embeddings as keys, rich factual vectors as values).

(d) **Parallel tracks.** Controller runs in parallel with a small fixed LoRA; their outputs mix via a learned gate per-token.

(e) **Metalearned weight updates.** Use hypernetwork / Mesa-optimizer (Von Oswald 2023) where the controller produces low-rank weight updates per instance.

For each, assess: (i) parameter budget, (ii) training stability, (iii) inference speed, (iv) expected gain on WinoGrande/HellaSwag, (v) expected preservation of SpatialGrid win.

### Q4. Zero-shot transfer — is there a multi-task meta-training regime that produces a *universal* controller?

Suppose we train a single controller on 6 benchmarks simultaneously (HS+WG+BQ+CSQA+SG+MZ). Does it transfer to a 7th held-out benchmark (e.g., ARC-Challenge, RACE, SocialIQA)? What's the theoretical basis for expecting this to work (or not)?

Related: MoR, MetaICL, FLAN-T5. Is the controller's per-benchmark-training requirement a fundamental limitation or a training-data artifact?

### Q5. The PaLiGemma win is a diamond — why and how to generalize?

Controller gives **+33pp on PaLiGemma + VSR**, vs only **+2.7pp on LLaVA + VSR**. Both are vision-LM. The LMs differ:
- PaLiGemma: Gemma2-3.5B, SigLIP vision, 42 LM layers, 3584 hidden.
- LLaVA: Mistral-7B, CLIP-L/14 vision, 32 LM layers, 4096 hidden.

And the tasks differ in how the vision tokens are fused (PaLiGemma prepends image tokens uniformly; LLaVA uses image-anywhere tokens with a unique `<image>` placeholder).

Derive: which architectural property of PaLiGemma makes the controller effective, and is there a principled way to predict which VLMs will be "controller-friendly"?

**Experiments to propose.** Outline 3 VLM architectures and 3 vision-reasoning benchmarks where the controller is *predicted* to give maximum gain vs LoRA.

### Q6. The "Fine-Tuning Collapse on Structured Reasoning" paper

I think the catastrophic LoRA collapse at 16k steps on SpatialGrid, reproducible across seeds, is a publishable finding in its own right. Can you:

1. Derive why this happens (Q1).
2. Connect to known phenomena: "model collapse" from self-distillation (Shumailov 2023), "mode collapse" in GANs, representation collapse in SSL (Chen 2020, BYOL), catastrophic forgetting (McCloskey 1989).
3. Propose a diagnostic metric that would predict collapse *before it happens* (e.g., a gradient-norm pattern, a rank-decay signal in the LoRA matrices).
4. Suggest a fix for LoRA that would prevent the collapse while preserving its good behavior on WinoGrande. (Regularization? Lower rank? Adaptive alpha? Warm-up schedule?)

### Q7. What's the strongest paper narrative?

Given the evidence, rank these three framings by chance of acceptance at a top venue (NeurIPS, ICLR, ICML) and say why:

**F1: "Task-Conditional Advantage of Iterative Latent Computation"**
- Core claim: iterative computation beats weight updates on structured reasoning tasks where the answer is an *operation* on input structure; loses on semantic memorization tasks.
- Flagship result: SpatialGrid +37pp over LoRA at matched compute.
- Mechanistic argument via H1–H4.

**F2: "Fine-Tuning Collapse on Structured Reasoning: When LoRA Catastrophically Fails"**
- Core claim: LoRA at 140M params catastrophically collapses to label prior on high-entropy structured reasoning tasks. Stable frozen-backbone controllers avoid this.
- Flagship result: 3/3 LoRA runs at 16k steps on SpatialGrid Inst collapse to 25%, deterministically; our controller: 87% stable.
- Framing: LoRA is not a panacea; here's a failure mode and a fix.

**F3: "Composable Test-Time-Scaling Modules for Frozen LLMs"**
- Core claim: a 140M controller gives K-scaling at inference (test-time compute lever LoRA lacks) and is parameter-efficient and composable.
- Flagship result: K-scaling monotonic improvement + multi-benchmark gains + frozen-backbone reusability.
- Framing: a new class of LLM add-ons.

What's the strongest? What additional experiments would nail it? What's the honest ceiling — is there a realistic scenario where this work is accepted at a main track?

### Q8. Concrete Action Plan — 3 Experiments This Week

Given 4×B200 for 2 weeks, suggest the **3 highest-value experiments** I should run now. For each:

- **Exact implementation** (code changes, hyperparams).
- **Expected result** (point estimate, approximate).
- **GPU-hours required.**
- **What it would prove or disprove.**
- **How it feeds the paper.**

Assume I can run ~5 concurrent jobs, each ~2 hours. So I can run ~40-60 experiments in a week if each is 2-4 hours.

### Q9. Theoretical depth — is there a unifying framework?

Our findings across DeepPass (layer duplication), Solver (bidirectional memory), Controller (iterative thoughts), and TRM (recursive 7M model) all point to "iteration over fixed weights > single pass" for reasoning tasks. Is there a deeper principle — information-theoretic, dynamical-systems, or representation-theoretic — that unifies these? Candidate frames:

- **Fixed-point dynamics / DEQ.** Iteration converges to a fixed point of a learned operator; more iterations refine the solution.
- **Implicit depth.** K rounds on a D-layer model ≈ K×D effective layers for certain tasks.
- **Compute-optimal training.** Given a fixed parameter budget, iterative models are more compute-efficient in a regime determined by task's intrinsic "iteration demand."
- **Representation smoothing.** Each iteration smooths the representation along a learned vector field; converges to a stable attractor.
- **Bayesian inference.** Rounds = sampling from a posterior; more rounds = better posterior estimate.

Which framework best fits our data? Are there testable predictions that distinguish them?

### Q10. Challenge me

Tell me honestly:
- Is this line of research going anywhere?
- Is my paper framing wrong?
- Am I over-investing in SpatialEval (a niche benchmark)?
- Should I abandon the controller and just use LoRA + prompt engineering?
- What's the single most important thing I'm missing?

Don't flatter me. If the project's best outcome is a workshop paper, say so. If it's a PhD-level research direction, say that too.

---

## 8. Practical Constraints

- **American LLMs only** (Meta Llama, Google Gemma). No Qwen, Chinese-developed models for new experiments.
- **4×B200 GPUs for 2 weeks** until undergraduate thesis deadline.
- Must be reproducible with open-source code.
- No TPU, no 1000-GPU cluster.
- Undergraduate thesis — target venue: workshop at NeurIPS/ICLR or specialized journal.

## 9. What I Want From You

1. **Diagnose** — Mechanistic arguments for the bifurcation, including challenges to my H1–H6 hypotheses. Accept or reject each with mathematical argument.
2. **Design** — Three concrete architectural/training modifications (ranked). Include math, pseudocode, expected gain with derivation.
3. **Action plan** — Three experiments for this week with exact specs and predictions.
4. **Paper framing** — Strongest narrative with evidence gap analysis. List missing experiments that would sharpen the story.
5. **Deeper theory** — A unifying framework (dynamical systems, information geometry, whatever fits) with testable predictions.
6. **Honesty check** — Challenge my framing; if the work is fundamentally limited, say so.

Think step by step. Use equations. Cite literature. Give me your most rigorous + creative analysis. I trust you to tell me when I'm wrong.

---

## 10. Appendix — Data & Artifacts

**Code:** `/blue/cis4914/jietao/DeepPass/solver/`
- `recurrent_deliberation.py` — base controller class.
- `eval_deliberation_creative.py` — MidLayerDeliberation + LowrankDeliberation.
- `mega_runner.py` — SpatialEval training/eval pipeline.
- `mega_runner_benchmarks.py` — HellaSwag/WG/BQ/CSQA/OBQA/ARC pipeline.
- `mega_runner_lora.py` — LoRA baseline pipeline.
- `mega_runner_lora_spatial.py` — LoRA baseline on SpatialEval.
- `eval_deliberation_vision.py` — vision-LM controller (LLaVA/PaLiGemma/Mllama).

**Results:** `/blue/cis4914/jietao/DeepPass/results/data/`
- `mega/` — 400+ SpatialEval controller runs (multi-seed, layer sweep, step sweep).
- `benchmarks/` — controller on HellaSwag/WG/BQ/CSQA/OBQA (18 JSON results).
- `lora_baseline/` — LoRA matched baselines (25 JSON results).
- `vision/` — controller on VSR (PaLiGemma L8/L16/L24/L32, LLaVA).
- `72b/`, `gemma3_27b/`, `qwen35/`, `moe/`, `quantization/` — DeepPass Track A results.

**Paper outline:** `/blue/cis4914/jietao/DeepPass/PAPER.md` (275 lines).
**Experimental history:** `/blue/cis4914/jietao/DeepPass/HISTORY.md` (~2700 lines).
**Previous GPT-5.4 Pro consultations:** `/blue/cis4914/jietao/DeepPass/prompts/` (20+ prior prompts, including `gpt54_generalization_v3.md` (503 lines) which is the most recent predecessor to this one).

**Related prior work:**
- LiquidTRM (CfC-inspired fractional second-pass architecture), at `/blue/cis4914/jietao/LNN/`.
- TRM interpretability (Alexia's 7M model): `/home/jietao/RR/SeniorProject/RR_Interpretability/`.

---

Please begin. No preamble. Proof-level rigor. High creativity.
