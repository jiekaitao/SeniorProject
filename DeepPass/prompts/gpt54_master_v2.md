# MASTER BRIEFING v2 — "LLMs Have ADHD" Senior Thesis: Full Research Context, Updated Data, and the Honest Open Problem

## 0. Instructions to You (GPT-5.4 Pro)

You are the technical director of this research direction. I need **proof-level mathematical rigor** — not hand-waving. For every claim:

- Provide mechanistic reasoning with equations or linear-algebra arguments.
- Cite relevant literature (COCONUT, Recurrent-Depth, MoR, Dr.LLM, Quiet-STaR, TTT, BLOOD, RYS, TRM, ACT, universal transformers, DEQs, HyperNetworks, OFT, VeRA, AdaLoRA, DoRA, Mesa-optimizers, fast-weights).
- Propose at least one falsifiable experiment that would verify or disprove each claim.
- Where possible, **derive** behavior from first principles (loss-landscape geometry, information bottleneck, representation theory, fixed-point dynamics, mean-field NN analysis, NTK, implicit bias).

I want **high creativity**. If the best answer requires inventing a new architecture, loss, or inference procedure — propose it. Don't settle for incremental tweaks.

I want **honesty**. Challenge my framing. If the work has a fundamental ceiling, say so. **I am specifically asking: does this line of research actually improve "general intelligence," or is it a niche capability?** If it's niche, help me position it honestly.

Output budget: as long as needed. Aim for ≥20 dense pages.

---

## 1. TL;DR — What I Need

After ~550 GPU experiments on 4×B200, I have a learned **recurrent deliberation controller** (~140M params) that bolts onto a frozen Llama 3.1 8B and writes iterative thought tokens into the residual stream. It **decisively beats parameter-matched LoRA on SpatialGrid** (+37pp, stable across 5 seeds) but **loses to LoRA on most semantic benchmarks** (WinoGrande, HellaSwag, CSQA, Mazenav).

The key new finding since last consultation: **LoRA's failure mode on SpatialGrid is perfectly deterministic within seed and highly bimodal across seeds.** Base LoRA SpatialGrid is 5/10 catastrophic collapses to 20% vs 5/10 successes at 90%. Inst LoRA SpatialGrid uniformly flatlines at +0 to +8pp and catastrophically collapses 3/3 times at 16k training steps. Our controller has **zero failures across 400+ runs**.

I need you to answer, in depth:

1. **Does iterative latent computation actually improve general intelligence, or is it a niche capability?** If niche, what's the honest paper narrative?
2. **Why** does iterative latent computation win on structured reasoning and lose on semantic tasks? Derive it mathematically, challenging my hypotheses.
3. **How** to redesign the controller so it wins everywhere — ideally zero-shot across benchmarks. Propose 3 concrete architectures with math and expected gain.
4. **Which 3 experiments** should I run this week (4×B200, 2 weeks to thesis deadline)?
5. **What's the strongest paper framing** given the evidence?
6. **Is there a deeper theoretical framework** (dynamical systems, information geometry, implicit depth) that unifies our findings?

---

## 2. Project Genealogy — Two Tracks

### Track A: DeepPass — Zero-Shot Layer Duplication (no training required)

We extended David Ng's RYS ("Repeat Your Self") discovery with:

**Spectral screening.** For a candidate duplicate block $[i, j)$:
$$\rho(i,j) = \frac{\|F_{i:j}(F_{i:j}(h)) - F_{i:j}(h)\|}{\|F_{i:j}(h) - h\|}$$
Low rho = safe to iterate. Reduces search 162×. 80% top-5 hit rate on 7B.

**SBUID screening metric on 72B:**
$$\text{SBUID}_0 = \text{BLOOD-impact} - 6000 \cdot \rho$$
First statistically significant screening metric on 72B: Spearman r=0.515, p=0.008. Cross-validated (train r=0.34 → test r=0.664).

**Greedy stacking + per-layer alpha.** 72B triple with 21 per-layer alphas: **84.07, +7.31 over Ng's 76.76**. No training, no new weights.

**Mechanistic finding.** Attention benefits from repetition (re-weighting operation). FFN hurts (key-value retrieval corrupted by perturbed query). Attention-only duplication resolves the trade-off. **"FFN re-retrieval corruption" hypothesis.** Supported by sublayer data: L2 attention-only (80.35) > full (77.45).

Cross-architecture: Qwen2-72B, Qwen3.5-27B/9B, Gemma3-27B, Qwen3-30B MoE, Llama 3.1 8B. 4-bit NF4 preserves gains. Inference overhead: 11% for a pair, 20% for a quad.

**TRM cross-scale validation.** Alexia's Tiny Recursive Model (7M params, 30×30 maze solver) shows the same attention-vs-FFN pattern as DeepPass (27B–72B). Both demonstrate: **iterative attention refinement is the primary driver of reasoning, at any scale.**

### Track B: Solver → Recurrent Deliberation Controller

**V1 SolverCore (bidirectional).** Bolts onto frozen Llama with two-level z_L/z_H hierarchy, shared weights across K iterations, projects 4096↔512.

**SpatialEval hit a 72% ceiling.** Diagnosed by you (GPT-5.4 Pro, prior consultation) as "frozen-decoder readout bottleneck" — solver's latent couldn't cross discrete token bottleneck.

**V3 Recurrent Deliberation Controller (current centerpiece).** Per your recommendation: controller writes thought tokens natively in the LM embedding manifold. Iterates across rounds. ~140M trainable params, frozen 8B backbone.

This system unlocked the 72% ceiling — SpatialGrid Instruct now **87.0% ± 0.8%** across 5 seeds (baseline ~48%).

---

## 3. The Current Controller — Architecture

Frozen Llama 3.1 8B (Base or Instruct), bf16.

### 3.1 State & Heads

- **Latent state:** $z \in \mathbb{R}^{1 \times 8 \times 512}$. Learnable init $z_0 \sim \mathcal{N}(0, 0.02^2 I)$.
- **Tapped layers:** $\mathcal{T} = \{8, 16, 24\}$. At each, mean-pool the residual stream over sequence.
- **Read head:** concatenates [flattened slot hidden states, mean-pooled taps, softmax(ABCD), entropy, margin] → Linear(2048) → GELU → Linear(8·512).
- **State update:** $z \leftarrow \text{RMSNorm}(z + \sigma \Delta z)$, $\sigma$ learnable scalar gate (init 0.1).

### 3.2 Write Head — Lowrank + Sparse Vocab Superposition

$$\text{thought}(z) = \underbrace{\sum_{k=1}^{64} \text{softmax}(\text{logits}_k(z)) \cdot E[\text{topk}_k(z)]}_{\text{sparse vocab superposition}} + \ 0.12 \cdot \underbrace{z \cdot W_\downarrow W_\uparrow}_{\text{lowrank residual}}$$

where $E \in \mathbb{R}^{|\mathcal{V}| \times d}$ is the frozen LM's embedding matrix, $W_\downarrow \in \mathbb{R}^{512 \times 64}$, $W_\uparrow \in \mathbb{R}^{64 \times 4096}$. The 0.12 coefficient was swept (0.05, 0.1, 0.12, 0.2, 0.5); 0.12 is the Pareto-optimal trade-off. **Eliminates seed variance**: std 12.8% → 0.1%.

### 3.3 Mid-Layer Injection

Thoughts are **NOT** prepended. Instead:
- Layers 0..11 run on `[prompt | answer_prefix]`.
- At layer 12, splice: `h ← [h_prompt, thought_emb, h_answer]`.
- Recompute positional embeddings.
- Layers 12..31 run on extended sequence.

Empirical: L0–L16 injection all give ~83% on SpatialGrid Instruct (layer-invariant). L18–L20 **catastrophically fails on mazenav only** (42% vs 70% at L17). Task-specific dead zone.

### 3.4 Training — Proven Recipe

- AdamW lr=1e-4, wd=0.05, cosine with 200-step warmup.
- **GA=16** (critical — killed seed variance).
- **8000 steps** (critical — at 2000 nothing worked).
- 5 rounds per training step; K-scaling at eval (3, 5, 8).
- Loss: $\text{CE}_{\text{final}} + 0.5 \cdot \frac{1}{R}\sum_r \text{BCE}(v_r, \mathbb{1}[\hat y_r = y]) + 0.1 \cdot \max(0, \text{CE}_{\text{final}} - \text{CE}_{\text{first}} + 0.1)$.

---

## 4. The Benchmarks

### 4.1 SpatialEval (NeurIPS 2024)

**SpatialGrid** (counting in tabular 5×5 grids):
```
Consider a 5x5 grid with animals from ['cat','dog','elephant','giraffe','rabbit']:
elephant | rabbit   | rabbit   | dog     | giraffe
cat      | rabbit   | elephant | dog     | cat
elephant | elephant | giraffe  | giraffe | rabbit
rabbit   | elephant | elephant | rabbit  | rabbit
elephant | cat      | elephant | cat     | cat
How many blocks contain dog? A. 5  B. 6  C. 2  D. 7
```
Llama 8B Inst baseline ~48–77%. Random = 25%. **Target task for iterative computation.**

**Mazenav** (count right turns on pre-marked path):
```
#######
#E# # #
#X# # #
#X#  S#
#X###X#
#XXXXX#
How many right turns from S to E? A. 4  B. 8  C. 2  D. 7
```
Baseline ~16% (**below random**). The X path is already marked — task is *secretly* pattern matching, not planning.

**SpatialMap, SpatialReal** — other SpatialEval tasks.

### 4.2 Standard Reasoning

HellaSwag (4-way), WinoGrande (2-way), BoolQ (2-way), CommonsenseQA (5-way), OpenBookQA (4-way), ARC-Challenge (4-way).

### 4.3 Vision

**VSR** (Visual Spatial Reasoning, cambridgeltl/vsr_random): binary True/False on spatial claims about images. Train 7.7k, test 2.2k. Images from COCO URLs.

Backbones tested:
- **PaLiGemma2-10B** (Google, Gemma2 backbone, SigLIP vision).
- **LLaVA-NeXT 7B** (UW-Madison + MSR, Mistral backbone, CLIP vision).
- Llama 3.2 11B Vision (access pending).

---

## 5. The Head-to-Head — Controller vs LoRA (~550 paired runs)

### 5.1 LoRA baseline spec

- **Rank r = 64** on all 7 linear projections per layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj across all 32 Llama layers. **~140M trainable params** (matches controller).
- Same train/test splits, AdamW lr=1e-4 + cosine + GA=16, 8000 steps, bfloat16.
- Single-pass inference.

### 5.2 Updated Results — The Bifurcation Is Real

**Benchmarks where LoRA wins:**

| Benchmark | Model | Controller Δ (n seeds) | LoRA Δ (n seeds) | LoRA − Controller |
|-----------|-------|------------------------|-------------------|-------------------|
| HellaSwag | Inst | +11.3pp (3) | +13.5pp (3) | +2.2pp |
| HellaSwag | Base | +12.6pp (2) | +18.6pp (1) | +6.0pp |
| WinoGrande | Inst | +7.0pp (3, range 7.2–9.0) | +18.3pp (3, replicates identical) | **+11.3pp** |
| WinoGrande | Base | +9.1pp (2) | +25.4pp (2, identical) | **+16.3pp** |
| CommonsenseQA | Inst | +1.7pp (2, range 0.4–3.0) | +8.8pp (1) | +7.1pp |
| OpenBookQA | Inst | +1.9pp (2) | +2.6pp (1) | ~tie |
| Mazenav | Inst | +40pp (peak 74%) | +73pp (peak 89%) | **+33pp** |

**Benchmarks where Controller wins decisively:**

| Benchmark | Model | Controller Δ (multi-seed) | LoRA Δ (multi-seed) |
|-----------|-------|---------------------------|----------------------|
| **SpatialGrid** | Inst | **+39pp stable** (87.0% ± 0.8% across 5 seeds) | +1.7pp mean, range **[−52pp @ 16k steps, +8.2pp @ 8k]** across 6 seeds; **0/6 above controller** |
| **SpatialGrid** | Base | **+34pp stable** (~79% across all seeds) | **Bimodal**: s100/s300 → 90% (+45pp), s42/s200 → 20% (−25pp, **below random**) — **5/10 catastrophic failures** |

**BoolQ — tied:**

| Model | Controller | LoRA |
|-------|-----------|------|
| Inst | +6.3pp | +7.2pp |
| Base | +6.2pp | +11.0pp |

### 5.3 The Striking New Finding — LoRA Determinism + Bimodality

Across ~40+ replicate LoRA runs, two facts emerge:

(a) **LoRA is perfectly deterministic within seed.** Same seed, same model, same hyperparams → identical final accuracy to 4 decimal places across 4+ replicates. This rules out training stochasticity.

(b) **LoRA is wildly bimodal across seeds on SpatialGrid Base.** 5/10 seeds converge to 90% (+45pp). 5/10 seeds collapse to 20% (below random, −25pp). Initialization fully determines outcome. **There's a sharp decision boundary in the LoRA loss landscape where SpatialGrid is on a knife's edge.**

(c) **16k-step LoRA on SpatialGrid Inst reliably collapses** to 25% across all 3 replicate runs. Longer training → catastrophic decay.

Compare:
- Controller Std < 1% across 10+ seeds.
- Controller 8k→16k steps: monotonically improves (HellaSwag +11.3 → +14.8pp).
- LoRA 8k→16k steps on SpatialGrid: catastrophically breaks (+1.7pp → −52pp).

### 5.4 Controller's Unique Properties LoRA Cannot Replicate

1. **K-scaling** at test time: 3 → 5 → 8 rounds monotonically improves. No LoRA analog.
2. **Stability** across seeds, steps, hyperparams. Zero failures.
3. **Training-compute scaling** (monotonically improves with more steps). LoRA has catastrophic regimes.
4. **Frozen backbone preserved** — controllers are composable / hot-swappable.

### 5.5 Vision — The PaLiGemma Diamond

**PaLiGemma2-10B + controller on VSR:**
- Baseline: **37.0%** (below random! PaLiGemma has a True/False calibration bias.)
- Inject L8 (8k): 46.7% (+9.7pp)
- Inject L16 (5k): 52.3% (+15.3pp)
- Inject L24 (8k): 69.3% (+32.3pp)
- **Inject L32 (8k): 70.6% (+33.6pp) — deeper = better, monotonic**

**LLaVA-NeXT 7B + controller on VSR:**
- Baseline 43.5%, final 46.2% (+2.7pp).

Both are VLMs with similar architecture. PaLiGemma gets **+33.6pp**; LLaVA gets **+2.7pp**. Same controller, same benchmark, wildly different outcomes. LoRA baseline on VSR not yet tested.

---

## 6. Full Ablation Summary (ran 500+ experiments)

### What we've varied

1. **Injection layer** (L0–L20): layer-invariant on SpatialGrid; task-specific dead zone at L18–L20 on mazenav.
2. **Base vs Instruct**: +18pp on SpatialGrid (Instruct advantage), +1.5pp on mazenav (small).
3. **Training rounds** (1, 2, 3, 5, 8): 5 is sweet spot.
4. **Eval rounds (K-scaling)** (3, 5, 8): monotonically improves, plateaus at 8.
5. **GA** (1, 4, 8, 16, 32, 64): GA=16 optimal.
6. **Steps** (1500, 3000, 5000, 8000, 12000, 16000): 8000 for SpatialGrid, 16000 better for HellaSwag.
7. **Writer architecture**: vocab-only (ceiling), lowrank-only (worse), vocab+lowrank(0.12) (best).
8. **Thought decoding**: opaque gibberish ("EEPROM", "avatel"). Not human-interpretable.
9. **Controller scale**: 70M, 140M, 280M, 560M, 1.1B. Marginal gains past 140M.
10. **Multi-task training**: jointly on maze+grid+map. Slightly worse than per-task (~3pp).
11. **Attention-only / FFN-only injection**: no clean gain from either.
12. **Vision layer sweep** (PaLiGemma): L8→L16→L24→L32 monotonically improves.

### What we haven't tried (or inconclusive)

1. **Zero-shot transfer across benchmarks.** Never tested whether a HellaSwag-trained controller works on WinoGrande.
2. **Cross-LLM transfer.** Only Llama tested.
3. **Test-time learning (TTT).** Online gradient updates on controller only.
4. **Retrieval-augmented thoughts.**
5. **Distillation from LoRA → controller.** Would teach the controller LoRA's outputs.
6. **Hybrid controller + small LoRA** (joint training).
7. **Very long training** (50k+) on controller.
8. **Longer-context prompts** (>1900 tokens).
9. **Chain-of-thought-style explicit thinking** (let controller emit *text* tokens between rounds, not just latent).
10. **Per-task thought-token specialization** (learn different thought "vocabularies" for different task types).

---

## 7. Mechanistic Hypotheses (H1–H6, to Evaluate)

### H1: SpatialGrid requires per-instance operation execution; LoRA can't learn operations.

The SpatialGrid answer is $y = \sum_{c} \mathbb{1}[c = \text{target}]$. Different grids, different targets, different counts. No fixed input→output mapping exists. LoRA's gradient signal is incoherent (same surface tokens → different labels). Best LoRA can do is learn the label prior → degenerate to modal-letter prediction → 25%. Controller has iterative rounds, can tally in latent state.

**Falsification:** test LoRA on TallyQA, other counting tasks. If LoRA fails on all counting, hypothesis confirmed.

### H2: Mazenav is "secretly pattern matching" (path pre-marked).

X-patterns have finite set. Train covers most. LoRA memorizes "this X-config → 3 turns." Controller's iteration doesn't help when answer is surface-retrievable.

**Falsification:** redesign mazenav without X pre-marking. Controller should win, LoRA should fail.

### H3: WinoGrande/HellaSwag require updating stored factual associations.

FFN layers store facts as key-value memories (Geva 2021). LoRA directly updates them. Controller can only reshape activations, not modify stored facts.

**Falsification:** probe controller's internal rep for factual modification. If only attention reshaping visible, hypothesis confirmed.

### H4: Unification — computation vs memorization primitives.

Two idealized capabilities:
- **Computation** (operation over input structure) — iteration is the natural primitive.
- **Memorization** (cached association) — weight update is the natural primitive.

Optimal system: **hybrid**. Controller for computation, LoRA (or equivalent) for memorization.

### H5: LoRA's "catastrophic collapse" = degenerate minimum of high-entropy labels.

SpatialGrid train labels are ~uniform over A/B/C/D (entropy ≈ log 4). If LoRA fails to learn the task, it degenerates to predicting the marginal. On 4-way uniform → 25% (catastrophic). On 2-way balanced → 50% (less obvious).

**Falsification:** check LoRA output entropy at end of training on SpatialGrid. If close to log 4, confirmed. Ablate the grid content (same prompt structure, random grid): if LoRA's output is invariant, confirmed degenerate.

### H6: K-scaling = implicit depth scaling, bounded by fixed-point convergence.

Round-to-round dynamics $z_{r+1} = \Phi(z_r)$ has a fixed point $z^*$. More rounds → closer to $z^*$. Rate bounded by spectral radius $\|\nabla_z \Phi\|$.

**Falsification:** measure $\|z_{r+1} - z_r\|$ per round. Correlate with K-scaling gain.

---

## 8. The Deep Questions

### Q1. Does our controller actually improve "general intelligence"?

Be brutally honest. My take: **no.** Here's what the evidence shows:

- Fails on semantic tasks (WinoGrande, HellaSwag) at matched compute.
- No zero-shot transfer across benchmarks.
- Can't add new knowledge or expand LLM capabilities.
- Wins only on structured computation (SpatialGrid) where fine-tuning can't learn the *operation*.

So we've built a **specific capability** (per-instance structured reasoning), not a **general upgrade**. K-scaling is cute but doesn't generalize.

Do you agree? If yes, help me position the paper. If no, what am I missing?

### Q2. Derive the LoRA catastrophic collapse threshold.

LoRA rank 64 × 8B Llama (FFN dim 14336, attn head dim 128) × SpatialGrid (label entropy ≈ log 4, 4-way) → collapses at ~16k steps, reproducibly across seeds.

Derive:
1. A formula for when LoRA will collapse as a function of (rank, label entropy, training steps, lr). Use NTK or implicit-bias analysis.
2. A diagnostic metric (gradient norm pattern, rank-decay signal in $BA^T$, anything) that predicts collapse **before** it happens.
3. A fix for LoRA that avoids the collapse (adaptive rank, regularization, warm-up schedule, ...). Want to know: is LoRA collapse preventable, or is it fundamental to the class of fine-tuning methods?

### Q3. Design 3 hybrid architectures (rank by expected impact).

Each must: (i) preserve controller's SpatialGrid stability, (ii) close the WinoGrande/HellaSwag gap vs LoRA. Be specific with math + pseudocode + expected gain.

Candidate directions:
- **Thought-gated LoRA.** Controller decides when to apply small LoRA delta.
- **Parallel tracks with gate.** Controller + small LoRA, output mixed by per-instance gate.
- **Metalearned weight updates.** Hypernetwork (Mesa-optimizer, Von Oswald 2023) produces low-rank deltas per instance.
- **Fast-weights / attention-as-memory.** Controller writes to fast-weight memory, reads back.
- **OFT / DoRA / VeRA / AdaLoRA.** Alternative parameter-efficient families — do they behave differently from LoRA on SpatialGrid?

### Q4. Design a zero-shot controller.

Multi-task meta-training on 6 benchmarks simultaneously, test on 7th. What's the minimum task-diversity / data-diversity to get transfer? MetaICL-like setup? MoR-style router?

### Q5. What makes PaLiGemma (+33pp) different from LLaVA (+2.7pp)?

Same task, same controller, same hyperparams, same param count. Different VLMs. Why?

Architectural differences:
- PaLiGemma: Gemma2-3.5B LM, SigLIP vision, 42 layers, 3584 hidden. Image tokens prepended uniformly.
- LLaVA: Mistral-7B LM, CLIP-L/14 vision, 32 layers, 4096 hidden. Image tokens use `<image>` placeholder replaced at specific position.

Also: PaLiGemma is **below random** on VSR (37% baseline). LLaVA is above random (43.5%).

Hypotheses:
(a) PaLiGemma's True/False calibration is miscalibrated — controller recalibrates (easy gain).
(b) Gemma2 backbone is "controller-friendlier" than Mistral because of different attention patterns.
(c) Uniform image-token prepending in PaLiGemma means thoughts interact cleanly with vision; LLaVA's placeholder replacement makes mid-layer injection awkward.

Derive which is correct. Predict 3 VLMs and 3 benchmarks where controller will give maximum gain.

### Q6. The LoRA catastrophic collapse is a paper on its own.

LoRA at 16k steps on SpatialGrid: deterministic 25% collapse across all 3 replicates. Is this:
1. A rare corner case?
2. A general phenomenon (any high-entropy structured reasoning task)?
3. Connected to known phenomena (model collapse, mode collapse, representation collapse, catastrophic forgetting)?

Propose:
1. Theoretical framework for the collapse.
2. Diagnostic metric that predicts it.
3. Benchmark suite to characterize the phenomenon (tasks where LoRA should fail).
4. Fix that preserves LoRA's good behavior.

### Q7. What's the strongest paper narrative?

Rank these three framings:

**F1: "Task-Conditional Advantage of Iterative Latent Computation."**
- Core claim: iterative computation beats weight updates on structured reasoning tasks where answer = operation on input structure.
- Flagship: SpatialGrid +37pp over LoRA at matched compute.
- Risk: venue reviewers ask "what about MMLU, BBH, GSM8K?" and we don't have answers.

**F2: "Fine-Tuning Collapse on Structured Reasoning: When LoRA Catastrophically Fails."**
- Core claim: LoRA at matched param budget collapses deterministically on high-entropy structured reasoning. Frozen-backbone controllers stable.
- Flagship: 3/3 LoRA runs at 16k steps on SpatialGrid Inst collapse to 25%; our controller 87% stable.
- Risk: too narrow? Reviewer says "just use rank-32 or lr=1e-5 and LoRA works."

**F3: "Composable Test-Time-Scaling Modules for Frozen LLMs."**
- Core claim: 140M controller gives K-scaling at inference; compute-efficient; composable; swappable.
- Flagship: K-scaling monotonic, frozen-backbone preserved.
- Risk: novelty contested (COCONUT, Recurrent-Depth, MoR all in this space).

What's strongest? What experiments needed to close the gap? Is there a **F4** I'm missing?

### Q8. Concrete action plan — 3 experiments this week.

4×B200 for 2 weeks. Propose **3 high-value experiments**. For each:
- Exact implementation (code changes, hyperparams).
- Expected result (point estimate).
- GPU-hours.
- What it proves/disproves.
- How it serves the paper.

Assume I can run ~5 concurrent jobs, each 2 hours → ~40 experiments/week.

### Q9. Deeper theory — is there a unifying framework?

DeepPass (layer duplication), TRM (7M recursive), Solver (bidirectional), Controller (iterative thoughts) all point to "iteration over fixed weights > single pass for reasoning." Unifying frameworks:

- **Fixed-point dynamics / DEQ.** Iteration converges to attractor.
- **Implicit depth.** K rounds on D layers ≈ KD effective layers.
- **Compute-optimal training.** Iteration is param-efficient in the "high iteration-demand" regime.
- **Representation stability / smoothing.** Each iteration reduces variance in representation.
- **Mean-field analysis.** Iteration allows mean-field convergence of posterior.
- **Bayesian inference.** Rounds = posterior sampling.

Which best fits the data? Testable predictions distinguishing them? Is there a theorem of the form:
$$\text{Gain}(K) - \text{Gain}(1) \geq f(\text{task complexity, input structure})$$

### Q10. Challenge me — honestly.

- Is the work actually novel given COCONUT, Recurrent-Depth, MoR?
- Is SpatialGrid too niche to support the paper?
- Should I abandon the controller and pivot to DeepPass (layer duplication) alone?
- What's the single biggest weakness that a reviewer will pounce on?
- Is the realistic best outcome a workshop paper, or can this be a main-conference paper?
- If you were me, what would you do differently in the next 2 weeks?

Don't flatter. Be direct.

---

## 9. Practical Constraints

- **American LLMs only** (Meta Llama, Google Gemma). No Qwen, Chinese models for new experiments.
- **4×B200 GPUs × 2 weeks** remaining to thesis deadline.
- Group CPU limit 32 cores total.
- Must be reproducible with open-source code.
- Target venue: **workshop at NeurIPS/ICLR or specialized journal**, possibly main conference if strong.

---

## 10. What I Want From You

1. **Is this useful or niche?** Brutal assessment of general-intelligence claim.
2. **Diagnose** — mechanistic arguments with math for the bifurcation. Accept/reject H1–H6.
3. **Design** — 3 hybrid architectures, ranked, with pseudocode + expected gain.
4. **Action plan** — 3 experiments for the week with specs and predictions.
5. **Paper framing** — strongest F1/F2/F3/F4, evidence gaps, missing experiments.
6. **Deeper theory** — unifying framework with testable predictions.
7. **Challenge** — direct critique; tell me what I'm missing.

Proof-level rigor. High creativity. Use equations. Cite literature. Don't hold back.

---

## 11. Appendix — Data & Artifacts

**Code:** `/blue/cis4914/jietao/DeepPass/solver/`
- `recurrent_deliberation.py` — base controller.
- `eval_deliberation_creative.py` — MidLayerDeliberation + LowrankDeliberation.
- `mega_runner.py` — SpatialEval pipeline.
- `mega_runner_benchmarks.py` — reasoning benchmark pipeline.
- `mega_runner_lora.py` — LoRA baseline pipeline.
- `mega_runner_lora_spatial.py` — LoRA on SpatialEval.
- `eval_deliberation_vision.py` — vision-LM controller.

**Results:**
- `results/data/mega/` — 400+ controller runs on SpatialEval.
- `results/data/benchmarks/` — controller on HS/WG/BQ/CSQA/OBQA (~20 results).
- `results/data/lora_baseline/` — LoRA matched baselines (~30 results, multi-seed).
- `results/data/vision/` — controller on VSR.
- `results/data/72b/`, `gemma3_27b/`, `qwen35/`, `moe/` — DeepPass Track A.

**Documents:**
- `PAPER.md` (275 lines) — DeepPass Track A paper outline.
- `HISTORY.md` (~2700 lines) — full experimental timeline.
- `prompts/` — 20+ prior GPT consultations.

**Related:**
- LiquidTRM (CfC-inspired), at `/blue/cis4914/jietao/LNN/`.
- TRM interpretability, at `/home/jietao/RR/SeniorProject/RR_Interpretability/`.

---

**Please begin. No preamble. Proof-level rigor. High creativity. Honest challenge.**
