# Make the Recurrent Deliberation Controller Beat LoRA Everywhere — A Deep Research Question

You are GPT-5.4 Pro, helping me design a research direction for my senior thesis at University of Florida (undergrad CIS 4914). I've been building a recurrent-deliberation controller that sits on top of a frozen LLM and iteratively writes "thought tokens" into the residual stream. After ~500 GPU experiments on a B200 cluster, I have hard head-to-head data against LoRA fine-tuning at matched parameter budget. **LoRA wins on most benchmarks. Our controller wins decisively on SpatialGrid and generally on structured reasoning, but loses on semantic/commonsense benchmarks.** I need your help figuring out how to make the controller win everywhere — ideally *without* benchmark-specific training (zero-shot or at least task-agnostic).

## 1. The System We Built

**Frozen backbone:** Llama 3.1 8B (Base or Instruct), bf16, ~15 GB VRAM.

**Trainable controller:** `MidLayerDeliberation + lowrank residual writer` — 140M parameters.

- **Hidden state:** `z ∈ R^(1×8×512)` — 8 "slots" of 512-dim latent state.
- **Initial state:** `z0 ∼ N(0, 0.02²·I)`.
- **Read head:** at tapped layers `(8, 16, 24)` of the frozen LM, mean-pool the residual stream hidden states. Concatenate with the 8 slot-hidden-states from the previous round + answer-logit softmax + entropy + top-2 margin. Feed through a `Linear(read_dim → 2048) → GELU → Linear(2048 → 8·512)` → residual update to `z`.
- **State update:** `z ← RMSNorm(z + σ · Δz)` with learned scalar gate `σ` (init 0.1).
- **Write head ("thought tokens"):**
  - Sparse vocab superposition: `z` → `Linear(d_state → vocab_size)` → top-k=64 softmax → weighted sum of vocab embeddings (shape `1×8×4096`).
  - Plus a **lowrank residual**: `thought += 0.12 · z · W_down · W_up` with `W_down ∈ R^(d_state × 64)`, `W_up ∈ R^(64 × d_model)`.
- **Injection:** thoughts are inserted into the residual stream *at layer 12* (of 32) between the prompt tokens and the answer-prefix tokens. The first 12 layers run on `[prompt | answer_prefix]`, then thought tokens are spliced into position, positional embeddings are recomputed, and layers 12–31 run on `[prompt | thoughts | answer_prefix]`.
- **Rounds:** during training we run 5 rounds per sample. At test time we do K-scaling (evaluate with 3, 5, 8 rounds).
- **Loss:** `CE(answer_logits, y) + 0.5·BCE(verifier_head, per-round correctness) + 0.1·max(0, CE_final − CE_first + 0.1)` (progress loss — penalize later rounds getting worse).

**Training recipe:**
- AdamW lr=1e-4, weight_decay=0.05
- 200-step warmup → cosine decay
- **Gradient accumulation = 16** (critical — killed seed variance from std 12.8% → 0.3%)
- **Total steps = 8000** (critical — at 2000 nothing worked)
- bfloat16, grad clip 1.0
- Batch size 1 per step

**Evaluation:** 500 held-out test samples per benchmark, pick best across K-scaled round counts.

## 2. The Benchmarks

We tested on two families:

### SpatialEval (NeurIPS 2024, MilaWang/SpatialEval, "tqa" split)

- **SpatialGrid** (counting in a tabular grid):
  ```
  Consider a 5x5 grid with animals from ['cat','dog','elephant','giraffe','rabbit']:
  elephant | rabbit   | rabbit   | dog     | giraffe
  cat      | rabbit   | elephant | dog     | cat
  elephant | elephant | giraffe  | giraffe | rabbit
  ...
  How many blocks contain dog? A. 5  B. 6  C. 2  D. 7
  ```
  Llama 8B Inst baseline ≈ 48–77% (depending on exact sampling). Random = 25%.

- **Mazenav** (count right turns on a *pre-marked* path):
  ```
  #######
  #E# # #
  #X# # #
  #X#  S#
  #X###X#
  #XXXXX#
  #######
  How many right turns from S to E? A. 4  B. 8  C. 2  D. 7
  ```
  Note: the X's already mark the path — the model doesn't have to plan, just count direction changes along a given trajectory. Baseline ≈ 16% (below random! the model tends to fail systematically).

- **SpatialMap** (spatial relations among objects, 4-way MC).
- **SpatialReal** (real-scene spatial questions).

### Standard reasoning benchmarks

- **HellaSwag** (sentence completion commonsense, 4-way)
- **WinoGrande** (coreference resolution, 2-way)
- **BoolQ** (passage → yes/no, 2-way)
- **CommonsenseQA** (5-way commonsense)
- **OpenBookQA** (4-way science with book)
- **ARC-Challenge** (4-way science)

## 3. The LoRA Baseline (matched compute)

To test whether our controller adds anything beyond "trainable adapter with iterative inference," I built an apples-to-apples LoRA baseline:
- LoRA rank 64 on all attention + MLP projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) across all 32 layers → **~140M trainable params (matches our controller exactly)**.
- Same train/test splits, same tokenizer prompts, same 8000 steps, same AdamW lr=1e-4 + GA=16.
- Single forward pass at inference (no multi-round).

## 4. The Results (what we learned over ~500 paired runs)

**"Delta" = accuracy improvement over zero-shot baseline of the same frozen LLM.**

### Benchmarks where LoRA wins (same 140M budget)

| Benchmark | Model | Controller Δ (mean ± sd) | LoRA Δ (mean ± sd) | Winner |
|-----------|-------|--------------------------|---------------------|--------|
| **HellaSwag** | Inst | +11.3pp (n=3 seeds) | +13.5pp (n=3) | LoRA by 2pp |
| **HellaSwag** | Base | +12.6pp (n=2) | +18.6pp (n=1) | LoRA by 6pp |
| **WinoGrande** | Inst | +7.0pp (n=3) | +18.3pp (n=3, all replicates identical) | **LoRA by 11pp** |
| **WinoGrande** | Base | +9.1pp (n=2) | +25.4pp (n=2) | **LoRA by 16pp** |
| **CommonsenseQA** | Inst | +1.7pp (n=2) | +8.8pp (n=1) | LoRA by 7pp |
| **OpenBookQA** | Inst | +1.9pp (n=2) | +2.6pp (n=1) | ~tie |
| **Mazenav** | Inst | +40pp (peak 74%) | +73pp (peak 89%) | **LoRA by 33pp** |

### Benchmarks where Controller wins decisively

| Benchmark | Model | Controller Δ | LoRA Δ | Notes |
|-----------|-------|-------------|--------|-------|
| **SpatialGrid** | Inst | **+39pp** (87.0% ± 0.8% across 5 seeds) | +1.7pp mean, range **[-52pp, +8.2pp]** across 4 seeds; catastrophically collapses to 25% (random) at 16k steps on 3/3 replicates | **Controller +37pp, stable** |
| **SpatialGrid** | Base | **+34pp** (~79% stable) | **Bimodal**: seeds 100 & 300 → 90% (+45pp), seeds 42 & 200 → 20% (-25pp, below random). 4/8 runs catastrophic. | **Controller is seed-stable; LoRA is not** |

### BoolQ

| Model | Controller | LoRA |
|-------|-----------|------|
| Inst | +6.3pp | +7.2pp |
| Base | +6.2pp | +11pp |

Roughly tied to slight LoRA lead.

### Vision (preliminary, no LoRA baseline yet)

- **PaLiGemma2-10B + our controller on VSR (Visual Spatial Reasoning)**: baseline 37% → +33pp at inject layer 32 (deeper = better, tested L8, L16, L24, L32). Baseline is below random (50%) because PaLiGemma is biased toward "False" before fine-tuning — the controller recalibrates.
- LLaVA-NeXT 7B + controller on VSR: baseline 43.5% → +2.7pp. Tiny gain, unclear why this architecture responds so poorly.

### K-scaling (controller only, LoRA has no analog)

| Training rounds | Eval at 3 rounds | Eval at 5 | Eval at 8 |
|-----------------|------------------|-----------|-----------|
| 5 (default) | lower | ~peak | slightly higher |

Concrete: HellaSwag Inst 8k→16k training goes from +11.3 → +14.8pp. More training helps the controller. More training **catastrophically breaks LoRA** on SpatialGrid (-52pp at 16k).

## 5. Mechanistic Hypotheses — Why the bifurcation?

### SpatialGrid — a genuine reasoning task LoRA can't learn

The question requires *counting occurrences in a structured layout*. Training labels are all over the place (answer = 2, 3, 5, 6, 7 etc. across samples). There is no fixed surface pattern. The model has to **execute the counting operation at inference time**. LoRA updates weights but has no mechanism to execute per-instance arithmetic. Hypothesis: LoRA's gradient signal is incoherent because "when given a different grid, count something different" isn't learnable as a static mapping. Hence LoRA collapses to a constant-prediction strategy (we see 25% = always-C at 16k steps). Our controller has iterative rounds where it can attend to different rows each round and accumulate a tally in latent state.

### Mazenav — secretly a pattern-matching task

The solution path is *pre-marked with X's*. The model doesn't need spatial planning; it just needs to count direction changes. X patterns repeat across train/test samples — LoRA can memorize "this X-shape → B." It hits 89% because the task surface is more static than it looks.

### WinoGrande / HellaSwag / BoolQ / CommonsenseQA — semantic pattern matching

These benchmarks rely on world-knowledge associations. Fine-tuning weights is a direct way to sharpen those associations. Our controller's thought tokens are a more indirect intervention — they steer the frozen LM through latent state but can't modify the LM's factual associations. LoRA can.

### Controller's strengths

- **K-scaling:** more test-time rounds → better accuracy (up to plateau ~8 rounds).
- **Stability:** zero catastrophic failures across 400+ runs. LoRA fails 3/4 times at 16k steps on SpatialGrid Inst, and is bimodal across seeds on SpatialGrid Base.
- **Frozen backbone preserved** — multiple controllers could be hot-swapped or combined.

## 6. What We've Tried

- **Mid-layer injection point:** swept L0, L2, L4, L6, L8, L12, L14, L16, L18, L20. L0–L16 all give ~83% on SpatialGrid Instruct (layer-invariant for this task). L18–L20 has a task-specific dead zone on mazenav (drops ~30pp) but not on other tasks.
- **Base vs Instruct:** Instruct adds +18pp on SpatialGrid vs only +1.5pp on mazenav over Base.
- **Rounds (train & eval):** trained with 1, 2, 3, 5, 8. 5 rounds is sweet spot for training. At eval, rounds 3–8 all similar.
- **Grad accumulation:** 1, 4, 8, 16, 32, 64. GA=16 best; std drops from 12.8% → 0.3%.
- **Steps:** 1500, 3000, 5000, 8000, 12000, 16000. 8000 optimal for SpatialGrid, diminishing returns after. 16000 hurts LoRA on SpatialGrid (collapse).
- **Writer architecture:** tried vocab-only superposition (hit ceiling), vocab + lowrank residual (current, eliminated variance), lowrank-only (worse). Vocab + lowrank(α=0.12) is best.
- **Thought token decoding:** decoded nearest vocab neighbors — they're opaque gibberish ("EEPROM", "avatel", ...) — no interpretable concepts.
- **Dead zone L18–L20 mazenav:** 42% vs 70% at L17. Task-specific, not architectural.
- **Multi-task training:** one controller trained on mazenav+spatialgrid+spatialmap simultaneously. Works OK but per-task is better.
- **Verifier head:** helpful for training stability, doesn't help inference.
- **Larger controllers:** 280M, 560M variants. Marginal improvement, not worth it.

## 7. What Has NOT Worked (or not tried carefully)

- **Generalizing across LLMs:** controller only tested on Llama 3.1 8B. Gemma 3 27B showed weaker results but not fully tested with best hyperparams. Would be a big win if we could show cross-architecture generalization.
- **Zero-shot transfer:** every benchmark requires its own controller training. No evidence that a controller trained on one benchmark transfers to another.
- **Instruction-following backbone:** we only use Llama. No tests with DPO'd / RLHF'd instruct models beyond Llama's own.
- **Retrieval-augmented thoughts:** we haven't tried letting the controller *retrieve* information, only letting it reorganize what's already in the LM.
- **Test-time learning:** controller is frozen at inference. Never tried online gradient updates per instance (hypernetwork, TTT-style).

## 8. The Deep Question For You

**Can we redesign the controller so it wins across ALL benchmarks (semantic + structured), ideally without per-benchmark training?**

Here are the specific questions I want you to think through, in depth, with concrete architectural / training-recipe / inference-time proposals:

### Q1 — Why does LoRA beat us on semantic tasks?
Is it fundamentally because LoRA can modify the LM's *stored associations* while our controller only modifies *activations*? If so, is there a way to let the controller effectively modify associations (e.g., via a small adapter inside the frozen LM that the controller conditionally activates)?

### Q2 — How do we keep controller's SpatialGrid advantage *while* closing the gap on WinoGrande?
What architectural addition would let the controller behave like LoRA on memorization tasks AND like a reasoner on structured tasks? One idea: hybrid controller + sparse task-gated adapter. Another: controller learns to decide per-instance whether it needs iterative compute or just a fact lookup.

### Q3 — Zero-shot or universal controller?
Is there a training recipe (multi-task meta-training across many benchmarks) that could produce a single controller checkpoint that transfers zero-shot? COCONUT, Recurrent-Depth, MoR, Dr.LLM, and Quiet-STaR are related — can you synthesize from them?

### Q4 — Test-time compute scaling
Our controller has K-scaling (more rounds = better). Can we make this sharper — e.g., adaptive early-stopping based on verifier head confidence, or test-time gradient updates on controller params only?

### Q5 — Why does the controller break LLaVA but work on PaLiGemma?
PaLiGemma VSR went 37% → 70% (+33pp). LLaVA VSR went 43.5% → 46.2% (+2.7pp). Same architecture pattern (vision encoder + projector + LM). What's different about LLaVA (Mistral backbone) that makes thought tokens less effective?

### Q6 — LoRA's catastrophic collapse on SpatialGrid is a diamond
LoRA at 16k steps on SpatialGrid Inst *reliably* goes to 25% (always-A prediction). It's not noise — it's a stable failure mode reproducible across seeds. Why? Entropy of the answer distribution in train is high (no dominant letter). Is LoRA collapsing to a degenerate minimum? Can we quantify this and make it the center of a paper?

### Q7 — The vision angle
Our controller works spectacularly on PaLiGemma + VSR (+33pp). This is a real win. How do we generalize this — which other vision benchmarks are *structured reasoning* tasks where LoRA would similarly fail? CLEVR, SuperCLEVR, visual-spatial-reasoning, ICON-QA, TallyQA... suggest a principled benchmark suite that would maximize our advantage.

### Q8 — Parameter-matched or parameter-scaled comparison?
At 140M params the controller loses on most things. What happens at 500M or 1B trainable params? Does the *architecture* have a scaling advantage over LoRA as compute grows? Unclear — our small-scale ablations were inconclusive.

### Q9 — The scientific framing
Given all the above, suggest **three** possible paper framings, ranked by likelihood of acceptance at a top venue:
1. "When iterative latent computation beats fine-tuning" — scoped claim, SpatialGrid as flagship result.
2. "Stable controllers vs unstable fine-tuning" — focus on LoRA's catastrophic collapse.
3. "Composable test-time-scaling modules for frozen LLMs" — emphasize K-scaling, swappability.

Which is the strongest story? What additional experiments would I need to run to nail it?

### Q10 — Concrete action plan
Given I have a B200 cluster and ~2 weeks before thesis deadline, what are the **3 highest-value experiments** I should run? Please be specific:
- Exactly what to implement
- Expected result
- How many GPU hours
- What it would prove / disprove

## 9. Constraints

- Only American LLMs (Meta Llama, Google Gemma). No Qwen, Mistral-direct, etc.
- 4×B200 GPU budget, ~2 weeks.
- Must be reproducible with open-source code.
- Undergrad thesis — no PhD-level infrastructure (no TPU pod, no cluster of 1000 GPUs).
- Prefer clean scientific contribution over chasing SOTA.

## 10. What I Want From You

1. **Diagnose** — Tell me what's really going on mechanistically. Don't just repeat my hypotheses; challenge them.
2. **Design** — Propose 3 concrete architectural/training modifications ranked by expected impact.
3. **Action plan** — The 3 experiments I should run this week. Be specific about code changes, hyperparams, and expected outcomes.
4. **Paper framing** — The strongest story, and what evidence I'm missing.
5. **Stretch** — If you think this line of work isn't going to beat LoRA on semantic tasks, say so honestly. What's the real ceiling?

Think step by step. Use concrete numbers where you can. Challenge my framing. I trust you to tell me when I'm wrong.
