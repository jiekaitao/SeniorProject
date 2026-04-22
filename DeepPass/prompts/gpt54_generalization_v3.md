# GPT-5.4 Pro Prompt: Why Doesn't Deliberation Generalize? How Do We Fix It?

## Instructions

Think extensively and comprehensively about this problem. I need proof-level reasoning, not hand-waving. For every claim you make, provide a mechanistic argument for WHY it's true, cite relevant literature if applicable, and suggest a concrete experiment to verify or falsify it. Be exhaustive — I'd rather have 10 pages of rigorous analysis than 2 pages of high-level suggestions.

## Context & History

We previously consulted you about a 72% accuracy ceiling on SpatialEval maze navigation. You diagnosed it as a "frozen-decoder readout bottleneck" and recommended the **Recurrent Deliberation Controller** architecture. We implemented it exactly as you specified. It works spectacularly on one task — but fails to generalize. We need to understand WHY and HOW to fix it.

## The Architecture (implemented exactly as recommended)

### Recurrent Deliberation Controller

A frozen LLM with a learned recurrent control interface. The controller maintains latent state z across rounds:

```python
class RecurrentDeliberation(nn.Module):
    def __init__(self, frozen_llm, d_state=512, n_slots=8,
                 tapped_layers=(8, 16, 24), topk_vocab=64):
        # Freeze the backbone
        for p in frozen_llm.parameters():
            p.requires_grad = False

        # Controller state
        self.z0 = nn.Parameter(torch.randn(1, n_slots, d_state) * 0.02)

        # READ: hidden states at tapped layers + slot hidden states + logits + uncertainty
        n_tap = len(tapped_layers)
        read_dim = n_tap * d_model + n_slots * d_model + 4 + 2  # 4=choice probs, 2=entropy+margin
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048), nn.GELU(),
            nn.Linear(2048, n_slots * d_state)
        )

        # STATE UPDATE: residual
        self.state_norm = RMSNorm(d_state)
        self.state_gate = nn.Parameter(torch.tensor(0.1))

        # WRITE: z → sparse vocab superposition → thought embeddings
        self.to_vocab_logits = nn.Linear(d_state, vocab_size, bias=False)
        # Top-K sparse selection, then weighted sum of vocab embeddings:
        # logits → topk(64) → softmax → weighted sum of E[idx] → thought embedding

        # VERIFIER: predict if current answer is correct
        self.verifier = nn.Sequential(
            nn.Linear(n_tap * d_model + n_slots * d_model + 4, 512),
            nn.GELU(), nn.Linear(512, 1)
        )

    def forward(self, prompt_emb, answer_emb, choice_ids, rounds=2):
        z = self.z0.expand(B, -1, -1).clone()
        for r in range(rounds):
            # 1. WRITE: z → thought embeddings (sparse vocab superposition)
            thought_emb = latent_to_thought_embs(z)  # (B, n_slots, d_model)

            # 2. RUN FROZEN DECODER: [prompt | thought_slots | answer_prefix]
            logits, think_h, tapped_pools = forward_frozen_round(
                prompt_emb, thought_emb, answer_emb)

            # 3. EXTRACT answer logits for A/B/C/D at last position
            ans_logits = logits[:, -1, choice_ids]  # (B, 4)

            # 4. BUILD FEATURES from frozen LM output
            feat = build_features(think_h, tapped_pools, ans_logits)

            # 5. VERIFIER: is current answer correct?
            verify = self.verifier(verify_feat)

            # 6. UPDATE controller state
            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        return all_choice_logits, all_verify

    def compute_loss(self, all_choice_logits, all_verify, answer_labels):
        # L = CE(final_answer, y)
        #   + 0.5 * mean_r BCE(v_r, correct_r?)
        #   + 0.1 * max(0, CE(final) - CE(first) + 0.1)
```

**Key design choices:**
- Thought tokens are **sparse vocab superpositions**: top-64 vocab embeddings weighted by softmax → stays native to LM embedding manifold
- Controller reads **hidden states at tapped layers** (mean-pooled over sequence) + think slot hidden states + answer logits + entropy + margin
- **Verifier head** predicts whether current round's answer is correct
- **Progress loss** ensures later rounds don't degrade vs round 1
- ~189M trainable params (controller only), frozen LLM is 8B

**Training setup:**
- Batch size 1, AdamW lr=1e-4, weight_decay=0.05, cosine schedule with 200-step warmup
- 2000-3000 steps depending on experiment
- All task formats: 4-way MCQA, extract logits for A/B/C/D tokens at last position

## Comprehensive Experimental Results (1000+ result files, 500+ configurations)

### Experiment 1: K-Scaling on SpatialEval Maze Navigation (Llama 3.1 8B)

**Does more deliberation at test time help?** Train with N rounds, evaluate at 1-20 rounds.

Train-3 (3000 steps), 2 seeds:
```
Rounds  Seed42   Seed7    Mean
base    33.4%    33.4%    33.4%
r=1     27.6%    26.4%    27.0%
r=2     47.2%    58.4%    52.8%
r=3     57.0%    65.6%    61.3%  ← trained here
r=5     59.4%    67.6%    63.5%
r=8     60.2%    68.2%    64.2%  ← peak
r=10    59.8%    67.6%    63.7%
r=15    59.2%    67.2%    63.2%
r=20    60.0%    67.2%    63.6%
```

Train-5 (3000 steps), 2 seeds:
```
Rounds  Seed42   Seed7    Mean
base    33.4%    33.4%    33.4%
r=1     22.4%    26.4%    24.4%
r=2     65.8%    56.6%    61.2%
r=3     70.4%    62.4%    66.4%
r=5     71.0%    69.0%    70.0%  ← trained here, peak
r=8     71.0%    68.6%    69.8%
r=10    71.0%    68.2%    69.6%
r=15    71.0%    68.0%    69.5%
```

Train-7 (3000 steps), 2 seeds:
```
Rounds  Seed42   Seed7    Mean
base    33.4%    33.4%    33.4%
r=1     26.4%    26.8%    26.6%
r=2     33.4%    58.0%    45.7%
r=3     62.2%    63.6%    62.9%
r=5     64.0%    66.0%    65.0%
r=7     64.2%    66.0%    65.1%  ← trained here
r=8     63.6%    66.2%    64.9%
r=10    64.2%    66.2%    65.2%
```

**Key findings:**
- Clear K-scaling: accuracy monotonically increases with rounds up to ~8, then plateaus
- Test-time extrapolation works: models trained with 3 rounds benefit from 8 rounds at eval
- Training with 5 rounds is optimal — train-7 is WORSE (harder optimization with more recurrent passes)
- Best result: 71.0% (seed 42, train-5) vs 33.4% baseline = **+37.6pp**

### Experiment 2: Adaptive Computation Time

**Can we stop deliberation early when confident?** Train with N rounds, evaluate with up to 10, use verifier/entropy to decide when to stop.

Train-5 adaptive (seed 42):
```
Strategy          Accuracy  Avg Rounds  Compute Saved
fixed_1           22.4%     1.00        —
fixed_2           65.8%     2.00        —
fixed_3           70.4%     3.00        —
fixed_5           71.0%     5.00        —
fixed_8           71.0%     8.00        —
adaptive_ent<0.8  71.0%     3.41        32% savings
adaptive_ent<0.5  71.0%     4.12        18% savings
```

Train-3 adaptive (seed 7):
```
fixed_8           68.2%     8.00        —
adaptive_ent<1.0  68.8%     3.25        59% savings
adaptive_ent<1.2  65.4%     2.45        69% savings
```

**Key finding:** Entropy-based early stopping achieves same accuracy with 32-59% fewer rounds.

### Experiment 3: Cross-Benchmark Generalization

All on Llama 3.1 8B, 8 slots, topk=64, tapped=(8,16,24).

**ARC-Challenge** (science reasoning, 4-way MCQA):
```
Config       Seed42   Seed7    Mean     Delta
baseline     77.4%    77.4%    77.4%    —
r=1          77.8%    76.4%    77.1%    -0.3pp
r=2          76.6%    77.6%    77.1%    -0.3pp
r=3          77.2%    77.8%    77.5%    +0.1pp
```
**Verdict: No effect.** Controller matches but cannot beat a strong baseline.

**HellaSwag** (commonsense reasoning, 4-way sentence completion):
```
Config       Seed42   Seed7    Mean     Delta
baseline     54.2%    54.2%    54.2%    —
r=1          57.0%    56.0%    56.5%    +2.3pp
r=2          57.8%    59.0%    58.4%    +4.2pp
r=3          57.6%    57.6%    57.6%    +3.4pp
```
**Verdict: Moderate positive.** +4.2pp mean at r=2. But NO K-scaling (flat across rounds).

**HellaSwag K-Scaling** (train-3, 3000 steps):
```
Rounds  Seed42   Seed7    Mean
base    54.4%    54.4%    54.4%
r=1     49.0%    46.8%    47.9%
r=2     60.0%    54.4%    57.2%
r=3     59.8%    56.0%    57.9%
r=5     59.6%    55.0%    57.3%
r=8     59.4%    54.8%    57.1%
r=10    60.4%    55.2%    57.8%
r=15    58.4%    55.4%    56.9%
```
**K-scaling does NOT generalize to HellaSwag** — accuracy is flat at all round counts (~57%).

**MMLU STEM** (knowledge + reasoning, 4-way MCQA, 8 subjects):
```
Config       Seed42   Seed7    Mean     Delta
baseline     40.7%    40.7%    40.7%    —
r=1          38.8%    40.1%    39.5%    -1.2pp
r=2          40.5%    39.7%    40.1%    -0.6pp
r=3          40.3%    39.2%    39.8%    -0.9pp
```
**Verdict: Slight negative.** Deliberation hurts knowledge retrieval.

### Experiment 4: Cross-Model Generalization

Each model gets a **freshly trained** controller (not transfer from Llama). Hook-based architecture for compatibility.

**Gemma 3 27B-IT on mazenav** (10 configs, 4 seeds):
```
Baseline: 47.0%
Deliberation r=1: ~25.0% (ALL configs)
Deliberation r=2: ~25.0% (ALL configs)
```
**Verdict: Catastrophic. -22pp.** Every single configuration collapses to ~25% (near random for 4-way MCQA).

**Gemma 4 31B-IT on mazenav** (2 configs):
```
Baseline: 33.2%
Deliberation r=2: ~23.4% (both seeds)
```
**Verdict: Negative. -10pp.**

**Llama 3.1 8B Instruct on mazenav** (3 seeds):
```
Baseline: 16.0%
Deliberation r=3: ~25.0%
```
**Verdict: Marginal positive (+9pp), but still very low.**

**Llama 3.1 8B Instruct on spatialmap:**
```
Baseline: 62.0%
Deliberation r=3: ~49.0%
```
**Verdict: Negative. -13pp.**

### Experiment 5: Cross-Task on Llama 8B

All with freshly trained controller per task.

```
Task         Baseline  Delib r=1  Delib r=3  Delta(best)
mazenav      33.4%     57.5%      57.6%      +24.2pp
spatialmap   37.2%     34.2%      34.1%      -3.0pp
spatialgrid  48.0%     44.2%      40.1%      -7.9pp
spatialreal  35.6%     —          26.7%      -8.9pp
```
**Verdict: Only mazenav benefits. All other SpatialEval tasks get worse.**

### Experiment 6: Partial Unfreezing

Unfreeze last K layers of decoder + norm + lm_head during deliberation training. Lower LR (1e-5) for unfrozen layers.

```
K   Unfrozen Params  Baseline(modified)  Delib r=3
0   0                33.4%               57.0%
2   961M             39.8%               39.8%
4   1.59B            59.8%               38.6%
8   2.46B            66.4%               39.8%
```

**Key observation:** Unfreezing k=8 layers makes the decoder itself reach 66.4% (without deliberation!), but adding deliberation DROPS it to 39.8%. The controller cannot coordinate with a non-stationary decoder.

### Experiment 7: Oracle Trace Control

Give the frozen decoder perfect BFS spatial information in the prompt (no training needed).

```
Mode                 Accuracy  Delta
baseline             33.2%     —
reachable_set        42.4%     +9.2pp
option_reachability  33.2%     +0.0pp
full_trace           30.2%     -3.0pp
```

**Key finding:** Even perfect oracle spatial information only helps +9pp. The deliberation controller's implicit attention steering (+37.6pp) is 4x more effective than explicit oracle information. The decoder cannot process explicit structured data well.

### Experiment 8: Configuration Ablations (500+ configs on mazenav)

**Slot count:**
```
Slots  Best Acc  Mean Acc (r=3)
4      47.0%    46.5%
8      72.0%    57.6%
12     70.0%    57.3%
16     71.0%    56.9%
32     68.8%    64.2%
```
8 slots is the sweet spot for peak; 32 has higher mean but lower peak.

**TopK vocab:**
```
TopK   Best Acc (r=3)
16     44.6%
32     63.6%
64     72.0%  ← optimal
128    53.4%
```

**Tapped layers (Llama 8B, 32 layers):**
```
Config                         Mean Acc
[8, 16, 24] (3 layers)        57.6%
[4,8,12,16,20,24,28] (7 layers)  57.6%
```
No difference between 3 and 7 tapped layers on Llama 8B.

**Answer distribution in dataset:** A=39.8%, B=28.0%, C=18.2%, D=14.0% (imbalanced)

## Previous Architecture (Before Deliberation)

We also tried a separate **SolverCore** prefix-based architecture (~12M params bidirectional module) that produces 32 memory tokens prepended to the prompt. Results on mazenav:
- Best: 72.0% (same ceiling)
- Mean across 50+ runs: ~50%
- Cross-task: spatialmap 23%, spatialgrid 25% (worse than deliberation)

## Summary of What Works and What Doesn't

**WORKS (only on Llama 8B + mazenav):**
- K-scaling: 33% → 71% with 5 training rounds
- Test-time extrapolation: train with fewer rounds, eval with more
- Adaptive stopping: 32-59% compute savings via entropy
- Multiple architectures reach the same ~72% ceiling

**DOESN'T WORK:**
- Cross-model: Gemma 3 (-22pp), Gemma 4 (-10pp)
- Cross-task: spatialmap (-3pp), spatialgrid (-8pp), spatialreal (-9pp)
- Cross-benchmark: ARC (0pp), MMLU (-1pp)
- HellaSwag has moderate gains (+4pp) but no K-scaling
- Partial unfreezing: catastrophic forgetting
- More training rounds (7 > 5): harder optimization
- More training steps (5000 > 3000): overfitting

## Questions for You (answer ALL with proof-level reasoning)

1. **WHY does the deliberation controller fail on Gemma 3/4 specifically?** The controller is retrained from scratch for each model, so it's not a transfer problem. What's mechanistically different about Gemma's internals that prevents the controller from learning?

2. **WHY does it fail on other SpatialEval tasks (spatialmap, spatialgrid, spatialreal)?** These are all spatial reasoning tasks on the same model. The controller is retrained per task. What's special about mazenav?

3. **WHY does HellaSwag show improvement but no K-scaling?** The +4pp suggests the controller learns something useful, but additional rounds don't help. What determines whether a task benefits from iterative refinement vs one-shot improvement?

4. **The tapped layer question:** We use evenly-spaced layers (8,16,24 for Llama 8B). We showed 3 vs 7 tapped layers doesn't matter on Llama 8B. But for Gemma 3 (62 layers, sliding window attention), could wrong layer selection explain the failure? What layers SHOULD we tap for different architectures?

5. **The vocab-space thought token assumption:** Thoughts are sparse superpositions of vocab embeddings. This keeps them "native" to the LM manifold. But does this actually constrain the thought space too much? Would continuous embeddings (not vocab-space) work better for generalization?

6. **The batch-size-1 problem:** We train with batch size 1 due to variable prompt lengths. Could this explain the high seed variance (8pp gap on train-3, 20pp on 5000-step)? How critical is batch size for recurrent architectures?

7. **Why does the oracle trace only help +9pp but the controller helps +37pp?** The controller doesn't have explicit spatial knowledge — it manipulates attention patterns. What does this tell us about how the frozen decoder processes information, and how should we exploit this?

8. **The unfreezing paradox:** Unfreezing k=8 layers makes the decoder alone reach 66.4%, but adding deliberation drops to 39.8%. The controller requires a FROZEN decoder. Why? And does this impose a fundamental limit on what deliberation can achieve?

9. **CONCRETE ARCHITECTURAL CHANGES** to make this generalize:
   - Should we change how we tap hidden states?
   - Should we change the thought token representation?
   - Should we change the training procedure?
   - Should we change how the controller reads the decoder's output?
   - Is there a fundamentally different approach that preserves K-scaling but generalizes?

10. **Literature you know about** that addresses iterative inference-time computation across multiple models/tasks. What approaches have actually been shown to generalize? What can we learn from them?

## PART B: Layer Duplication Work (DeepPass)

The deliberation controller is one half of our thesis. The other half is **layer duplication** — repeating transformer blocks at inference time without modifying weights. This is a complementary approach to adaptive computation: instead of a learned controller choosing WHAT to think, we're choosing WHICH layers to repeat.

### The Core Idea

Given a transformer with layers [L0, L1, ..., LN], we duplicate a contiguous block [Li, ..., Lj] so the model executes:
```
[L0, ..., Li, ..., Lj, Li, ..., Lj, ..., LN]
```
Same weights, run twice. No training. The second pass acts as iterative refinement: h → F(h) → F(F(h)).

### Spectral Screening: Displacement Rho

Instead of brute-force searching all (i,j) pairs (3,241 configs for 80-layer 72B), we use a **spectral displacement metric**:
```
rho = ||F(F(h)) - F(h)|| / ||F(h) - h||
```
Values < 1 mean the block converges when repeated. This predicts which blocks benefit from duplication.
- 7B: 80% top-5 hit rate, **162x reduction** in search cost
- 72B: rho alone fails (p=0.50) — need combined metric

### SBUID Screening Metric (BLOOD - λ*rho)

First statistically significant screening metric on 72B models:
- **Spearman r=0.515, p=0.008** on 72B
- Cross-validated: train r=0.34 → test r=0.664 (holds out of sample)
- Works on 27B+ models (p=0.038 on 27B), fails on small models (p=0.83 on 9B)
- Combines BLOOD impact (downstream Jacobian smoothness) with displacement rho
- Novel trajectory metrics we tried (OTAS, GCHS, CLRG) all failed — duplication is NOT about advancing along the base trajectory

### Multi-Block Stacking: Cross-Region Pairs

Complementary blocks from different network regions stack:
- 72B: (0,7)+(45,52) combined=79.91 beats all single blocks (Ng's best=76.76)
- Early blocks (layers 0-10) + deep blocks (layers 45-55) activate different circuits
- Same-region pairs interfere; cross-region pairs are complementary

### Per-Layer Alpha Tuning

Each layer within a duplicated block gets its own blending weight:
```
h_out = h1 + alpha_l * (h2_l - h1_l)   # per-layer alpha
```
- **Single block (45,52) with 7 per-layer alphas: 82.77** — nearly matches 4-block configurations
- **Triple with 21 per-layer alphas: 84.07** — all-time record, +7.31 over Ng
- Key finding: some layers should be boosted (alpha=1.3), others dampened (0.5) or disabled (0.0)
- Bayesian optimization (Optuna TPE) reaches 83.97 in only 60 evals vs 300 for grid search = **5x speedup**

### Key Results Table

| System | Combined Score | Delta vs Ng |
|--------|---------------|-------------|
| Baseline (72B) | 70.52 | — |
| Ng (45,52) @1.0 | 76.76 | — |
| Our pair (0,7)+(45,52) | 79.91 | +3.15 |
| Per-block quad (whisper alpha) | 82.58 | +5.82 |
| Per-layer single (45,52) | 82.77 | +6.01 |
| **Bayesian per-layer triple** | **83.97** | **+7.21** |
| **Grid per-layer triple** | **84.07** | **+7.31** |

### Cross-Architecture Generalization (Layer Duplication)

Layer duplication generalizes across ALL architectures tested:
| Model | Baseline | Best Config | Delta |
|-------|----------|-------------|-------|
| Qwen2-72B | 70.52 | Per-layer triple | +13.55 |
| Gemma3-27B | 80.54 | Quad @1.0 | +5.04 |
| Qwen3.5-27B | 42.86 | Triple | +37.19 |
| Qwen3-30B MoE | 27.76 | Single (8,9) | +12.66 |

**This is in stark contrast to the deliberation controller, which fails on Gemma entirely.**

### Why Duplication Hurts Knowledge: FFN Re-Retrieval Hypothesis

- FFN/MLP layers store facts as key-value associations
- On second pass, perturbed input causes FFN to retrieve nearby-but-wrong facts
- Attention benefits from repetition (re-weighting computation), FFN hurts (corrupts retrieval)
- lm-eval confirms: duplication helps reasoning (MuSR +4.3%, IFEval +2.2%) but hurts knowledge (MATH -5%, MMLU-PRO -1.9%)
- Sublayer analysis: attention-only duplication on L2 (80.35) beats full duplication (77.45)

### Scale-Dependent Mechanism

- **9B:** Second pass inflates norms by 42% (norm_ratio=1.42). Norm-preserving projection helps +13.6.
- **72B:** Second pass barely changes norms (norm_ratio=1.04, cosine=0.997). The perturbation is almost purely directional. Norm-preserving hurts -2.3.
- Alpha=1.25 beats standard on 72B — the directional refinement is slightly too conservative at 1.0.

### Multi-Pass Ceiling

3+ passes destroys EQ-bench on 72B (83→63 at 3x, 17 at 4x). Two passes is the ceiling for layer duplication.

### Negative Results (Layer Duplication)

- All trained routers fail (BrierHalting, ESR+DSG) — can't learn adaptive routing
- DICE pair prediction is weak (Spearman=0.191)
- Math probe doesn't generalize to lm-eval — duplication is task-selective
- Residual stability doesn't predict pair quality
- Novel screening metrics (OTAS, GCHS, CLRG) all fail on 72B
- Deeper stacking doesn't work on 9B (too small)

## THE BIG PICTURE: Two Approaches to Adaptive Computation

| Property | Layer Duplication | Deliberation Controller |
|----------|-------------------|------------------------|
| Training required? | NO | YES (189M params) |
| Cross-architecture? | YES (all 5 tested) | NO (only Llama 8B) |
| Cross-task? | Partially (helps reasoning, hurts knowledge) | NO (only mazenav + slight HellaSwag) |
| K-scaling? | NO (2 passes is ceiling) | YES (up to 8 rounds) |
| Adaptive stopping? | Not implemented | YES (32-59% savings) |
| Best improvement | +7.31 (84.07 combined on 72B) | +37.6pp (33→71% on mazenav) |
| Mechanism | Untrained iterative refinement | Learned thought token steering |

**The thesis needs BOTH to tell a complete story about adaptive computation time for LLMs.** Layer duplication shows untrained iteration helps across architectures. Deliberation shows learned iteration with K-scaling but only on specific tasks. Why the asymmetry?

## Questions (updated — answer ALL with proof-level reasoning)

11. **Why does layer duplication generalize across architectures but the deliberation controller doesn't?** Layer duplication is untrained — same weights, run twice — and works on Qwen, Gemma, and MoE models. The deliberation controller is trained per-model and still fails on Gemma. What's fundamentally different? Is it that training introduces model-specific overfitting? Or that vocab-space thought tokens are inherently architecture-dependent?

12. **Can we combine layer duplication with deliberation?** Layer duplication gives +7.31 on 72B. Deliberation gives +37.6pp on 8B mazenav. Could a deliberation controller that also controls WHICH layers to duplicate — and with what alpha — yield a system that generalizes? The controller would learn both thought tokens AND duplication routing.

13. **The multi-pass ceiling for duplication (2 passes) vs K-scaling for deliberation (up to 8 rounds):** Why can deliberation iterate 8 times without degrading but layer duplication degrades after 2 passes? Is it because deliberation updates its thought tokens between rounds (adaptive) while duplication uses the same weights (fixed)? What does this tell us about the role of adaptivity in iterative computation?

14. **The FFN re-retrieval hypothesis applies to duplication. Does something analogous happen in deliberation?** In layer duplication, FFN corrupts factual memory on the second pass. In deliberation, the thought tokens go through ALL layers (including FFN). Could FFN interference with thought tokens explain the cross-task failure?

15. **What additional innovations can you think of to further this research?** Beyond fixing generalization, what novel directions could this work open up? Think about: new training objectives, new ways to use the verifier, new controller architectures, connections to other fields (control theory, program synthesis, neuroscience), ways to combine our two approaches (layer duplication + deliberation), ways to make this publishable at a top venue, and anything else a world-class researcher would think of. Don't limit yourself to incremental improvements — propose bold ideas that could turn this into a significant contribution.

Be specific, mechanistic, and comprehensive. I have full GPU access and can implement anything you suggest.
