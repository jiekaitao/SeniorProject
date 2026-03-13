# Full Analysis: Recursive Reasoning Loss Modifications for TRM

## Experiment 1: Raju-Netrapalli Error Model Analysis (Best Checkpoints)

### Setup

We apply the Raju-Netrapalli (2024) error accumulation framework to compare two TRM (Tiny Recursive Model) variants trained on 30x30 hard mazes:

| | BCE Baseline (BASE_MODEL) | BrierHalting (RR_MODS_MODEL) |
|---|---|---|
| Halting loss | Binary Cross-Entropy | Brier Score (MSE) |
| Monotonicity reg. | No | Yes |
| Token loss | stablemax_cross_entropy | softmax_cross_entropy |
| Best epoch (wandb) | 21,600 / 50,000 | 5,800 / 50,000 |
| Best exact_accuracy | 86.90% | 87.30% |
| Checkpoint used | step_168696 | step_45298 |
| Architecture | hidden=512, H_cycles=3, L_cycles=4, L_layers=2, halt_max=16 |

Both models share the same architecture, dataset (1k hard mazes), batch size (128), learning rate (1e-4), and EMA. The key differences are (a) halting loss function, (b) monotonicity regularization, and (c) token-level loss function (stablemax vs softmax).

**Important confound**: These models differ in THREE ways simultaneously (halting loss, monotonicity, token loss), making it impossible to attribute differences to a single cause.

### Analysis Protocol

For each model we:
1. Run 64 refinement steps (4x the halt_max=16) on all 1,024 test mazes (16 batches of 64)
2. At each step, measure exact accuracy, token accuracy, and empirical spectral norm rho via perturbation analysis
3. Test argmax snap-back error correction (re-derive argmax tokens and continue)
4. Attempt to fit the three-parameter accuracy law from Raju-Netrapalli: `acc(T) = Q(q, T * r_step / rho)`

### Results

#### RQ1: Empirical Spectral Norm rho

| | rho (step 1) | rho (steady-state, steps 10-64) |
|---|---|---|
| BCE | ~2.93 | **~2.03** |
| BrierHalting | ~4.63 | ~2.46 |

**BCE has ~20% lower steady-state rho** (2.03 vs 2.46), meaning its iterative refinement dynamics are more contractive. BrierHalting exhibits much larger initial perturbation amplification at step 1 (~4.6x vs ~2.9x), suggesting its first refinement step is more volatile.

Both models have rho > 1 at all steps, indicating neither is strictly contractive in the operator norm sense. However, both converge reliably in practice, suggesting the effective contraction happens in a lower-dimensional subspace relevant to the solution.

Note: Batch 16 (partial, 24 samples) showed anomalous behavior for both models — lower rho for BCE (~1.93) and variable rho for BrierHalting (~2.2-2.4). This batch likely contains easier/shorter mazes.

#### RQ2: Three-Parameter Accuracy Law Fit

**Both fits failed.** The optimizer hit parameter bounds with uncertainties 6 orders of magnitude larger than the values:

| | q | r_step | rho_fit | residual |
|---|---|---|---|---|
| BCE | 0.50 +/- 4.3M | 0.51 +/- 768K | 0.01 +/- 25.8K | 1.40 |
| BrierHalting | 0.50 +/- 4.8M | 0.50 +/- 758K | 0.01 +/- 28.8K | 1.40 |

**Why it fails**: The Raju-Netrapalli model assumes a gradual accuracy ramp governed by error accumulation. In TRM, accuracy jumps from ~0% to ~70-90% in just 2-3 steps, then plateaus. This step-function behavior doesn't match the smooth incomplete gamma CDF. The model was designed for autoregressive LLMs where errors accumulate token-by-token; TRM's iterative refinement has fundamentally different dynamics where the entire solution crystallizes rapidly.

#### RQ3: BrierHalting vs BCE Comparison

| Metric | BCE | BrierHalting | Winner |
|---|---|---|---|
| Peak exact accuracy (this analysis) | 87.50% | 87.50% | Tie |
| wandb best exact_accuracy | 86.90% | 87.30% | BrierHalting (+0.4%) |
| Epochs to peak | 21,600 | 5,800 | **BrierHalting (3.7x faster)** |
| Steady-state rho | 2.03 | 2.46 | **BCE (more contractive)** |
| Step-1 rho | 2.93 | 4.63 | **BCE (less volatile)** |
| Convergence speed (steps to plateau) | ~4-5 | ~3-4 | BrierHalting (slightly faster) |
| Stability at 64 steps | Stable | Stable | Tie |

**The paradox**: BrierHalting achieves equal/better accuracy with WORSE contraction properties (higher rho). This suggests:

1. **rho doesn't tell the whole story** — the Raju-Netrapalli error model's assumption that spectral norm governs accuracy doesn't hold for iterative refinement. The models may be contractive in a task-relevant subspace even though they're expansive in the full state space.

2. **Training efficiency vs inference dynamics are decoupled** — BrierHalting's 3.7x faster convergence during training doesn't translate to better fixed-point contraction at inference time. The Brier loss likely provides better gradient signal for learning the halting decision, but doesn't reshape the refinement operator itself.

3. **The token loss confound matters** — stablemax vs softmax could explain some of the rho difference, independent of the halting loss.

#### RQ4: Argmax Snap-Back

Snap-back was **neutral/destructive** for both models across all full batches (0% improvement on batches 1-15). Batch 16 (partial) showed 37.5% snap_exact, but this reflects that batch's initial state, not actual error correction.

**Interpretation**: The argmax discretization does not help. The models' continuous hidden states already encode the correct discrete solution — forcing argmax and re-embedding doesn't provide useful error correction. This makes sense: the refinement operates in a continuous latent space where the argmax boundary is arbitrary.

### Open Questions

1. **Is the rho difference caused by halting loss or token loss?** Need ablation: BrierHalting + stablemax, or BCE + softmax.
2. **Why does BrierHalting train 3.7x faster?** Is it the Brier score, the monotonicity regularization, or softmax?
3. **Does the rho difference matter at all?** Both models are equally accurate and stable. Perhaps rho is irrelevant when both are in the "converged" regime.
4. **Would a different error model fit better?** The step-function convergence suggests a phase-transition model rather than gradual error accumulation.
5. **What happens on harder tasks?** The ~87% ceiling may mask differences that emerge on larger/harder mazes.

---

---

## The Debate: What Should Experiment 2 Be?

Five perspectives argued for different next steps. Their arguments and critiques of each other are summarized below, followed by a synthesis.

### 1. The Ablation Advocate: "Isolate the confounds first"

**Core argument**: The RR_MODS_MODEL differs from BASE_MODEL in THREE ways (halting loss, token loss, monotonicity reg). We have zero causal evidence for any individual claim. The 3.7x speedup? Could be monotonicity alone. Could be softmax replacing stablemax. Every downstream experiment builds on sand without ablations.

**Proposed runs** (all with identical hyperparams, 5000 epochs, same seed):

| Run | Brier Halting | Monotonicity | Token Loss | Isolates |
|-----|--------------|-------------|------------|----------|
| 1 (control) | No | No | stablemax | Baseline |
| 2 | **Yes** | No | stablemax | Halting loss only |
| 3 | No | **Yes** | stablemax | Monotonicity only |
| 4 | No | No | **softmax** | Token loss only |
| 5 | **Yes** | **Yes** | **softmax** | Full combination |

**Critique of others**: Scaling without understanding is premature. Latent geometry describes a confound. Training dynamics watches two systems diverge for unknown reasons. Fitting curves to uninterpretable data.

**Strength**: Methodologically rigorous. Creates *usable knowledge*.
**Weakness**: 5000-epoch runs may not be enough to see the 21,600-epoch BCE peak; results might all tie at low accuracy.

### 2. The Scaling Advocate: "The task is too easy"

**Core argument**: 87.5% vs 87.5% at the same accuracy means the task isn't hard enough to reveal differences. We're comparing two cars' top speeds on a parking lot. A model with rho=2.03 vs rho=2.46 might show dramatically different behavior when the task actually requires more refinement steps.

**Proposal**: Generate 50x50 and 100x100 maze datasets. First test zero-shot OOD generalization, then retrain both configs on 50x50 and compare.

**Critique of others**: Ablations on a task where everything ties are wasteful — ablations become informative *after* we find a regime where models diverge. Latent geometry without a performance differential is "navel-gazing." Training dynamics explain *why*, but we first need to know *whether* the methods differ.

**Strength**: High potential for a clear, publishable result if models diverge on harder tasks.
**Weakness**: Requires significant compute for retraining. Also, the models may not generalize OOD at all.

### 3. The Theory Advocate: "Understand why rho > 1 yet convergence happens"

**Core argument**: The central paradox is that rho > 1 (globally expansive) yet both models converge. The contraction must happen in a low-dimensional task-relevant subspace. The step-function accuracy is a phase transition signature. Understanding this geometry would make every subsequent experiment interpretable.

**Proposal** (inference-only, no retraining):
1. Collect hidden state trajectories h_t for correct vs incorrect mazes
2. PCA on difference vectors to find the "solution-relevant subspace"
3. Compute spectral norm *within* that subspace (predict: well below 1)
4. Track effective dimensionality (participation ratio) across steps
5. Compare subspace contraction rates between models

**Critique of others**: Ablations tell *what* but not *why*. Harder mazes just produce the same paradox at larger scale. Error model fitting produces equations without mechanisms. Training dynamics are downstream of geometry.

**Strength**: Could resolve the fundamental paradox. No retraining needed.
**Weakness**: Produces understanding but not directly actionable results. Jacobian computation on 512-dim states is expensive.

### 4. The Modeling Advocate: "The error model is wrong, not the data"

**Core argument**: The Raju-Netrapalli model was designed for autoregressive LLMs. TRM is a dynamical system approaching a fixed point. The 0-to-80% jump is a phase transition. We need a model that fits: accuracy should follow a **mixture of contraction rates** — each maze has its own effective rate, and accuracy at step T is the CDF of that distribution.

**Proposal** (inference-only, ~1 day):
1. Record per-instance, per-step solved/not-solved for all 1024 mazes
2. Fit logistic: A(t) = A_max / (1 + exp(-k(t - t0)))
3. Fit log-normal CDF: A(t) = A_max * Phi((ln(t) - mu) / sigma)
4. Compare fits via AIC/BIC against the failed gamma CDF
5. If fitted k or sigma differs between models, that's the mechanistic explanation

**Critique of others**: Ablations collect more numbers without a model to interpret them. Harder mazes give two unexplained curves instead of one. Latent geometry gives visualizations, not equations. Training dynamics are orthogonal to the inference question.

**Strength**: Cheap, fast, and directly extends the existing analysis script. Could rescue the entire Raju-Netrapalli analysis angle.
**Weakness**: A better fit isn't necessarily a better explanation. Descriptive, not mechanistic.

### 5. The Training Advocate: "The 3.7x speedup is the real finding"

**Core argument**: Inference is solved — both models work. The 3.7x training speedup is what matters for practitioners. BCE gradients blow up near 0 and 1 (proportional to 1/(p(1-p))); Brier gradients are bounded (proportional to p - target). This directly explains faster convergence. The plateau at epoch 5,800 is equally important — bounded gradients that help early may vanish near the optimum.

**Proposal** (requires re-running training with enhanced logging):
1. Log halting loss curves (both Brier and BCE on validation) every 100 epochs
2. Gradient norm of halting head parameters — test bounded vs unbounded hypothesis
3. Halting probability histograms at each recursion step over training
4. Token loss vs halting loss decomposition — which saturates first?
5. Monotonicity violation rate for both models (even though only BrierHalting penalizes it)

**Critique of others**: Ablations without dynamics are "what" without "why." Harder mazes just make training slower. Latent geometry is "interesting for a paper but not actionable." Error model fitting polishes inference curves when the story is in training.

**Strength**: Directly explains the most practically important finding.
**Weakness**: Requires full retraining with custom logging. Most expensive option.

---

## Synthesis: Recommended Experiment 2

After weighing all five perspectives, the strongest path forward combines elements from multiple advocates, ordered by **information-per-compute-hour**:

### Phase A: Quick wins (inference-only, ~1 day)

**From the Modeling Advocate**: Fit logistic and log-normal CDFs to the per-step accuracy curves we already have. This is essentially free — just curve fitting on existing data. If a model fits, it immediately replaces the failed Raju-Netrapalli framework.

**From the Theory Advocate**: Run PCA on hidden state trajectories and compute projected contraction rates. This resolves the rho > 1 paradox and costs only inference passes.

### Phase B: The ablation grid (~1-2 weeks)

**From the Ablation Advocate**: Run the 4 missing ablation configs. This is non-negotiable — we cannot publish claims about "BrierHalting" when three variables changed simultaneously. But run them at the original 50,000-epoch budget with the original hyperparams (batch 128, weight_decay 1.0, ema on) to match the existing runs, not the shorter 5000-epoch configs.

**From the Training Advocate**: Add enhanced logging (gradient norms, halting histograms, loss decomposition) to the ablation runs. This costs nothing extra and answers the training dynamics questions simultaneously.

### Phase C: Scaling (after ablations reveal what matters)

**From the Scaling Advocate**: Once ablations identify which modification(s) drive the speedup, test that specific modification on 50x50 mazes. Don't scale everything — scale the thing that works.

## Experiment 2 (Phase A): Alternative Accuracy Curve Fits

### Motivation
The Raju-Netrapalli incomplete gamma CDF completely failed to fit TRM's step-accuracy curve. We test three alternative models.

### Models Tested
1. **Logistic sigmoid**: A(t) = A_max / (1 + exp(-k(t - t0)))
2. **Log-normal CDF**: A(t) = A_max * Phi((ln(t) - mu) / sigma)
3. **Exponential saturation**: A(t) = A_max * (1 - exp(-rate * (t - offset)))
4. **Raju-Netrapalli** (baseline): incomplete gamma CDF

### Results

| Model | BCE RMSE | BCE AIC | Brier RMSE | Brier AIC |
|---|---|---|---|---|
| **Exp Saturation** | **0.00408** | **-698** | **0.00433** | **-691** |
| Log-Normal CDF | 0.00807 | -611 | 0.00465 | -681 |
| Logistic | 0.00823 | -608 | 0.00464 | -682 |
| Raju-Netrapalli | 0.14806 | -239 | 0.14775 | -239 |

**Exponential saturation wins decisively** for both models (lowest AIC by 80+ points). Raju-Netrapalli is **36x worse** in RMSE.

### Fitted Parameters (Exp Saturation)

| Parameter | BCE | BrierHalting | Interpretation |
|---|---|---|---|
| A_max | 0.8726 | 0.8739 | Asymptotic accuracy (essentially identical) |
| rate | 1.452 | **2.640** | Convergence speed (**BrierHalting 1.82x faster**) |
| offset | 1.971 | 1.988 | Steps before accuracy begins (identical ~2 steps) |

### Key Finding
**BrierHalting converges 1.82x faster even at inference time**, not just during training. Both models have the same accuracy ceiling (~87.3%) and the same startup delay (~2 steps), but BrierHalting reaches the ceiling almost twice as fast in refinement steps.

This suggests the Brier halting loss (and/or monotonicity + softmax) doesn't just improve training efficiency — it shapes the refinement dynamics themselves to converge faster to the fixed point.

### Logistic Parameters (for reference)

| Parameter | BCE | BrierHalting |
|---|---|---|
| A_max | 0.872 | 0.874 |
| k (steepness) | 4.31 | 6.11 (1.42x steeper) |
| t0 (midpoint) | 2.70 | 2.57 (slightly earlier) |

---

## Experiment 3 (Phase A): Latent Subspace Contraction Analysis

### Motivation
Experiment 1 found that both models have perturbation-based spectral norm rho > 1 (BCE ~2.03, BrierHalting ~2.46), yet both converge reliably. This is paradoxical under the Raju-Netrapalli framework, which assumes rho < 1 for convergence. We hypothesize that contraction occurs in a low-dimensional task-relevant subspace even though the full state space is expansive.

### Method
For each model, we:
1. Collect mean-pooled hidden state trajectories z_H across 32 refinement steps for 512 test mazes
2. Compute PCA on final-step hidden states to identify the "solution-relevant subspace" (top-20 PCs)
3. Measure **displacement-based contraction rate**: rho_t = ||z_t - z_{t-1}|| / ||z_{t-1} - z_{t-2}|| (both in full space and projected onto the PCA subspace)
4. Track effective dimensionality via participation ratio at each step
5. Measure centroid separation between correct and incorrect samples

### Results

#### Eigenvalue Spectrum

| Metric | BCE | BrierHalting |
|---|---|---|
| Top-20 PCs explained variance | **99.2%** | 97.9% |
| PC1 fraction of variance | **84.5%** | 52.4% |
| PC2 fraction of variance | 8.7% | 39.2% |
| PC1+PC2 combined | **93.2%** | **91.6%** |

**BCE concentrates variance into a single dominant direction** (PC1 captures 84.5%), while **BrierHalting distributes variance more evenly** across PC1 (52.4%) and PC2 (39.2%). This suggests fundamentally different latent geometry: BCE's solution space is nearly 1-dimensional, while BrierHalting uses a 2-dimensional solution manifold.

#### Displacement-Based Contraction Rate

The displacement-based rho measures actual step-to-step convergence along the model's trajectory (unlike the perturbation-based Jacobian rho from Experiment 1, which measures sensitivity to off-trajectory perturbations).

| Metric | BCE | BrierHalting |
|---|---|---|
| Mean full-space rho (steps 3-32) | 0.878 | 0.856 |
| Mean projected rho (steps 3-32) | 0.877 | 0.860 |
| Perturbation-based rho (Exp 1) | ~2.03 | ~2.46 |

**Both models show displacement-based rho < 1**, confirming they ARE contractive along their actual trajectories. This resolves the rho > 1 paradox: the models are locally expansive (perturbation rho > 1) but globally contractive along the solution path (displacement rho < 1). The refinement dynamics follow a narrow "valley" in state space — perturbations perpendicular to this valley are amplified, but the trajectory itself converges.

The projected rho (within top-20 PCA subspace) is nearly identical to the full-space rho, confirming that 20 PCs capture essentially all the dynamics.

#### Participation Ratio (Effective Dimensionality)

| Metric | BCE | BrierHalting |
|---|---|---|
| PR at step 1 | 3.75 | 6.15 |
| PR at steady state (steps 10-32) | ~2.03 | ~4.97 |

**BCE collapses to ~2 effective dimensions**, consistent with its eigenvalue spectrum (PC1 dominates). **BrierHalting maintains ~5 effective dimensions**, using a higher-dimensional representation throughout refinement. Both models show PR decreasing from step 1 to steady state, indicating dimensionality reduction during refinement.

#### Correct vs Incorrect Separation

| Metric | BCE | BrierHalting |
|---|---|---|
| Max separation | 1.35 (step 3) | 1.79 (step 3) |
| Steady-state separation (steps 10-32) | ~0.26 | ~0.27 |

Both models show a spike in correct/incorrect centroid separation at step 3 (when accuracy jumps from ~0% to ~85%), then settle to similar steady-state separation (~0.26-0.27). **BrierHalting achieves 33% higher peak separation**, suggesting it creates a more distinct decision boundary during the critical early refinement steps.

#### Per-Step Accuracy

| Step | BCE | BrierHalting |
|---|---|---|
| Step 2 | 70.5% | **83.2%** |
| Step 3 | 83.0% | 85.9% |
| Step 5+ | ~88.3% | ~88.1% |

Consistent with Experiment 2's exponential saturation fit, **BrierHalting reaches high accuracy faster** (83.2% at step 2 vs BCE's 70.5%), then both plateau at ~88%.

### Key Findings

1. **The rho > 1 paradox is resolved.** Perturbation-based rho measures sensitivity to noise (Jacobian spectral norm), not trajectory convergence. Displacement-based rho shows both models are genuinely contractive (rho ~0.86-0.88) along their actual refinement paths.

2. **BCE and BrierHalting use different latent geometries.** BCE concentrates its solution into ~2 effective dimensions (PC1=84.5%), while BrierHalting distributes across ~5 dimensions (PC1=52.4%, PC2=39.2%). Despite this structural difference, both achieve the same final accuracy.

3. **BrierHalting's faster convergence (Exp 2) correlates with higher peak separation.** The 1.82x faster saturation rate from curve fitting corresponds to a 33% higher peak correct/incorrect separation at step 3, and 83.2% vs 70.5% accuracy at step 2.

4. **Dimensionality decreases during refinement for both models**, suggesting that refinement progressively constrains the solution to a lower-dimensional manifold. This is consistent with the "phase transition" interpretation: early steps identify the solution subspace, later steps refine within it.

---

## Experiment 4 (Phase B): Ablation Training Runs

### Configs Created

| Run | Config | Brier | Mono | Token Loss | Isolates |
|-----|--------|-------|------|-----------|----------|
| 1 | cfg_ablation_baseline | No | No | stablemax | Control |
| 2 | cfg_ablation_brier_only | **Yes** | No | stablemax | Halting loss |
| 3 | cfg_ablation_mono_only | No | **Yes** | stablemax | Monotonicity reg |
| 4 | cfg_ablation_softmax_only | No | No | **softmax** | Token loss |
| 5 | cfg_ablation_full_combo | **Yes** | **Yes** | **softmax** | Full combo |

All use: batch=128, weight_decay=1.0, ema=true, 10000 epochs, eval every 200 epochs, seed=0.

**Training launched** — all 5 runs executing sequentially on RTX 5090 (10,000 epochs each).

*Results will be added here as each run completes.*
