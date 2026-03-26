# Ablation Study Results: Isolating Training Modifications in Tiny Recursive Models

## Overview

This document reports the results of a controlled ablation study on the Tiny Recursive Model (TRM) architecture for 30x30 maze solving. We trained 5 model variants that systematically toggle three modifications (Brier halting loss, softmax token loss, monotonicity regularization) and then performed post-training analysis using the Raju-Netrapalli error model framework and PCA-based latent subspace analysis.

## 1. Training Results

All models trained for 5,000 epochs on the maze-30x30-hard-1k dataset (1,000 mazes, 8x augmented to 8,000 training examples) with batch size 256, hidden size 512, 2 L-layers, H_cycles=3, L_cycles=6.

| Run | Brier | Mono | Softmax | Peak Token Acc | Peak Step | Best Exact Acc | Best Exact Step |
|-----|:-----:|:----:|:-------:|:--------------:|:---------:|:--------------:|:---------------:|
| 1. BASELINE | - | - | - | 97.94% | 5,859 | 9.10% | 15,624 |
| 2. BRIER_ONLY | Yes | - | - | 96.86% | 3,906 | 19.80% | 19,530 |
| 3. MONO_ONLY | - | Yes | - | 96.73% | 15,624 | 4.20% | 15,624 |
| 4. SOFTMAX_ONLY | - | - | Yes | 97.43% | 13,671 | 6.90% | 11,718 |
| 5. FULL_COMBO | Yes | Yes | Yes | 97.42% | 7,812 | 30.80% | 7,812 |

**Key finding:** Brier halting drives faster training convergence (peaked at step 3,906 vs baseline 5,859). FULL_COMBO achieves the highest exact accuracy (30.80%) by a large margin.

## 2. Raju-Netrapalli Error Model Analysis

We estimated the Jacobian spectral norm (rho) at each refinement step using perturbation analysis and power iteration, and measured accuracy across 64 refinement steps.

| Run | Perturbation rho | Power Iter rho | Peak Exact (64 steps) | Peak Step | Snap-back Peak |
|-----|:----------------:|:--------------:|:---------------------:|:---------:|:--------------:|
| BASELINE | 5.35 | 11.01 | 13.28% | 5 | 2.34% |
| BRIER_ONLY | 5.48 | 22.42 | 21.88% | 14 | 2.34% |
| MONO_ONLY | 4.88 | 12.57 | 8.20% | 5 | 2.34% |
| SOFTMAX_ONLY | 6.66 | N/A | 9.47% | 6 | 2.34% |
| FULL_COMBO | 9.64 | N/A | 32.42% | 7 | 2.34% |

**Key findings:**
- All models have perturbation rho >> 1 (globally expansive Jacobian)
- The 3-parameter accuracy law fit failed for all runs (hit parameter bounds), indicating the simple Raju-Netrapalli model doesn't directly apply to TRM's expansive regime
- Argmax snap-back uniformly hurt performance (2.34% vs 8-32%) — re-embedding destroys useful information in the continuous latent state
- BRIER_ONLY benefits from more refinement steps (peaks at step 14 vs baseline step 5)

## 3. PCA Latent Subspace Analysis

We tracked hidden state trajectories, computed displacement-based contraction rates, effective dimensionality, and correct/incorrect separation.

| Run | Displacement rho | Eff. Dim (PR) | PC1 Var% | Separation | Explained Var (top-20) |
|-----|:----------------:|:-------------:|:--------:|:----------:|:----------------------:|
| BASELINE | 0.847 | 1.7 | 76.9% | 0.80 | 99.2% |
| BRIER_ONLY | 0.836 | 2.1 | 68.5% | 0.54 | 98.9% |
| MONO_ONLY | 0.864 | 1.8 | 73.1% | 0.28 | 99.1% |
| SOFTMAX_ONLY | 0.860 | 4.1 | 42.8% | 2.18 | 97.0% |
| FULL_COMBO | 0.847 | 3.0 | 56.1% | 3.34 | 96.2% |

**Key findings:**

### The rho > 1 Paradox Resolution
All models have perturbation-based rho >> 1 (globally expansive) but displacement-based rho < 1 (trajectory-contractive at ~0.84-0.86). This means TRM's refinement map is expansive in most random directions but contractive along the actual solution trajectory. The model converges despite global instability because it operates on an attracting manifold.

### Softmax Shatters Latent Geometry
Switching from stablemax to softmax dramatically changes the representation:
- PC1 variance drops from 76.9% to 42.8% (the latent space becomes more distributed)
- Effective dimensionality jumps from 1.7 to 4.1
- Correct/incorrect separation increases from 0.80 to 2.18

### FULL_COMBO Creates Maximum Discrimination
The combination of all three modifications achieves:
- Highest correct/incorrect centroid separation (3.34 vs 0.80 baseline)
- Balanced dimensionality (PR=3.0)
- Strong trajectory contraction (rho=0.847)

### Monotonicity Alone Hurts
Monotonicity regularization in isolation produces the worst results:
- Highest displacement rho (0.864 — least contractive)
- Lowest separation (0.28)
- Worst peak accuracy (8.20%)

## 4. Summary of Which Modification Drives What

| Effect | Primary Driver | Evidence |
|--------|---------------|----------|
| Faster training convergence | Brier halting | Peaked at step 3,906 vs 5,859 baseline |
| Higher exact accuracy | All three combined | 30.80% (FULL_COMBO) vs 9.10% (BASELINE) |
| Most contractive trajectory | Brier halting | Lowest displacement rho (0.836) |
| Distributed latent geometry | Softmax token loss | PC1 variance 42.8% vs 76.9%, PR 4.1 vs 1.7 |
| Best correct/incorrect separation | All three combined | Separation 3.34 vs 0.80 baseline |
| Snap-back error correction | None (uniformly harmful) | 2.34% for all runs |

## 5. Implications

1. **Brier halting loss** is the single most impactful modification — it drives both faster training and more contractive inference dynamics. The bounded gradient signal (vs BCE's unbounded gradients) likely helps the model learn a cleaner attractor.

2. **The Raju-Netrapalli error model needs modification for TRM.** The perturbation-based spectral norm is not the right measure for iterative refinement — displacement-based contraction along the solution manifold is what matters.

3. **Argmax snap-back is harmful for TRM.** Unlike autoregressive models where discrete token projection acts as error correction, TRM's continuous latent state carries information that argmax destroys.

4. **The modifications are synergistic.** While Brier alone helps training speed and softmax alone changes geometry, the combination (FULL_COMBO) achieves dramatically better exact accuracy than any single modification.

## Files

- `results/ablation_analysis/<RUN>_rn/results.json` — Raju-Netrapalli analysis data
- `results/ablation_analysis/<RUN>_pca/pca_results.json` — PCA latent analysis data
- `results/ablation_analysis/<RUN>_rn/analysis.png` — RN analysis plots
- `results/ablation_analysis/<RUN>_pca/pca_analysis.png` — PCA analysis plots
- `results/ablation_data.json` — Aggregated data for all runs

## Wandb

All training runs logged to: https://wandb.ai/jiekaitao-university-of-florida/SeniorProjectTRM
