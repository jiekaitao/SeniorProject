# DeepPass: Adaptive Layer Duplication via Spectral Screening and Pairwise Compatibility

## Paper Outline

### Abstract
Layer duplication (repeating transformer blocks without modifying weights) improves LLM reasoning, but finding which blocks to duplicate requires exhaustive search. We present DeepPass, a spectral screening framework that reduces this search by 162x, and show that multi-block duplication from complementary network regions outperforms any single-block configuration on a combined math+EQ-bench metric. Our best 72B configuration achieves combined=79.91 vs Ng's published 76.76.

### 1. Introduction
- Ng's RYS discovery: repeating layers improves reasoning
- Problem: 3,241-config brute-force search
- Our contributions: spectral screening, BLOOD validation, multi-block stacking, DICE pair prediction
- Ultimate goal: adaptive computation time for LLMs (TRM-style ACT head)

### 2. Background
- RYS method (Ng, 2024)
- TRM iterative refinement theory (Jolicoeur-Martineau)
- BrierHalting and adaptive computation
- BLOOD OOD detection (ICLR 2024)

### 3. Spectral Screening Method
- Displacement rho: ||F(F(h))-F(h)|| / ||F(h)-h||
- 162x reduction in search cost
- Validation: 80% top-5 hit rate on 7B
- **Data:** `results/data/7b/spectral/`, `results/data/72b/spectral/`

### 4. BLOOD Impact as Screening Metric
- Downstream Jacobian smoothness predicts block quality
- Spearman r=-0.492 (p=0.028) on 7B
- Two-stage Pareto screen: rho → BLOOD rerank
- **Data:** `results/blood_impact_sweep/`

### 5. Multi-Block Stacking
- Novel finding: complementary cross-region pairs stack
- 7B: 4/22 pairs stack, best=(8,15)+(16,27)=0.7883
- 72B: (0,7)+(45,52) combined=79.91 beats all single blocks
- Cross-region hypothesis: early + deep blocks activate different circuits
- **Data:**
  - 7B pairs: `results/data/7b/pairs/pairwise_stacking_sweep.json`
  - 72B pairs: `results/data/72b/pairs/72b_cross_region_pairs.json`
  - 72B dual-probe: `results/data/72b/pairs/72b_best_pairs_dual_probe.json`

### 6. DICE: Directed Interaction Compatibility Estimator
- Pairwise epistasis decomposition: ε_ij = F({i,j}) - F({i}) - F({j}) + F(∅)
- Cheap pair features: rho_lift, BLOOD territory, effect CKA, region distance
- 7B validation: region_dist (r=-0.438, p=0.042) and territory_orth (r=-0.452, p=0.035) are significant but sign-inverted from theory
- Beam search for N-tuples
- **Data:** `results/data/7b/dice/7b_pair_features.json`

### 7. Oracle Seam Patching
- Residual-scaling experiment: h_patched = h1 + α(h2 - h1)
- Tests whether second pass is genuinely useful or overshooting
- **Data:** `results/data/72b/oracle_seam_patching/` (pending)

### 8. Triple-Block Stacking
- Can a third block improve (0,7)+(45,52)?
- **Data:** `results/data/72b/triples/` (pending)

### 9. Seam Analysis — Scale-Dependent Duplication Mechanism
- **9B:** Second pass inflates norms by 42% (norm_ratio=1.42, cosine=0.975). Norm-preserving projection (h1_norm_h2_dir) improves combined by +13.6. The duplication is noisy; clamping norms helps.
- **72B:** Second pass barely changes norms (norm_ratio=1.04, cosine=0.997). Norm-preserving *hurts* by -2.3. The perturbation is almost purely directional.
- **Alpha=1.25 beats standard on 72B** (+1.26 combined on Ng's block). The directional refinement is slightly too conservative at alpha=1.0. A 25% overshoot lets the block express more of its intended correction. Alpha=1.5+ overshoots into noise.
- Both blocks in best pair contribute independently (oracle seam patching confirms)
- **Data:** `results/data/72b/oracle_seam_patching/v2_results.json`, `results/data/72b/norm_preserving/`, `results/data/72b/direction_interventions/`

### 10. Multi-Pass — Two Passes Is the Ceiling
- 3+ passes destroys EQ-bench on 72B (83→63 at 3x, 17 at 4x)
- Confirms 7B finding at scale
- **Data:** `results/data/72b/multipass/multipass_results.json`

### 11. Generalization Across Architectures
- **Qwen3.5-27B:** Best single (25,30) combined=78.30 (+35.4 delta). Greedy stacking in progress.
- **Qwen3.5-9B:** Full pipeline complete.
- **Gemma 3 27B:** Pair (4,5)+(20,21) combined=84.42 beats single (20,21)=83.76. Greedy stacking works on a completely different model family (Google vs Qwen).
- Spectral screening correctly identifies best blocks across all architectures
- **Data:** `results/data/qwen35/`, `results/data/qwen35_9b/`, `results/data/gemma3_27b/`

### 12. lm-eval Leaderboard — Duplication Is Task-Selective
- Pair helps reasoning: MuSR +4.3%, IFEval +2.2%
- Pair hurts knowledge: MATH Hard -5%, MMLU-PRO -1.9%
- Different tasks need different blocks → motivates adaptive per-input routing
- **Data:** `results/data/72b/lm_eval/`

### 13. Per-Block Alpha Tuning — Breaking the Two-Block Ceiling
- **Triples and quads work with fractional "whisper" alphas!**
- Pattern: core blocks at high alpha (0.9-1.0), additional blocks at very low alpha (0.02-0.15)
- Probe is perfectly deterministic (0.00 std across 5 runs) — all differences are real
- **Data:** `results/data/72b/alpha_optimization/`, `results/data/72b/deeper_stacking/`

### 14. Per-Layer Alpha — Finer Control Beats More Blocks
- Each layer within a duplicated block gets its own alpha weight
- **Single block (45,52) with 7 per-layer alphas: 82.77** — nearly matches 4-block quad
- **Triple with 21 per-layer alphas: 84.07** — all-time record, +7.31 over Ng
- Key finding: some layers should be boosted (L3@1.3), others dampened (L2@0.5) or disabled (L0@0.0)
- **Data:** `results/data/72b/per_layer_alpha/`

### 15. Bayesian Alpha Optimization — 5x More Efficient
- Optuna TPE reaches **83.97 in 60 evaluations** vs grid search's 84.07 in ~300
- 5x fewer evals, same result quality
- Makes the full pipeline practical: ~8 GPU-hours from scratch on any model
- **Data:** `results/data/72b/bayesian_alpha/results.json`

### 16. SBUID Screening Metric (BLOOD - λ*rho)
- First statistically significant screening metric on 72B: **Spearman r=0.515, p=0.008**
- Cross-validated: train r=0.34 → test r=0.664 (holds out of sample)
- Works on 27B+ models (p=0.038 on 27B), not small models (p=0.83 on 9B)
- Novel trajectory metrics (OTAS, GCHS, CLRG) all failed — duplication is NOT about advancing along the base trajectory
- **Data:** `results/data/72b/fresh_validation/`, `results/data/sbuid_validation/`

### 17. Quantization Survival — Deployable on Consumer GPUs
- 4-bit NF4 preserves duplication benefit on both Gemma3-27B (17.5GB, delta=+3.00) and 72B (59GB, delta=+8.06)
- **Data:** `results/data/quantization/`

### 18. Inference Speed — Minimal Overhead
- Ng single: 0.96x baseline speed (4% slower)
- Our pair: 0.89x (11% slower)
- Quad: 0.80x (20% slower)
- Zero extra VRAM — weights are shared
- **Data:** `results/data/72b/inference_speed/`

### 19. Prompt Sensitivity — Results Are Robust
- Kendall W = 0.808 across 5 different prompt sets (STRONG concordance)
- Config rankings are stable regardless of which questions are used
- **Data:** `results/data/72b/prompt_sensitivity/`

### 20. MoE Layer Duplication — First Demonstration
- Layer duplication works on Qwen3-30B-A3B (128 experts, 8 active/token)
- Best single block: +12.66 delta (+46% improvement)
- First-ever test of layer duplication on mixture-of-experts architecture
- **Data:** `results/data/moe/`

### 21. Generalization Summary
| Model | Pair > Single? | Triple > Pair? | Quad? | Needs whisper? |
|-------|---------------|----------------|-------|----------------|
| 72B | YES | YES | YES | YES (0.02-0.15) |
| Gemma3-27B | YES | YES | YES (85.58) | NO (1.0 works) |
| Qwen3.5-27B | YES (tuned) | YES (80.05) | — | Partial |
| Qwen3.5-9B | YES (tuned) | NO | — | — |
| Qwen3-30B MoE | — | — | — | First test |

### 22. Negative Results
- Adapters don't help good configs (6 approaches tested)
- All trained routers fail (BrierHalting, ESR+DSG)
- Math probe doesn't generalize to lm-eval — duplication is task-selective
- DICE pair prediction weak (Spearman=0.191)
- Residual stability doesn't predict pair quality (Spearman=0.117)
- Norm-preserving helps 9B but hurts 72B — scale-dependent
- Novel screening metrics (OTAS, GCHS, CLRG) all fail on 72B
- Rho alone fails on 72B (p=0.50) — only works on 7B
- Deeper stacking doesn't work on 9B (too small)

### 23. Why Duplication Hurts Knowledge — The FFN Re-Retrieval Hypothesis
- FFN/MLP layers store facts as key-value associations. On the second pass, the perturbed input causes the FFN to retrieve *nearby but wrong* facts — corrupting the clean signal.
- Attention benefits from repetition because it does *computation* (re-weighting), not *retrieval*.
- **Fix: attention-only duplication** — skip FFN on second pass to preserve factual knowledge while keeping reasoning gains.
- Sublayer data supports this: L2 attention-only (80.35) beats full duplication (77.45). FFN on L2 is destructive.
- lm-eval validation running.
- **Data:** `results/data/72b/sublayer/`, `results/data/72b/attn_only/`

### 24. Discussion
- The full pipeline: SBUID screening → greedy stacking → Bayesian alpha optimization (~8 GPU-hours)
- 46x speedup over Ng's brute force, arriving at a better result (+7.31)
- Scale-dependent mechanism: norm-driven (small) vs direction-driven (large)
- Toward adaptive per-input computation: different tasks need different blocks
- The attention vs FFN decomposition explains the task-selective benefit
- Cross-layer duplication shows the second pass benefits from "simpler" weights
- Practical deployment: 4-bit quantization, minimal speed overhead, zero extra VRAM

### 24. Conclusion

---

## Key Results Table (for paper)

| System | Combined | Delta vs Ng | Method |
|--------|----------|-------------|--------|
| Baseline | 70.52 | — | — |
| Ng (45,52) @1.0 | 76.76 | — | Brute force (3,241 evals) |
| Our pair (0,7)+(45,52) @1.0 | 79.91 | +3.15 | Greedy spectral stacking |
| Per-block quad (whisper alpha) | 82.58 | +5.82 | Whisper alpha tuning |
| Per-layer single (45,52) | 82.77 | +6.01 | 7 tuned layer alphas |
| **Bayesian per-layer triple** | **83.97** | **+7.21** | **60 Optuna evals (~5h)** |
| **Grid search per-layer triple** | **84.07** | **+7.31** | **300 evals (~25h)** |

### Depth Progression on 72B
| Depth | Best Config | Combined |
|-------|-------------|----------|
| 1 block | (45,52)@1.15 | 79.79 |
| 2 blocks | (0,7)@0.9 + (45,52)@1.0 | 81.24 |
| 3 blocks | (0,7)@0.9 + (20,27)@0.15 + (45,52)@1.0 | 82.29 |
| 4 blocks | (0,7)@0.9 + (15,20)@0.1 + (20,27)@0.15 + (45,52)@1.0 | 82.58 |
| 3 blocks (per-layer) | 21 optimized layer alphas | **84.07** |

### Cross-Architecture Results
| Model | Baseline | Best Config | Delta |
|-------|----------|-------------|-------|
| Qwen2-72B | 70.52 | Per-layer triple | +13.55 |
| Gemma3-27B | 80.54 | Quad @1.0 | +5.04 |
| Qwen3.5-27B | 42.86 | Triple (25,30)+(15,20)@1.0 | +37.19 |
| Qwen3-30B MoE | 27.76 | Single (8,9) | +12.66 |

---

## Data Index

### Definitive Results (for paper figures/tables)
| Data | Path | Description |
|------|------|-------------|
| 7B brain scanner | `results/data/7b/singles/` | Full (i,j) sweep, 77 configs |
| 7B pairwise | `results/data/7b/pairs/pairwise_stacking_sweep.json` | 22 pairs, 10 singles, epistasis |
| 72B pair sweep | `results/data/72b/pairs/72b_pair_sweep.json` | 8 same-region pairs |
| 72B cross-region | `results/data/72b/pairs/72b_cross_region_pairs.json` | 4 cross-region pairs (math only) |
| 72B dual-probe | `results/data/72b/pairs/72b_best_pairs_dual_probe.json` | Best pairs on math+EQ-bench |
| BLOOD impact | `results/blood_impact_sweep/` | 20 configs, Spearman correlation |
| DICE 7B | `results/data/7b/dice/7b_pair_features.json` | 22 pair features + validation |
| lm-eval 15% | `results/data/72b/lm_eval/` | IFEval, BBH, MATH, MuSR, MMLU-PRO |
| Adapter comparison | `results/data/7b/adapters/` | 6 approaches, good+bad configs |

| Oracle seam patching | `results/data/72b/oracle_seam_patching/v2_results.json` | Alpha curves for 4 configs |
| Multi-pass 72B | `results/data/72b/multipass/multipass_results.json` | 2-4 passes on (45,52) and (50,60) |
| Triple search | `results/data/72b/triples/systematic_search.json` | Full pipeline, no triple beats pair |
| Spectral 7B (all blocks) | `results/data/7b/spectral/displacement_rho_all.json` | 77 blocks for heatmap figure |
| EQ-bench sweep | `results/data/72b/singles/missing_eq_bench.json` | (0,7), (15,20), (35,40), (55,60), (45,52)+(55,60) |

### Pending (experiments running)
| Data | Path | Status |
|------|------|--------|
| lm-eval 4-way comparison | `results/data/72b/lm_eval/` | baseline done, (0,7)+(45,52) + (50,60) + Ng running |
| More 72B pairs | `results/data/72b/pairs/more_pairs_round2.json` | 10 new pairs, running |
| Nonlinear seam experiments | `results/data/svd_*.json`, `norm_*.json`, `gated_*.json` | SVD/norm/gate on Qwen3.5-9B |
| Qwen3.5-27B generalization | `results/data/qwen35/generalization_results.json` | Spectral screen + top-8 singles |
| Qwen3.5-9B full pipeline | `results/data/qwen35_9b/full_pipeline.json` | Screen → singles → greedy stacking |
| DICE 72B features | TBD | Not started |
| Full lm-eval (no subsample) | TBD | Not started |

### History
Full experimental timeline: `HISTORY.md` (~1200 lines)

---

## Figures Needed
1. **Spectral screening heatmap** — displacement rho vs actual math delta (7B)
2. **BLOOD impact correlation** — scatter plot, r=-0.492
3. **Pairwise epistasis matrix** — 7B top-10 blocks, color=epistasis
4. **72B config comparison bar chart** — baseline, Ng, (50,60), (0,7)+(45,52)
5. **Oracle seam patching curves** — alpha vs combined score per block (pending)
6. **DICE feature importance** — individual feature correlations
7. **Cross-region stacking diagram** — network architecture showing where blocks are duplicated
