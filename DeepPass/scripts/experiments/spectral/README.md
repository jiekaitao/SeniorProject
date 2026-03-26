# Spectral Analysis Experiments

Uses TRM-inspired spectral metrics to predict which layer blocks benefit from duplication, replacing Ng's brute-force 3,241-config sweep.

## Key Metrics

- **Displacement rho:** `||F(F(h)) - F(h)|| / ||F(h) - h||` — convergence along refinement trajectory
- **Perturbation rho:** `||F(h+eps) - F(h)|| / ||eps||` — global Jacobian spectral norm
- **Residual magnitude:** `||F(F(h)) - F(h)||` — how much work a second pass does

## Files

- **`spectral_analysis.py`** — Core spectral screening on any model. Computes displacement/perturbation rho for all candidate blocks. Generates heatmaps and rankings.

- **`spectral_guided_search_72b.py`** — Used spectral screening to find (50,60) config on 72B that beats Ng's (45,52) by 74% on math probe, with 162x fewer evaluations.

- **`validate_spectral.py`** — Correlation analysis between spectral predictions and brain scanner ground truth. Result: 80% top-5 hit rate, but weak Spearman correlation.

- **`deeppass_analysis.py`** — Unified TRM-RYS framework. Connects spectral metrics to Raju-Netrapalli error model. Explains the "rho > 1 paradox": blocks are globally expansive but locally contractive along the solution manifold.

## Key Result

Spectral pre-screening efficiently finds math-probe-optimal configs. However, math probe improvement (+15.4%) does NOT generalize to lm-eval benchmarks (-1.1%). The spectral method needs a broader objective function or a per-input routing approach (see routing/).
