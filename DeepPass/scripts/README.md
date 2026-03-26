# DeepPass Scripts

## Directory Structure

```
scripts/
├── core/                          # Foundation modules (imported by everything)
│   ├── layer_duplicator.py        # RYS layer duplication engine
│   ├── math_probe.py              # Ng's hard math guesstimate probe
│   ├── save_duplicated_model.py   # Save duplicated models with deep-copied layers
│   └── compile_results.py         # Aggregate JSON results
│
├── experiments/
│   ├── discovery/                 # Brain scanner sweeps and benchmarks
│   │   ├── brain_scanner.py       # Full (i,j) heatmap sweep
│   │   ├── benchmark.py           # CLI wrapper for math probe
│   │   └── quick_test.py          # Minimal sanity check
│   │
│   ├── multi/                     # Multi-block and multi-pass experiments
│   │   ├── multi_block_test.py    # Dual-block duplication (interference, not stacking)
│   │   ├── multi_pass_test.py     # N-pass duplication (2 passes optimal)
│   │   ├── even_odd_test.py       # Parity hypothesis (rejected)
│   │   └── adaptive_depth.py      # Per-input adaptive pass count
│   │
│   ├── junction_ft/               # Junction fine-tuning V1 → V4
│   │   ├── junction_ft_v1.py      # V1: Logit KL, 2 layers
│   │   ├── junction_ft_v2.py      # V2: + Procrustes init
│   │   ├── junction_ft_v3.py      # V3: Hidden-state MSE, 4 layers, config-aware
│   │   ├── junction_ft_v3_72b.py  # V3 adapted for 72B (two-stage loading)
│   │   ├── junction_ft_v4_adapter.py  # V4: Bottleneck adapter (current best)
│   │   └── README.md              # Detailed version history
│   │
│   ├── spectral/                  # Spectral analysis and guided search
│   │   ├── spectral_analysis.py   # Core spectral screening
│   │   ├── spectral_guided_search_72b.py  # Found (50,60) beating Ng
│   │   ├── validate_spectral.py   # Correlation with brain scanner
│   │   └── deeppass_analysis.py   # Unified TRM-RYS framework
│   │
│   └── routing/                   # Adaptive iteration routing (new)
│       ├── routing_diagnostic.py  # Per-input vs per-task feasibility test
│       └── README.md              # ESR + DSG hybrid design
│
├── orchestration/                 # Shell scripts for running experiments
│   ├── overnight_runner.sh        # Original overnight suite
│   ├── overnight_v2.sh            # V2: (45,52) + junction FT + full eval
│   └── run_*.sh                   # Individual experiment runners
│
└── [original flat files]          # Legacy — kept for import compatibility
```

## Key Results Summary

1. Ng's RYS reproduced: +19.8% on math probe
2. Spectral search found (50,60) beating Ng by 74%, with 162x fewer evals
3. Math probe gains do NOT generalize to lm-eval benchmarks
4. (45,52) and (50,60) have complementary strengths → motivates adaptive routing
5. V4 adapter: amazing for bad configs (141% recovery), hurts good configs (KL loss fights improvement)
6. Routing diagnostic: per-input signal exists (86% within-task variance) → geometric router justified
