# Discovery Experiments

Foundational experiments: sweeps, benchmarks, and reproduction of Ng's results.

## Files

- **`brain_scanner.py`** — Full (i,j) sweep heatmap. Tests all possible layer duplication configs. Expensive (hours on 7B). Produced the ground truth for spectral validation.

- **`benchmark.py`** — CLI wrapper for math probe evaluation with optional layer duplication. `--model`, `--i`, `--j`, `--tag`.

- **`quick_test.py`** — Minimal sanity check for a model + config.
