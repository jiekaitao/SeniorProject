# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepPass improves LLM capabilities by duplicating specific transformer layer blocks at inference time — no training, no weight changes. Building on David Ng's RYS (Repeat Your Self) discovery, we developed greedy spectral stacking, per-layer alpha tuning, and sublayer-aware duplication that achieves +7.31 over Ng's result on a 72B model. The project spans 5 model architectures (Qwen2-72B, Qwen3.5-27B/9B, Gemma3-27B, Qwen3-30B MoE) with ~200 GPU experiments.

## Environment & Execution

```bash
# Activate environment
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass

# Interactive GPU session
srun --partition=hpg-b200 --gres=gpu:b200:1 --mem=32G --time=4:00:00 --cpus-per-task=4 --pty bash

# All scripts run from DeepPass root
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
```

**SLURM batch jobs** use account `cis4914`, partition `hpg-b200`, qos `cis4914`. Per GPU: 4 CPUs, 32GB RAM. Time limits: estimated runtime + 50% buffer (never blanket 3-day requests — hurts backfill scheduling).

**Never compile flash attention from source** — causes OOM on login nodes. Use PyTorch's built-in SDPA.

## Running Experiments

```bash
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

# Quick evaluation of a duplicated block
$PYTHON scripts/benchmark.py --model models/full/calme-2.1-qwen2-72b --i 45 --j 52

# Spectral screening (minutes, finds candidate blocks)
$PYTHON scripts/spectral_analysis.py --model models/full/calme-2.1-qwen2-72b --block-sizes "3,5,7"

# lm-eval benchmarks (hours on 72B)
$PYTHON scripts/experiments/lm_eval_runtime_dup.py \
    --model models/full/calme-2.1-qwen2-72b \
    --blocks "45,52" \
    --tasks leaderboard_ifeval,leaderboard_bbh,leaderboard_math_hard,leaderboard_musr,leaderboard_mmlu_pro \
    --limit 0.15

# Typical SLURM batch submission
sbatch scripts/experiments/spectral/per_layer_alpha_72b.sh
```

## Architecture

### Core Modules (scripts/ — no internal imports)

- **`layer_duplicator.py`** — `load_original_model()` handles Qwen/Gemma/GPT architectures (auto-detects `model.layers` vs `transformer.h` vs `model.language_model.layers`). `apply_layer_duplication(model, i, j)` modifies `nn.ModuleList` to run layers [i,j) twice. `generate_no_cache()` for token-by-token generation.

- **`math_probe.py`** — 16 hard arithmetic questions with Ng's partial-credit scoring. `run_math_probe(generate_fn)` → score 0-1. ~5 min per config on 72B.

- **`eq_bench_probe.py`** — 20/171 EQ-Bench questions (emotional intelligence). `run_eq_bench_probe(generate_fn)` → score 0-100. ~60s per config.

- **Combined metric:** `combined = math_score * 50 + eq_score * 0.5` (both contribute ~50 points max).

### Experiment Scripts (scripts/experiments/)

All experiments follow the same pattern:
```python
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe
```

Major subdirectories:
- **spectral/** — Screening metrics, alpha optimization, deeper stacking, sublayer analysis, cross-layer duplication, Bayesian optimization
- **multi/** — Multi-block/multi-pass experiments
- **routing/** — Per-input adaptive block selection (in progress)
- **dice/** — DICE pair prediction features

### Layer Duplication Mechanics

```python
# Standard duplication: layers [i,j) run twice
order = list(range(j)) + list(range(i, j)) + list(range(j, N))
inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
model.config.num_hidden_layers = len(order)

# For Gemma3: inner = model.model.language_model
# For num_hidden_layers: check model.config.text_config.num_hidden_layers
```

### Alpha Blending at Seam

Per-block alpha controls how much of the second pass's correction to keep:
```python
h_out = h1 + alpha * (h2 - h1)  # alpha=1.0 standard, alpha=0.1 "whisper"
```

Per-layer alpha gives each layer in the duplicated block its own weight. Per-sublayer alpha separates attention and FFN contributions.

## Critical Constraints

1. **`use_cache=False` is mandatory** for runtime-duplicated models. Shared layer modules break KV cache indexing (`self.layer_idx` collision). Deep-copied layers with unique `layer_idx` + `layer_types=None` in config work but cost extra VRAM.

2. **Gemma3 manual layer loop doesn't work** — sliding window attention breaks. Use full-model forward with ModuleList swapping only (`model(input_ids, use_cache=False)`).

3. **`save_pretrained` requires deep copy** — shared weight tensors cause errors. Also must set `model.config.layer_types = None` or extend the list to match new layer count.

4. **lm-eval limit parameter:** `--limit 1.0` means 1 sample, not 100%. Use `--limit 0` for full dataset (maps to `None` in our wrapper).

5. **72B model uses ~140-160GB VRAM** in bfloat16. Cannot fit two models simultaneously on one B200 (179GB).

## Models

| Path | Size | Layers | Notes |
|------|------|--------|-------|
| `models/small/Qwen2-7B-Instruct` | 15GB | 28 | Fast iteration |
| `models/full/calme-2.1-qwen2-72b` | 136GB | 80 | Main model (Ng's base) |
| `models/full/RYS-XLarge` | 146GB | 87 | Ng's published duplicated model |
| `models/full/gemma-3-27b-it` | 52GB | 62 | Google, cross-architecture |
| `models/full/Qwen3.5-27B` | 52GB | 64 | Generalization |
| `models/full/Qwen3.5-9B` | 19GB | 32 | Fast generalization |
| `models/full/Qwen3-30B-A3B` | 57GB | 48 | MoE (128 experts, 8 active/tok) |

## Key Results

| Config | Combined | Delta vs Ng |
|--------|----------|-------------|
| Baseline (72B) | 70.52 | — |
| Ng (45,52) @1.0 | 76.76 | — |
| Greedy pair (0,7)+(45,52) | 79.91 | +3.15 |
| Whisper-alpha quad (4 blocks) | 82.58 | +5.82 |
| Per-layer alpha single (45,52) | 82.77 | +6.01 |
| Per-layer alpha triple (grid, 300 evals) | 84.07 | +7.31 |
| Per-layer alpha triple (Bayesian, 60 evals) | 83.97 | +7.21 |
| Gemma3-27B quad (4,5)+(12,13)+(16,17)+(20,21) | 85.58 | N/A |

**Screening:** SBUID_0 (`BLOOD_impact - 6000*rho`) achieves Spearman r=0.515, p=0.008 on 72B.

**lm-eval:** Duplication helps reasoning (IFEval +2.3%, MuSR +1.3%) but hurts knowledge (MATH -6.4%). Sublayer analysis shows FFN re-retrieval corruption is the cause — attention-only duplication may fix this.

## Results Organization

```
results/data/
├── 7b/          # singles, pairs, spectral, blood, dice, adapters, routing
├── 72b/         # 35+ experiment categories (alpha_optimization, deeper_stacking, etc.)
├── gemma3_27b/  # comprehensive, cross_layer, mega_stacking, alpha_stacking
├── qwen35/      # e2e_pipeline, greedy_stacking, alpha_stacking
├── moe/         # basic_duplication, adaptive_routing
├── quantization/ # 4-bit NF4 survival tests
└── sbuid_validation/ # cross-model screening metric
```

All results saved as JSON. Experiment logs in `results/sbatch_*.log`.

## Documentation

- **HISTORY.md** (~1800 lines) — Complete experimental timeline with numbered sections
- **PAPER.md** (~250 lines) — Paper outline with section numbers, data paths, key tables
- **prompts/** — Comprehensive prompts sent to GPT-5.4 Pro for screening metric design and FFN analysis

## GPU Usage Rules

- Account `cis4914` has 8 B200 GPUs. Always leave 2 free for group members.
- Check group usage: `squeue -A cis4914 -o '%u|%t' -h | grep -v jietao | grep R`
- Monitor jobs after submission — check logs within 60-90s for errors.
- Set up recurring checks for long-running experiments.
