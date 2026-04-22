# CGAR_TRM

CGAR stands for Curriculum-Guided Adaptive Recursion. This is a separate methodology from the RR loss work in `RR_TRM/`. Where RR changes what loss the model is optimizing, CGAR changes how the training progresses over time.

It's two ideas glued together:

**Progressive Depth Curriculum (PDC)**. Start training with shallow recursion (H=1, L=2 cycles). For the first 30% of training, stay shallow. Ramp the depth up during the middle 30%. Finish with full depth (H=3, L=6) for the final 40%. The idea is that shallow recursion converges quickly and cheaply on easy patterns, and you only pay for full depth once the model has something to refine.

**Hierarchical Supervision Weighting (HSW)**. Apply exponential decay to the supervision signal across recursion steps. Step 0 gets weight 1.0, step 1 gets 0.7, step 2 gets 0.49, and so on. Early reasoning steps matter more, later ones get regularized. This helps training stability.

The payoff: 1.71x faster training on Sudoku-Extreme (6.38 hours vs 10.93 hours on an A100), for a 0.63% accuracy drop (86.02% vs 86.65%). The paper was published in the Journal of Artificial Intelligence Research, JAIR 2025, volume 83, article 27. `CITATION.bib` has the BibTeX if you want to cite it.

## What's in the top level

```
pretrain.py                              # base training loop (shared with RR_TRM variants)
pretrain_cgar.py                         # CGAR-specific wrapper that calls set_curriculum_depth() each step
puzzle_dataset.py                        # same iterable dataset as in RR_TRM
evaluate_checkpoints.py                  # loads CGAR checkpoints, scores them
evaluate_baseline_checkpoints.py         # parallel for baseline TRM
test_eval_single_batch.py                # minimal debug harness
create_comprehensive_visualizations.py   # 34 KB of plotting code for training curves, ablation plots
README.md                                # public-facing documentation (from the GitHub release)
RESULTS_VERIFICATION.md                  # cross-check that paper Tables 1/2/4 match what's in the README
FINAL_GITHUB_VERIFICATION.md             # pre-publication checklist (branding, .gitignore, ~3.8 GB excluded)
LORA_TRAINING_GUIDE.md                   # how to combine CGAR with LoRA for even faster training
CITATION.bib                             # JAIR + arXiv BibTeX
requirements.txt                         # torch, pydantic, hydra-core, wandb, coolname, argdantic
```

This folder is the cleanest of the TRM variants because we finished it first and actually shipped the writeup. `FINAL_GITHUB_VERIFICATION.md` walks through what got excluded from the release (docs, checkpoints, wandb artifacts, experiment outputs, data) and why.

The one odd thing here is that `pretrain.py` and `pretrain_cgar.py` both exist. `pretrain.py` is the base training loop, effectively shared with the RR_TRM siblings. `pretrain_cgar.py` imports from it and overrides `train_batch` to call `model.set_curriculum_depth(progress)` at each step, where `progress = current_step / total_steps`. That's the whole CGAR orchestration in 7 KB.

## Subfolders

### `config/`
Hyperparameters and architectures.

`config/arch/` has a pile of YAML files:
- `trm_cgar.yaml` — the main CGAR config.
- `ablation_baseline.yaml` — no CGAR, plain TRM.
- `ablation_curriculum_only.yaml` — PDC on, HSW off.
- `ablation_hierarchical_only.yaml` — HSW on, PDC off.
- `ablation_decay_0.5.yaml` through `ablation_decay_0.9.yaml` — sweep of the HSW decay factor.
- `trm.yaml`, `hrm.yaml`, `trm_hier6.yaml`, `trm_singlez.yaml`, `transformers_baseline.yaml` — baseline models for comparison.

Some of these are symlinks into `/data/TRM/ablations/configs/` on our local box. If you've cloned this fresh and those paths don't exist, look for the real YAMLs in the same directory.

### `models/`
`models/recursive_reasoning/` has the actual model classes:

- `trm_cgar.py` — the CGAR variant with `set_curriculum_depth()` that adjusts H_cycles and L_cycles on the inner layer.
- `trm.py` — base TRM, same as in RR_TRM.
- `hrm.py` — the hierarchical reasoning model baseline.
- `trm_hier6.py`, `trm_singlez.py` — alternative TRM variants we wired up for comparison but didn't highlight in the paper.
- `transformers_baseline.py` — plain transformer as a sanity-check baseline.

Directly under `models/`:

- `losses_cgar.py` — `ACTLossHead_CGAR` that applies the HSW decay (0.7^step) to supervision signals.
- `losses.py` — the base loss with the standard ACTLossHead. `losses_cgar.py` imports from here.

### `ablations/`
`ablations/scripts/` has two progress-monitor scripts, `monitor_progress.py` and `monitor_progress_fixed.py`. The "fixed" version is the one that actually works. These watch training runs across ablations and flag when convergence stalls.

### `evaluators/`
One file: `arc.py` (6.7 KB). Same ARC evaluator as RR_TRM.

### `utils/`
One file: `functions.py`. Dynamic model loader by string name.

### `visualizations/`
`create_visualizations.py` plus a `README.md` with visualization guidance. The big top-level `create_comprehensive_visualizations.py` (34 KB) is the production version that makes all the paper figures.

### `assets/`
Two PNGs. TRM architecture diagram, TRM pseudocode.

## How to train

Install deps:

```bash
pip install -r requirements.txt
```

Train CGAR:

```bash
python pretrain_cgar.py --config config/arch/trm_cgar.yaml --epochs 50000 --batch_size 256 --lr 0.001
```

Evaluate:

```bash
python evaluate_checkpoints.py --checkpoint checkpoints/cgar_50k.pth --dataset sudoku_extreme
```

The README has the full end-to-end commands with the exact flags. For a sanity check on a single batch (useful when debugging), use `test_eval_single_batch.py`.

Checkpoints load with a small quirk: `torch.compile` prefixes parameter names with `_orig_mod.`. The evaluation scripts strip this prefix automatically when loading, but if you're writing your own loader, keep it in mind.

## LoRA training

`LORA_TRAINING_GUIDE.md` walks through combining CGAR with Low-Rank Adaptation. The short version: freeze the base weights, add rank-r decomposition matrices to the attention QKV projections, attention outputs, and MLP layers. With rank=32 (the recommended value) you get 88% parameter reduction (~120K trainable from ~1M base) and 2-3x additional speedup for less than 1% accuracy drop.

The combination pitch is "LoRA for parameter reduction, CGAR for better learning dynamics." They're orthogonal axes, so they stack.

Rank tradeoffs in the guide:
- rank=16 → 94% parameter reduction, minor accuracy hit.
- rank=32 → 88% reduction, cleanest speedup (recommended).
- rank=64 → 76% reduction, better accuracy recovery.

LoRA uses the same loss and evaluation pipeline, so it's effectively a drop-in replacement for the full parameter path.

## Relationship to CGAR_RR_TRM

CGAR_TRM and `../CGAR_RR_TRM/` are parallel implementations of the same CGAR methodology applied to two different base architectures. If you diff the two folders:

- `pretrain.py` differs at byte 356 (a comment, I think).
- `trm.py` differs at byte 796 (also comments).
- The core CGAR code (`trm_cgar.py`, `losses_cgar.py`, `pretrain_cgar.py`) is identical.
- The configs are different, because CGAR_RR_TRM adds the RR loss flags.

CGAR_TRM targets standard TRM. CGAR_RR_TRM targets the RR-modified TRM. They're not forks, they're not diverging experiments. They're the same recipe applied to two different cakes. Treat them as a matched pair.

## Quirks

- `pretrain.py` has one FIXME noting that only model weights are saved, not optimizer state.
- Some `config/arch/` entries are symlinks to a local `/data/TRM/ablations/configs/` path. Check they resolve before training.
- The six model variants in `models/recursive_reasoning/` include some (`trm_hier6.py`, `trm_singlez.py`) that don't show up in the final paper. They're kept for comparison but aren't documented.
- No abandoned experiments, no half-finished features. This folder is the polished one.
