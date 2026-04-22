# RR_TRM

The main experiment folder. This is where we live-trained TRM with different halting losses and then beat it with analysis scripts until it told us what it was doing.

If you want the short version: swapping out the BCE halting loss for a Brier-based one, plus adding a monotonicity term on the halt probability, plus using softmax instead of stablemax for the token loss, converges the model noticeably faster at inference time. Best BCE checkpoint lands at 86.9% exact accuracy on Sudoku-Extreme. Best BrierHalting checkpoint hits 87.3% and gets there with fewer training epochs. The paper's in `results/full_analysis.md`.

## What's in the top level

```
pretrain.py                  # the training loop
puzzle_dataset.py            # dataloader (ARC, Sudoku, Maze)
requirements.txt             # torch, pydantic, hydra-core, wandb, coolname, argdantic
specific_requirements.txt    # pinned versions for reproducibility (CUDA 12.6, py3.10)
README.md                    # installation + quickstart commands
RESULTS.md                   # the 5-way ablation table with training metrics
LICENSE                      # Apache 2.0
```

`pretrain.py` is the orchestrator. It reads Hydra configs, sets up distributed training, instantiates a model, runs the training loop, and fires off periodic evals. It's about 25 KB and has one FIXME near the top of the save function because it only checkpoints model weights (not optimizer state). If you want to resume a run exactly, back up the whole directory, not just the `.pth`.

`puzzle_dataset.py` is an IterableDataset that streams puzzle batches with group-based shuffling. The group logic matters because Sudoku-Extreme has multiple puzzles that share a solution structure, and you don't want the same batch to pick up several of those.

## Subfolders

### `config/`
9 YAML files at the top level, plus an `arch/` subdirectory with 5 more. The top-level ones are the experiment configs: baseline, brier_only, mono_only, softmax_only, full_combo, two maze configs (BCE and Brier), a reduced MLP ablation, and the base pretrain config. The `arch/` subdirectory defines 5 architecture variants: TRM (the main one), HRM (the predecessor paper's model we compared against), transformers_baseline (plain transformer), TRM_hier6 (a 6-layer hierarchical variant), and TRM_singlez (single latent instead of split y/z).

The ablation configs are named after what they isolate. `cfg_ablation_baseline.yaml` turns off all three RR modifications. `cfg_ablation_brier_only.yaml` turns on just the Brier halting loss. And so on. The `full_combo` one turns all three on, which is what we call BrierHalting in the paper.

### `models/`
Five recursive reasoning implementations live in `models/recursive_reasoning/` (confusingly nested, but that's how it is): `trm.py`, `hrm.py`, `trm_hier6.py`, `trm_singlez.py`, `transformers_baseline.py`. Each is in the 295-340 line range.

Alongside those, a handful of support modules at the top of `models/`:

- `losses.py` (6.6 KB) — the `ACTLossHead` class with the three RR toggles (Brier halting, monotonicity, smoothness). This is the heart of the whole project.
- `layers.py` (6.3 KB) — transformer block definitions used by the models above.
- `sparse_embedding.py` (4.4 KB) — distributed sign-SGD embeddings for the large ARC vocabulary.
- `ema.py` (1.2 KB) — exponential moving average for checkpointing.
- `common.py` (1.2 KB) — small shared utilities.

The two variants `trm_hier6.py` and `trm_singlez.py` don't have matching config files or any entry in RESULTS.md. They were comparison baselines that we wired up but never ended up using in the final writeup. Feel free to ignore them.

### `dataset/`
Four dataset builders:

- `build_arc_dataset.py` (11.4 KB) — handles both ARC-AGI-1 and ARC-AGI-2 splits.
- `build_maze_dataset.py` (4.5 KB) — generates 30x30 hard mazes.
- `build_sudoku_dataset.py` (5.8 KB) — Sudoku-Extreme.
- `common.py` (1.4 KB) — the metadata dataclasses shared across builders.

Running any of these produces a data directory with pre-tokenized tensors and a metadata JSON. The training script reads that directory.

### `experiments/`
Post-hoc analysis scripts. This is where a lot of the paper came from.

- `analyze_raju_netrapalli.py` (27.7 KB) — fits the Raju-Netrapalli accuracy law to the model's per-step accuracy curve. Spoiler: it doesn't fit TRM well, because TRM's accuracy curve looks more like a step function than a smooth exponential. The paper talks about this.
- `fit_accuracy_curves.py` (7.4 KB) — tries three curve families (logistic, log-normal, exponential saturation) and compares AIC/BIC. Exp saturation wins. BrierHalting's convergence rate is 1.82x faster than BCE's.
- `latent_pca_analysis.py` (12.6 KB) — runs PCA on the hidden state trajectories through the recursion. The main finding is that BCE lives in a roughly 1D subspace (PC1 explains 84.5% of variance), but BrierHalting uses 2D (PC1 = 52.4%, PC2 = 39.2%).
- `compare_pca.py` (6.3 KB) — plots the two PCA trajectories side by side.

### `evaluators/`
One file, `arc.py` (6.7 KB). Implements the ARC-AGI exact match evaluator with the output-grid comparison logic. Used during training for periodic eval and by the post-training analysis scripts.

### `utils/`
One tiny file, `functions.py` (516 bytes). Dynamic model loader so the config can specify a model class by string and have it resolved at runtime.

### `ablation_logs/`
Four SLURM output files totaling about 7.8 MB. Mostly stderr. One of them is 6.6 MB and is the full stderr from a run that had warnings about deprecated PyTorch APIs. Kept around in case we ever want to dig in, but you can ignore it day-to-day.

### `results/`
`full_analysis.md` (21.9 KB) is the real artifact here. Full writeup of the ablation study, the Raju-Netrapalli analysis, the curve fitting, and the PCA geometry. If you only read one file in this folder, make it that one.

### `assets/`
Two PNGs: the TRM architecture diagram and the pseudocode figure. We used these in slides.

## How to reproduce

Install:

```bash
pip install -r specific_requirements.txt
```

Build the dataset (pick one):

```bash
python dataset/build_sudoku_dataset.py --output_dir data/sudoku_extreme
python dataset/build_maze_dataset.py   --output_dir data/maze_hard
python dataset/build_arc_dataset.py    --output_dir data/arc_agi
```

Train the BrierHalting variant on Sudoku:

```bash
python pretrain.py arch=trm data_paths=[data/sudoku_extreme] epochs=50000 eval_interval=5000 \
    loss.enable_brier_halting=true loss.enable_monotonicity=true loss.use_softmax=true
```

The README has a fuller version with the exact flags we used. For distributed training, swap `python` for `torchrun --nproc_per_node=N`.

Evaluate a checkpoint:

```bash
python experiments/analyze_raju_netrapalli.py --checkpoint checkpoints/step_45298.pth
python experiments/latent_pca_analysis.py     --checkpoint checkpoints/step_45298.pth
python experiments/fit_accuracy_curves.py     --checkpoint checkpoints/step_45298.pth
```

The best checkpoints we use in the paper are step_168696 (BCE, epoch 21600) and step_45298 (BrierHalting, epoch 5800).

## Relationship to sibling folders

RR_TRM is the parent branch of the puzzle track. `CGAR_RR_TRM` is essentially this codebase with CGAR's curriculum wrapper glued on top, which is why the two folders look almost identical. `RR_Interpretability` reads RR_TRM checkpoints (via hardcoded paths to `step_19530` and similar) and runs mechanistic probes on them.

There are no explicit imports across the folders. Each one has its own copy of `pretrain.py`, `puzzle_dataset.py`, and the model files. That was a deliberate choice so we could diverge configs without breaking anything, but it does mean any change to the base model needs to be applied by hand in up to three places.

## Quirks to know about

- The `cfg_ablation_reduced_mlp.yaml` config is referenced by a SLURM log but we never wrote the analysis for it. It's in an indeterminate state.
- The `pretrain.py` FIXME around saving: only weights are persisted, so resuming exactly from a crashed run isn't fully supported.
- The symlinked configs in `config/arch/` (in some folders) point at `/data/TRM/ablations/configs/`, which is a path on our local box. If you're cloning fresh, make sure they're intact or point them at the YAMLs directly.
