# RNAformer

This folder is an upstream project, vendored in. We didn't write any of it. Copyright Joerg K.H. Franke, University of Freiburg, 2023. Apache 2.0 license. Original GitHub repo link lives in `pyproject.toml` under `automl/RNAformer`.

We brought it in because we wanted a non-puzzle testbed for the TRM ideas, and RNA secondary structure prediction was a good fit. RNAformer is a transformer with a 2D latent space and axial attention that predicts which RNA bases pair with which, without needing multiple sequence alignment as input.

The modified version, where we added TRM-style recursion and halting, lives next door in `../RNAformer_TRM/`. This folder is the unmodified baseline.

## What's here

Top-level files:

```
train_RNAformer.py        # training entry point, PyTorch Lightning + DeepSpeed
infer_RNAformer.py        # inference script, supports LoRA fine-tuning
evaluate_RNAformer.py     # metrics (F1, sensitivity, PPV) across test sets
build_bprna_dataset.py    # pulls the bpRNA dataset from HuggingFace, parses dot-bracket
download_all_datasets.sh
download_all_models.sh
run_evaluation.sh
setup.py                  # package setup
pyproject.toml            # dependencies pinned: torch 2.1.0, pl 2.0.4, loralib 0.1.2, deepspeed 0.9.5
MANIFEST.in
requirements.txt
LICENSE                   # Apache 2.0
README.md                 # upstream README, detailed
```

The training script uses PyTorch Lightning with DeepSpeed for memory efficiency. The four pretrained checkpoints that come down via `download_all_models.sh` are 32M parameters each: a biophysical one, a bpRNA one, and two fine-tuned variants (intra-family, inter-family).

## Subfolders

### `config/`
One file, `default_config.yaml`. Hyperparameters and training settings. You override on the command line or supply your own config.

### `RNAformer/`
The Python package. Four subdirectories:

- `RNAformer/model/` — the actual architecture. `RNAformer.py` (~94 lines) is the main `RiboFormer` class. `RNAformer_stack.py` wraps the stacked transformer blocks. `RNAformer_block.py` is a single block.
- `RNAformer/module/` — the core layers. `Axial_attention.py` is the interesting one (axial attention over the 2D latent). Also `axial_dropout.py`, `embedding.py`, `feed_forward.py`.
- `RNAformer/pl_module/` — PyTorch Lightning glue. `datamodule_rna.py` is the DataModule, `rna_folding_trainer.py` is the LightningModule with train and val steps.
- `RNAformer/utils/` — configuration loading, instantiation helpers, evaluation, logging, folder management, parameter grouping. Plus two subfolders, `handler/` (checkpoint and directory management) and `optim/` (learning rate schedules).

## How to use it

Download the data and models:

```bash
bash download_all_datasets.sh
bash download_all_models.sh
```

Train from scratch:

```bash
python train_RNAformer.py --config config/default_config.yaml
```

Evaluate a checkpoint:

```bash
bash run_evaluation.sh
```

Or more manually:

```bash
python evaluate_RNAformer.py --checkpoint path/to/checkpoint.ckpt --dataset bprna
```

Inference on a single sequence (with optional LoRA):

```bash
python infer_RNAformer.py --sequence GCAUCGAU... --checkpoint path/to/checkpoint.ckpt
```

The upstream README has more detail on the data format and the fine-tuning workflow.

## Why it's here

Three reasons:

1. Baseline for the TRM extension in `../RNAformer_TRM/`. We needed the unmodified version to compare against.
2. Sanity check that our TRM modifications don't just work on Sudoku and mazes. RNA folding is a totally different domain (continuous sequence, structured output, biophysics-flavored) and we wanted to know if recursion helps there too.
3. Reference for how PyTorch Lightning plus DeepSpeed integrates, which we didn't want to reimplement from scratch for RNAformer_TRM.

We didn't change anything in this folder. If you spot a modification, it's a bug, please flag it.

## Gotchas

- The DeepSpeed version is pinned to 0.9.5. Newer DeepSpeed versions have broken the config format we're using, so if you bump it, expect to fix things.
- The LoRA layer insertion in `infer_RNAformer.py` expects specific target modules. If you're adapting this to a different transformer, you'll need to change the target list.
- `build_bprna_dataset.py` reaches out to HuggingFace. If the dataset moves or gets renamed, this will break.
