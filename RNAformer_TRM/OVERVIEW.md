# RNAformer_TRM

The RNAformer-meets-TRM experiment. Takes the vendored RNAformer from `../RNAformer/` and adds three modifications we'd been testing on Sudoku and maze puzzles. The goal was to check whether the TRM ideas generalize outside the discrete-grid puzzle regime.

The three modifications are rolled out in phases. Phase 1 and Phase 2 are architectural. Phase 3 is loss-side.

**Phase 1: CGAR for RNA.** Curriculum-Guided Adaptive Recycling. RNAformer already has an input recycling mechanism where you can pass the latent back through the stack multiple times. CGAR adds a schedule: start with fewer cycles during early training, increase to the full cycle count late. Same core idea as CGAR on TRM, but using RNAformer's recycling knob instead of TRM's H and L cycles.

**Phase 2: Dual-state recurrent architecture.** Maintain separate horizontal and vertical hidden states through H and L cycles of distinct recurrent passes. This was inspired by the split-state design in TRM, adapted for the 2D pair-representation that RNAformer uses.

**Phase 3: ACT with halting.** Add an adaptive computation time head that predicts when the model is confident enough to stop cycling. Comes with three accompanying regularizers: Brier loss on the halting prediction, a monotonicity constraint that penalizes q_probs from decreasing across cycles, and a smoothness regularizer on q_probs change.

All three phases run concurrently in the TRM variant. The comparison script trains both variants back-to-back.

## Top level

```
train_comparison.py   # 13.4 KB, main entry point, --variant baseline|trm
bench.py              # microbenchmark comparing 5 configs for fwd+bwd latency
run_comparison.sh     # sequential baseline-then-TRM training
run_trm_only.sh       # TRM variant only (reduced batch because dual-state is heavier)
__init__.py           # empty
```

`train_comparison.py` is the one you run. It instantiates the right trainer (either `RNATRMTrainer` or the baseline `RNAFoldingTrainer` from the parent RNAformer package) based on the `--variant` flag, wraps the optimizer with a picklable CosineWarmupLambda scheduler, and logs to a shared WandB project so you can compare F1 curves side by side.

`bench.py` is just for timing. Five configs (baseline, cycling-only, CGAR, dual-state, dual-state+ACT), 20 iterations each, batch size 4, length 32. Outputs relative fwd+bwd latencies and counts the RNAformer stack invocations per iteration. Useful for sanity-checking that the cycling math is doing what you think it's doing.

## Subfolders

### `losses/`
One file: `rna_act_loss.py`. Implements `RNAACTLoss`, which combines the contact BCE (the base RNAformer loss for base-pair prediction) with the halting loss (Brier or BCE), plus monotonicity and smoothness regularization, plus supervision decay across cycles. This is the Phase 3 piece.

### `model/`
Three files:

- `RNAformer_TRM.py` — the main `RiboFormerTRM` class. Manages the CGAR curriculum, the dual-state LayerNorms, the ACT q_head, the cycling initialization, and the forward logic. This is where all three phases live.
- `RNAformer_stack.py` — sequential layer stack wrapping the blocks, with a final LayerNorm.
- `RNAformer_block.py` — single transformer block with axial attention, dropout, and either a conv or dense feed-forward.

### `module/`
Reusable components, mostly lifted from RNAformer with minor adjustments:

- `axial_attention.py` — `AxialAttention` and `TriangleAttention`. Row-wise and column-wise self-attention, with optional rotary embeddings.
- `embedding.py` — `EmbedSequence2Matrix`, converts a sequence token stream to a pair representation (the 2D latent).
- `feed_forward.py` — `FeedForward` (dense) and `ConvFeedForward` (1D conv bottleneck, which is a little cheaper).
- `axial_dropout.py` — positional dropout for structured sequences.

### `trainer/`
One file: `rna_trm_trainer.py`. A PyTorch Lightning module that orchestrates training steps, ACT loss computation, curriculum progression, and validation metrics.

### `tests/`
Surprisingly thorough tests for a research project. Six files:

- `test_phase1_cgar.py` — CGAR curriculum depth staging (3-stage progression), input reinjection output differences.
- `test_phase2_dual_state.py` — dual-state forward pass, state recycling correctness.
- `test_phase3_act_losses.py` — ACT q_head outputs, Brier halting, monotonicity and smoothness.
- `test_backward_compat.py` — checks the TRM variant doesn't silently break the baseline RNAformer behavior.
- `test_integration.py` — end-to-end forward and backward pass.
- `conftest.py` — shared pytest fixtures (tiny config, synthetic batches, model factory).

If you're modifying anything in `model/` or `losses/`, run these first. They're fast.

### `utils/`
Thin re-export layer. Imports from the parent RNAformer package so we can refer to everything as `rnaformer_trm.utils.x` instead of reaching across folders.

## Relationship to ../RNAformer/

Not standalone. This folder imports a lot from the parent:

- `RNAformer.utils.configuration.Config` — config loader.
- `RNAformer.pl_module.datamodule_rna.DataModuleRNA` — RNA data loading.
- `RNAformer.utils.group_parameters` — optimizer parameter grouping.
- `RNAformer.utils.optim.lr_schedule` — learning rate schedule primitives.
- `RNAformer.pl_module.rna_folding_trainer.RNAFoldingTrainer` — the baseline trainer (used for comparison runs).

So you need the `../RNAformer/` package installed or on the Python path for anything here to work. The tests include a backward compatibility check that imports the original `RiboFormer` and confirms its behavior hasn't regressed.

## How to run

Install the parent RNAformer first (see `../RNAformer/OVERVIEW.md`). Then:

**Run the side-by-side comparison:**

```bash
bash run_comparison.sh
```

This trains baseline first (batch=2, accumulate=8), then the TRM variant (batch=1, accumulate=16, because dual-state roughly doubles memory). Both configs use identical hyperparameters otherwise: dim=128, 4 heads, 4 layers, cycling=6, 50 epochs. Effective batch size is 16 for both. Logs go to the WandB project `RNAformer-TRM-comparison`.

**Run TRM variant only:**

```bash
bash run_trm_only.sh
```

**Microbenchmark:**

```bash
python bench.py
```

## Tests

```bash
pytest tests/
```

All six test files should pass. If one fails, figure out why before training, because training takes hours.

## Gaps

- No inference script. The training code is there, but we haven't written a clean inference path yet. The baseline RNAformer's `infer_RNAformer.py` won't work for the TRM variant directly because of the dual-state forward signature.
- No checkpoints in the repo. You have to train from scratch.
- Brier halting and the monotonicity and smoothness regularizers are toggleable flags but we haven't done a full ablation sweep on them yet. Defaults work, but there's probably headroom.
- Dual-state memory overhead is noted in comments as a known issue. Batch size takes a hit. Probably worth revisiting whether we can share some of the horizontal/vertical parameters to recover the memory.
