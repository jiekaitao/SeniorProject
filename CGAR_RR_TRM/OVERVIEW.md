# CGAR_RR_TRM

The combined experiment. Takes the curriculum training schedule from `../CGAR_TRM/` and bolts on the loss modifications from `../RR_TRM/`. This is where we test whether CGAR and RR stack or cannibalize each other.

It's a fair question. Both methods aim at training dynamics. CGAR varies the recursion depth across training. RR changes what the halting loss is actually computing. A priori, they could either reinforce each other (both push the model toward smoother gradient flow) or step on each other (if CGAR's shallow-to-deep schedule confuses the monotonicity regularizer, for example). We wanted to measure which.

The RR modifications are exposed here as config flags, all off by default:

- `enable_brier_halting` — swap BCE for Brier on the halt prediction.
- `enable_monotonicity` — penalize q_probs from decreasing across iterations.
- `enable_smoothness` — L2 penalty on q_probs change between adjacent iterations.

Plus the curriculum setup from CGAR:

- Progressive depth: H_cycles ramps from 1 to 3, L_cycles from 2 to 6 across training stages.
- Hierarchical supervision weighting: `supervision_decay = 0.7` by default.

You can toggle any subset.

## What's in the top level

Everything looks familiar if you've read `../CGAR_TRM/OVERVIEW.md`:

```
pretrain.py                              # base training loop
pretrain_cgar.py                         # CGAR wrapper that updates curriculum depth each step
puzzle_dataset.py                        # same iterable dataset
evaluate_checkpoints.py                  # checkpoint evaluation
evaluate_baseline_checkpoints.py         # baseline evaluation
test_eval_single_batch.py                # debug harness
create_comprehensive_visualizations.py   # paper figure generator
CITATION.bib                             # same JAIR citation as CGAR_TRM
LICENSE                                  # Apache 2.0
.gitignore
requirements.txt
```

Notable **missing** files compared to `../CGAR_TRM/`:

- No `README.md`.
- No `RESULTS_VERIFICATION.md`.
- No `LORA_TRAINING_GUIDE.md`.
- No `FINAL_GITHUB_VERIFICATION.md`.

That's because this folder is the active experiment. CGAR_TRM was polished for a publication. CGAR_RR_TRM is still moving and we didn't want to ship it yet.

## Subfolders

### `config/`
The important difference vs. CGAR_TRM. `config/arch/trm_cgar.yaml` here references `losses_cgar@ACTLossHead_CGAR` with the RR flags listed. In CGAR_TRM the same file references plain `losses@ACTLossHead` with no flags.

`config/arch/` also has the same family of architecture configs (trm, hrm, transformers_baseline, trm_hier6, trm_singlez), plus the ablation variants. Some entries are symlinked to external `/data/TRM/ablations/configs/` paths.

### `models/`
`models/recursive_reasoning/` has six variants, same as CGAR_TRM:

- `trm.py` — base, with `prev_q` carried in state so the monotonicity and smoothness regularizers have something to compare against.
- `trm_cgar.py` — curriculum wrapper with `set_curriculum_depth()`.
- `trm_hier6.py` — six-layer hierarchical variant.
- `trm_singlez.py` — single hidden state (no split y/z).
- `hrm.py` — hierarchical reasoning model baseline.
- `transformers_baseline.py` — plain transformer.

At the top of `models/`:

- `losses.py` — base `ACTLossHead`. This is where the three RR toggles live. Brier halting switches the halt prediction from BCE to MSE. Monotonicity reg penalizes any decrease in q_probs across iterations. Smoothness reg is an L2 penalty on the q_probs delta.
- `losses_cgar.py` — `ACTLossHead_CGAR` that adds hierarchical supervision weighting on top. Early steps get roughly 7x higher loss weight than later ones with the default 0.7 decay.
- `layers.py`, `ema.py`, `sparse_embedding.py`, `common.py` — the usual supporting modules.
- A LoRA adapter module lives here too, though it's wired but not heavily used in the ablation runs.

### `ablations/`
`ablations/scripts/` has `monitor_progress.py` and `monitor_progress_fixed.py`. Same as CGAR_TRM. The "fixed" one is the one you want.

### `evaluators/`
`arc.py`, the ARC-AGI scoring logic.

### `utils/`
`functions.py`, the dynamic model loader.

### `visualizations/`
`create_visualizations.py` and a small README.

### `assets/`
TRM architecture PNG and pseudocode PNG.

## Training

Same general shape as `../CGAR_TRM/`:

```bash
pip install -r requirements.txt
python pretrain_cgar.py --config config/arch/trm_cgar.yaml --epochs 50000 --batch_size 256 --lr 0.001
```

To turn on the RR features, edit `trm_cgar.yaml` and set any combination of:

```yaml
loss:
  enable_brier_halting: true
  enable_monotonicity: true
  enable_smoothness: false     # we found this one was less impactful
  supervision_decay: 0.7
```

The curriculum progression is controlled in `pretrain_cgar.py` where `train_batch` gets overridden to call `set_curriculum_depth(progress)`. The formula is linear: first 30% of steps stay at shallow depth, ramp up through the middle, finish at full depth.

Evaluate with `evaluate_checkpoints.py`, same as the sibling folders.

## What's actually different from CGAR_TRM in code

If you're comparing the two folders file by file:

- `config/arch/trm_cgar.yaml` is the meaningful difference. CGAR_RR_TRM points at `losses_cgar@ACTLossHead_CGAR` with RR flags available.
- `pretrain.py` differs at byte 356 (a comment).
- `trm.py` differs at byte 796 (a comment, and the `prev_q` tracking).
- Everything else matches.

So 95% of the code is shared. CGAR_RR_TRM exists as its own folder because keeping a clean diff at the top level made it easier to run ablations without risking accidental cross-contamination.

## Quirks

- No README. That's intentional. This folder is active work.
- `pretrain.py` line 719 has a FIXME about only saving model weights, not optimizer state.
- The `ablations/` folder is thin here. The heavier orchestration logs live on the training box, not in the repo.
- If you look for LoRA code, it's wired in but we haven't done the full LoRA + CGAR + RR three-way ablation. That's on the wishlist.
