# Senior Project Overview

Okay, so this repo ended up a lot bigger than we first planned. What started as "let's poke at TRM and see if we can make it better" grew into eight sub-projects, a web app, a 16 KB paper draft, and one very honest 144 KB lab notebook. This file is the map. If you're looking for a specific experiment, the short blurbs below point you at the right folder, and each folder has its own OVERVIEW.md that goes into more detail.

## The team

Jie Tao, Edward Kempa, Linwei Zhang. A CIS senior project.

## What we were actually trying to figure out

The through-line is recursive reasoning. Most models today spend their compute linearly: you feed in a prompt, you pass through N layers once, you get an answer. We got interested in what happens when you let a small model iterate on its own hidden state, and we pulled at that thread from every angle we could fit into a semester.

Most of the work sits on top of a model family called TRM (Tiny Recursive Model). It's about 7M parameters. What's cool about it is that it gets surprisingly far on ARC-AGI and Sudoku by doing many forward passes through the same layers, with a learned halting signal deciding when it's done. From there we went in a few directions. We changed the loss function. We added a training curriculum. We tried the same idea on a completely different domain (RNA folding). We probed the trained weights to understand what the model actually learned. And eventually we built a small web app that lets a non-researcher train one of these models on their own problem.

If you read only one file after this one, read `RR_TRM/RESULTS.md`. That's the cleanest writeup of the core finding, which is that swapping BCE for a Brier halting loss plus a softmax-over-stablemax and a monotonicity term makes TRM converge about 1.82x faster at inference without hurting accuracy.

## The eight folders, in rough reading order

### `RR_TRM/`
The main experiment. Loss modifications for TRM, trained on Sudoku-Extreme and Maze-Hard. Baseline BCE halting vs. a "BrierHalting" variant that combines Brier loss, a monotonicity regularizer, and softmax instead of stablemax. Best BCE checkpoint hits 86.9% on Sudoku, best Brier hits 87.3%. Also contains the post-training analysis scripts we used for the paper (Raju-Netrapalli error fitting, PCA on latent trajectories, curve fitting for convergence dynamics). Start here to reproduce the headline numbers.

### `RR_Interpretability/`
The follow-up "why does it work" project. Flat directory of probing scripts: attention heads, induction heads, linear and MLP probes on hidden states, spatial propagation through the maze grid, a "can it BFS?" test, and a mechanistic investigation of the Brier halting dynamics. A lot of it is mid-iteration. There was a checkpoint-loading bug partway through that forced a re-run, which is why you'll see `proper_eval.py` and `verify_and_reprobe.py` sitting next to the earlier versions. The `VISUALIZATIONS/` subfolder has an interactive dashboard that renders per-cell softmax probabilities across ACT steps, which is pretty fun to look at.

### `CGAR_TRM/`
A separate methodology we tried alongside the loss work. CGAR is short for Curriculum-Guided Adaptive Recursion. Two ideas glued together: Progressive Depth Curriculum (start training with shallow recursion, gradually increase depth), and Hierarchical Supervision Weighting (weight early reasoning steps more heavily via a 0.7^step decay). The payoff was 1.71x faster training for a 0.63% accuracy drop. Published in JAIR 2025 volume 83, article 27. This folder has the cleanest docs of any subproject because it was the one we finished first.

### `CGAR_RR_TRM/`
CGAR plus RR. This is the combined version. It takes the curriculum schedule from CGAR_TRM and layers the BrierHalting loss family on top, exposing the RR components as toggleable flags in the config (enable_brier_halting, enable_monotonicity, enable_smoothness). The code overlaps with CGAR_TRM by a lot, which is on purpose: it's meant to isolate whether CGAR and RR stack. If you diff the two folders you'll notice CGAR_RR_TRM is missing the polish files like README and the GitHub verification checklists, because it's still an active experiment.

### `RNAformer/`
An upstream project from Freiburg (Joerg K.H. Franke, 2023), vendored in unchanged. Transformer for RNA secondary structure prediction using axial attention. We brought it in as a baseline. The license is Apache 2.0 and we didn't modify any of it, so treat this folder as a reference implementation.

### `RNAformer_TRM/`
This is the interesting one in the RNA direction. Takes RNAformer and bolts on the same three TRM ideas we applied elsewhere: curriculum-guided adaptive recycling, a dual-state recurrent architecture (separate horizontal and vertical hidden states through H and L cycles), and ACT with Brier halting and the monotonicity and smoothness regularizers. The `train_comparison.py` script trains both variants back-to-back so you can compare F1 on the same WandB project. Tests are decent. Inference code isn't finished.

### `DeepPass/`
Our biggest and messiest subproject, and also the one that produced the strongest empirical result. The question here is the opposite of RR_TRM: instead of training recursion from scratch, what if you take a big pretrained LLM and just repeat some of its layers at inference time? This builds on David Ng's "Repeat Your Self" paper. We added spectral screening to prune the 3,241-config search space down to ~20 candidates (162x speedup), greedy multi-block stacking to find complementary layer pairs, and per-layer alpha tuning with Optuna. Final result on Qwen2-72B: +7.31 points over Ng's published number on the combined math + EQ metric, with no training. The folder also contains two or three side experiments that didn't pan out (`psrt/`, `solver/`, `sirt/`), which are documented honestly in `HISTORY.md` as instructive dead ends. About 450+ GPU hours went into this one.

### `TRM_Spinner/`
The product-y piece. A full-stack web app that lets a user describe a reasoning problem in a chat interface, get the problem automatically classified, upload or generate training data, watch the model train on the GPU with live metrics, and download the weights. Next.js 15 frontend, FastAPI worker, Redis for the job queue, Appwrite for persistent user data. Uses authentic TRM training code from RR_TRM and CGAR_TRM (the `make prebuild` step copies model source files across). Local-first, designed for a single RTX 5090 box. Not deployed anywhere public.

## How the folders relate to each other

There are really three tracks running in parallel:

1. **The TRM loss track**: RR_TRM is the training code and main result, RR_Interpretability is the followup mechanistic work, CGAR_TRM and CGAR_RR_TRM are curriculum variants. These all share the same puzzle dataset loader and config scaffolding (`puzzle_dataset.py`, `pretrain.py`, Hydra configs under `config/`).

2. **The RNA track**: RNAformer is the baseline, RNAformer_TRM is the modified version. These are self-contained and don't share code with the puzzle track.

3. **The inference-time compute track**: DeepPass is its own universe. It has its own results, its own paper draft, and its own conclusion, which is that runtime layer duplication on big pretrained LLMs works, and trained recursion from scratch (what we did in RR_TRM) is harder to scale. Read `DeepPass/RESULTS_COMPREHENSIVE.md` for the honest verdict.

4. **The product**: TRM_Spinner wraps the RR_TRM and CGAR_TRM training code in a web UI so somebody without a GPU setup can kick off training.

## A few things to keep in mind when reading the code

The configs live in YAML and get loaded through Hydra. A lot of them are symlinked to paths under `/data/TRM/ablations/configs/`, which is a holdover from our local training box. If you clone this fresh and the symlinks break, check `CGAR_TRM/config/arch/` for the actual targets.

Some of the models save only weights, not optimizer state. There's a FIXME about this in `pretrain.py` line 301 (or thereabouts, depending on the folder). For long runs, back up the full directory, not just `.pth` files.

DeepPass has a painful history with KV caching. If you see scripts named `kv_cache_v1.sh` through `kv_cache_v9.sh`, that's not a joke, we genuinely needed nine iterations. The final workaround is to always pass `use_cache=False` when wrapping duplicated layers. See `DeepPass/CLAUDE.md` for the gory details.

The RR_Interpretability folder looks flat and disorganized on purpose. Scripts named `_v2` are the ones you want to run. The ones without `_v2` are earlier versions kept around so we could diff them. Read `GPT_Experiments_List.md` in that folder if you want the full research plan, though we didn't finish every experiment in there.

## Hardware and setup

Everything in the puzzle track runs on a single RTX 5090, which is what our local box has. The DeepPass experiments need B200-class GPUs (192 GB HBM3e) because we're loading 72B models in bf16, which won't fit on consumer cards. The RNA experiments are smaller and should run on anything with ~16 GB VRAM.

For the puzzle track, dependencies are in `RR_TRM/requirements.txt` (and the same file copied into the other puzzle folders). Python 3.10, PyTorch with CUDA 12.6.

For DeepPass, there's a conda env at `/blue/cis4914/jietao/DeepPass/envs/deeppass` on the HPC cluster. If you're setting up fresh, don't compile flash-attn from source, it uses 40 GB of RAM and will crash the login node. Grab a prebuilt wheel instead.

## One rule

Please don't commit on my behalf. CLAUDE.md at the root says this, and it's true. If you're using Claude Code in this repo, review the diff before anything hits the commit log.

## Where to go next

- Reading the paper: `DeepPass/PAPER.md` for the DeepPass draft, `RR_TRM/RESULTS.md` for the puzzle results.
- Running a training: `RR_TRM/OVERVIEW.md` or `CGAR_TRM/README.md`.
- Understanding what BrierHalting actually does: `RR_Interpretability/OVERVIEW.md` and the `VISUALIZATIONS/` dashboard inside it.
- Getting the web app up: `TRM_Spinner/OVERVIEW.md` and then `make dev`.
- The long story of everything that worked and didn't: `DeepPass/HISTORY.md`. Bring coffee.
