# RR_Interpretability

The followup project to RR_TRM. After we got BrierHalting to converge faster than the BCE baseline, we wanted to know why. This folder is a collection of probing scripts that poke at the trained models from every angle we could think of.

Fair warning: this folder is messy. It was never meant to be polished. Scripts named `_v2` are the ones you actually want to run. The ones without a suffix are earlier versions we kept around so we could diff them. A checkpoint-loading bug partway through the project forced a bunch of re-runs, which is also why you'll see `verify_and_reprobe.py` and `proper_eval.py` sitting next to the scripts they're "verifying" and "properly" evaluating.

If you want the finished story, open `VISUALIZATIONS/index.html` in a browser and look at the dashboard. It shows per-cell softmax probabilities evolving across all 16 ACT steps for each of the 5 ablations. That's the most intuitive way to see what's different.

## What we were testing

The core hypothesis is that BrierHalting isn't just arbitrary noise, it's changing the mechanism. We tested that from a bunch of directions:

- **Can the hidden states predict spatial labels?** Linear and MLP probes for wall/open/reachable on the maze task.
- **Does the model do BFS?** We probed it against a ground-truth BFS solver.
- **What are the attention heads doing?** Per-iteration attention pattern extraction, plus induction head detection.
- **How does confidence flow through iterations?** Tracked per-cell softmax across all 48 internal iterations (16 ACT steps times 3 H-cycles).
- **Does the model propagate information across the maze grid spatially?** Cell-level accuracy tracking by distance from start/goal.
- **Is the representation geometry different between BCE and BrierHalting?** Cosine similarity, Fisher discriminant, ECE calibration.

The short answer is that token accuracy is nearly identical across all 5 ablations (~97%). The differences are all in *how* the model gets there, not *whether* it does.

## The blueprint

`GPT_Experiments_List.md` in this folder is the research plan. It's ~47 KB and very dense. The main thing to know is that it was written early, it defines 13 formal hypotheses (H1 through H13) about TRM as a dynamical system, and it lays out a 4-week experiment plan. We didn't do all of it. The scripts in this folder correspond to maybe 60% of that plan. The rest is still on the wishlist.

## Top-level scripts, grouped by what they do

### Attention probing
- `attention_heads.py` — analyzes attention patterns per iteration, looking for whether heads switch roles (like from previous-token to induction) across H-cycles.
- `extract_attention.py` — monkey-patches PyTorch attention to capture weight matrices during a forward pass. This is what produces `attention_patterns.json` (which is a 175 KB JSON, so it's not the file you open casually).
- `induction_heads.py` — dedicated induction-head detector that loads the correct final checkpoint (after the bug was found) and manually computes attention matrices to check for spatial propagation and wall detection patterns.

### BFS analysis
- `can_it_bfs.py` — tests whether the model rediscovers BFS on text-encoded grids. Trains both linear and 2-layer MLP probes on the hidden states against a ground-truth BFS oracle.
- `can_it_bfs_results.json` — the results. MLP probes get ~77% cell accuracy with K=1 solver setup, which suggests partial BFS but not clean.
- `viz_soft_bfs.py` — generates the per-cell softmax trajectories that power the web dashboard.

### Linear and MLP probes
- `linear_probe.py` — trains simple linear probes on frozen z_H representations at each H-cycle to predict maze cell labels (wall, open, reachable).
- `linear_probe_results.json` — per-H-cycle probe accuracies across all 5 ablations. The main takeaway is that label 5 (unreachable cells far from start) is consistently near 0%. The model really doesn't build a representation of unreachability.

### Spatial propagation
- `spatial_propagation.py` and `spatial_propagation_v2.py` — track per-position accuracy on the maze grid and how it evolves across H-cycles. V2 has extended metrics.
- `spatial_propagation.json` and `spatial_propagation_v2.json` — the raw outputs.

### Evaluation and verification
- `proper_eval.py` — runs the FULL ACT loop (all 16 × 3 = 48 internal iterations) and probes z_H at multiple steps. This fixes the earlier bug where only 3 H-cycles were being evaluated.
- `proper_eval_results.json` — the headline file. Exact token accuracy (~97%), per-label accuracy, per-step LM and probe accuracy, reachability metrics, all broken out by ablation.
- `verify_and_reprobe.py` — loads the correct final checkpoint (`step_19530`, not the half-trained `step_9765` we were accidentally using before) and re-runs the linear and MLP probes.
- `verify_reprobe_results.json` — the re-verified probe accuracies.
- `testset_results.json` — a small file, mostly summary-only, maybe incomplete.

### Investigation
- `investigate_brier_mechanism.py` — the most specific script. Compares FULL_COMBO against BASELINE across confidence evolution, prediction stability, representation geometry (cosine similarity across iterations), separability (Fisher discriminant), ECE calibration, and per-cell convergence speed. This is where most of the mechanistic conclusions come from.
- `full_interp_corrected.py` — consolidates attention patterns, displacement, PCA, spatial propagation, and cosine similarity into one script with proper checkpoint loading.

### Orchestration
- `run_experiments.py` — iteration-resolved logit lens, Jacobian spectral analysis, state trajectory PCA, temporal activation analysis. Aims to cover several hypotheses at once.
- `run_v2.py` — cleaner version of the above.

### JSON summaries
- `experiment_results.json` — contains an error message about tensor shape expansion. Aborted run, ignore.
- `experiment_results_v2.json` — the rerun that worked.
- `corrected_interp_results.json` — aggregated metrics after the bug fix.
- `reduced_mlp_perstep.json` — summary of MLP probe accuracies per step.

## Subfolders

### `VISUALIZATIONS/`
The presentation layer. Contains:

- `index.html` — interactive dashboard that pulls the JSON files and renders maze grids with softmax probabilities evolving across ACT steps. This is the "show someone the result" artifact.
- `maze_data.js` — static maze data for the dashboard.
- `loss_landscape_analysis.md` — a writeup on why stablemax (which BASELINE uses) has a discontinuous second derivative at zero and why softmax (which FULL_COMBO uses) gives you C-infinity smoothness. This is the closest thing to a clean mechanistic story we have.
- 5 ablation-specific BFS soft reachability trajectory JSONs (~2 MB each).
- A few aggregated JSONs: `brier_mechanism_investigation.json`, `induction_head_analysis.json`, `five_way_ablation_summary.json`.
- A duplicate of `proper_eval_results.json` for easy access from the browser.

## What we found (short version)

1. Both models (BCE and BrierHalting) achieve about 97% token accuracy and 25% exact maze accuracy. Final performance is basically tied.
2. The probe accuracies for cell labels are in the 87-88% range for both linear and MLP probes, across all ablations. So the representations are "there" in both models, they're just being used differently.
3. The representation geometry differs. BCE sits in a nearly 1D subspace (PC1 around 84.5% of variance). BrierHalting spreads into 2D (PC1 ~52%, PC2 ~39%).
4. The loss landscape matters. Stablemax isn't twice-differentiable at zero. Softmax is smooth everywhere. For an iterative model that keeps revisiting its own outputs, this seems to give the BrierHalting variant a cleaner gradient path during early training.
5. Perturbation rho (how much a small input change amplifies) is > 1 for both models (BCE ~2.03, Brier ~2.46), but displacement rho (how much the hidden state moves between iterations) is < 1 for both (around 0.84-0.88). So the models contract in state space even though they'd be locally unstable under small input perturbations.

Full details are in `proper_eval_results.json`, `VISUALIZATIONS/loss_landscape_analysis.md`, and the RR_TRM `results/full_analysis.md`.

## Reading order

If you're new to this folder:

1. `VISUALIZATIONS/index.html` in a browser.
2. `VISUALIZATIONS/loss_landscape_analysis.md`.
3. `GPT_Experiments_List.md` (only if you want the theory).
4. `proper_eval.py` and `proper_eval_results.json` for the headline numbers.
5. `investigate_brier_mechanism.py` if you want the mechanistic comparison.

If you want to run anything, start with `verify_and_reprobe.py`, because it loads the correct checkpoint and the probes are fast. Most of the other scripts expect similar checkpoint paths, so once that one runs you're good.
