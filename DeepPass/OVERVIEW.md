# DeepPass

DeepPass is the biggest thing in this repo, both in code volume and in GPU hours. About 450+ GPU hours went into it. It's also the project that produced the cleanest empirical result.

## The question

What if you take a big pretrained LLM and, at inference time, just repeat some of its transformer layers? Not train on them, not fine-tune, just send the same token stream through a block twice in sequence. Does the model get smarter?

This isn't our original idea. David Ng wrote a paper called "Repeat Your Self" (RYS) that showed this works on a Qwen2-72B model, hitting 77.76 on a combined math + EQ-bench metric, up from the baseline. We took that as a starting point and asked:

1. Which blocks should you repeat? (Ng did a brute-force search. 3,241 configs. Slow.)
2. Can you do better by repeating multiple blocks together?
3. If you weight the repeated pass by alpha (so the output is `(1-a)*original + a*repeated`), what's the right alpha per block? Per layer? Per sublayer?
4. Does this generalize to other architectures? Llama, Gemma, Mistral, MoE variants?

## The answer (short version)

Yes to all four. Our final number on Qwen2-72B is **84.07** on the same combined metric, up from Ng's 77.76. That's +7.31 points. No training.

The pipeline to get there:

1. **Spectral screening (SBUID).** A cheap metric we built that predicts which blocks will benefit from duplication, using BLOOD impact minus a displacement rho term. Reduces the search space from 3,241 configs to about 20 candidates. ~162x faster than brute force.
2. **Greedy multi-block stacking.** Once you have a shortlist, apply the best block first, then find the best complementary block to add, and repeat until adding a block hurts.
3. **Per-layer alpha tuning.** Each of the 7 layers in a duplicated block gets its own alpha weight, tuned via Optuna Bayesian optimization (60-80 evals). This is where most of the final gains come from.
4. **lm-eval validation.** Confirm the improvements generalize beyond math and EQ to IFEval, BBH, MATH, GPQA, and MuSR.

Key empirical finding along the way: attention repetition helps reasoning, FFN repetition hurts factual knowledge. So the right blocks to duplicate are attention-heavy.

## The honest verdict

This repo also contains several tracks that didn't work. `RESULTS_COMPREHENSIVE.md` says it plainly: "Trained recursion does not help. Only untrained runtime layer duplication at inference provides genuine benefit." That conclusion came from bumping hard into limits in the from-scratch trained recursion tracks (`psrt/`, `solver/`). We document these as instructive dead ends in `HISTORY.md`.

## Top-level markdown files

There are a lot of these. Here's what each one is for:

- **`PAPER.md` (16 KB)** — the paper draft. Structured into Abstract, Introduction, Background (RYS, TRM, BLOOD, BrierHalting), Methods (spectral screening, stacking, DICE pair prediction, oracle seam patching, multi-pass, cross-arch, lm-eval, per-block and per-layer alpha, Bayesian optimization), and Results. Narrative arc: screening → validation → stacking → ablation → generalization → efficiency.
- **`CLAUDE.md` (11 KB)** — the operations manual. Conda setup on the HPC cluster, SLURM configuration, core module APIs, layer duplication mechanics, alpha blending formulas, critical constraints (use_cache=False, Gemma3 sliding window edge cases, lm-eval parameter mappings), model table, key results. If you're about to run something here, read this first.
- **`deeppass_blog_briefing.md` (21 KB)** — the blog-post draft. Heavy on analogies (there's a "20-feet car wash" thing in Part 1 you'll want to read), less equations, more story. Useful for explaining this to anyone who isn't already in the weeds.
- **`HISTORY.md` (141 KB)** — the lab notebook. Everything, chronologically. Chapters for each track: background reading, runtime duplication (Track 1), TRM 7M from-scratch (Track 2), ARR-PSRT 1.7B from-scratch (Track 3), solver on frozen LLMs (Track 4), cross-architecture (Track 5). Root-cause analyses of failures. Version histories of training configs. Brutally honest about what didn't work. Bring coffee.
- **`MEETING_PREP.md` (39 KB)** — advisor meeting notes for the April 9 meeting. Three narrative threads: 72B duplication +7.31, TRM mechanistic insights on overconfidence, solver on frozen LLMs doubling SpatialEval accuracy. Elevator pitches, transition strategies.
- **`RESULTS_COMPREHENSIVE.md` (43 KB)** — the executive summary of everything. Tables per track. If you want the verdict without the history, read this.
- **`CHECKLIST_CROSS_ARCH.md` (3.5 KB)** — cross-architecture validation status. Eight models tested. Three-tier experiments per model (spectral screening, neuron analysis, greedy pairing).

A lot of this is redundant on purpose. `HISTORY.md` is the raw data, `RESULTS_COMPREHENSIVE.md` is the distilled version, `PAPER.md` is the presented version. Different audiences.

## Subfolders

### `scripts/`
The core execution layer. Where you run things from. About 38 Python files plus 40+ bash orchestration scripts.

Subfolders inside:

- `scripts/core/` — the foundation. `layer_duplicator.py` is the engine (it's the thing that actually modifies a model's layer list at runtime). `math_probe.py` and `eq_bench_probe.py` are the two evaluation probes (16 math questions, 20 EQ questions, dual-probe evaluation). `save_duplicated_model.py` handles checkpoint serialization. `compile_results.py` aggregates JSON outputs. Imported by experiments.
- `scripts/experiments/` — ~45 shell scripts plus 4 subdirectories. Spectral screening runs, blood validation, greedy stacking, alpha tuning (Bayesian and grid), lm-eval benchmarks, KV cache debugging (yes, there are nine versions of the KV cache script, we really did need that many), junction finetuning attempts, and Gemma3 / Llama70B pipelines.
- `scripts/dice/` — pairwise compatibility features for DICE (a pair-prediction model we built).
- `scripts/orchestration/` — batch runners for chaining experiments overnight.

### `results/`
Where all the data lives. Flat structure, 32 named subdirectories plus a bunch of loose JSON files. Organization:

- Named experiment runs like `spectral_72B/`, `blood_impact_sweep/`, `greedy_stacking/`, `adaptive_router/`.
- Task-specific buckets: `dice/`, `eqbench_baseline/`, `brierhalting/`, `routing_diagnostic/`, `junction_diagnosis/`.
- Timestamped run dirs like `calme72b_calme-2.1-qwen2-72b_dup_45_52_20260314_002229/`.
- Loose JSON summaries at the top: `pairwise_stacking_sweep.json`, `72b_best_pairs_dual_probe.json`, `72b_pair_sweep.json`.

The subdirectory names are mostly self-explanatory once you're familiar with the vocabulary. The timestamped ones come from automated runs. The named ones are the cleaner curated results that get referenced in `PAPER.md`.

### `psrt/`
PSRT stands for Prompt-based Spectral Recursive Transformer. It's a 1.7B from-scratch model we trained to see if we could build recursion in from the start instead of finding it at inference time. The main file is `arr_psrt_v17.py`. Yes, v17. We iterated a lot trying to debug training instability. The honest answer is that we never got it to beat a dense baseline. `HISTORY.md` has the root-cause analysis, which comes down to a prompt bank rank bottleneck that's provably insufficient to carry information through the scratchpad. Kept here as a cautionary tale.

### `solver/`
Another dead-end track, kept for honesty. A ~12M parameter bidirectional solver head that we bolted onto frozen 8B LLMs. 28 Python files, mostly eval scripts testing different adapter and routing strategies. The solver doubles accuracy on SpatialEval maze navigation tasks (impressive on its own), but it plateaus because the frozen LLM's embeddings and decoder are the bottleneck. We tried unfreezing pieces, adding logit bias, oracle routing, deliberation loops, ensemble configurations. All hit a ceiling.

### `sirt/`
SIRT, Spatial Intelligence Reasoning Transformer. A wrapper around the SpatialEval benchmark. 30+ shell scripts for finetuning and evaluation, targeting Gemma3, Llama3, Mistral, SmolLM2. One subdirectory `eval_results/` for the outputs. Side track exploring whether spatial reasoning tasks specifically benefit from auxiliary modules. Mixed results, no clear signal.

### `SpatialEval/`
The official NeurIPS 2024 SpatialEval benchmark codebase. External project, we only use it. Subfolders: `configs/`, `evals/`, `models/`, `scripts/`, `utils/`. Read-only from our perspective.

### `experiment_a/`
Early LoRA + reasoning-eval experiments on 7B models. Superseded by everything else. Around 22 files. You can ignore this folder unless you're curious about how the project started.

### `prompts/`
28 markdown memos documenting internal reasoning about scaling, sublayer analysis, solver architectures, theoretical bottlenecks. Things like `gpt54_master_v1.md`, `trm_for_llm.md`, `screening_metric_gpt54pro.md`, `why_arr_loses_to_dense.md`. These are internal research conversations (with Claude or GPT-5.4 Pro) saved for reference. Not executable, not core, but useful if you want to understand the thinking.

### `webpage/`
Static HTML dashboard generation. `collect_data.py` aggregates JSON from `results/`. `index.html` and `everything.html` display the tables. `data.json` is the flattened cached data (~166 KB). Final report output.

## Running an experiment

The typical workflow is:

```bash
# 1. Spectral screening to find candidate blocks
sbatch scripts/experiments/spectral_screening_72b.sh

# 2. Greedy stacking (iterative)
sbatch scripts/experiments/greedy_stacking_72b.sh

# 3. Alpha tuning (Bayesian via Optuna)
sbatch scripts/experiments/bayesian_alpha_72b.sh

# 4. lm-eval validation
sbatch scripts/experiments/lmeval_final_72b.sh
```

Each of these spawns SLURM jobs on the HPC cluster. `monitor_jobs.sh` at the repo root tracks live status across all running jobs. `gpu_ping.sh` is a quick GPU-usage sanity check. `overnight_v2.sh` chains several SLURM submissions so you can queue a full pipeline and go to bed.

## Hardware

This does not run on a consumer GPU. Numbers:

- Qwen2-72B in bf16: ~140-160 GB VRAM. B200 (192 GB HBM3e) only.
- Gemma3-27B: ~50-55 GB. A100 80GB works.
- Llama-3 8B: ~15-20 GB. Works on pretty much anything.

Environment on the HPC cluster:

- Conda env: `/blue/cis4914/jietao/DeepPass/envs/deeppass`
- HF cache: `/blue/cis4914/jietao/hf_cache`
- Results: `/blue/cis4914/jietao/DeepPass/results/data/`
- Models: local paths under `models/full/` and `models/small/`
- SLURM partition: `hpg-b200` for 72B work
- CPU allocation: 4 CPUs per GPU, 32 GB RAM per job

## Things that will bite you

- **Do not compile flash-attn from source on the login node.** 40 GB RAM peak, it will crash things. Use a prebuilt wheel. CLAUDE.md has the link.
- **lm-eval `--limit` is weird.** `--limit 1.0` means 1 sample. `--limit 0.5` means 0.5 of the first sample, not half the dataset. Use `--limit 0` (maps to `None`) for the full dataset. There's a wrapper around lm-eval in this repo that normalizes this, but if you call the upstream tool directly, you'll get confused.
- **Gemma3 sliding window breaks naive layer duplication.** You need `use_cache=False` and you need to swap the ModuleList, not wrap layers in a Python loop. Details in `CLAUDE.md`.
- **KV cache and layer duplication do not mix cleanly.** We spent nine iterations debugging this (`scripts/experiments/kv_cache_v1.sh` through `kv_cache_v9.sh`). The final answer is always-off caching, which costs speed but works. Don't try to re-solve this.
- **The 72B model takes ~3 minutes just to load** from disk to VRAM. Factor that into your job timings.

## In progress and abandoned

**In progress:**

- Triple-block and quad-block alpha grids (in `results/data/72b/deeper_stacking/` and `results/data/72b/triples/`).
- Llama3-70B pipeline. The scripts exist but runs are incomplete.
- Gemma3 comprehensive search with mechanistic analysis. Multiple variant scripts, still running through the checklist.

**Abandoned / instructive dead ends** (kept for the writeup):

- ARR-PSRT from-scratch (`psrt/`). 16 training versions. Never beat dense.
- Solver on frozen LLM (`solver/`). Doubled SpatialEval but plateaued.
- Junction finetuning at the seam between duplicated blocks (`scripts/junction_*.py`). Multiple revisions, confusing results.
- SIRT finetuning (`sirt/`). No clear signal.
- `experiment_a/` LoRA + 7B reasoning. Pre-superseded.

## If you're starting cold

Read in this order:

1. `deeppass_blog_briefing.md` for the story (it's the most readable).
2. `PAPER.md` for the formal version.
3. `CLAUDE.md` before running anything.
4. `RESULTS_COMPREHENSIVE.md` for the numbers.
5. `HISTORY.md` only if you really want the full story. It's honest, but it's long.
