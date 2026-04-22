# GPT-5.4 Pro Prompt: Scaling Recursion Fine-Tuning to SOTA for 8B Models

## Context

We have a working technique for improving LLMs through recursion fine-tuning — teaching a pre-trained model to benefit from running certain transformer layers twice. We want to push this to achieve SOTA or near-SOTA results on the 8B model class.

## What We Have Working

### Recursion Fine-Tuning (300 steps, lr=5e-7, freeze all except core layers)

**LLaMA 3 8B Instruct** (32 layers, core=[10,13]):
- Dual probe: Baseline 63.76 → K=2 post-ft **66.98 (+3.22)**
- lm-eval: BBH **+0.9%**, MMLU-PRO **+0.1%**, MuSR **+3.5%**, MATH -1.0%
- Only 300 steps of fine-tuning with K=1-3 curriculum
- Core layers [10,13] frozen except during training

**Gemma3 27B** (62 layers, core=[11,14]):
- Dual probe: Baseline 76.96 → K=2 post-ft **81.08 (+4.11)**
- K=1 perfectly preserved (76.96 → 76.96)
- lm-eval still running

### Key Findings Across 8+ Models

1. **Attention benefits from repetition, FFN can hurt** — but the balance is architecture-dependent
2. **Whisper FFN (β=0.2) is the sweet spot** — on Mistral 7B: BBH +0.3%, MATH +2.5%, MuSR +2.6%, MMLU -0.1%
3. **SBUID screening** (r=0.668 on LLaMA 3 70B) identifies which blocks to duplicate
4. **KV cache works** with our LayerIdxWrapper fix
5. **Small models benefit less** — LLaMA 3 8B gets +3.22, but 70B gets +6.55 on dual probe
6. **Sliding window models are harder** — Gemma3's screening metrics fail, but duplication still helps

### Current Limitations

1. **K=1 degradation on LLaMA 3 8B** — post-ft K=1 drops 2.6 points. Fine-tuning shifts the model toward expecting recursion
2. **K>2 hurts** — K=3 gives diminishing returns or regression on most models
3. **Only 300 steps** — we haven't tried longer fine-tuning or more sophisticated approaches
4. **No sublayer control during fine-tuning** — we fine-tune full layers, not attention/FFN separately

## Our Scaling Experiments (Currently Running)

1. **Wider core [8,16]** — 8 layers instead of 3
2. **Longer training (1000 steps)** — more exposure to K=2/K=3
3. **Dual recursive zones [4,7]+[10,13]** — two separated blocks, early + mid

## What We Want

### Goal: Beat LLaMA 3 8B Instruct on Open LLM Leaderboard v2 benchmarks

Current LLaMA 3 8B Instruct scores (our 15% subsample):
- BBH: 47.5%
- MATH Hard: 9.9%
- MMLU-PRO: 27.7%
- MuSR: 34.8%

Our current best (recursion ft K=2):
- BBH: 48.4% (+0.9)
- MATH Hard: 8.9% (-1.0)
- MMLU-PRO: 27.8% (+0.1)
- MuSR: 38.3% (+3.5)

**We want ALL four to improve, or at least 3/4 with the 4th flat.**

### Specific Questions

**1. Architecture modifications during fine-tuning:**
- Should we add LoRA adapters at the recursion seam instead of full layer fine-tuning?
- Should we add a small "recursion adapter" module (e.g., a 2-layer MLP) between first and second pass?
- Should we use different LR for attention vs FFN within the core?
- Should we apply whisper FFN (β=0.2) during the second pass by default?

**2. Training strategy:**
- Is 300 steps enough or do we need 1000+?
- Should we use distillation (K=1 teacher → K=2 student) instead of curriculum?
- Should we progressively increase K (first 200 steps K=1-2 only, then add K=3)?
- Should we train on math/reasoning data instead of general text?

**3. Which layers to recurse:**
- Our SBUID screening found layers [8-15] are best on LLaMA 3. Should we use all 8?
- Early layers (4-7) or mid layers (10-13) — which gives more reasoning benefit?
- Should we recurse attention-heavy layers differently from FFN-heavy layers?

**4. Inference-time tricks:**
- Should we apply different β per layer during K=2 inference?
- Should we use our gate-margin gating (skip FFN when gates would flip)?
- Should we ensemble K=1 and K=2 outputs (e.g., weighted average of logits)?

**5. Evaluation strategy:**
- Should we run full lm-eval (not 15% subsample) for the final result?
- Should we include IFEval (requires generation, slow without cache)?
- What's the comparison baseline — raw LLaMA 3 8B or the best fine-tuned variant?

## Hardware & Constraints

- 1-4 NVIDIA B200 GPUs (192GB each)
- LLaMA 3 8B pre-downloaded on HiPerGator
- 6.1B tokens of fineweb-edu tokenized and ready
- Can run experiments for 2-3 more days
- KV cache fix works for cached generation/evaluation

## Papers to Reference and Search For

- Relaxed Recursive Transformers (arXiv 2410.20672) — LoRA per depth
- LayerSkip (arXiv 2404.16710) — shallow competency
- PonderNet (arXiv 2107.05407) — halting
- Mixture-of-Recursion (arXiv 2507.10524)
- SOLAR 10.7B (depth upscaling via layer duplication + fine-tuning)
- Ng's RYS method (HuggingFace dnhkng/RYS-XLarge)
- MobileLLM (arXiv 2402.14905) — deep-thin architectures

**Search the web for:**
- Any 2025-2026 papers on improving 8B models through architectural tricks
- Depth upscaling results on LLaMA 3 specifically
- Any work combining layer duplication with fine-tuning
- Best practices for fine-tuning recursive/iterative computation into pre-trained models
- Recent lm-eval leaderboard scores for 8B models to compare against

## Output Format

Give us:
1. **Top 3 most promising modifications** with mathematical justification and pseudocode
2. **Exact hyperparameter recommendations** for LLaMA 3 8B
3. **Training curriculum** (what to do in steps 0-100, 100-500, 500-2000)
4. **Expected improvements** (what BBH/MATH/MMLU/MuSR deltas to expect)
5. **Risk assessment** (what could go wrong, how to detect it)
6. **Comparison to existing methods** (how does this compare to SOLAR, depth upscaling, etc.)

Think deeply. Search recent literature. Be specific about implementation details.
