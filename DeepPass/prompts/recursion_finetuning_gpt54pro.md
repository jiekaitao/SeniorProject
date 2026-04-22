# GPT-5.4 Pro Prompt: How to Fine-Tune Pre-Trained LLMs for Recursive Computation Without Catastrophic Forgetting

## Context

We built SIRT (Sublayer-Iterated Recursive Transformer), a novel architecture where designated "core" layers can be repeated 1-4 times with shared weights, while attention iterates fully and FFN contribution is gated by a learned token-wise β. The architecture works when trained from scratch (loss drops, β learns, adaptive halting learns E[K]≈1.1).

**The problem:** When we try to convert a pre-trained model to support recursion via fine-tuning, we destroy the model's existing capabilities.

## What We Tried

### Approach: Freeze everything except core layers, fine-tune with K=1-3 curriculum

```python
# Freeze all
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only core layers [core_start, core_end)
for i in range(core_start, core_end):
    for param in original_layers[i].parameters():
        param.requires_grad = True

# Train with random K ∈ {1,2,3} per step
K = random.choice([1, 2, 3])
orig_layers, orig_N, new_N = build_recursive_model(model, core_start, core_end, K)
outputs = model(input_ids, use_cache=False)
loss = cross_entropy(outputs.logits, labels)
loss.backward()
restore_model(model, orig_layers, orig_N)
```

### Results on LLaMA 3 8B Instruct (32 layers, core=[8,14], lr=5e-6, 1000 steps)

| Config | Combined Score | Delta vs Baseline |
|--------|---------------|-------------------|
| Baseline (K=1, no dup) | **63.76** | — |
| Pre-finetune K=2 (raw dup) | 49.17 | -14.59 |
| **Post-finetune K=1** | **59.66** | **-4.10** (WORSE than baseline!) |
| Post-finetune K=2 | 50.70 | -13.06 |
| Post-finetune K=3 | 41.24 | -22.53 |

**The fine-tuning degraded the model.** Even at K=1 (no recursion), post-finetune is 4 points worse than the original baseline. The core layers were damaged by training with variable K.

### Key Observations

1. **Raw duplication (no fine-tuning) already helps on some models.** On our dual probe:
   - LLaMA 3 70B: +6.55 (pair duplication)
   - Mistral 7B: +2.96 (single block)
   - Gemma3 27B: +7.26 (triple block)
   - SmolLM-135M: +1.92

2. **But raw duplication hurts standardized benchmarks (lm-eval)** on most models:
   - LLaMA 3 70B: BBH -0.2%, MATH -2.0%, MMLU -1.9%
   - Gemma3 27B: BBH -3.4%, MMLU -4.8%
   - **Exception: Mistral 7B with whisper FFN (β=0.2): BBH +0.3%, MATH +2.5%, MuSR +2.6%, MMLU -0.1%**

3. **The attention-FFN asymmetry:** Attention benefits from repetition. FFN can corrupt factual recall. Whisper FFN (β=0.2) is the sweet spot — enough processing without full re-retrieval.

4. **SIRT trained from scratch** (172M, 705M tokens):
   - SIRT K=1 beats dense baseline: 9.60 vs 8.60
   - But K=2 and K=3 hurt (8.64 and 2.91)
   - E[K]≈1.1 — model learned to mostly skip recursion (insufficient training data)

## The Core Problem

We need to teach a pre-trained model to:
1. **Handle seeing core layers twice** without the second pass corrupting representations
2. **Maintain K=1 performance** at or near the original baseline
3. **Improve at K=2** over K=1, proving recursion adds value

The challenge is that pre-trained layers have learned representations calibrated for a single-pass pipeline. When you run them twice, the second-pass input is out-of-distribution for downstream layers. Our finding is that this OOD effect is specifically harmful in the FFN (basin-crossing in associative memory) but potentially helpful in attention (iterative refinement).

## What We Need From You

### Part 1: Diagnosis

Why does fine-tuning core layers with variable K hurt K=1 performance? Provide a mathematical analysis of:
- How the gradient signal from K=2 and K=3 training corrupts the K=1 weight space
- Why the core layers become "confused" about whether they're in first or second pass
- The connection to catastrophic forgetting in continual learning

### Part 2: Five Different Approaches

Design **5 fundamentally different approaches** to making recursion fine-tuning work. For each, provide mathematical formulation, pseudocode, and expected failure modes.

**Approach ideas to explore (not limited to these):**

1. **LoRA per recursion depth** — Following Relaxed Recursive Transformers (arXiv 2410.20672). Add rank-8 LoRA adapters that are ONLY active during recursion passes t>1. Base weights stay frozen. The LoRA learns to "translate" the refined representation into something the next layer expects.

2. **Adapter modules at recursion seams** — Insert small adapter networks (e.g., 2-layer MLP with bottleneck) between the first and second pass of core layers. These adapters learn to map the refined representation back to the distribution expected by the core layers.

3. **Distillation from K=1 to K=2** — Train with a distillation loss: the K=2 output should match or exceed the K=1 output on the same input. This prevents the model from forgetting what K=1 should produce.

4. **Progressive depth unfreezing** — Start by only duplicating 1 layer (the most beneficial from our SBUID analysis). Fine-tune. Then add a second duplicated layer. Fine-tune. Gradually build up to the full recursive core.

5. **Residual scaling + β warm-start** — Instead of full layer duplication, start with the second pass contributing α=0.01 (nearly zero) and gradually increase α during training. The model slowly adapts to the iterative signal.

6. **Mixture-of-Recursion (MoR)** — Route different tokens to different recursion depths. Some tokens get K=1, others get K=2. A tiny router learns which tokens benefit from recursion.

7. **Contrastive recursion training** — For each batch, compute loss at K=1 and K=2. Use a contrastive objective that ONLY updates when K=2 is better than K=1, preventing harmful updates.

### Part 3: Specific Implementation for LLaMA 3 8B

Give us a concrete, implementable plan for LLaMA 3 8B Instruct (32 layers, 8B params):
- Which layers to use as recursive core (we found layers 8-15 are best from SBUID screening)
- Exact hyperparameters (LR, steps, batch size, LoRA rank if applicable)
- Training curriculum (what to train first, what to add later)
- How to validate at each stage (what metrics indicate it's working vs failing)
- Total compute budget estimate (we have 1-4 B200 GPUs)

### Part 4: Why Mistral Works

Mistral 7B is the ONE model where raw duplication (no fine-tuning) improves standardized benchmarks. Why?
- Mistral uses sliding window attention (4096 window)
- Layer 28 is the best block — deep in the network
- FFN is essential (attn-only hurts by 2.85 points)

What makes Mistral's architecture uniquely suited to layer repetition? Can we use this understanding to design a fine-tuning approach that makes other models behave like Mistral?

## Papers to Reference

- **Relaxed Recursive Transformers** (arXiv 2410.20672) — LoRA per recursion depth
- **LayerSkip** (arXiv 2404.16710) — Shallow-competency training, early exit
- **PonderNet** (arXiv 2107.05407) — Learned halting
- **AdaPonderLM** — Adaptive pondering for language models
- **Universal Transformers** (arXiv 1807.03819) — Weight-tied recursion + halting
- **Mixture-of-Recursion (MoR)** (arXiv 2507.10524) — Token-level recursion routing
- **The Curse of Depth** (arXiv 2502.05795) — Why deep recursion degrades signal
- **LoRA** (arXiv 2106.09685) — Low-rank adaptation
- **Geva et al.** (arXiv 2012.14913) — FFN as key-value memory (why re-retrieval corrupts)

**Search the web for any 2025-2026 papers on:**
- Fine-tuning for recursive/iterative computation
- Preventing catastrophic forgetting during architecture modification
- Teaching pre-trained models new computation patterns
- Depth-wise adaptation without retraining

## Constraints

- We have LLaMA 3 8B Instruct pre-downloaded
- 1-4 B200 GPUs available (192GB each)
- Training budget: 1-3 days
- The result must show: **K=2 post-finetune > K=1 baseline** on dual probe
- K=1 post-finetune must be within 1 point of original baseline (no catastrophic forgetting)

## Output Format

For each approach:
```
## Approach N: [Name]
### Why it should work (mathematical argument)
### Pseudocode (PyTorch, implementable)
### Expected compute cost
### How to detect if it's failing (early stopping criteria)
### Risk assessment
```

Think deeply. Search recent literature. The key insight we need: **how to add iterative computation to a pre-trained model without breaking what it already knows.**
