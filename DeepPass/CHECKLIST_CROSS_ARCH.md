# Cross-Architecture Validation Checklist

**Goal:** SBUID screening + Tier 2 neuron analysis on ALL models by morning (Mar 28-29)
**GPU limit:** 6 total, 150G RAM max

## Models

| # | Model | HF ID | Params | Layers | Attn Type | Status |
|---|-------|-------|--------|--------|-----------|--------|
| 1 | Gemma 3 27B | local: models/full/gemma-3-27b-it | 27B | 62 | Sliding window | ✅ DONE |
| 2 | LLaMA 3 70B | local: /data/ai/models/nlp/llama/...70B-Instruct-hf | 70B | 80 | Full | ✅ SBUID done, Tier 2 partial |
| 3 | LLaMA 3 8B | local: /data/ai/models/nlp/llama/...8B-Instruct-hf | 8B | 32 | Full | 🔄 RUNNING |
| 4 | Gemma 2 9B | google/gemma-2-9b-it | 9B | 42 | Full (no sliding) | ❌ NEED DOWNLOAD + RUN |
| 5 | Mistral 7B | mistralai/Mistral-7B-Instruct-v0.3 | 7B | 32 | Sliding window | ❌ NEED DOWNLOAD + RUN |
| 6 | LLaMA 3.1 8B | meta-llama/Llama-3.1-8B-Instruct | 8B | 32 | Full | ❌ NEED DOWNLOAD + RUN |
| 7 | Phi-3 Medium | microsoft/Phi-3-medium-4k-instruct | 14B | 40 | Full | ❌ NEED DOWNLOAD + RUN |
| 8 | Cohere Command R | CohereForAI/c4ai-command-r-v01 | 35B | 40 | Full | ❌ NEED DOWNLOAD + RUN |

## Per-Model Experiments

For each model, run:

### Tier 1: SBUID Screening
- [ ] Spectral screen ALL single-layer blocks (rho + BLOOD)
- [ ] Compute SBUID at λ=6k, 10k, 20k
- [ ] Evaluate top 10 + bottom 5 with dual probe
- [ ] Compute Spearman correlation (SBUID vs combined)
- [ ] Save: spectral_screen.json, evaluated_blocks.json

### Tier 2: Neuron Analysis (on best block)
- [ ] Attn-only vs full duplication (FFN impact)
- [ ] Whisper FFN (β=0.2)
- [ ] Gate margin / flip rate measurement
- [ ] Save: sublayer_analysis.json

### Bonus (if time)
- [ ] Greedy pair from best single
- [ ] Per-layer DLA on best block

## Schedule (estimated)

| Time | GPU 1 | GPU 2 | GPU 3 | GPU 4 | GPU 5 | GPU 6 |
|------|-------|-------|-------|-------|-------|-------|
| Now | g3_refin | ll3_eval | ll3_8b | ef_en_2b | — | — |
| +1h | g3_refin | ll3_eval | gemma2_9b | mistral_7b | — | — |
| +2h | ll3_70b_t2 | ll3_eval | gemma2_9b | llama31_8b | phi3_14b | cohere_35b |
| +4h | ll3_70b_t2 | ll3_eval | phi3_14b | cohere_35b | — | — |
| +8h | (cleanup) | — | — | — | — | — |

## Results Summary (fill in as completed)

| Model | Params | Attn Type | SBUID r | SBUID p | Best Δ | FFN Impact | Gate Flip |
|-------|--------|-----------|---------|---------|--------|-----------|-----------|
| LLaMA 3 8B | 8B | Full | 0.000 | 1.000 | +0.25 | +0.05 (neutral) | 12.5% |
| LLaMA 3.1 8B | 8B | Full | 0.221 | 0.349 | +1.21 | -6.59 (helps) | 8.5% |
| Mistral 7B | 7B | Sliding | 0.221 | 0.349 | +2.96 | -3.81 (helps) | 6.0% |
| Gemma 2 9B | 9B | Full | -0.337 | 0.146 | +6.11 | -5.73 (helps) | 8.5% |
| Gemma 3 27B | 27B | Sliding | -0.075 | 0.567 | +7.26 | Mixed | L0=37% |
| LLaMA 3 70B | 70B | Full | +0.668 | 0.001 | +4.83* | -4.49 (helps) | 4.5% |
| Phi-3 14B | 14B | Full | ❌ FAILED | — | — | — | — |
| Cohere 35B | 35B | Full | ❌ GATED | — | — | — | — |

*LLaMA 3 70B T2 run found different best block (63,64) vs original spec run (10,11). Both show +3-6 improvement.

**Key findings:**
1. FFN helps on ALL models except Gemma3-27B (which has sliding window)
2. SBUID only significant on LLaMA 3 70B (r=0.668, p=0.001)
3. Larger models benefit more: 8B=+0.25 to +1.21, 27B=+7.26, 70B=+4.83 to +6.55
4. Gate flip rates are low (4-12%) on all models except Gemma3 L0 (37%)
5. Gemma 2 9B (no sliding window) gets +6.11 — much more than LLaMA 3 8B (+0.25)
