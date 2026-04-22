# GPT-5.4 Pro Response: Breaking the 72% Ceiling

**Date:** April 8, 2026
**Diagnosis:** The 72% ceiling is a frozen-decoder interface bottleneck, not a solver limitation.

## Key Insight
The frozen decoder can't effectively read prepended memory tokens because:
1. Memory tokens compete with prompt tokens for normalized attention mass (softmax budget)
2. Memory[0] replaces BOS — the model's natural attention sink
3. The gradient signal through 32 frozen layers is too diluted

## Top 5 Fixes (ranked by expected value)

### 1. Move memory AFTER the prompt (one-line change!)
`[prompt | memory | answer]` instead of `[memory | prompt | answer]`
- Preserves BOS behavior
- Memory adjacent to answer tokens
- Natural causal position

### 2. Preserve BOS / exploit attention sink
Don't let memory[0] be position 0. Keep BOS, inject memory after it.

### 3. Probe z_H to confirm decoder is the bottleneck
Train a tiny MLP on solver's z_H to predict the 4-way answer directly.
If probe > 72%, the solver knows the answer but the decoder can't read it.

### 4. Add auxiliary choice/reachability loss
Direct supervised loss on z_H for answer + per-cell reachability.
Reduces gradient path and training variance.

### 5. Cross-attention sidecars (CALM-style)
If probe gap is real, add external cross-attention at layers 4,8,12,16.
Separate channel for memory — no softmax competition with prompt.

## What NOT to do
- Stop scaling solver width/depth
- Stop adding more K iterations
- Stop adding more prefix slots
- Stop training longer

## Implementation Priority
Days 1-2: Probe + placement sweep (4 positions × 4 seeds)
Days 3-5: BOS-anchor + auxiliary loss
Days 6-10: Cross-attention sidecars
Days 11-14: Logit bias fallback if needed

[Full response with math, code, and citations saved in this file]
