# Adaptive Iteration Routing

Per-input block selection: dynamically choose which layer block to duplicate at test time.

## Motivation

Fixed config (50,60) helps math (+15.4%) but hurts IFEval (-4.2%). Config (45,52) helps IFEval (+2.4%) but hurts MATH Hard (-6.4%). Different inputs need different blocks. A router that selects the right block per-input could recover gains while avoiding losses.

## Architecture (GPT-5.4 Pro + Claude design)

**Hybrid ESR + DSG:**
1. **Exact Spectral Router (ESR):** Measures displacement rho + margin gain directly for K candidate blocks. Expensive but accurate. Used as teacher/oracle.
2. **Distilled Spectral Gate (DSG):** Tiny MLP trained on ESR labels. Takes hidden states from anchor layer, outputs block selection. Cheap at inference.
3. **Cascaded hybrid:** DSG when confident, ESR fallback on top-2 when not.

**Key design decisions:**
- Arm 0 = "no duplication" (first-class, not an ablation)
- Candidates from multiple network regions, not just one neighborhood
- Last-token + learned projection (not wide pooling)
- Hierarchical: none/region first, then block within region

## Files

- **`routing_diagnostic.py`** — Critical feasibility test. Answers: does the optimal block vary per-input or per-task? If per-task, a simple task classifier suffices. If per-input, geometric routing is justified.

## Status

Diagnostic complete on 7B. Initial ESR scoring (geometric-only) was dominated by residual magnitude and picked wrong blocks. V2 scoring adds LM-head margin gain. Full ESR+DSG implementation pending.

## Key Finding

Pure geometric metrics (displacement rho, residual) are insufficient for routing. Destructive blocks show low rho + high residual, mimicking "good contraction." An output-quality signal (logit margin gain) is essential.
