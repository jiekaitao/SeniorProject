# Junction Fine-Tuning Experiments (V1 → V4)

Progressive evolution of junction fine-tuning approaches.

## Version History

### V1 — `junction_ft_v1.py` (originally `junction_finetune.py`)
- **Loss:** Logit KL divergence with base model
- **Trainable:** 2 junction layers (where duplicated block loops back)
- **Result on 7B (15,18):** +6.0% relative improvement (0.3901 → 0.4137)
- **Limitation:** KL on full vocabulary is diluted, slow convergence

### V2 — `junction_ft_v2.py`
- **Added:** Procrustes initialization (analytical zero-shot alignment via SVD)
- **Loss:** Same as V1
- **Improvement:** Better starting point, slightly faster convergence

### V3 — `junction_ft_v3.py` / `junction_ft_v3_72b.py`
- **Loss:** Hidden-state MSE at junction (direct gradient, not logit KL)
- **Trainable:** 4 junction layers with per-layer learning rates
- **Config-aware:** Good configs get MSE only; bad configs get MSE + KL
- **Result on 7B:** Good configs preserved (+2.4%), bad configs recovered (102%)
- **Result on 72B:** HURT the model (-0.0796 on math probe)
- **Problem:** MSE loss pushes h_59 → h_49, which FIGHTS iterative refinement (see V4)

### V4 — `junction_ft_v4_adapter.py` (Current best design)
- **Architecture:** Tiny bottleneck adapter (8192→256→8192) inserted at junction
- **Key insight:** Don't train the duplicated layers or align hidden states. Instead, insert a minimal "voltage converter" that adjusts signal format without erasing refinement content.
- **Loss:** End-to-end logit KL with base model
- **Trainable:** ~8.4M params (0.01% of 72B model) — adapters only, everything else frozen
- **Residual connection:** Starts as identity (near-zero init), training adds minimal correction
- **Why V4 > V3:** V3 fights TRM-style iterative refinement by making the second pass input look like the first pass never happened. V4 preserves the refinement while making the junction functional.
- **Status:** Implemented, pending validation

## Supporting Files

- **`junction_ft_72b.py`** — Earlier 72B attempt (OOM'd)
- **`junction_diagnosis.py`** — Comprehensive diagnostic: analyzes hidden state distributions at junction points
- **`comprehensive_junction_ft.py`** — Multi-config runner for V1/V2/V3 comparison
