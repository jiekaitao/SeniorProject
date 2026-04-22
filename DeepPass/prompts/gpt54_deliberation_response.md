# GPT-5.4 Pro Response: Recurrent Deliberation Controller

**Date:** April 8, 2026
**Key thesis reframe:** "A frozen LLM with a learned recurrent control interface that allocates extra latent compute only when needed."

## Core Architecture

Frozen decoder F_ω + trainable controller C_θ with latent state z_r.

Round r:
- (h_r, p_r) = F_ω(x, m_r, u_r)  — frozen LM forward with thought slots m_r and steering u_r
- z_{r+1} = C_θ(z_r, φ(h_r, p_r))  — controller updates state from hidden states + logits
- m_{r+1} = W_m z_{r+1}  — new thought slots
- u_{r+1} = W_u z_{r+1}  — new steering signals

## Key Design Decisions

1. **Thought slots in vocab space**: m_r = Σ α_v E_v (sparse superposition of vocab embeddings via sparsemax). Keeps signal native to frozen LM manifold.

2. **Read hidden states**: tap layers 8, 16, 24 + think slot hidden states + answer logits + entropy + margin.

3. **Verifier head**: v_r = σ(w^T s_r), predicts whether current answer is correct. Critical for meaningful self-improvement (solver-verifier gap).

4. **Loss**: CE(p_2, y) + λ_v BCE(v_r, correct?) + λ_p max(0, CE(p_2,y) - CE(p_1,y) + δ)
   - Final answer must be right
   - Verifier must know when round 1 is wrong
   - Round 2 should not be worse than round 1

## Why This Is Different From Current Solver

- Current: one-shot preconditioning of prompt
- Proposed: closed-loop recurrent control over frozen model's internal computation
- Task-agnostic: works for math, code, proofs, spatial reasoning
- Controller outputs HOW the frozen model should think, not the answer itself

## Modular Actions (Not Domain Routing)

a_r ~ π_θ(a | s_r), a ∈ {plan, verify, lookahead, halt}

Route by computation type, not by domain. This is why MoERM collapsed — it tried to route by domain on homogeneous inputs.

## Implementation Plan (48 hours)

Minimal version:
- 2 rounds (fixed)
- 8 latent think slots
- Tapped layers: 8, 16, 24
- Sequence: [prompt | THINK_1...THINK_8 | Answer:]
- Controller: d_state=512, read MLP + state update + vocab writer + verifier
- Train on SpatialEval as proof of concept, then generalize

## References

- COCONUT (COLM 2025): hidden-state feedback, latent BFS behavior
- Recurrent-depth (NeurIPS 2025): test-time recurrence for reasoning
- MoR: adaptive recursive depth
- Dr.LLM: retrofitted layer routing
- StateLM: internal reasoning loops
- ASM: dynamic history-dependent steering
- Solver-verifier gap theory: gains from verification, not raw recurrence

## TRM's Role

Demoted to:
- Positive control (what good iteration looks like on structured data)
- Teacher for auxiliary supervision
- Diagnostic baseline (separates "mechanism works" from "interface fails")

NOT the headline architecture.
