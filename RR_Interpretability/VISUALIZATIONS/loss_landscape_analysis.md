# Loss Landscape Analysis: Why Brier + Monotonicity + Softmax Accelerates Convergence

## The Three Modifications (BASELINE → FULL_COMBO)

### 1. StableMax → Softmax (token classification loss)

StableMax is a piecewise function:
```
s(x) = 1/(1-x)   for x < 0    (hyperbolic)
s(x) = x + 1      for x ≥ 0    (linear)
```

While C¹-continuous at x=0 (both left and right derivatives equal 1), the **second derivative is discontinuous**:
- For x < 0: d²s/dx² = 2/(1-x)³  (nonzero curvature)
- For x ≥ 0: d²s/dx² = 0  (zero curvature, perfectly linear)

**Impact on iterative computation:** When logits hover near zero during refinement across ACT steps, the Hessian has a ridge at z=0. This creates an unstable optimization landscape for shared-weight iteration — the gradient direction changes abruptly as logits cross zero.

Softmax (exp(x)) is C∞ — smooth everywhere, with consistent curvature. The iterative process follows a clean gradient trajectory.

**Measured effect:** Switching to softmax distributes latent geometry from 1.7 → 4.1 effective dimensions (PC1 variance: 76.9% → 42.8%).

### 2. BCE → Brier Score (halting loss)

**BCE:**
```
L = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
```
- Unbounded: L → ∞ as σ(z) → 0 when y=1
- Gradient: ∂L/∂z = σ(z) - y (bounded in [-1,1])
- But loss SPIKES when iterative refinement flips a correct → incorrect prediction

**Brier:**
```
L = (σ(z) - y)²
```
- Bounded: L ∈ [0, 1]
- Gradient: ∂L/∂z = 2(σ(z)-y)·σ(z)·(1-σ(z)) (also bounded)
- Quadratic: proportional penalty, smooth landscape

**Critical for iterative models:** In an ACT loop with 16 steps and shared weights, BPTT propagates gradients through all steps. A single BCE spike at step k creates gradient explosions through steps 0..k-1. Brier's bounded loss prevents this.

**Calibration:** Brier uniquely encourages probabilistic calibration — the model outputs σ(z)=0.7 when 70% sure, not a binary 0/1. This gives the halting mechanism meaningful, gradual probability estimates that can evolve smoothly across ACT steps.

### 3. Monotonicity Regularization

```
L_mono = relu(q_prev - q_curr)² · 1{step > 0}
```

Penalizes the halting probability from INCREASING across steps. This enforces that the model's "confidence in continuing" only decreases — once it's ready to halt, it shouldn't change its mind.

**Key insight:** Monotonicity alone HURTS (4.2% < 9.1% baseline). BCE-trained halting probs are poorly calibrated (binary near 0 or 1), so forcing monotonicity on them constrains exploration. Monotonicity only helps with Brier-calibrated probabilities.

## Ablation Results (Exact Accuracy on 30×30 Mazes)

| Modification | Brier | Mono | Softmax | Exact Acc | Δ vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| BASELINE | - | - | - | 9.10% | — |
| BRIER_ONLY | ✓ | - | - | **19.80%** | **+10.7pp** |
| MONO_ONLY | - | ✓ | - | 4.20% | -4.9pp |
| SOFTMAX_ONLY | - | - | ✓ | 6.90% | -2.2pp |
| FULL_COMBO | ✓ | ✓ | ✓ | **30.80%** | **+21.7pp** |

## Synergy Analysis

```
Individual effects: +10.7 + (-4.9) + (-2.2) = +3.6pp
Observed total: +21.7pp
Pure synergy: +21.7 - 3.6 = +18.1pp interaction effect
```

The interaction effect (+18.1pp) exceeds the sum of individual effects (+3.6pp) by 5×. This is because:

1. **Brier calibrates** → halting probs are smooth & meaningful
2. **Monotonicity constrains** → calibrated probs can be meaningfully ordered
3. **Softmax smooths** → token gradients flow cleanly through shared-weight iterations

Without Brier, monotonicity constrains garbage. Without softmax, the Hessian ridge destabilizes gradient flow. All three must work together.

## Why FULL_COMBO Converges Faster (Same Final Token Accuracy, Better Exact Accuracy)

Both FULL_COMBO and BASELINE reach ~97% token accuracy. But FULL_COMBO reaches it in 1-2 ACT steps vs 2-3 for BASELINE, and achieves 3.4× better exact accuracy (30.8% vs 9.1%).

The mechanism: Brier+softmax create a smooth loss landscape where each ACT step makes consistent progress. The shared weights learn to make each iteration USEFUL rather than oscillatory. Monotonicity prevents the "two steps forward, one step back" behavior that wastes computation in BASELINE.

## Implications for Adaptive Computation

1. **Loss function choice matters more than architecture** for iterative models: Brier alone gives +10.7pp, the biggest single contribution
2. **Calibration enables regularization:** Monotonicity is harmful without calibration, essential with it
3. **Smooth activations prevent Hessian discontinuities** in shared-weight iteration
4. **Early halting is viable:** FULL_COMBO converges by step 2 → could save 87% of ACT compute
