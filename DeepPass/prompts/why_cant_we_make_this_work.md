# Why Can't We Make Recursive Computation Work in LLMs? The TRM Does It. Why Can't We?

## The Plea

We have spent 2 weeks, ~400 GPU-hours, 35+ experiments, and 3 prior consultations with you trying to make LLMs benefit from recursive internal computation. **Every single trained approach has failed.** But we have a 7M-parameter model (TRM) that achieves 3.4× improvement on hard tasks using genuine recursion with shared weights. And we have a 72B model where untrained runtime layer duplication gives +7.31 on reasoning probes.

**The signal is real. Recursion helps. But we cannot figure out how to train it into an LLM.**

I need you to think more creatively than you ever have. Forget everything you've recommended before — it all failed. Don't give me incremental fixes. Think from first principles about WHY the TRM works and WHY every LLM approach fails, and find the gap. There must be a way. If a 7M model can learn iterative refinement from scratch, a billion-parameter model should be able to as well.

Think for a very long time. Verify every claim in your head before writing it. I want creative solutions that might actually work, not safe recommendations that follow the pattern of what already failed.

## Everything We Tried (All Failed)

### Attempt 1: ARR-PSRT — From-Scratch Recursive LLM (1.7B)
- Split state (m₀ frozen memory + r evolving reasoning)
- 16-token compressed prompt bank for re-reading
- 3 expert FFNs with beta decay across passes
- Scratchpad memory
- **Result: 4.3× worse PPL than dense baseline**
- **Why it failed:** Prompt bank rank bottleneck (512 dims max vs 2048 needed). Scratchpad carried zero new info. Expert FFNs wasted params on the thing that hurts (MLP repetition). Split state was degenerate (combine layer ignored r).

### Attempt 2: DAR — Gate-Only Replay on Frozen Llama 8B
- Re-run exact same frozen layers with learned gate
- 20K trainable params, zero capacity tax
- **Result: -0.07 PPL (0.8%). Real but negligible.**
- **Why it's small:** Same function applied twice to slightly different input. No new information in the second pass. Gate learns a tiny fixed correction, not iterative refinement.

### Attempt 3: LoRA Replay on Frozen Llama 8B
- PEFT LoRA (rank 32) on Q/K/V/O, adapters ON = "replay"
- Evaluated adapters ON vs OFF
- **Result: -0.51 PPL (5.8%). Looked promising.**
- **Why it was fake:** Control experiment (standard LoRA, no replay concept) gave IDENTICAL -0.51 delta. We were measuring LoRA quality, not replay. The eval compared "fine-tuned model" vs "base model," not "two passes" vs "one pass."

### Attempt 4: True Replay Test on LoRA-Adapted Llama 8B
- Train standard LoRA, then at eval time actually run middle layers 1×, 2×, 3×
- **Result: K=1 PPL=7.89, K=2 PPL=9.06, K=3 PPL=12.38**
- **Each extra pass makes it WORSE.** The LoRA-adapted layers were optimized for single-pass. Running them twice pushes states out of distribution.

### Attempt 5: CIRRA — Contained Input-Reinjected Recurrent Attention
- YOUR recommended design from our last consultation
- Separate trainable 4-layer core (deep-copied from base layers 12-15)
- Shared weights across K cycles (true iteration)
- Dense reinjection of full-rank layer-12 state every cycle
- Attention-dominant with tiny bottleneck MLP (scale init=0)
- Trained with K ∈ {2,4,8}, evaluated at K=1,2,4,8
- 4-arm experiment: full CIRRA, no-reinjection ablation, always-on control, seed repeat
- **Result:**

| K | Control (extra params, no loop) | CIRRA (no reinject) | CIRRA (full) | CIRRA (seed 2) |
|---|-------------------------------|-------------------|-------------|---------------|
| 1 | **8.36** | 8.74 | 8.74 | 8.74 |
| 2 | — | 10.26 | 10.25 | 10.23 |
| 4 | — | 10.04 | **9.77** | 9.91 |
| 8 | — | 10.24 | 9.88 | 10.09 |

- **K=1 is always best. No K-scaling. CIRRA fails.**
- The always-on control (8.36) beats everything — extra parameters help as a single-pass enhancement, not as a recurrent loop.
- Dense reinjection helps slightly (D1 K=4=9.77 vs C K=4=10.04) but neither beats K=1.

## The One Thing That DOES Work: TRM

The Tiny Recursive Model (7M params) trained from scratch on 30×30 maze solving:

```python
# The core loop
for H_step in range(3):
    for L_step in range(6):
        z_L = L_level(z_L, z_H + input_embeddings)  # input injected EVERY step
    z_H = L_level(z_H, z_L)

output = lm_head(z_H)
```

- **Config:** hidden=512, 2 L-layers, H_cycles=3, L_cycles=6, 8 heads
- **Non-causal attention** (bidirectional)
- **Shared weights** (same L_level blocks every cycle)
- **Input injection** every L-step (raw embeddings, not compressed)
- **Two-level hierarchy** (z_H and z_L ping-pong)
- **Gradient truncation** (only last H-cycle gets gradients)
- **ACT with Q-learning** for adaptive halting
- **Reduced-MLP ablation works** (attention-only in early L-level blocks)
- **Result: 30.8% exact maze accuracy vs 9.1% baseline (3.4× improvement)**
- **Contraction rate: ρ ≈ 0.84-0.86** (converging on learned attracting manifold)

## The Other Thing That Works: 72B Runtime Duplication

- Take pretrained Qwen2-72B
- Duplicate layers 45-52 at inference (run them twice, no training)
- **+7.31 combined score improvement**
- Reasoning: +2.3% IFEval, +1.3% MuSR
- Knowledge: -6.4% MATH (FFN repetition hurts factual recall)

## The Paradox

| Approach | Works? | Key Property |
|----------|--------|-------------|
| TRM from scratch on mazes | **YES (3.4×)** | Shared weights, input injection, bidirectional, hard task |
| 72B runtime duplication | **YES (+7.31)** | No training, inference-time, on pretrained model |
| ARR from scratch on LM | NO (4.3× worse) | Compressed bank, split state, expert FFNs |
| Gate replay on pretrained LM | BARELY (-0.07) | Same weights twice, no differentiation |
| LoRA replay on pretrained LM | FAKE (-0.51) | Was measuring LoRA quality, not replay |
| True replay (layers 2×) on LM | NO (K=2 hurts) | Layers trained for single-pass |
| CIRRA on pretrained LM | NO (K=1 best) | Separate core, dense reinjection, shared weights |

**Why does TRM work but nothing transfers to LLMs?**

## My Hypotheses (Challenge These)

### H1: The task is wrong
TRM solves mazes — a task that REQUIRES iterative computation (finding paths through a graph). Next-token prediction on web text mostly doesn't require iteration. Maybe recursion only helps on tasks with inherent computational depth.

**But:** 72B duplication helps on IFEval and MuSR (reasoning tasks), and we still can't train replay even on math-heavy data.

### H2: Causal masking kills it
TRM uses bidirectional attention. Every token sees every other token. LLMs use causal masking — each token only sees previous tokens. This fundamentally limits what the second pass can learn from.

**But:** CIRRA had full access to all previous tokens on every pass. Bidirectional should help but shouldn't be necessary for the second pass to refine an already-processed sequence.

### H3: Pretrained models are too "single-pass hardened"
The pretrained LLM has learned to put ALL useful computation into one forward pass. Its internal representations, layer norms, residual stream magnitudes — everything is calibrated for single execution. Inserting any recurrence disrupts this calibration catastrophically.

**But:** TRM trains from scratch, so it doesn't have this problem. Could we train a recursive LM from scratch at scale? We tried with ARR but the architecture was wrong (rank bottleneck). What if we used TRM's architecture at LLM scale?

### H4: The two-level structure is essential
TRM has z_H and z_L that ping-pong. This creates a hierarchical refinement process where high-level plan (z_H) and low-level execution (z_L) iterate. All our LLM attempts used a single hidden stream. Maybe single-stream recurrence is fundamentally limited.

### H5: Gradient truncation is essential
TRM only backpropagates through the last H-cycle. Earlier cycles run with `torch.no_grad()`. This prevents the gradient from having to navigate through 18 sequential applications of the same weights. All our LLM training backpropagated through all passes.

### H6: We need to train from scratch with the RIGHT architecture
ARR failed because of the rank bottleneck, not because from-scratch is impossible. TRM proves from-scratch works. What if we built TRM's exact architecture but at LLM scale (512→4096 hidden, 2→12 L-layers) and trained it on text? Not next-token prediction on web text, but on specifically HARD text tasks?

### H7: The scale is wrong
TRM is 7M params. Maybe recursion helps at small scale where the model doesn't have enough single-pass capacity, but at billion-param scale the model already has enough depth that recursion is redundant.

**But:** 72B duplication helps, so even massive models benefit from extra computation. The issue isn't capacity — it's that we can't TRAIN the recursion.

### H8: We're not training long enough or on the right data
Our CIRRA experiment ran 10K steps on mixed web+math data. TRM trained for 100K epochs on mazes. Maybe the recurrent core needs much more training specifically on depth-demanding tasks.

## What I Want From You

1. **Tell me which of my hypotheses are right and which are wrong.** Be specific. Use formal arguments.

2. **Identify what I'm missing.** There's something fundamental about why TRM works that we haven't captured. What is it?

3. **Design something that will actually work.** Not another incremental variant. Something fundamentally different from everything we've tried. Think about this from the perspective of: "What does TRM have that all our LLM attempts lack? How do we give an LLM that same property?"

4. **If this genuinely cannot work for autoregressive language modeling, explain WHY with a formal proof.** I'll accept a proof of impossibility. But it has to account for TRM working on mazes and 72B duplication working on reasoning.

5. **Think creatively.** Here are some wild ideas — tell me if any have merit:
   - Train a small TRM-style model as a "reasoning coprocessor" that runs alongside the LLM and injects its refined state back
   - Use the LLM's own hidden states as "maze-like" inputs to a recursive solver, where the "maze" is "which tokens should attend to which"
   - Train bidirectional recursion on the prompt only (like Bitune) and measure reasoning improvement on the answer
   - Forget LLMs entirely — scale TRM to 1B params and train it on text-encoded reasoning tasks (not next-token prediction)
   - Use reinforcement learning instead of supervised training for the recurrent core (reward = hard-token accuracy, not average NLL)
   - Train the recurrent core on synthetic "algorithmic" tasks (sorting, graph traversal, arithmetic) then transfer to natural language reasoning

6. **Be brutally honest but don't give up.** If a 7M model can iterate to solve mazes, there MUST be a way to make a larger model iterate to solve harder reasoning problems. The question is how. Don't tell me it's impossible unless you can prove TRM is a special case that genuinely cannot generalize.

## Compute Budget
- 4 × NVIDIA B200 (192GB VRAM each)
- Available for ~1 week
- Can train from scratch (we have the infrastructure)
- Llama 3.1 8B, Gemma 3 27B, Gemma 4 31B available locally
- TRM codebase available at /blue/cis4914/jietao/SeniorProject/RR_TRM/

## Detailed Architecture Code

### TRM Block (the thing that works)
```python
class Block(nn.Module):
    def __init__(self, config, has_mlp=True):
        self.self_attn = Attention(
            hidden_size=config.hidden_size,      # 512
            head_dim=config.hidden_size // config.num_heads,  # 64
            num_heads=config.num_heads,           # 8
            causal=False                          # BIDIRECTIONAL
        )
        if has_mlp:
            self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=4)
        self.has_mlp = has_mlp

    def forward(self, hidden_states, cos_sin):
        # Post-norm
        hidden_states = rms_norm(hidden_states + self.self_attn(hidden_states, cos_sin))
        if self.has_mlp:
            hidden_states = rms_norm(hidden_states + self.mlp(hidden_states))
        return hidden_states

class ReasoningModule(nn.Module):
    def forward(self, hidden_states, input_injection, **kwargs):
        hidden_states = hidden_states + input_injection  # RAW INPUT ADDED
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states
```

### TRM Forward Pass
```python
def forward(self, carry, batch):
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    z_H, z_L = carry.z_H, carry.z_L
    
    # H_cycles-1 WITHOUT gradient
    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):  # 2 cycles no grad
            for _L_step in range(self.config.L_cycles):    # 6 L-steps each
                z_L = self.L_level(z_L, z_H + input_embeddings)
            z_H = self.L_level(z_H, z_L)
    
    # Last H_cycle WITH gradient
    for _L_step in range(self.config.L_cycles):            # 6 L-steps
        z_L = self.L_level(z_L, z_H + input_embeddings)
    z_H = self.L_level(z_H, z_L)
    
    # Detach carry for next sequence
    new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    output = self.lm_head(z_H)
    return new_carry, output
```

Key detail: TRM maintains carry state ACROSS SEQUENCES. z_H and z_L persist between batches and are reset only when the ACT halting mechanism fires. This means the model can accumulate information across multiple inputs.

### TRM Training Details
- **Optimizer:** AdamATan2 or Muon (not standard AdamW)
- **LR:** 1e-4 with 2000 warmup, min_ratio=1.0 (no decay)
- **Batch size:** 768 (global)
- **Epochs:** 100,000
- **Weight decay:** 0.1
- **Puzzle embeddings:** learned per-puzzle-type, LR=1e-2 (10× higher)
- **Data:** 1000 mazes, 8× augmented = 8000 training examples
- **Loss:** stablemax cross-entropy (not standard softmax CE)
- **ACT:** Q-learning with exploration prob=0.1, max_steps=16

### CIRRA Architecture (our latest attempt, failed)
```python
class CIRRA(nn.Module):
    def forward(self, input_ids, labels=None, K=1):
        h = embed(input_ids)
        position_embeddings = rotary_emb(h, position_ids)
        
        # Frozen early trunk (layers 0-11)
        for i in range(12):
            h = base_layers[i](h, position_embeddings=position_embeddings)
        
        u = h.clone()  # Dense anchor for reinjection
        
        # Frozen warm-start (base layers 12-15)
        z = h
        for i in range(12, 16):
            z = base_layers[i](z, position_embeddings=position_embeddings)
        
        # Recurrent replay (K-1 cycles, shared trainable core)
        if K > 1:
            gamma = sigmoid(self.gamma)  # ~0.25 init
            for k in range(K - 1):
                inp = rmsnorm(z + alpha * u)  # Dense reinjection
                cand = inp
                for layer in self.replay_core:  # 4 shared layers (copied from base 12-15)
                    cand = layer(cand, position_embeddings=position_embeddings)
                z = z + gamma * (cand - z) + tiny_mlp(z)  # Relaxation + tiny channel mix
        
        # Frozen late trunk (layers 16-31)
        for i in range(16, 32):
            z = base_layers[i](z, position_embeddings=position_embeddings)
        
        logits = lm_head(norm(z))
        return logits
```

### ARR-PSRT Forward Pass (our first attempt, failed)
```python
for t in range(K):
    bank_scratch = cat([bank, scratchpad])  # (B, 24, D) — 16 bank + 8 scratch
    expert_weights = uniform(1/3, 1/3, 1/3)
    beta_t = expert_betas[:, t]  # t=0: [0.25,0.80,0.10], t=1: [0.10,0.20,0.02]
    
    r_prev = r
    for blk in core_blocks:  # 6 core blocks, each:
        c = cross_attn(norm(r), bank_scratch)  # Re-read compressed input
        r = r + c
        h = r + m_0                             # Add frozen memory
        r = r + self_attn(norm(h))             # Self-attend on sum
        z = norm(r)
        ffn_out = sum(w_e * beta_e * FFN_e(z) for e if beta_e > 0)
        r = r + ffn_out
    
    r = (1-alpha) * r_prev + alpha * r          # Alpha mix (~0.5)
    scratchpad = 0.95 * scratchpad + 0.1 * gated_write(delta_r)

h = Linear(cat([m_0, r]))  # Combine → coda → logits
```

## Dense Baseline Comparison (1.185B params, same data)

```
Step  2K:  PPL 567      Step 30K: PPL 100
Step  4K:  PPL 393      Step 50K: PPL  79
Step  6K:  PPL 306      Step 80K: PPL  71
Step 10K:  PPL 192      Step 100K: PPL 63
Step 14K:  PPL 164      Best: PPL 59.15
Step 20K:  PPL 123
```

ARR v16b at step 14K: PPL 703 (K=2). Dense at step 14K: PPL 164. **4.3× gap.**

## Mathematical Results From Prior GPT Analyses

### 1. Prompt Bank Rank Bottleneck (proven)
For cross-attention with M=16 bank tokens, d_h=64 head dim, 32 heads:
- Max fresh info per reread: 16 × 32 = 512 dimensions
- Model hidden dim: 2048
- **Bank reread is hard-capped at 25% of hidden rank**

### 2. Scratchpad Information Theorem (proven by induction)
S_0 is input-independent. If S_t = g_t(r_0, m_0, B), then S_{t+1} = g_{t+1}(r_0, m_0, B).
Therefore I(S_t; X | r_0, m_0, B) = 0 for all t.
**Scratchpad cannot compensate for bank bottleneck.**

### 3. Alpha Mixing Expansion
With α=0.5: R_2 = 0.25·R_0 + 0.25·U_1 + 0.50·U_2
**Pass 2 controls 50% of final state despite having ~32 dimensions of new info.**

### 4. Shared-Weight Fixed-Point Theory
With tied weights z_{k+1} = T(z_k; u), contraction requires spectral radius ρ < 1.
TRM achieved ρ ≈ 0.84-0.86 on attracting manifold.
**But ρ_perturbation > 1 (globally expansive). Convergence is trajectory-specific, not global.**

### 5. FFN Memory Corruption Model
Attention Jacobian ≈ [[A_r, 0], [0, I]] with |A_r| < 1 (contracts reasoning).
MLP Jacobian ≈ [[I, 0], [0, M_k]] with ρ(M_k) > 1 (expands factual drift).
**Repeating attention shrinks error. Repeating MLP grows knowledge drift.**

### 6. Information Gain Scaling
ΔI(K, w, r) ≈ (1/2) Σ log((η² + σ²)/(η² + ρ^{2(K-1)}·σ²))
- More K: exponential saturation
- More rank: saturates when captured spectrum exhausted
- More width: saturates when no new answer-relevant modes
**Explains rank 64 not beating 32, and delta plateau at -0.51.**

## What GPT Previously Recommended (All Failed)

### Consultation 1: Scratchpad diagnosis
- Identified unbounded accumulation as NaN root cause ✅ (this worked)
- Recommended skip zero-beta experts ✅ (this worked)
- Recommended Phase B1/B2 K=3 ramp ✅ (this worked for stability)

### Consultation 2: K=2 worse than K=1 analysis
- Recommended: separate self-attn from cross-attn to m₀ ❌ (entropy collapsed)
- Recommended: bank size 16→64 ❌ (part of failed v17)
- Recommended: slot-attention scratchpad ❌ (part of failed v17)
- Recommended: per-pass eta gates ❌ (part of failed v17)
- Recommended: gated combine ❌ (part of failed v17)

### Consultation 3: Why ARR loses to dense
- Recommended: dense containment ✅ (correct principle)
- Recommended: near-zero parameter replay ✅ (we built DAR)
- Recommended: attention-dominant, MLP gates near zero ✅ (correct principle)
- Recommended: LoRA on replay layers ❌ (was measuring LoRA quality, not replay)
- Recommended: easy-token KL distillation — not tested in isolation
- Recommended: hard-token routing — neutral effect (-0.07 with or without)

### Consultation 4: CIRRA design
- Recommended: separate trainable core copied from base ❌ (K=1 always best)
- Recommended: dense reinjection every cycle ❌ (helped slightly but K=1 still best)
- Recommended: attention-dominant with tiny MLP ❌ (MLP scale stayed at 0)
- Recommended: K ∈ {1,2,4,8} training ❌ (no K-scaling observed)
- Recommended: contraction penalty ❌ (didn't prevent failure)
- Recommended: always-on control experiment ✅ (correctly identified it as critical)

## What TRM Has That We Never Replicated

| Property | TRM | Our Best LLM Attempt (CIRRA) |
|----------|-----|------------------------------|
| **Bidirectional attention** | Yes (causal=False) | No (causal throughout) |
| **Input injection** | Every L-step (18× per forward) | Every K-cycle (up to 7× per forward) |
| **Two-level hierarchy** | z_H + z_L ping-pong | Single stream z |
| **Gradient truncation** | Only last H-cycle has grad | All cycles have grad |
| **Training epochs** | 100,000 | 10,000 steps |
| **Task type** | Maze solving (requires iteration) | Next-token prediction (mostly doesn't) |
| **Carry state** | Persists across sequences | Resets every sequence |
| **Optimizer** | Muon/AdamATan2 | AdamW |
| **Loss** | Stablemax CE | Standard CE |
| **Data** | 8000 mazes (8× augmented) | Millions of text tokens |
| **Model size** | 7M | 8B (frozen) + ~200M (trainable) |
| **Architecture** | Fully recursive from scratch | Pretrained + bolted-on recursion |

## The Emotional State

We're at the edge of giving up. The data says recursion helps (TRM, 72B duplication). But we can't train it into an LLM no matter what we try. This is either a fundamental impossibility that we should accept, or we're missing something obvious. Help us figure out which.
