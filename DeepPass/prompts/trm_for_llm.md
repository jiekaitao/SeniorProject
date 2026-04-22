# How to Convert a Pretrained LLM Into an Internally-Reasoning Model Using TRM Principles

## What I Need From You

We have a working small-scale recursive reasoning architecture (TRM — Tiny Recursive Model, ~7M params) that successfully trains from scratch with genuine recursion on hard puzzle tasks (maze solving, ARC pattern induction). We want to apply its design principles to a **pretrained large language model** (e.g., Qwen-7B, Llama-8B, or Gemma-27B) to give it **internal recursive reasoning** — multiple passes of iterative refinement before producing output.

Think extensively about:
1. **How** to inject TRM-style recursion into a pretrained LLM's forward pass
2. **What** to train (fine-tune) and what to freeze
3. **Whether** the recursion should be applied to all tokens or adaptively to hard tokens
4. **How** to train with the ACT/halting mechanism on LM tasks
5. **Whether** this should be autoregressive (causal) or if the reasoning passes should be bidirectional
6. **What** the expected compute-accuracy tradeoff looks like

We want mathematical rigor, concrete architecture proposals, and a training recipe. Not hand-waving.

## Research Background

### Our Core Finding
We discovered that repeating transformer layers at inference time helps reasoning but hurts factual recall:
- On a 72B model: IFEval +2.3%, MuSR +1.3%, MATH -6.4%
- Sublayer analysis: **attention repetition helps, FFN/MLP repetition hurts**
- The FFN stores factual associations; re-running it overshoots retrieval basins

### What Failed: Building a Recursive LLM From Scratch (ARR-PSRT)
We spent a week trying to train a 1.7B-parameter recursive transformer from scratch for next-token prediction. Key results:
- The model trained stably after fixing scratchpad overflow issues
- **K=2 (two passes) gave up to 12% better PPL than K=1 within the architecture**
- But the architecture was **4x worse than a standard dense transformer of the same size** on absolute PPL
- Root cause (proven formally): the re-reading mechanism (16-token compressed prompt bank, 3 expert FFNs, scratchpad) consumed too many parameters while providing a rank-bottlenecked information channel

### What Also Failed: Dense Attention Replay (DAR)
We then tried the simplest possible approach: a standard dense transformer with near-zero-parameter attention replay gates on middle layers (24K extra params on 1.185B model). Results after 14K steps:
- **Dense containment confirmed**: K=1 PPL matches pure dense exactly (64.7 vs 65.4)
- **K=2 delta plateaued at +3 to +5** — the attention replay gates learned to reduce K=2 harm by 80% (from +15 to +3) but could never make K=2 actually better than K=1
- Conclusion: for general next-token prediction on web text, attention replay during training doesn't improve average PPL

### What Works: TRM (Tiny Recursive Model)
A ~7M parameter model trained from scratch on hard puzzle tasks (30x30 maze solving, ARC patterns). **This works.** Key architecture:

```python
# Two recurrent states: z_H (high-level) and z_L (low-level)
# Same L_level blocks (shared weights) run repeatedly
# Input is INJECTED every L-step (no lossy compression)

for H_step in range(H_cycles):          # typically 3
    for L_step in range(L_cycles):       # typically 6
        z_L = L_level(z_L, z_H + input_embeddings)  # L refines using H + raw input
    z_H = L_level(z_H, z_L)                          # H refines using L

output = lm_head(z_H)
```

Key design properties:
1. **Full input injection every step** — raw `input_embeddings` added at every L-cycle. No lossy bank compression.
2. **Shared weights** — the SAME `L_level` blocks run every cycle. True fixed-point iteration, not "different passes with different behavior."
3. **Two-level hierarchy** — z_H and z_L ping-pong, creating a structured refinement process.
4. **Non-causal attention** — `causal=False`. Full bidirectional attention within the reasoning loop.
5. **ACT with Q-learning** — adaptive halting decides how many cycles per sequence.
6. **Reduced-MLP ablation worked** — removing MLP from all but the last L-level block (attention-only recursion) performed well.
7. **Gradient truncation** — only the last H_cycle gets gradients. Earlier cycles run with `torch.no_grad()`.

### TRM Results on Mazes
- Config: hidden_size=512, 2 L-layers, H_cycles=3, L_cycles=6, 8 attention heads
- Baseline (no training modifications): 9.10% exact accuracy, 97.94% token accuracy
- Full combination (Brier halting + monotonicity + softmax): **30.80% exact accuracy**
- Displacement-based contraction rate: ~0.84-0.86 (converging along solution manifold despite globally expansive Jacobian)
- The model operates on an **attracting manifold** — the spectral radius ρ > 1 globally, but trajectories contract on the learned subspace

### TRM Config
```yaml
H_cycles: 3      # outer refinement loops
L_cycles: 6      # inner refinement loops per H-step
L_layers: 2      # transformer blocks in L_level
hidden_size: 512
num_heads: 8
expansion: 4      # FFN expansion ratio
puzzle_emb_len: 16
pos_encodings: rope
causal: false     # bidirectional attention
reduced_mlp: false  # ablation: attention-only in early L_level blocks
```

### The TRM Reasoning Module (Full Code)
```python
class ReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, input_injection, **kwargs):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states
```

### The TRM Forward Pass (Full Code)
```python
def forward(self, carry, batch):
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    
    z_H, z_L = carry.z_H, carry.z_L
    
    # H_cycles-1 without gradient (cheaper)
    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings)
            z_H = self.L_level(z_H, z_L)
    
    # Last H_cycle with gradient
    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings)
    z_H = self.L_level(z_H, z_L)
    
    # Output from z_H
    output = self.lm_head(z_H)
    return output
```

### The TRM Block
```python
class Block(nn.Module):
    def __init__(self, config, has_mlp=True):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            causal=False  # BIDIRECTIONAL
        )
        if has_mlp:
            self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.has_mlp = has_mlp

    def forward(self, hidden_states, cos_sin):
        # Post-norm
        hidden_states = rms_norm(hidden_states + self.self_attn(hidden_states, cos_sin))
        if self.has_mlp:
            hidden_states = rms_norm(hidden_states + self.mlp(hidden_states))
        return hidden_states
```

## The Key Question: How to Apply This to a Pretrained LLM

The TRM works because:
1. It's bidirectional (sees all tokens simultaneously)
2. It injects raw input every cycle (no information bottleneck)
3. It uses shared weights (true iterative refinement)
4. It's trained on HARD tasks where recursion genuinely helps
5. It has two-level structure (z_H, z_L) for hierarchical refinement

A pretrained LLM:
1. Is causal/autoregressive (can't attend to future tokens)
2. Has its own strong single-pass representations already
3. Has separate weights per layer (not shared)
4. Is trained on general text (mostly easy next-token prediction)
5. Has a single hidden stream

### Specific Design Questions

1. **Where in the LLM should the recursive loop go?** Between which layers? Should it replace middle layers, or be inserted as an additional processing step? TRM puts the recursion as the ENTIRE model. For an LLM, should it be a "reasoning module" inserted in the middle of the existing layer stack?

2. **Should the reasoning passes be bidirectional?** TRM uses non-causal attention because it's solving puzzles where the full input is available. LLMs are autoregressive. Options:
   - Use causal attention even in the reasoning loop (limited but compatible)
   - Use bidirectional attention in the reasoning loop for prefix tokens, causal for generation
   - Use a separate bidirectional "reasoning pass" then continue causal generation

3. **How to handle the two-level (z_H, z_L) structure?** TRM has two states that ping-pong. A pretrained LLM has one hidden stream. Options:
   - Split the hidden state into z_H and z_L via learned projections
   - Use z_H = the LLM's existing hidden state, z_L = a new learned reasoning state
   - Just use single-level recursion (simpler, but TRM data suggests two-level helps)

4. **What about input injection?** TRM adds raw `input_embeddings` at every L-step. For an LLM, this would mean re-injecting the token embeddings (or early-layer representations) at each refinement cycle. This is what makes TRM avoid the rank bottleneck that killed ARR. But it's expensive — it means running embedding/early layers once, then repeatedly running middle layers with the early-layer output re-injected.

5. **What to train?** Options:
   - Freeze the entire pretrained LLM, add a small trainable reasoning module (like TRM's L_level)
   - Fine-tune middle layers to be weight-shared across cycles (effectively converting 6 unique middle layers into 1 shared layer run 6 times)
   - LoRA adapters on the shared reasoning layers
   - Train from scratch with TRM-style architecture at LLM scale

6. **How to handle adaptive computation (ACT)?** TRM uses Q-learning for halting. For an LLM:
   - Could use per-token halting (some tokens get more refinement)
   - Could use per-sequence halting (entire sequence gets N cycles)
   - The Q-learning approach needs a reward signal — what's the reward for LM?

7. **The gradient truncation trick.** TRM only backpropagates through the LAST H-cycle (`torch.no_grad()` on earlier cycles). This is essential for training deep recursion without exploding memory. Is this sufficient for an LLM, or does it lose too much learning signal?

8. **The reduced-MLP finding.** TRM's ablation showed attention-only blocks work well in the recursion (only last block has MLP). This aligns perfectly with our finding that "attention repetition helps, FFN repetition hurts." Should the recursive module in the LLM be attention-only?

## What We Want From You

1. **A concrete architecture proposal** for injecting TRM-style recursion into a pretrained 7B+ LLM. Not conceptual — give us the forward pass pseudocode, which layers to modify, what to freeze, what to train.

2. **A training recipe** — what data, what loss function, how to handle the ACT mechanism, how many cycles to use, whether to use gradient truncation.

3. **Mathematical analysis** of whether this can provably improve over the base LLM:
   - Under what conditions does the recursive module's fixed point improve on the base model's single-pass output?
   - What is the contraction rate needed for convergence, and how does it relate to the number of cycles?
   - Is there an information-theoretic argument for when recursion helps LM and when it doesn't?

4. **Your honest assessment** of whether this approach has a chance of working for:
   - General next-token prediction (our DAR results suggest: probably not)
   - Reasoning-heavy tasks (TRM results suggest: yes)
   - A hybrid: general LM that's better on hard tokens and neutral on easy ones

5. **What experiments to run first** — the minimal experiment that would tell us if this approach is viable before we invest weeks of GPU time.
