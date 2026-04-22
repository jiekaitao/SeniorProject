# GPT-5.4 Pro Prompt: Breaking the 72% SpatialEval Ceiling

## Context

We built a **12M-parameter bidirectional recursive solver** that bolts onto a frozen Llama 3.1 8B and doubles spatial reasoning accuracy on SpatialEval (NeurIPS 2024 maze navigation benchmark) from 33.4% to ~72%. But we've hit a hard ceiling at 72% that nothing breaks.

**We need you to diagnose the bottleneck and propose concrete fixes.** Search the web for relevant work on memory-augmented LLM inference, prefix tuning limitations, and attention over prepended tokens.

## The Architecture (Full Detail)

```
Input maze prompt → Frozen Llama 3.1 8B embeds (4096-dim)
    → Solver(prompt_emb) → 32 memory tokens (4096-dim)
    → Frozen Llama decoder sees: [32 memory tokens | full prompt] → "Answer: B"
```

### SolverCore (exact architecture, ~12M params):

```python
class SolverCore(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ffn_dim=1024,
                 n_L_layers=2, n_memory_slots=32):
        # Projection from LLM embedding space to solver space
        self.proj_in = nn.Linear(4096, d_model, bias=False)   # 4096 → 512

        # L-level: token-aligned workspace (bidirectional self-attn + cross-attn from H)
        self.L_self = [BidirectionalBlock(d_model, n_heads, ffn_dim) for _ in range(n_L_layers)]
        self.L_cross_H = CrossAttention(d_model, n_heads)

        # H-level: global memory slots (self-attn + cross-attn from L)
        self.H_self = BidirectionalBlock(d_model, n_heads, ffn_dim)
        self.H_cross_L = CrossAttention(d_model, n_heads)

        # Learned initial memory slots
        self.H_init = nn.Parameter(torch.randn(1, n_memory_slots, d_model) * 0.02)
        self.L_init_scale = nn.Parameter(torch.tensor(0.1))

        # Output projection: solver space → LLM embedding space
        self.proj_out = nn.Linear(d_model, 4096, bias=False)  # 512 → 4096
        self.out_norm = RMSNorm(4096)

    def forward(self, prompt_embeddings, K_inner=4, K_outer=3, grad_last_only=True):
        e = self.proj_in(prompt_embeddings)      # (B, T, 512)
        z_L = self.L_init_scale * e              # Start from projected prompt
        z_H = self.H_init.expand(B, -1, -1)     # (B, 32, 512) learned init

        for s in range(K_outer):                 # Outer refinement rounds
            for _ in range(K_inner):             # Inner L-level steps
                z_L = z_L + e                    # RAW PROMPT RE-INJECTED every step
                z_L = z_L + self.L_cross_H(z_L, z_H)  # L attends to H
                for layer in self.L_self:
                    z_L = layer(z_L)             # Bidirectional self-attention

            z_H = z_H + self.H_cross_L(z_H, z_L)  # H attends to L
            z_H = self.H_self(z_H)

        memory = self.out_norm(self.proj_out(z_H))   # (B, 32, 4096)
        return memory
```

### How Memory Tokens Are Fed to the Frozen Decoder:

```python
# BYPASS mode: decoder sees [memory | full prompt]
all_emb = frozen_llama.model.embed_tokens(input_ids)  # (B, T, 4096)
prompt_emb = all_emb[:, :prompt_len]

memory = solver(prompt_emb, K_inner=4, K_outer=K)     # (B, 32, 4096)

# Concatenate: [memory tokens | original prompt embeddings | answer embeddings]
dec_input = torch.cat([memory, all_emb[:, :prompt_len], all_emb[:, prompt_len:]], dim=1)

# Generate position IDs for the full sequence (memory gets positions 0-31)
M = memory.shape[1]  # 32
T = dec_input.shape[1]
pos_ids = torch.arange(T, device=device).unsqueeze(0)  # [0, 1, 2, ..., M+T-1]

# Run through frozen Llama's RoPE and all 32 decoder layers
pos_emb = frozen_llama.model.rotary_emb(dec_input, pos_ids)  # (cos, sin) for RoPE
h = dec_input
for layer in frozen_llama.model.layers:
    h = layer(h, position_embeddings=pos_emb)
h = frozen_llama.model.norm(h)
logits = frozen_llama.lm_head(h)

# Loss on answer tokens only (after memory + prompt)
ans_start = M + prompt_len
loss = CE(logits[:, ans_start-1:-1], input_ids[:, prompt_len:])
```

### Key Points About the Current Implementation:
1. Memory tokens occupy RoPE positions 0-31, pushing real prompt to position 32+
2. The frozen decoder's causal attention mask allows memory tokens to attend to each other but NOT to future prompt tokens; prompt tokens CAN attend to all memory tokens
3. Gradients flow from CE loss → through all 32 frozen Llama layers (frozen, no param updates) → through proj_out → through solver → to solver params
4. The solver's BidirectionalBlock uses `is_causal=False` (full bidirectional attention), unlike the decoder's causal attention
5. `out_norm` is RMSNorm applied to the projected 4096-dim memory tokens before they enter the decoder

### Training Details:
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Schedule: cosine with 200-step warmup
- Batch size: 1 (single maze per step)
- Total steps: 2000-3000 (5-8 minutes on B200)
- K-curriculum: K=1 early → K=4 late
- Loss: standard cross-entropy on answer tokens only
- Best-of-N training: screen 10 seeds for 500 steps, continue top 2 for 3000+ steps

### What SpatialEval Maze-Nav Looks Like:
The input is a text description of a maze with walls (#), open spaces (.), start (S), and end (E), plus a multiple-choice question like "Which cell is reachable from S?" with options A/B/C/D. The maze is described in text, not as an image. The prompt is ~1000-1500 tokens.

## The 72% Ceiling — Exhaustive Evidence

We've thrown everything at this and nothing breaks past 72%:

| Experiment | Best K=1 | Notes |
|:---|:---:|:---|
| 12M, best-of-10 seeds, 3000 steps | **72.0%** | Reliable with seed screening |
| 12M, best-of-20 seeds, 5000 steps | **72.4%** | Marginal gain |
| 12M, winning seed, 10000 steps | 39.8% | Massively overfits |
| 42M (d=1024), best-of-10 own seeds | 67.4% | WORSE than 12M |
| 42M, winning 12M seed, 3000 steps | 72.8% | Best single run, but 12M seeds don't transfer |
| 30M_deep (d=768, L=4), winning seed | 72.6% | Same ceiling |
| mem8 slots | 39.0% | Too few |
| mem16 slots | 70.2% best | Same ceiling |
| mem32 slots | 70.6% best | Same ceiling |
| mem64 slots | 69.6% best | Same ceiling |
| mem128 slots | 39.0% | Too many to train |
| Gemma 4 31B-IT + solver | 34.6% | Solver helps but lower ceiling |
| EMA + lower LR + grad accum | 55-68% | Helps stability but not ceiling |

**Key observations:**
1. K=1 ≈ K=2 ≈ K=4 ≈ K=8 on SpatialEval. More solver iterations don't help.
2. mem16 through mem64 all cap at ~70-72%. Memory capacity isn't the bottleneck.
3. Wider (42M) and deeper (30M_deep, L=4) solvers don't beat 12M.
4. Training variance is huge (avg ~40%, best ~70%), but best-of-N screening reliably hits 72%.
5. Longer training (>3000 steps) overfits and degrades.
6. The baseline (no solver, K=0) is 33.4% — so the solver provides +38.6pp.

## What We Think the Bottleneck Might Be

### Hypothesis 1: Frozen decoder can't attend properly to memory tokens
The frozen Llama was never trained to attend to prepended memory tokens. Its positional encoding assumes position 0 starts the real text. Our memory tokens occupy positions 0-31, pushing the real prompt to positions 32+. The decoder may struggle to learn the attention pattern "look at these special tokens for spatial hints."

### Hypothesis 2: Memory token representation mismatch
The solver outputs 4096-dim vectors via a linear projection + RMSNorm, but these may not live in the same manifold as Llama's natural token embeddings. The decoder's attention and FFN layers expect embeddings that follow certain distributional properties (norm, direction, subspace occupancy). Our memory tokens may be "out of distribution" for the frozen decoder.

### Hypothesis 3: 32 tokens can't encode a 30×30 maze
A 30×30 maze has 900 cells. Compressing the reachability structure into 32 tokens (32 × 4096 = 131K parameters) may be fundamentally insufficient. But mem64 doesn't help, so either the compression limit is lower than we think, or the bottleneck is elsewhere.

### Hypothesis 4: CE loss on answer tokens is too indirect
We train the solver by backpropagating from the answer token CE loss through the entire frozen decoder back to the solver. The gradient signal is extremely diluted — it has to flow through 32 frozen transformer layers. The solver may not get a clear enough training signal.

### Hypothesis 5: Position encoding clash
Llama uses RoPE. Our memory tokens get positions 0-31, and the real prompt gets 32+. This might create a systematic bias where the decoder treats memory tokens as "nearby context" rather than "special augmentation."

## What We Need

1. **Diagnose:** Which hypothesis (or combination) is most likely? Search for papers on prefix tuning limitations, memory token attention patterns, and position encoding issues with prepended tokens.

2. **Propose fixes** (ranked by effort and expected impact):
   - What's the simplest thing that might break the ceiling?
   - What's the most principled fix?
   - What would require significant architecture changes?

3. **Specific questions:**
   - Should we use cross-attention injection (like CALM) instead of prefix prepending?
   - Should we add learned position offsets for memory tokens?
   - Should we train a small adapter on the first few decoder layers to help them attend to memory?
   - Would a different memory injection point (mid-layer rather than input) help?
   - Is there a way to make the memory tokens "look more like" real embeddings to the decoder?

4. **Think outside the box.** We may be framing this wrong entirely. Maybe the solver shouldn't produce "memory tokens" at all. Maybe there's a completely different way to inject spatial reasoning into a frozen LLM.

## Constraints
- 4× NVIDIA B200 (192GB each)
- Frozen Llama 3.1 8B (we CANNOT fine-tune the decoder — that defeats the "external module" thesis)
- Timeline: 2 weeks
- Current codebase has working SolverCore, train/eval pipeline, best-of-N training
- The 12M solver trains in ~5 min per run (fast iteration)

## Additional Context

### Cross-Model Results
- Gemma 4 31B-IT baseline: 24.5% (WORSE than Llama 8B's 33.4% — scale doesn't solve spatial reasoning)
- Gemma 4 31B-IT + solver: 34.6% (solver helps but ceiling much lower — Gemma 4's hybrid sliding-window/global attention may not leverage prepended memory tokens well)
- Llama 3.1 8B + solver: 72% (our best)

### TRM Interpretability Findings (relevant background)
We separately analyzed a 7M-parameter Tiny Recursive Model (TRM) that uses the same z_H/z_L two-level iterative architecture on maze-solving tasks:

1. **Overconfidence trap:** Standard training (BCE + StableMax) makes the model 98.5% confident but only 80.6% correct on reachability. Representations freeze at a wrong fixed point. Switching to Brier loss + softmax + monotonicity regularization breaks the trap → 98.1% accuracy.

2. **Probe gap:** MLP probes on z_H reveal 95.3% reachable accuracy, but the output head (lm_head) only achieves 83.2%. The iterative computation builds useful spatial knowledge that the output layer fails to fully exploit.

3. **Parallel soft BFS:** The TRM solves 30×30 mazes (needing ~80 BFS steps) in just 1-2 ACT steps via bidirectional attention that propagates reachability info across the entire grid simultaneously.

4. **5-way loss ablation:** Brier+monotonicity+softmax have +18.1pp synergy — individual effects sum to +3.6pp but the combination gives +21.7pp.

These findings suggest that the SOLVER may also suffer from similar issues — the solver might be building good representations internally but the projection to LLM space + the frozen decoder's attention patterns lose the information.

### Related Approaches (from our literature review)
- **CALM (ICLR 2024):** Composes frozen LLM with specialist via learned cross-attention between intermediate layers — NOT prefix prepending. This avoids our position encoding problem entirely.
- **UniR (2025):** Composes reasoning modules via logit addition, not embedding prepending.
- **Proxy Tuning (COLM 2024):** Adds logit differences from small models to frozen large models.
- **COCONUT (COLM 2025):** Feeds hidden states back as input embeddings for latent reasoning.
- **TokMem (ICLR 2026):** Memory tokens placed within (not prepended to) the input sequence, with routing.

### What We've Explicitly Ruled Out
1. **More solver capacity** — 25M, 30M, 42M all equal or worse than 12M
2. **More iterations** — K=1 through K=8 give identical results
3. **More memory slots** — mem16 through mem64 all cap at ~72%
4. **Longer training** — >3000 steps overfits
5. **Bigger base model** — Gemma 4 31B actually performs worse

### SpatialEval Benchmark Details
- Source: MilaWang/SpatialEval on HuggingFace, 'tqa' split, 'mazenav' subset
- 1500 total samples, we split 1000 train / 500 eval
- Multiple choice (A/B/C/D), 25% is random chance
- Input is a textual maze description (~1000-1500 tokens) with a navigation question
- Published at NeurIPS 2024

### The Question
The solver IS learning something massive (33.4% → 72% = +38.6pp). But what stops it at 72%? Is it:
- How the frozen decoder processes prepended memory tokens?
- The information bottleneck of 32 × 4096 floating-point values?
- Position encoding interference?
- Gradient signal dilution through 32 frozen layers?
- Something else entirely?

**Please search the web** for work on prefix tuning limitations, soft prompt ceilings, memory-augmented inference, and attention patterns over prepended tokens. We need concrete, implementable fixes — not theoretical suggestions.
