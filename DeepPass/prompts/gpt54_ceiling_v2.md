# GPT-5.4 Pro Prompt: The 72% Ceiling is Fundamental — Now What?

## Context

Previously, you diagnosed our 72% SpatialEval ceiling as a **"frozen-decoder readout bottleneck"** and recommended: (1) move memory after prompt, (2) preserve BOS, (3) probe z_H, (4) auxiliary choice loss, (5) cross-attention sidecars.

**We tested ALL of your recommendations plus 50+ additional experiments. Your diagnosis was WRONG.** The ceiling is not a decoder interface problem — it's a fundamental capability limit of the solver+frozen-decoder system. We need a completely new strategy.

## The Architecture (unchanged from previous prompt)

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
        self.proj_in = nn.Linear(4096, d_model, bias=False)   # 4096 → 512
        self.L_self = nn.ModuleList([BidirectionalBlock(d_model, n_heads, ffn_dim) for _ in range(n_L_layers)])
        self.L_cross_H = CrossAttention(d_model, n_heads)
        self.H_self = BidirectionalBlock(d_model, n_heads, ffn_dim)
        self.H_cross_L = CrossAttention(d_model, n_heads)
        self.H_init = nn.Parameter(torch.randn(1, n_memory_slots, d_model) * 0.02)
        self.L_init_scale = nn.Parameter(torch.tensor(0.1))
        self.proj_out = nn.Linear(d_model, 4096, bias=False)
        self.out_norm = RMSNorm(4096)

    def forward(self, prompt_embeddings, K_inner=4, K_outer=3, grad_last_only=True):
        e = self.proj_in(prompt_embeddings)      # (B, T, 512)
        z_L = self.L_init_scale * e              # Start from projected prompt
        z_H = self.H_init.expand(B, -1, -1)     # (B, 32, 512)
        for s in range(K_outer):
            for _ in range(K_inner):
                z_L = z_L + e                    # RAW PROMPT RE-INJECTED every step
                z_L = z_L + self.L_cross_H(z_L, z_H)
                for layer in self.L_self:
                    z_L = layer(z_L)             # Bidirectional self-attention
            z_H = z_H + self.H_cross_L(z_H, z_L)
            z_H = self.H_self(z_H)
        memory = self.out_norm(self.proj_out(z_H))   # (B, 32, 4096)
        return memory
```

### How Memory Tokens Are Fed to the Frozen Decoder:

```python
with torch.no_grad():
    all_emb = frozen_llama.model.embed_tokens(input_ids)  # (B, T, 4096)
prompt_emb = all_emb[:, :prompt_len]
memory = solver(prompt_emb, K_inner=4, K_outer=K)     # (B, 32, 4096)
dec_input = torch.cat([memory, all_emb], dim=1)       # [memory | full sequence]
M = memory.shape[1]
pos_ids = torch.arange(M + T, device=device).unsqueeze(0)
pos_emb = frozen_llama.model.rotary_emb(dec_input, pos_ids)
h = dec_input
for layer in frozen_llama.model.layers:
    h = layer(h, position_embeddings=pos_emb)
h = frozen_llama.model.norm(h)
logits = frozen_llama.lm_head(h)
loss = CE(logits[:, M+prompt_len-1:-1], input_ids[:, prompt_len:])
```

### Key Implementation Details:
1. Memory tokens get RoPE positions 0-31, real prompt starts at 32+
2. Causal attention: memory attends to memory only; prompt attends to all memory + preceding prompt
3. Gradients flow: CE loss → 32 frozen Llama layers → proj_out → solver params
4. BidirectionalBlock uses `is_causal=False` (full bidirectional attention)
5. `grad_last_only=True`: only the last K_outer iteration retains gradient (saves memory)

### Training Details:
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Schedule: cosine with 200-step warmup
- Batch size: 1 (single maze per step)
- Total steps: 2000-3000 (5-8 minutes on B200)
- K-curriculum: K=1 early → K=4 late (random weighted sampling)
- Loss: standard cross-entropy on answer tokens only
- Best-of-N training: screen 10 seeds for 500 steps, continue top 2 for 3000+ steps

### SpatialEval Maze-Nav Benchmark:
- Source: MilaWang/SpatialEval on HuggingFace, 'tqa' split, 'mazenav' subset
- Published at NeurIPS 2024
- 1500 total samples, we split 1000 train / 500 eval
- Multiple choice (A/B/C/D), 25% random chance
- Input: ASCII maze description with #=wall, .=open, S=start, E=end, X=path markers
- **All mazes are 7×7** (tiny), text is 900-1200 chars, zero truncation at our 1500-char limit
- Answer distribution is IMBALANCED: A=39.8%, B=28.0%, C=18.2%, D=14.0%

## What We Tested (60+ Experiments, Exhaustive)

### Your Recommendation #1: Memory Placement (4 positions tested)
| Placement | Best K=1 |
|-----------|----------|
| prefix (before prompt) | 72.6% |
| after_bos | ~71% |
| after_prompt | ~71% |
| before_question | ~71% |

**Result: All placements cap at ~71-72%.** Moving memory doesn't help.

### Your Recommendation #3: Probe z_H
- z_H probe accuracy: **33.7% = random chance**
- z_H does NOT encode the answer. Solver works through **implicit attention steering**.

### Your Recommendation #4: Auxiliary Choice Loss
Added direct CE loss on z_H → A/B/C/D classification.

| lambda_aux | Decoder Acc | Aux Head Acc |
|-----------|------------|-------------|
| 0.1 | 71.2-71.8% | **39.8%** |
| 0.5 | 38-44% | **39.8%** |
| 1.0 | 38.4% | **39.8%** |
| 2.0 | 38.8-72.2% | **39.8%** |

**The aux head ALWAYS predicts 39.8% — that's the class prior for answer "A" (A=39.8%, B=28%, C=18.2%, D=14%).** z_H literally cannot be trained to encode the answer. Higher lambda destroys decoder accuracy without helping aux.

### Your Recommendation #5: Cross-Attention Sidecars (CALM-style)
Bottleneck cross-attention (d_bn=256, 4 heads) at frozen decoder layers.

| Sidecar Layers | d_bottleneck | K=1 Acc |
|---------------|-------------|---------|
| [4,8,12,16] | 256 | **71.8%** (6 seeds, ALL identical) |
| [8,16,24] | 256 | **71.8%** (2 seeds, identical) |
| [2,4,6,8,10,12,14,16] | 256 | **71.8%** (3 seeds) |
| [4,8,12,16] | 512 | **71.8%** (2 seeds) |
| [24,26,28,30] (late) | 256 | 37-60% (WORSE — info can't be processed) |
| Combined: sidecar + prefix | 256 | 71.6-73.2% |

**Every sidecar configuration converges to EXACTLY 359/500 = 71.8%.** The model finds the same 359 solvable samples regardless of injection method. Sidecar gates learned small values (0.003-0.006) and DECREASED during training — the model suppresses the sidecars. Late-layer injection is catastrophically worse because the decoder needs many layers to process injected information.

### Additional Experiment: Pure Logit Bias (bypass decoder entirely)
Solver produces 4-dim bias for A/B/C/D logits, added to frozen decoder's own logits. Gradient path: loss → bias_head → z_H → solver (NO frozen decoder in gradient path).

| Mode | Bias Acc | Notes |
|------|---------|-------|
| Pure (6 configs) | **25-26%** | **BELOW random chance (25%) and below baseline (33.2%)** |
| Hybrid (memory+bias) | 26-32% bias, 38-41% decoder | Two losses compete |

**THE SOLVER CANNOT INDEPENDENTLY CLASSIFY THE ANSWER.** When forced to produce a direct prediction (bypassing the decoder), it performs at random chance. The solver has NO intrinsic maze-solving ability — it only helps the decoder through implicit attention reorganization.

### Additional Experiment: Ensemble Voting (5 independent solvers)
Loaded 5 best independently-trained solvers and majority-voted on each eval sample.

| Metric | Value |
|--------|-------|
| Best individual solver | **72.6%** (363/500) |
| Ensemble-3 majority vote | **72.8%** (364/500) |
| Ensemble-5 majority vote | **72.8%** (364/500) |
| All 5 solvers correct | 64.4% (322/500) |
| All 5 solvers WRONG | **20.8% (104/500)** |
| At least 1 correct | 79.2% (396/500) |

**ERRORS ARE PERFECTLY CORRELATED.** All 5 independently-trained solvers (different seeds, different training trajectories) fail on the EXACT same 104 mazes. Only 74/500 samples have any disagreement between solvers. The theoretical maximum for ANY ensemble is 79.2%.

### Additional Experiment: MoERM (Mixture of External Reasoning Modules)
4-expert mixture with sequence-level soft routing and Perceiver-style fusion.

| Config | Params | K=1 Acc | Router |
|--------|--------|---------|--------|
| Full (4 separate experts) | 321M | 38.6-39.4% | Collapsed to [0.25,0.25,0.25,0.25] |
| Shared core (expert-specific inits) | 283M | 38.8-69.4% | Collapsed to uniform |
| Heterogeneous K (K=1,2,4,8) | 321M | 38.8-39.6% | Near-uniform |

**The router ALWAYS collapses to uniform.** There is no useful routing signal — all mazes look the same to the router. MoERM with 321M params is consistently WORSE than a single 12M solver (39% vs 72%).

### Additional Experiment: Hard Maze Curriculum
Identified the 248/1000 training samples that no solver gets right, then oversampled them.

| Strategy | Hard Weight | K=1 Acc |
|---------|------------|---------|
| Weighted sampling | 0.7 | **28.0%** (WORSE than baseline!) |
| Weighted sampling | 0.5 | **28.0%** |
| Diverse pass (perturbed solver) | — | 38.8% |

**Oversampling hard mazes HURTS.** The solver overfits to specific hard training mazes without learning generalizable spatial reasoning.

### Additional Experiment: Embedding Adapter
Trainable residual adapter (4096-dim MLP) between frozen embeddings and solver.

| Mode | d_hidden | K=1 Acc | Gate |
|------|---------|---------|------|
| Solver-only | 512 | **71.8%** | 0.007 |
| Solver-only | 1024 | 62.2% | 0.010 |
| Shared (decoder too) | 512 | 70.8% | 0.007 |

The adapter gate stays tiny (~0.007) and DECREASES during training — the model learns to suppress the adapter. The frozen embeddings are already optimal for this task.

### Previously Tested (from first prompt)
| Experiment | K=1 Acc |
|-----------|---------|
| 42M solver (d=1024) | 67-72.8% |
| 30M deep (d=768, L=4) | 72.6% |
| mem8 slots | 39.0% |
| mem16-64 slots | 70-72% |
| mem128 slots | 39.0% |
| 10000 training steps | 39.8% (overfits) |
| Gemma 4 31B baseline | 24.5% |
| Gemma 4 31B + solver | 34.6% |

## The Complete Picture: Why 72% is Fundamental

### What the solver actually does
The solver provides **implicit attention steering** — it produces memory tokens that reorganize the prompt's attention patterns in the frozen decoder. This gives the decoder better access to relevant maze information, boosting accuracy from 33.4% to 72%.

### What the solver CANNOT do
1. **Solve mazes** — logit bias at 25% proves the solver has no intrinsic maze-solving ability
2. **Encode the answer** — z_H probe at 33.7% and aux head at 39.8% (class prior) prove the answer is not in z_H
3. **Help with fundamentally hard mazes** — 104/500 eval mazes are impossible for ALL solver variants

### The real bottleneck: decoder reasoning ceiling
The **frozen Llama 8B decoder** is the bottleneck. It has limited spatial reasoning ability (33.4% baseline). The solver's memory tokens help it access maze information more efficiently, boosting it to 72%. But the remaining 28% of mazes require multi-step spatial reasoning that the decoder simply cannot perform, regardless of how the information is presented.

Evidence:
- Sidecar, prefix, combined, adapter all give the same ceiling → injection method doesn't matter
- Solver can't classify on its own → it's not computing the answer
- All solver variants fail on the same mazes → it's a decoder limitation
- The maze is tiny (7×7) and fully visible (no truncation) → not an information availability issue

### The 104 "impossible" mazes
These represent mazes where the correct answer requires reasoning the frozen decoder cannot do:
- 7×7 mazes, all under 1200 chars, zero truncation
- All 5 independently-trained solvers fail on the same ones
- Oversampling during training doesn't help (28%)
- No adapter, sidecar, or MoERM variant cracks them

## Cross-Model Results
- Llama 3.1 8B baseline: 33.4% → +solver: **72%** (our best)
- Gemma 4 31B-IT baseline: 24.5% (WORSE than 8B!) → +solver: 34.6%
- Scale does NOT solve spatial reasoning. Architecture augmentation is the right approach.

## Your Previous Hypotheses — ALL Refuted
1. ~~Frozen decoder can't attend to memory tokens~~ → Sidecar (different interface) = same ceiling
2. ~~Memory token representation mismatch~~ → Adapter (transform embeddings) doesn't help
3. ~~32 tokens can't encode a maze~~ → mem16-64 all same ceiling; logit bias (4 dims!) also fails
4. ~~CE loss too indirect (gradient dilution)~~ → Aux head (direct short gradient) can't learn either
5. ~~Position encoding clash~~ → Placement sweep shows all positions equivalent

## What We Need From You

### 1. Validate or Refute Our New Diagnosis
Is "decoder reasoning ceiling" the right frame? Or is there something we're missing? Search for work on:
- Frozen LLM spatial reasoning limits
- Attention steering vs reasoning augmentation
- When memory-augmented inference hits fundamental ceilings

### 2. The Solver↔Decoder Feedback Loop (Our New Idea)
We've been iterating the SOLVER (K iterations) but not the SOLVER↔DECODER SYSTEM. What if:
1. Solver produces memory tokens → Decoder generates partial answer/reasoning
2. Solver reads decoder's output → produces UPDATED memory tokens
3. Decoder generates better answer with updated guidance

This is **system-level iterative computation** — the thesis is "LLMs Have ADHD" (they need iterative computation). We've been giving iterations to the solver but not to the solver↔decoder feedback loop.

**Is this viable?** How should we implement it? Key questions:
- How does the solver "read" the decoder's output? (attend to decoder hidden states? read generated tokens?)
- What should the decoder generate as intermediate output? (partial answer? cell-by-cell reasoning? confidence signal?)
- How many rounds? Training procedure?

### 3. Chain-of-Thought with Memory Tokens
Instead of outputting a single answer token, let the decoder generate multi-step reasoning:
"Let me trace from S... (3,4) is a wall... going down to (4,4)... The reachable cells are... Answer: B"

The memory tokens would guide this step-by-step generation. This converts the spatial reasoning from a single-token prediction to a sequential reasoning chain, which transformers are better at.

**How to train this?** We only have A/B/C/D labels, not step-by-step reasoning traces. Can the solver learn to guide CoT generation with only answer supervision?

### 4. Making MoERM Work
MoERM's router collapsed because there's no useful routing signal in the prompt embeddings. Can you propose:
- A different routing mechanism that CAN distinguish maze types?
- Expert-specific auxiliary losses that force genuine specialization?
- Pre-training strategies for individual experts?

### 5. Completely New Approaches
We may be fundamentally limited by the "external memory token" paradigm. Are there approaches that:
- Modify the frozen decoder's computation without changing its weights? (e.g., activation patching, steering vectors per layer)
- Use the solver as a "tool" that the decoder can query multiple times during generation?
- Let the solver generate a compact "program" that the decoder executes?

### 6. The User's Original Vision: Adaptive Computation for LLMs
The thesis is about proving that iterative latent computation helps LLMs. The SpatialEval result (+38.6pp with solver) already proves this. But the 72% ceiling limits the impact.

Can you help us frame this result positively while being honest about the ceiling? What claims can we make, and what caveats should we include?

## TRM Interpretability Findings (relevant background)

We separately analyzed a 7M-parameter Tiny Recursive Model (TRM) that uses the same z_H/z_L two-level iterative architecture on direct grid-input maze-solving (not text):

1. **Overconfidence trap:** BCE+StableMax makes model 98.5% confident but 80.6% correct. Representations freeze at wrong fixed point. Brier loss + softmax + monotonicity regularization breaks the trap → 98.1% accuracy.
2. **Probe gap:** MLP probes on TRM's z_H: 95.3% reachable accuracy, but lm_head only achieves 83.2%. Internal representations ARE good, the output head can't fully exploit them.
3. **Parallel soft BFS:** TRM solves 30×30 mazes (needing ~80 BFS steps) in 1-2 ACT steps via bidirectional attention. Reachability info propagates across the entire grid simultaneously.
4. **5-way loss ablation:** Brier+monotonicity+softmax have +18.1pp synergy — individual effects sum to +3.6pp but combination gives +21.7pp.

**Critical difference:** TRM works on GRID DATA directly, not text. It achieves 87.9% on 30×30 mazes. Our SpatialEval solver works on TEXT DESCRIPTIONS of 7×7 mazes and hits 72%. The text→spatial conversion is likely where information is lost.

## Related Approaches (from literature)
- **CALM (ICLR 2024):** Composes frozen LLM with specialist via learned cross-attention between intermediate layers. We tested this (sidecars) — same ceiling.
- **UniR (2025):** Composes reasoning modules via logit addition. Similar to our logit bias — doesn't work.
- **Proxy Tuning (COLM 2024):** Adds logit differences from small models to frozen large models.
- **COCONUT (COLM 2025):** Feeds hidden states back as input embeddings for latent reasoning. **THIS IS CLOSEST TO OUR FEEDBACK LOOP IDEA.**
- **TokMem (ICLR 2026):** Memory tokens placed within (not prepended to) input sequence, with routing.
- **Activation Steering (Representation Engineering, 2024):** Adds learned vectors to intermediate decoder activations to steer behavior. Different from sidecars (continuous steering vs cross-attention).

## Constraints
- 4× NVIDIA B200 (192GB each)
- Frozen Llama 3.1 8B (CANNOT fine-tune decoder — that's the thesis constraint)
- Timeline: advisor meeting Thursday April 10 (2 days)
- Fast iteration (~5 min per training run)
- 60+ experiments already completed, fully documented
- Only American LLMs (Meta Llama, Google Gemma)

## What Success Looks Like
- Break 75% on SpatialEval (even 73-74% consistently would be meaningful)
- OR: a compelling theoretical framework for WHY 72% is the ceiling
- OR: a completely new approach that sidesteps the ceiling
- OR: transfer the solver approach to a different benchmark where the ceiling is higher

**Please search the web** for work on iterative inference, solver-decoder feedback loops, frozen LLM reasoning augmentation, and spatial reasoning benchmarks. We need genuinely new ideas — we've exhausted all the standard approaches.
