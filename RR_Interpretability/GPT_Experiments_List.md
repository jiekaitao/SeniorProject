I grounded this in your supplied TRM interpretability blueprint and system brief, then checked the official TRM paper/repo and the current primary literature on induction heads, SAEs, circuit tracing, Universal Transformers, and recurrent reasoning. Where your note and the public implementation diverge, I would trust the paper/repo for reproduction details—for example, the released README uses task-specific cycle counts (Sudoku `H_cycles=3, L_cycles=6`; Maze/ARC `H_cycles=3, L_cycles=4`) and its example commands use different optimizer details than the informal summary, so I would pin experiments to the public code/configs rather than the prose summary.   ([arXiv][1])

## 1. Formalize TRM as an autonomous dynamical system

Let the tokenized input be (x\in\mathcal X), sequence length (L), width (d), and flattened state
[
s_u=(y_u,z_u)\in\mathbb R^{D},\qquad D=2Ld
]
for answer-state (y_u) and latent-state (z_u). At the paper level, one outer TRM step is a deterministic map
[
s_{u+1}=F_x(s_u),
]
where (F_x) is the composition of (n) latent updates followed by one answer update. Writing (G_x) for one latent update and (A) for one answer update,
[
G_x(y,z)=(y,; g_\theta(x,y,z)),\qquad
A(y,z)=(a_\theta(y,z),; z),
]
so
[
F_x = A\circ G_x^{,n}.
]
The paper’s recursive training scheme then applies this outer map (T) times per supervision window, with the first (T-1) passes under `torch.no_grad()`, and repeats that supervision window up to (N_{\text{sup}}) times with detach-and-carry-forward. The official TRM paper explicitly emphasizes that the model does **not** rely on a fixed-point theorem and that replacing the explicit recursion with TorchDEQ-style fixed-point iteration was slower and generalized worse. ([arXiv][2])

Two immediate consequences matter for interpretability:

**Autonomy.** There is no explicit iteration index in (F_x). So any “phase-specific” behavior must be encoded in the state (s_u), not in a separate clock.

**Virtual depth.** Unrolling TRM for (N) outer updates gives a depth-(N) computation graph with shared weights. Mechanistically, this means every induction-like circuit in TRM must be represented on a graph whose nodes are indexed by **(module, virtual time)** rather than by distinct physical layers.

That is the right abstraction for the rest of the answer.

---

## 2. Additional hypotheses beyond H1–H6

Your H1–H6 are strong. I would add the following.

### H7. **Lag-spectrum temporal induction**

In standard transformers, induction is mostly a two-layer phenomenon. In TRM, the relevant composition may span **multiple iteration lags**.

Define a causal lag matrix for head (h):
[
M_h(\tau_{\text{src}},\tau_{\text{dst}})
========================================

\mathbb E\left[\Delta \mathcal M ;\middle|; \text{patch head }h\text{ output from }\tau_{\text{src}}\text{ into }\tau_{\text{dst}}\right],
]
where (\mathcal M) is a task metric such as logit difference on the correct digit/path token.

**Prediction.** If TRM has selective temporal induction, (M_h) is not concentrated on (\tau_{\text{dst}}=\tau_{\text{src}}+1); it has multiple ridges at distinct lags, analogous to selective induction over multiple Markov lags. ([arXiv][3])

### H8. **Shared basis, phase-specific coefficients**

Weight tying may not force complete polysemantic chaos. Instead, the model may reuse a mostly shared latent basis (U) while changing coefficients over time:
[
h_\tau(c)\approx U,a_\tau(c),
]
where (c) is the puzzle context and (a_\tau(c)) are iteration-dependent coordinates.

**Prediction.** A pooled SAE with a small iteration embedding or affine adapter should reconstruct nearly as well as fully separate SAEs:
[
\min_{U,{a_\tau}}\sum_\tau |H_\tau-Ua_\tau|_F^2
]
should be close to the sum of per-(\tau) optima.

### H9. **Orbit-basin hypothesis**

Different puzzles may induce trajectories that fall into a small number of orbit families. Hard puzzles spend longer in transient regions before entering the same late-time basin.

Define an orbit distance, e.g. dynamic-time-warped or Procrustes-aligned:
[
d_{\text{orbit}}(s^{(1)},s^{(2)})=
\min_{\phi}\sum_u |s^{(1)}_u-\phi_u(s^{(2)}_u)|_2 .
]

**Prediction.** Orbit clusters correlate more with puzzle type / reasoning mode than with raw token identity.

### H10. **Jacobian bifurcation during training**

Recursive reasoning may emerge when the local Jacobian spectrum changes shape, not merely when certain heads appear.

Let
[
J_x(s)=\frac{\partial F_x}{\partial s}(s),\qquad
\rho_u=\rho!\left(J_x(s_u)\right),\qquad
\kappa_u=|J_x(s_{u+1})-J_x(s_u)|_2.
]

**Prediction.** Early in training and early in inference, (\rho_u) is closer to or above 1 (exploratory / unstable); later in successful trajectories it moves toward near-contractive behavior. This is the dynamical-systems analogue of an induction-head phase transition.

### H11. **Constraint-graph equivariance**

For Sudoku and Maze, useful reasoning should respect graph symmetries.

Let (g) be a symmetry of the puzzle graph and (\Pi_g) the induced permutation on token/state positions. Then approximate equivariance means
[
F_{g\cdot x}(\Pi_g s)\approx \Pi_g F_x(s).
]

**Prediction.** Many late-time features are tied to relational roles (same row, same box, adjacent maze cell, wall, frontier) rather than absolute coordinates.

### H12. **Halting / confidence is a value-of-computation estimator**

If TRM’s recursion is genuinely useful, a scalar readout from (s_\tau) should predict whether another iteration helps.

Let
[
\Delta_\tau = \mathcal L_\tau - \mathcal L_{\tau+1},
]
where (\mathcal L_\tau) is task loss after iteration (\tau).

**Prediction.** There exists a simple readout (r(s_\tau)) such that
[
r(s_\tau)\approx \mathbb E[\Delta_\tau\mid s_\tau].
]
If true, this gives a principled adaptive-depth interpretation.

### H13. **Iterative syndrome decoding**

Rather than copying symbols directly, TRM may implement iterative error-correcting updates on a learned constraint code.

Let (C(s)) be a probe-decoded constraint syndrome (row/column/box violations, path inconsistency, dead ends).

**Prediction.**
[
|C(s_{\tau+1})| < |C(s_\tau)|
]
on most successful trajectories, even when token-level logits change little.

### H14. **Attention is optional on Sudoku but essential for induction-like phenomena**

Because the MLP-Mixer TRM is strongest on Sudoku while attention TRM is strongest on Maze/ARC, induction-like circuits should be strongest in the attention variants and weakest or absent in Sudoku-MLP. That makes Sudoku-MLP the right recurrence-only control. ([arXiv][1])

---

## 3. What can actually be proved?

## 3.1 Unique trajectory: the clean theorem

For fixed (x), suppose (F_x:\mathbb R^D\to\mathbb R^D) is a well-defined deterministic map.

**Theorem 1 (trajectory uniqueness from a fixed initial state).**
For any initial state (s_0), the sequence
[
s_{u+1}=F_x(s_u)
]
is unique.

**Proof.** Immediate by recursion. Once (s_u) is fixed, (s_{u+1}) is fixed because (F_x) is single-valued. ∎

So the answer to “under what Jacobian conditions does the ((y,z)) system admit a unique trajectory?” is:

* **Given a fixed (s_0):** no Jacobian condition is needed beyond forward well-definedness.
* **Independent of initialization / unique attracting orbit:** you need contraction-type assumptions.

### 3.2 Contraction gives uniqueness independent of initialization

Assume (F_x) is differentiable and define
[
L_x = \sup_s |J_x(s)|_2,\qquad J_x(s)=\frac{\partial F_x}{\partial s}(s).
]

**Theorem 2 (global contraction).**
If (L_x<1), then:

1. (F_x) has a unique fixed point (s^\star(x)).
2. Every trajectory converges to (s^\star(x)).
3. Perturbations decay geometrically:
   [
   |s_u-\tilde s_u|_2 \le L_x^u |s_0-\tilde s_0|_2.
   ]

**Proof sketch.** By the mean value inequality,
[
|F_x(s)-F_x(\tilde s)|_2 \le L_x |s-\tilde s|_2.
]
So (F_x) is a contraction. Banach’s fixed-point theorem gives a unique fixed point and geometric convergence. ∎

This is the strongest clean answer to your Jacobian question.

### 3.3 Local contraction is enough for local uniqueness/stability

If
[
\sup_{s\in U}|J_x(s)|_2 < 1
]
on a forward-invariant neighborhood (U), then any trajectory entering (U) has locally unique stable continuation and converges to the unique fixed point in (U).

### 3.4 What if (|J|_2 \ge 1)?

Then none of the above is guaranteed. You can still have:

* a unique trajectory from each initial state,
* multiple attractors,
* limit cycles,
* or sensitive dependence on initial conditions.

That matters because the TRM paper explicitly says fixed-point convergence is not essential and reports that DEQ-style fixed-point enforcement hurt performance. So the relevant mechanistic regime may be **useful transient computation**, not contraction to a single attractor. ([arXiv][2])

### 3.5 No-clock theorem for autonomous recurrence

**Theorem 3 (no intrinsic clock).**
Assume (s_{u+1}=F_x(s_u)) with no explicit (u)-dependence. If (s_u=s_v) for some (u<v), then
[
s_{u+t}=s_{v+t}\qquad \forall t\ge 0.
]

**Proof.** By induction:
[
s_{u+1}=F_x(s_u)=F_x(s_v)=s_{v+1},
]
and so on. ∎

**Interpretation.** If the same head behaves like a previous-token head at one iteration and like an induction head at another, that difference is not because of an external time index. It must come from the orbit entering different state regions.

That is a strong conceptual difference from standard “spatial” induction circuits.

---

## 3.6 Can you prove a bound on Sudoku iterations?

Unconditionally, not in any interesting general sense.

Generalized Sudoku is NP-complete, so absent additional structure you should not expect a universal polynomial-time bound for a generic learned local-update solver over arbitrary (n^2\times n^2) Sudoku families. ([Oxford Computer Science][4])

But you **can** prove conditional bounds under progress assumptions.

### Cell-potential bound

Let
[
\Phi_{\text{cell}}(s)=#{\text{unsolved cells at state }s}.
]

**Proposition 1.**
If every TRM outer step that has not yet solved the puzzle fixes at least one new cell and never unfixes a previously correct cell, then for a 9x9 puzzle with (k) clues,
[
U_{\max}\le 81-k.
]

**Proof.** Initially (\Phi_{\text{cell}}(s_0)\le 81-k). Each successful step decreases (\Phi_{\text{cell}}) by at least 1 and it is bounded below by 0. ∎

### Candidate-potential bound

Let (C_i(s)\subseteq{1,\dots,9}) be the candidate set for cell (i), decoded either exactly from the model state or from an external candidate probe. Define
[
\Phi_{\text{cand}}(s)=\sum_{i=1}^{81}(|C_i(s)|-1).
]

If clue cells are fixed and unsolved cells begin with all 9 candidates, then
[
\Phi_{\text{cand}}(s_0)\le 8(81-k).
]

**Proposition 2.**
If each TRM outer step removes at least (m\ge 1) invalid candidates and never reintroduces any removed candidate, then
[
U_{\max}\le \left\lceil \frac{\Phi_{\text{cand}}(s_0)}{m}\right\rceil
\le
\left\lceil \frac{8(81-k)}{m}\right\rceil .
]

**Proof.** Same descent argument. ∎

These are the right kinds of theorem statements to use in a paper: they are true, interpretable, and make explicit which assumptions are algorithmic rather than neural.

---

## 3.7 Computational complexity: TRM with (N) iterations vs. an (N)-layer transformer

Let (L) be sequence length and (d) width.

For one self-attention block, ignoring constants and head splits,
[
\text{cost}_{\text{block}} = \Theta(L^2 d + L d^2).
]

So a TRM with (N) virtual steps has forward cost
[
\text{cost}*{\text{TRM}}(N)=\Theta!\big(N(L^2 d + L d^2)\big),
]
which is the same asymptotic inference cost as an untied (N)-layer transformer:
[
\text{cost}*{\text{std}}(N)=\Theta!\big(N(L^2 d + L d^2)\big).
]

The real asymptotic difference is **parameter complexity**:

* standard (N)-layer transformer:
  [
  \Theta(N d^2)
  ]
* tied TRM:
  [
  \Theta(d^2)
  ]
  up to embeddings/output heads and constant-factor architectural extras.

So:

**Conclusion 1.** With fixed iteration budget (N), TRM and a standard (N)-layer transformer are in the same broad runtime class; TRM is basically an (N)-layer transformer with hard weight sharing.

**Conclusion 2.** Mechanistically, the difference is not asymptotic inference cost but **description length** and **the need for time-mediated reuse of the same parameters**.

This is exactly why induction-like circuits, if present, should look different.

As a comparison point, the Universal Transformer is also an iterative self-attentive architecture with shared weights and dynamic halting; the original paper argues that under suitable assumptions it can achieve stronger algorithmic expressivity than fixed-depth transformers. For your comparison framework, UT is the right control for “weight sharing + recurrence” without TRM’s explicit (y/z) state split and deep supervision. ([arXiv][5])

---

## 4. Concrete code for adapting MI tooling to TRM

Below I’ll keep the code architecture-agnostic enough to survive small implementation differences.

## 4.1 Hook every virtual iteration

```python
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TraceStore:
    save: set[str]
    cache: dict = field(default_factory=lambda: defaultdict(dict))
    patch: dict = field(default_factory=dict)

    def tap(self, name: str, tau: int, x: torch.Tensor) -> torch.Tensor:
        if name in self.save:
            self.cache[name][tau] = x.detach().to(torch.float32).cpu()
        fn = self.patch.get((name, tau), None)
        if fn is not None:
            x = fn(x)
        return x
```

Paper-level wrapper:

```python
class InstrumentedTRM(nn.Module):
    """
    Wrap your public TRM model so every recursive application gets a virtual time index tau.
    Adapt self.trm.latent_block / self.trm.answer_block to your real code.
    """
    def __init__(self, trm: nn.Module):
        super().__init__()
        self.trm = trm

    def latent_step(self, x, y, z, tau: int, trace: TraceStore | None):
        if trace is not None:
            y = trace.tap("y_pre_latent", tau, y)
            z = trace.tap("z_pre_latent", tau, z)

        z = self.trm.latent_block(x=x, y=y, z=z)   # adapt signature

        if trace is not None:
            z = trace.tap("z_post_latent", tau, z)
        return y, z

    def answer_step(self, y, z, tau: int, trace: TraceStore | None):
        if trace is not None:
            y = trace.tap("y_pre_answer", tau, y)
            z = trace.tap("z_pre_answer", tau, z)

        y = self.trm.answer_block(y=y, z=z)        # adapt signature

        if trace is not None:
            y = trace.tap("y_post_answer", tau, y)
        return y, z

    def outer_step(self, x, y, z, n: int, tau0: int, trace: TraceStore | None):
        tau = tau0
        for _ in range(n):
            y, z = self.latent_step(x, y, z, tau=tau, trace=trace)
            tau += 1
        y, z = self.answer_step(y, z, tau=tau, trace=trace)
        tau += 1
        return y, z, tau

    def forward(self, input_ids, y0, z0, n: int = 6, T: int = 3,
                trace: TraceStore | None = None):
        x = self.trm.embed_input(input_ids)
        y, z = y0, z0
        tau = 0

        with torch.no_grad():
            for _ in range(T - 1):
                y, z, tau = self.outer_step(x, y, z, n=n, tau0=tau, trace=trace)

        y, z, tau = self.outer_step(x, y, z, n=n, tau0=tau, trace=trace)

        logits = self.trm.unembed(y)
        if trace is not None:
            trace.cache["logits"][tau - 1] = logits.detach().cpu()

        return {"y": y.detach(), "z": z.detach(), "logits": logits, "trace": trace}
```

If you instrument the released repo more literally, use the repo’s task-specific cycle structure (`H_cycles`, `L_cycles`) and assign one (\tau) to each application of the tied inner block. The repo README confirms that these cycle counts differ by task, so do **not** hard-code one universal unroll schedule across Sudoku and Maze. ([GitHub][6])

## 4.2 Iteration-resolved logit lens

```python
@torch.no_grad()
def iteration_logit_lens(state_cache: dict[int, torch.Tensor],
                         unembed: nn.Module,
                         final_norm: nn.Module | None = None):
    out = {}
    for tau, h in state_cache.items():
        h_tau = final_norm(h) if final_norm is not None else h
        logits = unembed(h_tau)
        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
        out[tau] = {
            "logits": logits.cpu(),
            "entropy": entropy.cpu(),
            "top1": probs.argmax(dim=-1).cpu(),
            "confidence": probs.max(dim=-1).values.cpu(),
        }
    return out
```

A tuned-lens version:

```python
class TauTunedLens(nn.Module):
    def __init__(self, n_tau: int, d_model: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d_model).repeat(n_tau, 1, 1))
        self.b = nn.Parameter(torch.zeros(n_tau, 1, d_model))

    def forward(self, h: torch.Tensor, tau: int) -> torch.Tensor:
        return h @ self.A[tau].transpose(-1, -2) + self.b[tau]
```

Train (A_\tau,b_\tau) by KL-minimizing against final logits or against the target distribution. Tuned lenses are especially useful when the hidden-state basis drifts across iterations. ([GitHub][7])

## 4.3 Iteration-specific vs. pooled SAEs

Simple TopK SAE:

```python
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_latents: int, k: int):
        super().__init__()
        self.enc = nn.Linear(d_in, n_latents, bias=False)
        self.dec = nn.Linear(n_latents, d_in, bias=False)
        self.k = k

    def forward(self, x: torch.Tensor):
        pre = self.enc(x)
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
        x_hat = self.dec(acts)
        return x_hat, acts, pre
```

Pooled SAE with iteration embedding:

```python
class PooledTopKSAE(nn.Module):
    def __init__(self, d_in: int, n_latents: int, n_tau: int, k: int, d_tau: int = 32):
        super().__init__()
        self.tau_emb = nn.Embedding(n_tau, d_tau)
        self.enc = nn.Linear(d_in + d_tau, n_latents, bias=False)
        self.dec = nn.Linear(n_latents, d_in, bias=False)
        self.k = k

    def forward(self, x: torch.Tensor, tau: torch.Tensor):
        e = self.tau_emb(tau)
        inp = torch.cat([x, e], dim=-1)
        pre = self.enc(inp)
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
        x_hat = self.dec(acts)
        return x_hat, acts, pre
```

Training loop:

```python
def sae_loss(x, x_hat, acts, beta_mse=1.0):
    recon = beta_mse * F.mse_loss(x_hat, x)
    l0 = (acts > 0).float().sum(dim=-1).mean()
    return recon, {"recon": recon.item(), "l0": l0.item()}
```

For serious runs, use SAELens as the trainer and feed it activations from your own TRM dataloader / activation store. NNsight is the easiest official library for model-agnostic activation capture; SAELens is the mature training stack for SAE sweeps. ([GitHub][8])

## 4.4 Temporal activation patching

```python
@torch.no_grad()
def temporal_patch(model: InstrumentedTRM,
                   clean_batch: dict,
                   corrupt_batch: dict,
                   hook_name: str,
                   src_tau: int,
                   dst_tau: int,
                   metric_fn):
    clean_trace = TraceStore(save={hook_name})
    clean_out = model(**clean_batch, trace=clean_trace)
    clean_act = clean_out["trace"].cache[hook_name][src_tau].to(clean_out["logits"].device)

    patch_trace = TraceStore(save=set())
    patch_trace.patch[(hook_name, dst_tau)] = lambda x: clean_act

    corrupt_out = model(**corrupt_batch, trace=None)
    patched_out = model(**corrupt_batch, trace=patch_trace)

    return metric_fn(corrupt_out["logits"], patched_out["logits"])
```

Masked variant:

```python
def make_masked_patch(clean_act, mask):
    def fn(x):
        return x * (1 - mask) + clean_act * mask
    return fn
```

This lets you patch:

* same hook, different (\tau),
* different hook families (`z_post_latent` into `y_pre_answer` if dimensions line up),
* whole states, heads, or selected cells/tokens.

Use this to construct the lag matrix (M(\tau_{\text{src}},\tau_{\text{dst}})).

## 4.5 Iteration-resolved prefix matching

For the attention TRM, create repeated synthetic sequences:

```python
def repeated_sequence_batch(B: int, half_len: int, vocab_size: int, device):
    first = torch.randint(0, vocab_size, (B, half_len), device=device)
    return torch.cat([first, first], dim=1)
```

Then score attention on the “token after previous occurrence” target:

```python
def prefix_matching_score(attn: torch.Tensor) -> torch.Tensor:
    """
    attn: [B, H, 2T, 2T] for a repeated-sequence batch x[:T] + x[:T]
    returns: [H]
    """
    B, H, S, _ = attn.shape
    T = S // 2

    q_positions = torch.arange(T + 1, 2 * T, device=attn.device)
    k_positions = torch.arange(1, T, device=attn.device)

    # query at T+t should attend to key t+1
    block = attn[:, :, q_positions[:, None], k_positions[None, :]]  # [B,H,T-1,T-1]
    diag = block.diagonal(dim1=-2, dim2=-1)                         # [B,H,T-1]
    return diag.mean(dim=(0, 2))
```

## 4.6 Iteration-resolved copying score

```python
def mean_target_logit(logits: torch.Tensor, targets: torch.Tensor, positions: torch.Tensor):
    selected = logits[:, positions]  # [B, P, V]
    good = selected.gather(-1, targets[:, positions].unsqueeze(-1)).squeeze(-1)
    return good.mean()

@torch.no_grad()
def head_copying_score(model, batch, tau, hook_name, head_idx, positions):
    base_logits = model(**batch, trace=None)["logits"]

    def zero_head(x):
        x = x.clone()
        x[:, :, head_idx, :] = 0.0
        return x

    patch_trace = TraceStore(save=set())
    patch_trace.patch[(hook_name, tau)] = zero_head
    patched_logits = model(**batch, trace=patch_trace)["logits"]

    return (
        mean_target_logit(base_logits, batch["targets"], positions)
        - mean_target_logit(patched_logits, batch["targets"], positions)
    )
```

This is the attention-model analogue of the standard copying score; in TRM you compute it **per virtual iteration**.

## 4.7 Task-native analogues of induction metrics

Synthetic repeated-sequence scores are only half the story. For Sudoku and Maze, define native metrics.

For Sudoku, let (c) be an unsolved cell, (S(c)) the set of solved support cells in its row/column/box, and (v(j)) the digit at support cell (j). Then define an invalid-digit suppression score
[
\text{SCS}*h(\tau)=
\mathbb E\Big[
\sum*{j\in S(c)} A_h^\tau(c,j);
\big(\ell^{\text{abl}}*{c,v(j)}-\ell^{\text{clean}}*{c,v(j)}\big)
\Big].
]
Positive values mean the head helps suppress forbidden digits.

For Maze, if (N(i)) is the graph neighborhood of cell (i), define
[
\text{AdjCopy}*h(\tau)=
\mathbb E\Big[
\sum*{j\in N(i)} A_h^\tau(i,j);
\Delta\ell_{i,\text{next-step}(j)}
\Big].
]
These are much more important than raw prefix matching for adjudicating H2 vs. H3.

---

## 5. The complete 4-week experiment plan

I would not spend the first month primarily on ARC. The official repo shows Sudoku and Maze are much cheaper and cleaner to instrument, while ARC training is materially longer. Use Sudoku and Maze as the core science loop, then do ARC only as a stretch validation. ([GitHub][6])

I would run **three core TRM models**:

1. **Sudoku-MLP TRM** — recurrence-only control
2. **Sudoku-attention TRM** — same task, attention-enabled
3. **Maze-attention TRM** — strongest induction-like candidate

Plus two baselines:
4. **Untied transformer baseline**
5. **Universal Transformer baseline**

All under matched tokenizer / data / width / optimizer as much as possible.

The compute estimates below are **planning estimates**, anchored by the public repo’s L40S/H100 runtimes. On 4×B200, model training is not the bottleneck; instrumentation and analysis volume are. ([GitHub][6])

---

## Week 1 — Reproduction, instrumentation, checkpoints

### Experiment 1. Exact reproduction and dense checkpoint sweep

**Question.** Where in training do recursive behaviors and any induction-like signatures first appear?
**Method.** Reproduce the three core TRM models above with dense early checkpointing. Save every step for the first 500 steps, every 100 to 10k, then every 1k.
**Code excerpt.**

```python
if step < 500 or step % 100 == 0 or (step < 10000 and step % 100 == 0) or step % 1000 == 0:
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        ckpt_dir / f"step_{step:07d}.pt"
    )
```

**Expected compute.** Roughly 1–2 days wall-clock total for Sudoku + Maze TRM training runs on your setup, plus I/O overhead.
**Positive result.** Clear emergence window for recursive dynamics / circuit signatures.
**Negative result.** Either circuits form too early to resolve or there is no sharp emergence, which itself would distinguish TRM from standard IH phase transitions.
**Why grounded.** The official repo reports hours-scale Sudoku/Maze training, not weeks. ([GitHub][6])

### Experiment 2. Instrumented virtual-time trace capture

**Question.** Can you reliably capture every virtual step (\tau) for all major state types?
**Method.** Integrate `InstrumentedTRM` and cache `y`, `z`, per-head attention maps, block outputs, and final logits.
**Code excerpt.**

```python
trace = TraceStore(save={
    "y_pre_latent", "z_pre_latent", "z_post_latent",
    "y_pre_answer", "z_pre_answer", "y_post_answer"
})
out = model(input_ids=batch["input_ids"], y0=y0, z0=z0, trace=trace)
torch.save(out["trace"].cache, out_path)
```

**Expected compute.** 2–6 B200-hours for the full implementation/debug cycle; ongoing extraction costs dominate.
**Positive result.** Stable trace schema across models/tasks.
**Negative result.** If hook granularity is too coarse, you refine the wrapper before doing any SAE work.

### Experiment 3. Behavior-by-iteration curves

**Question.** Does task accuracy improve mostly in the first few iterations, as reported in the follow-up ARC analysis, or do Sudoku/Maze show deeper iterative gains?
**Method.** Decode predictions after every (\tau) with the logit lens and compute per-iteration task accuracy.
**Code excerpt.**

```python
lens = iteration_logit_lens(trace.cache["y_post_answer"], model.trm.unembed)
acc_tau = {
    tau: (lens[tau]["top1"] == batch["targets"].cpu()).float().mean().item()
    for tau in lens
}
```

**Expected compute.** 1–2 B200-hours per 100k-example evaluation sweep.
**Positive result.** Meaningful late-(\tau) gains support genuine iterative computation.
**Negative result.** Near-saturation at the first step supports the “effectively shallow recursion” concern.
**Why grounded.** Roye-Azar et al. report that much of final ARC performance appears at the first recursion step. ([arXiv][9])

### Experiment 4. Jacobian spectrum and orbit geometry

**Question.** Are successful trajectories contractive, marginally stable, or expansive at different (\tau)?
**Method.** Estimate (|J_x(s_\tau)|_2), leading eigenvalue proxies, and step-to-step state distances.
**Code excerpt.**

```python
def F_state(s_flat):
    y, z = unpack_state(s_flat)
    y1, z1, _ = model.outer_step(x_embed, y, z, n=n, tau0=0, trace=None)
    return pack_state(y1, z1)

rho = power_iteration_jacobian(F_state, pack_state(y_tau, z_tau))
dist = (pack_state(y_tau1, z_tau1) - pack_state(y_tau, z_tau)).norm(dim=-1).mean()
```

**Expected compute.** 4–8 B200-hours per full checkpoint family if done on a few hundred puzzles.
**Positive result.** Late-time contraction with early-time expansion supports H10.
**Negative result.** Flat spectral behavior suggests recursive benefit comes from simple repeated refinement, not dynamical phase structure.

---

## Week 2 — Lenses, probes, SAEs, symmetry

### Experiment 5. Iteration-resolved logit lens / tuned lens

**Question.** Does TRM progressively assemble solutions, and which cells/tokens resolve first?
**Method.** Run raw logit lens and tau-specific tuned lens; compute per-cell entropy trajectories.
**Code excerpt.**

```python
lens_out = iteration_logit_lens(trace.cache["y_post_answer"], model.trm.unembed)
entropy_tau = {tau: lens_out[tau]["entropy"].mean().item() for tau in lens_out}
```

**Expected compute.** 1–3 B200-hours.
**Positive result.** Monotone or near-monotone entropy drop supports H5.
**Negative result.** Flat or oscillatory entropy implies more hidden-state computation than visible partial-solution assembly.

### Experiment 6. Linear probes for constraint syndromes

**Question.** Is the latent state encoding explicit Sudoku/Maze constraint information before final answer decoding?
**Method.** Train linear probes from (z_\tau) and (y_\tau) to predict row/column/box violations, candidate masks, path frontier, dead ends, graph distance to goal.
**Code excerpt.**

```python
probe = nn.Linear(d_model, n_probe_targets).cuda()
loss = F.binary_cross_entropy_with_logits(probe(z_feats), probe_targets.float())
```

**Expected compute.** 2–4 B200-hours for all probe families.
**Positive result.** Strong decodability of syndromes supports H13.
**Negative result.** Weak syndrome probes but strong final logits suggest highly distributed or nonlinear computation.

### Experiment 7. Iteration-specific SAE sweep

**Question.** Are features strongly phase-specific?
**Method.** Train separate SAEs on selected (\tau) slices. Start with width sweep ({4k,16k,64k}), (k\in{32,64,128}).
**Code excerpt.**

```python
sae = TopKSAE(d_in=d_model, n_latents=16384, k=64).cuda()
for x in act_loader_tau7:
    x_hat, acts, _ = sae(x.cuda())
    loss, stats = sae_loss(x.cuda(), x_hat, acts)
    loss.backward(); opt.step(); opt.zero_grad()
```

**Expected compute.** 2–8 B200-hours per medium SAE; 1–2 days for the full sweep.
**Positive result.** Different decoder dictionaries across (\tau) support H4/H6.
**Negative result.** Near-identical dictionaries across (\tau) support H8 shared-basis reuse.

### Experiment 8. Pooled SAE with iteration embedding

**Question.** Are iteration differences mostly basis changes or coefficient changes?
**Method.** Train `PooledTopKSAE` on concatenated activations from all (\tau).
**Code excerpt.**

```python
pooled = PooledTopKSAE(d_in=d_model, n_latents=16384, n_tau=n_virtual_steps, k=64).cuda()
x_hat, acts, _ = pooled(x.cuda(), tau.cuda())
```

**Expected compute.** 4–10 B200-hours.
**Positive result.** If pooled SAE matches separate SAEs, H8 is favored.
**Negative result.** Large reconstruction gap means truly distinct per-(\tau) feature sets.

### Experiment 9. Seed-stability and subspace overlap

**Question.** Are discovered features stable enough to make circuit claims?
**Method.** Train 2–3 SAEs per condition and compare decoder subspaces via Procrustes/CCA rather than one-to-one feature matching.
**Code excerpt.**

```python
U1 = F.normalize(sae1.dec.weight.data, dim=0)
U2 = F.normalize(sae2.dec.weight.data, dim=0)
overlap = torch.linalg.svdvals(U1.T @ U2).mean().item()
```

**Expected compute.** 1 extra day if done on the main conditions only.
**Positive result.** High subspace overlap makes causal feature interventions credible.
**Negative result.** Use subspace-level, not feature-level, claims in the paper.

### Experiment 10. Symmetry-equivariance test

**Question.** Does TRM reason in relational coordinates?
**Method.** Apply Sudoku row/column/digit permutations or Maze rotations/reflections and test
[
|F_{g\cdot x}(\Pi_g s)-\Pi_g F_x(s)|.
]
**Code excerpt.**

```python
err = (state_next_transformed - permute_state(state_next_original, g)).pow(2).mean().sqrt()
```

**Expected compute.** 2–4 B200-hours.
**Positive result.** Supports H11 and makes sparse features easier to interpret.
**Negative result.** Suggests strong absolute-position anchoring.

---

## Week 3 — Causal interventions and temporal circuit discovery

### Experiment 11. Temporal patch matrix

**Question.** Which virtual times causally feed which later virtual times?
**Method.** Build a full matrix (M(\tau_{\text{src}},\tau_{\text{dst}})) by patching hidden states or head outputs.
**Code excerpt.**

```python
M = torch.zeros(n_tau, n_tau)
for src in range(n_tau):
    for dst in range(n_tau):
        M[src, dst] = temporal_patch(
            model, clean_batch, corrupt_batch,
            hook_name="z_post_latent",
            src_tau=src,
            dst_tau=dst,
            metric_fn=sudoku_logit_diff_metric
        )
```

**Expected compute.** 4–12 B200-hours, depending on matrix size and example count.
**Positive result.** Off-diagonal ridges imply lagged temporal circuits (H7).
**Negative result.** Mostly local/diagonal effects imply shallow refinement.

### Experiment 12. Head and channel ablation over virtual time

**Question.** Are there canonical induction-like heads, or is causality diffuse?
**Method.** Zero/mean ablate specific heads/channels at each (\tau); compare with task-native neighbor/constraint ablations.
**Code excerpt.**

```python
def ablate_head(x, head_idx):
    x = x.clone()
    x[:, :, head_idx, :] = 0.0
    return x
trace.patch[("attn_out", tau)] = lambda x: ablate_head(x, head_idx)
```

**Expected compute.** 4–8 B200-hours.
**Positive result.** Sparse causal peaks support discrete circuit claims.
**Negative result.** Diffuse effects point toward distributed message passing (H2).

### Experiment 13. Prefix-matching / copying scores by virtual time

**Question.** Does the attention TRM ever exhibit classical IH signatures on synthetic repeated sequences?
**Method.** Compute prefix-matching and copying scores at every (\tau).
**Code excerpt.**

```python
pm_tau = prefix_matching_score(attn_cache[tau])
cp_tau = head_copying_score(model, rep_batch, tau, "attn_out", head_idx, positions)
```

**Expected compute.** 2–4 B200-hours.
**Positive result.** Strong late-(\tau) synthetic IH signatures support H1/H7.
**Negative result.** If Maze still works while these scores are weak, H2/H3 becomes more plausible than literal token-style induction.

### Experiment 14. Task-native constraint-copy / adjacency-copy metrics

**Question.** If there are no classical IHs, is there a domain-specific analogue?
**Method.** Compute `SCS_h(τ)` for Sudoku and `AdjCopy_h(τ)` for Maze.
**Code excerpt.**

```python
score = 0.0
for (cell, support_cell, bad_digit) in constraint_examples:
    score += attn[b, h, cell, support_cell] * (
        logits_abl[b, cell, bad_digit] - logits_clean[b, cell, bad_digit]
    )
score /= len(constraint_examples)
```

**Expected compute.** 2–6 B200-hours.
**Positive result.** Strong native scores with weak synthetic IH scores strongly favors H3.
**Negative result.** Weak native and synthetic scores push you toward distributed recurrence rather than discrete copy circuits.

### Experiment 15. Attribution patching / edge attribution on the virtual graph

**Question.** Can you extract a sparse temporal circuit graph without exhaustive patching?
**Method.** Use gradient-based attribution patching or edge attribution over the unrolled virtual-time graph.
**Code excerpt.**

```python
clean = clean_act.requires_grad_(False)
corr = corr_act.requires_grad_(True)
metric = logit_diff(model_out, target_info)
metric.backward()
approx_effect = ((corr.grad) * (clean - corr)).sum().item()
```

**Expected compute.** 2–5 B200-hours once the wrapper is stable.
**Positive result.** Sparse temporal graph enables ACDC-/AutoCircuit-style pruning.
**Negative result.** Use attribution only as hypothesis generator, then validate with explicit patching.
**Why grounded.** Attribution patching is standard scalable practice; circuit-tracing methods and graph-style causal decompositions are now mature in the open-source ecosystem. ([Transformer Circuits][10])

---

## Week 4 — Cross-architecture comparison and validation

### Experiment 16. Matched-compute baseline comparison

**Question.** Are the discovered circuits specific to TRM or generic to any model with enough depth/compute?
**Method.** Train:

* untied transformer with same virtual depth,
* Universal Transformer with shared weights and attention,
* TRM.

Match tokenizer, task data, width, optimizer, and approximate training FLOPs.
**Code excerpt.**

```python
models = {
    "trm": trm_model,
    "untied": untied_transformer,
    "ut": universal_transformer,
}
for name, model in models.items():
    train_model(model, same_data_cfg, same_opt_cfg, same_budget_cfg)
```

**Expected compute.** 2–4 days total wall-clock for the whole comparison set.
**Positive result.**

* TRM-specific temporal circuits -> explicit state split (y/z) matters.
* UT matches TRM -> weight sharing + recurrence are sufficient.
* Untied transformer differs -> temporal reuse, not just depth, is the key distinction.
  **Negative result.** Similar circuits across all three would imply convergence to a common algorithmic solution.

### Experiment 17. Cross-task feature overlap

**Question.** Are there universal iterative-reasoning features shared between Sudoku and Maze?
**Method.** Train pooled SAEs or probes on one task, test transfer to the other. Compute subspace overlap and intervention transfer.
**Code excerpt.**

```python
transfer_mse = F.mse_loss(sae_sudoku(x_maze)[0], x_maze)
subspace = torch.linalg.svdvals(
    F.normalize(W_sudoku, dim=0).T @ F.normalize(W_maze, dim=0)
).mean()
```

**Expected compute.** 4–8 B200-hours.
**Positive result.** Supports a universality hypothesis for iterative constraint solving.
**Negative result.** Suggests task-specific algorithm families.

### Experiment 18. Causal validation by feature steering

**Question.** Are SAE features merely correlated, or actually causal?
**Method.** Add or remove specific SAE features at selected (\tau) and test whether they change only the predicted concept (row conflict, path frontier, etc.).
**Code excerpt.**

```python
def steer_feature(x, sae, feat_idx, delta):
    x_hat, acts, pre = sae(x)
    acts = acts.clone()
    acts[..., feat_idx] += delta
    return sae.dec(acts)

trace.patch[("z_post_latent", tau)] = lambda x: steer_feature(x, sae_tau, feat_idx, delta=3.0)
```

**Expected compute.** 2–6 B200-hours.
**Positive result.** Strong concept-selective steering validates feature interpretation.
**Negative result.** Correlation without causality -> be cautious with SAE semantics.

### Experiment 19. Early-vs-late phase swap

**Question.** Are there true phase-locked computational roles?
**Method.** Swap the entire state from early (\tau) into late (\tau), and vice versa.
**Code excerpt.**

```python
trace.patch[("z_post_latent", late_tau)] = lambda x: clean_trace.cache["z_post_latent"][early_tau].to(x.device)
```

**Expected compute.** 2–4 B200-hours.
**Positive result.** Strong asymmetry in swap effects supports H4 phase-locking.
**Negative result.** Iterations are fungible, favoring H8 shared basis.

### Experiment 20. Optional ARC transfer validation

**Question.** Do the same circuit motifs appear on ARC?
**Method.** Reuse probes / SAEs / patch metrics from Maze attention TRM on a smaller ARC slice or on released checkpoints.
**Expected compute.** 1–3 extra days, so treat as stretch.
**Positive result.** Big external validity win.
**Negative result.** Still acceptable—the first paper can focus on Sudoku and Maze.

---

## 6. Comparison framework: TRM vs. standard transformer vs. Universal Transformer

This comparison should be one of the paper’s core contributions.

## 6.1 Match the right things

Hold fixed:

* tokenizer / puzzle serialization
* training data and augmentation
* width (d), head count, FF multiplier
* training budget (tokens or FLOPs)
* optimizer, LR schedule, batch size where possible
* output head / loss

Vary only:

* **weight sharing**
* **explicit recurrence**
* **explicit split state (y/z)**
* **deep supervision + detach-and-carry-forward**

## 6.2 Use a common virtual-depth index

For all models define a “virtual depth” (v):

* TRM: (v=\tau) (iteration index)
* Untied transformer: (v=\ell) (layer index)
* UT: (v=t) (recurrent step)

Then compare the same metrics as functions of (v):

1. task accuracy vs. (v)
2. entropy vs. (v)
3. prefix-matching score vs. (v)
4. native constraint-copy score vs. (v)
5. Jacobian norm / local sensitivity vs. (v)
6. SAE feature sparsity / stability vs. (v)
7. patch matrix over ((v_{\text{src}},v_{\text{dst}}))

## 6.3 What each comparison means

### TRM vs untied transformer

This isolates the effect of **weight sharing**. If both solve via similar circuits, then tying mainly compresses parameters. If they differ sharply, then tying changes the learned algorithm.

### TRM vs Universal Transformer

This isolates the effect of **TRM’s explicit state split and supervision scheme**. If UT reproduces the same temporal circuits, recurrence + sharing are enough. If not, TRM’s (y/z) factorization and detach-supervision matter mechanistically. ([arXiv][5])

### Attention TRM vs MLP TRM

This isolates the effect of **attention itself**. If Sudoku-MLP shows similar recursive phases but no induction-like signatures, then induction is not the whole story. If only attention TRM shows copy-like circuitry, that is a very clean result. ([arXiv][1])

---

## 7. Risk analysis and what to do if things fail

### Risk 1. Hidden states are too low-dimensional for SAEs to be useful

If (d) is small and features are already nearly axis-aligned, SAEs may add little.

**Fallbacks.**
Use:

* neuron/channel ablations,
* linear probes,
* sparse PCA / ICA / NMF,
* direct causal patching on raw channels,
* dynamic mode decomposition (DMD) / Koopman fits to (s_{\tau+1}\approx Ks_\tau).

A DMD baseline is especially good here because TRM is genuinely a dynamical system.

### Risk 2. The model is too small for strong polysemanticity

That is not a failure. It would actually be scientifically useful: it would mean weight-tied small recursive models achieve high performance without the heavy polysemanticity seen in larger LMs.

**Action.** Shift emphasis from SAE discovery to exact circuit tracing on raw channels/heads.

### Risk 3. Synthetic induction metrics say “no,” even if the model reasons well

That is entirely plausible. TRM is not trained as a language model, and Sudoku/Maze may rely on constraint propagation instead of token copying.

**Action.** Make the paper’s central fork explicit:

* **Classical temporal induction**
* **Domain-specific constraint-copy**
* **Distributed message passing with no discrete induction circuit**

That fork is itself publishable.

### Risk 4. Weight tying breaks existing circuit-discovery tooling assumptions

Yes. Standard tooling assumes spatially distinct layers.

**Action.** Treat the unrolled computation graph as the object of analysis. Use NNsight or a custom hook wrapper first; use TransformerLens only for matched standard-transformer baselines. ([GitHub][7])

### Risk 5. Feature dictionaries are unstable across seeds

Then do not write “feature 15342 is the row-conflict feature.” Write subspace-level or intervention-level claims.

### Risk 6. EMA weights and raw weights tell different stories

The TRM paper reports EMA is important for avoiding collapse and improving generalization. Analyze both when feasible, or at least state clearly which one you use for MI. ([arXiv][2])

### Risk 7. Storage explodes

Caching every position, every (\tau), every checkpoint is easy to overdo.

**Action.**
Save:

* selected checkpoints,
* selected (\tau),
* pooled activation subsets for SAE training,
* summary statistics for full runs,
* full traces only for curated puzzle subsets.

### Risk 8. Circuit-tracing libraries do not plug in cleanly

Anthropic’s circuit-tracing stack is oriented around supported model families; the open-source `circuit-tracer` stack can use an NNsight backend for broader support, but for TRM I would still treat it as a second-phase tool after your custom unrolled wrapper is working. ([Transformer Circuits][10])

---

## 8. The strongest paper narrative

If I were writing the project as a paper series, I would target these claims in order:

### Paper 1: **Mechanistic interpretability of a weight-tied iterative model**

Main result:

* first circuit-level analysis of TRM / any comparable weight-tied iterative architecture,
* explicit unrolled virtual-time graph,
* evidence for one of three mechanisms:

  1. temporal induction,
  2. constraint-copy induction analogue,
  3. distributed iterative message passing.

### Paper 2: **Temporal vs spatial induction**

Main result:

* matched comparison of TRM, UT, and untied transformer,
* show whether weight tying forces qualitatively different circuit reuse.

### Paper 3: **Feature geometry of recursive reasoning**

Main result:

* shared-basis vs phase-specific feature dictionaries,
* Jacobian-spectrum analysis of recursive reasoning.

---

## 9. My recommended priority order

If you want the highest expected scientific yield in one month:

1. **Instrument the unrolled TRM and reproduce Sudoku-attn, Maze-attn, Sudoku-MLP**
2. **Run iteration-resolved logit lens + behavior-by-iteration curves**
3. **Build temporal patch matrices**
4. **Run task-native constraint-copy / adjacency-copy metrics**
5. **Train iteration-specific and pooled SAEs on a small number of key (\tau) and checkpoints**
6. **Train UT and untied baselines**
7. **Do seed-stability and symmetry tests**
8. **Use ARC only as a stretch validation**

That ordering maximizes the chance you end month one with a strong result even if SAEs or synthetic IH metrics underdeliver.

The key intellectual move is this: **for TRM, the right object is not a layer circuit but an orbit circuit.** Once you treat ((\text{module}, \tau)) as the unit of analysis, almost every standard MI tool becomes usable again—and you get a genuinely new scientific question instead of a small variant of transformer interpretability.

[1]: https://arxiv.org/abs/2510.04871?utm_source=chatgpt.com "Less is More: Recursive Reasoning with Tiny Networks"
[2]: https://arxiv.org/html/2510.04871v1 "Less is More: Recursive Reasoning with Tiny Networks"
[3]: https://arxiv.org/html/2402.13055v1?utm_source=chatgpt.com "Identifying Semantic Induction Heads to Understand In-Context Learning"
[4]: https://www.cs.ox.ac.uk/people/paul.goldberg/FCS/sudoku.html?utm_source=chatgpt.com "MSc course: Foundations of Computer Science"
[5]: https://arxiv.org/abs/1807.03819?utm_source=chatgpt.com "Universal Transformers"
[6]: https://github.com/SamsungSAILMontreal/TinyRecursiveModels "GitHub - SamsungSAILMontreal/TinyRecursiveModels · GitHub"
[7]: https://github.com/TransformerLensOrg/TransformerLens?utm_source=chatgpt.com "GitHub - TransformerLensOrg/TransformerLens: A library for mechanistic ..."
[8]: https://github.com/decoderesearch/SAELens?utm_source=chatgpt.com "GitHub - decoderesearch/SAELens: Training Sparse Autoencoders on ..."
[9]: https://arxiv.org/abs/2512.11847?utm_source=chatgpt.com "Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute"
[10]: https://transformer-circuits.pub/2025/attribution-graphs/methods.html?utm_source=chatgpt.com "Circuit Tracing: Revealing Computational Graphs in Language Models"

