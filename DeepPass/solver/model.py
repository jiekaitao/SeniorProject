"""
Prompt Solver: A separate bidirectional recursive reasoning module
that feeds memory tokens to a frozen LLM decoder.

GPT-5.4 Pro's key insight: "separate the thinker from the talker."
TRM is a thinker. LLMs are talkers. Don't make the talker think.

Architecture:
  1. Frozen LLM embeds the prompt
  2. Bidirectional recursive solver iterates over prompt embeddings
     - z_L: token-aligned workspace (refines per-token understanding)
     - z_H: global memory slots (planner/summary)
     - Raw prompt embeddings re-injected every cycle
     - Shared weights across cycles (true iteration)
  3. Solver outputs memory tokens
  4. Frozen LLM decoder generates answer conditioned on memory + prompt
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class BidirectionalBlock(nn.Module):
    """Transformer block with BIDIRECTIONAL attention (no causal mask)."""
    def __init__(self, d_model, n_heads, ffn_dim, has_ffn=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.has_ffn = has_ffn
        if has_ffn:
            self.norm2 = RMSNorm(d_model)
            self.wg = nn.Linear(d_model, ffn_dim, bias=False)
            self.wu = nn.Linear(d_model, ffn_dim, bias=False)
            self.wd = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)

        q = self.wq(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # BIDIRECTIONAL — no causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(out)

        if self.has_ffn:
            h = self.norm2(x)
            x = x + self.wd(F.silu(self.wg(h)) * self.wu(h))

        return x


class CrossAttention(nn.Module):
    """Cross-attention: Q from one state, KV from another."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_input, kv_input):
        B, Tq, D = q_input.shape
        Tkv = kv_input.shape[1]

        q = self.wq(self.norm_q(q_input)).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(self.norm_kv(kv_input)).view(B, Tkv, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(self.norm_kv(kv_input)).view(B, Tkv, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.wo(out)


class SolverCore(nn.Module):
    """
    Two-level recursive solver (like TRM's z_H/z_L).
    Shared weights across all cycles.

    Each inner step:
      z_L = L_block(z_L + proj(prompt_embs) + cross_attn(z_L, z_H))
      z_H = H_block(z_H + cross_attn(z_H, z_L))
    """
    def __init__(self, d_model=1024, n_heads=16, ffn_dim=2816,
                 n_L_layers=2, n_memory_slots=16):
        super().__init__()
        self.d_model = d_model
        self.n_memory_slots = n_memory_slots

        # Projection from LLM embedding space to solver space
        self.proj_in = nn.Linear(4096, d_model, bias=False)  # 4096 = Llama hidden

        # L-level: token-aligned workspace (bidirectional self-attn + cross-attn from H)
        self.L_self = nn.ModuleList([
            BidirectionalBlock(d_model, n_heads, ffn_dim, has_ffn=(i == n_L_layers - 1))
            for i in range(n_L_layers)
        ])
        self.L_cross_H = CrossAttention(d_model, n_heads)

        # H-level: global memory slots (self-attn + cross-attn from L)
        self.H_self = BidirectionalBlock(d_model, n_heads, ffn_dim, has_ffn=True)
        self.H_cross_L = CrossAttention(d_model, n_heads)

        # Learned initial memory slots
        self.H_init = nn.Parameter(torch.randn(1, n_memory_slots, d_model) * 0.02)
        self.L_init_scale = nn.Parameter(torch.tensor(0.1))

        # Output projection: solver space -> LLM embedding space
        self.proj_out = nn.Linear(d_model, 4096, bias=False)
        self.out_norm = RMSNorm(4096)

    def forward(self, prompt_embeddings, K_inner=6, K_outer=3, grad_last_only=True):
        """
        prompt_embeddings: (B, T, 4096) from frozen LLM embedding layer
        Returns: memory_tokens (B, M, 4096) to prepend to decoder
        """
        B, T, _ = prompt_embeddings.shape
        e = self.proj_in(prompt_embeddings)  # (B, T, d_solver)

        # Initialize states
        z_L = self.L_init_scale * e  # Start from projected prompt
        z_H = self.H_init.expand(B, -1, -1).clone()

        # Outer refinement rounds (like TRM's H_cycles)
        for s in range(K_outer):
            use_grad = (not grad_last_only) or (s == K_outer - 1)
            ctx = torch.enable_grad() if use_grad else torch.no_grad()

            with ctx:
                # Inner steps (like TRM's L_cycles)
                for _ in range(K_inner):
                    # L refines using prompt + H guidance
                    z_L_input = z_L + e  # RAW PROMPT RE-INJECTED
                    z_L_input = z_L_input + self.L_cross_H(z_L_input, z_H)
                    for layer in self.L_self:
                        z_L_input = layer(z_L_input)
                    z_L = z_L_input

                # H refines using L
                z_H = z_H + self.H_cross_L(z_H, z_L)
                z_H = self.H_self(z_H)

            if grad_last_only and s < K_outer - 1:
                z_L = z_L.detach()
                z_H = z_H.detach()

        # Project memory tokens back to LLM space
        memory = self.out_norm(self.proj_out(z_H))  # (B, M, 4096)
        return memory


class PromptSolverLLM(nn.Module):
    """
    Complete system: recursive solver + frozen LLM decoder.

    The solver processes the prompt bidirectionally with K iterations.
    Its output (memory tokens) is prepended to the prompt for the frozen decoder.
    """
    def __init__(self, base_model, solver_d=1024, solver_heads=16,
                 solver_ffn=2816, solver_L_layers=2, n_memory=16):
        super().__init__()
        self.base_model = base_model

        # Freeze entire base model
        for p in base_model.parameters():
            p.requires_grad = False

        self.solver = SolverCore(
            d_model=solver_d, n_heads=solver_heads, ffn_dim=solver_ffn,
            n_L_layers=solver_L_layers, n_memory_slots=n_memory,
        )

    def forward(self, input_ids, labels=None, prompt_len=None,
                K_inner=6, K_outer=3, grad_last_only=True):
        """
        input_ids: (B, T) full sequence [prompt | answer]
        prompt_len: where prompt ends and answer begins
        """
        base = self.base_model
        model = base.model
        B, T = input_ids.shape

        if prompt_len is None:
            prompt_len = T // 2  # Default: first half is prompt

        # Get raw prompt embeddings from frozen LLM
        with torch.no_grad():
            all_embeds = model.embed_tokens(input_ids)  # (B, T, 4096)

        prompt_embeds = all_embeds[:, :prompt_len]  # (B, P, 4096)

        # Run recursive solver on prompt
        memory = self.solver(prompt_embeds, K_inner=K_inner, K_outer=K_outer,
                            grad_last_only=grad_last_only)  # (B, M, 4096)

        # Prepend memory tokens to the full sequence embeddings
        augmented = torch.cat([memory, all_embeds], dim=1)  # (B, M+T, 4096)

        # Run frozen decoder on augmented sequence
        M = memory.shape[1]
        position_ids = torch.arange(M + T, device=input_ids.device).unsqueeze(0)
        position_embeddings = model.rotary_emb(augmented, position_ids)

        h = augmented
        for layer in model.layers:
            h = layer(h, position_embeddings=position_embeddings)

        h = model.norm(h)
        logits = base.lm_head(h)

        # Remove memory token positions from logits — only score real tokens
        logits = logits[:, M:]  # (B, T, V)

        loss = None
        if labels is not None:
            # Answer-only loss: only score tokens after prompt_len
            answer_logits = logits[:, prompt_len-1:-1]  # predict answer tokens
            answer_labels = labels[:, prompt_len:]
            if answer_logits.shape[1] > 0 and answer_labels.shape[1] > 0:
                min_len = min(answer_logits.shape[1], answer_labels.shape[1])
                loss = F.cross_entropy(
                    answer_logits[:, :min_len].contiguous().view(-1, logits.shape[-1]),
                    answer_labels[:, :min_len].contiguous().view(-1),
                    ignore_index=-100,
                )

        return logits, loss

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        return sum(p.numel() for p in self.parameters())
