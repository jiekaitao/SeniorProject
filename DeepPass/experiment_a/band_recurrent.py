"""
Experiment A: Band-Recurrent Decoder on Llama 3.1 8B

GPT-5.4 Pro design: take a frozen pretrained LLM, add attention-dominant
recurrent replay on a middle band of layers, targeting only hard tokens.

Architecture:
    Lower (layers 0-11):  frozen, produces prefix representation e
    Band  (layers 12-15): frozen first pass + trainable replay pass
    Upper (layers 16-31): frozen, produces logits

    First pass: exact pretrained model (dense containment)
    Extra pass: h += tanh(gate) * replay_attn(replay_norm(h))
                Only on top 25% entropy tokens (hardest predictions)

Trainable params: ~5M (replay norms + gates + Q/K/V/O LoRA rank 16)
Frozen params: ~8B
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class LoRALinear(nn.Module):
    """Low-rank adapter for attention projections."""
    def __init__(self, base_linear, rank=16):
        super().__init__()
        self.base = base_linear  # frozen
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_a = nn.Linear(in_f, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_f, bias=False)
        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x))


class ExtraPassLayer(nn.Module):
    """Replay layer: runs the frozen base Llama layer again with gated output.

    MVP: no LoRA, no projection swaps. Just re-run the exact same layer
    and gate the delta. The only trainable params are the gate and a norm.
    If this helps, we add LoRA later.
    """
    def __init__(self, base_layer, d_model, lora_rank=16):
        super().__init__()
        self.base_layer = base_layer  # frozen Llama decoder layer
        self.replay_norm = RMSNorm(d_model)

        # Gate: small positive init so gradient flows through replay from the start
        # tanh(0.1) ≈ 0.1, replay contributes ~10% initially
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_states, position_embeddings, token_mask=None):
        """One replay pass: re-run frozen base layer, gate the delta."""
        h_in = hidden_states
        target_dtype = self.base_layer.input_layernorm.weight.dtype

        # Re-run the exact same frozen layer, ensuring correct dtype
        layer_out = self.base_layer(
            hidden_states.to(target_dtype),
            position_embeddings=position_embeddings,
        )

        # Gate the delta (keep in target dtype)
        delta = layer_out.to(target_dtype) - h_in.to(target_dtype)
        if token_mask is not None:
            delta = delta * token_mask.unsqueeze(-1)

        gate = torch.tanh(self.gate.to(target_dtype))
        return (h_in.to(target_dtype) + gate * delta)


class BandRecurrentLlama(nn.Module):
    """
    Wraps a pretrained Llama model with recurrent replay on middle layers.

    First pass: exact pretrained forward (frozen).
    Extra passes: replay attention on band layers with LoRA + gating.
    """
    def __init__(self, base_model, replay_layer_ids=(12, 13, 14, 15),
                 lora_rank=16, max_extra_passes=1):
        super().__init__()
        self.base_model = base_model
        self.replay_layer_ids = set(replay_layer_ids)
        self.max_extra = max_extra_passes
        d_model = base_model.config.hidden_size

        # Freeze entire base model
        for p in base_model.parameters():
            p.requires_grad = False

        # Build replay layers for the band
        self.extra_layers = nn.ModuleDict()
        for layer_id in sorted(replay_layer_ids):
            base_layer = base_model.model.layers[layer_id]
            self.extra_layers[str(layer_id)] = ExtraPassLayer(
                base_layer, d_model, lora_rank
            )

        # Reinjection gate (full lower-stack representation)
        self.inj_gate = nn.Parameter(torch.zeros(d_model))

        # Mix gate: small positive init so replay layers get gradient from the start
        # tanh(0.1) ≈ 0.1, so replay contributes ~10% initially
        self.mix_gate = nn.Parameter(torch.tensor(0.1))

    def _to_device(self, device, dtype):
        """Move wrapper params to match hidden state device and dtype."""
        self.inj_gate.data = self.inj_gate.data.to(device=device, dtype=dtype)
        self.mix_gate.data = self.mix_gate.data.to(device=device, dtype=dtype)
        for extra in self.extra_layers.values():
            extra.gate.data = extra.gate.data.to(device=device, dtype=dtype)
            extra.replay_norm.to(device=device, dtype=dtype)

    def forward(self, input_ids, labels=None, K=1, hard_frac=0.25):
        """
        K=1: exact pretrained model.
        K=2: pretrained pass + one replay pass on band layers.
        """
        # Full pretrained forward pass, capturing intermediate states
        base_model = self.base_model
        model = base_model.model

        # Embedding
        hidden_states = model.embed_tokens(input_ids)

        # Ensure wrapper params match hidden state device and dtype
        self._to_device(hidden_states.device, hidden_states.dtype)

        # Build position embeddings
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        # Run all layers (first pass = exact pretrained)
        band_input = None  # save the state entering the band
        for i, layer in enumerate(model.layers):
            if i == min(self.replay_layer_ids):
                band_input = hidden_states.clone()  # save for reinjection
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)

        # Extra replay passes
        if K > 1 and band_input is not None:
            h_prev = hidden_states

            # Reinject lower-stack representation (keep dtype consistent)
            dtype = hidden_states.dtype
            hidden_states = hidden_states + torch.tanh(self.inj_gate.to(dtype).to(hidden_states.device)) * band_input

            # Compute token mask based on base model entropy
            token_mask = None
            if hard_frac < 1.0:
                with torch.no_grad():
                    normed = model.norm(h_prev)
                    pre_logits = base_model.lm_head(normed.to(base_model.lm_head.weight.dtype))
                    probs = F.softmax(pre_logits[:, :-1].float(), dim=-1)
                    entropy = -(probs * (probs + 1e-8).log()).sum(-1)  # (B, T-1)
                    # Pad to match sequence length
                    entropy = F.pad(entropy, (0, 1), value=0.0)  # (B, T)
                    thresh = torch.quantile(entropy.flatten(), 1.0 - hard_frac)
                    token_mask = (entropy >= thresh).float()

            # Replay band layers
            for layer_id in sorted(self.replay_layer_ids):
                extra = self.extra_layers[str(layer_id)]
                hidden_states = extra(hidden_states, position_embeddings, token_mask)

            # Mix: h_final = h_prev + mix_gate * (h_replay - h_prev)
            mix = torch.tanh(self.mix_gate.to(dtype).to(hidden_states.device))
            hidden_states = h_prev + mix * (hidden_states - h_prev)

        # Final norm + LM head (ensure bfloat16 for lm_head)
        hidden_states = model.norm(hidden_states)
        logits = base_model.lm_head(hidden_states.to(base_model.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        return sum(p.numel() for p in self.parameters())
