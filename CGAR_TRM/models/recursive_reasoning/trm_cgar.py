"""
TRM-CGAR: TRM with Curriculum-Guided Adaptive Recursion
Adds progressive depth curriculum for faster, more stable training
"""
import math
from typing import Tuple, List, Dict, Optional
import torch
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry
)


class TinyRecursiveReasoningModel_ACTV1_Inner_CGAR(TinyRecursiveReasoningModel_ACTV1_Inner):
    """
    Enhanced TRM Inner with dynamic depth control for progressive curriculum.
    """
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__(config)
        # Store base config values
        self.base_H_cycles = config.H_cycles
        self.base_L_cycles = config.L_cycles
        # Current values (can be dynamically adjusted)
        self.current_H_cycles = config.H_cycles
        self.current_L_cycles = config.L_cycles

    def set_curriculum_depth(self, progress: float):
        """
        Set recursion depth based on training progress.

        Args:
            progress: Training progress from 0.0 (start) to 1.0 (end)
        """
        stage1_H = max(1, math.ceil(self.base_H_cycles / 3))
        stage1_L = max(2, math.ceil(self.base_L_cycles / 3))
        stage2_H = max(stage1_H, math.ceil(2 * self.base_H_cycles / 3))
        stage2_L = max(stage1_L, math.ceil(2 * self.base_L_cycles / 3))

        if progress < 0.3:
            # First 30%: Shallow (fast warmup)
            self.current_H_cycles = stage1_H
            self.current_L_cycles = stage1_L
        elif progress < 0.6:
            # Middle 30%: Medium
            self.current_H_cycles = min(self.base_H_cycles, stage2_H)
            self.current_L_cycles = min(self.base_L_cycles, stage2_L)
        else:
            # Final 40%: Full depth
            self.current_H_cycles = self.base_H_cycles
            self.current_L_cycles = self.base_L_cycles

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations with DYNAMIC depth
        z_H, z_L = carry.z_H, carry.z_L
        # Use current_H_cycles and current_L_cycles instead of config values
        H_cycles = self.current_H_cycles if self.training else self.base_H_cycles
        L_cycles = self.current_L_cycles if self.training else self.base_L_cycles

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(H_cycles - 1):
                for _L_step in range(L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1_CGAR(TinyRecursiveReasoningModel_ACTV1):
    """ACT wrapper with CGAR enhancements."""

    def __init__(self, config_dict: dict):
        # Don't call super().__init__ yet, we need to replace inner first
        super(TinyRecursiveReasoningModel_ACTV1, self).__init__()  # Skip parent init
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner_CGAR(self.config)  # Use CGAR inner

    def set_curriculum_depth(self, progress: float):
        """Forward curriculum setting to inner model."""
        self.inner.set_curriculum_depth(progress)
