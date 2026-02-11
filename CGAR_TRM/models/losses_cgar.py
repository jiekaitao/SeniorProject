"""
CGAR (Curriculum-Guided Adaptive Recursion) Loss Head
Adds hierarchical supervision weighting for faster convergence
"""
from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn
from models.losses import ACTLossHead, IGNORE_LABEL_ID, softmax_cross_entropy

class ACTLossHead_CGAR(ACTLossHead):
    """
    Enhanced ACTLossHead with hierarchical supervision weighting.

    Key improvement: Earlier supervision steps are weighted more heavily,
    focusing learning on the most impactful gradient updates.
    """
    def __init__(self, model: nn.Module, loss_type: str, supervision_decay: float = 0.7):
        super().__init__(model, loss_type)
        self.supervision_decay = supervision_decay  # Decay factor for supervision weighting

    def get_supervision_weight(self, step: int, max_steps: int = 16) -> float:
        """
        Compute exponential decay weight for supervision step.

        Args:
            step: Current supervision step (0-indexed)
            max_steps: Maximum supervision steps

        Returns:
            Weight multiplier (early steps ~7x higher than late steps with decay=0.7)
        """
        return self.supervision_decay ** step

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Get carry to extract current step
        carry = model_kwargs.get("carry")

        # Call parent forward
        new_carry, loss, metrics, detached_outputs, all_halted = super().forward(
            return_keys=return_keys,
            **model_kwargs
        )

        # Apply supervision weighting (only during training)
        if self.training and carry is not None:
            # Get current supervision step from carry
            # Steps are per-sample, so we use the mean for weighting
            mean_step = carry.steps.float().mean().item()

            # Compute weight
            weight = self.get_supervision_weight(int(mean_step))

            # Apply weight to loss
            weighted_loss = weight * loss

            # Add weight to metrics for monitoring
            count_tensor = metrics.get("count")
            if count_tensor is not None:
                count_tensor = count_tensor.to(loss.dtype)
                metrics["supervision_weight"] = count_tensor * torch.tensor(weight, device=loss.device, dtype=loss.dtype)
                metrics["supervision_step"] = count_tensor * torch.tensor(mean_step, device=loss.device, dtype=loss.dtype)
            else:
                metrics["supervision_weight"] = torch.tensor(weight, device=loss.device, dtype=loss.dtype)
                metrics["supervision_step"] = torch.tensor(mean_step, device=loss.device, dtype=loss.dtype)

            return new_carry, weighted_loss, metrics, detached_outputs, all_halted

        return new_carry, loss, metrics, detached_outputs, all_halted
