import torch
import torch.nn as nn
import torch.nn.functional as F


class RNAACTLoss(nn.Module):
    """ACT + RR losses adapted for RNA contact matrix prediction.

    Components:
    - Contact matrix loss (BCEWithLogitsLoss on valid positions)
    - Halting loss (BCE or Brier score)
    - Monotonicity regularization
    - Smoothness regularization
    - Supervision decay weighting
    """

    def __init__(self, enable_brier_halting=False, enable_monotonicity=False,
                 enable_smoothness=False, supervision_decay=0.7):
        super().__init__()
        self.contact_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.enable_brier_halting = enable_brier_halting
        self.enable_monotonicity = enable_monotonicity
        self.enable_smoothness = enable_smoothness
        self.supervision_decay = supervision_decay

    def forward(self, logits, target, mask, q_halt_logit=None, prev_q=None, cycle_step=0):
        """
        Args:
            logits: [B, L, L, 1] raw contact predictions
            target: [B, L, L] binary contact matrix
            mask: [B, L, L] bool mask for valid positions
            q_halt_logit: [B] raw halting logit (optional, only with ACT)
            prev_q: [B] previous halting probability (optional, for monotonicity/smoothness)
            cycle_step: int, current cycle step for supervision weighting

        Returns:
            total_loss: scalar tensor
            metrics: dict of detached metric tensors
        """
        # Contact matrix loss
        pred = logits[mask][:, 0]  # Extract valid predictions
        target_flat = target[mask].float()
        contact_loss = self.contact_loss_fn(pred, target_flat)
        contact_loss = contact_loss[~torch.isnan(contact_loss)].mean()

        metrics = {'contact_loss': contact_loss.detach()}
        total_loss = contact_loss

        if q_halt_logit is not None:
            # Compute correctness target for halting
            with torch.no_grad():
                B = logits.shape[0]
                pred_mat = torch.sigmoid(logits[..., -1]) > 0.5  # [B, L, L]
                # Per-sample: check if ALL valid positions match
                is_correct_list = []
                for b in range(B):
                    b_mask = mask[b]
                    if b_mask.any():
                        correct = (pred_mat[b][b_mask] == target[b][b_mask].bool()).all()
                    else:
                        correct = torch.tensor(True, device=logits.device)
                    is_correct_list.append(correct.float())
                target_y = torch.stack(is_correct_list)  # [B]

            q_probs = torch.sigmoid(q_halt_logit)  # [B]

            # Halting loss
            if self.enable_brier_halting:
                halt_loss = F.mse_loss(q_probs, target_y, reduction='mean')
            else:
                halt_loss = F.binary_cross_entropy_with_logits(
                    q_halt_logit, target_y, reduction='mean')

            total_loss = total_loss + 0.5 * halt_loss
            metrics['halt_loss'] = halt_loss.detach()

            # Monotonicity regularization
            if self.enable_monotonicity and prev_q is not None:
                mono_violation = F.relu(prev_q - q_probs) ** 2
                mono_loss = mono_violation.mean()
                total_loss = total_loss + mono_loss
                metrics['monotonicity_loss'] = mono_loss.detach()

            # Smoothness regularization
            if self.enable_smoothness and prev_q is not None:
                smooth_loss = ((q_probs - prev_q) ** 2).mean()
                total_loss = total_loss + smooth_loss
                metrics['smoothness_loss'] = smooth_loss.detach()

        # Supervision decay weighting
        weight = self.supervision_decay ** cycle_step
        total_loss = weight * total_loss
        metrics['supervision_weight'] = weight

        return total_loss, metrics
