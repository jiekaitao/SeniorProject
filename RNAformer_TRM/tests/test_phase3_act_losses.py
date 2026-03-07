"""Phase 3: ACT Halting + RR Loss Tests"""
import pytest
import torch
import torch.nn.functional as F
from RNAformer_TRM.losses.rna_act_loss import RNAACTLoss
from RNAformer_TRM.tests.conftest import make_model, make_tiny_config


class TestACTLosses:
    def test_q_head_outputs_halt_logit(self, synthetic_batch):
        """Model with act_enabled=True returns q_halt_logit."""
        model = make_model(dual_state=True, H_cycles=2, L_cycles=2, act_enabled=True)
        model.eval()

        with torch.no_grad():
            result = model(synthetic_batch['src_seq'], synthetic_batch['length'],
                           synthetic_batch['pdb_sample'])

        assert len(result) == 3, "ACT model should return (logits, mask, q_halt_logit)"
        logits, mask, q_halt_logit = result
        assert q_halt_logit.shape == (2,), f"q_halt_logit shape should be [B], got {q_halt_logit.shape}"

    def test_brier_halting_loss(self):
        """Brier loss = mean((q - y)^2)."""
        loss_fn = RNAACTLoss(enable_brier_halting=True)

        logits = torch.zeros(2, 4, 4, 1)
        target = torch.zeros(2, 4, 4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)

        q_halt_logit = torch.tensor([1.3863, -1.3863])  # sigmoid -> ~0.8, ~0.2
        prev_q = None

        total_loss, metrics = loss_fn(logits, target, mask, q_halt_logit=q_halt_logit)

        assert 'halt_loss' in metrics
        # q_probs ≈ [0.8, 0.2], target depends on correctness
        # The halt_loss should be computed as MSE

    def test_bce_halting_loss_default(self):
        """Without brier, standard BCE is used."""
        loss_fn = RNAACTLoss(enable_brier_halting=False)

        logits = torch.zeros(2, 4, 4, 1)
        target = torch.zeros(2, 4, 4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)
        q_halt_logit = torch.tensor([0.0, 0.0])

        total_loss, metrics = loss_fn(logits, target, mask, q_halt_logit=q_halt_logit)
        assert 'halt_loss' in metrics
        # BCE should be well-defined
        assert not torch.isnan(total_loss)

    def test_monotonicity_loss(self):
        """Monotonicity violation: relu(prev_q - q_probs)^2."""
        loss_fn = RNAACTLoss(enable_monotonicity=True)

        logits = torch.zeros(2, 4, 4, 1)
        target = torch.zeros(2, 4, 4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)

        # prev_q > q_probs for first sample (violation), prev_q < q_probs for second (ok)
        q_halt_logit = torch.tensor([0.0, 2.0])  # sigmoid -> [0.5, 0.88]
        prev_q = torch.tensor([0.6, 0.8])

        total_loss, metrics = loss_fn(logits, target, mask, q_halt_logit=q_halt_logit, prev_q=prev_q)

        assert 'monotonicity_loss' in metrics
        mono = metrics['monotonicity_loss']

        q_probs = torch.sigmoid(q_halt_logit)
        expected_mono = (F.relu(prev_q - q_probs) ** 2).mean()
        assert mono.item() == pytest.approx(expected_mono.item(), rel=1e-4)

    def test_smoothness_loss(self):
        """Smoothness = mean((q_probs - prev_q)^2)."""
        loss_fn = RNAACTLoss(enable_smoothness=True)

        logits = torch.zeros(2, 4, 4, 1)
        target = torch.zeros(2, 4, 4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)

        q_halt_logit = torch.tensor([0.0, 2.0])
        prev_q = torch.tensor([0.6, 0.8])

        total_loss, metrics = loss_fn(logits, target, mask, q_halt_logit=q_halt_logit, prev_q=prev_q)

        assert 'smoothness_loss' in metrics
        smooth = metrics['smoothness_loss']

        q_probs = torch.sigmoid(q_halt_logit)
        expected_smooth = ((q_probs - prev_q) ** 2).mean()
        assert smooth.item() == pytest.approx(expected_smooth.item(), rel=1e-4)

    def test_supervision_decay_weighting(self):
        """Weight at step i = decay^i."""
        loss_fn = RNAACTLoss(supervision_decay=0.7)

        logits = torch.randn(1, 4, 4, 1)
        target = torch.zeros(1, 4, 4)
        mask = torch.ones(1, 4, 4, dtype=torch.bool)

        _, m0 = loss_fn(logits, target, mask, cycle_step=0)
        _, m1 = loss_fn(logits, target, mask, cycle_step=1)
        _, m5 = loss_fn(logits, target, mask, cycle_step=5)

        assert m0['supervision_weight'] == pytest.approx(1.0)
        assert m1['supervision_weight'] == pytest.approx(0.7)
        assert m5['supervision_weight'] == pytest.approx(0.7**5, rel=1e-4)

    def test_act_all_losses_combined(self):
        """Total loss includes all components when all enabled."""
        loss_fn = RNAACTLoss(
            enable_brier_halting=True,
            enable_monotonicity=True,
            enable_smoothness=True,
            supervision_decay=0.7,
        )

        logits = torch.randn(2, 4, 4, 1)
        target = torch.zeros(2, 4, 4)
        mask = torch.ones(2, 4, 4, dtype=torch.bool)
        q_halt_logit = torch.tensor([0.5, -0.5])
        prev_q = torch.tensor([0.4, 0.7])

        total_loss, metrics = loss_fn(logits, target, mask, q_halt_logit=q_halt_logit,
                                       prev_q=prev_q, cycle_step=1)

        assert 'contact_loss' in metrics
        assert 'halt_loss' in metrics
        assert 'monotonicity_loss' in metrics
        assert 'smoothness_loss' in metrics
        assert 'supervision_weight' in metrics
        assert metrics['supervision_weight'] == pytest.approx(0.7)
        assert not torch.isnan(total_loss)

    def test_act_disabled_no_q_head(self, synthetic_batch):
        """When act_enabled=False, no q_head and standard return."""
        model = make_model(act_enabled=False)
        assert not hasattr(model, 'q_head') or model.q_head is None or not model.act_enabled

        with torch.no_grad():
            result = model(synthetic_batch['src_seq'], synthetic_batch['length'],
                           synthetic_batch['pdb_sample'])
        assert len(result) == 2, "Non-ACT model should return (logits, mask)"
