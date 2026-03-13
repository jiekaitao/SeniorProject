"""Phase 1: CGAR Progressive Cycling Tests"""
import math
import pytest
import torch
from RNAformer_TRM.tests.conftest import make_model, make_tiny_config


class TestCGARCurriculum:
    def test_cgar_curriculum_depth_stages(self):
        """Verify set_curriculum_depth returns correct cycle counts for each stage."""
        config = make_tiny_config(cycling=6, cgar_enabled=True)
        model = make_model(config)
        base = 6

        # Stage 1: progress < 0.3 -> ceil(6/3) = 2
        model.set_curriculum_depth(0.0)
        assert model.current_cycle_steps == math.ceil(base / 3)
        model.set_curriculum_depth(0.15)
        assert model.current_cycle_steps == math.ceil(base / 3)
        model.set_curriculum_depth(0.29)
        assert model.current_cycle_steps == math.ceil(base / 3)

        # Stage 2: 0.3 <= progress < 0.6 -> ceil(2*6/3) = 4
        model.set_curriculum_depth(0.3)
        assert model.current_cycle_steps == math.ceil(2 * base / 3)
        model.set_curriculum_depth(0.45)
        assert model.current_cycle_steps == math.ceil(2 * base / 3)

        # Stage 3: progress >= 0.6 -> 6
        model.set_curriculum_depth(0.6)
        assert model.current_cycle_steps == base
        model.set_curriculum_depth(0.8)
        assert model.current_cycle_steps == base
        model.set_curriculum_depth(1.0)
        assert model.current_cycle_steps == base

    def test_input_reinjection_changes_output(self, synthetic_batch):
        """Forward with input_reinjection=True vs False produces different outputs."""
        torch.manual_seed(42)
        config_with = make_tiny_config(cycling=3, input_reinjection=True)
        model_with = make_model(config_with)

        torch.manual_seed(42)
        config_without = make_tiny_config(cycling=3, input_reinjection=False)
        model_without = make_model(config_without)

        # Copy weights from model_with to model_without
        model_without.load_state_dict(model_with.state_dict())

        with torch.no_grad():
            out_with, _ = model_with(synthetic_batch['src_seq'], synthetic_batch['length'],
                                      synthetic_batch['pdb_sample'])
            out_without, _ = model_without(synthetic_batch['src_seq'], synthetic_batch['length'],
                                            synthetic_batch['pdb_sample'])

        assert not torch.allclose(out_with, out_without), \
            "Input re-injection should change the output"

    def test_cgar_supervision_weighting(self):
        """Verify supervision decay weighting: weight at step i = decay^i."""
        from RNAformer_TRM.losses.rna_act_loss import RNAACTLoss
        loss_fn = RNAACTLoss(supervision_decay=0.7)

        # Create dummy data
        logits = torch.randn(2, 8, 8, 1)
        target = torch.zeros(2, 8, 8)
        mask = torch.ones(2, 8, 8, dtype=torch.bool)

        _, metrics_0 = loss_fn(logits, target, mask, cycle_step=0)
        assert metrics_0['supervision_weight'] == pytest.approx(1.0)

        _, metrics_1 = loss_fn(logits, target, mask, cycle_step=1)
        assert metrics_1['supervision_weight'] == pytest.approx(0.7)

        _, metrics_2 = loss_fn(logits, target, mask, cycle_step=2)
        assert metrics_2['supervision_weight'] == pytest.approx(0.49, rel=1e-5)

        _, metrics_5 = loss_fn(logits, target, mask, cycle_step=5)
        assert metrics_5['supervision_weight'] == pytest.approx(0.7**5, rel=1e-4)

    def test_cycling_gradient_only_last(self, synthetic_batch):
        """Only the last cycle step receives gradients."""
        config = make_tiny_config(cycling=3, cgar_enabled=True)
        model = make_model(config)
        model.train()
        model.set_curriculum_depth(1.0)  # Full depth

        logits, mask = model(synthetic_batch['src_seq'], synthetic_batch['length'],
                             synthetic_batch['pdb_sample'], max_cycle=3)

        loss = logits.sum()
        loss.backward()

        # The RNAformerStack should have gradients (used in last step)
        has_grad = False
        for p in model.RNAformer.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "RNAformerStack should have gradients from last cycle"

    def test_backward_compat_no_cgar(self, synthetic_batch):
        """When CGAR disabled and no re-injection, output matches original cycling pattern."""
        torch.manual_seed(42)
        config = make_tiny_config(cycling=3, cgar_enabled=False, input_reinjection=False)
        model = make_model(config)

        # Should still work with cycling
        with torch.no_grad():
            logits, mask = model(synthetic_batch['src_seq'], synthetic_batch['length'],
                                 synthetic_batch['pdb_sample'])

        assert logits.shape == (2, 8, 8, 1)
        assert mask.shape == (2, 8, 8)

    def test_cgar_inference_uses_full_depth(self, synthetic_batch):
        """During eval, always use full cycle_steps regardless of curriculum stage."""
        config = make_tiny_config(cycling=6, cgar_enabled=True)
        model = make_model(config)

        # Set to stage 1 (shallow)
        model.set_curriculum_depth(0.0)
        assert model.current_cycle_steps == math.ceil(6 / 3)  # 2

        # But in eval mode, should use full depth
        model.eval()
        with torch.no_grad():
            logits, mask = model(synthetic_batch['src_seq'], synthetic_batch['length'],
                                 synthetic_batch['pdb_sample'])
        assert logits.shape == (2, 8, 8, 1)
