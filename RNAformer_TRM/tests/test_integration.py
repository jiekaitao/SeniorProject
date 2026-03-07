"""End-to-end integration tests."""
import pytest
import torch
from RNAformer_TRM.tests.conftest import make_model, make_tiny_config
from RNAformer_TRM.losses.rna_act_loss import RNAACTLoss


class TestIntegration:
    def test_full_forward_all_features(self, synthetic_batch):
        """All features enabled: forward + backward with gradients."""
        model = make_model(
            cycling=6, cgar_enabled=True, input_reinjection=True,
            dual_state=True, H_cycles=2, L_cycles=2,
            act_enabled=True,
        )
        model.train()
        model.set_curriculum_depth(0.5)

        logits, mask, q_halt = model(
            synthetic_batch['src_seq'], synthetic_batch['length'],
            synthetic_batch['pdb_sample']
        )

        loss = logits.sum() + q_halt.sum()
        loss.backward()

        # Verify gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0, "Should have gradients after backward"

    def test_training_step_reduces_loss(self, synthetic_batch):
        """Run 5 training steps, verify loss decreases."""
        model = make_model(
            dual_state=True, H_cycles=1, L_cycles=1,
            act_enabled=True,
        )
        model.train()

        loss_fn = RNAACTLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(5):
            optimizer.zero_grad()
            logits, mask, q_halt = model(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )
            total_loss, metrics = loss_fn(
                logits, synthetic_batch['trg_mat'], synthetic_batch['mask'],
                q_halt_logit=q_halt
            )
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Loss should generally decrease (allow some tolerance)
        assert losses[-1] < losses[0], \
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"

    def test_config_yaml_roundtrip(self, tmp_path):
        """Write config to YAML, reload, create model."""
        from RNAformer.utils.configuration import Config

        config = make_tiny_config(
            dual_state=True, H_cycles=2, L_cycles=2, act_enabled=True,
        )

        config.save_config(str(tmp_path), "test_config.yml")
        reloaded = Config(config_file=str(tmp_path / "test_config.yml"))

        model = make_model(reloaded)
        assert model.dual_state == True
        assert model.act_enabled == True

    def test_original_weights_loadable(self, synthetic_batch):
        """Original RNAformer state dict loads into TRM model with features disabled."""
        from RNAformer.model.RNAformer import RiboFormer

        config = make_tiny_config()

        # Create original model
        original = RiboFormer(config.RNAformer)
        original.eval()

        # Create TRM model with all features disabled
        trm = make_model(config)

        # Load original weights
        trm.load_state_dict(original.state_dict(), strict=True)

        # Compare outputs
        with torch.no_grad():
            out_orig, mask_orig = original(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )
            out_trm, mask_trm = trm(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )

        assert torch.allclose(out_orig, out_trm, atol=1e-6), \
            "TRM with disabled features should match original output"
        assert torch.equal(mask_orig, mask_trm)
