"""Backward compatibility tests with original RNAformer."""
import pytest
import torch
from RNAformer_TRM.tests.conftest import make_model, make_tiny_config


class TestBackwardCompat:
    def test_default_config_matches_original(self, synthetic_batch):
        """RiboFormerTRM with defaults produces identical output to RiboFormer."""
        from RNAformer.model.RNAformer import RiboFormer

        config = make_tiny_config()

        torch.manual_seed(42)
        original = RiboFormer(config.RNAformer)
        original.eval()

        # Load same weights into TRM model
        trm = make_model(config)
        trm.load_state_dict(original.state_dict(), strict=True)
        trm.eval()

        with torch.no_grad():
            out_orig, mask_orig = original(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )
            out_trm, mask_trm = trm(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )

        assert torch.allclose(out_orig, out_trm, atol=1e-6)
        assert torch.equal(mask_orig, mask_trm)

    def test_state_dict_compatible(self):
        """Original state dict loads into TRM model."""
        from RNAformer.model.RNAformer import RiboFormer

        config = make_tiny_config()
        original = RiboFormer(config.RNAformer)
        trm = make_model(config)

        orig_sd = original.state_dict()
        trm_sd = trm.state_dict()

        # All original keys should exist in TRM
        for key in orig_sd:
            assert key in trm_sd, f"Missing key in TRM model: {key}"
            assert orig_sd[key].shape == trm_sd[key].shape, \
                f"Shape mismatch for {key}: {orig_sd[key].shape} vs {trm_sd[key].shape}"

    def test_cycling_backward_compat(self, synthetic_batch):
        """Cycling without CGAR/dual-state matches original behavior."""
        from RNAformer.model.RNAformer import RiboFormer

        config = make_tiny_config(cycling=3, cgar_enabled=False, input_reinjection=False)

        torch.manual_seed(42)
        original = RiboFormer(config.RNAformer)
        original.eval()

        trm = make_model(config)
        trm.load_state_dict(original.state_dict(), strict=True)
        trm.eval()

        with torch.no_grad():
            out_orig, _ = original(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )
            out_trm, _ = trm(
                synthetic_batch['src_seq'], synthetic_batch['length'],
                synthetic_batch['pdb_sample']
            )

        assert torch.allclose(out_orig, out_trm, atol=1e-6), \
            "Cycling without TRM features should match original"
