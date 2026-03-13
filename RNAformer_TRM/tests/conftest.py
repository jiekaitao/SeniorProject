"""Shared test fixtures for RNAformer_TRM tests."""
import sys
import os
import pytest
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RNAformer.utils.configuration import Config


def make_tiny_config(**overrides):
    """Create a minimal config for testing (no GPU, no flash attn)."""
    base = {
        'RNAformer': {
            'model_dim': 32,
            'num_head': 2,
            'n_layers': 2,
            'seq_vocab_size': 5,
            'max_len': 16,
            'initializer_range': 0.02,
            'ln_eps': 1e-6,
            'learn_ln': True,
            'resi_dropout': 0.0,
            'ff_factor': 2,
            'ff_kernel': 0,
            'use_bias': True,
            'use_glu': False,
            'zero_init': False,
            'softmax_scale': True,
            'flash_attn': False,
            'rotary_emb': False,
            'rel_pos_enc': True,
            'pos_embedding': True,
            'pdb_flag': True,
            'binary_output': True,
            'precision': 'fp32',
            # Cycling defaults
            'cycling': False,
            # Phase 1: CGAR
            'cgar_enabled': False,
            'input_reinjection': False,
            'supervision_decay': 0.7,
            # Phase 2: Dual-state
            'dual_state': False,
            'H_cycles': 1,
            'L_cycles': 1,
            # Phase 3: ACT
            'act_enabled': False,
        }
    }
    # Apply overrides to the RNAformer sub-dict
    for k, v in overrides.items():
        base['RNAformer'][k] = v
    return Config(config_dict=base)


@pytest.fixture
def tiny_config():
    """Minimal config for testing."""
    return make_tiny_config()


@pytest.fixture
def cycling_config():
    """Config with cycling enabled."""
    return make_tiny_config(cycling=4)


@pytest.fixture
def cgar_config():
    """Config with CGAR enabled."""
    return make_tiny_config(cycling=6, cgar_enabled=True, input_reinjection=True)


@pytest.fixture
def dual_state_config():
    """Config with dual-state enabled."""
    return make_tiny_config(
        dual_state=True,
        H_cycles=2,
        L_cycles=3,
    )


@pytest.fixture
def act_config():
    """Config with ACT halting enabled."""
    return make_tiny_config(
        dual_state=True,
        H_cycles=2,
        L_cycles=2,
        act_enabled=True,
    )


@pytest.fixture
def full_config():
    """Config with all TRM features enabled."""
    return make_tiny_config(
        cycling=6,
        cgar_enabled=True,
        input_reinjection=True,
        supervision_decay=0.7,
        dual_state=True,
        H_cycles=2,
        L_cycles=3,
        act_enabled=True,
    )


@pytest.fixture
def synthetic_batch():
    """Create a synthetic batch for testing. B=2, L=8."""
    B, L = 2, 8
    src_seq = torch.randint(0, 5, (B, L))
    length = torch.tensor([L, 6])  # Second sequence shorter
    pdb_sample = torch.ones(B, 1)

    # Create a simple target contact matrix (symmetric, sparse)
    trg_mat = torch.zeros(B, L, L)
    # Add a few contacts
    for b in range(B):
        seq_len = length[b].item()
        for i in range(0, seq_len - 4, 2):
            j = seq_len - 1 - i
            if j > i:
                trg_mat[b, i, j] = 1.0
                trg_mat[b, j, i] = 1.0

    # Create mask (valid positions)
    mask = torch.zeros(B, L, L, dtype=torch.bool)
    for b in range(B):
        sl = length[b].item()
        mask[b, :sl, :sl] = True

    return {
        'src_seq': src_seq,
        'length': length,
        'pdb_sample': pdb_sample,
        'trg_mat': trg_mat,
        'mask': mask,
    }


def make_model(config=None, **overrides):
    """Factory to create RiboFormerTRM with config overrides."""
    from RNAformer_TRM.model.RNAformer_TRM import RiboFormerTRM
    if config is None:
        config = make_tiny_config(**overrides)
    model = RiboFormerTRM(config.RNAformer)
    model.eval()
    return model
