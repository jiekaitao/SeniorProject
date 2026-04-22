"""
TRM Interpretability v2: Use the actual TRM repo's forward pass.
Don't reimplement — hook into the real model.
"""
import os, sys, time, json, math
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import yaml

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)

from utils.functions import load_model_class


def load_model_and_data(ckpt_name='ABLATION_FULL_COMBO', device='cuda'):
    """Load TRM using the repo's own infrastructure."""
    ckpt_dir = os.path.join(TRM_ROOT, 'checkpoints/SeniorProjectTRM', ckpt_name)

    # Load config
    with open(os.path.join(ckpt_dir, 'all_config.yaml')) as f:
        full_config = yaml.safe_load(f)

    arch_config = full_config['arch']

    # Load dataset metadata to get vocab_size, seq_len, etc.
    data_path = full_config.get('data_paths', ['data/maze-30x30-hard-1k'])[0]
    data_full_path = os.path.join(TRM_ROOT, data_path)

    inputs = np.load(os.path.join(data_full_path, 'train/all__inputs.npy'))
    labels = np.load(os.path.join(data_full_path, 'train/all__labels.npy'))
    puzzle_ids = np.load(os.path.join(data_full_path, 'train/all__puzzle_identifiers.npy'))

    seq_len = inputs.shape[1]
    vocab_size = int(inputs.max()) + 1
    num_puzzles = int(puzzle_ids.max()) + 1

    # Build config dict
    config = dict(arch_config)
    config['seq_len'] = seq_len
    config['vocab_size'] = vocab_size
    config['num_puzzle_identifiers'] = num_puzzles
    config['batch_size'] = 32

    # Load model
    model_cls = load_model_class(arch_config['name'])
    model = model_cls(config)

    # Load weights
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')])
    if not ckpt_files:
        raise FileNotFoundError(f"No step_* checkpoints in {ckpt_dir}")
    latest = ckpt_files[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f'Loading {ckpt_path}', flush=True)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # Sample data
    idx = np.random.choice(len(inputs), size=min(100, len(inputs)), replace=False)
    sample_inputs = torch.tensor(inputs[idx], dtype=torch.long, device=device)
    sample_labels = torch.tensor(labels[idx], dtype=torch.long, device=device)
    sample_pids = torch.tensor(puzzle_ids[idx], dtype=torch.long, device=device)

    return model, config, sample_inputs, sample_labels, sample_pids


def experiment_displacement_analysis(model, config, inputs, labels, pids, device='cuda'):
    """
    Track hidden state displacement across iterations.
    How much does z_H / z_L change per iteration? Does it contract?
    """
    print(f'\n=== Displacement Analysis ===', flush=True)

    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    B = min(inputs.shape[0], 32)

    batch = {
        'inputs': inputs[:B],
        'puzzle_identifiers': pids[:B],
    }

    with torch.no_grad():
        input_emb = inner._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        # H_init/L_init are (hidden_size,) — broadcast to (B, total_len, hidden_size)
        puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
        _seq_len = config.get('seq_len', 900)
        total_len = _seq_len + puzzle_emb_len
        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

        z_H_history = [z_H.float().cpu()]
        z_L_history = [z_L.float().cpu()]
        displacements_L = []
        displacements_H = []

        for h in range(H_cycles):
            for l in range(L_cycles):
                z_L_prev = z_L.clone()
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                d = (z_L - z_L_prev).float().norm(dim=-1).mean().item()
                displacements_L.append(d)
                z_L_history.append(z_L.float().cpu())

            z_H_prev = z_H.clone()
            z_H = inner.L_level(z_H, z_L, **seq_info)
            d = (z_H - z_H_prev).float().norm(dim=-1).mean().item()
            displacements_H.append(d)
            z_H_history.append(z_H.float().cpu())

    # Compute contraction rates
    print(f'\n  z_L displacements per inner step:', flush=True)
    for i, d in enumerate(displacements_L):
        rate = d / displacements_L[i-1] if i > 0 and displacements_L[i-1] > 1e-8 else float('nan')
        print(f'    step {i:2d}: disp={d:.6f}  rate={rate:.4f}', flush=True)

    print(f'\n  z_H displacements per outer step:', flush=True)
    for i, d in enumerate(displacements_H):
        rate = d / displacements_H[i-1] if i > 0 and displacements_H[i-1] > 1e-8 else float('nan')
        print(f'    step {i:2d}: disp={d:.6f}  rate={rate:.4f}', flush=True)

    avg_L = np.mean([displacements_L[i]/displacements_L[i-1]
                     for i in range(1, len(displacements_L))
                     if displacements_L[i-1] > 1e-8])
    print(f'\n  Avg z_L contraction rate: {avg_L:.4f} ({"contracting" if avg_L < 1 else "expanding"})', flush=True)

    return {
        'displacements_L': displacements_L,
        'displacements_H': displacements_H,
        'avg_contraction_L': float(avg_L),
    }


def experiment_logit_evolution(model, config, inputs, labels, pids, device='cuda'):
    """
    How does the output quality evolve across H-cycles?
    Decode z_H at each outer step and measure accuracy.
    """
    print(f'\n=== Logit Evolution Across H-cycles ===', flush=True)

    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    B = min(inputs.shape[0], 32)

    batch_inputs = inputs[:B]
    batch_labels = labels[:B]
    batch_pids = pids[:B]

    with torch.no_grad():
        input_emb = inner._input_embeddings(batch_inputs, batch_pids)
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        # H_init/L_init are (hidden_size,) — broadcast to (B, total_len, hidden_size)
        puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
        _seq_len = config.get('seq_len', 900)
        total_len = _seq_len + puzzle_emb_len
        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

        puzzle_emb_len = inner.puzzle_emb_len if hasattr(inner, 'puzzle_emb_len') else 16

        for h in range(H_cycles):
            for l in range(L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)

            # Decode and measure
            logits = inner.lm_head(z_H)[:, puzzle_emb_len:]
            preds = logits.argmax(dim=-1)
            acc = (preds == batch_labels).float().mean().item()
            entropy = -(F.softmax(logits.float(), dim=-1) *
                       F.log_softmax(logits.float(), dim=-1)).sum(-1).mean().item()
            confidence = F.softmax(logits.float(), dim=-1).max(-1).values.mean().item()

            print(f'  H-cycle {h}: accuracy={acc:.4f}  entropy={entropy:.4f}  confidence={confidence:.4f}', flush=True)

    return {}


def experiment_pca(model, config, inputs, labels, pids, device='cuda'):
    """PCA of z_H trajectories across H-cycles."""
    print(f'\n=== PCA of z_H Trajectories ===', flush=True)

    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    B = min(inputs.shape[0], 32)

    with torch.no_grad():
        input_emb = inner._input_embeddings(inputs[:B], pids[:B])
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        # H_init/L_init are (hidden_size,) — broadcast to (B, total_len, hidden_size)
        puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
        _seq_len = config.get('seq_len', 900)
        total_len = _seq_len + puzzle_emb_len
        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

        # Collect z_H at each outer step, pooled over tokens
        states = [z_H.float().mean(dim=1).cpu().numpy()]

        for h in range(H_cycles):
            for l in range(L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)
            states.append(z_H.float().mean(dim=1).cpu().numpy())

    Z = np.stack(states, axis=1)  # (B, T, D)
    Z_flat = Z.reshape(-1, Z.shape[-1])
    Z_centered = Z_flat - Z_flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)

    explained = (S ** 2) / (S ** 2).sum()
    pr = (S ** 2).sum() ** 2 / (S ** 4).sum()

    print(f'  Top-5 explained variance:', flush=True)
    for i in range(min(5, len(explained))):
        print(f'    PC{i+1}: {explained[i]:.4f} (cumul: {explained[:i+1].sum():.4f})', flush=True)
    print(f'  Effective dimensionality: {pr:.2f}', flush=True)

    return {'eff_dim': float(pr), 'top5_variance': explained[:5].tolist()}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for ckpt in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE', 'ABLATION_REDUCED_MLP']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt}', flush=True)
        print(f'{"="*60}', flush=True)
        try:
            model, config, inputs, labels, pids = load_model_and_data(ckpt, device)
            r1 = experiment_displacement_analysis(model, config, inputs, labels, pids, device)
            r2 = experiment_logit_evolution(model, config, inputs, labels, pids, device)
            r3 = experiment_pca(model, config, inputs, labels, pids, device)
            results[ckpt] = {'displacement': r1, 'logit_evolution': r2, 'pca': r3}
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}', flush=True)
            traceback.print_exc()
            results[ckpt] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/experiment_results_v2.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
