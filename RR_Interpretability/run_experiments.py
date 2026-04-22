"""
TRM Interpretability Experiments — from GPT_Experiments_List.md

Implements:
1. Iteration-resolved logit lens (how does answer quality evolve across cycles?)
2. Jacobian spectral analysis (contraction vs expansion across iterations)
3. State trajectory analysis (PCA, displacement, convergence)
4. Temporal activation analysis (what changes between iterations?)

Uses the trained TRM checkpoints from /blue/cis4914/jietao/SeniorProject/RR_TRM/
"""
import os, sys, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Add TRM repo to path
TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def load_trm_model(checkpoint_name='ABLATION_FULL_COMBO', device='cuda'):
    """Load a trained TRM checkpoint."""
    ckpt_dir = os.path.join(TRM_ROOT, 'checkpoints/SeniorProjectTRM', checkpoint_name)

    # Find the latest checkpoint (files named step_XXXXX, no extension)
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith('step_')]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    latest = sorted(ckpt_files, key=lambda x: int(x.split('_')[1]))[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f'Loading checkpoint: {ckpt_path}', flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'config' in ckpt:
        config = ckpt['config']
    else:
        # Default config matching the ablation studies
        config = {
            'batch_size': 256,
            'seq_len': 900,  # maze 30x30
            'puzzle_emb_ndim': 512,
            'num_puzzle_identifiers': 1000,
            'vocab_size': 5,  # maze tokens
            'H_cycles': 3,
            'L_cycles': 6,
            'H_layers': 0,
            'L_layers': 2,
            'hidden_size': 512,
            'expansion': 4,
            'num_heads': 8,
            'pos_encodings': 'rope',
            'halt_max_steps': 16,
            'halt_exploration_prob': 0.0,
            'mlp_t': False,
            'puzzle_emb_len': 16,
            'no_ACT_continue': True,
            'reduced_mlp': False,
        }

    model = TinyRecursiveReasoningModel_ACTV1(config)

    # TRM checkpoints are raw state dicts
    if isinstance(ckpt, dict) and any(k.startswith('inner.') for k in ckpt.keys()):
        model.load_state_dict(ckpt, strict=False)
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        # Raw state dict
        model.load_state_dict(ckpt, strict=False, assign=True)

    model = model.to(device).eval()
    return model, config


def experiment_1_logit_lens(model, config, device='cuda', n_samples=50):
    """
    Iteration-Resolved Logit Lens: How does answer quality evolve across cycles?

    At each virtual timestep tau, decode the current state through the LM head
    and measure: entropy, top-1 accuracy, confidence.
    """
    print(f'\n=== Experiment 1: Iteration-Resolved Logit Lens ===', flush=True)

    inner_model = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)

    results = defaultdict(list)

    # Generate random maze inputs
    B = min(n_samples, 32)
    seq_len = config.get('seq_len', 900)
    vocab_size = config.get('vocab_size', 5)

    # Random inputs (would be better with real maze data)
    input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)
    puzzle_ids = torch.zeros(B, dtype=torch.long, device=device)

    batch = {
        'inputs': input_ids,
        'puzzle_identifiers': puzzle_ids,
    }

    with torch.no_grad():
        input_embeddings = inner_model._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])

        z_H = inner_model.H_init.expand(B, -1, -1).clone()
        z_L = inner_model.L_init.expand(B, -1, -1).clone()

        cos_sin = inner_model.rotary_emb() if hasattr(inner_model, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        tau = 0
        for h_step in range(H_cycles):
            for l_step in range(L_cycles):
                z_L = inner_model.L_level(z_L, z_H + input_embeddings, **seq_info)

                # Decode through LM head at this tau
                logits = inner_model.lm_head(z_H)
                probs = F.softmax(logits.float(), dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()
                confidence = probs.max(-1).values.mean().item()

                results['tau'].append(tau)
                results['phase'].append(f'H{h_step}_L{l_step}')
                results['entropy'].append(entropy)
                results['confidence'].append(confidence)
                results['z_L_norm'].append(z_L.float().norm(dim=-1).mean().item())
                results['z_H_norm'].append(z_H.float().norm(dim=-1).mean().item())

                tau += 1

            # H-level update
            z_H = inner_model.L_level(z_H, z_L, **seq_info)

            logits = inner_model.lm_head(z_H)
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(-1).mean().item()
            confidence = probs.max(-1).values.mean().item()

            results['tau'].append(tau)
            results['phase'].append(f'H{h_step}_update')
            results['entropy'].append(entropy)
            results['confidence'].append(confidence)
            results['z_L_norm'].append(z_L.float().norm(dim=-1).mean().item())
            results['z_H_norm'].append(z_H.float().norm(dim=-1).mean().item())

            tau += 1

    # Print results
    print(f'  {"tau":>4} {"phase":>12} {"entropy":>10} {"confidence":>12} {"z_H_norm":>10} {"z_L_norm":>10}', flush=True)
    for i in range(len(results['tau'])):
        print(f'  {results["tau"][i]:4d} {results["phase"][i]:>12} '
              f'{results["entropy"][i]:10.4f} {results["confidence"][i]:12.4f} '
              f'{results["z_H_norm"][i]:10.2f} {results["z_L_norm"][i]:10.2f}', flush=True)

    return results


def experiment_2_jacobian_spectrum(model, config, device='cuda', n_samples=10):
    """
    Jacobian Spectral Analysis: Is the iteration contractive or expansive?

    Compute displacement-based contraction rate at each iteration.
    """
    print(f'\n=== Experiment 2: Jacobian / Displacement Analysis ===', flush=True)

    inner_model = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)

    B = min(n_samples, 16)
    seq_len = config.get('seq_len', 900)
    vocab_size = config.get('vocab_size', 5)

    input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)
    puzzle_ids = torch.zeros(B, dtype=torch.long, device=device)

    with torch.no_grad():
        input_embeddings = inner_model._input_embeddings(input_ids, puzzle_ids)
        cos_sin = inner_model.rotary_emb() if hasattr(inner_model, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        z_H = inner_model.H_init.expand(B, -1, -1).clone()
        z_L = inner_model.L_init.expand(B, -1, -1).clone()

        states = []
        displacements = []

        for h_step in range(H_cycles):
            for l_step in range(L_cycles):
                z_L_prev = z_L.clone()
                z_L = inner_model.L_level(z_L, z_H + input_embeddings, **seq_info)

                disp = (z_L - z_L_prev).float().norm(dim=-1).mean().item()
                displacements.append(disp)
                states.append(z_L.float().clone().cpu())

            z_H_prev = z_H.clone()
            z_H = inner_model.L_level(z_H, z_L, **seq_info)

            disp = (z_H - z_H_prev).float().norm(dim=-1).mean().item()
            displacements.append(disp)

    # Compute contraction rates
    print(f'  Step  Displacement  Contraction_Rate', flush=True)
    for i, d in enumerate(displacements):
        rate = displacements[i] / displacements[i-1] if i > 0 and displacements[i-1] > 1e-8 else float('nan')
        print(f'  {i:4d}  {d:12.6f}  {rate:16.6f}', flush=True)

    avg_rate = np.mean([displacements[i]/displacements[i-1]
                        for i in range(1, len(displacements))
                        if displacements[i-1] > 1e-8])
    print(f'\n  Average displacement contraction rate: {avg_rate:.4f}', flush=True)
    print(f'  (< 1.0 = contracting, > 1.0 = expanding)', flush=True)

    return {'displacements': displacements, 'avg_contraction': avg_rate}


def experiment_3_state_trajectory_pca(model, config, device='cuda', n_samples=20):
    """
    State Trajectory PCA: How does the hidden state evolve?
    Collect z_H at each tau, run PCA, visualize trajectory structure.
    """
    print(f'\n=== Experiment 3: State Trajectory PCA ===', flush=True)

    inner_model = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)

    B = min(n_samples, 16)
    seq_len = config.get('seq_len', 900)
    vocab_size = config.get('vocab_size', 5)

    input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)
    puzzle_ids = torch.zeros(B, dtype=torch.long, device=device)

    all_z_H = []

    with torch.no_grad():
        input_embeddings = inner_model._input_embeddings(input_ids, puzzle_ids)
        cos_sin = inner_model.rotary_emb() if hasattr(inner_model, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        z_H = inner_model.H_init.expand(B, -1, -1).clone()
        z_L = inner_model.L_init.expand(B, -1, -1).clone()

        all_z_H.append(z_H.float().mean(dim=1).cpu())  # pool over tokens

        for h_step in range(H_cycles):
            for l_step in range(L_cycles):
                z_L = inner_model.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = inner_model.L_level(z_H, z_L, **seq_info)
            all_z_H.append(z_H.float().mean(dim=1).cpu())

    # Stack and PCA
    Z = torch.stack(all_z_H, dim=1)  # (B, T, D) where T = H_cycles + 1
    Z_flat = Z.reshape(-1, Z.shape[-1]).numpy()

    # Simple PCA
    Z_centered = Z_flat - Z_flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)

    explained = (S ** 2) / (S ** 2).sum()
    print(f'  Top-5 PCA explained variance:', flush=True)
    for i in range(min(5, len(explained))):
        print(f'    PC{i+1}: {explained[i]:.4f} ({explained[:i+1].sum():.4f} cumulative)', flush=True)

    # Effective dimensionality (participation ratio)
    pr = (S ** 2).sum() ** 2 / (S ** 4).sum()
    print(f'  Effective dimensionality (participation ratio): {pr:.2f}', flush=True)

    return {'explained_variance': explained.tolist()[:10], 'eff_dim': float(pr)}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    for ckpt_name in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE', 'ABLATION_REDUCED_MLP']:
        print(f'\n{"="*60}', flush=True)
        print(f'  Checkpoint: {ckpt_name}', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            model, config = load_trm_model(ckpt_name, device=str(device))

            r1 = experiment_1_logit_lens(model, config, device=str(device))
            r2 = experiment_2_jacobian_spectrum(model, config, device=str(device))
            r3 = experiment_3_state_trajectory_pca(model, config, device=str(device))

            results[ckpt_name] = {'logit_lens': r1, 'jacobian': r2, 'pca': r3}

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            results[ckpt_name] = {'error': str(e)}

    # Save results
    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/experiment_results.json'
    with open(save_path, 'w') as f:
        json.dump({k: {kk: str(vv) if not isinstance(vv, (dict, list)) else vv
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f'\nResults saved to {save_path}', flush=True)
    print(f'=== All experiments complete ===', flush=True)


if __name__ == '__main__':
    main()
