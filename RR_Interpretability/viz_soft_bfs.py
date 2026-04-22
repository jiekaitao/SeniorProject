"""
Soft BFS Visualization Data — Extract per-cell reachability predictions at EVERY ACT step.
Produces JSON files for web app animation: user clicks through steps and sees reachability
"flooding" the maze like a wavefront.

For each sample maze:
  - The actual maze grid (walls, open, start, end, reachable ground truth)
  - Per-cell softmax probabilities at each of the 16 ACT steps (48 H-iterations)
  - Per-cell predicted label at each step
  - Per-cell confidence at each step
  - Convergence metrics: when each cell "locks in" its final prediction
"""
import os, sys, torch, json, math
import numpy as np

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)
from utils.functions import load_model_class
import yaml

VIZ_DIR = '/home/jietao/RR/SeniorProject/RR_Interpretability/VISUALIZATIONS'
os.makedirs(VIZ_DIR, exist_ok=True)


def load_model(ckpt_name, device='cuda'):
    ckpt_dir = os.path.join(TRM_ROOT, 'checkpoints/SeniorProjectTRM', ckpt_name)
    with open(os.path.join(ckpt_dir, 'all_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    ac = cfg['arch']
    dp = os.path.join(TRM_ROOT, cfg.get('data_paths', ['data/maze-30x30-hard-1k'])[0])
    inputs = np.load(os.path.join(dp, 'train/all__inputs.npy'))
    labels = np.load(os.path.join(dp, 'train/all__labels.npy'))
    pids = np.load(os.path.join(dp, 'train/all__puzzle_identifiers.npy'))

    config = dict(ac)
    config['seq_len'] = inputs.shape[1]
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )
    sd = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location=device)
    embed_key = [k for k in sd.keys() if 'embed_tokens' in k][0]
    config['vocab_size'] = sd[embed_key].shape[0]
    config['num_puzzle_identifiers'] = int(pids.max()) + 1
    config['batch_size'] = 1

    model = load_model_class(ac['name'])(config)
    sd_fixed = {k.replace('model.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd_fixed, strict=False)
    model = model.to(device).eval()
    return model, config, inputs, labels, pids


def extract_per_step_predictions(model, config, inp, lbl, pid, device='cuda'):
    """Run full ACT loop on a single maze, extract predictions at every step."""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    halt_max = config.get('halt_max_steps', 16)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel
    vocab = config['vocab_size']

    inp_t = torch.tensor(inp, dtype=torch.long, device=device).unsqueeze(0)
    pid_t = torch.tensor(pid, dtype=torch.long, device=device).unsqueeze(0)

    steps_data = []

    with torch.no_grad():
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}
        input_emb = inner._input_embeddings(inp_t, pid_t)

        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(1, tl, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(1, tl, -1).clone()

        for act_step in range(halt_max):
            for h_step in range(H_cycles):
                for l_step in range(L_cycles):
                    z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = inner.L_level(z_H, z_L, **seq_info)

            # Get predictions at this step
            logits = inner.lm_head(z_H)[:, pel:]  # (1, seq_len, vocab)
            probs = torch.softmax(logits.float(), dim=-1)  # (1, seq_len, vocab)
            preds = logits.argmax(dim=-1)  # (1, seq_len)
            confidence = probs.max(dim=-1).values  # (1, seq_len)

            # Extract per-cell data
            step_info = {
                'act_step': act_step,
                'h_iterations': (act_step + 1) * H_cycles,
                'predictions': preds[0].cpu().tolist(),
                'confidence': confidence[0].cpu().tolist(),
                'probs_reachable': probs[0, :, 5].cpu().tolist() if vocab > 5 else [0] * probs.shape[1],  # label 5 = reachable
                'probs_open': probs[0, :, 2].cpu().tolist() if vocab > 2 else [0] * probs.shape[1],  # label 2 = open
                'token_accuracy': (preds[0].cpu() == torch.tensor(lbl)).float().mean().item(),
            }
            # Reachable-only accuracy
            reach_mask = (torch.tensor(lbl) == 5)
            if reach_mask.sum() > 0:
                step_info['reachable_accuracy'] = (preds[0].cpu()[reach_mask] == torch.tensor(lbl)[reach_mask]).float().mean().item()
                step_info['reachable_confidence'] = confidence[0].cpu()[reach_mask].mean().item()
            else:
                step_info['reachable_accuracy'] = 0.0
                step_info['reachable_confidence'] = 0.0

            steps_data.append(step_info)

    return steps_data


def compute_convergence_map(steps_data, seq_len):
    """For each cell, find the first ACT step where its prediction matches its final prediction."""
    final_preds = steps_data[-1]['predictions']
    convergence = [len(steps_data)] * seq_len  # default: never converged

    for pos in range(seq_len):
        for step_idx, step in enumerate(steps_data):
            if step['predictions'][pos] == final_preds[pos]:
                # Check if it stays converged for all remaining steps
                stayed = all(s['predictions'][pos] == final_preds[pos] for s in steps_data[step_idx:])
                if stayed:
                    convergence[pos] = step_idx
                    break

    return convergence


def tokens_to_grid(tokens, labels, grid_size=30):
    """Convert flat token/label sequences to 2D grid for visualization."""
    # TRM mazes are flattened 30x30 grids
    sl = len(tokens)
    side = int(math.sqrt(sl))
    if side * side != sl:
        side = grid_size  # fallback

    grid_tokens = []
    grid_labels = []
    for r in range(side):
        row_t = []
        row_l = []
        for c in range(side):
            idx = r * side + c
            if idx < sl:
                row_t.append(int(tokens[idx]))
                row_l.append(int(labels[idx]))
            else:
                row_t.append(0)
                row_l.append(0)
        grid_tokens.append(row_t)
        grid_labels.append(row_l)
    return grid_tokens, grid_labels


# Label map for the web app
LABEL_MAP = {
    0: 'padding',
    1: 'wall',
    2: 'open',
    3: 'start',
    4: 'end',
    5: 'reachable'
}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = 10  # 10 diverse mazes for the web app

    for ckpt_name in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE',
                       'ABLATION_BRIER_ONLY', 'ABLATION_MONO_ONLY', 'ABLATION_SOFTMAX_ONLY']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt_name} — Soft BFS Visualization Data', flush=True)
        print(f'{"="*60}', flush=True)

        model, config, inputs, labels, pids = load_model(ckpt_name, device)
        seq_len = config['seq_len']
        halt_max = config.get('halt_max_steps', 16)

        # Pick diverse mazes (different reachable fractions)
        np.random.seed(42)
        n_total = len(inputs)
        reachable_fracs = [(labels[i] == 5).mean() for i in range(min(200, n_total))]
        # Sort by reachable fraction, pick evenly spaced
        sorted_idx = np.argsort(reachable_fracs)
        sample_indices = sorted_idx[np.linspace(0, len(sorted_idx)-1, n_samples, dtype=int)]

        all_mazes = []
        for i, maze_idx in enumerate(sample_indices):
            print(f'  Processing maze {i+1}/{n_samples} (idx={maze_idx})...', flush=True)

            inp = inputs[maze_idx]
            lbl = labels[maze_idx]
            pid = pids[maze_idx]

            # Extract per-step data
            steps_data = extract_per_step_predictions(model, config, inp, lbl, pid, device)

            # Compute convergence map
            convergence = compute_convergence_map(steps_data, seq_len)

            # Convert to grid
            grid_tokens, grid_labels = tokens_to_grid(inp, lbl)

            # Build per-step grids for animation
            step_grids = []
            for step in steps_data:
                _, pred_grid = tokens_to_grid(step['predictions'], step['predictions'])
                conf_flat = step['confidence']
                reach_prob_flat = step['probs_reachable']

                # Build confidence and reachable probability grids
                side = len(grid_tokens)
                conf_grid = []
                reach_grid = []
                for r in range(side):
                    conf_row = []
                    reach_row = []
                    for c in range(side):
                        idx = r * side + c
                        conf_row.append(round(conf_flat[idx], 4) if idx < len(conf_flat) else 0)
                        reach_row.append(round(reach_prob_flat[idx], 4) if idx < len(reach_prob_flat) else 0)
                    conf_grid.append(conf_row)
                    reach_grid.append(reach_row)

                step_grids.append({
                    'act_step': step['act_step'],
                    'h_iterations': step['h_iterations'],
                    'predicted_grid': pred_grid,
                    'confidence_grid': conf_grid,
                    'reachable_prob_grid': reach_grid,
                    'token_accuracy': round(step['token_accuracy'], 4),
                    'reachable_accuracy': round(step.get('reachable_accuracy', 0), 4),
                })

            # Convergence grid
            _, conv_grid = tokens_to_grid(convergence, convergence)

            maze_data = {
                'maze_index': int(maze_idx),
                'grid_size': len(grid_tokens),
                'input_grid': grid_tokens,
                'label_grid': grid_labels,
                'label_map': LABEL_MAP,
                'n_reachable': int((np.array(lbl) == 5).sum()),
                'n_open': int((np.array(lbl) == 2).sum()),
                'n_walls': int((np.array(lbl) == 1).sum()),
                'reachable_fraction': round(float((np.array(lbl) == 5).mean()), 4),
                'convergence_grid': conv_grid,
                'avg_convergence_step': round(float(np.mean([c for c in convergence if c < halt_max])), 2) if any(c < halt_max for c in convergence) else halt_max,
                'steps': step_grids,
            }
            all_mazes.append(maze_data)

            # Print summary
            acc_by_step = [s['reachable_accuracy'] for s in steps_data]
            print(f'    Reachable acc by step: {" → ".join(f"{a:.1%}" for a in acc_by_step[:5])} ... {acc_by_step[-1]:.1%}', flush=True)
            print(f'    Avg convergence: step {maze_data["avg_convergence_step"]}', flush=True)

        # Save
        output = {
            'model': ckpt_name,
            'config': {
                'halt_max_steps': halt_max,
                'H_cycles': config.get('H_cycles', 3),
                'L_cycles': config.get('L_cycles', 6),
                'hidden_size': config.get('hidden_size', 512),
                'seq_len': seq_len,
            },
            'mazes': all_mazes,
        }
        save_path = os.path.join(VIZ_DIR, f'soft_bfs_{ckpt_name}.json')
        with open(save_path, 'w') as f:
            json.dump(output, f)
        print(f'\n  Saved: {save_path} ({os.path.getsize(save_path)/1024:.0f} KB)', flush=True)

        del model; torch.cuda.empty_cache()

    print('\n=== Soft BFS visualization data complete ===', flush=True)


if __name__ == '__main__':
    main()
