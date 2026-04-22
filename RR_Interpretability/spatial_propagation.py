"""
TRM Spatial Propagation Analysis — Does accuracy propagate like BFS across H-cycles?

For each H-cycle:
- Measure per-position accuracy on the maze grid
- Track which positions become correct first
- Measure distance from start/goal of newly-correct positions
- Compute cross-position cosine similarity (do neighbors become more similar?)
"""
import os, sys, torch, json, math
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)

from utils.functions import load_model_class
import yaml


def load_model_and_data(ckpt_name, device='cuda', n_samples=32):
    ckpt_dir = os.path.join(TRM_ROOT, 'checkpoints/SeniorProjectTRM', ckpt_name)
    with open(os.path.join(ckpt_dir, 'all_config.yaml')) as f:
        full_config = yaml.safe_load(f)

    arch_config = full_config['arch']
    data_path = full_config.get('data_paths', ['data/maze-30x30-hard-1k'])[0]
    data_full_path = os.path.join(TRM_ROOT, data_path)

    inputs = np.load(os.path.join(data_full_path, 'train/all__inputs.npy'))
    labels = np.load(os.path.join(data_full_path, 'train/all__labels.npy'))
    puzzle_ids = np.load(os.path.join(data_full_path, 'train/all__puzzle_identifiers.npy'))

    config = dict(arch_config)
    config['seq_len'] = inputs.shape[1]
    config['vocab_size'] = int(inputs.max()) + 1
    config['num_puzzle_identifiers'] = int(puzzle_ids.max()) + 1
    config['batch_size'] = n_samples

    model_cls = load_model_class(arch_config['name'])
    model = model_cls(config)
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')])
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    idx = np.random.choice(len(inputs), size=n_samples, replace=False)
    return (model, config,
            torch.tensor(inputs[idx], dtype=torch.long, device=device),
            torch.tensor(labels[idx], dtype=torch.long, device=device),
            torch.tensor(puzzle_ids[idx], dtype=torch.long, device=device),
            inputs[idx], labels[idx])


def compute_maze_distances(labels_np, maze_size=30):
    """Compute BFS distance from the 'start' cell for one maze."""
    seq_len = labels_np.shape[0]
    grid_len = min(seq_len, maze_size * maze_size)

    # Find reachable cells (label=5) — the path
    reachable = set(np.where(labels_np[:grid_len] == 5)[0])
    # Find open cells (label=2)
    open_cells = set(np.where(labels_np[:grid_len] == 2)[0])
    passable = reachable | open_cells

    # Start = first reachable cell, goal = last reachable cell
    reachable_sorted = sorted(reachable) if reachable else []
    if not reachable_sorted:
        return np.full(seq_len, -1)

    start = reachable_sorted[0]

    # BFS from start
    from collections import deque
    dist = np.full(seq_len, -1)
    queue = deque([(start, 0)])
    dist[start] = 0
    visited = {start}

    while queue:
        pos, d = queue.popleft()
        r, c = pos // maze_size, pos % maze_size
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0 <= nr < maze_size and 0 <= nc < maze_size:
                npos = nr * maze_size + nc
                if npos not in visited and npos in passable:
                    visited.add(npos)
                    dist[npos] = d + 1
                    queue.append((npos, d + 1))

    return dist


def analyze_spatial_propagation(model, config, inputs, labels, pids,
                                 inputs_np, labels_np, device='cuda'):
    """Track per-position accuracy across H-cycles."""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
    total_len = config['seq_len'] + puzzle_emb_len
    B = inputs.shape[0]
    seq_len = config['seq_len']
    maze_size = 30

    # Compute BFS distances for each maze
    all_distances = []
    for b in range(B):
        d = compute_maze_distances(labels_np[b], maze_size)
        all_distances.append(d)
    all_distances = np.stack(all_distances)  # (B, seq_len)

    results = {}

    with torch.no_grad():
        input_emb = inner._input_embeddings(inputs, pids)
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

        prev_correct = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        for h_step in range(H_cycles):
            for l_step in range(L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)

            # Decode and measure per-position accuracy
            logits = inner.lm_head(z_H)[:, puzzle_emb_len:]  # (B, seq_len, V)
            preds = logits.argmax(dim=-1)  # (B, seq_len)
            correct = (preds == labels)  # (B, seq_len)

            # Overall accuracy
            total_acc = correct.float().mean().item()

            # Per-label-type accuracy
            label_accs = {}
            for lv in [1, 2, 3, 4, 5]:
                mask = (labels == lv)
                if mask.sum() > 0:
                    label_accs[lv] = correct[mask].float().mean().item()

            # Per-distance accuracy (distance from start via BFS)
            dist_accs = {}
            max_dist = int(all_distances.max())
            dist_tensor = torch.tensor(all_distances, device=device)
            for d in range(0, min(max_dist + 1, 60), 2):  # bin by 2
                mask = (dist_tensor >= d) & (dist_tensor < d + 2) & (dist_tensor >= 0)
                if mask.sum() > 0:
                    dist_accs[d] = correct[mask].float().mean().item()

            # Newly correct positions (weren't correct at previous H-cycle)
            newly_correct = correct & ~prev_correct
            n_new = newly_correct.sum().item()
            if n_new > 0:
                new_pos = newly_correct.nonzero(as_tuple=False)  # (N, 2)
                new_dists = []
                for row in new_pos:
                    b, p = row[0].item(), row[1].item()
                    if all_distances[b, p] >= 0:
                        new_dists.append(all_distances[b, p])
                avg_new_dist = np.mean(new_dists) if new_dists else -1
                median_new_dist = np.median(new_dists) if new_dists else -1
            else:
                avg_new_dist, median_new_dist = -1, -1

            # Cross-position cosine similarity for maze body
            z_H_maze = z_H[:, puzzle_emb_len:, :]  # (B, seq_len, D)
            # Sample pairs of neighbors vs non-neighbors
            n_pairs = 500
            neighbor_sims = []
            distant_sims = []
            for _ in range(n_pairs):
                b = np.random.randint(B)
                p1 = np.random.randint(min(seq_len, maze_size * maze_size))
                r, c = p1 // maze_size, p1 % maze_size
                # Random neighbor
                neighbors = []
                if r > 0: neighbors.append(p1 - maze_size)
                if r < maze_size - 1: neighbors.append(p1 + maze_size)
                if c > 0: neighbors.append(p1 - 1)
                if c < maze_size - 1: neighbors.append(p1 + 1)
                if neighbors:
                    p2 = neighbors[np.random.randint(len(neighbors))]
                    v1 = z_H_maze[b, p1].float()
                    v2 = z_H_maze[b, p2].float()
                    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    neighbor_sims.append(sim)
                # Random distant
                p3 = np.random.randint(min(seq_len, maze_size * maze_size))
                while abs(p3 // maze_size - r) + abs(p3 % maze_size - c) < 5:
                    p3 = np.random.randint(min(seq_len, maze_size * maze_size))
                v3 = z_H_maze[b, p3].float()
                v1 = z_H_maze[b, p1].float()
                sim = F.cosine_similarity(v1.unsqueeze(0), v3.unsqueeze(0)).item()
                distant_sims.append(sim)

            cycle_results = {
                'total_acc': total_acc,
                'label_accs': {str(k): v for k, v in label_accs.items()},
                'dist_accs': {str(k): v for k, v in dist_accs.items()},
                'n_newly_correct': n_new,
                'avg_new_dist_from_start': avg_new_dist,
                'median_new_dist_from_start': median_new_dist,
                'neighbor_cosine_sim': float(np.mean(neighbor_sims)) if neighbor_sims else 0,
                'distant_cosine_sim': float(np.mean(distant_sims)) if distant_sims else 0,
                'sim_ratio': (float(np.mean(neighbor_sims)) / max(float(np.mean(distant_sims)), 1e-8)
                              if neighbor_sims and distant_sims else 0),
            }

            print(f'\n  H-cycle {h_step}:', flush=True)
            print(f'    Total acc: {total_acc:.4f}', flush=True)
            print(f'    Label accs: {label_accs}', flush=True)
            print(f'    Newly correct: {n_new} (avg dist={avg_new_dist:.1f}, median={median_new_dist:.1f})', flush=True)
            print(f'    Neighbor sim: {cycle_results["neighbor_cosine_sim"]:.4f}  '
                  f'Distant sim: {cycle_results["distant_cosine_sim"]:.4f}  '
                  f'Ratio: {cycle_results["sim_ratio"]:.4f}', flush=True)
            print(f'    Accuracy by BFS distance:', flush=True)
            for d in sorted(dist_accs.keys()):
                print(f'      d={d:2d}-{d+1}: {dist_accs[d]:.4f}', flush=True)

            results[f'h_cycle_{h_step}'] = cycle_results
            prev_correct = correct.clone()

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt in ['ABLATION_BASELINE', 'ABLATION_FULL_COMBO', 'ABLATION_REDUCED_MLP']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt}', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            model, config, inputs, labels, pids, inp_np, lbl_np = load_model_and_data(
                ckpt, device, n_samples=32)
            results = analyze_spatial_propagation(
                model, config, inputs, labels, pids, inp_np, lbl_np, device)
            all_results[ckpt] = results
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f'ERROR: {e}', flush=True)
            traceback.print_exc()
            all_results[ckpt] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/spatial_propagation.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
