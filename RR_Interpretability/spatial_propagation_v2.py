"""
TRM Spatial Propagation v2 — Larger sample (100 mazes), fixed seed for reproducibility.
Focus on BASELINE (known to K-scale to 50% accuracy) and BFS distance analysis.
"""
import os, sys, torch, json, math
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)

from utils.functions import load_model_class
import yaml


def load_model_and_data(ckpt_name, device='cuda', n_samples=100, seed=42):
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

    np.random.seed(seed)
    idx = np.random.choice(len(inputs), size=n_samples, replace=False)
    return (model, config,
            torch.tensor(inputs[idx], dtype=torch.long, device=device),
            torch.tensor(labels[idx], dtype=torch.long, device=device),
            torch.tensor(puzzle_ids[idx], dtype=torch.long, device=device),
            inputs[idx], labels[idx])


def compute_bfs_distances(labels_np, maze_size=30):
    seq_len = labels_np.shape[0]
    grid_len = min(seq_len, maze_size * maze_size)
    reachable = set(np.where(labels_np[:grid_len] == 5)[0])
    open_cells = set(np.where(labels_np[:grid_len] == 2)[0])
    passable = reachable | open_cells
    reachable_sorted = sorted(reachable)
    if not reachable_sorted:
        return np.full(seq_len, -1)
    start = reachable_sorted[0]
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


def run_analysis(model, config, inputs, labels, pids, labels_np, device='cuda'):
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
    total_len = config['seq_len'] + puzzle_emb_len
    seq_len = config['seq_len']
    maze_size = 30

    # Process in mini-batches to avoid OOM
    B_total = inputs.shape[0]
    mini_bs = 16
    all_preds = {h: [] for h in range(H_cycles)}
    all_z_H_norms = {h: [] for h in range(H_cycles)}

    for start in range(0, B_total, mini_bs):
        end = min(start + mini_bs, B_total)
        batch_inputs = inputs[start:end]
        batch_pids = pids[start:end]
        B = batch_inputs.shape[0]

        with torch.no_grad():
            input_emb = inner._input_embeddings(batch_inputs, batch_pids)
            cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
            seq_info = {'cos_sin': cos_sin}
            z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
            z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

            for h_step in range(H_cycles):
                for l_step in range(L_cycles):
                    z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = inner.L_level(z_H, z_L, **seq_info)
                logits = inner.lm_head(z_H)[:, puzzle_emb_len:]
                all_preds[h_step].append(logits.argmax(dim=-1).cpu())
                all_z_H_norms[h_step].append(z_H[:, puzzle_emb_len:].float().norm(dim=-1).cpu())

    # Concat predictions
    for h in range(H_cycles):
        all_preds[h] = torch.cat(all_preds[h], dim=0)
        all_z_H_norms[h] = torch.cat(all_z_H_norms[h], dim=0)

    labels_cpu = labels.cpu()

    # Compute BFS distances
    all_distances = np.stack([compute_bfs_distances(labels_np[b], maze_size) for b in range(B_total)])

    results = {}
    prev_correct = None

    for h_step in range(H_cycles):
        preds = all_preds[h_step]
        correct = (preds == labels_cpu)
        total_acc = correct.float().mean().item()

        # Per-label accuracy
        label_accs = {}
        for lv in [1, 2, 3, 4, 5]:
            mask = (labels_cpu == lv)
            if mask.sum() > 0:
                label_accs[lv] = correct[mask].float().mean().item()

        # Per-BFS-distance accuracy (finer bins)
        dist_accs = {}
        max_dist = int(all_distances.max())
        for d in range(0, min(max_dist + 1, 80), 2):
            mask = (all_distances >= d) & (all_distances < d + 2) & (all_distances >= 0)
            mask_t = torch.tensor(mask, dtype=torch.bool)
            if mask_t.sum() > 0:
                dist_accs[d] = correct[mask_t].float().mean().item()

        # Newly correct
        if prev_correct is not None:
            newly = correct & ~prev_correct
            n_new = newly.sum().item()
            if n_new > 0:
                new_pos = newly.nonzero(as_tuple=False)
                new_dists = [all_distances[b.item(), p.item()] for b, p in new_pos
                             if all_distances[b.item(), p.item()] >= 0]
                avg_d = np.mean(new_dists) if new_dists else -1
                med_d = np.median(new_dists) if new_dists else -1
            else:
                n_new, avg_d, med_d = 0, -1, -1
        else:
            n_new = correct.sum().item()
            new_pos = correct.nonzero(as_tuple=False)
            new_dists = [all_distances[b.item(), p.item()] for b, p in new_pos
                         if all_distances[b.item(), p.item()] >= 0]
            avg_d = np.mean(new_dists) if new_dists else -1
            med_d = np.median(new_dists) if new_dists else -1

        # z_H norm by cell type
        norms = all_z_H_norms[h_step]
        norm_by_type = {}
        for lv in [1, 2, 5]:
            mask = (labels_cpu == lv)
            if mask.sum() > 0:
                norm_by_type[lv] = norms[mask].mean().item()

        cycle_r = {
            'total_acc': total_acc,
            'label_accs': {str(k): v for k, v in label_accs.items()},
            'dist_accs': {str(k): v for k, v in dist_accs.items()},
            'n_newly_correct': n_new,
            'avg_new_dist': float(avg_d),
            'median_new_dist': float(med_d),
            'z_H_norm_wall': norm_by_type.get(1, 0),
            'z_H_norm_open': norm_by_type.get(2, 0),
            'z_H_norm_reachable': norm_by_type.get(5, 0),
        }

        print(f'\n  H-cycle {h_step}:', flush=True)
        print(f'    Total acc: {total_acc:.4f}', flush=True)
        print(f'    Per-label: {label_accs}', flush=True)
        print(f'    New: {n_new} (avg_d={avg_d:.1f}, med_d={med_d:.1f})', flush=True)
        print(f'    z_H norm — wall={norm_by_type.get(1,0):.3f} open={norm_by_type.get(2,0):.3f} reach={norm_by_type.get(5,0):.3f}', flush=True)
        print(f'    BFS distance curve:', flush=True)
        for d in sorted(dist_accs.keys()):
            bar = '#' * int(dist_accs[d] * 50)
            print(f'      d={d:2d}: {dist_accs[d]:.4f} {bar}', flush=True)

        results[f'h_cycle_{h_step}'] = cycle_r
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
                ckpt, device, n_samples=100, seed=42)
            results = run_analysis(model, config, inputs, labels, pids, lbl_np, device)
            all_results[ckpt] = results
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f'ERROR: {e}', flush=True)
            traceback.print_exc()
            all_results[ckpt] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/spatial_propagation_v2.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
