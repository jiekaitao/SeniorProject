"""
CORRECTED TRM Interpretability — All experiments with proper checkpoint loading.
Fixes: model. prefix stripping, numeric sort, vocab_size from checkpoint, full ACT loop.

Runs: attention patterns, displacement/PCA, spatial propagation, cosine similarity.
All on FULL_COMBO (the best model) with correct weights.
"""
import os, sys, torch, json, math
import torch.nn.functional as F
import numpy as np
from collections import deque
import einops

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)
from utils.functions import load_model_class
import yaml


def load_model_correct(ckpt_name, device='cuda'):
    """Load TRM with ALL fixes applied."""
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
    config['num_puzzle_identifiers'] = int(pids.max()) + 1
    config['batch_size'] = 16

    # FIX 1: numeric sort
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')],
                        key=lambda x: int(x.split('_')[1]))
    latest = ckpt_files[-1]

    # FIX 2: vocab_size from checkpoint
    sd = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
    embed_key = [k for k in sd.keys() if 'embed_tokens' in k][0]
    config['vocab_size'] = sd[embed_key].shape[0]

    model = load_model_class(ac['name'])(config)

    # FIX 3: strip model. prefix
    sd_fixed = {k.replace('model.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd_fixed, strict=True)
    model = model.to(device).eval()
    print(f'  Loaded {ckpt_name} ({latest}), vocab={config["vocab_size"]}', flush=True)
    return model, config, inputs, labels, pids


def run_act_steps(model, config, inp, pid, device, n_act_steps=None):
    """Run the full ACT loop, return z_H at each ACT step."""
    inner = model.inner
    H = config.get('H_cycles', 3)
    L = config.get('L_cycles', 6)
    if n_act_steps is None:
        n_act_steps = config.get('halt_max_steps', 16)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel
    B = inp.shape[0]

    with torch.no_grad():
        ie = inner._input_embeddings(inp, pid)
        cs = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        si = {'cos_sin': cs}
        zH = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
        zL = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()

        z_H_history = []
        for act in range(n_act_steps):
            for h in range(H):
                for l in range(L):
                    zL = inner.L_level(zL, zH + ie, **si)
                zH = inner.L_level(zH, zL, **si)
            z_H_history.append(zH[:, pel:].clone())

    return z_H_history  # list of (B, seq_len, D)


def experiment_displacement(z_H_history, config):
    """Track z_H displacement across ACT steps."""
    results = []
    for i in range(len(z_H_history)):
        if i == 0:
            results.append({'step': 0, 'displacement': 0.0})
        else:
            d = (z_H_history[i] - z_H_history[i-1]).float().norm(dim=-1).mean().item()
            results.append({'step': i, 'displacement': d})
    return results


def experiment_attention_patterns(model, config, inp, pid, device, n_act=3):
    """Extract attention weights with correct model weights for first n_act ACT steps."""
    inner = model.inner
    H = config.get('H_cycles', 3)
    L = config.get('L_cycles', 6)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel
    B = inp.shape[0]

    results = {}

    with torch.no_grad():
        ie = inner._input_embeddings(inp, pid)
        cs = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        si = {'cos_sin': cs}
        zH = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
        zL = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()

        tau = 0
        for act in range(n_act):
            for h_step in range(H):
                for l_step in range(L):
                    # Manually compute attention for each layer
                    x = zL
                    input_inj = zH + ie
                    for layer_idx, layer in enumerate(inner.L_level.layers):
                        # RMSNorm + attention
                        normed = layer.self_attn_norm(x) if hasattr(layer, 'self_attn_norm') else x
                        # Compute QKV manually
                        attn = layer.self_attn
                        qkv = attn.qkv_proj(normed)
                        bs, sl, _ = normed.shape
                        qkv = qkv.view(bs, sl, attn.num_heads + 2*attn.num_key_value_heads, attn.head_dim)
                        q = qkv[:, :, :attn.num_heads]
                        k = qkv[:, :, attn.num_heads:attn.num_heads+attn.num_key_value_heads]
                        v = qkv[:, :, attn.num_heads+attn.num_key_value_heads:]
                        if cs is not None:
                            from models.layers import apply_rotary_pos_emb
                            q, k = apply_rotary_pos_emb(q, k, *cs)
                        q = einops.rearrange(q, 'B S H D -> B H S D')
                        k = einops.rearrange(k, 'B S H D -> B H S D')
                        scale = 1.0 / math.sqrt(attn.head_dim)
                        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                        weights = F.softmax(scores.float(), dim=-1)
                        # Average over batch
                        avg_w = weights.mean(0).cpu()  # (H, S, S)

                        # Analyze maze portion only
                        maze_w = avg_w[:, pel:, pel:]
                        n_heads = maze_w.shape[0]
                        for hi in range(n_heads):
                            hw = maze_w[hi]
                            self_attn = hw.diagonal().mean().item()
                            entropy = -(hw * (hw + 1e-10).log()).sum(-1).mean().item()
                            tag = f'act{act}_H{h_step}_L{l_step}'
                            results[f'{tag}_L{layer_idx}H{hi}'] = {
                                'self': round(self_attn, 6),
                                'entropy': round(entropy, 2),
                                'tau': tau,
                            }

                    zL = inner.L_level(zL, input_inj, **si)
                    tau += 1
                zH = inner.L_level(zH, zL, **si)
                tau += 1

    return results


def experiment_spatial(model, config, inputs_np, labels_np, pids_np, device, n_samples=100):
    """Spatial analysis: per-BFS-distance accuracy at each ACT step."""
    inner = model.inner
    pel = getattr(inner, 'puzzle_emb_len', 16)
    ms = 30
    n_act = config.get('halt_max_steps', 16)

    np.random.seed(42)
    idx = np.random.choice(len(inputs_np), n_samples, replace=False)
    inp = torch.tensor(inputs_np[idx], dtype=torch.long, device=device)
    lbl = torch.tensor(labels_np[idx], dtype=torch.long, device=device)
    pid = torch.tensor(pids_np[idx], dtype=torch.long, device=device)
    lbl_np = labels_np[idx]

    # BFS distances
    def bfs_dist(labels_1d, maze_size=30):
        gl = min(len(labels_1d), maze_size**2)
        reach = set(np.where(labels_1d[:gl] == 5)[0])
        opens = set(np.where(labels_1d[:gl] == 2)[0])
        passable = reach | opens
        rs = sorted(reach)
        if not rs: return np.full(len(labels_1d), -1)
        start = rs[0]
        dist = np.full(len(labels_1d), -1)
        q = deque([(start, 0)]); dist[start] = 0; vis = {start}
        while q:
            p, d = q.popleft()
            r, c = p // maze_size, p % maze_size
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0 <= nr < maze_size and 0 <= nc < maze_size:
                    np2 = nr*maze_size+nc
                    if np2 not in vis and np2 in passable:
                        vis.add(np2); dist[np2] = d+1; q.append((np2, d+1))
        return dist

    all_dist = np.stack([bfs_dist(lbl_np[b]) for b in range(n_samples)])

    # Run in mini-batches
    bs = 16
    # Only probe steps 0, 1, 2, 7, 15
    probe_steps = [0, 1, 2, 7, 15]
    step_preds = {s: [] for s in probe_steps}

    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        bi, bp = inp[start:end], pid[start:end]
        z_hist = run_act_steps(model, config, bi, bp, device, n_act)
        for s in probe_steps:
            if s < len(z_hist):
                logits = inner.lm_head(z_hist[s].to(device))
                step_preds[s].append(logits.argmax(dim=-1).cpu())

    results = {}
    for s in probe_steps:
        preds = torch.cat(step_preds[s], dim=0)
        correct = (preds == lbl.cpu())

        # Per-BFS-distance
        dist_accs = {}
        max_d = int(all_dist.max())
        for d in range(0, min(max_d+1, 80), 4):
            mask = (all_dist >= d) & (all_dist < d+4) & (all_dist >= 0)
            mask_t = torch.tensor(mask, dtype=torch.bool)
            if mask_t.sum() > 0:
                dist_accs[d] = correct[mask_t].float().mean().item()

        # Per-label
        label_accs = {}
        for lv in [1, 2, 5]:
            m = (lbl.cpu() == lv)
            if m.sum() > 0:
                label_accs[lv] = correct[m].float().mean().item()

        total = correct.float().mean().item()
        print(f'  ACT step {s:2d}: total={total:.4f} wall={label_accs.get(1,0):.4f} '
              f'open={label_accs.get(2,0):.4f} reach={label_accs.get(5,0):.4f}', flush=True)

        results[s] = {
            'total_acc': total,
            'label_accs': {str(k): v for k, v in label_accs.items()},
            'dist_accs': {str(k): v for k, v in dist_accs.items()},
        }

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt in ['ABLATION_FULL_COMBO']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt} — CORRECTED INTERPRETABILITY', flush=True)
        print(f'{"="*60}', flush=True)

        model, config, inputs, labels, pids = load_model_correct(ckpt, device)

        # 1. Displacement across ACT steps
        print(f'\n--- Displacement ---', flush=True)
        np.random.seed(42)
        idx = np.random.choice(len(inputs), 50, replace=False)
        inp = torch.tensor(inputs[idx], dtype=torch.long, device=device)
        pid = torch.tensor(pids[idx], dtype=torch.long, device=device)
        z_hist = run_act_steps(model, config, inp, pid, device, n_act_steps=16)
        disp = experiment_displacement(z_hist, config)
        for d in disp:
            print(f'  ACT step {d["step"]:2d}: displacement={d["displacement"]:.4f}', flush=True)

        # PCA on z_H trajectory
        print(f'\n--- PCA ---', flush=True)
        traj = torch.stack([zh[0].float().mean(dim=0) for zh in z_hist])  # (16, D)
        traj_centered = traj - traj.mean(0)
        U, S, V = torch.svd(traj_centered)
        var = (S**2) / (S**2).sum()
        eff_dim = 1.0 / (var**2).sum().item()
        print(f'  PC1: {var[0]:.4f}, PC2: {var[1]:.4f}, eff_dim: {eff_dim:.2f}', flush=True)

        # 2. Spatial propagation with BFS distance
        print(f'\n--- Spatial Propagation ---', flush=True)
        spatial = experiment_spatial(model, config, inputs, labels, pids, device, n_samples=100)

        # 3. Attention patterns (first 2 ACT steps only — expensive)
        print(f'\n--- Attention Patterns (first 2 ACT steps) ---', flush=True)
        np.random.seed(42)
        idx2 = np.random.choice(len(inputs), 8, replace=False)
        inp2 = torch.tensor(inputs[idx2], dtype=torch.long, device=device)
        pid2 = torch.tensor(pids[idx2], dtype=torch.long, device=device)
        attn = experiment_attention_patterns(model, config, inp2, pid2, device, n_act=2)
        # Print summary: L0H0 and L0H5 evolution
        for key in sorted(attn.keys()):
            if 'L0H0' in key or 'L0H5' in key:
                v = attn[key]
                print(f'  {key}: self={v["self"]:.4f} entropy={v["entropy"]:.2f}', flush=True)

        all_results[ckpt] = {
            'displacement': disp,
            'pca': {'var': var[:5].tolist(), 'eff_dim': eff_dim},
            'spatial': {str(k): v for k, v in spatial.items()},
        }

        del model; torch.cuda.empty_cache()

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/corrected_interp_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
