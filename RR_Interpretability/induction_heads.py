"""
TRM Induction Head Analysis — Proper version with correct checkpoint loading + full ACT.

Key questions:
1. Do specific heads specialize in spatial propagation (reachable flooding)?
2. Do other heads detect walls/boundaries?
3. At which ACT STEP do specialized patterns emerge? (analog of training inflection point)
4. Do heads in FULL_COMBO specialize differently than BASELINE?

Method: Manually compute Q @ K^T / sqrt(d) to get attention matrices,
since F.scaled_dot_product_attention doesn't return weights.
"""
import os, sys, torch, json, math
import numpy as np
from collections import defaultdict

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)
from utils.functions import load_model_class
import yaml

VIZ_DIR = '/home/jietao/RR/SeniorProject/RR_Interpretability/VISUALIZATIONS'


def load_model_correct(ckpt_name, device='cuda'):
    """Load model with all 3 fixes (key prefix, numeric sort, vocab from checkpoint)."""
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
        key=lambda x: int(x.split('_')[1])  # FIX: numeric sort
    )
    sd = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location=device)
    embed_key = [k for k in sd.keys() if 'embed_tokens' in k][0]
    config['vocab_size'] = sd[embed_key].shape[0]  # FIX: vocab from checkpoint
    config['num_puzzle_identifiers'] = int(pids.max()) + 1
    config['batch_size'] = 1

    model = load_model_class(ac['name'])(config)
    sd_fixed = {k.replace('model.', '', 1): v for k, v in sd.items()}  # FIX: key prefix
    model.load_state_dict(sd_fixed, strict=False)
    model = model.to(device).eval()
    return model, config, inputs, labels, pids


def compute_attention_weights(block, x):
    """
    Manually compute attention weights from a BidirectionalBlock.
    Returns: attn_weights (B, n_heads, T, T) and output
    """
    B, T, D = x.shape
    n_heads = block.n_heads
    d_head = block.d_head

    h = block.norm1(x)
    q = block.wq(h).view(B, T, n_heads, d_head).transpose(1, 2)  # (B, H, T, d)
    k = block.wk(h).view(B, T, n_heads, d_head).transpose(1, 2)
    v = block.wv(h).view(B, T, n_heads, d_head).transpose(1, 2)

    # Manual attention: softmax(Q @ K^T / sqrt(d))
    scale = math.sqrt(d_head)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)
    attn_weights = torch.softmax(attn_scores.float(), dim=-1)

    # Also compute output for forward pass continuity
    out = torch.matmul(attn_weights.to(v.dtype), v)
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    x_out = x + block.wo(out)
    if block.has_ffn:
        h2 = block.norm2(x_out)
        x_out = x_out + block.wd(torch.nn.functional.silu(block.wg(h2)) * block.wu(h2))

    return attn_weights, x_out


def analyze_attention_at_step(attn_weights, labels, puzzle_emb_len, grid_size=30):
    """
    Analyze attention patterns:
    - Self-attention (diagonal)
    - Neighbor attention (adjacent cells in grid)
    - Reachable→Reachable attention
    - Reachable→Wall attention
    - Open→Reachable attention
    - Entropy per head
    """
    B, H, T, T2 = attn_weights.shape
    lbl = labels[0].cpu().numpy()  # (seq_len,)
    pel = puzzle_emb_len

    # Only look at maze tokens (skip puzzle embedding)
    maze_attn = attn_weights[:, :, pel:, pel:]  # (B, H, seq_len, seq_len)
    seq_len = maze_attn.shape[2]

    head_metrics = []
    for head in range(H):
        a = maze_attn[0, head].cpu().numpy()  # (seq_len, seq_len)

        # Self attention (diagonal mean)
        self_attn = np.mean(np.diag(a))

        # Neighbor attention (4-connected grid adjacency)
        neighbor_sum = 0
        neighbor_count = 0
        for pos in range(seq_len):
            r, c = pos // grid_size, pos % grid_size
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    npos = nr * grid_size + nc
                    if npos < seq_len:
                        neighbor_sum += a[pos, npos]
                        neighbor_count += 1
        neighbor_attn = neighbor_sum / max(neighbor_count, 1)

        # Reachable→Reachable attention
        reach_mask = (lbl[:seq_len] == 5)
        wall_mask = (lbl[:seq_len] == 1)
        open_mask = (lbl[:seq_len] == 2)

        r2r = 0
        r2w = 0
        r2o = 0
        o2r = 0
        if reach_mask.sum() > 0:
            reach_rows = a[reach_mask]  # (n_reach, seq_len)
            r2r = reach_rows[:, reach_mask].mean() if reach_mask.sum() > 0 else 0
            r2w = reach_rows[:, wall_mask].mean() if wall_mask.sum() > 0 else 0
            r2o = reach_rows[:, open_mask].mean() if open_mask.sum() > 0 else 0
        if open_mask.sum() > 0 and reach_mask.sum() > 0:
            open_rows = a[open_mask]
            o2r = open_rows[:, reach_mask].mean()

        # Entropy
        a_flat = a.flatten()
        a_flat = a_flat[a_flat > 1e-10]
        entropy = -np.sum(a_flat * np.log(a_flat)) / len(a_flat) if len(a_flat) > 0 else 0

        head_metrics.append({
            'head': head,
            'self_attn': float(self_attn),
            'neighbor_attn': float(neighbor_attn),
            'reach_to_reach': float(r2r),
            'reach_to_wall': float(r2w),
            'reach_to_open': float(r2o),
            'open_to_reach': float(o2r),
            'entropy': float(entropy),
            'spatial_ratio': float(r2r / max(r2w, 1e-10)),  # How much more does reach attend to reach vs wall?
        })

    return head_metrics


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt_name in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt_name} — Induction Head Analysis', flush=True)
        print(f'{"="*60}', flush=True)

        model, config, inputs_np, labels_np, pids_np = load_model_correct(ckpt_name, device)
        inner = model.inner
        H_cycles = config.get('H_cycles', 3)
        L_cycles = config.get('L_cycles', 6)
        halt_max = config.get('halt_max_steps', 16)
        pel = getattr(inner, 'puzzle_emb_len', 16)
        tl = config['seq_len'] + pel
        n_heads = config.get('n_heads', 8)

        # Use 5 samples and average
        np.random.seed(42)
        sample_idx = np.random.choice(len(inputs_np), 5, replace=False)

        step_head_data = {}  # act_step -> list of head metrics

        for act_step in range(min(halt_max, 8)):  # First 8 ACT steps (enough to see emergence)
            all_head_metrics = [[] for _ in range(n_heads)]

            for si in sample_idx:
                inp = torch.tensor(inputs_np[si:si+1], dtype=torch.long, device=device)
                lbl = torch.tensor(labels_np[si:si+1], dtype=torch.long, device=device)
                pid = torch.tensor(pids_np[si:si+1], dtype=torch.long, device=device)

                with torch.no_grad():
                    cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
                    seq_info = {'cos_sin': cos_sin}
                    input_emb = inner._input_embeddings(inp, pid)

                    z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(1, tl, -1).clone()
                    z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(1, tl, -1).clone()

                    # Run to the target ACT step
                    for s in range(act_step + 1):
                        for l_step in range(L_cycles):
                            z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                        z_H = inner.L_level(z_H, z_L, **seq_info)

                    # Now extract attention weights from the LAST L_level call (H update)
                    # We need to re-run the last H update step manually to get attention weights
                    # The H update is: z_H = inner.L_level(z_H, z_L, **seq_info)
                    # L_level is a sequence of BidirectionalBlocks

                    # Extract from each block in L_level
                    h_input = z_H.clone()
                    # L_level processes: input goes through layers with cross-attention-like behavior
                    # Since L_level is used for both L and H updates, let's extract from the H-update
                    # by running through the blocks manually

                    # Get the L_level blocks
                    if hasattr(inner.L_level, 'layers'):
                        blocks = inner.L_level.layers
                    elif hasattr(inner.L_level, 'children'):
                        blocks = list(inner.L_level.children())
                    else:
                        blocks = [inner.L_level]

                    # Extract attention from first block (main attention layer)
                    for block in blocks:
                        if hasattr(block, 'wq') and hasattr(block, 'wk'):
                            attn_w, _ = compute_attention_weights(block, z_H)
                            head_mets = analyze_attention_at_step(attn_w, lbl, pel)
                            for hm in head_mets:
                                all_head_metrics[hm['head']].append(hm)
                            break  # Only first attention block

            # Average across samples
            avg_metrics = []
            for head in range(n_heads):
                if all_head_metrics[head]:
                    avg = {}
                    for key in all_head_metrics[head][0]:
                        if key == 'head':
                            avg[key] = head
                        else:
                            vals = [m[key] for m in all_head_metrics[head]]
                            avg[key] = float(np.mean(vals))
                    avg_metrics.append(avg)

            step_head_data[act_step] = avg_metrics

            # Print summary
            print(f'\n  ACT step {act_step} ({(act_step+1)*H_cycles} H-iter):', flush=True)
            for hm in avg_metrics:
                flag = ''
                if hm['spatial_ratio'] > 2.0:
                    flag = ' *** SPATIAL HEAD'
                if hm['neighbor_attn'] > 0.05:
                    flag += ' +LOCAL'
                print(f'    Head {hm["head"]}: self={hm["self_attn"]:.3f} '
                      f'neighbor={hm["neighbor_attn"]:.4f} '
                      f'R→R={hm["reach_to_reach"]:.4f} '
                      f'R→W={hm["reach_to_wall"]:.4f} '
                      f'spatial_ratio={hm["spatial_ratio"]:.2f} '
                      f'entropy={hm["entropy"]:.3f}{flag}', flush=True)

        # Check for emergence: does spatial_ratio increase across ACT steps?
        print(f'\n  --- Spatial Head Emergence ---', flush=True)
        for head in range(n_heads):
            ratios = []
            for step in sorted(step_head_data.keys()):
                for hm in step_head_data[step]:
                    if hm['head'] == head:
                        ratios.append(hm['spatial_ratio'])
            if ratios:
                trend = ratios[-1] - ratios[0]
                print(f'    Head {head}: spatial_ratio step0={ratios[0]:.2f} → step{len(ratios)-1}={ratios[-1]:.2f} '
                      f'({"INCREASING" if trend > 0.5 else "STABLE" if abs(trend) < 0.5 else "DECREASING"})',
                      flush=True)

        # Save
        all_results[ckpt_name] = {
            'n_heads': n_heads,
            'H_cycles': H_cycles,
            'steps_analyzed': list(step_head_data.keys()),
            'step_head_data': {str(k): v for k, v in step_head_data.items()},
        }

        del model; torch.cuda.empty_cache()

    # Save all
    save_path = os.path.join(VIZ_DIR, 'induction_head_analysis.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {save_path}', flush=True)
    print(f'\n=== Induction head analysis complete ===', flush=True)


if __name__ == '__main__':
    main()
