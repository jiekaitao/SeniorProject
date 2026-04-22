"""
TRM Attention Pattern Extraction — monkey-patch to capture actual attention weights.
Captures per-head, per-layer, per-iteration attention maps.
Analyzes: Which heads attend to neighbors (BFS-like)? Do patterns change across H-cycles?
"""
import os, sys, torch, math, json
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)

from utils.functions import load_model_class
import yaml


def load_model_and_data(ckpt_name='ABLATION_BASELINE', device='cuda'):
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
    config['batch_size'] = 8

    model_cls = load_model_class(arch_config['name'])
    model = model_cls(config)
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')])
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    idx = np.random.choice(len(inputs), size=8, replace=False)
    return (model, config,
            torch.tensor(inputs[idx], dtype=torch.long, device=device),
            torch.tensor(labels[idx], dtype=torch.long, device=device),
            torch.tensor(puzzle_ids[idx], dtype=torch.long, device=device))


def monkey_patch_attention(model):
    """Replace scaled_dot_product_attention with manual computation that stores weights."""
    import einops

    attn_storage = {}

    for layer_idx, layer in enumerate(model.inner.L_level.layers):
        attn_mod = layer.self_attn
        original_forward = attn_mod.forward

        def make_patched_forward(orig_fwd, mod, l_idx):
            def patched_forward(cos_sin, hidden_states):
                batch_size, seq_len, _ = hidden_states.shape
                qkv = mod.qkv_proj(hidden_states)
                qkv = qkv.view(batch_size, seq_len,
                                mod.num_heads + 2 * mod.num_key_value_heads,
                                mod.head_dim)
                query = qkv[:, :, :mod.num_heads]
                key = qkv[:, :, mod.num_heads:mod.num_heads + mod.num_key_value_heads]
                value = qkv[:, :, mod.num_heads + mod.num_key_value_heads:]

                if cos_sin is not None:
                    from models.layers import apply_rotary_pos_emb
                    query, key = apply_rotary_pos_emb(query, key, *cos_sin)

                query = einops.rearrange(query, 'B S H D -> B H S D')
                key = einops.rearrange(key, 'B S H D -> B H S D')
                value = einops.rearrange(value, 'B S H D -> B H S D')

                # Manual attention computation to capture weights
                scale = 1.0 / math.sqrt(mod.head_dim)
                scores = torch.matmul(query, key.transpose(-2, -1)) * scale
                attn_weights = F.softmax(scores.float(), dim=-1).to(value.dtype)

                # Store attention weights (averaged over batch)
                attn_storage[l_idx] = attn_weights.float().mean(0).detach().cpu()  # (H, S, S)

                attn_output = torch.matmul(attn_weights, value)
                attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
                attn_output = attn_output.reshape(batch_size, seq_len, mod.head_dim * mod.num_heads)
                return mod.o_proj(attn_output)
            return patched_forward

        attn_mod.forward = make_patched_forward(original_forward, attn_mod, layer_idx)

    return attn_storage


def analyze_attention_patterns(attn_weights, config, labels_0, maze_size=30):
    """Analyze attention patterns for BFS-like behavior."""
    puzzle_emb_len = config.get('puzzle_emb_len', 16)
    seq_len = config['seq_len']
    results = {}

    label_np = labels_0.cpu().numpy()
    wall_mask = (label_np == 1)
    open_mask = (label_np == 2)
    reachable_mask = (label_np == 5)

    for layer_idx, weights in attn_weights.items():
        # weights shape: (H, total_len, total_len)
        H = weights.shape[0]
        # Focus on maze positions (skip puzzle embedding positions)
        maze_attn = weights[:, puzzle_emb_len:, puzzle_emb_len:]  # (H, seq_len, seq_len)

        layer_results = {}
        for head_idx in range(H):
            head_attn = maze_attn[head_idx]  # (seq_len, seq_len)

            # Diagonal dominance (self-attention)
            diag = head_attn.diagonal().mean().item()

            # Neighbor attention: for maze, position (r,c) = index r*maze_size+c
            # Neighbors are +/-1 (left/right) and +/-maze_size (up/down)
            neighbor_attn = 0.0
            n_valid = 0
            for pos in range(min(seq_len, maze_size * maze_size)):
                r, c = pos // maze_size, pos % maze_size
                neighbors = []
                if r > 0: neighbors.append(pos - maze_size)
                if r < maze_size - 1: neighbors.append(pos + maze_size)
                if c > 0: neighbors.append(pos - 1)
                if c < maze_size - 1: neighbors.append(pos + 1)
                for n in neighbors:
                    if n < seq_len:
                        neighbor_attn += head_attn[pos, n].item()
                        n_valid += 1
            avg_neighbor = neighbor_attn / max(n_valid, 1)

            # Attention from reachable to reachable vs reachable to wall
            if reachable_mask.sum() > 0 and wall_mask.sum() > 0:
                reach_idx = np.where(reachable_mask)[0]
                wall_idx = np.where(wall_mask)[0]
                r2r = head_attn[np.ix_(reach_idx, reach_idx)].mean().item()
                r2w = head_attn[np.ix_(reach_idx, wall_idx)].mean().item()
            else:
                r2r, r2w = 0.0, 0.0

            # Entropy of attention distribution
            entropy = -(head_attn * (head_attn + 1e-10).log()).sum(-1).mean().item()

            layer_results[head_idx] = {
                'self_attn': diag,
                'neighbor_attn': avg_neighbor,
                'reach_to_reach': r2r,
                'reach_to_wall': r2w,
                'entropy': entropy,
            }

        results[layer_idx] = layer_results

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt in ['ABLATION_BASELINE', 'ABLATION_FULL_COMBO']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt}', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            model, config, inputs, labels, pids = load_model_and_data(ckpt, device)
            inner = model.inner
            H_cycles = config.get('H_cycles', 3)
            L_cycles = config.get('L_cycles', 6)
            puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
            total_len = config['seq_len'] + puzzle_emb_len
            B = inputs.shape[0]

            # Monkey-patch to capture attention weights
            attn_storage = monkey_patch_attention(model)

            ckpt_results = {}
            with torch.no_grad():
                input_emb = inner._input_embeddings(inputs, pids)
                cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
                seq_info = {'cos_sin': cos_sin}

                z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
                z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

                tau = 0
                for h_step in range(H_cycles):
                    for l_step in range(L_cycles):
                        z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)

                        # Capture attention patterns for this tau
                        patterns = analyze_attention_patterns(
                            attn_storage, config, labels[0])

                        tag = f'H{h_step}_L{l_step}'
                        print(f'\n  tau={tau} [{tag}]:', flush=True)
                        for layer_idx, heads in sorted(patterns.items()):
                            for head_idx, stats in sorted(heads.items()):
                                print(f'    L{layer_idx}H{head_idx}: '
                                      f'self={stats["self_attn"]:.4f} '
                                      f'neighbor={stats["neighbor_attn"]:.4f} '
                                      f'R→R={stats["reach_to_reach"]:.6f} '
                                      f'R→W={stats["reach_to_wall"]:.6f} '
                                      f'entropy={stats["entropy"]:.2f}', flush=True)

                        ckpt_results[f'tau_{tau}_{tag}'] = {
                            str(li): {str(hi): s for hi, s in hs.items()}
                            for li, hs in patterns.items()
                        }
                        tau += 1

                    # H-update
                    z_H = inner.L_level(z_H, z_L, **seq_info)
                    patterns = analyze_attention_patterns(attn_storage, config, labels[0])
                    tag = f'H{h_step}_update'
                    print(f'\n  tau={tau} [{tag}]:', flush=True)
                    for layer_idx, heads in sorted(patterns.items()):
                        for head_idx, stats in sorted(heads.items()):
                            print(f'    L{layer_idx}H{head_idx}: '
                                  f'self={stats["self_attn"]:.4f} '
                                  f'neighbor={stats["neighbor_attn"]:.4f} '
                                  f'R→R={stats["reach_to_reach"]:.6f} '
                                  f'R→W={stats["reach_to_wall"]:.6f} '
                                  f'entropy={stats["entropy"]:.2f}', flush=True)
                    ckpt_results[f'tau_{tau}_{tag}'] = {
                        str(li): {str(hi): s for hi, s in hs.items()}
                        for li, hs in patterns.items()
                    }
                    tau += 1

            all_results[ckpt] = ckpt_results
            del model; torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f'ERROR: {e}', flush=True)
            traceback.print_exc()
            all_results[ckpt] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/attention_patterns.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
