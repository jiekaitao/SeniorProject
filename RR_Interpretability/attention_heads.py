"""
TRM Induction Head Analysis: Extract and analyze attention patterns per iteration.

Key questions:
- Do specific heads change role across iterations? (previous-token → induction)
- Is there a lag structure? (head at tau=2 attends to pattern from tau=0)
- Which heads matter most for solving? (activation patching)
"""
import os, sys, torch, numpy as np, yaml, json
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, '/blue/cis4914/jietao/SeniorProject/RR_TRM')
from utils.functions import load_model_class


def extract_attention_patterns(model, config, inputs, pids, device='cuda'):
    """
    Run TRM with hooks to capture attention weights at every iteration.
    Returns: dict[tau][layer][head] -> attention matrix (T, T)
    """
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
    total_len = config['seq_len'] + puzzle_emb_len
    B = inputs.shape[0]

    # Hook storage
    attn_store = defaultdict(dict)
    hooks = []

    def make_attn_hook(layer_idx, store_key):
        def hook_fn(module, args, output):
            # Try to get attention weights from the module
            # TransformerEngine / custom attention may store them differently
            if hasattr(module, '_attn_weights'):
                attn_store[store_key][layer_idx] = module._attn_weights.detach().cpu()
        return hook_fn

    # Register hooks on attention modules in L_level
    for i, layer in enumerate(inner.L_level.layers):
        if hasattr(layer, 'self_attn'):
            h = layer.self_attn.register_forward_hook(make_attn_hook(i, 'default'))
            hooks.append(h)

    # Run forward pass, collecting per-tau data
    tau_data = {}

    with torch.no_grad():
        input_emb = inner._input_embeddings(inputs, pids)
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

        tau = 0
        for h_step in range(H_cycles):
            for l_step in range(L_cycles):
                z_L_prev = z_L.clone()
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)

                # Compute what changed
                delta = (z_L - z_L_prev).float()
                delta_norm = delta.norm(dim=-1).mean(dim=0)  # per-position

                tau_data[tau] = {
                    'type': f'L_step',
                    'h_step': h_step,
                    'l_step': l_step,
                    'delta_norm_mean': delta_norm.mean().item(),
                    'delta_norm_std': delta_norm.std().item(),
                    'delta_norm_max_pos': delta_norm.argmax().item(),
                }
                tau += 1

            z_H_prev = z_H.clone()
            z_H = inner.L_level(z_H, z_L, **seq_info)

            delta = (z_H - z_H_prev).float()
            delta_norm = delta.norm(dim=-1).mean(dim=0)

            tau_data[tau] = {
                'type': 'H_update',
                'h_step': h_step,
                'delta_norm_mean': delta_norm.mean().item(),
                'delta_norm_std': delta_norm.std().item(),
                'delta_norm_max_pos': delta_norm.argmax().item(),
            }
            tau += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    return tau_data


def analyze_representation_similarity(model, config, inputs, labels, pids, device='cuda'):
    """
    Analyze how representations at specific maze positions evolve.
    Group by: wall cells vs open cells vs reachable cells.
    """
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
    total_len = config['seq_len'] + puzzle_emb_len
    B = inputs.shape[0]

    print(f'\n=== Representation Analysis by Cell Type ===', flush=True)

    with torch.no_grad():
        input_emb = inner._input_embeddings(inputs, pids)
        cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
        seq_info = {'cos_sin': cos_sin}

        z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()
        z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, total_len, -1).clone()

        # Classify tokens by type (using labels as proxy)
        # In maze data: different label values correspond to different cell types
        label_types = labels[0].cpu().numpy()
        unique_labels = np.unique(label_types)
        print(f'  Unique label values: {unique_labels}', flush=True)
        print(f'  Label distribution: {[(l, (label_types == l).sum()) for l in unique_labels]}', flush=True)

        for h_step in range(H_cycles):
            for l_step in range(L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)

            # Get logits and check accuracy per label type
            logits = inner.lm_head(z_H)[:, puzzle_emb_len:]
            preds = logits.argmax(dim=-1)

            for label_val in unique_labels:
                mask = (labels == label_val)
                if mask.sum() > 0:
                    acc = (preds[mask] == labels[mask]).float().mean().item()
                    conf = F.softmax(logits.float(), dim=-1).max(-1).values[mask].mean().item()
                    print(f'  H-cycle {h_step}, label={label_val}: acc={acc:.4f} conf={conf:.4f} (n={mask.sum().item()})',
                          flush=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for CKPT in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {CKPT}', flush=True)
        print(f'{"="*60}', flush=True)

        ckpt_dir = f'/blue/cis4914/jietao/SeniorProject/RR_TRM/checkpoints/SeniorProjectTRM/{CKPT}'
        with open(os.path.join(ckpt_dir, 'all_config.yaml')) as f:
            full_cfg = yaml.safe_load(f)

        arch_cfg = full_cfg['arch']
        data_path = full_cfg.get('data_paths', ['data/maze-30x30-hard-1k'])[0]
        data_full = os.path.join('/blue/cis4914/jietao/SeniorProject/RR_TRM', data_path)

        inputs_np = np.load(os.path.join(data_full, 'train/all__inputs.npy'))
        labels_np = np.load(os.path.join(data_full, 'train/all__labels.npy'))
        pids_np = np.load(os.path.join(data_full, 'train/all__puzzle_identifiers.npy'))

        config = dict(arch_cfg)
        config['seq_len'] = inputs_np.shape[1]
        config['vocab_size'] = int(inputs_np.max()) + 1
        config['num_puzzle_identifiers'] = int(pids_np.max()) + 1
        config['batch_size'] = 16

        model_cls = load_model_class(arch_cfg['name'])
        model = model_cls(config)
        ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')])
        state_dict = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

        B = 16
        idx = np.random.choice(len(inputs_np), B, replace=False)
        inputs = torch.tensor(inputs_np[idx], dtype=torch.long, device=device)
        labels = torch.tensor(labels_np[idx], dtype=torch.long, device=device)
        pids = torch.tensor(pids_np[idx], dtype=torch.long, device=device)

        # Experiment 1: Per-tau delta analysis
        print(f'\n--- Per-Tau Delta Analysis ---', flush=True)
        tau_data = extract_attention_patterns(model, config, inputs, pids, device)
        for tau, data in sorted(tau_data.items()):
            print(f'  tau={tau:2d} [{data["type"]:>8}] delta_mean={data["delta_norm_mean"]:.4f} '
                  f'delta_std={data["delta_norm_std"]:.4f} max_pos={data["delta_norm_max_pos"]}',
                  flush=True)

        # Experiment 2: Per-label accuracy across iterations
        analyze_representation_similarity(model, config, inputs, labels, pids, device)

        del model
        torch.cuda.empty_cache()

    print(f'\n=== Done ===', flush=True)


if __name__ == '__main__':
    main()
