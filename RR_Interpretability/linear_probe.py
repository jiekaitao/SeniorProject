"""
TRM Linear Probe: Does z_H encode reachability even if lm_head doesn't use it?

Train a lightweight linear probe on frozen z_H representations at each H-cycle
to predict: wall vs open vs reachable. If the probe succeeds where lm_head fails,
the iteration process builds useful spatial representations that the output head
doesn't exploit.
"""
import os, sys, torch, json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)
from utils.functions import load_model_class
import yaml


def load_model_and_data(ckpt_name, device='cuda', seed=42):
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
    config['batch_size'] = 16

    model_cls = load_model_class(arch_config['name'])
    model = model_cls(config)
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('step_')])
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    np.random.seed(seed)
    n = len(inputs)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:min(200, n)], idx[min(200, n):min(300, n)]

    return (model, config,
            torch.tensor(inputs[train_idx], dtype=torch.long, device=device),
            torch.tensor(labels[train_idx], dtype=torch.long, device=device),
            torch.tensor(puzzle_ids[train_idx], dtype=torch.long, device=device),
            torch.tensor(inputs[test_idx], dtype=torch.long, device=device),
            torch.tensor(labels[test_idx], dtype=torch.long, device=device),
            torch.tensor(puzzle_ids[test_idx], dtype=torch.long, device=device))


def extract_representations(model, config, inputs, pids, device='cuda'):
    """Extract z_H at each H-cycle, return dict[h_cycle] -> (N, seq_len, D)"""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
    total_len = config['seq_len'] + puzzle_emb_len
    B = inputs.shape[0]
    mini_bs = 16

    all_z_H = {h: [] for h in range(H_cycles)}

    for start in range(0, B, mini_bs):
        end = min(start + mini_bs, B)
        batch_in = inputs[start:end]
        batch_pid = pids[start:end]
        mb = batch_in.shape[0]

        with torch.no_grad():
            input_emb = inner._input_embeddings(batch_in, batch_pid)
            cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
            seq_info = {'cos_sin': cos_sin}
            z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(mb, total_len, -1).clone()
            z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(mb, total_len, -1).clone()

            for h_step in range(H_cycles):
                for l_step in range(L_cycles):
                    z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = inner.L_level(z_H, z_L, **seq_info)
                # Keep maze tokens only (skip puzzle embedding)
                all_z_H[h_step].append(z_H[:, puzzle_emb_len:].float().cpu())

    for h in range(H_cycles):
        all_z_H[h] = torch.cat(all_z_H[h], dim=0)  # (N, seq_len, D)

    return all_z_H


def train_linear_probe(z_H_train, labels_train, z_H_test, labels_test, hidden_size, n_classes=6):
    """Train a linear probe (no hidden layers) on frozen representations."""
    device = 'cpu'  # small enough for CPU
    probe = nn.Linear(hidden_size, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    # Flatten: (N, seq_len, D) -> (N*seq_len, D)
    X_train = z_H_train.reshape(-1, hidden_size)
    y_train = labels_train.reshape(-1)
    X_test = z_H_test.reshape(-1, hidden_size)
    y_test = labels_test.reshape(-1)

    # Subsample for speed if too large
    if X_train.shape[0] > 200000:
        idx = torch.randperm(X_train.shape[0])[:200000]
        X_train, y_train = X_train[idx], y_train[idx]

    n_train = X_train.shape[0]
    batch_size = 4096

    for epoch in range(20):
        perm = torch.randperm(n_train)
        total_loss = 0
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i+batch_size]
            logits = probe(X_train[batch_idx])
            loss = F.cross_entropy(logits, y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate
    with torch.no_grad():
        test_logits = probe(X_test)
        test_preds = test_logits.argmax(dim=-1)
        overall_acc = (test_preds == y_test).float().mean().item()

        per_label = {}
        for lv in range(n_classes):
            mask = (y_test == lv)
            if mask.sum() > 0:
                per_label[lv] = (test_preds[mask] == y_test[mask]).float().mean().item()

    return overall_acc, per_label


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt in ['ABLATION_BASELINE', 'ABLATION_FULL_COMBO', 'ABLATION_REDUCED_MLP']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt}', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            (model, config,
             train_in, train_lbl, train_pid,
             test_in, test_lbl, test_pid) = load_model_and_data(ckpt, device)

            hidden_size = config.get('hidden_size', 512)
            H_cycles = config.get('H_cycles', 3)

            print(f'  Extracting train representations...', flush=True)
            z_H_train = extract_representations(model, config, train_in, train_pid, device)
            print(f'  Extracting test representations...', flush=True)
            z_H_test = extract_representations(model, config, test_in, test_pid, device)

            # Also get lm_head accuracy for comparison
            print(f'\n  --- lm_head accuracy (baseline) ---', flush=True)
            ckpt_results = {}
            for h in range(H_cycles):
                with torch.no_grad():
                    lm_logits = model.inner.lm_head(z_H_test[h].to(device))
                    lm_preds = lm_logits.argmax(dim=-1).cpu()
                    lm_acc = (lm_preds == test_lbl.cpu()).float().mean().item()
                    lm_per_label = {}
                    for lv in range(6):
                        mask = (test_lbl.cpu() == lv)
                        if mask.sum() > 0:
                            lm_per_label[lv] = (lm_preds[mask.expand_as(lm_preds)] == test_lbl.cpu()[mask.expand_as(test_lbl.cpu())]).float().mean().item() if mask.any() else 0
                print(f'    H-cycle {h}: lm_head acc = {lm_acc:.4f}', flush=True)

            print(f'\n  --- Linear probe accuracy ---', flush=True)
            for h in range(H_cycles):
                acc, per_label = train_linear_probe(
                    z_H_train[h], train_lbl.cpu(), z_H_test[h], test_lbl.cpu(), hidden_size)
                print(f'    H-cycle {h}: probe acc = {acc:.4f}', flush=True)
                print(f'      Per-label: {per_label}', flush=True)

                # Also compare: probe on INITIAL z_H (before any iteration)
                ckpt_results[f'h_cycle_{h}'] = {
                    'probe_acc': acc,
                    'probe_per_label': {str(k): v for k, v in per_label.items()},
                }

            # Probe on z_H at initialization (before any iteration)
            print(f'\n  --- Probe on INITIAL z_H (no iteration) ---', flush=True)
            inner = model.inner
            puzzle_emb_len = getattr(inner, 'puzzle_emb_len', 16)
            total_len = config['seq_len'] + puzzle_emb_len
            with torch.no_grad():
                z_H_init = inner.H_init.unsqueeze(0).unsqueeze(0).expand(
                    1, total_len, -1)[:, puzzle_emb_len:].float().cpu()
                # All samples get the same init — repeat for train/test
                z_H_init_train = z_H_init.expand(train_in.shape[0], -1, -1)
                z_H_init_test = z_H_init.expand(test_in.shape[0], -1, -1)
            init_acc, init_per = train_linear_probe(
                z_H_init_train, train_lbl.cpu(), z_H_init_test, test_lbl.cpu(), hidden_size)
            print(f'    Init: probe acc = {init_acc:.4f}', flush=True)
            print(f'      Per-label: {init_per}', flush=True)
            ckpt_results['init'] = {
                'probe_acc': init_acc,
                'probe_per_label': {str(k): v for k, v in init_per.items()},
            }

            all_results[ckpt] = ckpt_results
            del model; torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f'ERROR: {e}', flush=True)
            traceback.print_exc()
            all_results[ckpt] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/linear_probe_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
