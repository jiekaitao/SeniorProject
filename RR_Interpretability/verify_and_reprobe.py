"""
CRITICAL FIX: Previous experiments loaded step_9765 (half-trained) instead of
step_19530 (fully trained) due to alphabetical sort bug.

This script:
1. Loads the CORRECT final checkpoint (step_19530)
2. Verifies token accuracy matches Alexia's paper (~97% token, ~30% exact)
3. Re-runs linear AND MLP probes on the fully-trained model
"""
import os, sys, torch, json
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TRM_ROOT = '/blue/cis4914/jietao/SeniorProject/RR_TRM'
sys.path.insert(0, TRM_ROOT)
from utils.functions import load_model_class
import yaml


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def load_correct_checkpoint(ckpt_name, device='cuda'):
    """Load the FINAL checkpoint (numeric sort, not alphabetical)."""
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
    config['vocab_size'] = int(inputs.max()) + 1
    config['num_puzzle_identifiers'] = int(pids.max()) + 1
    config['batch_size'] = 16

    model = load_model_class(ac['name'])(config)

    # CORRECT: numeric sort
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )
    latest = ckpt_files[-1]
    print(f'  Loading checkpoint: {latest} (step {latest.split("_")[1]})', flush=True)

    sd = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, config, inputs, labels, pids


def verify_accuracy(model, config, inputs, labels, pids, device='cuda', n_samples=200):
    """Verify token accuracy matches paper's reported ~97%."""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel
    sl = config['seq_len']
    bs = 16

    np.random.seed(42)
    idx = np.random.choice(len(inputs), n_samples, replace=False)
    inp = torch.tensor(inputs[idx], dtype=torch.long, device=device)
    lbl = torch.tensor(labels[idx], dtype=torch.long, device=device)
    pid = torch.tensor(pids[idx], dtype=torch.long, device=device)

    all_correct = 0
    all_total = 0
    exact_correct = 0
    per_label_correct = {}
    per_label_total = {}

    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        bi, bl, bp = inp[start:end], lbl[start:end], pid[start:end]
        B = bi.shape[0]

        with torch.no_grad():
            ie = inner._input_embeddings(bi, bp)
            cs = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
            si = {'cos_sin': cs}
            zH = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
            zL = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()

            for h in range(H_cycles):
                for l in range(L_cycles):
                    zL = inner.L_level(zL, zH + ie, **si)
                zH = inner.L_level(zH, zL, **si)

            logits = inner.lm_head(zH)[:, pel:]
            preds = logits.argmax(dim=-1)

            correct = (preds == bl)
            all_correct += correct.sum().item()
            all_total += correct.numel()

            for b in range(B):
                if correct[b].all():
                    exact_correct += 1

            for lv in range(6):
                mask = (bl == lv)
                if mask.sum() > 0:
                    if lv not in per_label_correct:
                        per_label_correct[lv] = 0
                        per_label_total[lv] = 0
                    per_label_correct[lv] += correct[mask].sum().item()
                    per_label_total[lv] += mask.sum().item()

    token_acc = all_correct / all_total
    exact_acc = exact_correct / n_samples

    print(f'\n  === Verification (n={n_samples}) ===', flush=True)
    print(f'  Token accuracy: {token_acc:.4f} (paper reports ~0.97)', flush=True)
    print(f'  Exact accuracy: {exact_acc:.4f} (paper reports ~0.09-0.31)', flush=True)
    print(f'  Per-label accuracy:', flush=True)
    for lv in sorted(per_label_total.keys()):
        acc = per_label_correct[lv] / per_label_total[lv]
        print(f'    Label {lv}: {acc:.4f} (n={per_label_total[lv]})', flush=True)

    return token_acc, exact_acc, {lv: per_label_correct[lv]/per_label_total[lv] for lv in per_label_total}


def extract_z_H(model, config, inputs, pids, data_idx, device='cuda', bs=16):
    """Extract z_H at each H-cycle."""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel

    inp = torch.tensor(inputs[data_idx], dtype=torch.long, device=device)
    pid = torch.tensor(pids[data_idx], dtype=torch.long, device=device)

    z_H_all = {h: [] for h in range(H_cycles)}
    for start in range(0, len(data_idx), bs):
        end = min(start + bs, len(data_idx))
        bi, bp = inp[start:end], pid[start:end]
        B = bi.shape[0]
        with torch.no_grad():
            ie = inner._input_embeddings(bi, bp)
            cs = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
            si = {'cos_sin': cs}
            zH = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
            zL = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
            for h in range(H_cycles):
                for l in range(L_cycles):
                    zL = inner.L_level(zL, zH + ie, **si)
                zH = inner.L_level(zH, zL, **si)
                z_H_all[h].append(zH[:, pel:].float().cpu())
    for h in range(H_cycles):
        z_H_all[h] = torch.cat(z_H_all[h], dim=0)
    return z_H_all


def train_probe(X_train, y_train, X_test, y_test, hidden_size, probe_type='linear', epochs=30):
    """Train linear or MLP probe."""
    if probe_type == 'linear':
        probe = nn.Linear(hidden_size, 6)
    else:
        probe = MLPProbe(hidden_size, hidden_dim=256, n_classes=6)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    X_tr = X_train.reshape(-1, hidden_size)
    y_tr = y_train.reshape(-1)
    X_te = X_test.reshape(-1, hidden_size)
    y_te = y_test.reshape(-1)

    if X_tr.shape[0] > 200000:
        si = torch.randperm(X_tr.shape[0])[:200000]
        X_tr, y_tr = X_tr[si], y_tr[si]

    for epoch in range(epochs):
        perm = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 4096):
            bi = perm[i:i+4096]
            logits = probe(X_tr[bi])
            loss = F.cross_entropy(logits, y_tr[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_te).argmax(dim=-1)
        overall = (preds == y_te).float().mean().item()
        per_label = {}
        for lv in range(6):
            mask = (y_te == lv)
            if mask.sum() > 0:
                per_label[lv] = (preds[mask] == y_te[mask]).float().mean().item()
    return overall, per_label


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt_name in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE', 'ABLATION_REDUCED_MLP']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt_name} (CORRECT checkpoint)', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            model, config, inputs, labels, pids = load_correct_checkpoint(ckpt_name, device)
            hidden_size = config.get('hidden_size', 512)
            H_cycles = config.get('H_cycles', 3)

            # Step 1: Verify accuracy matches paper
            token_acc, exact_acc, label_accs = verify_accuracy(
                model, config, inputs, labels, pids, device, n_samples=200)

            # Step 2: Extract representations
            np.random.seed(42)
            idx = np.random.permutation(len(inputs))
            train_idx, test_idx = idx[:300], idx[300:400]

            print(f'\n  Extracting representations (step_19530)...', flush=True)
            z_train = extract_z_H(model, config, inputs, pids, train_idx, device)
            z_test = extract_z_H(model, config, inputs, pids, test_idx, device)
            lbl_train = torch.tensor(labels[train_idx], dtype=torch.long)
            lbl_test = torch.tensor(labels[test_idx], dtype=torch.long)

            ckpt_results = {
                'token_acc': token_acc,
                'exact_acc': exact_acc,
                'label_accs': {str(k): v for k, v in label_accs.items()},
            }

            # Step 3: Linear and MLP probes at each H-cycle
            for h in range(H_cycles):
                print(f'\n  --- H-cycle {h} ---', flush=True)

                lin_acc, lin_per = train_probe(z_train[h], lbl_train, z_test[h], lbl_test,
                                               hidden_size, 'linear')
                mlp_acc, mlp_per = train_probe(z_train[h], lbl_train, z_test[h], lbl_test,
                                               hidden_size, 'mlp', epochs=50)

                print(f'    Linear probe: {lin_acc:.4f}  reachable(5)={lin_per.get(5, "N/A")}', flush=True)
                print(f'    MLP probe:    {mlp_acc:.4f}  reachable(5)={mlp_per.get(5, "N/A")}', flush=True)
                print(f'    Linear per-label: {lin_per}', flush=True)
                print(f'    MLP per-label:    {mlp_per}', flush=True)

                ckpt_results[f'h_cycle_{h}'] = {
                    'linear_acc': lin_acc, 'linear_per_label': {str(k): v for k, v in lin_per.items()},
                    'mlp_acc': mlp_acc, 'mlp_per_label': {str(k): v for k, v in mlp_per.items()},
                }

            all_results[ckpt_name] = ckpt_results
            del model; torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f'ERROR: {e}', flush=True)
            traceback.print_exc()
            all_results[ckpt_name] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/verify_reprobe_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
