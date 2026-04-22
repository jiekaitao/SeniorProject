"""
PROPER TRM Evaluation — Uses the FULL ACT loop (16 steps × 3 H-cycles = 48 iterations).

Previous experiments only ran 3 H-cycles (1 ACT step). The trained model uses 16 ACT steps.
This explains why we got 25% accuracy instead of the paper's 97%.

This script:
1. Runs the model through its FULL forward pass (16 ACT steps)
2. Verifies token accuracy matches paper (~97%)
3. Probes z_H at multiple ACT steps to see when reachability emerges
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
    # Derive vocab_size from checkpoint to avoid mismatch
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )
    sd_peek = torch.load(os.path.join(ckpt_dir, ckpt_files[-1]), map_location='cpu')
    embed_key = [k for k in sd_peek.keys() if 'embed_tokens' in k][0]
    config['vocab_size'] = sd_peek[embed_key].shape[0]
    del sd_peek
    config['num_puzzle_identifiers'] = int(pids.max()) + 1
    config['batch_size'] = 16

    model = load_model_class(ac['name'])(config)
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )
    latest = ckpt_files[-1]
    print(f'  Loading {ckpt_name} checkpoint: {latest}', flush=True)
    sd = torch.load(os.path.join(ckpt_dir, latest), map_location=device)
    # Strip "model." prefix — checkpoints saved from DDP/EMA wrapper
    sd_fixed = {k.replace('model.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd_fixed, strict=False)
    print(f'  Loaded: {len(sd_fixed) - len(missing)} keys, missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
    if missing:
        print(f'  Missing keys sample: {missing[:3]}', flush=True)
    model = model.to(device).eval()
    return model, config, inputs, labels, pids


def run_full_act(model, config, inputs_np, labels_np, pids_np, device='cuda',
                 n_samples=200, probe_steps=None):
    """Run the FULL ACT loop and collect z_H at specified steps."""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    halt_max = config.get('halt_max_steps', 16)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel
    hidden_size = config.get('hidden_size', 512)
    bs = 16

    if probe_steps is None:
        probe_steps = [0, 1, 3, 7, 15]  # ACT steps to probe

    np.random.seed(42)
    idx = np.random.choice(len(inputs_np), n_samples, replace=False)

    # Collect final predictions and z_H at probe steps
    all_preds = []
    all_labels = []
    z_H_at_step = {s: [] for s in probe_steps}

    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        batch_idx = idx[start:end]
        B = len(batch_idx)

        inp = torch.tensor(inputs_np[batch_idx], dtype=torch.long, device=device)
        lbl = torch.tensor(labels_np[batch_idx], dtype=torch.long, device=device)
        pid = torch.tensor(pids_np[batch_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
            seq_info = {'cos_sin': cos_sin}
            input_emb = inner._input_embeddings(inp, pid)

            # Initialize carry
            z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
            z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()

            # Run FULL ACT loop: halt_max_steps × (H_cycles × L_cycles)
            for act_step in range(halt_max):
                # Each ACT step = one full inner forward (H_cycles-1 no-grad + 1 with-grad)
                # At eval, we just run all H_cycles
                for h_step in range(H_cycles):
                    for l_step in range(L_cycles):
                        z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                    z_H = inner.L_level(z_H, z_L, **seq_info)

                # Collect z_H at probe steps
                if act_step in probe_steps:
                    z_H_at_step[act_step].append(z_H[:, pel:].float().cpu())

            # Final predictions
            logits = inner.lm_head(z_H)[:, pel:]
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(lbl.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    for s in probe_steps:
        z_H_at_step[s] = torch.cat(z_H_at_step[s], dim=0)

    return all_preds, all_labels, z_H_at_step


def train_probe(X_train, y_train, X_test, y_test, hidden_size, probe_type='mlp', epochs=30):
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
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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

    for ckpt_name in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt_name} — FULL ACT EVALUATION', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            model, config, inputs, labels, pids = load_model(ckpt_name, device)
            halt_max = config.get('halt_max_steps', 16)
            H_cycles = config.get('H_cycles', 3)
            hidden_size = config.get('hidden_size', 512)
            print(f'  ACT steps: {halt_max}, H_cycles per step: {H_cycles}', flush=True)
            print(f'  Total H-iterations: {halt_max * H_cycles}', flush=True)

            # Split data
            np.random.seed(42)
            perm = np.random.permutation(len(inputs))
            train_idx, test_idx = perm[:300], perm[300:400]

            # Run full ACT on test set
            probe_steps = list(range(halt_max))  # Probe ALL steps
            print(f'\n  Running FULL ACT ({halt_max} steps)...', flush=True)
            preds, lbls, z_H_steps = run_full_act(
                model, config, inputs[test_idx], labels[test_idx], pids[test_idx],
                device, n_samples=100, probe_steps=probe_steps)

            # Final accuracy
            correct = (preds == lbls)
            token_acc = correct.float().mean().item()
            exact_acc = correct.all(dim=1).float().mean().item()
            per_label_acc = {}
            for lv in range(6):
                mask = (lbls == lv)
                if mask.sum() > 0:
                    per_label_acc[lv] = correct[mask].float().mean().item()

            print(f'\n  === FINAL ACCURACY (after {halt_max} ACT steps) ===', flush=True)
            print(f'  Token accuracy: {token_acc:.4f} (paper: ~0.97)', flush=True)
            print(f'  Exact accuracy: {exact_acc:.4f}', flush=True)
            print(f'  Per-label:', flush=True)
            for lv in sorted(per_label_acc.keys()):
                print(f'    Label {lv}: {per_label_acc[lv]:.4f}', flush=True)

            ckpt_results = {
                'token_acc': token_acc,
                'exact_acc': exact_acc,
                'per_label': {str(k): v for k, v in per_label_acc.items()},
                'act_steps': halt_max,
                'total_H_iters': halt_max * H_cycles,
            }

            # Measure accuracy and probe at each ACT step
            print(f'\n  === ACCURACY BY ACT STEP ===', flush=True)

            # Also run full ACT on train set for probes
            print(f'  Running FULL ACT on train set for probes...', flush=True)
            _, train_lbls, z_H_train_steps = run_full_act(
                model, config, inputs[train_idx], labels[train_idx], pids[train_idx],
                device, n_samples=200, probe_steps=probe_steps)

            test_lbls_cpu = lbls.cpu()
            train_lbls_cpu = train_lbls.cpu()  # FIX: use aligned labels from run_full_act, not sequential slice

            step_results = {}
            for step in probe_steps:
                if step not in z_H_steps:
                    continue
                z_test = z_H_steps[step]
                z_train = z_H_train_steps[step]

                # lm_head accuracy at this step
                with torch.no_grad():
                    lm_logits = model.inner.lm_head(z_test.to(device))
                    lm_preds = lm_logits.argmax(dim=-1).cpu()
                    lm_acc = (lm_preds == test_lbls_cpu).float().mean().item()
                    lm_reach = 0.0
                    reach_mask = (test_lbls_cpu == 5)
                    if reach_mask.sum() > 0:
                        lm_reach = (lm_preds[reach_mask] == test_lbls_cpu[reach_mask]).float().mean().item()

                # MLP probe
                mlp_acc, mlp_per = train_probe(z_train, train_lbls_cpu,
                                                z_test, test_lbls_cpu, hidden_size, 'mlp', 30)

                total_h = (step + 1) * H_cycles
                print(f'  ACT step {step:2d} (H-iter {total_h:2d}): '
                      f'lm_head={lm_acc:.4f}  probe={mlp_acc:.4f}  '
                      f'reach_lm={lm_reach:.4f}  reach_probe={mlp_per.get(5, 0):.4f}', flush=True)

                step_results[step] = {
                    'lm_acc': lm_acc,
                    'probe_acc': mlp_acc,
                    'lm_reach': lm_reach,
                    'probe_reach': mlp_per.get(5, 0),
                    'probe_per_label': {str(k): v for k, v in mlp_per.items()},
                }

            ckpt_results['by_step'] = step_results
            all_results[ckpt_name] = ckpt_results
            del model; torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f'ERROR: {e}', flush=True)
            traceback.print_exc()
            all_results[ckpt_name] = {'error': str(e)}

    save_path = '/home/jietao/RR/SeniorProject/RR_Interpretability/proper_eval_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
