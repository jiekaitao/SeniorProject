"""
WHY does Brier+monotonicity (FULL_COMBO) converge faster than BASELINE?

Hypotheses to test:
H1: Brier loss calibrates confidence → model commits to predictions earlier
H2: Monotonicity loss prevents prediction oscillation across steps → smoother convergence
H3: RR mods create better gradient highways → representations change more purposefully per step
H4: FULL_COMBO learns more separable representations from the start → easier to classify

Experiments:
1. Confidence evolution: Compare confidence distributions at each ACT step
2. Prediction stability: How many cells flip their prediction between consecutive steps?
3. Representation geometry: cosine similarity between z_H at consecutive steps (how much does each step change?)
4. Separability: Fisher discriminant ratio for reachable vs open at each step
5. Logit calibration: ECE (Expected Calibration Error) at each step
6. Per-cell convergence speed: histogram of "when does each cell lock in?"
"""
import os, sys, torch, json, math
import numpy as np
from collections import defaultdict

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
    config['batch_size'] = 16

    model = load_model_class(ac['name'])(config)
    sd_fixed = {k.replace('model.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd_fixed, strict=False)
    model = model.to(device).eval()
    return model, config, inputs, labels, pids


def full_act_with_diagnostics(model, config, inputs_np, labels_np, pids_np,
                               device='cuda', n_samples=200):
    """Run full ACT, collect z_H + predictions + logits at every step."""
    inner = model.inner
    H_cycles = config.get('H_cycles', 3)
    L_cycles = config.get('L_cycles', 6)
    halt_max = config.get('halt_max_steps', 16)
    pel = getattr(inner, 'puzzle_emb_len', 16)
    tl = config['seq_len'] + pel
    bs = 16

    np.random.seed(42)
    idx = np.random.choice(len(inputs_np), min(n_samples, len(inputs_np)), replace=False)

    # Collect per-step diagnostics
    step_diagnostics = {s: {
        'logits': [],       # for confidence/calibration
        'preds': [],        # for stability
        'z_H_norm': [],     # representation magnitude
        'z_H_flat': [],     # flattened z_H for geometry analysis (subsample)
    } for s in range(halt_max)}

    all_labels = []
    prev_z_H = None
    step_cos_sims = defaultdict(list)  # step -> list of cosine sims with prev step

    for start in range(0, len(idx), bs):
        end = min(start + bs, len(idx))
        batch_idx = idx[start:end]
        B = len(batch_idx)

        inp = torch.tensor(inputs_np[batch_idx], dtype=torch.long, device=device)
        lbl = torch.tensor(labels_np[batch_idx], dtype=torch.long, device=device)
        pid = torch.tensor(pids_np[batch_idx], dtype=torch.long, device=device)

        if start == 0:
            all_labels.append(lbl[:, :].cpu())  # we'll trim puzzle_emb later

        with torch.no_grad():
            cos_sin = inner.rotary_emb() if hasattr(inner, 'rotary_emb') else None
            seq_info = {'cos_sin': cos_sin}
            input_emb = inner._input_embeddings(inp, pid)

            z_H = inner.H_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()
            z_L = inner.L_init.unsqueeze(0).unsqueeze(0).expand(B, tl, -1).clone()

            prev_z_H_batch = None

            for act_step in range(halt_max):
                for h_step in range(H_cycles):
                    for l_step in range(L_cycles):
                        z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                    z_H = inner.L_level(z_H, z_L, **seq_info)

                # Extract maze-only representations
                z_maze = z_H[:, pel:].float()  # (B, seq_len, D)
                logits = inner.lm_head(z_H)[:, pel:]  # (B, seq_len, vocab)
                preds = logits.argmax(dim=-1)

                # Store diagnostics (first batch only for z_H to save memory)
                step_diagnostics[act_step]['logits'].append(logits.cpu())
                step_diagnostics[act_step]['preds'].append(preds.cpu())
                step_diagnostics[act_step]['z_H_norm'].append(z_maze.norm(dim=-1).mean().item())

                if start == 0:
                    step_diagnostics[act_step]['z_H_flat'].append(z_maze.cpu())

                # Cosine similarity with previous step
                if prev_z_H_batch is not None:
                    cos = torch.nn.functional.cosine_similarity(
                        z_maze.reshape(-1, z_maze.shape[-1]),
                        prev_z_H_batch.reshape(-1, prev_z_H_batch.shape[-1]),
                        dim=-1
                    ).mean().item()
                    step_cos_sims[act_step].append(cos)

                prev_z_H_batch = z_maze.clone()

    all_labels = torch.cat(all_labels, dim=0) if all_labels else None

    return step_diagnostics, all_labels, step_cos_sims


def analyze_diagnostics(step_diagnostics, labels, step_cos_sims, config, model_name):
    """Compute all the diagnostic metrics."""
    halt_max = config.get('halt_max_steps', 16)
    seq_len = config['seq_len']
    results = {'model': model_name, 'steps': []}

    # Get labels for first batch (matches z_H_flat)
    lbl_flat = labels[0] if labels is not None and len(labels.shape) > 1 else labels

    prev_preds = None

    for step in range(halt_max):
        sd = step_diagnostics[step]
        logits_all = torch.cat(sd['logits'], dim=0)  # (N, seq_len, vocab)
        preds_all = torch.cat(sd['preds'], dim=0)    # (N, seq_len)
        N = logits_all.shape[0]

        probs_all = torch.softmax(logits_all.float(), dim=-1)

        # 1. Confidence statistics
        max_probs = probs_all.max(dim=-1).values  # (N, seq_len)
        avg_confidence = max_probs.mean().item()
        std_confidence = max_probs.std().item()

        # Confidence for reachable cells specifically (label 5)
        if lbl_flat is not None:
            reach_mask = (lbl_flat == 5)
            if reach_mask.any():
                reach_conf = max_probs[0][reach_mask[:max_probs.shape[1]]].mean().item() if max_probs.shape[0] > 0 else 0
            else:
                reach_conf = 0
        else:
            reach_conf = 0

        # 2. Prediction stability (flips from previous step)
        if prev_preds is not None:
            flips = (preds_all != prev_preds).float().mean().item()
        else:
            flips = 1.0  # first step: all new
        prev_preds = preds_all.clone()

        # 3. Representation change (cosine similarity with prev step)
        avg_cos = np.mean(step_cos_sims[step]) if step_cos_sims[step] else 0.0

        # 4. z_H norm
        avg_norm = np.mean(sd['z_H_norm'])

        # 5. Fisher discriminant ratio for reachable (5) vs open (2)
        fisher_ratio = 0.0
        if sd['z_H_flat'] and lbl_flat is not None:
            z_flat = torch.cat(sd['z_H_flat'], dim=0)  # (B, seq_len, D)
            z_2d = z_flat[0]  # first sample (seq_len, D)
            lbl_1d = lbl_flat[:z_2d.shape[0]]

            reach_z = z_2d[lbl_1d == 5]
            open_z = z_2d[lbl_1d == 2]
            if len(reach_z) > 1 and len(open_z) > 1:
                mu_r = reach_z.mean(dim=0)
                mu_o = open_z.mean(dim=0)
                var_r = reach_z.var(dim=0).mean()
                var_o = open_z.var(dim=0).mean()
                between = (mu_r - mu_o).pow(2).mean()
                within = (var_r + var_o) / 2 + 1e-8
                fisher_ratio = (between / within).item()

        # 6. ECE (Expected Calibration Error) - 10 bins
        ece = 0.0
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        flat_probs = max_probs.reshape(-1)
        flat_correct = (preds_all == preds_all).reshape(-1).float()  # placeholder — need true labels
        # Use first batch labels for ECE
        if lbl_flat is not None and preds_all.shape[0] > 0:
            first_preds = preds_all[0]
            first_probs = max_probs[0]
            first_correct = (first_preds == lbl_flat[:first_preds.shape[0]]).float()
            for b in range(n_bins):
                mask = (first_probs >= bin_boundaries[b]) & (first_probs < bin_boundaries[b+1])
                if mask.sum() > 0:
                    avg_conf = first_probs[mask].mean().item()
                    avg_acc = first_correct[mask].mean().item()
                    ece += mask.sum().item() * abs(avg_conf - avg_acc)
            ece /= first_probs.numel()

        # 7. Reachable accuracy
        reach_acc = 0.0
        if lbl_flat is not None and preds_all.shape[0] > 0:
            first_preds = preds_all[0]
            reach_mask = (lbl_flat[:first_preds.shape[0]] == 5)
            if reach_mask.sum() > 0:
                reach_acc = (first_preds[reach_mask] == 5).float().mean().item()

        # 8. Entropy of predictions (how uncertain is the model?)
        entropy = -(probs_all * (probs_all + 1e-10).log()).sum(dim=-1).mean().item()

        step_data = {
            'act_step': step,
            'h_iterations': (step + 1) * config.get('H_cycles', 3),
            'avg_confidence': round(avg_confidence, 4),
            'std_confidence': round(std_confidence, 4),
            'reachable_confidence': round(reach_conf, 4),
            'prediction_flip_rate': round(flips, 4),
            'cos_sim_with_prev': round(avg_cos, 6),
            'representation_change': round(1 - avg_cos, 6) if avg_cos > 0 else 1.0,
            'z_H_norm': round(avg_norm, 4),
            'fisher_discriminant': round(fisher_ratio, 4),
            'ece': round(ece, 4),
            'reachable_accuracy': round(reach_acc, 4),
            'prediction_entropy': round(entropy, 4),
        }
        results['steps'].append(step_data)

        print(f'  Step {step:2d} (H={step_data["h_iterations"]:2d}): '
              f'conf={avg_confidence:.3f} flips={flips:.3f} '
              f'cos={avg_cos:.4f} fisher={fisher_ratio:.2f} '
              f'ece={ece:.3f} reach={reach_acc:.3f}', flush=True)

    return results


def compute_convergence_histogram(step_diagnostics, config):
    """When does each cell lock in its final prediction?"""
    halt_max = config.get('halt_max_steps', 16)
    all_preds = []
    for step in range(halt_max):
        preds = torch.cat(step_diagnostics[step]['preds'], dim=0)
        all_preds.append(preds)

    # Stack: (halt_max, N, seq_len)
    pred_stack = torch.stack(all_preds, dim=0)
    final_preds = pred_stack[-1]  # (N, seq_len)

    # For each position, find first step where prediction matches final and stays
    N, S = final_preds.shape
    convergence = torch.full((N, S), halt_max, dtype=torch.long)

    for step in range(halt_max):
        matches_final = (pred_stack[step:] == final_preds.unsqueeze(0)).all(dim=0)  # (N, S)
        newly_converged = matches_final & (convergence == halt_max)
        convergence[newly_converged] = step

    # Histogram
    hist = {}
    for step in range(halt_max + 1):
        count = (convergence == step).sum().item()
        hist[step] = count

    return hist, convergence


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for ckpt_name in ['ABLATION_FULL_COMBO', 'ABLATION_BASELINE',
                       'ABLATION_BRIER_ONLY', 'ABLATION_MONO_ONLY', 'ABLATION_SOFTMAX_ONLY']:
        print(f'\n{"="*60}', flush=True)
        print(f'  {ckpt_name} — Mechanistic Investigation', flush=True)
        print(f'{"="*60}', flush=True)

        model, config, inputs, labels, pids = load_model(ckpt_name, device)

        print(f'\n  --- Running full ACT with diagnostics (200 samples) ---', flush=True)
        step_diag, lbl_tensor, cos_sims = full_act_with_diagnostics(
            model, config, inputs, labels, pids, device, n_samples=200)

        print(f'\n  --- Analyzing diagnostics ---', flush=True)
        results = analyze_diagnostics(step_diag, lbl_tensor, cos_sims, config, ckpt_name)

        print(f'\n  --- Convergence histogram ---', flush=True)
        hist, conv_tensor = compute_convergence_histogram(step_diag, config)
        results['convergence_histogram'] = hist
        total_cells = sum(hist.values())
        cumulative = 0
        for step in sorted(hist.keys()):
            cumulative += hist[step]
            pct = cumulative / total_cells * 100
            if hist[step] > 0:
                print(f'    Step {step:2d}: {hist[step]:6d} cells converge ({pct:.1f}% cumulative)', flush=True)

        all_results[ckpt_name] = results
        del model; torch.cuda.empty_cache()

    # Comparative summary
    print(f'\n{"="*60}', flush=True)
    print(f'  COMPARATIVE SUMMARY: FULL_COMBO vs BASELINE', flush=True)
    print(f'{"="*60}', flush=True)

    fc = all_results.get('ABLATION_FULL_COMBO', {}).get('steps', [])
    bl = all_results.get('ABLATION_BASELINE', {}).get('steps', [])

    if fc and bl:
        print(f'\n  {"Step":>4s} | {"FC conf":>8s} {"BL conf":>8s} | {"FC flips":>9s} {"BL flips":>9s} | {"FC fisher":>10s} {"BL fisher":>10s} | {"FC reach":>9s} {"BL reach":>9s}')
        print(f'  {"-"*90}')
        for i in range(min(len(fc), len(bl))):
            f, b = fc[i], bl[i]
            print(f'  {i:4d} | {f["avg_confidence"]:8.3f} {b["avg_confidence"]:8.3f} | '
                  f'{f["prediction_flip_rate"]:9.4f} {b["prediction_flip_rate"]:9.4f} | '
                  f'{f["fisher_discriminant"]:10.2f} {b["fisher_discriminant"]:10.2f} | '
                  f'{f["reachable_accuracy"]:9.3f} {b["reachable_accuracy"]:9.3f}')

    # Save all results
    save_path = os.path.join(VIZ_DIR, 'brier_mechanism_investigation.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved: {save_path}', flush=True)

    # Interpret findings
    print(f'\n{"="*60}', flush=True)
    print(f'  INTERPRETATION', flush=True)
    print(f'{"="*60}', flush=True)
    if fc and bl:
        # H1: Confidence
        fc_conf_0 = fc[0]['avg_confidence']
        bl_conf_0 = bl[0]['avg_confidence']
        print(f'\n  H1 (Brier → earlier commitment):')
        print(f'    Step 0 confidence: FC={fc_conf_0:.3f} vs BL={bl_conf_0:.3f}')
        if fc_conf_0 > bl_conf_0:
            print(f'    SUPPORTED: FULL_COMBO is more confident from the start')
        else:
            print(f'    NOT SUPPORTED: BASELINE is equally or more confident')

        # H2: Prediction stability
        fc_flips_1 = fc[1]['prediction_flip_rate'] if len(fc) > 1 else 1
        bl_flips_1 = bl[1]['prediction_flip_rate'] if len(bl) > 1 else 1
        print(f'\n  H2 (Monotonicity → fewer flips):')
        print(f'    Step 1 flip rate: FC={fc_flips_1:.4f} vs BL={bl_flips_1:.4f}')
        if fc_flips_1 < bl_flips_1:
            print(f'    SUPPORTED: FULL_COMBO has fewer prediction flips')
        else:
            print(f'    NOT SUPPORTED: comparable flip rates')

        # H3: Representation change
        fc_change_1 = fc[1].get('representation_change', 0) if len(fc) > 1 else 0
        bl_change_1 = bl[1].get('representation_change', 0) if len(bl) > 1 else 0
        print(f'\n  H3 (Better gradient highways):')
        print(f'    Step 1 repr change: FC={fc_change_1:.6f} vs BL={bl_change_1:.6f}')

        # H4: Separability
        fc_fisher_0 = fc[0]['fisher_discriminant']
        bl_fisher_0 = bl[0]['fisher_discriminant']
        print(f'\n  H4 (More separable from the start):')
        print(f'    Step 0 Fisher ratio: FC={fc_fisher_0:.2f} vs BL={bl_fisher_0:.2f}')
        if fc_fisher_0 > bl_fisher_0:
            print(f'    SUPPORTED: FULL_COMBO has better reachable/open separation from step 0')
        else:
            print(f'    NOT SUPPORTED: comparable separability')

    print(f'\n=== Investigation complete ===', flush=True)


if __name__ == '__main__':
    main()
