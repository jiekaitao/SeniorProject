"""
Ensemble Voting — Combine predictions from multiple trained solvers.

If individual solver errors are uncorrelated, majority voting can break 72%.
If errors ARE correlated (all solvers fail on the same hard mazes), ensemble
won't help — confirms a fundamental capability limit.

Loads top-N checkpoints from best-of-N training runs.
For each eval sample, runs all N solvers and takes majority vote.
"""
import os, sys, torch, json, random, time, argparse
from collections import Counter
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def predict_with_solver(solver, base_model, lm_model, tokenizer, text, K=1):
    """Get prediction from one solver."""
    prompt = text + "\nAnswer:"
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        emb = lm_model.embed_tokens(enc['input_ids'])
        mem = solver(emb, K_inner=4, K_outer=K, grad_last_only=False)
        dec = torch.cat([mem, emb], dim=1)
        T = dec.shape[1]
        pid = torch.arange(T, device=device).unsqueeze(0)
        pe = lm_model.rotary_emb(dec, pid)
        h = dec
        for layer in lm_model.layers:
            h = layer(h, position_embeddings=pe)
        h = lm_model.norm(h)
        lg = base_model.lm_head(h)
        pred = tokenizer.decode([lg[0, -1].argmax().item()]).strip()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-n', type=int, default=5, help='Number of solvers to ensemble')
    args = parser.parse_args()

    print('Loading model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    maze_data = load_maze_nav()
    random.seed(0)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    eval_idx = indices[1000:]

    # Find best checkpoints
    ckpt_scores = []
    for f in os.listdir(RESULTS_DIR):
        if f.startswith('spatialeval_bestofn_') and f.endswith('.json'):
            path = os.path.join(RESULTS_DIR, f)
            d = json.load(open(path))
            k1_acc = d.get('results', {}).get('K=1', {}).get('accuracy', 0)
            ckpt_name = f.replace('spatialeval_', 'solver_').replace('.json', '.pt')
            ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)
            if os.path.exists(ckpt_path) and k1_acc > 0.5:
                ckpt_scores.append((k1_acc, ckpt_path, d.get('tag', '?')))

    ckpt_scores.sort(reverse=True)
    top_n = min(args.top_n, len(ckpt_scores))
    print(f'\nTop {top_n} checkpoints:', flush=True)
    for acc, path, tag in ckpt_scores[:top_n]:
        print(f'  {tag}: {acc:.4f}', flush=True)

    if top_n < 3:
        print('Not enough checkpoints for meaningful ensemble!', flush=True)
        return

    # Load all solvers
    solvers = []
    for _, ckpt_path, tag in ckpt_scores[:top_n]:
        solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                            n_L_layers=2, n_memory_slots=32).to(device=device, dtype=torch.bfloat16)
        solver.load_state_dict(torch.load(ckpt_path, map_location=device))
        solver.eval()
        solvers.append((solver, tag))
        print(f'  Loaded: {tag}', flush=True)

    # Evaluate: individual + ensemble
    print(f'\n=== Evaluating {len(eval_idx)} samples ===', flush=True)
    t0 = time.time()

    individual_correct = [0] * top_n
    ensemble_correct = {3: 0, 5: 0, 'all': 0}
    per_sample_preds = []
    n_eval = len(eval_idx)

    for eval_i, idx in enumerate(eval_idx):
        sample = maze_data[idx]
        text = sample['text'][:1500]
        oracle = sample['oracle_option'].strip().upper()

        preds = []
        for si, (solver, tag) in enumerate(solvers):
            pred = predict_with_solver(solver, base_model, lm_model, tokenizer, text, K=1)
            is_correct = oracle in pred.upper()[:10]
            if is_correct:
                individual_correct[si] += 1
            preds.append(pred.upper()[:3])

        per_sample_preds.append({
            'idx': idx, 'oracle': oracle, 'preds': preds,
            'individual_correct': [oracle in p[:10] for p in preds]
        })

        # Majority voting for different ensemble sizes
        for ens_size in [3, 5]:
            if ens_size <= top_n:
                votes = preds[:ens_size]
                # Extract first letter for voting
                letters = [p[0] if p else '?' for p in votes]
                majority = Counter(letters).most_common(1)[0][0]
                if majority == oracle[0]:
                    ensemble_correct[ens_size] += 1

        # All-solver ensemble
        letters = [p[0] if p else '?' for p in preds]
        majority = Counter(letters).most_common(1)[0][0]
        if majority == oracle[0]:
            ensemble_correct['all'] += 1

        if (eval_i + 1) % 100 == 0:
            ens_acc = ensemble_correct['all'] / (eval_i + 1)
            best_ind = max(individual_correct[i] / (eval_i + 1) for i in range(top_n))
            print(f'  {eval_i+1}/{n_eval} | best_individual={best_ind:.3f} | ensemble_{top_n}={ens_acc:.3f} | {time.time()-t0:.0f}s', flush=True)

    # Results
    print(f'\n=== Final Results ===', flush=True)
    results = {}
    for si, (solver, tag) in enumerate(solvers):
        acc = individual_correct[si] / n_eval
        results[f'individual_{tag}'] = acc
        print(f'  {tag}: {acc:.4f} ({individual_correct[si]}/{n_eval})', flush=True)

    for ens_size in [3, 5, 'all']:
        if ens_size == 'all' or ens_size <= top_n:
            acc = ensemble_correct[ens_size] / n_eval
            results[f'ensemble_{ens_size}'] = acc
            print(f'  Ensemble-{ens_size}: {acc:.4f} ({ensemble_correct[ens_size]}/{n_eval})', flush=True)

    # Error analysis: how many samples do ALL solvers get wrong?
    all_wrong = sum(1 for s in per_sample_preds if not any(s['individual_correct']))
    any_right = sum(1 for s in per_sample_preds if any(s['individual_correct']))
    all_right = sum(1 for s in per_sample_preds if all(s['individual_correct']))
    print(f'\n  Error correlation:', flush=True)
    print(f'    All {top_n} correct: {all_right}/{n_eval} ({all_right/n_eval:.3f})', flush=True)
    print(f'    At least 1 correct: {any_right}/{n_eval} ({any_right/n_eval:.3f})', flush=True)
    print(f'    All {top_n} wrong: {all_wrong}/{n_eval} ({all_wrong/n_eval:.3f})', flush=True)

    results['all_correct'] = all_right / n_eval
    results['any_correct'] = any_right / n_eval
    results['all_wrong'] = all_wrong / n_eval

    # Agreement analysis
    agreement_counts = Counter()
    for s in per_sample_preds:
        n_correct = sum(s['individual_correct'])
        agreement_counts[n_correct] += 1
    print(f'\n  Agreement distribution:', flush=True)
    for k in sorted(agreement_counts.keys()):
        print(f'    {k}/{top_n} correct: {agreement_counts[k]} samples', flush=True)

    results['agreement'] = {str(k): v for k, v in agreement_counts.items()}

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': f'ensemble_top{top_n}',
        'method': 'ensemble_majority_vote',
        'n_solvers': top_n,
        'checkpoints': [tag for _, tag in solvers],
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_ensemble_top{top_n}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'\nSaved: spatialeval_ensemble_top{top_n}.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
