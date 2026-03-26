#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_bayesian_%j.log
#SBATCH --job-name=deeppass_bayes

# Bayesian optimization for per-layer alpha on triple (0,7)+(20,27)+(45,52)
# Compare: can 50 Bayesian evals match 300 grid search evals?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Bayesian Alpha Optimization ==="
echo "Started: $(date)"

# Install optuna if needed
$PYTHON -m pip install optuna -q 2>/dev/null

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCKS = [(0, 7), (20, 27), (45, 52)]
MAX_EVALS = 60

print('=' * 70)
print('BAYESIAN ALPHA OPTIMIZATION (Optuna)')
print(f'Triple: {BLOCKS}')
print(f'Budget: {MAX_EVALS} evaluations')
print(f'Comparison: grid search needed ~300 evals to reach 84.07')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def generate_per_layer_alpha(prompt, blocks, layer_alphas, max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)
    layer_order = build_order(sorted_blocks, N)
    step_alphas = {}
    for block in sorted_blocks:
        i, j = block
        block_layers = list(range(i, j))
        count = {}
        offset = 0
        for step, idx in enumerate(layer_order):
            if idx in block_layers:
                count[idx] = count.get(idx, 0) + 1
                if count[idx] == 2:
                    key = (block, offset)
                    step_alphas[step] = layer_alphas.get(key, 1.0)
                    offset += 1
    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                h_before = h.clone()
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h_after = out[0] if isinstance(out, tuple) else out
                if step_idx in step_alphas:
                    alpha = step_alphas[step_idx]
                    h = h_before + alpha * (h_after - h_before)
                else:
                    h = h_after
            h = inner.norm(h)
            logits = model.lm_head(h)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

eval_count = [0]
best_so_far = [0]
history = []

def objective(trial):
    # Sample alphas for each layer in each block
    layer_alphas = {}
    for block in BLOCKS:
        i, j = block
        block_size = j - i
        for offset in range(block_size):
            if block == (0, 7):
                a = trial.suggest_float(f'b0_L{offset}', 0.0, 1.5)
            elif block == (20, 27):
                a = trial.suggest_float(f'b1_L{offset}', 0.0, 0.5)
            else:  # (45, 52)
                a = trial.suggest_float(f'b2_L{offset}', 0.3, 1.5)
            layer_alphas[(block, offset)] = a

    gen = lambda p: generate_per_layer_alpha(p, BLOCKS, layer_alphas, max_new_tokens=64)
    gen_long = lambda p: generate_per_layer_alpha(p, BLOCKS, layer_alphas, max_new_tokens=128)
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5

    eval_count[0] += 1
    if combined > best_so_far[0]:
        best_so_far[0] = combined
        alphas_str = ', '.join(f'{k[0]}L{k[1]}={v:.2f}' for k, v in sorted(layer_alphas.items()))
        print(f'  [{eval_count[0]:3d}/{MAX_EVALS}] NEW BEST: {combined:.2f} (math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f})', flush=True)
    elif eval_count[0] % 10 == 0:
        print(f'  [{eval_count[0]:3d}/{MAX_EVALS}] current: {combined:.2f} best: {best_so_far[0]:.2f}', flush=True)

    history.append({
        'eval': eval_count[0],
        'combined': combined,
        'math': math_r['score'],
        'eq': eq_r['score'],
        'alphas': {f'{k[0]}_{k[1]}': v for k, v in layer_alphas.items()},
    })

    return combined

# Run Bayesian optimization
print(f'\\n--- Starting Bayesian optimization ({MAX_EVALS} evals) ---', flush=True)
t0 = time.time()

study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15))

# Seed with known good alphas
study.enqueue_trial({
    'b0_L0': 0.9, 'b0_L1': 0.9, 'b0_L2': 0.9, 'b0_L3': 0.9, 'b0_L4': 0.9, 'b0_L5': 0.9, 'b0_L6': 0.9,
    'b1_L0': 0.15, 'b1_L1': 0.15, 'b1_L2': 0.15, 'b1_L3': 0.15, 'b1_L4': 0.15, 'b1_L5': 0.15, 'b1_L6': 0.15,
    'b2_L0': 1.1, 'b2_L1': 1.0, 'b2_L2': 0.5, 'b2_L3': 1.3, 'b2_L4': 1.0, 'b2_L5': 0.9, 'b2_L6': 1.1,
})

study.optimize(objective, n_trials=MAX_EVALS)

elapsed = time.time() - t0
best_trial = study.best_trial

print(f'\\n{\"=\" * 70}')
print('BAYESIAN OPTIMIZATION RESULTS')
print(f'{\"=\" * 70}')
print(f'Best combined: {best_trial.value:.2f}')
print(f'Found at eval: {best_trial.number + 1}/{MAX_EVALS}')
print(f'Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)')
print(f'Time per eval: {elapsed/MAX_EVALS:.0f} seconds')
print(f'\\nBest alphas:')
for k, v in sorted(best_trial.params.items()):
    print(f'  {k}: {v:.3f}')

print(f'\\nComparison:')
print(f'  Grid search (300 evals, ~25h): 84.07')
print(f'  Bayesian ({MAX_EVALS} evals, {elapsed/3600:.1f}h): {best_trial.value:.2f}')
print(f'  Speedup: {300/MAX_EVALS:.1f}x fewer evals', flush=True)

# Convergence curve
print(f'\\nConvergence:')
running_best = 0
for h in history:
    if h['combined'] > running_best:
        running_best = h['combined']
        print(f'  Eval {h[\"eval\"]:3d}: {running_best:.2f}')

os.makedirs('results/data/72b/bayesian_alpha', exist_ok=True)
with open('results/data/72b/bayesian_alpha/results.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'max_evals': MAX_EVALS,
        'best_combined': best_trial.value,
        'best_eval_number': best_trial.number + 1,
        'elapsed_hours': elapsed / 3600,
        'best_alphas': dict(best_trial.params),
        'history': history,
        'grid_search_reference': 84.07,
    }, f, indent=2, default=str)
print(f'Saved to results/data/72b/bayesian_alpha/results.json', flush=True)
"

echo "=== Done at $(date) ==="
