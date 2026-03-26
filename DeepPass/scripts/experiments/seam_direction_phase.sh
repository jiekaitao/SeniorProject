#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_seam_phase_%j.log
#SBATCH --job-name=deeppass_sphase

# Seam-Direction Phase Diagram (72B)
#
# Tests whether factual corruption from FFN re-retrieval is direction-specific.
# For block (45,52), collects the seam perturbation delta = h2 - h1, then
# generates random and orthogonal perturbations with the same norm.
# Sweeps alpha from 0 to 2.0 and tracks phase transitions in model answers.
#
# Key question: does the corruption follow the SPECIFIC direction of delta,
# or would ANY perturbation of the same magnitude cause equal harm?

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Seam-Direction Phase Diagram ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
BLOCK = (45, 52)
ALPHAS = [round(a * 0.2, 1) for a in range(11)]  # 0.0 to 2.0 step 0.2

FACTUAL_PROMPTS = [
    'What is the capital of France?',
    'Who wrote Romeo and Juliet?',
    'What year did WW2 end?',
    'What is the chemical formula for water?',
    'What planet is closest to the sun?',
    'What is the speed of light in m/s?',
    'Who painted the Mona Lisa?',
    'What is the atomic number of carbon?',
]

REASONING_PROMPTS = [
    'If A>B and B>C, is A>C?',
    'What is 127*348?',
    'If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?',
    'A bat and ball cost \$1.10 total. The bat costs \$1 more than the ball. How much does the ball cost?',
    'What emotion would someone feel after losing a close friend?',
    'If f(x)=3x^2-2x+1, what is f(5)?',
    'How does anticipation differ from anxiety?',
    'Describe the feeling of watching a sunset after a difficult day.',
]

print('=' * 70)
print('SEAM-DIRECTION PHASE DIAGRAM')
print('Is factual corruption direction-specific?')
print(f'Block: {BLOCK}, Alphas: {ALPHAS}')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers on {device}', flush=True)

# =====================================================================
# Build layer order with duplication
# =====================================================================

def build_order(block, N):
    i, j = block
    order = list(range(j)) + list(range(i, j)) + list(range(j, N))
    return order

def find_seam_positions(layer_order, block_start, block_end):
    last_layer = block_end - 1
    occurrences = [step for step, layer_idx in enumerate(layer_order) if layer_idx == last_layer]
    assert len(occurrences) >= 2, f'Block not duplicated: layer {last_layer} appears {len(occurrences)} time(s)'
    return occurrences[0], occurrences[1]

layer_order = build_order(BLOCK, N)
first_end, second_end = find_seam_positions(layer_order, BLOCK[0], BLOCK[1])

# =====================================================================
# Collect seam perturbation delta for a prompt (single forward, no gen)
# =====================================================================

def collect_seam_delta(prompt):
    \"\"\"Run full duplicated forward once, return (h1, delta=h2-h1) at seam.\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, pos_ids)

        h_after_first = None
        for step_idx, layer_idx in enumerate(layer_order):
            layer = original_layers[layer_idx]
            out = layer(h, position_embeddings=pos_embeds, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out

            if step_idx == first_end:
                h_after_first = h.clone()

            if step_idx == second_end and h_after_first is not None:
                delta = h - h_after_first  # h2 - h1
                return h_after_first, delta

    raise RuntimeError('Seam not reached')


def generate_random_perturbation(delta):
    \"\"\"Random direction perturbation with same norm as delta.\"\"\"
    rand = torch.randn_like(delta)
    rand_norm = rand.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_norm = delta.norm(dim=-1, keepdim=True)
    return rand / rand_norm * delta_norm


def generate_orthogonal_perturbation(delta):
    \"\"\"Perturbation orthogonal to delta with same norm. Uses Gram-Schmidt.\"\"\"
    rand = torch.randn_like(delta)
    # Project out the delta component: rand_perp = rand - (rand . delta / delta . delta) * delta
    delta_flat = delta.reshape(-1).float()
    rand_flat = rand.reshape(-1).float()
    proj = (rand_flat @ delta_flat) / (delta_flat @ delta_flat + 1e-12)
    perp_flat = rand_flat - proj * delta_flat
    perp = perp_flat.reshape(delta.shape).to(delta.dtype)
    # Rescale to match delta norm
    perp_norm = perp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_norm = delta.norm(dim=-1, keepdim=True)
    return perp / perp_norm * delta_norm


# =====================================================================
# Generate with patched hidden state: h_patched = h1 + alpha * perturbation
# =====================================================================

def generate_with_perturbation(prompt, h1_cached, perturbation, alpha, max_new_tokens=64):
    \"\"\"
    Generate text where at the seam, we inject h_patched = h1 + alpha * perturbation.
    h1_cached and perturbation are for the INITIAL prompt (first token generation step).
    For subsequent tokens, we recompute the full forward (perturbation only on first step).

    Actually, to be consistent, we apply the perturbation at every token generation step.
    The perturbation is cached for the original prompt length; for extended sequences,
    we pad/recompute.
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    original_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            h_after_first = None
            for step_idx, layer_idx in enumerate(layer_order):
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

                if step_idx == first_end:
                    h_after_first = h.clone()

                if step_idx == second_end and h_after_first is not None:
                    # Apply perturbation: h = h1 + alpha * perturbation_direction
                    # The perturbation is recomputed live as (h - h_after_first) is the
                    # real delta at current length. We replace it with the desired perturbation type.
                    current_delta = h - h_after_first
                    if perturbation is None:
                        # Use real delta (standard duplication with alpha scaling)
                        h = h_after_first + alpha * current_delta
                    else:
                        # For the perturbation type, we scale the cached direction to match
                        # current delta norm, then apply alpha
                        # Recompute perturbation for current sequence length
                        current_norm = current_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        if perturbation == 'random':
                            p = generate_random_perturbation(current_delta)
                        elif perturbation == 'orthogonal':
                            p = generate_orthogonal_perturbation(current_delta)
                        else:
                            p = current_delta  # fallback
                        h = h_after_first + alpha * p

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


def get_logit_info(prompt, perturbation_type, alpha):
    \"\"\"Run single forward pass with perturbation, return logit info for last token.\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    with torch.no_grad():
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, pos_ids)

        h_after_first = None
        for step_idx, layer_idx in enumerate(layer_order):
            layer = original_layers[layer_idx]
            out = layer(h, position_embeddings=pos_embeds, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out

            if step_idx == first_end:
                h_after_first = h.clone()

            if step_idx == second_end and h_after_first is not None:
                current_delta = h - h_after_first
                if perturbation_type == 'real':
                    h = h_after_first + alpha * current_delta
                elif perturbation_type == 'random':
                    p = generate_random_perturbation(current_delta)
                    h = h_after_first + alpha * p
                elif perturbation_type == 'orthogonal':
                    p = generate_orthogonal_perturbation(current_delta)
                    h = h_after_first + alpha * p

        h = inner.norm(h)
        logits = model.lm_head(h)

    # Top-1 info
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top1_prob, top1_idx = probs.max(dim=-1)
    top1_token = tokenizer.decode([top1_idx.item()])

    return {
        'top1_prob': top1_prob.item(),
        'top1_token': top1_token.strip(),
        'top1_idx': top1_idx.item(),
    }


# =====================================================================
# Main experiment
# =====================================================================

all_results = {}
perturbation_types = ['real', 'random', 'orthogonal']

for prompt_category, prompts in [('factual', FACTUAL_PROMPTS), ('reasoning', REASONING_PROMPTS)]:
    print(f'\n{\"=\" * 70}')
    print(f'CATEGORY: {prompt_category.upper()} ({len(prompts)} prompts)')
    print(f'{\"=\" * 70}', flush=True)

    category_results = []

    for pidx, prompt in enumerate(prompts):
        print(f'\n  Prompt {pidx}: \"{prompt}\"', flush=True)

        # First, get the baseline answer (alpha=0, just h1)
        baseline_info = get_logit_info(prompt, 'real', 0.0)
        print(f'    alpha=0 baseline: top1=\"{baseline_info[\"top1_token\"]}\" p={baseline_info[\"top1_prob\"]:.4f}', flush=True)

        # Also get standard dup answer (alpha=1, real delta)
        standard_info = get_logit_info(prompt, 'real', 1.0)
        print(f'    alpha=1 standard: top1=\"{standard_info[\"top1_token\"]}\" p={standard_info[\"top1_prob\"]:.4f}', flush=True)

        # Also generate full answers at alpha=0 and alpha=1
        answer_alpha0 = generate_with_perturbation(prompt, None, None, 0.0, max_new_tokens=48)
        answer_alpha1 = generate_with_perturbation(prompt, None, None, 1.0, max_new_tokens=48)
        print(f'    gen alpha=0: {answer_alpha0[:80]}', flush=True)
        print(f'    gen alpha=1: {answer_alpha1[:80]}', flush=True)

        prompt_results = {
            'prompt': prompt,
            'category': prompt_category,
            'baseline_token': baseline_info['top1_token'],
            'baseline_prob': baseline_info['top1_prob'],
            'standard_token': standard_info['top1_token'],
            'standard_prob': standard_info['top1_prob'],
            'answer_alpha0': answer_alpha0,
            'answer_alpha1': answer_alpha1,
            'sweeps': {},
        }

        for ptype in perturbation_types:
            print(f'    --- {ptype} perturbation ---', flush=True)
            sweep_data = []

            for alpha in ALPHAS:
                info = get_logit_info(prompt, ptype, alpha)
                # Check phase transition: did top-1 token change from baseline?
                phase_changed = (info['top1_idx'] != baseline_info['top1_idx'])

                sweep_data.append({
                    'alpha': alpha,
                    'top1_token': info['top1_token'],
                    'top1_prob': info['top1_prob'],
                    'top1_idx': info['top1_idx'],
                    'phase_changed': phase_changed,
                })

                marker = ' PHASE!' if phase_changed else ''
                print(f'      alpha={alpha:.1f}: top1=\"{info[\"top1_token\"]}\" p={info[\"top1_prob\"]:.4f}{marker}', flush=True)

            # Compute phase transition sharpness
            # Find the alpha range over which the answer transitions
            changed = [d for d in sweep_data if d['phase_changed']]
            if changed:
                first_change_alpha = changed[0]['alpha']
                # Sharpness = how narrow the transition region is
                # Measure: what fraction of alpha range has the original answer?
                original_range = sum(1 for d in sweep_data if not d['phase_changed']) / len(sweep_data)
                sharpness = 1.0 - original_range  # higher = sharper transition
            else:
                first_change_alpha = None
                sharpness = 0.0

            prompt_results['sweeps'][ptype] = {
                'data': sweep_data,
                'first_change_alpha': first_change_alpha,
                'sharpness': sharpness,
                'num_phase_changes': len(changed),
            }

            print(f'      first_change_alpha={first_change_alpha}, sharpness={sharpness:.3f}', flush=True)

        category_results.append(prompt_results)
    all_results[prompt_category] = category_results

# =====================================================================
# Summary analysis
# =====================================================================

print(f'\n{\"=\" * 70}')
print('PHASE TRANSITION SUMMARY')
print(f'{\"=\" * 70}', flush=True)

for category in ['factual', 'reasoning']:
    print(f'\n  --- {category.upper()} ---', flush=True)
    for ptype in perturbation_types:
        sharpnesses = []
        first_alphas = []
        for pr in all_results[category]:
            s = pr['sweeps'][ptype]
            sharpnesses.append(s['sharpness'])
            if s['first_change_alpha'] is not None:
                first_alphas.append(s['first_change_alpha'])

        mean_sharp = np.mean(sharpnesses) if sharpnesses else 0.0
        mean_first = np.mean(first_alphas) if first_alphas else float('nan')
        n_transitioned = len(first_alphas)

        print(f'    {ptype:12s}: mean_sharpness={mean_sharp:.3f} '
              f'mean_first_change_alpha={mean_first:.2f} '
              f'transitioned={n_transitioned}/{len(all_results[category])}', flush=True)

# Cross-category comparison
print(f'\n  --- DIRECTION SPECIFICITY ---', flush=True)
for category in ['factual', 'reasoning']:
    real_sharp = np.mean([pr['sweeps']['real']['sharpness'] for pr in all_results[category]])
    rand_sharp = np.mean([pr['sweeps']['random']['sharpness'] for pr in all_results[category]])
    orth_sharp = np.mean([pr['sweeps']['orthogonal']['sharpness'] for pr in all_results[category]])
    specificity = real_sharp - (rand_sharp + orth_sharp) / 2
    print(f'    {category:10s}: real={real_sharp:.3f} random={rand_sharp:.3f} orthogonal={orth_sharp:.3f} '
          f'specificity={specificity:+.3f}', flush=True)

# Save
os.makedirs('results/data/72b/mechanistic', exist_ok=True)
outpath = 'results/data/72b/mechanistic/seam_direction_phase.json'

# Convert for JSON serialization
def jsonify(obj):
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonify(x) for x in obj]
    return obj

with open(outpath, 'w') as f:
    json.dump(jsonify({
        'date': datetime.now().isoformat(),
        'model': 'calme-2.1-qwen2-72b',
        'block': list(BLOCK),
        'alphas': ALPHAS,
        'perturbation_types': perturbation_types,
        'factual_prompts': FACTUAL_PROMPTS,
        'reasoning_prompts': REASONING_PROMPTS,
        'results': all_results,
    }), f, indent=2)
print(f'\nSaved to {outpath}', flush=True)
print('DONE', flush=True)
"

echo "=== Done at $(date) ==="
