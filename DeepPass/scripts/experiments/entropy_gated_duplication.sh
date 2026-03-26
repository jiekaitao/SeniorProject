#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_entropy_gate_%j.log
#SBATCH --job-name=deeppass_egate

# Entropy-Gated Duplication Experiment (72B)
#
# Uses the model's output entropy to decide whether to duplicate per-input.
# Hypothesis: duplication helps most when the model is "uncertain" (high entropy
# at the seam point), and can be safely skipped for confident predictions.
#
# Approach:
#   1. For ~60 diverse prompts, measure seam entropy after first pass
#   2. Correlate entropy with duplication benefit
#   3. Find optimal entropy threshold for gating
#   4. Compare fixed duplication vs entropy-gated duplication

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Entropy-Gated Duplication Experiment ==="
echo "Started: $(date)"

$PYTHON -c "
import sys, os, json, time, torch, torch.nn as nn, numpy as np
from datetime import datetime

sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model
from math_probe import run_math_probe, calculate_score, extract_number, SYSTEM_PROMPT, USER_TEMPLATE
from eq_bench_probe import run_eq_bench_probe

MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'

print('=' * 70)
print('ENTROPY-GATED DUPLICATION EXPERIMENT')
print('Can we skip duplication when the model is already confident?')
print(f'Date: {datetime.now().isoformat()}')
print('=' * 70, flush=True)

model, tokenizer = load_original_model(MODEL_PATH)
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device
print(f'Loaded: {N} layers', flush=True)

# =====================================================================
# Diverse prompt set: 15 per category, 60 total
# =====================================================================

MATH_PROMPTS = [
    {'prompt': 'What is 78313086360375 multiplied by 88537453126609?', 'answer': 6933174468959498727528375, 'type': 'math'},
    {'prompt': 'What is the cube root of 74088893247?', 'answer': 4201, 'type': 'math'},
    {'prompt': 'What is 9999999 multiplied by 9999999?', 'answer': 99999980000001, 'type': 'math'},
    {'prompt': 'What is 123456789 multiplied by 987654321?', 'answer': 121932631112635269, 'type': 'math'},
    {'prompt': 'What is the square root of 152399025?', 'answer': 12345, 'type': 'math'},
    {'prompt': 'What is 7777777 multiplied by 3333333?', 'answer': 25925923703641, 'type': 'math'},
    {'prompt': 'What is 456789 raised to the power of 2?', 'answer': 208655854521, 'type': 'math'},
    {'prompt': 'What is 2 raised to the power of 48?', 'answer': 281474976710656, 'type': 'math'},
    {'prompt': 'What is 314159 multiplied by 271828?', 'answer': 85397342252, 'type': 'math'},
    {'prompt': 'What is 847293 multiplied by 192837?', 'answer': 163399178841, 'type': 'math'},
    {'prompt': 'What is 19 raised to the power of 4?', 'answer': 130321, 'type': 'math'},
    {'prompt': 'What is 65536 multiplied by 65536?', 'answer': 4294967296, 'type': 'math'},
    {'prompt': 'What is 3 raised to the power of 20?', 'answer': 3486784401, 'type': 'math'},
    {'prompt': 'What is 88888 multiplied by 77777?', 'answer': 6913580136, 'type': 'math'},
    {'prompt': 'What is 54321 multiplied by 12345?', 'answer': 670592745, 'type': 'math'},
]

REASONING_PROMPTS = [
    {'prompt': 'If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer YES or NO with one sentence of explanation.', 'type': 'reasoning'},
    {'prompt': 'A is taller than B. C is shorter than B. D is taller than A. Who is the shortest?', 'type': 'reasoning'},
    {'prompt': 'If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?', 'type': 'reasoning'},
    {'prompt': 'A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost?', 'type': 'reasoning'},
    {'prompt': 'Three friends each tell one lie and one truth. Alice says: I am not the oldest. Bob says: I am not the youngest. Charlie says: I am not the oldest. If Alice is the youngest, who is the oldest?', 'type': 'reasoning'},
    {'prompt': 'You have 8 balls. One is heavier. You have a balance scale. What is the minimum number of weighings to find the heavy ball?', 'type': 'reasoning'},
    {'prompt': 'A farmer has 17 sheep. All but 9 die. How many are left?', 'type': 'reasoning'},
    {'prompt': 'If you rearrange the letters CIFAIPC, you get the name of a(n): ocean, city, animal, or country?', 'type': 'reasoning'},
    {'prompt': 'I have two coins that total 30 cents. One of them is not a nickel. What are the two coins?', 'type': 'reasoning'},
    {'prompt': 'A man is looking at a portrait. Someone asks him whose portrait it is. He replies: Brothers and sisters I have none, but this mans father is my fathers son. Whose portrait is the man looking at?', 'type': 'reasoning'},
    {'prompt': 'If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?', 'type': 'reasoning'},
    {'prompt': 'There are 3 boxes: one has only apples, one has only oranges, one has both. All labels are wrong. You pick one fruit from one box. Which box do you pick from to determine all labels?', 'type': 'reasoning'},
    {'prompt': 'A snail climbs 3 feet during the day but slides back 2 feet at night. The well is 30 feet deep. How many days to climb out?', 'type': 'reasoning'},
    {'prompt': 'Five people are sitting in a row. A is to the left of B. C is to the right of D. B is to the right of D. E is between A and D. What is the order from left to right?', 'type': 'reasoning'},
    {'prompt': 'If the day after tomorrow is two days before Thursday, what day is it today?', 'type': 'reasoning'},
]

KNOWLEDGE_PROMPTS = [
    {'prompt': 'What year did the Berlin Wall fall?', 'type': 'knowledge'},
    {'prompt': 'What is the atomic number of gold?', 'type': 'knowledge'},
    {'prompt': 'Who wrote the novel One Hundred Years of Solitude?', 'type': 'knowledge'},
    {'prompt': 'What is the speed of light in meters per second?', 'type': 'knowledge'},
    {'prompt': 'What is the capital of Mongolia?', 'type': 'knowledge'},
    {'prompt': 'In what year was the Magna Carta signed?', 'type': 'knowledge'},
    {'prompt': 'What is the chemical formula for sulfuric acid?', 'type': 'knowledge'},
    {'prompt': 'Who painted the ceiling of the Sistine Chapel?', 'type': 'knowledge'},
    {'prompt': 'What is the largest desert in the world by area?', 'type': 'knowledge'},
    {'prompt': 'How many chromosomes do humans have?', 'type': 'knowledge'},
    {'prompt': 'What is the boiling point of water in Kelvin?', 'type': 'knowledge'},
    {'prompt': 'Who developed the theory of general relativity?', 'type': 'knowledge'},
    {'prompt': 'What is the longest river in Africa?', 'type': 'knowledge'},
    {'prompt': 'In what year was the transistor invented?', 'type': 'knowledge'},
    {'prompt': 'What element has the symbol Pb?', 'type': 'knowledge'},
]

CREATIVE_PROMPTS = [
    {'prompt': 'Write a haiku about a forgotten umbrella in the rain.', 'type': 'creative'},
    {'prompt': 'Describe the feeling of waking up to realize a nightmare was just a dream, in exactly two sentences.', 'type': 'creative'},
    {'prompt': 'Complete this analogy creatively: Time is to a river as memory is to ___', 'type': 'creative'},
    {'prompt': 'Write a one-sentence story that contains both immense joy and deep sorrow.', 'type': 'creative'},
    {'prompt': 'If sadness had a flavor, what would it taste like? Explain in one sentence.', 'type': 'creative'},
    {'prompt': 'Describe a sunset to someone who has never seen one, using only sounds and textures.', 'type': 'creative'},
    {'prompt': 'Write a short metaphor comparing loneliness to an everyday household object.', 'type': 'creative'},
    {'prompt': 'In one sentence, describe what courage looks like from the inside.', 'type': 'creative'},
    {'prompt': 'Finish this story in one sentence: The last librarian on Earth opened the final book and found...', 'type': 'creative'},
    {'prompt': 'Describe the sound of silence in a way a musician would understand.', 'type': 'creative'},
    {'prompt': 'Write a two-line poem about a robot discovering it can dream.', 'type': 'creative'},
    {'prompt': 'If nostalgia were a place, what would the entrance look like? One sentence.', 'type': 'creative'},
    {'prompt': 'Complete this: Forgiveness is not about the other person, it is about ___', 'type': 'creative'},
    {'prompt': 'Describe the taste of your favorite childhood memory in one sentence.', 'type': 'creative'},
    {'prompt': 'Write a fortune cookie message for someone about to make a difficult decision.', 'type': 'creative'},
]

ALL_PROMPTS = MATH_PROMPTS + REASONING_PROMPTS + KNOWLEDGE_PROMPTS + CREATIVE_PROMPTS
print(f'Total prompts: {len(ALL_PROMPTS)} ({len(MATH_PROMPTS)} math, {len(REASONING_PROMPTS)} reasoning, {len(KNOWLEDGE_PROMPTS)} knowledge, {len(CREATIVE_PROMPTS)} creative)', flush=True)

# =====================================================================
# Core functions
# =====================================================================

def build_order(blocks, N):
    s = sorted(blocks)
    order, prev = [], 0
    for (i, j) in s:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order

def forward_through_layers(h, pos_embeds, layer_indices):
    \"\"\"Run h through a sequence of layer indices.\"\"\"
    for idx in layer_indices:
        layer = original_layers[idx]
        out = layer(h, position_embeddings=pos_embeds, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    return h

def compute_entropy(logits, top_k=100):
    \"\"\"Compute entropy of the token distribution over top-k tokens.\"\"\"
    # logits: [1, vocab_size] or [1, seq_len, vocab_size]
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # last token

    # Top-k filtering
    top_vals, _ = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(top_vals.float(), dim=-1)

    # Entropy: -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.item()

def get_seam_entropy(input_ids, block, pos_embeds_fn):
    \"\"\"
    After first pass through block, compute entropy by temporarily running
    through remaining layers + norm + lm_head.

    For block (i, j):
      1. Run layers 0..j-1 (first pass includes the block)
      2. At the seam (after layer j-1), compute entropy by running j..N-1 + norm + lm_head
    \"\"\"
    i_block, j_block = block
    with torch.no_grad():
        h = inner.embed_tokens(input_ids)
        seq_len = h.shape[1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = inner.rotary_emb(h, pos_ids)

        # First pass: layers 0 through j-1
        h = forward_through_layers(h, pos_embeds, range(0, j_block))

        # At the seam: compute what the model would output without second pass
        # Run remaining layers j..N-1 to get the full forward pass
        h_probe = forward_through_layers(h.clone(), pos_embeds, range(j_block, N))
        h_probe = inner.norm(h_probe)
        logits = model.lm_head(h_probe)

        entropy = compute_entropy(logits)
    return entropy

def generate_with_optional_dup(prompt, blocks, do_duplicate, max_new_tokens=64):
    \"\"\"Generate with or without duplication (alpha=1.0 when active).\"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)

    if do_duplicate:
        layer_order = build_order(sorted_blocks, N)
    else:
        layer_order = list(range(N))

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            for layer_idx in layer_order:
                layer = original_layers[layer_idx]
                out = layer(h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out

            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

def generate_entropy_gated(prompt, blocks, entropy_threshold, max_new_tokens=64):
    \"\"\"
    Generate with entropy-gated duplication.
    At each token step, compute seam entropy and only duplicate if entropy > threshold.
    \"\"\"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    sorted_blocks = sorted(blocks)

    dup_count = 0
    skip_count = 0

    for step in range(max_new_tokens):
        with torch.no_grad():
            h = inner.embed_tokens(input_ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)

            # For each block, decide whether to duplicate based on seam entropy
            prev_end = 0
            for block in sorted_blocks:
                i_block, j_block = block

                # Run layers from prev_end to j_block (first pass)
                h = forward_through_layers(h, pos_embeds, range(prev_end, j_block))

                # Compute seam entropy
                h_probe = forward_through_layers(h.clone(), pos_embeds, range(j_block, N))
                h_probe = inner.norm(h_probe)
                logits_probe = model.lm_head(h_probe)
                entropy = compute_entropy(logits_probe)

                if entropy > entropy_threshold:
                    # High entropy: duplicate (second pass through block)
                    h = forward_through_layers(h, pos_embeds, range(i_block, j_block))
                    dup_count += 1
                else:
                    skip_count += 1

                prev_end = j_block

            # Run remaining layers
            h = forward_through_layers(h, pos_embeds, range(prev_end, N))
            h = inner.norm(h)
            logits = model.lm_head(h)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    prompt_len = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
    text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
    total = dup_count + skip_count
    dup_rate = dup_count / total if total > 0 else 0
    return text, dup_rate

# =====================================================================
# PHASE 1: Collect entropy and duplication benefit per prompt
# =====================================================================
print(f'\n{\"=\" * 70}')
print('PHASE 1: Collecting seam entropy and duplication benefit per prompt')
print(f'{\"=\" * 70}', flush=True)

# Configuration: best pair
pair_blocks = [(0, 7), (45, 52)]
# We measure seam entropy at (45, 52) since it is the primary block
entropy_block = (45, 52)

per_prompt_data = []
t0 = time.time()

for idx, p in enumerate(ALL_PROMPTS):
    prompt_text = p['prompt']
    prompt_type = p['type']

    # Compute seam entropy
    input_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to(device)
    seam_entropy = get_seam_entropy(input_ids, entropy_block, None)

    # Score with duplication
    if prompt_type == 'math':
        full_prompt = f'System: {SYSTEM_PROMPT}\n\nUser: {USER_TEMPLATE.format(question=prompt_text)}\n\nAssistant:'
        resp_dup = generate_with_optional_dup(full_prompt, pair_blocks, do_duplicate=True, max_new_tokens=64)
        resp_nodup = generate_with_optional_dup(full_prompt, pair_blocks, do_duplicate=False, max_new_tokens=64)
        est_dup = extract_number(resp_dup)
        est_nodup = extract_number(resp_nodup)
        try:
            score_dup = calculate_score(p['answer'], est_dup)
        except Exception:
            score_dup = 0.0
        try:
            score_nodup = calculate_score(p['answer'], est_nodup)
        except Exception:
            score_nodup = 0.0
    else:
        # For non-math prompts, use generation length and coherence as proxy
        # We measure by generating with and without dup and computing perplexity
        resp_dup = generate_with_optional_dup(prompt_text, pair_blocks, do_duplicate=True, max_new_tokens=64)
        resp_nodup = generate_with_optional_dup(prompt_text, pair_blocks, do_duplicate=False, max_new_tokens=64)

        # Use response length ratio as a proxy (longer coherent responses = better)
        # But more importantly, we score via next-token log-likelihood on the prompt
        with torch.no_grad():
            ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to(device)

            # Baseline (no dup) perplexity
            h = inner.embed_tokens(ids)
            seq_len = h.shape[1]
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for li in range(N):
                out = original_layers[li](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            h = inner.norm(h)
            logits_nodup = model.lm_head(h)
            # Shift for next-token prediction
            shift_logits = logits_nodup[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss_nodup = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

            # Duplicated perplexity
            layer_order_dup = build_order(sorted(pair_blocks), N)
            h = inner.embed_tokens(ids)
            pos_embeds = inner.rotary_emb(h, pos_ids)
            for li in layer_order_dup:
                out = original_layers[li](h, position_embeddings=pos_embeds, use_cache=False)
                h = out[0] if isinstance(out, tuple) else out
            h = inner.norm(h)
            logits_dup = model.lm_head(h)
            shift_logits = logits_dup[:, :-1, :].contiguous()
            loss_dup = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

        # Lower loss = better. Convert to a 0-1 score: benefit of duplication
        score_nodup = max(0, 1.0 - loss_nodup / 10.0)  # rough normalization
        score_dup = max(0, 1.0 - loss_dup / 10.0)

    benefit = score_dup - score_nodup

    entry = {
        'idx': idx,
        'prompt': prompt_text[:80],
        'type': prompt_type,
        'seam_entropy': seam_entropy,
        'score_dup': score_dup,
        'score_nodup': score_nodup,
        'benefit': benefit,
    }
    per_prompt_data.append(entry)

    print(f'  [{idx+1:2d}/{len(ALL_PROMPTS)}] {prompt_type:>10s} entropy={seam_entropy:.4f} dup={score_dup:.4f} nodup={score_nodup:.4f} benefit={benefit:+.4f}', flush=True)

elapsed_phase1 = time.time() - t0
print(f'\nPhase 1 completed in {elapsed_phase1/60:.1f} minutes', flush=True)

# =====================================================================
# PHASE 2: Analysis — correlate entropy with duplication benefit
# =====================================================================
print(f'\n{\"=\" * 70}')
print('PHASE 2: Correlation analysis')
print(f'{\"=\" * 70}', flush=True)

entropies = np.array([d['seam_entropy'] for d in per_prompt_data])
benefits = np.array([d['benefit'] for d in per_prompt_data])

# Overall correlation
from scipy.stats import pearsonr, spearmanr
if len(entropies) > 2:
    pearson_r, pearson_p = pearsonr(entropies, benefits)
    spearman_r, spearman_p = spearmanr(entropies, benefits)
else:
    pearson_r, pearson_p = 0, 1
    spearman_r, spearman_p = 0, 1

print(f'Overall entropy-benefit correlation:', flush=True)
print(f'  Pearson r  = {pearson_r:.4f} (p={pearson_p:.4f})', flush=True)
print(f'  Spearman r = {spearman_r:.4f} (p={spearman_p:.4f})', flush=True)

# Per-type analysis
for ptype in ['math', 'reasoning', 'knowledge', 'creative']:
    mask = [d['type'] == ptype for d in per_prompt_data]
    type_ent = entropies[mask]
    type_ben = benefits[mask]
    if len(type_ent) > 2:
        pr, _ = pearsonr(type_ent, type_ben)
        sr, _ = spearmanr(type_ent, type_ben)
    else:
        pr, sr = 0, 0
    mean_ent = np.mean(type_ent)
    mean_ben = np.mean(type_ben)
    print(f'  {ptype:>10s}: mean_entropy={mean_ent:.4f} mean_benefit={mean_ben:+.4f} pearson={pr:.4f} spearman={sr:.4f}', flush=True)

# Entropy distribution
print(f'\nEntropy distribution:', flush=True)
for ptype in ['math', 'reasoning', 'knowledge', 'creative']:
    mask = [d['type'] == ptype for d in per_prompt_data]
    type_ent = entropies[mask]
    print(f'  {ptype:>10s}: min={np.min(type_ent):.4f} median={np.median(type_ent):.4f} max={np.max(type_ent):.4f} std={np.std(type_ent):.4f}', flush=True)

# =====================================================================
# PHASE 3: Threshold gating — sweep entropy thresholds
# =====================================================================
print(f'\n{\"=\" * 70}')
print('PHASE 3: Entropy threshold sweep')
print(f'{\"=\" * 70}', flush=True)

# Compute candidate thresholds from percentiles of the entropy distribution
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
thresholds = [float(np.percentile(entropies, p)) for p in percentiles]
# Also add some fixed thresholds
thresholds.extend([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
thresholds = sorted(set(thresholds))

print(f'Testing {len(thresholds)} thresholds', flush=True)

# For the threshold analysis, use the per-prompt data we already collected.
# For each threshold T:
#   - If entropy >= T: use duplicated score
#   - If entropy < T: use non-duplicated score
#   - Average across all prompts

threshold_results = []
for T in thresholds:
    gated_scores = []
    dup_count = 0
    for d in per_prompt_data:
        if d['seam_entropy'] >= T:
            gated_scores.append(d['score_dup'])
            dup_count += 1
        else:
            gated_scores.append(d['score_nodup'])

    avg_gated = np.mean(gated_scores)
    dup_rate = dup_count / len(per_prompt_data)

    threshold_results.append({
        'threshold': T,
        'avg_score': float(avg_gated),
        'dup_rate': dup_rate,
        'dup_count': dup_count,
        'skip_count': len(per_prompt_data) - dup_count,
    })
    print(f'  T={T:6.3f}: avg_score={avg_gated:.4f} dup_rate={dup_rate:.1%} ({dup_count}/{len(per_prompt_data)})', flush=True)

# Fixed reference scores
avg_always_dup = np.mean([d['score_dup'] for d in per_prompt_data])
avg_never_dup = np.mean([d['score_nodup'] for d in per_prompt_data])
print(f'\n  Always duplicate:      avg_score={avg_always_dup:.4f} (dup_rate=100%)', flush=True)
print(f'  Never duplicate:       avg_score={avg_never_dup:.4f} (dup_rate=0%)', flush=True)

# Find best threshold
best_threshold_entry = max(threshold_results, key=lambda x: x['avg_score'])
print(f'  Best gated threshold:  T={best_threshold_entry[\"threshold\"]:.3f} avg_score={best_threshold_entry[\"avg_score\"]:.4f} dup_rate={best_threshold_entry[\"dup_rate\"]:.1%}', flush=True)

# =====================================================================
# PHASE 4: Validate entropy-gated generation end-to-end
# =====================================================================
print(f'\n{\"=\" * 70}')
print('PHASE 4: End-to-end validation of entropy-gated generation')
print(f'{\"=\" * 70}', flush=True)

best_T = best_threshold_entry['threshold']
print(f'Using best threshold T={best_T:.3f}', flush=True)

# Validate on math probe (official 16 questions)
print(f'\n  --- Math Probe Validation ---', flush=True)

gen_always = lambda p: generate_with_optional_dup(p, pair_blocks, do_duplicate=True, max_new_tokens=64)
gen_never = lambda p: generate_with_optional_dup(p, pair_blocks, do_duplicate=False, max_new_tokens=64)
gen_gated = lambda p: generate_entropy_gated(p, pair_blocks, best_T, max_new_tokens=64)[0]

math_always = run_math_probe(gen_always, verbose=False)
math_never = run_math_probe(gen_never, verbose=False)
math_gated = run_math_probe(gen_gated, verbose=False)

print(f'  Always dup: math={math_always[\"score\"]:.4f}', flush=True)
print(f'  Never dup:  math={math_never[\"score\"]:.4f}', flush=True)
print(f'  Gated:      math={math_gated[\"score\"]:.4f}', flush=True)

# Validate on EQ-bench
print(f'\n  --- EQ-Bench Validation ---', flush=True)

gen_always_long = lambda p: generate_with_optional_dup(p, pair_blocks, do_duplicate=True, max_new_tokens=128)
gen_never_long = lambda p: generate_with_optional_dup(p, pair_blocks, do_duplicate=False, max_new_tokens=128)
gen_gated_long = lambda p: generate_entropy_gated(p, pair_blocks, best_T, max_new_tokens=128)[0]

eq_always = run_eq_bench_probe(gen_always_long, verbose=False)
eq_never = run_eq_bench_probe(gen_never_long, verbose=False)
eq_gated = run_eq_bench_probe(gen_gated_long, verbose=False)

print(f'  Always dup: eq={eq_always[\"score\"]:.1f}/100', flush=True)
print(f'  Never dup:  eq={eq_never[\"score\"]:.1f}/100', flush=True)
print(f'  Gated:      eq={eq_gated[\"score\"]:.1f}/100', flush=True)

# Combined scores
combined_always = math_always['score'] * 50 + eq_always['score'] * 0.5
combined_never = math_never['score'] * 50 + eq_never['score'] * 0.5
combined_gated = math_gated['score'] * 50 + eq_gated['score'] * 0.5

print(f'\n  Combined scores:', flush=True)
print(f'  Always dup: {combined_always:.2f}', flush=True)
print(f'  Never dup:  {combined_never:.2f}', flush=True)
print(f'  Gated (T={best_T:.3f}): {combined_gated:.2f}', flush=True)

# =====================================================================
# PHASE 5: Per-token duplication rate analysis
# =====================================================================
print(f'\n{\"=\" * 70}')
print('PHASE 5: Per-token duplication rate with gated generation')
print(f'{\"=\" * 70}', flush=True)

# Run gated generation on a subset of prompts and track per-token dup rates
sample_prompts = ALL_PROMPTS[:20]  # 5 from each category
dup_rates_by_type = {'math': [], 'reasoning': [], 'knowledge': [], 'creative': []}

for idx, p in enumerate(sample_prompts):
    prompt_text = p['prompt']
    if p['type'] == 'math':
        full_prompt = f'System: {SYSTEM_PROMPT}\n\nUser: {USER_TEMPLATE.format(question=prompt_text)}\n\nAssistant:'
    else:
        full_prompt = prompt_text

    _, dup_rate = generate_entropy_gated(full_prompt, pair_blocks, best_T, max_new_tokens=64)
    dup_rates_by_type[p['type']].append(dup_rate)
    print(f'  [{idx+1:2d}/{len(sample_prompts)}] {p[\"type\"]:>10s} dup_rate={dup_rate:.1%} {prompt_text[:50]}', flush=True)

print(f'\n  Per-type average duplication rates with gating:', flush=True)
for ptype in ['math', 'reasoning', 'knowledge', 'creative']:
    rates = dup_rates_by_type[ptype]
    if rates:
        print(f'    {ptype:>10s}: mean={np.mean(rates):.1%} min={np.min(rates):.1%} max={np.max(rates):.1%}', flush=True)

overall_dup_rate = np.mean([r for rates in dup_rates_by_type.values() for r in rates])
print(f'    Overall dup rate: {overall_dup_rate:.1%}', flush=True)
compute_savings = 1.0 - overall_dup_rate
print(f'    Compute savings vs always-dup: {compute_savings:.1%}', flush=True)

# =====================================================================
# GRAND SUMMARY
# =====================================================================
print(f'\n{\"=\" * 70}')
print('GRAND SUMMARY')
print(f'{\"=\" * 70}', flush=True)

print(f'\n  Entropy-Benefit Correlation:', flush=True)
print(f'    Pearson r  = {pearson_r:.4f} (p={pearson_p:.4f})', flush=True)
print(f'    Spearman r = {spearman_r:.4f} (p={spearman_p:.4f})', flush=True)
if abs(pearson_r) > 0.3:
    print(f'    ==> Significant correlation: entropy IS predictive of duplication benefit', flush=True)
else:
    print(f'    ==> Weak correlation: entropy may NOT be a good gate signal', flush=True)

print(f'\n  Best Configuration:', flush=True)
print(f'    Threshold T = {best_T:.3f}', flush=True)
print(f'    Gated combined score:  {combined_gated:.2f}', flush=True)
print(f'    Always-dup combined:   {combined_always:.2f}', flush=True)
print(f'    Never-dup combined:    {combined_never:.2f}', flush=True)
print(f'    Gated dup rate:        {overall_dup_rate:.1%}', flush=True)

quality_loss = combined_always - combined_gated
print(f'\n  Quality loss from gating: {quality_loss:+.2f} ({quality_loss/combined_always*100:+.1f}%)', flush=True)
print(f'  Compute savings:          {compute_savings:.1%}', flush=True)

if quality_loss < 1.0 and compute_savings > 0.2:
    print(f'  ==> VIABLE: small quality loss with significant compute savings', flush=True)
elif quality_loss < 2.0:
    print(f'  ==> MARGINAL: moderate quality loss, worth further optimization', flush=True)
else:
    print(f'  ==> NOT VIABLE at this threshold: quality loss too high', flush=True)

# =====================================================================
# Save results
# =====================================================================
os.makedirs('results/data/72b/entropy_gate', exist_ok=True)
outpath = 'results/data/72b/entropy_gate/results.json'

with open(outpath, 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'model': 'calme-2.1-qwen2-72b',
        'pair_blocks': [list(b) for b in pair_blocks],
        'entropy_block': list(entropy_block),
        'num_prompts': len(ALL_PROMPTS),
        'per_prompt_data': per_prompt_data,
        'correlation': {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
        },
        'threshold_sweep': threshold_results,
        'best_threshold': best_T,
        'validation': {
            'math': {
                'always_dup': math_always['score'],
                'never_dup': math_never['score'],
                'gated': math_gated['score'],
            },
            'eq_bench': {
                'always_dup': eq_always['score'],
                'never_dup': eq_never['score'],
                'gated': eq_gated['score'],
            },
            'combined': {
                'always_dup': combined_always,
                'never_dup': combined_never,
                'gated': combined_gated,
            },
        },
        'dup_rates_by_type': {k: [float(r) for r in v] for k, v in dup_rates_by_type.items()},
        'overall_dup_rate': float(overall_dup_rate),
        'compute_savings': float(compute_savings),
        'quality_loss': float(quality_loss),
    }, f, indent=2)

print(f'\nSaved to {outpath}', flush=True)
print('DONE', flush=True)
"

echo "=== Done at $(date) ==="
