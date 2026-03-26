#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_speed_%j.log
#SBATCH --job-name=deeppass_speed

# Inference speed benchmark for layer duplication configs on 72B
# Measures TTFT, tokens/sec at various generation lengths, and peak VRAM

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Inference Speed Benchmark (72B) ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, gc
import torch
import torch.nn as nn

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = 'models/full/calme-2.1-qwen2-72b'
PROMPT = (
    'The theory of general relativity, proposed by Albert Einstein in 1915, '
    'fundamentally changed our understanding of gravity, space, and time. '
    'It describes gravity not as a force between masses, but as a curvature '
    'of spacetime caused by mass and energy.'
)
PROMPT_TOKENS = 128  # target prompt length for TTFT
GEN_LENGTHS = [64, 128, 256]
NUM_RUNS = 3

CONFIGS = {
    'baseline_cache':   {'blocks': [],                         'use_cache': True},
    'baseline_nocache': {'blocks': [],                         'use_cache': False},
    'ng_45_52':         {'blocks': [(45, 52)],                 'use_cache': False},
    'pair_0_7_45_52':   {'blocks': [(0, 7), (45, 52)],        'use_cache': False},
    'quad_4block':      {'blocks': [(0, 7), (15, 20), (20, 27), (45, 52)], 'use_cache': False},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_layer_order(blocks, N):
    sorted_blocks = sorted(blocks)
    order = []
    prev = 0
    for (i, j) in sorted_blocks:
        order.extend(range(prev, j))
        order.extend(range(i, j))
        prev = j
    order.extend(range(prev, N))
    return order


def apply_config(model, original_layers, original_num_layers, blocks):
    \"\"\"Apply layer duplication for a config (or restore baseline).\"\"\"
    inner = model.model
    if not blocks:
        # Restore baseline
        inner.layers = nn.ModuleList(list(original_layers))
        model.config.num_hidden_layers = original_num_layers
        return original_num_layers
    order = build_layer_order(blocks, original_num_layers)
    inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
    model.config.num_hidden_layers = len(order)
    return len(order)


def get_peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def measure_ttft(model, input_ids, use_cache):
    \"\"\"Measure time-to-first-token: one forward pass on the prompt.\"\"\"
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        _ = model(input_ids, use_cache=use_cache)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)  # milliseconds


def measure_generation_speed(model, tokenizer, input_ids, max_new_tokens, use_cache):
    \"\"\"
    Measure tokens/sec for generation.
    - With cache: use model.generate()
    - Without cache: use manual token-by-token loop (like generate_no_cache)
    Returns (elapsed_ms, tokens_generated).
    \"\"\"
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if use_cache:
        start.record()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        end.record()
        torch.cuda.synchronize()
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    else:
        ids = input_ids.clone()
        tokens_generated = 0
        start.record()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(ids, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, next_token], dim=-1)
            tokens_generated += 1
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms, tokens_generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print('Loading model...', flush=True)
model, tokenizer = load_original_model(MODEL_PATH)
device = next(model.parameters()).device

inner = model.model
original_layers = list(inner.layers)
original_num_layers = len(original_layers)
print(f'Model loaded: {original_num_layers} layers', flush=True)

# Tokenize prompt and pad/truncate to PROMPT_TOKENS
encoded = tokenizer(PROMPT, return_tensors='pt')['input_ids']
if encoded.shape[1] < PROMPT_TOKENS:
    # Repeat prompt to reach target length
    text = PROMPT
    while True:
        encoded = tokenizer(text, return_tensors='pt')['input_ids']
        if encoded.shape[1] >= PROMPT_TOKENS:
            break
        text = text + ' ' + PROMPT
    encoded = encoded[:, :PROMPT_TOKENS]
else:
    encoded = encoded[:, :PROMPT_TOKENS]
input_ids = encoded.to(device)
print(f'Prompt tokens: {input_ids.shape[1]}', flush=True)

all_results = {}

for config_name, config in CONFIGS.items():
    print(f'\\n{\"=\" * 60}', flush=True)
    print(f'Config: {config_name}', flush=True)
    print(f'  blocks: {config[\"blocks\"]}', flush=True)
    print(f'  use_cache: {config[\"use_cache\"]}', flush=True)

    # Apply config
    num_layers = apply_config(model, original_layers, original_num_layers, config['blocks'])
    print(f'  effective layers: {num_layers}', flush=True)

    use_cache = config['use_cache']
    config_results = {
        'blocks': [list(b) for b in config['blocks']],
        'use_cache': use_cache,
        'num_layers': num_layers,
    }

    # Reset peak VRAM tracker
    torch.cuda.reset_peak_memory_stats()

    # --- TTFT ---
    print(f'  Measuring TTFT ({NUM_RUNS} runs)...', flush=True)
    ttft_times = []
    for run in range(NUM_RUNS):
        ttft_ms = measure_ttft(model, input_ids, use_cache)
        ttft_times.append(ttft_ms)
        print(f'    run {run+1}: {ttft_ms:.1f} ms', flush=True)
    config_results['ttft_ms'] = {
        'runs': ttft_times,
        'mean': sum(ttft_times) / len(ttft_times),
    }

    # --- Tokens/sec for each generation length ---
    gen_results = {}
    for gen_len in GEN_LENGTHS:
        print(f'  Measuring generation speed @ {gen_len} tokens ({NUM_RUNS} runs)...', flush=True)
        run_data = []
        for run in range(NUM_RUNS):
            elapsed_ms, tokens_gen = measure_generation_speed(
                model, tokenizer, input_ids, gen_len, use_cache
            )
            tps = (tokens_gen / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
            run_data.append({
                'elapsed_ms': elapsed_ms,
                'tokens_generated': tokens_gen,
                'tokens_per_sec': tps,
            })
            print(f'    run {run+1}: {tokens_gen} tok in {elapsed_ms:.0f} ms = {tps:.2f} tok/s', flush=True)

        mean_tps = sum(r['tokens_per_sec'] for r in run_data) / len(run_data)
        mean_ms = sum(r['elapsed_ms'] for r in run_data) / len(run_data)
        gen_results[str(gen_len)] = {
            'runs': run_data,
            'mean_tokens_per_sec': mean_tps,
            'mean_elapsed_ms': mean_ms,
        }

    config_results['generation'] = gen_results
    config_results['peak_vram_gb'] = get_peak_vram_gb()
    print(f'  Peak VRAM: {config_results[\"peak_vram_gb\"]:.2f} GB', flush=True)

    all_results[config_name] = config_results

    # Cleanup between configs
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
print(f'\\n\\n{\"=\" * 90}', flush=True)
print(f'{\"INFERENCE SPEED COMPARISON TABLE\":^90}', flush=True)
print(f'{\"=\" * 90}', flush=True)

header = f'{\"Config\":<22} {\"Layers\":>6} {\"TTFT(ms)\":>10} {\"VRAM(GB)\":>9}'
for gl in GEN_LENGTHS:
    header += f' {f\"tok/s@{gl}\":>12}'
print(header, flush=True)
print('-' * 90, flush=True)

for config_name, r in all_results.items():
    row = f'{config_name:<22} {r[\"num_layers\"]:>6} {r[\"ttft_ms\"][\"mean\"]:>10.1f} {r[\"peak_vram_gb\"]:>9.2f}'
    for gl in GEN_LENGTHS:
        tps = r['generation'][str(gl)]['mean_tokens_per_sec']
        row += f' {tps:>12.2f}'
    print(row, flush=True)

print('-' * 90, flush=True)

# Relative slowdown vs baseline_cache
baseline = all_results.get('baseline_cache')
if baseline:
    print(f'\\nRelative to baseline_cache:', flush=True)
    for config_name, r in all_results.items():
        if config_name == 'baseline_cache':
            continue
        ttft_ratio = r['ttft_ms']['mean'] / baseline['ttft_ms']['mean']
        row = f'  {config_name:<22} TTFT: {ttft_ratio:.2f}x'
        for gl in GEN_LENGTHS:
            b_tps = baseline['generation'][str(gl)]['mean_tokens_per_sec']
            c_tps = r['generation'][str(gl)]['mean_tokens_per_sec']
            ratio = c_tps / b_tps if b_tps > 0 else 0
            row += f'  tok/s@{gl}: {ratio:.2f}x'
        print(row, flush=True)

print(f'\\n{\"=\" * 90}', flush=True)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output_dir = 'results/data/72b/inference_speed'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'results.json')

with open(output_path, 'w') as f:
    json.dump({
        'model': MODEL_PATH,
        'prompt_tokens': PROMPT_TOKENS,
        'generation_lengths': GEN_LENGTHS,
        'num_runs': NUM_RUNS,
        'configs': all_results,
    }, f, indent=2)
print(f'\\nResults saved to {output_path}', flush=True)
print(f'Done at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}', flush=True)
"

echo "=== Done at $(date) ==="
