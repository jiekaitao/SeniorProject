#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_spectral_validation_%j.log
#SBATCH --job-name=deeppass_spectral

cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== Spectral screening validation data for paper ==="
echo "Started: $(date)"

# Compute displacement rho for ALL 77 brain scanner configs on 7B
# and correlate with actual math probe scores
# This produces the data for the spectral screening heatmap figure
$PYTHON -c "
import sys, json, torch, numpy as np
sys.path.insert(0, 'scripts')
from layer_duplicator import load_original_model, generate_no_cache
import torch.nn as nn

model, tokenizer = load_original_model('models/small/Qwen2-7B-Instruct')
inner = model.model
original_layers = list(inner.layers)
N = len(original_layers)
device = next(model.parameters()).device

# Calibration prompts
prompts = [
    'What is 127 * 348?',
    'What is 99999 * 99999?',
    'Calculate 15! / 13!',
    'What is 2^16?',
    'What is the sum of all integers from 1 to 100?',
    'What is 7^5?',
    'If f(x) = 3x^2 - 2x + 1, what is f(5)?',
    'What is 13^3?',
]

# Compute displacement rho for all blocks with step=2
results = []
for start in range(0, N-1, 2):
    for end in range(start+2, min(start+15, N+1), 2):
        block = (start, end)

        rhos = []
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                h = inner.embed_tokens(ids['input_ids'])
                seq_len = h.shape[1]
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_embeds = inner.rotary_emb(h, pos_ids)

                for layer_idx in range(start):
                    out = inner.layers[layer_idx](h, position_embeddings=pos_embeds, use_cache=False)
                    h = out[0] if isinstance(out, tuple) else out

                h_input = h.clone()

                h1 = h_input.clone()
                for layer_idx in range(start, end):
                    out = inner.layers[layer_idx](h1, position_embeddings=pos_embeds, use_cache=False)
                    h1 = out[0] if isinstance(out, tuple) else out

                h2 = h1.clone()
                for layer_idx in range(start, end):
                    out = inner.layers[layer_idx](h2, position_embeddings=pos_embeds, use_cache=False)
                    h2 = out[0] if isinstance(out, tuple) else out

                num = torch.norm(h2 - h1).item()
                den = torch.norm(h1 - h_input).item()
                if den > 1e-8:
                    rhos.append(num / den)

        mean_rho = float(np.mean(rhos)) if rhos else 1.0
        results.append({
            'block': list(block),
            'displacement_rho': mean_rho,
            'block_size': end - start,
        })
        print(f'  ({start:2d},{end:2d}): rho={mean_rho:.4f}')

import os
os.makedirs('results/data/7b/spectral', exist_ok=True)
with open('results/data/7b/spectral/displacement_rho_all.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved {len(results)} blocks')
"

echo "=== Done at $(date) ==="
