#!/bin/bash
#SBATCH --job-name=l12_conf
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_l12_confirm_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== L12 confirmation: 3 seeds + cross-task ==="
for SEED in 42 7 137 2024; do
    $PYTHON -c "
import sys, os
os.environ['HF_HOME']='/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, 'solver')
from eval_deliberation_creative import run_midlayer, get_choice_token_ids
import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained('models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
for p in base_model.parameters(): p.requires_grad = False

ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
maze_data = [s for s in ds if s['id'].startswith('mazenav')]
random.seed(0)
indices = list(range(len(maze_data)))
random.shuffle(indices)
train_idx, eval_idx = indices[:1000], indices[1000:]
choice_ids = get_choice_token_ids(tokenizer)
run_midlayer($SEED, 3000, 3, 12, tokenizer, base_model, maze_data, train_idx, eval_idx, choice_ids)
"
done
