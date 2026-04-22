#!/bin/bash
#SBATCH --job-name=llama32
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_llama32_3b_%j.log
#SBATCH --account=cis4914
#SBATCH --partition=hpg-b200
#SBATCH --qos=cis4914
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
module load conda/25.7.0
eval "$(conda shell.bash hook)"
conda activate /blue/cis4914/jietao/DeepPass/envs/deeppass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
cd /blue/cis4914/jietao/DeepPass
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

# Check if model exists, if not download
if [ ! -d "models/full/Llama-3.2-3B" ]; then
    echo "Downloading Llama 3.2 3B..."
    $PYTHON -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
tok.save_pretrained('models/full/Llama-3.2-3B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')
model.save_pretrained('models/full/Llama-3.2-3B')
print('Downloaded!')
"
fi

echo "=== Llama 3.2 3B on SpatialEval mazenav + spatialmap ==="
$PYTHON solver/eval_deliberation_multi_benchmark.py \
    --benchmarks winogrande,piqa,openbookqa \
    --model models/full/Llama-3.2-3B \
    --rounds 3 --seeds 42,7 --steps 2000 --grad_accum 8

echo "=== Llama 3.2 3B on SpatialEval ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task mazenav --rounds 3 \
    --seeds 42 --steps 2000
