#!/bin/bash
#SBATCH --job-name=pk_3.2
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_packed_llama32_%j.log
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

# Download if needed
if [ ! -d "models/full/Llama-3.2-3B" ]; then
    echo "Downloading Llama 3.2 3B..."
    $PYTHON -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os; os.environ['HF_HOME']='/blue/cis4914/jietao/hf_cache'
t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
t.save_pretrained('models/full/Llama-3.2-3B')
m = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')
m.save_pretrained('models/full/Llama-3.2-3B')
print('Done!')
"
fi

echo "=== PACKED: Llama 3.2 3B on ALL benchmarks ==="
$PYTHON solver/eval_deliberation_multi_benchmark.py \
    --benchmarks winogrande,piqa,openbookqa,boolq \
    --model models/full/Llama-3.2-3B \
    --rounds 3 --seeds 42,7 --steps 2000 --grad_accum 8

echo "=== Llama 3.2 3B on SpatialEval ==="
$PYTHON solver/eval_deliberation_hybrid_writer.py \
    --modes lowrank_only --task mazenav --rounds 3 \
    --seeds 42 --steps 2000
