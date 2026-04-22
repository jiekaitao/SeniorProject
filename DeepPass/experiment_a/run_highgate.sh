#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_expa_highgate_%j.log
#SBATCH --job-name=expa
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Exp A HIGH GATE (init=0.5) ==="
envs/deeppass/bin/python -c "
import sys; sys.path.insert(0, 'experiment_a')
from band_recurrent import BandRecurrentLlama, ExtraPassLayer
import torch

# Monkey-patch gate init to 0.5
orig_init = ExtraPassLayer.__init__
def new_init(self, base_layer, d_model, lora_rank=16):
    orig_init(self, base_layer, d_model, lora_rank)
    self.gate = torch.nn.Parameter(torch.tensor(0.5))
ExtraPassLayer.__init__ = new_init

# Also patch mix gate
orig_band_init = BandRecurrentLlama.__init__
def new_band_init(self, *args, **kwargs):
    orig_band_init(self, *args, **kwargs)
    self.mix_gate = torch.nn.Parameter(torch.tensor(0.5))
BandRecurrentLlama.__init__ = new_band_init

from train import train
train(replay_layers=(12,13,14,15))
"
exit 0
