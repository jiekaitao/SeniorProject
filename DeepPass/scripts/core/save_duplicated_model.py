"""Save the layer-duplicated model with deep-copied layers so save_pretrained works."""
import sys, os, copy, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_duplicator import load_original_model
import torch.nn as nn

MODEL_PATH = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b"
SAVE_PATH = "/blue/cis4914/jietao/DeepPass/models/full/calme-2.1-qwen2-72b-dup-45-52"

I, J = 45, 52

print("Loading model...")
model, tokenizer = load_original_model(MODEL_PATH)

inner = model.model
layers = list(inner.layers)
N = len(layers)

# Build duplicated layer sequence with DEEP COPIES for the repeated block
# [0..J-1] + deep_copy([I..J-1]) + [J..N-1]
print(f"Deep-copying layers {I}-{J-1} for duplication...")
new_layers = layers[:J]
for idx in range(I, J):
    new_layers.append(copy.deepcopy(layers[idx]))
new_layers.extend(layers[J:])

inner.layers = nn.ModuleList(new_layers)
model.config.num_hidden_layers = len(new_layers)
print(f"New model: {len(new_layers)} layers ({N} original + {J-I} duplicated)")

print(f"Saving to {SAVE_PATH}...")
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH, max_shard_size="5GB")
tokenizer.save_pretrained(SAVE_PATH)
print("Done saving!")

del model
torch.cuda.empty_cache()

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(SAVE_PATH)
print(f"Verified: num_hidden_layers={cfg.num_hidden_layers}")
