# Core Modules

Foundation scripts imported by all experiments.

## Files

- **`layer_duplicator.py`** — Engine for RYS layer duplication. `apply_layer_duplication(model, i, j)` modifies the layer list to run layers [i,j) twice. `generate_no_cache()` generates text without KV cache (required for duplicated models). `load_original_model()` loads HF models in bfloat16.

- **`math_probe.py`** — Ng's hard math guesstimate probe. 16 difficult arithmetic questions scored with partial credit. Primary evaluation metric across all experiments.

- **`save_duplicated_model.py`** — Saves a duplicated model with deep-copied layers via `save_pretrained()`. Must deep-copy shared layers first.

- **`compile_results.py`** — Aggregates JSON results from all experiments into unified format.

## Usage

All experiment scripts import from core via:
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer_duplicator import load_original_model, apply_layer_duplication, generate_no_cache
from math_probe import run_math_probe
```
