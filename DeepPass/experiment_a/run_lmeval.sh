#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_lmeval_%j.log
#SBATCH --job-name=lmeval
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6

echo "=== LM-Eval: Base Llama vs LoRA Replay ==="
echo "Started: $(date)"

envs/deeppass/bin/python -c "
import os, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import lm_eval
from lm_eval.models.huggingface import HFLM

model_path = 'models/full/Llama-3.1-8B'
print(f'Loading {model_path}...', flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 1. Evaluate BASE model (no LoRA)
# ============================================================
print(f'\n=== BASELINE (no LoRA) ===', flush=True)
base_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='auto')
print(f'  Loaded in {time.time()-t0:.0f}s', flush=True)

lm = HFLM(pretrained=base_model, tokenizer=tokenizer, batch_size=4)

tasks = [
    'arc_challenge',
    'hellaswag',
    'winogrande',
    'gsm8k',
    'mmlu',
]

print(f'  Running benchmarks: {tasks}', flush=True)
base_results = lm_eval.simple_evaluate(
    model=lm,
    tasks=tasks,
    num_fewshot=0,
    batch_size=4,
    limit=200,  # 200 samples per task for speed
)

print(f'\n--- BASELINE RESULTS ---', flush=True)
for task, res in base_results['results'].items():
    acc = res.get('acc,none', res.get('acc_norm,none', res.get('exact_match,strict-match', '?')))
    print(f'  {task}: {acc}', flush=True)

# Save
with open('experiment_a/lmeval_baseline.json', 'w') as f:
    json.dump({k: {kk: str(vv) for kk, vv in v.items()} for k, v in base_results['results'].items()}, f, indent=2)

# ============================================================
# 2. Evaluate LoRA model (best config: L0-19 r32)
# ============================================================
print(f'\n=== LoRA REPLAY (L0-19 r32) ===', flush=True)

# Apply LoRA to layers 0-19
target_layers = list(range(20))
target_modules = [f'model.layers.{l}.self_attn.{p}_proj' for l in target_layers for p in 'qkvo']
lora_config = LoraConfig(r=32, lora_alpha=16, target_modules=target_modules, lora_dropout=0.0, bias='none', task_type=TaskType.CAUSAL_LM)
lora_model = get_peft_model(base_model, lora_config)

# Load best checkpoint if available
ckpt_dir = 'experiment_a/checkpoints_lora/best'
if os.path.exists(ckpt_dir):
    print(f'  Loading checkpoint from {ckpt_dir}', flush=True)
    from peft import PeftModel
    # Reload with saved adapters
    del lora_model
    lora_model = PeftModel.from_pretrained(base_model, ckpt_dir)
    print(f'  Loaded saved LoRA adapters', flush=True)
else:
    print(f'  No checkpoint found — using untrained LoRA (will match baseline)', flush=True)

lora_model.enable_adapter_layers()
lm_lora = HFLM(pretrained=lora_model, tokenizer=tokenizer, batch_size=4)

print(f'  Running benchmarks: {tasks}', flush=True)
lora_results = lm_eval.simple_evaluate(
    model=lm_lora,
    tasks=tasks,
    num_fewshot=0,
    batch_size=4,
    limit=200,
)

print(f'\n--- LoRA REPLAY RESULTS ---', flush=True)
for task, res in lora_results['results'].items():
    acc = res.get('acc,none', res.get('acc_norm,none', res.get('exact_match,strict-match', '?')))
    print(f'  {task}: {acc}', flush=True)

with open('experiment_a/lmeval_lora.json', 'w') as f:
    json.dump({k: {kk: str(vv) for kk, vv in v.items()} for k, v in lora_results['results'].items()}, f, indent=2)

# ============================================================
# 3. Comparison
# ============================================================
print(f'\n=== COMPARISON ===', flush=True)
print(f'  {\"Task\":<20} {\"Baseline\":>10} {\"LoRA\":>10} {\"Delta\":>10}', flush=True)
print(f'  {\"-\"*20} {\"-\"*10} {\"-\"*10} {\"-\"*10}', flush=True)
for task in base_results['results']:
    if task in lora_results['results']:
        b = base_results['results'][task]
        l = lora_results['results'][task]
        b_acc = b.get('acc,none', b.get('acc_norm,none', b.get('exact_match,strict-match', 0)))
        l_acc = l.get('acc,none', l.get('acc_norm,none', l.get('exact_match,strict-match', 0)))
        try:
            b_val = float(b_acc)
            l_val = float(l_acc)
            delta = l_val - b_val
            print(f'  {task:<20} {b_val:>10.4f} {l_val:>10.4f} {delta:>+10.4f}', flush=True)
        except (ValueError, TypeError):
            print(f'  {task:<20} {b_acc:>10} {l_acc:>10}', flush=True)

print(f'\n=== Done: $(date) ===', flush=True)
"

echo "=== Finished: $(date) ==="
exit 0
