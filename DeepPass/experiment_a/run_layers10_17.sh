#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_expa_L10_17_%j.log
#SBATCH --job-name=expa

# Experiment A: WIDE band — replay layers 10-17 (8 layers, double the standard)
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== Exp A (layers 10-17, wide band) ===" && echo "Started: $(date)"

envs/deeppass/bin/python -c "
import sys; sys.path.insert(0, 'experiment_a')
from train import *

device = torch.device('cuda')
model_path = 'models/full/Llama-3.1-8B'
print(f'Loading {model_path}...', flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='auto')
print(f'  Loaded in {time.time()-t0:.0f}s', flush=True)

model = BandRecurrentLlama(base_model, replay_layer_ids=(10,11,12,13,14,15,16,17), lora_rank=16, max_extra_passes=1)
print(f'  Trainable: {model.count_trainable():,} ({model.count_trainable()/model.count_total()*100:.3f}%)', flush=True)
print(f'  Replay layers: 10-17 (8 layers)', flush=True)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
total_steps = 10000; warmup = 500
def lr_sched(step):
    if step < warmup: return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

data_stream = get_data_stream(tokenizer, seq_len=1024, batch_size=2)
model.train()
step = 0; running_loss = 0; t0 = time.time()

for batch in data_stream:
    if step >= total_steps: break
    batch = batch.to(device)
    input_ids = batch[:, :-1]; labels = batch[:, :-1]
    K = 2
    _, loss = model(input_ids, labels=labels, K=K, hard_frac=0.25)
    if loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    running_loss += loss.item(); step += 1
    if step % 50 == 0:
        gates = [f'{torch.tanh(e.gate).item():.4f}' for e in model.extra_layers.values()]
        mix = torch.tanh(model.mix_gate.to(torch.float32)).item()
        print(f'  step {step:5d} | loss={running_loss/50:.4f} | mix={mix:.4f} | gates=[{\",\".join(gates)}] | {time.time()-t0:.0f}s', flush=True)
        running_loss = 0
    if step % 500 == 0:
        ppl = evaluate(model, tokenizer, device, n_batches=15, seq_len=1024, bs=2)
        delta = ppl['K=2'] - ppl['K=1']
        print(f'  --- EVAL step {step}: PPL K=1={ppl[\"K=1\"]:.2f} K=2={ppl[\"K=2\"]:.2f} (delta={delta:+.2f}) ---', flush=True)

ppl = evaluate(model, tokenizer, device, n_batches=30, seq_len=1024, bs=2)
print(f'\n=== Complete: PPL K=1={ppl[\"K=1\"]:.2f} K=2={ppl[\"K=2\"]:.2f} (delta={ppl[\"K=2\"]-ppl[\"K=1\"]:+.2f}) ===', flush=True)
"

echo "=== Finished: $(date) ==="
