#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dar_1b_noreplay_%j.log
#SBATCH --job-name=dar1b

# DAR 1B control: K=1 ONLY (no replay, pure dense baseline)
# This is the exact same model as DAR but never uses replay
# Should match our original dense baseline PPL (~59-63)
# Confirms dense containment empirically
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== DAR 1B (no replay control) ===" && echo "Started: $(date)"

# Hack: set replay probability to 0 by using K=1 always
# We modify the training to always use K=1
envs/deeppass/bin/python -c "
import sys; sys.path.insert(0, 'psrt')
from train_dar import *
import argparse

# Override: always K=1
args = argparse.Namespace(
    size='1b', total_steps=100000, batch_size=4, seq_len=2048,
    lr=1.5e-4, save_dir='psrt/checkpoints/dar_noreplay'
)

device = torch.device('cuda')
model, cfg = create_dar(args.size)
model = model.to(device)
tokenizer = get_tokenizer()

total = args.total_steps
warmup = int(total * 0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

def lr_schedule(step):
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

print(f'\n=== DAR Training (K=1 only, no replay) ===', flush=True)
print(f'  {model.count_params()/1e6:.0f}M params', flush=True)

gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
model.train()
step = 0; running_loss = 0; t0 = time.time(); best_ppl = float('inf')

for input_ids, labels in gen:
    if step >= total: break
    input_ids = input_ids.to(device)
    _, loss, _ = model(input_ids, labels=input_ids, K=1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    running_loss += loss.item(); step += 1

    if step % 100 == 0:
        print(f'  step {step:6d} | loss={running_loss/100:.4f} | {time.time()-t0:.0f}s', flush=True)
        running_loss = 0
    if step % 2000 == 0:
        ppl = evaluate_ppl(model, tokenizer, device, n=15, seq_len=args.seq_len, bs=args.batch_size)
        print(f'  --- EVAL step {step}: PPL K=1={ppl[\"K=1\"]:.2f} K=2={ppl[\"K=2\"]:.2f} (delta={ppl[\"K=2\"]-ppl[\"K=1\"]:+.2f}) ---', flush=True)
        if ppl['K=1'] < best_ppl:
            best_ppl = ppl['K=1']
            torch.save({'step': step, 'model_state': model.state_dict(), 'config': cfg.__dict__, 'ppl': ppl}, f'{save_dir}/best.pt')
            print(f'  --- SAVED best ---', flush=True)

ppl = evaluate_ppl(model, tokenizer, device, n=30, seq_len=args.seq_len, bs=args.batch_size)
print(f'\n=== Complete: PPL K=1={ppl[\"K=1\"]:.2f} K=2={ppl[\"K=2\"]:.2f} Best={best_ppl:.2f} ===', flush=True)
"

echo "=== Finished: $(date) ==="
