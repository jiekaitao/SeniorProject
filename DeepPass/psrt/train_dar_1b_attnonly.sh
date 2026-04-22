#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_dar_1b_attnonly_%j.log
#SBATCH --job-name=dar1b

# DAR 1B attention-only replay: K=2 ALWAYS, MLP gate forced to 0
# Tests pure attention replay benefit without any MLP replay
# This is the strongest version of "repeat attention, not FFN"
cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
echo "=== DAR 1B (attention-only, K=2 always) ===" && echo "Started: $(date)"

envs/deeppass/bin/python -c "
import sys; sys.path.insert(0, 'psrt')
from train_dar import *
import argparse

args = argparse.Namespace(
    size='1b', total_steps=100000, batch_size=4, seq_len=2048,
    lr=1.5e-4, save_dir='psrt/checkpoints/dar_attnonly'
)

device = torch.device('cuda')
model, cfg = create_dar(args.size)
model = model.to(device)
tokenizer = get_tokenizer()

# Force MLP replay gates to stay at -10 (effectively zero)
for blk in model.blocks:
    if hasattr(blk, 'replay_mlp_gate'):
        blk.replay_mlp_gate.requires_grad = False
        blk.replay_mlp_gate.fill_(-10.0)

total = args.total_steps
warmup = int(total * 0.05)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
)

def lr_schedule(step):
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

print(f'\n=== DAR Training (attention-only replay, K=2 always) ===', flush=True)
print(f'  {model.count_params()/1e6:.0f}M params', flush=True)

gen = mixed_stream(tokenizer, args.seq_len, args.batch_size)
model.train()
step = 0; running_loss = 0; t0 = time.time(); best_ppl = float('inf')

for input_ids, labels in gen:
    if step >= total: break
    input_ids = input_ids.to(device)

    # Always K=2 with attention-only replay
    _, loss, aux = model(input_ids, labels=input_ids, K=2)

    # Also keep K=1 path healthy
    _, loss_k1, _ = model(input_ids, labels=input_ids, K=1)
    loss = loss + 0.5 * loss_k1

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    running_loss += loss.item(); step += 1

    if step % 100 == 0:
        ag = [f'{torch.tanh(blk.replay_attn_gate[0]).item():.4f}' for blk in model.blocks if hasattr(blk, 'replay_attn_gate')]
        print(f'  step {step:6d} | loss={running_loss/100:.4f} | attn_g=[{\",\".join(ag)}] | {time.time()-t0:.0f}s', flush=True)
        running_loss = 0
    if step % 2000 == 0:
        ppl = evaluate_ppl(model, tokenizer, device, n=15, seq_len=args.seq_len, bs=args.batch_size)
        delta = ppl['K=2'] - ppl['K=1']
        print(f'  --- EVAL step {step}: PPL K=1={ppl[\"K=1\"]:.2f} K=2={ppl[\"K=2\"]:.2f} (delta={delta:+.2f}) ---', flush=True)
        if ppl['K=1'] < best_ppl:
            best_ppl = ppl['K=1']
            torch.save({'step': step, 'model_state': model.state_dict(), 'config': cfg.__dict__, 'ppl': ppl}, f'{save_dir}/best.pt')
            print(f'  --- SAVED best ---', flush=True)

ppl = evaluate_ppl(model, tokenizer, device, n=30, seq_len=args.seq_len, bs=args.batch_size)
print(f'\n=== Complete: PPL K=1={ppl[\"K=1\"]:.2f} K=2={ppl[\"K=2\"]:.2f} Best={best_ppl:.2f} ===', flush=True)
"

echo "=== Finished: $(date) ==="
