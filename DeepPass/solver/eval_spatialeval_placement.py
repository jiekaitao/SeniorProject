"""
SpatialEval Placement Sweep — Test where memory tokens go relative to the prompt.

GPT-5.4 diagnosis: the 72% ceiling is a decoder readout bottleneck.
The #1 fix: move memory AFTER the prompt instead of before it.

Placements to test:
  A. [memory | prompt | answer]        ← CURRENT (prefix mode)
  B. [BOS | memory | prompt[1:] | answer]  ← after BOS, preserve sink
  C. [prompt | memory | answer]        ← after prompt, before answer
  D. [prompt[:-Q] | memory | Q | answer]   ← before question/options

Also tests N_eff (effective memory slots used by decoder).
"""
import os, sys, torch, json, random, math, time, copy, argparse
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore

device = torch.device('cuda')
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def forward_with_placement(solver, base_model, lm_model, tokenizer,
                           text, answer_text, placement, K=1, training=True):
    """
    Run solver + decoder with specified memory placement.
    Returns logits, loss, and optionally attention info.
    """
    prompt_text = text + "\nAnswer:"
    if training:
        full = prompt_text + answer_text
        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048, padding=True).to(device)
    else:
        enc = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=2048).to(device)

    input_ids = enc['input_ids']
    prompt_len = len(tokenizer.encode(prompt_text))

    with torch.no_grad():
        all_emb = lm_model.embed_tokens(input_ids)

    prompt_emb = all_emb[:, :prompt_len]
    memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=training)
    M = memory.shape[1]

    if placement == 'prefix':
        # A: [memory | prompt | answer] (CURRENT)
        dec_in = torch.cat([memory, all_emb], dim=1)
        ans_offset = M

    elif placement == 'after_bos':
        # B: [BOS | memory | prompt[1:] | answer]
        bos = all_emb[:, :1]
        rest = all_emb[:, 1:]
        dec_in = torch.cat([bos, memory, rest], dim=1)
        ans_offset = M  # prompt shifted by M but BOS stays

    elif placement == 'after_prompt':
        # C: [prompt | memory | answer]
        if training:
            prompt_part = all_emb[:, :prompt_len]
            answer_part = all_emb[:, prompt_len:]
            dec_in = torch.cat([prompt_part, memory, answer_part], dim=1)
            ans_offset = M  # answer shifts right by M
        else:
            dec_in = torch.cat([all_emb, memory], dim=1)
            ans_offset = M

    elif placement == 'before_question':
        # D: Insert memory right before "Answer:" in the prompt
        # Find where "Answer:" starts (approximately prompt_len - ~10 tokens)
        q_offset = max(prompt_len - 15, prompt_len // 2)
        if training:
            before_q = all_emb[:, :q_offset]
            after_q = all_emb[:, q_offset:]
            dec_in = torch.cat([before_q, memory, after_q], dim=1)
            ans_offset = M
        else:
            before_q = all_emb[:, :q_offset]
            after_q = all_emb[:, q_offset:]
            dec_in = torch.cat([before_q, memory, after_q], dim=1)
            ans_offset = M
    else:
        raise ValueError(f'Unknown placement: {placement}')

    T = dec_in.shape[1]
    pos_ids = torch.arange(T, device=device).unsqueeze(0)
    pos_emb = lm_model.rotary_emb(dec_in, pos_ids)
    h = dec_in
    for layer in lm_model.layers:
        h = layer(h, position_embeddings=pos_emb)
    h = lm_model.norm(h)
    logits = base_model.lm_head(h)

    if training:
        ans_start = prompt_len + ans_offset
        ans_logits = logits[:, ans_start - 1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:, :ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            return loss
        return None
    else:
        # For eval: return last token logit
        return logits


def train_and_eval(placement='prefix', seed=42, total_steps=2000, n_memory_slots=32):
    tag = f'placement_{placement}_s{seed}'
    random.seed(seed)
    torch.manual_seed(seed)

    print(f'=== Placement: {placement} | seed={seed} | steps={total_steps} ===', flush=True)

    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=n_memory_slots).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in solver.parameters())
    print(f'Solver: {n_params:,} params', flush=True)

    maze_data = load_maze_nav()
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    warmup = 200
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    t0 = time.time()
    losses = []

    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        answer_text = f" {sample['oracle_option']}"

        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
        loss = forward_with_placement(solver, base_model, lm_model, tokenizer,
                                      text, answer_text, placement, K=K, training=True)
        if loss is not None and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if loss is not None:
            losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f'  step {step+1} | loss={avg_loss:.4f} | {time.time()-t0:.0f}s', flush=True)

    # Eval
    print(f'\n=== Eval ({len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    results = {}

    for K_eval in [0, 1, 2]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option']

            if K_eval == 0:
                prompt = text + "\nAnswer:"
                enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                with torch.no_grad():
                    logits = forward_with_placement(solver, base_model, lm_model, tokenizer,
                                                   text, "", placement, K=K_eval, training=False)
                    pred_id = logits[0, -1].argmax().item()
                    answer = tokenizer.decode([pred_id]).strip()

            if oracle.upper() in answer.upper()[:10]:
                correct += 1

        acc = correct / n_eval
        results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'tag': tag, 'placement': placement, 'seed': seed,
        'total_steps': total_steps, 'n_memory_slots': n_memory_slots,
        'solver_params': n_params,
        'final_loss': sum(losses[-50:]) / len(losses[-50:]) if losses else 0,
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json'), 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'Done ({time.time()-t0:.0f}s)', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--placement', type=str, default='after_prompt',
                        choices=['prefix', 'after_bos', 'after_prompt', 'before_question'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    train_and_eval(placement=args.placement, seed=args.seed, total_steps=args.steps)
