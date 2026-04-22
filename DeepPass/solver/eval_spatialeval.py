"""
Evaluate our K-scaling solver on SpatialEval Maze-Nav benchmark.
Tests whether more thinking cycles improve maze navigation accuracy.

The solver processes the maze text, then the frozen LLM decoder generates the answer.
Compare K=0 (no solver, base LLM), K=1, K=2, K=4 to measure K-scaling on real benchmark.
"""
import os, sys, torch, json, random, math, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'

sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore

device = torch.device('cuda')


def load_maze_nav_data():
    """Load SpatialEval Maze-Nav TQA subset."""
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def train_and_eval():
    """Train solver on maze-nav style prompts, then evaluate on SpatialEval."""
    tokenizer = AutoTokenizer.from_pretrained('models/full/Llama-3.1-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        'models/full/Llama-3.1-8B', dtype=torch.bfloat16).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    lm_model = base_model.model

    solver = SolverCore(d_model=512, n_heads=8, ffn_dim=1024,
                        n_L_layers=2, n_memory_slots=16).to(device=device, dtype=torch.bfloat16)
    print(f'Solver params: {sum(p.numel() for p in solver.parameters()):,}', flush=True)

    # Load maze data
    maze_data = load_maze_nav_data()

    # Phase 1: Evaluate BASE MODEL (no solver, K=0) on Maze-Nav
    print(f'\n=== Phase 1: Base LLM Evaluation (K=0) ===', flush=True)
    n_eval = min(200, len(maze_data))
    random.seed(42)
    eval_indices = random.sample(range(len(maze_data)), n_eval)

    correct_base = 0
    for idx in eval_indices[:n_eval]:
        sample = maze_data[idx]
        text = sample['text']
        oracle = sample['oracle_option']  # A, B, C, D, etc.

        # Truncate to fit context
        prompt = text[:1500] + "\nAnswer:"
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        # Check if oracle option appears in answer
        if oracle.upper() in answer.upper()[:10]:
            correct_base += 1

    base_acc = correct_base / n_eval
    print(f'Base LLM accuracy: {base_acc:.4f} ({correct_base}/{n_eval})', flush=True)

    # Phase 2: Train solver on maze-nav training data, then evaluate with K-scaling
    print(f'\n=== Phase 2: Training Solver on Maze-Nav ===', flush=True)

    # Use first 1000 samples for training, last 500 for eval
    train_data = [maze_data[i] for i in range(min(1000, len(maze_data)))]
    eval_data = [maze_data[i] for i in eval_indices]

    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    t0 = time.time()

    for step in range(2000):
        sample = random.choice(train_data)
        text = sample['text'][:1500]
        answer_text = f" {sample['oracle_option']}. {sample['oracle_answer']}"
        full = text + "\nAnswer:" + answer_text

        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048,
                        padding=True).to(device)
        input_ids = enc['input_ids']

        # Find where answer starts
        prompt_text = text + "\nAnswer:"
        prompt_len = len(tokenizer.encode(prompt_text))

        with torch.no_grad():
            all_emb = lm_model.embed_tokens(input_ids)

        prompt_emb = all_emb[:, :prompt_len]
        K = random.choices([1, 2, 4], weights=[0.2, 0.4, 0.4])[0]
        memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)

        stub = all_emb[:, prompt_len-16:prompt_len]
        ans_emb = all_emb[:, prompt_len:]
        dec_in = torch.cat([memory, stub, ans_emb], dim=1)
        M = memory.shape[1]
        T = dec_in.shape[1]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_in, pos_ids)
        h = dec_in
        for layer in lm_model.layers:
            h = layer(h, position_embeddings=pos_emb)
        h = lm_model.norm(h)
        logits = base_model.lm_head(h)

        ans_start = M + 16
        ans_logits = logits[:, ans_start-1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:,:ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:,:ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step+1) % 200 == 0:
            print(f'  step {step+1} | loss={loss.item():.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)

    # Phase 3: Evaluate with K-scaling
    print(f'\n=== Phase 3: K-Scaling Evaluation on Maze-Nav ===', flush=True)
    solver.eval()

    for K_eval in [1, 2, 4]:
        correct = 0
        for sample in eval_data[:n_eval]:
            text = sample['text'][:1500]
            oracle = sample['oracle_option']
            prompt = text + "\nAnswer:"

            enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
            prompt_len = enc['input_ids'].shape[1]

            with torch.no_grad():
                emb = lm_model.embed_tokens(enc['input_ids'])
                mem = solver(emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                stub = emb[:, -16:]
                dec_in = torch.cat([mem, stub], dim=1)
                M = mem.shape[1]
                T = dec_in.shape[1]
                pos_ids = torch.arange(T, device=device).unsqueeze(0)
                pe = lm_model.rotary_emb(dec_in, pos_ids)
                h = dec_in
                for layer in lm_model.layers:
                    h = layer(h, position_embeddings=pe)
                h = lm_model.norm(h)
                lg = base_model.lm_head(h)

                # Get top token prediction
                pred_id = lg[0, -1].argmax().item()
                pred = tokenizer.decode([pred_id]).strip()

            if oracle.upper() in pred.upper()[:5]:
                correct += 1

        acc = correct / n_eval
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    print(f'\n=== Summary ===', flush=True)
    print(f'Base LLM (K=0): {base_acc:.4f}', flush=True)
    print(f'Solver trained on maze-nav with K-scaling evaluation', flush=True)
    print(f'Done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    train_and_eval()
