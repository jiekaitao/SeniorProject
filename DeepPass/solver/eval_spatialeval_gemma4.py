"""
SpatialEval with Gemma 4 31B-IT as the frozen decoder + our solver.
Adapts SolverCore projection layers from 4096 (Llama) to 5376 (Gemma 4).
"""
import os, sys, torch, json, random, math, time
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_HOME'] = '/blue/cis4914/jietao/hf_cache'
sys.path.insert(0, os.path.dirname(__file__))

device = torch.device('cuda')
GEMMA_PATH = '/blue/cis4914/jietao/gemma/gemma-4-31b-it'
RESULTS_DIR = '/blue/cis4914/jietao/DeepPass/results/data/spatialeval'


class SolverCoreAdapted(nn.Module):
    """SolverCore adapted for arbitrary LLM hidden dim."""
    def __init__(self, llm_dim=5376, d_model=512, n_heads=8, ffn_dim=1024,
                 n_L_layers=2, n_memory_slots=32):
        super().__init__()
        self.d_model = d_model
        self.n_memory_slots = n_memory_slots

        self.proj_in = nn.Linear(llm_dim, d_model, bias=False)

        # L-level
        from model import BidirectionalBlock, CrossAttention, RMSNorm
        self.L_self = nn.ModuleList([
            BidirectionalBlock(d_model, n_heads, ffn_dim, has_ffn=(i == n_L_layers - 1))
            for i in range(n_L_layers)
        ])
        self.L_cross_H = CrossAttention(d_model, n_heads)

        # H-level
        self.H_self = BidirectionalBlock(d_model, n_heads, ffn_dim, has_ffn=True)
        self.H_cross_L = CrossAttention(d_model, n_heads)

        self.H_init = nn.Parameter(torch.randn(1, n_memory_slots, d_model) * 0.02)
        self.L_init_scale = nn.Parameter(torch.tensor(0.1))

        self.proj_out = nn.Linear(d_model, llm_dim, bias=False)
        self.out_norm = RMSNorm(llm_dim)

    def forward(self, prompt_embeddings, K_inner=4, K_outer=3, grad_last_only=True):
        B, T, _ = prompt_embeddings.shape
        e = self.proj_in(prompt_embeddings)
        z_L = self.L_init_scale * e
        z_H = self.H_init.expand(B, -1, -1).clone()

        for s in range(K_outer):
            use_grad = (not grad_last_only) or (s == K_outer - 1)
            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                for _ in range(K_inner):
                    z_L_input = z_L + e
                    z_L_input = z_L_input + self.L_cross_H(z_L_input, z_H)
                    for layer in self.L_self:
                        z_L_input = layer(z_L_input)
                    z_L = z_L_input
                z_H = z_H + self.H_cross_L(z_H, z_L)
                z_H = self.H_self(z_H)
            if grad_last_only and s < K_outer - 1:
                z_L = z_L.detach()
                z_H = z_H.detach()

        memory = self.out_norm(self.proj_out(z_H))
        return memory


def load_maze_nav():
    from datasets import load_dataset
    ds = load_dataset('MilaWang/SpatialEval', 'tqa', split='test')
    maze = [s for s in ds if s['id'].startswith('mazenav')]
    print(f'Loaded {len(maze)} Maze-Nav samples', flush=True)
    return maze


def train_and_eval(n_memory_slots=32, total_steps=2000, tag='gemma4_v1'):
    print("Loading Gemma 4 31B-IT...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH, dtype=torch.bfloat16, device_map='auto')
    for p in base_model.parameters():
        p.requires_grad = False

    # Get inner model
    lm_model = base_model.model.language_model
    llm_dim = lm_model.layers[0].self_attn.q_proj.in_features
    print(f"Loaded in {time.time()-t0:.0f}s. LLM dim={llm_dim}, Layers={len(lm_model.layers)}", flush=True)

    solver = SolverCoreAdapted(
        llm_dim=llm_dim, d_model=512, n_heads=8, ffn_dim=1024,
        n_L_layers=2, n_memory_slots=n_memory_slots
    ).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in solver.parameters())
    print(f'Solver params: {n_params:,} | memory_slots={n_memory_slots}', flush=True)

    maze_data = load_maze_nav()
    random.seed(42)
    indices = list(range(len(maze_data)))
    random.shuffle(indices)
    train_idx = indices[:1000]
    eval_idx = indices[1000:]

    # Training
    print(f'\n=== Training ({total_steps} steps, bypass mode, Gemma 4) ===', flush=True)
    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4, weight_decay=0.05)
    warmup = 200

    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    losses = []
    for step in range(total_steps):
        sample = maze_data[train_idx[step % len(train_idx)]]
        text = sample['text'][:1500]
        answer_text = f" {sample['oracle_option']}"

        # Use chat template for training too
        messages = [{"role": "user", "content": text + "\nAnswer with just the letter:"}]
        chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = chat_input + answer_text

        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=2048, padding=True).to(device)
        input_ids = enc['input_ids']
        prompt_len = len(tokenizer.encode(chat_input))

        with torch.no_grad():
            all_emb = lm_model.embed_tokens(input_ids)

        prompt_emb = all_emb[:, :prompt_len]

        # K curriculum
        if step < total_steps // 4:
            K = random.choices([1, 2], weights=[0.7, 0.3])[0]
        elif step < total_steps // 2:
            K = random.choices([1, 2, 4], weights=[0.2, 0.5, 0.3])[0]
        else:
            K = random.choices([1, 2, 4], weights=[0.1, 0.3, 0.6])[0]

        memory = solver(prompt_emb, K_inner=4, K_outer=K, grad_last_only=True)

        # BYPASS: [memory][prompt][answer]
        ans_emb = all_emb[:, prompt_len:]
        dec_in = torch.cat([memory, all_emb[:, :prompt_len], ans_emb], dim=1)
        M = memory.shape[1]
        T = dec_in.shape[1]
        pos_ids = torch.arange(T, device=device).unsqueeze(0)

        # Use model's own forward with inputs_embeds to handle Gemma 4's hybrid attention
        with torch.no_grad():
            # Temporarily disable grad for frozen model forward
            pass
        out = base_model(inputs_embeds=dec_in, use_cache=False)
        logits = out.logits

        ans_start = M + prompt_len
        ans_logits = logits[:, ans_start - 1:-1]
        ans_labels = input_ids[:, prompt_len:]
        ml = min(ans_logits.shape[1], ans_labels.shape[1])
        if ml > 0:
            loss = F.cross_entropy(ans_logits[:, :ml].reshape(-1, logits.shape[-1]),
                                   ans_labels[:, :ml].reshape(-1), ignore_index=tokenizer.pad_token_id)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            print(f'  step {step+1} | loss={avg_loss:.4f} | K={K} | {time.time()-t0:.0f}s', flush=True)

    # Save checkpoint
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(RESULTS_DIR, f'solver_{tag}.pt')
    torch.save(solver.state_dict(), ckpt_path)
    print(f'Saved: {ckpt_path}', flush=True)

    # Eval
    print(f'\n=== Evaluation (bypass, {len(eval_idx)} samples) ===', flush=True)
    solver.eval()
    results = {}

    for K_eval in [0, 1, 2, 4]:
        correct = 0
        n_eval = len(eval_idx)
        for idx in eval_idx:
            sample = maze_data[idx]
            text = sample['text'][:1500]
            oracle = sample['oracle_option']
            messages = [{"role": "user", "content": text + "\nAnswer with just the letter (A, B, C, or D):"}]
            chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat_input, return_tensors='pt', truncation=True, max_length=2048).to(device)

            if K_eval == 0:
                with torch.no_grad():
                    out = base_model.generate(enc['input_ids'], max_new_tokens=5, do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                with torch.no_grad():
                    all_emb = lm_model.embed_tokens(enc['input_ids'])
                    mem = solver(all_emb, K_inner=4, K_outer=K_eval, grad_last_only=False)
                    dec_in = torch.cat([mem, all_emb], dim=1)
                    M = mem.shape[1]
                    T_dec = dec_in.shape[1]
                    out = base_model(inputs_embeds=dec_in, use_cache=False)
                    lg = out.logits
                    pred_id = lg[0, -1].argmax().item()
                    answer = tokenizer.decode([pred_id]).strip()

            if oracle.upper() in answer.upper()[:10]:
                correct += 1

        acc = correct / n_eval
        results[f'K={K_eval}'] = {'accuracy': acc, 'correct': correct, 'total': n_eval}
        print(f'  K={K_eval}: accuracy={acc:.4f} ({correct}/{n_eval})', flush=True)

    result_data = {
        'tag': tag, 'base_model': 'gemma-4-31b-it',
        'n_memory_slots': n_memory_slots, 'total_steps': total_steps,
        'solver_params': n_params, 'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    result_path = os.path.join(RESULTS_DIR, f'spatialeval_{tag}.json')
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f'Saved: {result_path}', flush=True)
    print(f'Done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    train_and_eval(n_memory_slots=32, total_steps=2000, tag='gemma4_mem32')
