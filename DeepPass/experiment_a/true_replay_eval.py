"""
TRUE REPLAY TEST: Train standard LoRA, then at eval time run
LoRA-adapted middle layers 1x vs 2x vs 3x.

This isolates genuine replay benefit from LoRA quality.
"""
import os, sys, time, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_lora_any import get_data_stream

def main():
    device = torch.device('cuda')
    model_path = 'models/full/Llama-3.1-8B'

    print(f'Loading {model_path}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='auto')

    # Standard LoRA on L0-19
    layers = list(range(20))
    target_modules = [f'model.layers.{l}.self_attn.{p}_proj' for l in layers for p in 'qkvo']
    lora_config = LoraConfig(r=32, lora_alpha=16, target_modules=target_modules,
                             lora_dropout=0.0, bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Phase 1: Train
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = 5000
    warmup = 250
    def lr_sched(step):
        if step < warmup: return step / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    print(f'\n=== Phase 1: Standard LoRA training (5000 steps) ===', flush=True)
    data_stream = get_data_stream(tokenizer, seq_len=1024, batch_size=2)
    model.train()
    step = 0
    running_loss = 0
    t0 = time.time()

    for batch in data_stream:
        if step >= total_steps:
            break
        batch = batch.to(device)
        out = model(batch[:, :-1], labels=batch[:, :-1])
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += out.loss.item()
        step += 1
        if step % 500 == 0:
            print(f'  step {step:5d} | loss={running_loss/500:.4f} | {time.time()-t0:.0f}s', flush=True)
            running_loss = 0

    # Phase 2: TRUE REPLAY EVALUATION
    print(f'\n=== Phase 2: TRUE REPLAY EVALUATION ===', flush=True)
    print(f'Running LoRA-adapted middle layers 1x, 2x, 3x', flush=True)

    # Build eval data
    ds = iter(load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                           split='train', streaming=True))
    buf = []
    while len(buf) < 1025 * 2 * 30 + 500:
        ex = next(ds)
        t = ex.get('text', '')
        if t and len(t) > 50:
            toks = tokenizer.encode(t, add_special_tokens=False, truncation=True, max_length=2048)
            toks.append(tokenizer.eos_token_id)
            buf.extend(toks)

    model.eval()
    model.enable_adapter_layers()  # LoRA ON for ALL passes

    # Access inner model layers
    inner = model.base_model.model
    replay_layers = set(range(12, 16))  # replay middle band

    for n_passes in [1, 2, 3]:
        total_loss = 0
        total_tok = 0
        buf_copy = list(buf)

        with torch.no_grad():
            for _ in range(20):
                batch = []
                for _ in range(2):
                    batch.append(buf_copy[:1025])
                    buf_copy = buf_copy[1024:]
                t = torch.tensor(batch, dtype=torch.long).to(device)
                input_ids = t[:, :-1]
                targets = t[:, 1:]

                # Manual forward with replay
                h = inner.model.embed_tokens(input_ids)
                position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
                position_embeddings = inner.model.rotary_emb(h, position_ids)

                for i, layer in enumerate(inner.model.layers):
                    h = layer(h, position_embeddings=position_embeddings)
                    # Extra passes on replay layers
                    if i in replay_layers:
                        for _ in range(n_passes - 1):
                            h = layer(h, position_embeddings=position_embeddings)

                h = inner.model.norm(h)
                logits = inner.lm_head(h.to(inner.lm_head.weight.dtype))
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                total_loss += loss.item() * targets.numel()
                total_tok += targets.numel()

        ppl = math.exp(total_loss / max(total_tok, 1))
        print(f'  K={n_passes}: PPL={ppl:.2f}', flush=True)

    print(f'\nIf K=2 < K=1: genuine replay benefit on LoRA-adapted model', flush=True)
    print(f'If K=2 >= K=1: replay does NOT help even with adapted weights', flush=True)
    print(f'=== Done ===', flush=True)


if __name__ == '__main__':
    main()
