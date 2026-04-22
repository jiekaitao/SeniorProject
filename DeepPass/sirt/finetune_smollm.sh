#!/bin/bash
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --account=cis4914
#SBATCH --qos=cis4914
#SBATCH --output=/blue/cis4914/jietao/DeepPass/results/sbatch_sirt_finetune_%j.log
#SBATCH --job-name=sirt_ft

# Convert SmolLM-135M to SIRT format and fine-tune Stages 2+3
# Skip Stage 1 entirely — SmolLM already trained on 600B tokens

cd /blue/cis4914/jietao/DeepPass
export LD_PRELOAD=/blue/cis4914/jietao/DeepPass/envs/deeppass/lib/libstdc++.so.6
PYTHON=/blue/cis4914/jietao/DeepPass/envs/deeppass/bin/python

echo "=== SIRT Fine-tune from SmolLM-135M ==="
echo "Started: $(date)"

$PYTHON -c "
import os, sys, json, time, math, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, 'sirt')
sys.path.insert(0, 'scripts')

CACHE_DIR = '/blue/cis4914/jietao/hf_cache'
DATA_DIR = 'sirt/data'
SAVE_DIR = 'sirt/checkpoints_ft'
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================================
# Step 1: Load SmolLM-135M and inspect architecture
# ======================================================================
print('=== Step 1: Load SmolLM-135M ===', flush=True)
model_id = 'HuggingFaceTB/SmolLM-135M'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_id, cache_dir=CACHE_DIR, trust_remote_code=True)

print(f'Config:', flush=True)
print(f'  hidden_size: {config.hidden_size}', flush=True)
print(f'  num_layers: {config.num_hidden_layers}', flush=True)
print(f'  num_heads: {config.num_attention_heads}', flush=True)
print(f'  num_kv_heads: {getattr(config, \"num_key_value_heads\", config.num_attention_heads)}', flush=True)
print(f'  intermediate_size: {config.intermediate_size}', flush=True)
print(f'  vocab_size: {config.vocab_size}', flush=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True
)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Total params: {n_params:,}', flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Get layer structure
inner = model.model if hasattr(model, 'model') else model.transformer
layers = inner.layers if hasattr(inner, 'layers') else inner.h
N = len(layers)
print(f'  Layers: {N}', flush=True)

# ======================================================================
# Step 2: Apply SIRT-style recursion using layer duplication
# ======================================================================
print('\\n=== Step 2: Convert to SIRT-style recursive model ===', flush=True)

# SmolLM-135M has 30 layers. Split into zones:
#   Prelude: layers 0-8 (9 layers, standard)
#   Recursive core: layers 9-14 (6 layers -> 3 blocks of 2, share weights)
#   Coda: layers 15-29 (15 layers, standard)
#
# Actually, simpler approach: just duplicate the best block(s) like DeepPass
# and train with Stage 2+3 curriculum to teach recursion

# First, find baseline performance
from math_probe import run_math_probe
from eq_bench_probe import run_eq_bench_probe, _load_questions

eq_all = _load_questions()

def gen(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def gen_long(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print('Baseline evaluation...', flush=True)
t0 = time.time()
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, questions=eq_all, verbose=False)
baseline = math_r['score'] * 50 + eq_r['score'] * 0.5
print(f'Baseline: math={math_r[\"score\"]:.4f} eq={eq_r[\"score\"]:.1f} combined={baseline:.2f} ({time.time()-t0:.0f}s)', flush=True)

# ======================================================================
# Step 3: SBUID quick screen to find best block
# ======================================================================
print('\\n=== Step 3: Quick SBUID Screen ===', flush=True)

original_layers = list(layers)
cal_prompts = ['What is 127 * 348?', 'What is 99999 * 99999?',
               'What is the capital of France?', 'Explain entropy simply.']

sbuid_data = []
for start in range(N - 1):
    block = (start, start + 1)
    rhos, bloods = [], []
    for prompt in cal_prompts[:3]:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out_base = model(inputs['input_ids'], use_cache=False)
            logits_base = out_base.logits[:, -1, :].float()

            hb, ha, ctr = [None], [None], [0]
            def make_bh(c, b, a):
                def hook(mod, inp, out):
                    c[0] += 1
                    h = out[0] if isinstance(out, tuple) else out
                    if c[0] % 2 == 1: b[0] = h.detach().float()
                    else: a[0] = h.detach().float()
                    return out
                return hook
            bh = original_layers[start].register_forward_hook(make_bh(ctr, hb, ha))

            order = list(range(start + 1)) + [start] + list(range(start + 1, N))
            inner.layers = nn.ModuleList([original_layers[idx] for idx in order])
            model.config.num_hidden_layers = len(order)

            out_dup = model(inputs['input_ids'], use_cache=False)
            logits_dup = out_dup.logits[:, -1, :].float()
            bh.remove()

            inner.layers = nn.ModuleList(original_layers)
            model.config.num_hidden_layers = N

            num = torch.norm(logits_dup - logits_base).item()
            den = torch.norm(logits_base).item()
            if den > 1e-8: rhos.append(num / den)
            if hb[0] is not None and ha[0] is not None:
                bloods.append(torch.norm(ha[0] - hb[0]).item())

    mr = float(np.mean(rhos)) if rhos else 1.0
    mb = float(np.mean(bloods)) if bloods else 0.0
    sbuid_data.append({'layer': start, 'rho': mr, 'blood': mb, 'sbuid': mb - 6000*mr})
    if start % 5 == 0:
        print(f'  L{start}: sbuid={mb-6000*mr:.0f}', flush=True)

sbuid_data.sort(key=lambda x: x['sbuid'], reverse=True)
print(f'\\nTop 5 by SBUID:', flush=True)
for s in sbuid_data[:5]:
    print(f'  L{s[\"layer\"]}: sbuid={s[\"sbuid\"]:.0f}', flush=True)

# ======================================================================
# Step 4: Evaluate top blocks with dual probe
# ======================================================================
print('\\n=== Step 4: Evaluate Top Blocks ===', flush=True)

# LayerIdxWrapper for KV cache
class LayerIdxWrapper(nn.Module):
    def __init__(self, layer, new_idx):
        super().__init__()
        self.layer = layer
        self.new_layer_idx = new_idx
        self.orig_idx = layer.layer_idx
        self.orig_attn = layer.self_attn.layer_idx if hasattr(layer, 'self_attn') else None
    def forward(self, *args, **kwargs):
        self.layer.layer_idx = self.new_layer_idx
        if hasattr(self.layer, 'self_attn'): self.layer.self_attn.layer_idx = self.new_layer_idx
        try: return self.layer(*args, **kwargs)
        finally:
            self.layer.layer_idx = self.orig_idx
            if self.orig_attn is not None: self.layer.self_attn.layer_idx = self.orig_attn
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self.layer, name)

def apply_dup(start):
    order = list(range(start + 1)) + [start] + list(range(start + 1, N))
    seen = set()
    new_layers = []
    for pi, oi in enumerate(order):
        l = original_layers[oi]
        if oi in seen:
            new_layers.append(LayerIdxWrapper(l, pi))
        else:
            l.layer_idx = pi
            if hasattr(l, 'self_attn'): l.self_attn.layer_idx = pi
            new_layers.append(l)
        seen.add(oi)
    inner.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(order)
    model.config.use_cache = True

def restore():
    inner.layers = nn.ModuleList(original_layers)
    model.config.num_hidden_layers = N
    for i, l in enumerate(original_layers):
        l.layer_idx = i
        if hasattr(l, 'self_attn'): l.self_attn.layer_idx = i

results = []
for s in sbuid_data[:8]:
    layer = s['layer']
    apply_dup(layer)
    t0 = time.time()
    math_r = run_math_probe(gen, verbose=False)
    eq_r = run_eq_bench_probe(gen_long, verbose=False)
    combined = math_r['score'] * 50 + eq_r['score'] * 0.5
    delta = combined - baseline
    restore()
    print(f'  L{layer}: combined={combined:.2f} delta={delta:+.2f} ({time.time()-t0:.0f}s)', flush=True)
    results.append({'layer': layer, 'combined': combined, 'delta': delta,
                    'math': math_r['score'], 'eq': eq_r['score']})

results.sort(key=lambda x: x['combined'], reverse=True)
best = results[0]
print(f'\\nBest: L{best[\"layer\"]} = {best[\"combined\"]:.2f} (+{best[\"delta\"]:.2f})', flush=True)

# ======================================================================
# Step 5: Sublayer analysis on best block
# ======================================================================
print(f'\\n=== Step 5: Sublayer Analysis L{best[\"layer\"]} ===', flush=True)

best_layer = best['layer']

# Attn-only
apply_dup(best_layer)
module = original_layers[best_layer]
ctr = [0]
def make_zero_ffn(c):
    def hook(mod, inp, out):
        c[0] += 1
        if c[0] % 2 == 0:
            return (0.0 * out[0],) + out[1:] if isinstance(out, tuple) else 0.0 * out
        return out
    return hook
h = module.mlp.register_forward_hook(make_zero_ffn(ctr))
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, verbose=False)
attn_only = math_r['score'] * 50 + eq_r['score'] * 0.5
h.remove(); restore()
print(f'  Attn-only: {attn_only:.2f} (FFN impact: {attn_only - best[\"combined\"]:+.2f})', flush=True)

# Whisper FFN
apply_dup(best_layer)
ctr2 = [0]
def make_whisper(c):
    def hook(mod, inp, out):
        c[0] += 1
        if c[0] % 2 == 0:
            return (0.2 * out[0],) + out[1:] if isinstance(out, tuple) else 0.2 * out
        return out
    return hook
h = module.mlp.register_forward_hook(make_whisper(ctr2))
math_r = run_math_probe(gen, verbose=False)
eq_r = run_eq_bench_probe(gen_long, verbose=False)
whisper = math_r['score'] * 50 + eq_r['score'] * 0.5
h.remove(); restore()
print(f'  Whisper: {whisper:.2f}', flush=True)

# Save everything
with open(f'{SAVE_DIR}/smollm_analysis.json', 'w') as f:
    json.dump({
        'model': model_id, 'params': n_params, 'layers': N,
        'baseline': baseline,
        'sbuid_top5': sbuid_data[:5],
        'evaluated': results,
        'best_block': best,
        'attn_only': attn_only,
        'whisper_ffn02': whisper,
        'ffn_impact': attn_only - best['combined'],
    }, f, indent=2)
print(f'\\nSaved to {SAVE_DIR}/smollm_analysis.json', flush=True)

# ======================================================================
# Summary
# ======================================================================
print(f'\\n{\"=\" * 50}', flush=True)
print(f'SUMMARY — SmolLM-135M Layer Duplication', flush=True)
print(f'Baseline: {baseline:.2f}', flush=True)
print(f'Best dup L{best[\"layer\"]}: {best[\"combined\"]:.2f} (+{best[\"delta\"]:.2f})', flush=True)
print(f'Attn-only: {attn_only:.2f}', flush=True)
print(f'Whisper: {whisper:.2f}', flush=True)
print(f'FFN impact: {attn_only - best[\"combined\"]:+.2f}', flush=True)
print('COMPLETE', flush=True)
"

echo "=== Finished: $(date) ==="
