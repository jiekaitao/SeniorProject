"""Collect all experimental data into a single JSON for the static webpage.
Real data only. No synthetic."""
import os, json, glob, re
from collections import defaultdict

ROOT = '/blue/cis4914/jietao/DeepPass/results/data'
OUT = '/blue/cis4914/jietao/DeepPass/webpage/data.json'

# Also gather from new dirs added in the recent wave
EXTRA_DIRS = ['peft_variants', 'hybrid_aggressive', 'multitask', 'surgery',
              'vision_lora', 'distill']

def safe_load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def get_best_controller_acc(d):
    """Get best accuracy across K-scaling rounds."""
    if 'results' not in d:
        return None
    best = None
    for k, v in d['results'].items():
        if k.startswith('rounds') and isinstance(v, dict) and 'accuracy' in v:
            if best is None or v['accuracy'] > best:
                best = v['accuracy']
    return best

# ============================================================
# 1. Controller benchmark results (HellaSwag, WinoGrande, etc.)
# ============================================================
bench_results = []
for f in sorted(glob.glob(f'{ROOT}/benchmarks/*.json')):
    d = safe_load(f)
    if not d: continue
    bl = d['results'].get('baseline', {}).get('accuracy', 0)
    best = get_best_controller_acc(d) or 0
    # Parse tag: bench_inst_hellaswag_L12_8k_s42
    tag = d.get('tag', os.path.basename(f).replace('.json',''))
    model = 'inst' if 'inst' in tag else 'base'
    bench_results.append({
        'tag': tag, 'benchmark': d.get('benchmark', 'unknown'),
        'model': model, 'inject_layer': d.get('inject_layer', 12),
        'total_steps': d.get('total_steps', 8000),
        'seed': d.get('seed', 42), 'baseline': bl, 'best': best,
        'delta': best - bl,
        'all_rounds': {k: v.get('accuracy', 0) for k, v in d['results'].items()
                       if k.startswith('rounds') and isinstance(v, dict)},
    })

# ============================================================
# 2. LoRA baseline results
# ============================================================
lora_results = []
for f in sorted(glob.glob(f'{ROOT}/lora_baseline/*.json')):
    d = safe_load(f)
    if not d: continue
    bl = d['results'].get('baseline', {}).get('accuracy', 0)
    fn = d['results'].get('lora_final', {}).get('accuracy', 0)
    tag = d.get('tag', os.path.basename(f).replace('.json',''))
    model = 'inst' if 'inst' in tag else 'base'
    lora_results.append({
        'tag': tag, 'benchmark': d.get('benchmark', 'unknown'),
        'model': model, 'lora_r': d.get('lora_r', 64),
        'total_steps': d.get('total_steps', 8000),
        'seed': d.get('seed', 42), 'baseline': bl, 'final': fn,
        'delta': fn - bl,
    })

# ============================================================
# 3. SpatialEval controller results (mega/)
# ============================================================
mega_results = []
for f in sorted(glob.glob(f'{ROOT}/mega/*.json')):
    d = safe_load(f)
    if not d: continue
    if 'results' not in d: continue
    best = get_best_controller_acc(d)
    if best is None: continue
    tag = d.get('tag', os.path.basename(f).replace('.json',''))
    # Guess task
    task = 'spatialgrid'
    if 'mz_' in tag or 'mazenav' in tag: task = 'mazenav'
    elif 'sm_' in tag or 'spatialmap' in tag: task = 'spatialmap'
    elif 'sr_' in tag or 'spatialreal' in tag: task = 'spatialreal'
    elif 'sg_' in tag or 'spatialgrid' in tag: task = 'spatialgrid'
    model = 'inst' if ('inst' in tag.lower() or 'Inst' in tag) else 'base'
    mega_results.append({
        'tag': tag, 'task': task, 'model': model,
        'inject_layer': d.get('inject_layer', -1),
        'total_steps': d.get('total_steps', 0),
        'seed': d.get('seed', -1),
        'best': best,
        'all_rounds': {k: v.get('accuracy', 0) for k, v in d['results'].items()
                       if k.startswith('rounds') and isinstance(v, dict)},
    })

# ============================================================
# 4. LoRA on SpatialEval
# ============================================================
lora_spatial_results = []
for f in sorted(glob.glob(f'{ROOT}/lora_baseline/lora_*_spatialgrid_*.json') +
                glob.glob(f'{ROOT}/lora_baseline/lora_*_mazenav_*.json') +
                glob.glob(f'{ROOT}/lora_baseline/lora_*_spatialmap_*.json')):
    d = safe_load(f)
    if not d: continue
    bl = d['results'].get('baseline', {}).get('accuracy', 0)
    fn = d['results'].get('lora_final', {}).get('accuracy', 0)
    tag = d.get('tag', '')
    model = 'inst' if 'inst' in tag else 'base'
    task = d.get('benchmark', '')
    lora_spatial_results.append({
        'tag': tag, 'task': task, 'model': model,
        'lora_r': d.get('lora_r', 64),
        'total_steps': d.get('total_steps', 8000),
        'seed': d.get('seed', 42),
        'baseline': bl, 'final': fn, 'delta': fn - bl,
    })

# ============================================================
# 5. Vision results
# ============================================================
vision_results = []
for f in sorted(glob.glob(f'{ROOT}/vision/*.json')):
    d = safe_load(f)
    if not d: continue
    tag = d.get('tag', '')
    bl = d['results'].get('baseline', {}).get('accuracy', 0)
    # Best across round=3,5,8
    best = 0
    for k, v in d['results'].items():
        if k.startswith('rounds') and isinstance(v, dict):
            if v.get('accuracy', 0) > best:
                best = v['accuracy']
    vision_results.append({
        'tag': tag,
        'model_type': d.get('model_type', 'unknown'),
        'inject_layer': d.get('inject_layer', -1),
        'total_steps': d.get('total_steps', 0),
        'seed': d.get('seed', 42),
        'baseline': bl, 'best': best, 'delta': best - bl,
    })

# ============================================================
# 6. DeepPass 72B results (key configs)
# ============================================================
# Key 72B results extracted from PAPER.md
deeppass_72b = [
    {'config': 'Baseline (Qwen2-72B)', 'combined': 70.52, 'delta': 0, 'method': 'no modification'},
    {'config': "Ng's (45,52) @1.0", 'combined': 76.76, 'delta': 6.24, 'method': 'single block duplicate'},
    {'config': '(50,60) @1.0', 'combined': 76.49, 'delta': 5.97, 'method': 'single block, different region'},
    {'config': 'Pair (0,7)+(45,52) @1.0', 'combined': 79.91, 'delta': 9.39, 'method': 'greedy spectral stacking'},
    {'config': 'Whisper-alpha quad', 'combined': 82.58, 'delta': 12.06, 'method': 'per-block alpha tuning'},
    {'config': 'Per-layer single (45,52)', 'combined': 82.77, 'delta': 12.25, 'method': '7 layer alphas'},
    {'config': 'Bayesian per-layer triple', 'combined': 83.97, 'delta': 13.45, 'method': '60 Optuna evals'},
    {'config': 'Grid per-layer triple (BEST)', 'combined': 84.07, 'delta': 13.55, 'method': '300 evals, 21 alphas'},
]

# Cross-architecture DeepPass
deeppass_cross = [
    {'model': 'Qwen2-72B', 'baseline': 70.52, 'best': 84.07, 'delta': 13.55, 'layers': 80},
    {'model': 'Gemma3-27B', 'baseline': 80.54, 'best': 85.58, 'delta': 5.04, 'layers': 62},
    {'model': 'Qwen3.5-27B', 'baseline': 42.86, 'best': 80.05, 'delta': 37.19, 'layers': 64},
    {'model': 'Qwen3.5-9B', 'baseline': 40.19, 'best': 54.27, 'delta': 14.08, 'layers': 32},
    {'model': 'Qwen3-30B MoE', 'baseline': 27.76, 'best': 40.42, 'delta': 12.66, 'layers': 48},
]

# Depth progression on 72B
deeppass_depth = [
    {'depth': 1, 'config': '(45,52)@1.15', 'combined': 79.79},
    {'depth': 2, 'config': '(0,7)@0.9 + (45,52)@1.0', 'combined': 81.24},
    {'depth': 3, 'config': '(0,7)+(20,27)@0.15+(45,52)', 'combined': 82.29},
    {'depth': 4, 'config': 'quad with whisper alphas', 'combined': 82.58},
    {'depth': 3, 'config': 'per-layer triple', 'combined': 84.07},
]

# K-scaling data from controller
# From benchmarks — use mean across seeds
controller_kscaling = defaultdict(lambda: {'rounds=3': [], 'rounds=5': [], 'rounds=8': []})
for b in bench_results:
    key = f"{b['benchmark']}_{b['model']}"
    for k, v in b['all_rounds'].items():
        if k in controller_kscaling[key]:
            controller_kscaling[key][k].append(v)

# Mean for each
kscaling_data = {}
for key, rounds_d in controller_kscaling.items():
    means = {k: (sum(v)/len(v) if v else 0) for k,v in rounds_d.items()}
    kscaling_data[key] = means

# Paired comparison: controller vs LoRA
paired = []
# Group by (benchmark, model)
ctrl_by_key = defaultdict(list)
for b in bench_results:
    if 'L12' in b['tag'] and '8k' in b['tag']:  # canonical settings
        key = f"{b['benchmark']}_{b['model']}"
        ctrl_by_key[key].append(b)

lora_by_key = defaultdict(list)
for l in lora_results:
    if '8k' in l['tag']:
        key = f"{l['benchmark']}_{l['model']}"
        lora_by_key[key].append(l)

for key in sorted(set(list(ctrl_by_key.keys()) + list(lora_by_key.keys()))):
    ctrls = ctrl_by_key.get(key, [])
    loras = lora_by_key.get(key, [])
    bench_name, model = key.rsplit('_', 1)
    if not ctrls and not loras: continue
    ctrl_deltas = [c['delta'] for c in ctrls]
    lora_deltas = [l['delta'] for l in loras]
    ctrl_accs = [c['best'] for c in ctrls]
    lora_accs = [l['final'] for l in loras]
    paired.append({
        'benchmark': bench_name, 'model': model,
        'n_ctrl': len(ctrls), 'n_lora': len(loras),
        'ctrl_delta_mean': sum(ctrl_deltas)/max(len(ctrl_deltas),1),
        'lora_delta_mean': sum(lora_deltas)/max(len(lora_deltas),1),
        'ctrl_acc_mean': sum(ctrl_accs)/max(len(ctrl_accs),1),
        'lora_acc_mean': sum(lora_accs)/max(len(lora_accs),1),
        'ctrl_deltas_all': ctrl_deltas, 'lora_deltas_all': lora_deltas,
    })

# SpatialGrid LoRA instability analysis
sg_lora_runs = [l for l in lora_spatial_results if l['task'] == 'spatialgrid']
# sg controller from mega/
sg_ctrl_runs = [m for m in mega_results if m['task'] == 'spatialgrid' and m['total_steps'] == 8000 and m['inject_layer'] == 12]

# ============================================================
# New experiments — collect from latest dirs
# ============================================================
extra = {}
for d in EXTRA_DIRS:
    items = []
    for f in sorted(glob.glob(f'{ROOT}/{d}/*.json')):
        js = safe_load(f)
        if js:
            items.append(js)
    extra[d] = items

# Write out
data = {
    'meta': {
        'total_experiments': len(bench_results) + len(lora_results) + len(mega_results) + len(vision_results),
        'controller_benchmark_runs': len(bench_results),
        'lora_baseline_runs': len(lora_results),
        'spatialeval_controller_runs': len(mega_results),
        'spatialeval_lora_runs': len(lora_spatial_results),
        'vision_runs': len(vision_results),
    },
    'bench_results': bench_results,
    'lora_results': lora_results,
    'mega_results': mega_results,
    'lora_spatial_results': lora_spatial_results,
    'vision_results': vision_results,
    'deeppass_72b': deeppass_72b,
    'deeppass_cross': deeppass_cross,
    'deeppass_depth': deeppass_depth,
    'paired_controller_vs_lora': paired,
    'sg_lora_runs': sg_lora_runs,
    'sg_ctrl_runs': sg_ctrl_runs,
    'kscaling_data': kscaling_data,
    'new_peft_variants': extra.get('peft_variants', []),
    'new_hybrid_aggressive': extra.get('hybrid_aggressive', []),
    'new_multitask': extra.get('multitask', []),
    'new_surgery': extra.get('surgery', []),
    'new_vision_lora': extra.get('vision_lora', []),
    'new_distill': extra.get('distill', []),
}

with open(OUT, 'w') as f:
    json.dump(data, f, indent=1)

print(f'Wrote {OUT}')
print(f'Total experiments: {data["meta"]["total_experiments"]}')
for k, v in data['meta'].items():
    print(f'  {k}: {v}')
