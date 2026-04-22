"""Aggregate and report all findings with real data."""
import json, glob, os
from collections import defaultdict

ROOT = '/blue/cis4914/jietao/DeepPass/results/data'

def load_all(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


# ============================================================
# Key findings table
# ============================================================
print('=' * 75)
print('                       CONTROLLER vs PEFT — ALL BENCHMARKS')
print('=' * 75)

# Controller results (from benchmarks/ and mega/)
def bench_best(d):
    return max((v['accuracy'] for k,v in d.get('results', {}).items()
                if k.startswith('rounds') and isinstance(v, dict)), default=0)

def lora_final(d):
    r = d.get('results', {}).get('final') or d.get('results', {}).get('lora_final')
    return r['accuracy'] if r else 0

# Aggregate by (benchmark, model)
controller_by_key = defaultdict(list)   # key -> list of (acc, seed, steps)
for f in glob.glob(f'{ROOT}/benchmarks/*.json'):
    d = json.load(open(f))
    if 'L12' in d.get('tag', '') and d.get('total_steps') == 8000:
        key = f"{d['benchmark']}_{ 'inst' if 'inst' in d['tag'] else 'base'}"
        controller_by_key[key].append((bench_best(d), d['seed'], d['total_steps'],
                                        d['results']['baseline']['accuracy']))

# LoRA results (vanilla)
lora_by_key = defaultdict(list)
for f in glob.glob(f'{ROOT}/lora_baseline/*.json'):
    d = json.load(open(f))
    # skip the spatialgrid/mazenav/spatialmap (those are SpatialEval)
    if 'spatialgrid' in d['tag'] or 'mazenav' in d['tag'] or 'spatialmap' in d['tag']:
        continue
    if d.get('total_steps', 0) != 8000: continue
    key = f"{d['benchmark']}_{ 'inst' if 'inst' in d['tag'] else 'base'}"
    lora_by_key[key].append((lora_final(d), d['seed'], d['total_steps'],
                              d['results']['baseline']['accuracy']))

# Advanced PEFT on SpatialGrid
peft_sg = {}
for f in glob.glob(f'{ROOT}/peft_variants/peft_*_spatialgrid_*.json'):
    d = json.load(open(f))
    k = (d['method'], 'inst' if 'inst' in d['tag'] else 'base', d['seed'])
    peft_sg[k] = (lora_final(d), d['results']['baseline']['accuracy'])

# Advanced PEFT on other benchmarks
peft_other = {}
for f in glob.glob(f'{ROOT}/peft_variants/peft_*.json'):
    d = json.load(open(f))
    if 'spatialgrid' in d['tag']: continue
    k = (d['method'], d['benchmark'], 'inst' if 'inst' in d['tag'] else 'base', d['seed'])
    peft_other[k] = (lora_final(d), d['results']['baseline']['accuracy'])

# SpatialGrid controller from mega/ (L12, 8k, 5 rounds)
sg_ctrl = []
for f in glob.glob(f'{ROOT}/mega/*L12*8k*.json'):
    d = json.load(open(f))
    if d.get('inject_layer') != 12 or d.get('total_steps') != 8000: continue
    task = 'spatialgrid' if ('spatialgrid' in f or '_sg_' in f or '_sg_8k' in d.get('tag','')) else None
    if task != 'spatialgrid': continue
    model = 'inst' if ('inst' in f.lower()) else 'base'
    best = bench_best(d)
    sg_ctrl.append((model, d.get('seed', -1), best))

def summarize_ctrl(key):
    items = controller_by_key.get(key, [])
    if not items: return None
    accs = [a for a, _, _, _ in items]
    bls = [bl for _, _, _, bl in items]
    return {'mean': sum(accs)/len(accs), 'n': len(accs),
            'min': min(accs), 'max': max(accs), 'baseline': sum(bls)/len(bls)}

def summarize_lora(key):
    items = lora_by_key.get(key, [])
    if not items: return None
    accs = [a for a, _, _, _ in items]
    bls = [bl for _, _, _, bl in items]
    return {'mean': sum(accs)/len(accs), 'n': len(accs),
            'min': min(accs), 'max': max(accs), 'baseline': sum(bls)/len(bls)}

print(f'{"Task":20s} {"Model":5s} {"Ctrl mean":>10s} {"LoRA mean":>10s} {"Winner":>12s}')
print('-' * 70)
for key in sorted(set(list(controller_by_key.keys()) + list(lora_by_key.keys()))):
    bench, model = key.rsplit('_', 1)
    cs = summarize_ctrl(key)
    ls = summarize_lora(key)
    if not cs and not ls: continue
    cm = f"{cs['mean']:.3f} (n={cs['n']})" if cs else 'n/a'
    lm = f"{ls['mean']:.3f} (n={ls['n']})" if ls else 'n/a'
    winner = ''
    if cs and ls:
        if cs['mean'] - ls['mean'] > 0.02: winner = 'CONTROLLER'
        elif ls['mean'] - cs['mean'] > 0.02: winner = 'LORA'
        else: winner = 'tie'
    print(f'{bench:20s} {model:5s} {cm:>15s} {lm:>15s} {winner:>12s}')

print()
print('=' * 75)
print('                       PEFT VARIANTS ON SPATIALGRID')
print('=' * 75)
for (method, model, seed), (acc, bl) in sorted(peft_sg.items()):
    status = 'COLLAPSE' if acc < 0.3 else 'OK' if acc > bl else 'flat'
    print(f'  {method:10s} {model:4s} seed={seed}: baseline={bl:.3f} -> {acc:.3f} '
          f'(delta {acc-bl:+.3f}) {status}')

if peft_other:
    print()
    print('=' * 75)
    print('                       PEFT ON OTHER BENCHMARKS')
    print('=' * 75)
    for (method, bench, model, seed), (acc, bl) in sorted(peft_other.items()):
        print(f'  {method:10s} {bench:15s} {model:4s}: baseline={bl:.3f} -> {acc:.3f} '
              f'(delta {acc-bl:+.3f})')

# Multitask
print()
print('=' * 75)
print('                       MULTITASK ZERO-SHOT TRANSFER')
print('=' * 75)
for f in glob.glob(f'{ROOT}/multitask/*.json'):
    d = json.load(open(f))
    print(f'Trained on: {d["train_benchmarks"]}')
    for b, rs in d['results'].items():
        best = max(v['accuracy'] for k, v in rs.items() if 'rounds' in k)
        held = any(v.get('is_heldout') for v in rs.values() if isinstance(v, dict))
        tag = '[HELDOUT]' if held else '[train]'
        print(f'  {b:20s} {tag}: {best:.4f}')

# Mazenav surgery
print()
print('=' * 75)
print('                       MAZENAV SURGERY (no-X hypothesis test)')
print('=' * 75)
for f in sorted(glob.glob(f'{ROOT}/surgery/*.json')):
    d = json.load(open(f))
    bl = d.get('results', {}).get('baseline', {}).get('accuracy', 0)
    lf = lora_final(d)
    best_r = bench_best(d) if d.get('results', {}).get('rounds=5') else lf
    method = 'LORA' if 'lora' in d.get('tag', '') else 'CTRL'
    print(f'  {method:4s} {d.get("tag", "")}: baseline={bl:.3f} -> {best_r:.3f}')

# Vision
print()
print('=' * 75)
print('                       VISION — PaLiGemma VSR')
print('=' * 75)
for f in sorted(glob.glob(f'{ROOT}/vision/*.json') + glob.glob(f'{ROOT}/vision_lora/*.json')):
    d = json.load(open(f))
    bl = d.get('results', {}).get('baseline', {}).get('accuracy', 0)
    if 'lora_final' in d.get('results', {}):
        fn = d['results']['lora_final']['accuracy']
        method = 'LORA'
    else:
        fn = bench_best(d)
        method = 'CTRL'
    tag = d.get('tag', os.path.basename(f))
    print(f'  {method:4s} {tag}: baseline={bl:.3f} -> {fn:.3f} (delta {fn-bl:+.3f})')
