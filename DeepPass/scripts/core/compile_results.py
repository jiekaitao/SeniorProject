"""
DeepPass Results Compiler

Run this after all experiments finish to generate a comprehensive summary.
Compares baseline vs duplicated across all metrics and validates the
spectral prediction hypothesis.
"""

import json, os, sys
from pathlib import Path
from datetime import datetime


RESULTS_DIR = Path("/blue/cis4914/jietao/DeepPass/results")


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def find_lm_eval_results(result_dir):
    """Find and parse lm-eval output files."""
    result_dir = Path(result_dir)
    if not result_dir.exists():
        return None

    # lm-eval saves results in a subdirectory
    for f in result_dir.rglob("results*.json"):
        data = load_json(str(f))
        if data and "results" in data:
            return data["results"]
        elif data:
            return data
    return None


def compile_all():
    print("=" * 80)
    print("DEEPPASS COMPREHENSIVE RESULTS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Math Probe Results
    print("\n" + "=" * 60)
    print("1. MATH PROBE (Ng's hard guesstimate)")
    print("=" * 60)

    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and "calme" in d.name:
            math = load_json(str(d / "math_probe.json"))
            if math:
                print(f"  {d.name}: score={math['score']:.4f}")

    # 2. Leaderboard Benchmarks
    print("\n" + "=" * 60)
    print("2. LEADERBOARD BENCHMARKS (lm-eval)")
    print("=" * 60)

    baseline_lm = find_lm_eval_results(RESULTS_DIR / "lm_eval_baseline_72b")
    dup_lm = find_lm_eval_results(RESULTS_DIR / "lm_eval_dup45_52_72b")

    if baseline_lm:
        print("\n  Baseline (calme-2.1-qwen2-72b, 80 layers):")
        for task, metrics in sorted(baseline_lm.items()):
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k in ['acc', 'acc_norm', 'exact_match']:
                        print(f"    {task:40s} {k}: {v:.4f}")
    else:
        print("  Baseline: NOT YET AVAILABLE")

    if dup_lm:
        print("\n  Duplicated (45,52) — 87 layers:")
        for task, metrics in sorted(dup_lm.items()):
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and k in ['acc', 'acc_norm', 'exact_match']:
                        print(f"    {task:40s} {k}: {v:.4f}")
    else:
        print("  Duplicated: NOT YET AVAILABLE")

    if baseline_lm and dup_lm:
        print("\n  COMPARISON:")
        for task in baseline_lm:
            if task in dup_lm:
                b = baseline_lm[task]
                d = dup_lm[task]
                if isinstance(b, dict) and isinstance(d, dict):
                    for k in ['acc', 'acc_norm', 'exact_match']:
                        if k in b and k in d:
                            delta = d[k] - b[k]
                            pct = (delta / b[k] * 100) if b[k] > 0 else 0
                            print(f"    {task:30s} {k}: {b[k]:.4f} → {d[k]:.4f} ({delta:+.4f}, {pct:+.1f}%)")

    # 3. Spectral Analysis
    print("\n" + "=" * 60)
    print("3. SPECTRAL ANALYSIS")
    print("=" * 60)

    for name in ["spectral_7B", "spectral_72B"]:
        spec = load_json(str(RESULTS_DIR / name / "spectral_results.json"))
        if spec:
            results = spec.get("results", {})
            print(f"\n  {name} ({spec.get('num_layers', '?')} layers, {len(results)} configs):")
            # Find configs with lowest displacement rho
            scored = []
            for k, v in results.items():
                if "error" not in v and "displacement_rho" in v:
                    i, j = map(int, k.split(","))
                    scored.append((k, v["displacement_rho"], v.get("perturbation_rho", 0)))
            if scored:
                scored.sort(key=lambda x: x[1])
                print(f"    Top 5 by displacement rho (lower = more contractive):")
                for k, dr, pr in scored[:5]:
                    print(f"      {k:>10s}  disp_rho={dr:.4f}  pert_rho={pr:.4f}")

    # 4. Brain Scanner
    print("\n" + "=" * 60)
    print("4. BRAIN SCANNER (7B)")
    print("=" * 60)

    sweep = load_json(str(RESULTS_DIR / "sweep_7B" / "sweep_results.json"))
    if sweep:
        results = sweep.get("results", {})
        baseline = sweep.get("baseline_score", 0)
        print(f"  Baseline: {baseline:.4f}")
        print(f"  Configs evaluated: {len(results)}")
        # Top 5
        scored = [(k, v["delta"]) for k, v in results.items() if "error" not in v]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            print(f"  Top 5 configs:")
            for k, d in scored[:5]:
                print(f"    {k:>10s}  delta={d:+.4f}")
            print(f"  Worst 3:")
            for k, d in scored[-3:]:
                print(f"    {k:>10s}  delta={d:+.4f}")

    # 5. Validation
    print("\n" + "=" * 60)
    print("5. SPECTRAL PREDICTION VALIDATION")
    print("=" * 60)

    val = load_json(str(RESULTS_DIR / "validation_7B" / "validation_results.json"))
    if val:
        print(f"  Configs compared: {val.get('num_configs', 0)}")
        corrs = val.get("correlations", {})
        for metric, vals in corrs.items():
            print(f"    {metric:40s} r={vals['spearman_r']:+.4f} (p={vals['spearman_p']:.4f})")
    else:
        print("  NOT YET AVAILABLE")

    # 6. Multi-pass
    print("\n" + "=" * 60)
    print("6. MULTI-PASS TEST (7B)")
    print("=" * 60)

    mp = load_json(str(RESULTS_DIR / "multi_pass_Qwen2-7B-Instruct" / "multi_pass_results.json"))
    if mp:
        for n in sorted(mp.keys(), key=int):
            r = mp[n]
            print(f"  {r['n_passes']}x pass: score={r['score']:.4f} ({r['total_layers']} layers)")
    else:
        print("  NOT YET AVAILABLE")

    # 7. Junction Fine-tuning
    print("\n" + "=" * 60)
    print("7. JUNCTION FINE-TUNING (7B)")
    print("=" * 60)

    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and "junction_ft" in d.name:
            jft = load_json(str(d / "junction_ft_results.json"))
            if jft:
                print(f"  Pre-finetune:  {jft['pre_score']:.4f}")
                print(f"  Post-finetune: {jft['post_score']:.4f}")
                print(f"  Delta:         {jft['delta']:+.4f}")
                print(f"  Trainable params: {jft['trainable_params']:,} ({100*jft['trainable_params']/jft['total_params']:.4f}%)")

    # 8. DeepPass Analysis
    print("\n" + "=" * 60)
    print("8. DEEPPASS UNIFIED ANALYSIS")
    print("=" * 60)

    for name in ["deeppass_7B", "deeppass_72B"]:
        dp = load_json(str(RESULTS_DIR / name / "deeppass_full_results.json"))
        if dp:
            print(f"\n  {name}:")
            print(f"    Baseline math: {dp.get('baseline_math_score', 'N/A')}")
            print(f"    Prediction hit rate: {dp.get('prediction_hit_rate', 'N/A')}")
            p2 = dp.get("phase2_results", {})
            if p2:
                for k, v in sorted(p2.items(), key=lambda x: x[1].get("math_delta", 0), reverse=True)[:5]:
                    print(f"    Config {k}: math_delta={v['math_delta']:+.4f} deeppass_score={v['deeppass_score']:.4f}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    compile_results = compile_all()
