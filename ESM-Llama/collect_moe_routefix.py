#!/usr/bin/env python3
"""Collect MoE route-then-fix experiment results into a comparison table.

Scans results/moe_routefix/<mode>/ for each mode's per-task outputs, extracts the
5 task scores (reusing scripts/compare_with_experts.py extractors) and the timing
breakdown (timing.json written by the per-sample eval drivers), and prints a table
plus writes comparison.csv.

Usage:
    python collect_moe_routefix.py [results_root] [mode1,mode2,...]
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_with_experts as C  # noqa: E402

TASK_ORDER = ['gsm8k_cot', 'ifeval', 'bigcode', 'multilingual', 'safety']
TASK_COLS = ['math', 'instr', 'code', 'multi', 'safety']
DEFAULT_MODES = ['base', 'oracle', 'fix_agg']
TIMING_KEYS = ['route_s', 'merge_s', 'restore_s', 'infer_s', 'calls', 'eval_total_s']


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_mode(mode_root: Path):
    """Return (task_scores, timing) for one mode root."""
    scores, timing = {}, None
    if not mode_root.exists():
        return scores, timing

    for p in mode_root.rglob('*.json'):
        data = _load(p)
        if not isinstance(data, dict):
            continue

        if p.name == 'timing.json':
            if timing is None:
                timing = {k: 0.0 for k in TIMING_KEYS}
            for k in TIMING_KEYS:
                timing[k] += data.get(k) or 0
            continue

        # bigcode driver output: {task: {pass@k}, ..., 'config': {...}}
        if 'humanevalplus' in data or 'mbppplus' in data:
            s = C.get_bigcode_score(data)
            if s is not None:
                scores['bigcode'] = s
        # safety driver output: {task: metrics}
        if 'wildguardtest' in data or 'harmbench' in data or 'xstest' in data:
            s = C.get_safety_score(data)
            if s is not None:
                scores['safety'] = s
        # lm_eval driver output: {task: metrics} (already unwrapped) -> wrap back
        if 'gsm8k_cot' in data or 'ifeval' in data or 'm_mmlu_fr' in data:
            wrapped = {'results': data}
            if 'gsm8k_cot' in data:
                scores['gsm8k_cot'] = C.get_gsm8k_score(wrapped)
            if 'ifeval' in data:
                scores['ifeval'] = C.get_ifeval_score(wrapped)
            if 'm_mmlu_fr' in data:
                scores['multilingual'] = C.get_multilingual_score(wrapped)

    return scores, timing


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results/moe_routefix'
    modes = sys.argv[2].split(',') if len(sys.argv) > 2 else DEFAULT_MODES

    experts = C.get_expert_scores(str(ROOT))

    hdr = (f"{'mode':<16} " + " ".join(f"{n:>8}" for n in TASK_COLS)
           + f"{'avg_abs':>9}{'avg_norm':>10}"
           + f"{'route':>8}{'merge':>8}{'infer':>9}")
    print(hdr)
    print('-' * len(hdr))

    rows = []
    for mode in modes:
        scores, timing = extract_mode(root / mode)
        abss, norms, cells = [], [], []
        for k in TASK_ORDER:
            v = scores.get(k)
            cells.append(f"{v*100:>8.2f}" if v is not None else f"{'-':>8}")
            if v is not None:
                abss.append(v * 100)
                e = experts.get(k)
                if e:
                    norms.append(v / e * 100)
        a_abs = sum(abss) / len(abss) if abss else float('nan')
        a_norm = sum(norms) / len(norms) if norms else float('nan')
        t = timing or {}
        print(f"{mode:<16} " + " ".join(cells) + f"{a_abs:>9.2f}{a_norm:>9.1f}%"
              + f"{t.get('route_s',0):>8.1f}{t.get('merge_s',0):>8.1f}{t.get('infer_s',0):>9.1f}")
        rows.append((mode, {k: scores.get(k) for k in TASK_ORDER}, a_abs, a_norm, t))

    out = root / 'comparison.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode'] + TASK_COLS + ['avg_abs', 'avg_norm'] + TIMING_KEYS)
        for mode, s, a_abs, a_norm, t in rows:
            w.writerow([mode] + [s[k] for k in TASK_ORDER] + [a_abs, a_norm]
                       + [t.get(k, '') for k in TIMING_KEYS])
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
