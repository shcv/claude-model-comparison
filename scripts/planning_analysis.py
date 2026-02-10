#!/usr/bin/env python3
"""Cross-tabulate planning mode usage by complexity and alignment scores.

For each model, identifies tasks that used EnterPlanMode (from canonical task
`used_planning` field) and cross-tabulates planning rates by complexity bin.
Joins with LLM analysis alignment scores to compare outcomes for planned vs
unplanned tasks.

Output (to analysis/):
  - planning-analysis.json: per-model planning rates and alignment correlations

Usage:
    python scripts/planning_analysis.py \
      --data-dir comparisons/opus-4.5-vs-4.6/data \
      --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models

COMPLEXITY_ORDER = ["trivial", "simple", "moderate", "complex", "major"]


def load_tasks(data_dir, model):
    """Load canonical tasks for a model."""
    # Prefer canonical tasks
    canonical_path = data_dir / f"tasks-canonical-{model}.json"
    if canonical_path.exists():
        with open(canonical_path) as f:
            return json.load(f)
    # Fall back to classified tasks
    path = data_dir / f"tasks-classified-{model}.json"
    with open(path) as f:
        return json.load(f)


def load_llm_analysis(analysis_dir, model):
    path = analysis_dir / f"llm-analysis-{model}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {item["task_id"]: item for item in data}


def has_planning(task):
    """Check if a task used planning mode.

    Uses canonical `used_planning` field if available, falls back to
    checking tool_calls for EnterPlanMode.
    """
    if 'used_planning' in task:
        return bool(task['used_planning'])
    # Fallback for classified tasks without used_planning
    for tc in task.get("tool_calls", []):
        if tc.get("name") == "EnterPlanMode":
            return True
    return False


def get_complexity(task):
    """Extract complexity from a canonical or classified task."""
    # Canonical tasks may not have classification dict; check both
    cl = task.get("classification", {})
    if isinstance(cl, dict):
        return cl.get("complexity", "unknown")
    return "unknown"


def analyze_model(data_dir, analysis_dir, model):
    """Analyze planning usage for a model. Returns JSON-serializable result dict."""
    tasks = load_tasks(data_dir, model)
    llm = load_llm_analysis(analysis_dir, model)

    # Filter meta tasks
    tasks = [t for t in tasks if not t.get('is_meta', False)]

    # For canonical tasks, load classified data for complexity info
    classified_by_id = {}
    classified_path = data_dir / f"tasks-classified-{model}.json"
    if classified_path.exists():
        with open(classified_path) as f:
            for t in json.load(f):
                classified_by_id[t['task_id']] = t

    # Build cross-tab: complexity -> {planned: count, unplanned: count}
    cross_tab = defaultdict(lambda: {"planned": 0, "unplanned": 0})
    # Alignment scores: complexity -> {planned: [scores], unplanned: [scores]}
    alignment = defaultdict(lambda: {"planned": [], "unplanned": []})

    for task in tasks:
        task_id = task['task_id']

        # Get complexity from classified data if available
        if task_id in classified_by_id:
            complexity = get_complexity(classified_by_id[task_id])
        else:
            complexity = get_complexity(task)

        planned = has_planning(task)
        key = "planned" if planned else "unplanned"
        cross_tab[complexity][key] += 1

        if task_id in llm and llm[task_id].get("alignment_score") is not None:
            alignment[complexity][key].append(llm[task_id]["alignment_score"])

    total_tasks = len(tasks)
    total_planned = sum(1 for t in tasks if has_planning(t))

    # Collapse complex+major if either has <10 planned tasks
    complex_planned = cross_tab.get("complex", {}).get("planned", 0)
    major_planned = cross_tab.get("major", {}).get("planned", 0)
    collapse = complex_planned < 10 or major_planned < 10

    # Build by_complexity dict
    by_complexity = {}
    for c in COMPLEXITY_ORDER:
        if collapse and c in ("complex", "major"):
            continue
        if c not in cross_tab:
            continue
        p = cross_tab[c]["planned"]
        u = cross_tab[c]["unplanned"]
        n = p + u
        ps = alignment[c]["planned"]
        us = alignment[c]["unplanned"]
        p_mean = sum(ps) / len(ps) if ps else None
        u_mean = sum(us) / len(us) if us else None
        delta = round(p_mean - u_mean, 4) if p_mean is not None and u_mean is not None else None
        by_complexity[c] = {
            "n": n,
            "planned": p,
            "unplanned": u,
            "planning_rate_pct": round(100 * p / n, 1) if n > 0 else 0,
            "alignment_planned": round(p_mean, 4) if p_mean is not None else None,
            "alignment_unplanned": round(u_mean, 4) if u_mean is not None else None,
            "alignment_delta": delta,
            "n_scored_planned": len(ps),
            "n_scored_unplanned": len(us),
        }

    # Add collapsed complex+ bin if needed
    if collapse and ("complex" in cross_tab or "major" in cross_tab):
        p = cross_tab.get("complex", {}).get("planned", 0) + cross_tab.get("major", {}).get("planned", 0)
        u = cross_tab.get("complex", {}).get("unplanned", 0) + cross_tab.get("major", {}).get("unplanned", 0)
        n = p + u
        ps = alignment.get("complex", {}).get("planned", []) + alignment.get("major", {}).get("planned", [])
        us = alignment.get("complex", {}).get("unplanned", []) + alignment.get("major", {}).get("unplanned", [])
        p_mean = sum(ps) / len(ps) if ps else None
        u_mean = sum(us) / len(us) if us else None
        delta = round(p_mean - u_mean, 4) if p_mean is not None and u_mean is not None else None
        if n > 0:
            by_complexity["complex+"] = {
                "n": n,
                "planned": p,
                "unplanned": u,
                "planning_rate_pct": round(100 * p / n, 1),
                "alignment_planned": round(p_mean, 4) if p_mean is not None else None,
                "alignment_unplanned": round(u_mean, 4) if u_mean is not None else None,
                "alignment_delta": delta,
                "n_scored_planned": len(ps),
                "n_scored_unplanned": len(us),
            }

    # Overall alignment
    all_planned = []
    all_unplanned = []
    for c in alignment:
        all_planned.extend(alignment[c]["planned"])
        all_unplanned.extend(alignment[c]["unplanned"])

    overall_ap = round(sum(all_planned) / len(all_planned), 4) if all_planned else None
    overall_au = round(sum(all_unplanned) / len(all_unplanned), 4) if all_unplanned else None
    overall_delta = round(overall_ap - overall_au, 4) if overall_ap is not None and overall_au is not None else None

    result = {
        "total_tasks": total_tasks,
        "total_planned": total_planned,
        "planning_rate_pct": round(100 * total_planned / total_tasks, 1) if total_tasks > 0 else 0,
        "by_complexity": by_complexity,
        "overall_alignment_planned": overall_ap,
        "overall_alignment_unplanned": overall_au,
        "overall_alignment_delta": overall_delta,
        "coverage": {
            "tasks_with_alignment": len(all_planned) + len(all_unplanned),
            "pct": round(100 * (len(all_planned) + len(all_unplanned)) / total_tasks, 1) if total_tasks > 0 else 0,
        },
    }

    # Print stdout summary
    print(f"\n{'='*70}")
    print(f"  {model}")
    print(f"{'='*70}")
    print(f"  Total tasks: {total_tasks}  |  Used planning: {total_planned} ({result['planning_rate_pct']:.1f}%)")

    print(f"\n  Planning Mode Usage by Complexity")
    print(f"  {'Complexity':<12} {'Planned':>8} {'Unplanned':>10} {'Total':>7} {'% Planned':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*7} {'-'*10}")
    for c in COMPLEXITY_ORDER + ["complex+"]:
        if c not in by_complexity:
            continue
        b = by_complexity[c]
        print(f"  {c:<12} {b['planned']:>8} {b['unplanned']:>10} {b['n']:>7} {b['planning_rate_pct']:>9.1f}%")

    tp = sum(v["planned"] for v in cross_tab.values())
    tu = sum(v["unplanned"] for v in cross_tab.values())
    tt = tp + tu
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*7} {'-'*10}")
    print(f"  {'TOTAL':<12} {tp:>8} {tu:>10} {tt:>7} {100*tp/tt:>9.1f}%")

    print(f"\n  Alignment Scores: Planned vs Unplanned by Complexity")
    print(f"  {'Complexity':<12} {'Planned':>18} {'Unplanned':>18} {'Delta':>8}")
    print(f"  {'':<12} {'mean (n)':>18} {'mean (n)':>18} {'':>8}")
    print(f"  {'-'*12} {'-'*18} {'-'*18} {'-'*8}")
    for c in COMPLEXITY_ORDER + ["complex+"]:
        if c not in by_complexity:
            continue
        b = by_complexity[c]
        p_str = f"{b['alignment_planned']:.2f} (n={b['n_scored_planned']})" if b['alignment_planned'] is not None else "-- (n=0)"
        u_str = f"{b['alignment_unplanned']:.2f} (n={b['n_scored_unplanned']})" if b['alignment_unplanned'] is not None else "-- (n=0)"
        d_str = f"{b['alignment_delta']:+.2f}" if b['alignment_delta'] is not None else "--"
        print(f"  {c:<12} {p_str:>18} {u_str:>18} {d_str:>8}")

    print(f"  {'-'*12} {'-'*18} {'-'*18} {'-'*8}")
    if overall_ap is not None and overall_au is not None:
        print(f"  {'OVERALL':<12} {f'{overall_ap:.2f} (n={len(all_planned)})':>18} {f'{overall_au:.2f} (n={len(all_unplanned)})':>18} {overall_delta:>+8.2f}")
    else:
        print(f"  {'OVERALL':<12} insufficient data")

    return result


def main():
    parser = argparse.ArgumentParser(description='Cross-tabulate planning mode usage by complexity')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Data directory with tasks-canonical-*.json files')
    parser.add_argument('--analysis-dir', type=Path, default=None,
                        help='Output directory for analysis results (default: data/../analysis)')
    args = parser.parse_args()

    data_dir = args.data_dir
    analysis_dir = (args.analysis_dir or data_dir.parent / 'analysis').resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover models from canonical or classified files
    models = discover_models(data_dir, prefix="tasks-canonical")
    if not models:
        models = discover_models(data_dir, prefix="tasks-classified")
    if not models:
        print(f"Error: No task files found in {data_dir}")
        return

    print(f"Found {len(models)} model(s): {', '.join(models)}")

    results = {}
    for model in models:
        results[model] = analyze_model(data_dir, analysis_dir, model)

    # Cross-model comparison summary
    if len(models) > 1:
        print(f"\n{'='*70}")
        print(f"  Cross-Model Planning Rate Comparison")
        print(f"{'='*70}")

        for model in models:
            tasks = load_tasks(data_dir, model)
            tasks = [t for t in tasks if not t.get('is_meta', False)]

            # Load classified for complexity
            classified_by_id = {}
            classified_path = data_dir / f"tasks-classified-{model}.json"
            if classified_path.exists():
                with open(classified_path) as f:
                    for t in json.load(f):
                        classified_by_id[t['task_id']] = t

            by_complexity = defaultdict(lambda: {"total": 0, "planned": 0})
            for t in tasks:
                tid = t['task_id']
                if tid in classified_by_id:
                    c = get_complexity(classified_by_id[tid])
                else:
                    c = get_complexity(t)
                by_complexity[c]["total"] += 1
                if has_planning(t):
                    by_complexity[c]["planned"] += 1

            print(f"\n  {model}:")
            for c in COMPLEXITY_ORDER:
                if c not in by_complexity:
                    continue
                d = by_complexity[c]
                pct = 100 * d["planned"] / d["total"] if d["total"] > 0 else 0
                bar = "#" * int(pct / 2)
                print(f"    {c:<12} {d['planned']:>3}/{d['total']:<4} ({pct:>5.1f}%) {bar}")

    # Write JSON output
    output_path = analysis_dir / 'planning-analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPlanning analysis -> {output_path}")


if __name__ == "__main__":
    main()
