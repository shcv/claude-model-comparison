#!/usr/bin/env python3
"""Generate dataset overview statistics from session and task data.

Usage:
    python scripts/analyze_dataset.py --data-dir comparisons/opus-4.5-vs-4.6/data \
        --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import load_canonical_tasks


def compute_cleaning_stats(data_dir: Path, model: str) -> dict:
    """Compute data cleaning exclusion and flag counts from canonical tasks."""
    all_tasks = load_canonical_tasks(data_dir, model, include_excluded=True)
    total = len(all_tasks)
    included = sum(1 for t in all_tasks if not t.get('exclude_reason'))
    excluded = total - included

    reasons = Counter(t['exclude_reason'] for t in all_tasks if t.get('exclude_reason'))
    flags = Counter()
    for t in all_tasks:
        if not t.get('exclude_reason'):
            for f in t.get('flags', []):
                flags[f] += 1

    return {
        'total_extracted': total,
        'included': included,
        'excluded': excluded,
        'exclusion_rate': round(excluded / total, 4) if total else 0,
        'exclusion_reasons': dict(reasons.most_common()),
        'flag_counts': dict(flags.most_common()),
    }


def analyze_model(data_dir: Path, analysis_dir: Path, model: str) -> dict:
    """Compute overview statistics for a single model."""
    sessions_path = data_dir / f"sessions-{model}.json"
    classified_path = data_dir / f"tasks-classified-{model}.json"
    token_analysis_path = analysis_dir / "token-analysis.json"

    with open(sessions_path) as f:
        sessions = json.load(f)
    with open(classified_path) as f:
        tasks = json.load(f)
    with open(token_analysis_path) as f:
        token_analysis = json.load(f)

    ta = token_analysis[model]["overall"]

    # Date range
    dates = sorted(s["start_time"][:10] for s in sessions if s.get("start_time"))

    # Session durations (filter multi-day outliers >24h)
    durations = [
        s["duration_minutes"]
        for s in sessions
        if s.get("duration_minutes") and s["duration_minutes"] < 1440
    ]

    # Aggregate tool usage from sessions
    tool_totals = Counter()
    for s in sessions:
        for tool, count in s.get("tools_used", {}).items():
            tool_totals[tool] += count

    # Classification breakdowns
    types = Counter(t["classification"]["type"] for t in tasks)
    complexity_order = ["trivial", "simple", "moderate", "complex", "major"]
    complexities = Counter(t["classification"]["complexity"] for t in tasks)

    # Code output
    total_files = sum(t.get("total_files_touched", 0) for t in tasks)
    total_lines_add = sum(t.get("total_lines_added", 0) for t in tasks)
    total_lines_rm = sum(t.get("total_lines_removed", 0) for t in tasks)

    # Task durations
    task_durations = sorted(
        t["duration_seconds"]
        for t in tasks
        if t.get("duration_seconds") and t["duration_seconds"] > 0
    )

    # Projects and messages
    projects = sorted(set(s.get("project_path", "") for s in sessions))
    total_user = sum(s.get("user_message_count", 0) for s in sessions)
    total_asst = sum(s.get("assistant_message_count", 0) for s in sessions)
    total_tools = sum(s.get("tool_call_count", 0) for s in sessions)

    return {
        "sessions": len(sessions),
        "tasks": len(tasks),
        "tasks_per_session": round(len(tasks) / len(sessions), 1),
        "projects": len(projects),
        "project_names": projects,
        "date_range": {
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
        },
        "total_user_messages": total_user,
        "total_assistant_messages": total_asst,
        "total_tool_calls": total_tools,
        "total_session_hours": round(sum(durations) / 60, 1),
        "median_session_minutes": (
            round(durations[len(durations) // 2], 1) if durations else 0
        ),
        "total_files_touched": total_files,
        "total_lines_added": total_lines_add,
        "total_lines_removed": total_lines_rm,
        "avg_task_duration_sec": (
            round(sum(task_durations) / len(task_durations), 1)
            if task_durations
            else 0
        ),
        "median_task_duration_sec": (
            round(task_durations[len(task_durations) // 2], 1)
            if task_durations
            else 0
        ),
        "total_cost_usd": ta["total_cost_usd"],
        "total_input_tokens": ta["total_input_tokens"],
        "total_output_tokens": ta["total_output_tokens"],
        "cache_read_tokens": ta["cache_read_tokens"],
        "cache_write_tokens": ta["cache_write_tokens"],
        "estimated_thinking_tokens": ta["estimated_thinking_tokens"],
        "estimated_text_tokens": ta["estimated_text_tokens"],
        "thinking_ratio": ta["thinking_ratio"],
        "avg_requests_per_task": ta["avg_requests_per_task"],
        "task_types": dict(types.most_common()),
        "complexity_distribution": {
            k: complexities[k] for k in complexity_order if k in complexities
        },
        "tool_usage": dict(tool_totals.most_common(15)),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate dataset overview stats")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument(
        "--analysis-dir", required=True, help="Path to analysis directory"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = Path(args.analysis_dir)

    # Discover models from session files
    models = sorted(
        p.stem.replace("sessions-", "")
        for p in data_dir.glob("sessions-*.json")
    )

    result = {}
    for model in models:
        print(f"Analyzing {model}...")
        result[model] = analyze_model(data_dir, analysis_dir, model)
        result[model]['data_cleaning'] = compute_cleaning_stats(data_dir, model)

    # Combined stats
    all_projects = set()
    for m in models:
        all_projects.update(result[m]["project_names"])

    combined = {"projects": len(all_projects)}
    sum_fields = [
        "sessions", "tasks", "total_cost_usd", "total_user_messages",
        "total_assistant_messages", "total_tool_calls", "total_input_tokens",
        "total_output_tokens", "cache_read_tokens", "cache_write_tokens",
        "total_files_touched", "total_lines_added", "total_lines_removed",
    ]
    for field in sum_fields:
        combined[field] = sum(result[m][field] for m in models)
    combined["total_cost_usd"] = round(combined["total_cost_usd"], 2)

    # Date range across all models
    starts = [result[m]["date_range"]["start"] for m in models if result[m]["date_range"]["start"]]
    ends = [result[m]["date_range"]["end"] for m in models if result[m]["date_range"]["end"]]
    combined["date_range"] = {
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
    }

    # Combined cleaning stats
    combined_cleaning = {
        'total_extracted': sum(result[m]['data_cleaning']['total_extracted'] for m in models),
        'included': sum(result[m]['data_cleaning']['included'] for m in models),
        'excluded': sum(result[m]['data_cleaning']['excluded'] for m in models),
    }
    total_ext = combined_cleaning['total_extracted']
    combined_cleaning['exclusion_rate'] = round(combined_cleaning['excluded'] / total_ext, 4) if total_ext else 0
    # Merge reason/flag counts
    all_reasons = Counter()
    all_flags = Counter()
    for m in models:
        for r, c in result[m]['data_cleaning']['exclusion_reasons'].items():
            all_reasons[r] += c
        for f, c in result[m]['data_cleaning']['flag_counts'].items():
            all_flags[f] += c
    combined_cleaning['exclusion_reasons'] = dict(all_reasons.most_common())
    combined_cleaning['flag_counts'] = dict(all_flags.most_common())
    combined['data_cleaning'] = combined_cleaning

    result["combined"] = combined

    out_path = analysis_dir / "dataset-overview.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"  Sessions: {combined['sessions']}")
    print(f"  Tasks: {combined['tasks']}")
    print(f"  Projects: {combined['projects']}")
    print(f"  Cost: ${combined['total_cost_usd']:,.2f}")


if __name__ == "__main__":
    main()
