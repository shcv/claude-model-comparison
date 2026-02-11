#!/usr/bin/env python3
"""Timing Analysis: Turn Duration, Explore Phase, Active-Time Cost

Four sub-analyses:
  a) Explore phase: time from task start to first edit/write, by complexity
  b) Request/turn timing: per-request streaming duration, turn_duration_ms stats
  c) Active-time cost: session cost / active hours (gaps capped at threshold)
  d) Session overlap: overlapping sessions, activity periods, true idle vs context switches

Output (to analysis/):
  - timing-analysis.json: per-model timing metrics with all four sub-analyses

Usage:
    python scripts/analyze_timing.py \
      --data-dir comparisons/opus-4.5-vs-4.6/data \
      --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models, load_canonical_tasks


def parse_ts(ts: str) -> datetime | None:
    """Parse ISO 8601 timestamp."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


def percentiles(values: list[float], ps=(10, 25, 50, 75, 90)) -> dict:
    """Compute percentiles for a list of floats. Returns empty dict if empty."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in ps:
        k = (p / 100) * (n - 1)
        f = int(k)
        c = f + 1 if f + 1 < n else f
        d = k - f
        result[f"p{p}"] = round(sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f]), 3)
    return result


def safe_stats(values: list[float]) -> dict:
    """Compute mean/median/stdev/count + percentiles, or empty dict."""
    if not values:
        return {"count": 0}
    result = {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }
    if len(values) >= 2:
        result["stdev"] = round(statistics.stdev(values), 3)
    result.update(percentiles(values))
    return result


# ---------------------------------------------------------------------------
# a) Explore phase analysis
# ---------------------------------------------------------------------------

WRITE_TOOLS = {'Edit', 'Write', 'NotebookEdit'}


def analyze_explore_phase(tasks: list[dict]) -> dict:
    """Measure time from task start to first edit/write tool call.

    For tasks with edits, the "explore phase" is the read-only period before
    the model starts modifying files.
    """
    explore_durations = []      # seconds
    explore_ratios = []         # fraction of task duration
    by_complexity = defaultdict(lambda: {"durations": [], "ratios": []})
    explore_only_count = 0      # tasks with reads but no writes
    tasks_with_edits = 0

    for task in tasks:
        if task.get('exclude_reason'):
            continue
        tool_calls = task.get('tool_calls', [])
        if not tool_calls:
            continue

        task_start = parse_ts(task.get('start_time', ''))
        if not task_start:
            continue

        duration = task.get('duration_seconds', 0)
        if duration <= 0:
            continue

        # Find first write/edit tool call with a timestamp
        first_write_ts = None
        has_write = False
        for tc in tool_calls:
            if tc.get('name') in WRITE_TOOLS:
                has_write = True
                ts = parse_ts(tc.get('timestamp', ''))
                if ts:
                    first_write_ts = ts
                    break

        if not has_write:
            # Read-only task (explore only)
            if any(tc.get('name') == 'Read' for tc in tool_calls):
                explore_only_count += 1
            continue

        tasks_with_edits += 1

        if first_write_ts is None:
            continue

        explore_sec = (first_write_ts - task_start).total_seconds()
        if explore_sec < 0:
            continue

        explore_ratio = explore_sec / duration if duration > 0 else 0

        explore_durations.append(explore_sec)
        explore_ratios.append(explore_ratio)

        complexity = task.get('complexity', 'unknown')
        by_complexity[complexity]["durations"].append(explore_sec)
        by_complexity[complexity]["ratios"].append(explore_ratio)

    # Build complexity breakdown
    complexity_breakdown = {}
    for level, data in sorted(by_complexity.items()):
        complexity_breakdown[level] = {
            "explore_duration_seconds": safe_stats(data["durations"]),
            "explore_ratio": safe_stats(data["ratios"]),
        }

    return {
        "tasks_with_edits": tasks_with_edits,
        "tasks_with_explore_data": len(explore_durations),
        "explore_only_tasks": explore_only_count,
        "explore_duration_seconds": safe_stats(explore_durations),
        "explore_ratio": safe_stats(explore_ratios),
        "by_complexity": complexity_breakdown,
    }


# ---------------------------------------------------------------------------
# b) Request/turn timing
# ---------------------------------------------------------------------------

def analyze_request_timing(tasks: list[dict]) -> dict:
    """Analyze per-request streaming duration and turn_duration_ms.

    Uses turn_duration_ms (from Claude Code's own timing) as primary signal
    where available, and request_timings as supplementary data.
    """
    turn_durations_ms = []          # from turn_duration_ms field
    request_durations_sec = []      # from request_timings timestamps
    first_request_durations = []    # first request per task
    by_complexity = defaultdict(lambda: {"turn_ms": [], "request_sec": []})
    tasks_with_turn_data = 0

    for task in tasks:
        if task.get('exclude_reason'):
            continue

        complexity = task.get('complexity', 'unknown')

        # turn_duration_ms: Claude Code's own measurement
        turn_ms = task.get('turn_duration_ms', 0)
        if turn_ms > 0:
            turn_durations_ms.append(turn_ms)
            tasks_with_turn_data += 1
            by_complexity[complexity]["turn_ms"].append(turn_ms)

        # request_timings: per-request streaming windows
        timings = task.get('request_timings', [])
        is_first = True
        for rt in timings:
            first_ts = parse_ts(rt.get('first_timestamp', ''))
            last_ts = parse_ts(rt.get('last_timestamp', ''))
            if first_ts and last_ts:
                dur = (last_ts - first_ts).total_seconds()
                if dur >= 0:
                    request_durations_sec.append(dur)
                    by_complexity[complexity]["request_sec"].append(dur)
                    if is_first:
                        first_request_durations.append(dur)
                        is_first = False

    # Model time ratio: turn_duration / task wall time
    model_time_ratios = []
    for task in tasks:
        if task.get('exclude_reason'):
            continue
        turn_ms = task.get('turn_duration_ms', 0)
        duration = task.get('duration_seconds', 0)
        if turn_ms > 0 and duration > 0:
            ratio = (turn_ms / 1000.0) / duration
            if 0 < ratio <= 1.0:
                model_time_ratios.append(ratio)

    complexity_breakdown = {}
    for level, data in sorted(by_complexity.items()):
        complexity_breakdown[level] = {
            "turn_duration_ms": safe_stats(data["turn_ms"]),
            "request_duration_seconds": safe_stats(data["request_sec"]),
        }

    return {
        "tasks_with_turn_duration": tasks_with_turn_data,
        "turn_duration_ms": safe_stats(turn_durations_ms),
        "request_streaming_seconds": safe_stats(request_durations_sec),
        "first_request_duration_seconds": safe_stats(first_request_durations),
        "model_time_ratio": safe_stats(model_time_ratios),
        "by_complexity": complexity_breakdown,
    }


# ---------------------------------------------------------------------------
# c) Active-time cost
# ---------------------------------------------------------------------------

def analyze_active_time_cost(tasks: list[dict], gap_thresholds_min=(2, 5, 10, 20, 30, 60)) -> dict:
    """Compute cost per active hour by grouping tasks into sessions.

    Active time = sum of task durations + inter-task gaps below threshold.
    This filters out long idle periods (lunch breaks, overnight) that inflate
    wall-clock session duration.
    """
    # Group tasks by session
    session_tasks = defaultdict(list)
    for task in tasks:
        if task.get('exclude_reason'):
            continue
        session_tasks[task['session_id']].append(task)

    results_by_threshold = {}

    for threshold_min in gap_thresholds_min:
        threshold_sec = threshold_min * 60
        total_active_seconds = 0
        total_cost = 0
        session_details = []

        for sid, stasks in session_tasks.items():
            stasks.sort(key=lambda t: t.get('start_time', ''))

            session_cost = sum(t.get('estimated_cost', 0) for t in stasks)
            total_cost += session_cost

            # Calculate active time
            active_sec = 0
            prev_end = None

            for task in stasks:
                start = parse_ts(task.get('start_time', ''))
                end = parse_ts(task.get('end_time', ''))
                dur = task.get('duration_seconds', 0)

                if dur > 0:
                    active_sec += dur

                # Add gap if within threshold
                if prev_end and start:
                    gap = (start - prev_end).total_seconds()
                    if 0 < gap <= threshold_sec:
                        active_sec += gap

                if end:
                    prev_end = end

            total_active_seconds += active_sec
            if active_sec > 0:
                session_details.append({
                    "session_id": sid,
                    "active_hours": round(active_sec / 3600, 4),
                    "cost": round(session_cost, 4),
                    "task_count": len(stasks),
                })

        active_hours = total_active_seconds / 3600
        cost_per_active_hour = (
            round(total_cost / active_hours, 4) if active_hours > 0 else 0
        )

        # Per-session cost rates
        session_rates = []
        for sd in session_details:
            if sd["active_hours"] > 0:
                session_rates.append(sd["cost"] / sd["active_hours"])

        results_by_threshold[f"{threshold_min}min"] = {
            "total_active_hours": round(active_hours, 4),
            "total_cost": round(total_cost, 4),
            "cost_per_active_hour": cost_per_active_hour,
            "session_count": len(session_details),
            "per_session_rate": safe_stats(session_rates),
        }

    # Gap classification: short (typing) vs break
    gap_durations = []
    for sid, stasks in session_tasks.items():
        stasks.sort(key=lambda t: t.get('start_time', ''))
        prev_end = None
        for task in stasks:
            start = parse_ts(task.get('start_time', ''))
            end = parse_ts(task.get('end_time', ''))
            if prev_end and start:
                gap = (start - prev_end).total_seconds()
                if gap > 0:
                    gap_durations.append(gap)
            if end:
                prev_end = end

    gap_classification = {
        "total_gaps": len(gap_durations),
        "all_gaps_seconds": safe_stats(gap_durations),
    }
    if gap_durations:
        short = [g for g in gap_durations if g <= 300]  # <=5min
        long = [g for g in gap_durations if g > 300]
        gap_classification["short_gaps_lte_5min"] = {
            "count": len(short),
            "stats": safe_stats(short),
        }
        gap_classification["long_gaps_gt_5min"] = {
            "count": len(long),
            "stats": safe_stats(long),
        }

    return {
        "by_threshold": results_by_threshold,
        "gap_classification": gap_classification,
    }


# ---------------------------------------------------------------------------
# d) Session overlap
# ---------------------------------------------------------------------------

def analyze_session_overlap(data_dir: Path, tasks: list[dict]) -> dict:
    """Detect overlapping sessions and compute activity periods.

    Loads session metadata (start/end times) and identifies concurrent sessions,
    then merges overlapping sessions into contiguous activity periods.
    """
    # Load all session metadata
    sessions = []
    for sf in sorted(data_dir.glob("sessions-*.json")):
        with open(sf) as f:
            sessions.extend(json.load(f))

    # Parse session time ranges
    session_ranges = []
    for s in sessions:
        start = parse_ts(s.get('start_time', ''))
        end = parse_ts(s.get('end_time', ''))
        if start and end and end > start:
            session_ranges.append({
                "session_id": s["session_id"],
                "start": start,
                "end": end,
                "duration_min": s.get("duration_minutes", 0),
            })

    if not session_ranges:
        return {"session_count": 0}

    session_ranges.sort(key=lambda s: s["start"])

    # Detect overlapping pairs
    overlap_count = 0
    for i in range(len(session_ranges)):
        for j in range(i + 1, len(session_ranges)):
            si, sj = session_ranges[i], session_ranges[j]
            if sj["start"] >= si["end"]:
                break  # no more overlaps with si (sorted by start)
            overlap_count += 1

    # Max concurrency: sweep line
    events = []
    for s in session_ranges:
        events.append((s["start"], 1))
        events.append((s["end"], -1))
    events.sort()
    max_concurrent = 0
    current = 0
    for _, delta in events:
        current += delta
        max_concurrent = max(max_concurrent, current)

    # Merge into activity periods (contiguous work blocks)
    activity_periods = []
    current_start = session_ranges[0]["start"]
    current_end = session_ranges[0]["end"]

    for s in session_ranges[1:]:
        if s["start"] <= current_end:
            # Overlapping or adjacent — extend
            current_end = max(current_end, s["end"])
        else:
            activity_periods.append((current_start, current_end))
            current_start = s["start"]
            current_end = s["end"]
    activity_periods.append((current_start, current_end))

    period_hours = [
        (end - start).total_seconds() / 3600
        for start, end in activity_periods
    ]

    # Classify inter-task gaps as "true idle" vs "context switch"
    # Build a set of all session intervals for overlap checking
    true_idle_gaps = []
    context_switch_gaps = []

    # Group tasks by session for gap analysis
    session_task_map = defaultdict(list)
    for task in tasks:
        if not task.get('exclude_reason'):
            session_task_map[task['session_id']].append(task)

    for sid, stasks in session_task_map.items():
        stasks.sort(key=lambda t: t.get('start_time', ''))
        prev_end = None
        for task in stasks:
            start = parse_ts(task.get('start_time', ''))
            end = parse_ts(task.get('end_time', ''))
            if prev_end and start:
                gap_sec = (start - prev_end).total_seconds()
                if gap_sec <= 0:
                    if end:
                        prev_end = end
                    continue

                # Check if another session was active during this gap
                gap_mid = prev_end + timedelta(seconds=gap_sec / 2)
                other_active = False
                for sr in session_ranges:
                    if sr["session_id"] == sid:
                        continue
                    if sr["start"] <= gap_mid <= sr["end"]:
                        other_active = True
                        break

                if other_active:
                    context_switch_gaps.append(gap_sec)
                else:
                    true_idle_gaps.append(gap_sec)

            if end:
                prev_end = end

    return {
        "session_count": len(session_ranges),
        "overlapping_pairs": overlap_count,
        "max_concurrency": max_concurrent,
        "activity_periods": {
            "count": len(activity_periods),
            "total_hours": round(sum(period_hours), 4),
            "period_hours": safe_stats(period_hours),
        },
        "gap_classification": {
            "true_idle": {
                "count": len(true_idle_gaps),
                "stats_seconds": safe_stats(true_idle_gaps),
            },
            "context_switch": {
                "count": len(context_switch_gaps),
                "stats_seconds": safe_stats(context_switch_gaps),
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Timing analysis')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Data directory with tasks-canonical-*.json')
    parser.add_argument('--analysis-dir', type=Path, default=None,
                        help='Output directory (default: data/../analysis)')
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    analysis_dir = (args.analysis_dir or data_dir.parent / 'analysis').resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(data_dir, prefix="tasks-canonical")
    if not models:
        print("No tasks-canonical-*.json files found")
        sys.exit(1)

    print(f"Timing analysis for {len(models)} model(s): {', '.join(models)}")

    # Session overlap is cross-model (uses all session files)
    all_tasks = []
    for model in models:
        all_tasks.extend(load_canonical_tasks(data_dir, model, include_excluded=True))
    session_overlap = analyze_session_overlap(data_dir, all_tasks)

    result = {
        "session_overlap": session_overlap,
    }

    for model in models:
        tasks = load_canonical_tasks(data_dir, model, include_excluded=True)
        included = [t for t in tasks if not t.get('exclude_reason')]
        print(f"\n  {model}: {len(included)} included tasks ({len(tasks)} total)")

        # Load annotated tasks if available (for complexity field)
        annotated_file = data_dir / f'tasks-annotated-{model}.json'
        if annotated_file.exists():
            with open(annotated_file) as f:
                annotated = json.load(f)
            # Build lookup by task_id
            complexity_map = {t['task_id']: t.get('complexity', 'unknown') for t in annotated}
            for task in included:
                task['complexity'] = complexity_map.get(task['task_id'], 'unknown')
        else:
            for task in included:
                task['complexity'] = 'unknown'

        explore = analyze_explore_phase(included)
        request_timing = analyze_request_timing(included)
        active_cost = analyze_active_time_cost(included)

        model_result = {
            "explore_phase": explore,
            "request_timing": request_timing,
            "active_time_cost": active_cost,
        }

        # Print summary
        ep = explore
        print(f"    Explore phase: {ep['tasks_with_explore_data']}/{ep['tasks_with_edits']} editing tasks with timing data")
        if ep['explore_ratio'].get('count', 0) > 0:
            print(f"      Median explore ratio: {ep['explore_ratio']['median']:.1%}")
            print(f"      Median explore duration: {ep['explore_duration_seconds']['median']:.1f}s")

        rt = request_timing
        print(f"    Turn duration: {rt['tasks_with_turn_duration']} tasks with data")
        if rt['turn_duration_ms'].get('count', 0) > 0:
            print(f"      Median turn: {rt['turn_duration_ms']['median'] / 1000:.1f}s")
        if rt['model_time_ratio'].get('count', 0) > 0:
            print(f"      Median model time ratio: {rt['model_time_ratio']['median']:.1%}")

        ac = active_cost.get('by_threshold', {}).get('5min', {})
        if ac:
            print(f"    Active-time cost (5min gap): ${ac['cost_per_active_hour']:.2f}/active-hour")
            print(f"      Total active: {ac['total_active_hours']:.1f}h, cost: ${ac['total_cost']:.2f}")

        result[model] = model_result

    # Session overlap summary
    so = session_overlap
    print(f"\n  Session overlap: {so.get('overlapping_pairs', 0)} overlapping pairs, "
          f"max concurrency: {so.get('max_concurrency', 0)}")
    ap = so.get('activity_periods', {})
    if ap.get('count'):
        print(f"    Activity periods: {ap['count']}, total: {ap['total_hours']:.1f}h")

    output_file = analysis_dir / "timing-analysis.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
