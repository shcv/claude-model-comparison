#!/usr/bin/env python3
"""Compaction Analysis: Measure context window compaction rates and outcome impact.

Parses compaction events (compact_boundary system messages) from session JSONL
files and correlates with task-level outcomes. For sessions with compaction,
splits tasks into pre/post groups based on the first compaction timestamp.
A control group (non-compacting sessions) uses a synthetic split at the median
compaction position to measure baseline position effects.

Output (to analysis/):
  - compaction-analysis.json: per-model compaction rates and outcome correlations

Usage:
    python scripts/analyze_compaction.py \
      --data-dir comparisons/opus-4.5-vs-4.6/data \
      --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts: str) -> datetime | None:
    """Parse ISO 8601 timestamp."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


def extract_compaction_events(file_path: Path) -> list[dict]:
    """Extract compaction events from a session JSONL file.

    Returns list of dicts with keys: timestamp, trigger, pre_tokens, msg_index, total_msgs.
    """
    events = []
    msg_index = 0
    total_msgs = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        total_msgs = len(lines)

        for i, line in enumerate(lines):
            if 'compact_boundary' not in line:
                continue
            msg = json.loads(line)
            if msg.get('type') == 'system' and msg.get('subtype') == 'compact_boundary':
                meta = msg.get('compactMetadata', {})
                events.append({
                    'timestamp': msg.get('timestamp', ''),
                    'trigger': meta.get('trigger', 'unknown'),
                    'pre_tokens': meta.get('preTokens', 0),
                    'msg_index': i,
                    'total_msgs': total_msgs,
                })
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")

    return events


def compute_outcome_metrics(tasks: list[dict]) -> dict:
    """Compute aggregate outcome metrics for a group of tasks."""
    if not tasks:
        return {}

    n = len(tasks)
    alignment_scores = [t['alignment_score'] for t in tasks if t.get('alignment_score') is not None]
    sentiments = [t['normalized_user_sentiment'] for t in tasks if t.get('normalized_user_sentiment')]
    completions = [t['task_completion'] for t in tasks if t.get('task_completion')]

    # Edit metrics (may not exist for all tasks)
    edit_tasks = [t for t in tasks if 'user_corrections' in t]
    total_edits = sum(t.get('edit_count', 0) for t in edit_tasks)
    total_uc = sum(t.get('user_corrections', 0) for t in edit_tasks)

    result = {'n': n}

    if alignment_scores:
        result['avg_alignment'] = round(statistics.mean(alignment_scores), 3)
        result['median_alignment'] = round(statistics.median(alignment_scores), 1)

    if sentiments:
        sat_count = sum(1 for s in sentiments if s == 'satisfied')
        dissat_count = sum(1 for s in sentiments if s == 'dissatisfied')
        result['satisfaction_rate'] = round(sat_count / len(sentiments), 4)
        result['dissatisfaction_rate'] = round(dissat_count / len(sentiments), 4)

    if completions:
        complete_count = sum(1 for c in completions if c == 'complete')
        result['completion_rate'] = round(complete_count / len(completions), 4)

    if total_edits > 0:
        result['user_correction_rate'] = round(total_uc / total_edits, 4)
        result['total_edits'] = total_edits
        result['total_user_corrections'] = total_uc

    return result


def compute_delta(pre: dict, post: dict) -> dict:
    """Compute post - pre deltas for numeric metrics."""
    delta = {}
    for key in ['avg_alignment', 'satisfaction_rate', 'dissatisfaction_rate',
                 'completion_rate', 'user_correction_rate']:
        if key in pre and key in post:
            delta[key] = round(post[key] - pre[key], 4)
    return delta


def analyze_model_compaction(sessions_file: Path, model: str, data_dir: Path,
                             analysis_dir: Path) -> dict:
    """Analyze compaction for a single model."""
    with open(sessions_file) as f:
        sessions = json.load(f)

    # Filter meta sessions
    sessions = [s for s in sessions if not s.get('is_meta', False)]

    # Load classified tasks + LLM analysis + edit metrics
    classified_path = data_dir / f'tasks-classified-{model}.json'
    llm_path = analysis_dir / f'llm-analysis-{model}.json'
    edit_metrics_path = analysis_dir / f'edit-metrics-{model}.json'

    task_data = {}  # task_id -> merged task info

    if classified_path.exists():
        with open(classified_path) as f:
            for t in json.load(f):
                task_data[t['task_id']] = {
                    'task_id': t['task_id'],
                    'session_id': t['session_id'],
                    'start_time': t.get('start_time', ''),
                    'duration_seconds': t.get('duration_seconds', 0),
                }

    if llm_path.exists():
        with open(llm_path) as f:
            for t in json.load(f):
                tid = t['task_id']
                if tid in task_data:
                    task_data[tid]['alignment_score'] = t.get('alignment_score')
                    task_data[tid]['normalized_user_sentiment'] = t.get('normalized_user_sentiment')
                    task_data[tid]['task_completion'] = t.get('task_completion')

    if edit_metrics_path.exists():
        with open(edit_metrics_path) as f:
            for m in json.load(f):
                tid = m['task_id']
                if tid in task_data:
                    task_data[tid]['edit_count'] = m.get('edit_count', 0)
                    task_data[tid]['write_count'] = m.get('write_count', 0)
                    task_data[tid]['user_corrections'] = m.get('user_corrections', 0)

    # Parse compaction events for each session
    compacting_sessions = []
    non_compacting_sessions = []
    all_events = []

    for s in sessions:
        fp = Path(s['file_path'])
        if not fp.exists():
            continue

        events = extract_compaction_events(fp)
        sid = s['session_id']

        if events:
            compacting_sessions.append({
                'session_id': sid,
                'events': events,
                'duration_minutes': s.get('duration_minutes', 0),
            })
            all_events.extend(events)
        else:
            non_compacting_sessions.append({
                'session_id': sid,
                'duration_minutes': s.get('duration_minutes', 0),
            })

    total_sessions = len(sessions)
    n_compacting = len(compacting_sessions)
    n_events = len(all_events)

    # Basic stats
    triggers = defaultdict(int)
    pre_tokens_list = []
    positions = []
    for e in all_events:
        triggers[e['trigger']] += 1
        if e['pre_tokens'] > 0:
            pre_tokens_list.append(e['pre_tokens'])
        if e['total_msgs'] > 0:
            positions.append(e['msg_index'] / e['total_msgs'] * 100)

    # Events per hour across compacting sessions
    total_hours = sum(s['duration_minutes'] for s in compacting_sessions) / 60
    events_per_hour = round(n_events / total_hours, 2) if total_hours > 0 else 0

    result = {
        'total_sessions': total_sessions,
        'sessions_with_compaction': n_compacting,
        'compaction_rate_pct': round(n_compacting / total_sessions * 100, 1) if total_sessions > 0 else 0,
        'total_events': n_events,
        'trigger_breakdown': dict(triggers),
        'avg_events_per_compacting_session': round(n_events / n_compacting, 2) if n_compacting > 0 else 0,
        'avg_pre_tokens': round(statistics.mean(pre_tokens_list)) if pre_tokens_list else 0,
        'avg_position_pct': round(statistics.mean(positions), 1) if positions else 0,
        'events_per_hour': events_per_hour,
    }

    # --- Outcome correlation: pre/post compaction split ---
    if n_compacting > 0 and task_data:
        compacting_pre = []
        compacting_post = []

        for cs in compacting_sessions:
            sid = cs['session_id']
            # Use first compaction timestamp as split point
            first_event = cs['events'][0]
            split_ts = parse_timestamp(first_event['timestamp'])
            if not split_ts:
                continue

            session_tasks = [t for t in task_data.values() if t.get('session_id') == sid]
            for t in session_tasks:
                task_ts = parse_timestamp(t.get('start_time', ''))
                if not task_ts:
                    continue
                if task_ts < split_ts:
                    compacting_pre.append(t)
                else:
                    compacting_post.append(t)

        # Control group: non-compacting sessions, synthetic split at median position
        median_position_pct = statistics.median(positions) if positions else 50
        control_pre = []
        control_post = []

        for ncs in non_compacting_sessions:
            sid = ncs['session_id']
            session_tasks = sorted(
                [t for t in task_data.values() if t.get('session_id') == sid],
                key=lambda t: t.get('start_time', '')
            )
            if not session_tasks:
                continue
            split_idx = int(len(session_tasks) * median_position_pct / 100)
            split_idx = max(1, min(split_idx, len(session_tasks) - 1))
            control_pre.extend(session_tasks[:split_idx])
            control_post.extend(session_tasks[split_idx:])

        pre_metrics = compute_outcome_metrics(compacting_pre)
        post_metrics = compute_outcome_metrics(compacting_post)
        ctrl_pre_metrics = compute_outcome_metrics(control_pre)
        ctrl_post_metrics = compute_outcome_metrics(control_post)

        delta = compute_delta(pre_metrics, post_metrics)
        ctrl_delta = compute_delta(ctrl_pre_metrics, ctrl_post_metrics)

        # Compaction effect = compacting delta - control delta
        compaction_effect = {}
        for key in delta:
            if key in ctrl_delta:
                compaction_effect[key] = round(delta[key] - ctrl_delta[key], 4)

        result['outcome_correlation'] = {
            'compacting': {
                'pre': pre_metrics,
                'post': post_metrics,
                'delta': delta,
            },
            'control': {
                'pre': ctrl_pre_metrics,
                'post': ctrl_post_metrics,
                'delta': ctrl_delta,
            },
            'compaction_effect': compaction_effect,
            'median_split_position_pct': round(median_position_pct, 1),
        }

        print(f"    Compacting sessions: {n_compacting} ({len(compacting_pre)} pre, {len(compacting_post)} post tasks)")
        print(f"    Control sessions: {len(non_compacting_sessions)} ({len(control_pre)} pre, {len(control_post)} post tasks)")
    else:
        print(f"    Insufficient data for outcome correlation")

    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze compaction events and their impact')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Data directory with sessions-*.json files')
    parser.add_argument('--analysis-dir', type=Path, default=None,
                        help='Output directory for analysis results (default: data/../analysis)')
    args = parser.parse_args()

    data_dir = args.data_dir
    analysis_dir = (args.analysis_dir or data_dir.parent / 'analysis').resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    session_files = sorted(data_dir.glob('sessions-*.json'))
    if not session_files:
        print(f"Error: No sessions-*.json files found in {data_dir}")
        return

    models = {}
    for sf in session_files:
        model_name = sf.stem.replace('sessions-', '')
        models[model_name] = sf

    print(f"Found {len(models)} model(s): {', '.join(models.keys())}")

    results = {}
    for model, sf in models.items():
        print(f"\nAnalyzing compaction for {model}...")
        results[model] = analyze_model_compaction(sf, model, data_dir, analysis_dir)

        r = results[model]
        print(f"  Sessions: {r['sessions_with_compaction']}/{r['total_sessions']} ({r['compaction_rate_pct']}%)")
        print(f"  Events: {r['total_events']}, triggers: {r['trigger_breakdown']}")
        if r['avg_pre_tokens'] > 0:
            print(f"  Avg preTokens: {r['avg_pre_tokens']:,}")
        print(f"  Avg position: {r['avg_position_pct']}% through session")

    output_path = analysis_dir / 'compaction-analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nCompaction analysis -> {output_path}")


if __name__ == '__main__':
    main()
