#!/usr/bin/env python3
"""
Session-Level Analysis: Warm-up Effects, Effort Distribution, Session Length

Analyzes how task quality and behavior vary within and across sessions:
- Warm-up: Do later tasks in a session perform better than early ones?
- Effort distribution: What fraction of tool calls are research vs implementation?
- Session length: Do longer sessions produce different quality patterns?

Input:  data/tasks-classified-{model}.json, analysis/llm-analysis-{model}.json
Output: analysis/session-analysis.json + printed summary tables
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


MODELS = ['opus-4-5', 'opus-4-6']

RESEARCH_TOOLS = {'Read', 'Grep', 'Glob', 'Task', 'WebSearch', 'WebFetch'}
IMPLEMENTATION_TOOLS = {'Edit', 'Write', 'NotebookEdit'}


def load_data(data_dir: Path, analysis_dir: Path, model: str):
    """Load tasks-classified and llm-analysis for a model."""
    tasks_path = data_dir / f'tasks-classified-{model}.json'
    llm_path = analysis_dir / f'llm-analysis-{model}.json'

    with open(tasks_path) as f:
        tasks = json.load(f)

    with open(llm_path) as f:
        llm_analyses = json.load(f)

    # Index LLM analysis by task_id for fast lookup
    llm_by_id = {a['task_id']: a for a in llm_analyses}

    return tasks, llm_by_id


def classify_tool(name: str) -> str:
    """Classify a tool call as research, implementation, or other."""
    if name in RESEARCH_TOOLS:
        return 'research'
    elif name in IMPLEMENTATION_TOOLS:
        return 'implementation'
    else:
        return 'other'


def get_tool_names(task: dict) -> list[str]:
    """Extract ordered list of tool names from a task's tool_calls."""
    tool_calls = task.get('tool_calls', [])
    names = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            name = tc.get('name', '')
            if name:
                names.append(name)
    return names


def safe_mean(values: list) -> float:
    """Calculate mean, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def safe_pct(n: int, total: int) -> float:
    """Calculate percentage, returning 0.0 for zero denominator."""
    return 100.0 * n / total if total > 0 else 0.0


# ─── Analysis 1: Warm-up Effect ───────────────────────────────────────────────

def analyze_warmup(tasks: list, llm_by_id: dict, model: str) -> dict:
    """For sessions with 5+ tasks, compare first-3 vs later task metrics."""

    # Group tasks by session, sort by start_time
    sessions = defaultdict(list)
    for task in tasks:
        sid = task.get('session_id')
        if sid:
            sessions[sid].append(task)

    for sid in sessions:
        sessions[sid].sort(key=lambda t: t.get('start_time', ''))

    # Filter to sessions with 5+ tasks
    qualifying = {sid: tlist for sid, tlist in sessions.items() if len(tlist) >= 5}

    early_scores = []
    later_scores = []
    early_satisfaction = []
    later_satisfaction = []
    early_tools_per_file = []
    later_tools_per_file = []
    early_completion = []
    later_completion = []

    for sid, tlist in qualifying.items():
        early = tlist[:3]
        later = tlist[3:]

        for group, scores, satis, tpf, comp in [
            (early, early_scores, early_satisfaction, early_tools_per_file, early_completion),
            (later, later_scores, later_satisfaction, later_tools_per_file, later_completion),
        ]:
            for task in group:
                tid = task['task_id']
                llm = llm_by_id.get(tid)
                if not llm:
                    continue

                alignment = llm.get('alignment_score')
                if alignment is not None:
                    scores.append(alignment)

                sentiment = llm.get('normalized_user_sentiment', '')
                satis.append(1 if sentiment == 'satisfied' else 0)

                tpf_val = llm.get('tools_per_file', 0.0)
                if tpf_val > 0:
                    tpf.append(tpf_val)

                tc = llm.get('task_completion', '')
                comp.append(1 if tc == 'complete' else 0)

    return {
        'model': model,
        'qualifying_sessions': len(qualifying),
        'early_tasks': len(early_scores),
        'later_tasks': len(later_scores),
        'early': {
            'avg_alignment_score': round(safe_mean(early_scores), 2),
            'satisfaction_rate': round(safe_pct(sum(early_satisfaction), len(early_satisfaction)), 1),
            'avg_tools_per_file': round(safe_mean(early_tools_per_file), 2),
            'completion_rate': round(safe_pct(sum(early_completion), len(early_completion)), 1),
        },
        'later': {
            'avg_alignment_score': round(safe_mean(later_scores), 2),
            'satisfaction_rate': round(safe_pct(sum(later_satisfaction), len(later_satisfaction)), 1),
            'avg_tools_per_file': round(safe_mean(later_tools_per_file), 2),
            'completion_rate': round(safe_pct(sum(later_completion), len(later_completion)), 1),
        },
    }


# ─── Analysis 2: Effort Distribution ──────────────────────────────────────────

def analyze_effort_distribution(tasks: list, llm_by_id: dict, model: str) -> dict:
    """Analyze research vs implementation tool ratio per task."""

    task_ratios = []
    research_frontload_scores = []

    for task in tasks:
        tool_names = get_tool_names(task)
        if not tool_names:
            continue

        total = len(tool_names)
        research_count = sum(1 for t in tool_names if classify_tool(t) == 'research')
        impl_count = sum(1 for t in tool_names if classify_tool(t) == 'implementation')

        read_ratio = research_count / total if total > 0 else 0.0
        impl_ratio = impl_count / total if total > 0 else 0.0

        task_ratios.append({
            'task_id': task['task_id'],
            'total_tools': total,
            'research_count': research_count,
            'impl_count': impl_count,
            'read_ratio': round(read_ratio, 3),
            'impl_ratio': round(impl_ratio, 3),
        })

        # Frontloading analysis: is research concentrated in the first half?
        if total >= 4:
            midpoint = total // 2
            first_half = tool_names[:midpoint]
            second_half = tool_names[midpoint:]

            first_research = sum(1 for t in first_half if classify_tool(t) == 'research')
            second_research = sum(1 for t in second_half if classify_tool(t) == 'research')

            first_research_pct = first_research / len(first_half) if first_half else 0
            second_research_pct = second_research / len(second_half) if second_half else 0

            # Frontload score: positive means research is front-loaded
            frontload_score = first_research_pct - second_research_pct
            research_frontload_scores.append(frontload_score)

    all_ratios = [r['read_ratio'] for r in task_ratios]
    all_impl = [r['impl_ratio'] for r in task_ratios]

    return {
        'model': model,
        'tasks_analyzed': len(task_ratios),
        'avg_research_ratio': round(safe_mean(all_ratios), 3),
        'avg_impl_ratio': round(safe_mean(all_impl), 3),
        'tasks_with_frontload_data': len(research_frontload_scores),
        'avg_frontload_score': round(safe_mean(research_frontload_scores), 3),
        'frontload_positive_pct': round(
            safe_pct(
                sum(1 for s in research_frontload_scores if s > 0),
                len(research_frontload_scores)
            ), 1
        ),
    }


# ─── Analysis 3: Session Length Effects ────────────────────────────────────────

def analyze_session_length(tasks: list, llm_by_id: dict, model: str) -> dict:
    """Group sessions by length and compare quality metrics."""

    # Group tasks by session
    sessions = defaultdict(list)
    for task in tasks:
        sid = task.get('session_id')
        if sid:
            sessions[sid].append(task)

    buckets = {
        'short (1-3)': [],
        'medium (4-8)': [],
        'long (9+)': [],
    }

    for sid, tlist in sessions.items():
        n = len(tlist)
        if n <= 3:
            bucket = 'short (1-3)'
        elif n <= 8:
            bucket = 'medium (4-8)'
        else:
            bucket = 'long (9+)'
        buckets[bucket].append((sid, tlist))

    results = {}
    for bucket_name, session_list in buckets.items():
        scores = []
        satisfaction = []
        completion = []
        tools_per_file = []
        task_count = 0

        for sid, tlist in session_list:
            for task in tlist:
                task_count += 1
                tid = task['task_id']
                llm = llm_by_id.get(tid)
                if not llm:
                    continue

                alignment = llm.get('alignment_score')
                if alignment is not None:
                    scores.append(alignment)

                sentiment = llm.get('normalized_user_sentiment', '')
                satisfaction.append(1 if sentiment == 'satisfied' else 0)

                tc = llm.get('task_completion', '')
                completion.append(1 if tc == 'complete' else 0)

                tpf = llm.get('tools_per_file', 0.0)
                if tpf > 0:
                    tools_per_file.append(tpf)

        results[bucket_name] = {
            'sessions': len(session_list),
            'tasks': task_count,
            'avg_alignment_score': round(safe_mean(scores), 2),
            'satisfaction_rate': round(safe_pct(sum(satisfaction), len(satisfaction)), 1),
            'completion_rate': round(safe_pct(sum(completion), len(completion)), 1),
            'avg_tools_per_file': round(safe_mean(tools_per_file), 2),
        }

    return {
        'model': model,
        'buckets': results,
    }


# ─── Output ───────────────────────────────────────────────────────────────────

def print_warmup_table(warmup_results: list):
    """Print warm-up analysis comparison."""
    print("\n" + "=" * 75)
    print("WARM-UP ANALYSIS: First 3 Tasks vs Later Tasks (sessions with 5+ tasks)")
    print("=" * 75)

    for res in warmup_results:
        model = res['model']
        print(f"\n  {model}: {res['qualifying_sessions']} qualifying sessions "
              f"({res['early_tasks']} early, {res['later_tasks']} later tasks)")

    print(f"\n{'Metric':<30} ", end='')
    for res in warmup_results:
        print(f"{'Early':>10} {'Later':>10} {'Delta':>8}  ", end='')
    print()
    print("-" * 75)

    metrics = [
        ('Alignment score', 'avg_alignment_score'),
        ('Satisfaction %', 'satisfaction_rate'),
        ('Completion %', 'completion_rate'),
        ('Tools per file', 'avg_tools_per_file'),
    ]

    for label, key in metrics:
        print(f"{label:<30} ", end='')
        for res in warmup_results:
            early_val = res['early'][key]
            later_val = res['later'][key]
            delta = later_val - early_val
            sign = '+' if delta >= 0 else ''
            print(f"{early_val:>10.1f} {later_val:>10.1f} {sign}{delta:>7.1f}  ", end='')
        print()


def print_effort_table(effort_results: list):
    """Print effort distribution comparison."""
    print("\n" + "=" * 75)
    print("EFFORT DISTRIBUTION: Research vs Implementation Tools")
    print("=" * 75)
    print(f"  Research tools: {', '.join(sorted(RESEARCH_TOOLS))}")
    print(f"  Implementation tools: {', '.join(sorted(IMPLEMENTATION_TOOLS))}")

    print(f"\n{'Metric':<40} ", end='')
    for res in effort_results:
        print(f"{res['model']:>15} ", end='')
    print()
    print("-" * 75)

    rows = [
        ('Tasks analyzed', 'tasks_analyzed'),
        ('Avg research ratio', 'avg_research_ratio'),
        ('Avg implementation ratio', 'avg_impl_ratio'),
        ('Tasks w/ frontload data (4+ tools)', 'tasks_with_frontload_data'),
        ('Avg frontload score (-1 to +1)', 'avg_frontload_score'),
        ('% tasks research front-loaded', 'frontload_positive_pct'),
    ]

    for label, key in rows:
        print(f"{label:<40} ", end='')
        for res in effort_results:
            val = res[key]
            if isinstance(val, float):
                print(f"{val:>15.3f} ", end='')
            else:
                print(f"{val:>15} ", end='')
        print()


def print_session_length_table(length_results: list):
    """Print session length effects comparison."""
    print("\n" + "=" * 75)
    print("SESSION LENGTH EFFECTS: Quality by Session Size")
    print("=" * 75)

    bucket_names = ['short (1-3)', 'medium (4-8)', 'long (9+)']

    for res in length_results:
        model = res['model']
        print(f"\n  {model}:")
        print(f"  {'Bucket':<16} {'Sessions':>8} {'Tasks':>8} {'Align':>8} "
              f"{'Satis%':>8} {'Compl%':>8} {'T/File':>8}")
        print("  " + "-" * 68)

        for bucket in bucket_names:
            b = res['buckets'].get(bucket, {})
            print(f"  {bucket:<16} {b.get('sessions', 0):>8} {b.get('tasks', 0):>8} "
                  f"{b.get('avg_alignment_score', 0):>8.2f} "
                  f"{b.get('satisfaction_rate', 0):>8.1f} "
                  f"{b.get('completion_rate', 0):>8.1f} "
                  f"{b.get('avg_tools_per_file', 0):>8.2f}")


def main():
    parser = argparse.ArgumentParser(description='Session-level analysis')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory (default: data)')
    parser.add_argument('--analysis-dir', type=Path, default=Path('analysis'),
                        help='Analysis directory (default: analysis)')
    args = parser.parse_args()

    warmup_results = []
    effort_results = []
    length_results = []

    for model in MODELS:
        tasks_path = args.data_dir / f'tasks-classified-{model}.json'
        llm_path = args.analysis_dir / f'llm-analysis-{model}.json'

        if not tasks_path.exists() or not llm_path.exists():
            print(f"Skipping {model}: missing data files")
            continue

        print(f"Analyzing {model}...")
        tasks, llm_by_id = load_data(args.data_dir, args.analysis_dir, model)

        warmup_results.append(analyze_warmup(tasks, llm_by_id, model))
        effort_results.append(analyze_effort_distribution(tasks, llm_by_id, model))
        length_results.append(analyze_session_length(tasks, llm_by_id, model))

    # Print tables
    print_warmup_table(warmup_results)
    print_effort_table(effort_results)
    print_session_length_table(length_results)

    # Save JSON output
    output_path = args.analysis_dir / 'session-analysis.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'warmup': warmup_results,
        'effort_distribution': effort_results,
        'session_length': length_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
