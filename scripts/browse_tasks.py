#!/usr/bin/env python3
"""
Task Browser for Model Comparison

Browse, filter, and compare tasks across models.
"""

import json
import re
from pathlib import Path
from typing import Optional
import argparse


def load_tasks(model: str, data_dir: Path = Path('data'), classified: bool = True) -> list[dict]:
    """Load task data for a model, preferring classified version."""
    if classified:
        tasks_file = data_dir / f'tasks-classified-{model}.json'
        if tasks_file.exists():
            with open(tasks_file) as f:
                return json.load(f)

    tasks_file = data_dir / f'tasks-canonical-{model}.json'
    if not tasks_file.exists():
        return []
    with open(tasks_file) as f:
        return json.load(f)


def format_task_brief(task: dict, show_outcome: bool = True, show_type: bool = True) -> str:
    """Format task as single line."""
    tid = task['task_id'][-20:]  # Truncate
    prompt = task['user_prompt'][:50].replace('\n', ' ')
    tools = len(task.get('tool_calls', []))
    files = task.get('total_files_touched', 0)
    duration = task.get('duration_seconds', 0)

    # Classification info
    classification = task.get('classification', {})
    task_type = classification.get('type', '?')
    complexity = classification.get('complexity', '?')

    type_str = f"[{task_type[:4]}/{complexity[:3]}] " if show_type else ""
    line = f"{type_str}{tid}: {prompt}... | {tools} tools, {files} files, {duration:.0f}s"

    if show_outcome:
        outcome = task.get('outcome_category', '?')
        evidence = task.get('outcome_evidence', '')[:20]
        line += f" | {outcome}"
        if evidence:
            line += f" ({evidence})"

    return line


def format_task_detail(task: dict) -> str:
    """Format task with full details."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"TASK: {task['task_id']}")
    lines.append(f"Model: {task['model']} | Project: {task.get('project_path', '?')}")
    lines.append("=" * 70)

    lines.append("\n## User Request")
    lines.append(task.get('user_prompt_full', task.get('user_prompt', ''))[:1000])

    lines.append("\n## Scale Metrics")
    lines.append(f"  Duration: {task.get('duration_seconds', 0):.1f}s")
    lines.append(f"  Tool calls: {len(task.get('tool_calls', []))}")
    lines.append(f"  Files touched: {task.get('total_files_touched', 0)}")
    lines.append(f"  Lines added: {task.get('total_lines_added', 0)}")
    lines.append(f"  Lines removed: {task.get('total_lines_removed', 0)}")

    lines.append("\n## Tool Sequence")
    lines.append(f"  {task.get('tool_sequence', 'N/A')}")

    if task.get('files_read'):
        lines.append("\n## Files Read")
        for f in task['files_read'][:10]:
            lines.append(f"  - {f}")
        if len(task['files_read']) > 10:
            lines.append(f"  ... and {len(task['files_read']) - 10} more")

    if task.get('files_written'):
        lines.append("\n## Files Written")
        for f in task['files_written']:
            lines.append(f"  - {f}")

    if task.get('files_edited'):
        lines.append("\n## Files Edited")
        for f in task['files_edited'][:10]:
            lines.append(f"  - {f}")
        if len(task['files_edited']) > 10:
            lines.append(f"  ... and {len(task['files_edited']) - 10} more")

    if task.get('bash_commands'):
        lines.append("\n## Bash Commands")
        for cmd in task['bash_commands'][:5]:
            lines.append(f"  $ {cmd[:80]}")
        if len(task['bash_commands']) > 5:
            lines.append(f"  ... and {len(task['bash_commands']) - 5} more")

    lines.append("\n## Outcome (from next user message)")
    lines.append(f"  Category: {task.get('outcome_category', '?')}")
    lines.append(f"  Evidence: {task.get('outcome_evidence', 'N/A')}")
    if task.get('next_user_message'):
        lines.append(f"  Next message: {task['next_user_message'][:200]}...")

    return '\n'.join(lines)


def filter_tasks(tasks: list[dict],
                 min_tools: int = 0,
                 min_files: int = 0,
                 outcome: Optional[str] = None,
                 prompt_pattern: Optional[str] = None,
                 project: Optional[str] = None,
                 task_type: Optional[str] = None,
                 complexity: Optional[str] = None,
                 exclude_continuations: bool = False) -> list[dict]:
    """Filter tasks by criteria."""
    filtered = []
    for t in tasks:
        if len(t.get('tool_calls', [])) < min_tools:
            continue
        if t.get('total_files_touched', 0) < min_files:
            continue
        if outcome and t.get('outcome_category') != outcome:
            continue
        if prompt_pattern and not re.search(prompt_pattern, t.get('user_prompt', ''), re.I):
            continue
        if project and project.lower() not in t.get('project_path', '').lower():
            continue

        # Classification filters
        classification = t.get('classification', {})
        if task_type and classification.get('type') != task_type:
            continue
        if complexity and classification.get('complexity') != complexity:
            continue
        if exclude_continuations and classification.get('type') == 'continuation':
            continue

        filtered.append(t)
    return filtered


def find_similar_tasks(task: dict, all_tasks: list[dict], top_n: int = 5) -> list[tuple[dict, float]]:
    """Find tasks similar to the given task based on prompt and structure."""
    prompt_words = set(task.get('user_prompt', '').lower().split())
    tool_set = set(t['name'] for t in task.get('tool_calls', []))

    scores = []
    for other in all_tasks:
        if other['task_id'] == task['task_id']:
            continue

        # Word overlap score
        other_words = set(other.get('user_prompt', '').lower().split())
        if not prompt_words or not other_words:
            word_score = 0
        else:
            word_score = len(prompt_words & other_words) / len(prompt_words | other_words)

        # Tool overlap score
        other_tools = set(t['name'] for t in other.get('tool_calls', []))
        if not tool_set or not other_tools:
            tool_score = 0
        else:
            tool_score = len(tool_set & other_tools) / len(tool_set | other_tools)

        # Scale similarity (files touched)
        files1 = task.get('total_files_touched', 0)
        files2 = other.get('total_files_touched', 0)
        if files1 + files2 > 0:
            scale_score = 1 - abs(files1 - files2) / max(files1 + files2, 1)
        else:
            scale_score = 1

        # Combined score
        score = 0.5 * word_score + 0.3 * tool_score + 0.2 * scale_score
        scores.append((other, score))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def cmd_list(args):
    """List tasks for a model."""
    tasks = load_tasks(args.model, args.data_dir)
    if not tasks:
        print(f"No tasks found for {args.model}")
        return

    # Apply filters
    tasks = filter_tasks(
        tasks,
        min_tools=args.min_tools,
        min_files=args.min_files,
        outcome=args.outcome,
        prompt_pattern=args.pattern,
        project=args.project,
        task_type=args.type,
        complexity=args.complexity,
        exclude_continuations=args.no_continuations,
    )

    # Sort
    if args.sort == 'tools':
        tasks.sort(key=lambda t: len(t.get('tool_calls', [])), reverse=True)
    elif args.sort == 'files':
        tasks.sort(key=lambda t: t.get('total_files_touched', 0), reverse=True)
    elif args.sort == 'duration':
        tasks.sort(key=lambda t: t.get('duration_seconds', 0), reverse=True)
    elif args.sort == 'lines':
        tasks.sort(key=lambda t: t.get('total_lines_added', 0), reverse=True)

    # Limit
    tasks = tasks[:args.limit]

    print(f"\n{args.model.upper()} Tasks ({len(tasks)} shown):\n")
    for task in tasks:
        print(format_task_brief(task))


def cmd_show(args):
    """Show detailed task info."""
    tasks = load_tasks(args.model, args.data_dir)
    for task in tasks:
        if args.task_id in task['task_id']:
            print(format_task_detail(task))
            return
    print(f"Task not found: {args.task_id}")


def cmd_similar(args):
    """Find similar tasks across models."""
    source_tasks = load_tasks(args.model, args.data_dir)
    target_model = 'opus-4-6' if args.model == 'opus-4-5' else 'opus-4-5'
    target_tasks = load_tasks(target_model, args.data_dir)

    # Find source task
    source_task = None
    for task in source_tasks:
        if args.task_id in task['task_id']:
            source_task = task
            break

    if not source_task:
        print(f"Task not found: {args.task_id}")
        return

    print(f"\nSource task ({args.model}):")
    print(format_task_brief(source_task))
    print(f"\nSimilar tasks in {target_model}:\n")

    similar = find_similar_tasks(source_task, target_tasks, top_n=args.count)
    for task, score in similar:
        print(f"  [{score:.2f}] {format_task_brief(task)}")


def cmd_compare(args):
    """Compare two specific tasks side by side."""
    opus_4_5_tasks = load_tasks('opus-4-5', args.data_dir)
    opus_4_6_tasks = load_tasks('opus-4-6', args.data_dir)

    task1 = None
    task2 = None

    for t in opus_4_5_tasks + opus_4_6_tasks:
        if args.task1 in t['task_id']:
            task1 = t
        if args.task2 in t['task_id']:
            task2 = t

    if not task1:
        print(f"Task 1 not found: {args.task1}")
        return
    if not task2:
        print(f"Task 2 not found: {args.task2}")
        return

    print("\n" + "=" * 70)
    print("TASK COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Task 1 (' + task1['model'] + ')':<25} {'Task 2 (' + task2['model'] + ')':<25}")
    print("-" * 75)

    metrics = [
        ('Tool calls', len(task1.get('tool_calls', [])), len(task2.get('tool_calls', []))),
        ('Files touched', task1.get('total_files_touched', 0), task2.get('total_files_touched', 0)),
        ('Lines added', task1.get('total_lines_added', 0), task2.get('total_lines_added', 0)),
        ('Lines removed', task1.get('total_lines_removed', 0), task2.get('total_lines_removed', 0)),
        ('Duration (s)', f"{task1.get('duration_seconds', 0):.1f}", f"{task2.get('duration_seconds', 0):.1f}"),
        ('Outcome', task1.get('outcome_category', '?'), task2.get('outcome_category', '?')),
    ]

    for name, v1, v2 in metrics:
        print(f"{name:<25} {str(v1):<25} {str(v2):<25}")

    print("\n## Prompts")
    print(f"\nTask 1: {task1.get('user_prompt', '')[:200]}")
    print(f"\nTask 2: {task2.get('user_prompt', '')[:200]}")

    print("\n## Tool Sequences")
    print(f"Task 1: {task1.get('tool_sequence', 'N/A')}")
    print(f"Task 2: {task2.get('tool_sequence', 'N/A')}")


def cmd_stats(args):
    """Show statistics for both models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON STATISTICS")
    print("=" * 70)

    for model in ['opus-4-5', 'opus-4-6']:
        tasks = load_tasks(model, args.data_dir)
        if not tasks:
            continue

        print(f"\n{model.upper()} ({len(tasks)} tasks)")
        print("-" * 40)

        # Outcome distribution
        outcomes = {}
        for t in tasks:
            cat = t.get('outcome_category', 'unknown')
            outcomes[cat] = outcomes.get(cat, 0) + 1

        print("Outcomes (from user's next message):")
        for cat, count in sorted(outcomes.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(tasks)
            print(f"  {cat}: {count} ({pct:.1f}%)")

        # Scale metrics
        with_changes = [t for t in tasks if t.get('total_files_touched', 0) > 0]
        if with_changes:
            avg_files = sum(t['total_files_touched'] for t in with_changes) / len(with_changes)
            avg_lines = sum(t.get('total_lines_added', 0) for t in with_changes) / len(with_changes)
            avg_tools = sum(len(t.get('tool_calls', [])) for t in with_changes) / len(with_changes)
            avg_duration = sum(t.get('duration_seconds', 0) for t in with_changes) / len(with_changes)

            print(f"\nScale (tasks with file changes, n={len(with_changes)}):")
            print(f"  Avg files touched: {avg_files:.1f}")
            print(f"  Avg lines added: {avg_lines:.1f}")
            print(f"  Avg tool calls: {avg_tools:.1f}")
            print(f"  Avg duration: {avg_duration:.1f}s ({avg_duration/60:.1f}m)")


def compute_efficiency_metrics(tasks: list[dict]) -> dict:
    """Compute efficiency metrics for a set of tasks."""
    if not tasks:
        return {}

    # Filter to tasks with actual changes
    with_changes = [t for t in tasks if t.get('total_files_touched', 0) > 0]
    with_lines = [t for t in tasks if (t.get('total_lines_added', 0) + t.get('total_lines_removed', 0)) > 0]

    def safe_avg(vals):
        return sum(vals) / len(vals) if vals else 0

    # Basic metrics
    avg_tools = safe_avg([len(t.get('tool_calls', [])) for t in tasks])
    avg_files = safe_avg([t.get('total_files_touched', 0) for t in with_changes]) if with_changes else 0
    avg_lines = safe_avg([t.get('total_lines_added', 0) + t.get('total_lines_removed', 0) for t in with_lines]) if with_lines else 0

    # Efficiency metrics (lower is more efficient)
    tools_per_file = []
    tools_per_100_lines = []

    for t in with_changes:
        tools = len(t.get('tool_calls', []))
        files = t.get('total_files_touched', 0)
        lines = t.get('total_lines_added', 0) + t.get('total_lines_removed', 0)

        if files > 0:
            tools_per_file.append(tools / files)
        if lines > 0:
            tools_per_100_lines.append(tools / (lines / 100))

    return {
        'count': len(tasks),
        'with_changes': len(with_changes),
        'avg_tools': avg_tools,
        'avg_files': avg_files,
        'avg_lines': avg_lines,
        'tools_per_file': safe_avg(tools_per_file),
        'tools_per_100_lines': safe_avg(tools_per_100_lines),
        'dissat_pct': 100 * sum(1 for t in tasks if t.get('outcome_category') == 'dissatisfaction') / len(tasks) if tasks else 0,
    }


def cmd_compare_types(args):
    """Compare models by task type."""
    opus_4_5_tasks = load_tasks('opus-4-5', args.data_dir)
    opus_4_6_tasks = load_tasks('opus-4-6', args.data_dir)

    # Filter out continuations for fair comparison
    opus_4_5_tasks = [t for t in opus_4_5_tasks if t.get('classification', {}).get('type') != 'continuation']
    opus_4_6_tasks = [t for t in opus_4_6_tasks if t.get('classification', {}).get('type') != 'continuation']

    print("\n" + "=" * 120)
    print("MODEL COMPARISON BY TASK TYPE (excluding continuations)")
    print("=" * 120)

    # Get all types
    all_types = set()
    for t in opus_4_5_tasks + opus_4_6_tasks:
        all_types.add(t.get('classification', {}).get('type', 'unknown'))

    print(f"\n{'Type':<12} {'#O':<5} {'#F':<5} {'O-Tools':<8} {'F-Tools':<8} {'O-Files':<8} {'F-Files':<8} {'O-Lines':<8} {'F-Lines':<8} {'O-T/File':<8} {'F-T/File':<8} {'O-Dis%':<7} {'F-Dis%':<7}")
    print("-" * 120)

    for task_type in sorted(all_types):
        opus_4_5_typed = [t for t in opus_4_5_tasks if t.get('classification', {}).get('type') == task_type]
        opus_4_6_typed = [t for t in opus_4_6_tasks if t.get('classification', {}).get('type') == task_type]

        if not opus_4_5_typed and not opus_4_6_typed:
            continue

        om = compute_efficiency_metrics(opus_4_5_typed)
        fm = compute_efficiency_metrics(opus_4_6_typed)

        print(f"{task_type:<12} {om.get('count',0):<5} {fm.get('count',0):<5} "
              f"{om.get('avg_tools',0):<8.1f} {fm.get('avg_tools',0):<8.1f} "
              f"{om.get('avg_files',0):<8.1f} {fm.get('avg_files',0):<8.1f} "
              f"{om.get('avg_lines',0):<8.0f} {fm.get('avg_lines',0):<8.0f} "
              f"{om.get('tools_per_file',0):<8.1f} {fm.get('tools_per_file',0):<8.1f} "
              f"{om.get('dissat_pct',0):<7.1f} {fm.get('dissat_pct',0):<7.1f}")

    print("\n" + "=" * 120)
    print("MODEL COMPARISON BY COMPLEXITY (with efficiency metrics)")
    print("=" * 120)

    print(f"\n{'Complexity':<10} {'#O':<5} {'#F':<5} {'O-Tools':<8} {'F-Tools':<8} {'O-Lines':<8} {'F-Lines':<8} {'O-T/File':<8} {'F-T/File':<8} {'O-T/100L':<8} {'F-T/100L':<8} {'O-Dis%':<7} {'F-Dis%':<7}")
    print("-" * 120)

    for complexity in ['trivial', 'simple', 'moderate', 'complex', 'major']:
        opus_4_5_comp = [t for t in opus_4_5_tasks if t.get('classification', {}).get('complexity') == complexity]
        opus_4_6_comp = [t for t in opus_4_6_tasks if t.get('classification', {}).get('complexity') == complexity]

        om = compute_efficiency_metrics(opus_4_5_comp)
        fm = compute_efficiency_metrics(opus_4_6_comp)

        print(f"{complexity:<10} {om.get('count',0):<5} {fm.get('count',0):<5} "
              f"{om.get('avg_tools',0):<8.1f} {fm.get('avg_tools',0):<8.1f} "
              f"{om.get('avg_lines',0):<8.0f} {fm.get('avg_lines',0):<8.0f} "
              f"{om.get('tools_per_file',0):<8.1f} {fm.get('tools_per_file',0):<8.1f} "
              f"{om.get('tools_per_100_lines',0):<8.1f} {fm.get('tools_per_100_lines',0):<8.1f} "
              f"{om.get('dissat_pct',0):<7.1f} {fm.get('dissat_pct',0):<7.1f}")

    # Summary efficiency comparison
    print("\n" + "=" * 120)
    print("EFFICIENCY SUMMARY (tasks with file changes only)")
    print("=" * 120)

    opus_4_5_with_changes = [t for t in opus_4_5_tasks if t.get('total_files_touched', 0) > 0]
    opus_4_6_with_changes = [t for t in opus_4_6_tasks if t.get('total_files_touched', 0) > 0]

    om = compute_efficiency_metrics(opus_4_5_with_changes)
    fm = compute_efficiency_metrics(opus_4_6_with_changes)

    print(f"\n{'Metric':<25} {'Opus 4.5':<15} {'Opus 4.6':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Tasks with changes':<25} {om['count']:<15} {fm['count']:<15}")
    print(f"{'Avg tools/task':<25} {om['avg_tools']:<15.1f} {fm['avg_tools']:<15.1f} {fm['avg_tools']-om['avg_tools']:+.1f}")
    print(f"{'Avg files/task':<25} {om['avg_files']:<15.1f} {fm['avg_files']:<15.1f} {fm['avg_files']-om['avg_files']:+.1f}")
    print(f"{'Avg lines/task':<25} {om['avg_lines']:<15.0f} {fm['avg_lines']:<15.0f} {fm['avg_lines']-om['avg_lines']:+.0f}")
    print(f"{'Tools per file':<25} {om['tools_per_file']:<15.2f} {fm['tools_per_file']:<15.2f} {fm['tools_per_file']-om['tools_per_file']:+.2f}")
    print(f"{'Tools per 100 lines':<25} {om['tools_per_100_lines']:<15.2f} {fm['tools_per_100_lines']:<15.2f} {fm['tools_per_100_lines']-om['tools_per_100_lines']:+.2f}")
    print(f"{'Dissatisfaction %':<25} {om['dissat_pct']:<15.1f} {fm['dissat_pct']:<15.1f} {fm['dissat_pct']-om['dissat_pct']:+.1f}")


def main():
    parser = argparse.ArgumentParser(description='Browse and compare tasks')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    subparsers = parser.add_subparsers(dest='command', required=True)

    # List command
    list_parser = subparsers.add_parser('list', help='List tasks')
    list_parser.add_argument('model', choices=['opus-4-5', 'opus-4-6'])
    list_parser.add_argument('--limit', type=int, default=20)
    list_parser.add_argument('--min-tools', type=int, default=0)
    list_parser.add_argument('--min-files', type=int, default=0)
    list_parser.add_argument('--outcome', choices=['satisfaction', 'dissatisfaction', 'continuation', 'new_task', 'session_end', 'unclear'])
    list_parser.add_argument('--pattern', help='Regex pattern to match in prompt')
    list_parser.add_argument('--project', help='Filter by project path')
    list_parser.add_argument('--type', choices=['investigation', 'bugfix', 'refactor', 'port', 'greenfield', 'feature', 'sysadmin', 'docs', 'simple', 'continuation', 'unknown'])
    list_parser.add_argument('--complexity', choices=['trivial', 'simple', 'moderate', 'complex', 'major'])
    list_parser.add_argument('--no-continuations', action='store_true', help='Exclude continuation tasks')
    list_parser.add_argument('--sort', choices=['tools', 'files', 'duration', 'lines'], default='tools')
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser('show', help='Show task details')
    show_parser.add_argument('model', choices=['opus-4-5', 'opus-4-6'])
    show_parser.add_argument('task_id', help='Task ID (partial match)')
    show_parser.set_defaults(func=cmd_show)

    # Similar command
    similar_parser = subparsers.add_parser('similar', help='Find similar tasks in other model')
    similar_parser.add_argument('model', choices=['opus-4-5', 'opus-4-6'])
    similar_parser.add_argument('task_id', help='Task ID (partial match)')
    similar_parser.add_argument('--count', type=int, default=5)
    similar_parser.set_defaults(func=cmd_similar)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two tasks')
    compare_parser.add_argument('task1', help='First task ID')
    compare_parser.add_argument('task2', help='Second task ID')
    compare_parser.set_defaults(func=cmd_compare)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show model statistics')
    stats_parser.set_defaults(func=cmd_stats)

    # Compare types command
    typecmp_parser = subparsers.add_parser('compare-types', help='Compare models by task type')
    typecmp_parser.set_defaults(func=cmd_compare_types)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
