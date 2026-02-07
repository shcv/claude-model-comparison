#!/usr/bin/env python3
"""
Task Replication for Direct Model Comparison

Replay a task from one model with another model for direct comparison.
This creates controlled experiments where we can compare how different
models handle the exact same prompt.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_task(task_id: str, data_dir: Path = Path('data')) -> Optional[dict]:
    """Load a specific task by ID (partial match)."""
    for model in ['opus-4-5', 'opus-4-6']:
        tasks_file = data_dir / f'tasks-deep-{model}.json'
        if not tasks_file.exists():
            continue
        with open(tasks_file) as f:
            tasks = json.load(f)
            for task in tasks:
                if task_id in task['task_id']:
                    return task
    return None


def format_replication_prompt(task: dict, include_context: bool = True) -> str:
    """Format a task as a replication prompt."""
    prompt = task.get('user_prompt_full', task.get('user_prompt', ''))

    if include_context:
        # Add context about what we're testing
        context = f"""# Task Replication

Original task: {task['task_id']}
Original model: {task['model']}
Original outcome: {task.get('outcome_category', 'unknown')}

## Original Prompt

{prompt}

## Instructions

Please complete this task as you normally would. This is a replication test
to compare model behavior on identical prompts."""
        return context

    return prompt


def export_for_replication(task: dict, output_file: Path):
    """Export task details for manual replication."""
    export_data = {
        'original_task_id': task['task_id'],
        'original_model': task['model'],
        'prompt': task.get('user_prompt_full', task.get('user_prompt', '')),
        'original_metrics': {
            'tool_calls': len(task.get('tool_calls', [])),
            'files_touched': task.get('total_files_touched', 0),
            'lines_added': task.get('total_lines_added', 0),
            'lines_removed': task.get('total_lines_removed', 0),
            'duration_seconds': task.get('duration_seconds', 0),
            'outcome': task.get('outcome_category', 'unknown'),
        },
        'original_tool_sequence': task.get('tool_sequence', ''),
        'files_read': task.get('files_read', [])[:10],
        'files_written': task.get('files_written', []),
        'files_edited': task.get('files_edited', [])[:10],
        'project_path': task.get('project_path', ''),
        'exported_at': datetime.now().isoformat(),
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported task to {output_file}")
    return export_data


def run_replication(task: dict, model: str = 'opus-4-5', dry_run: bool = True) -> Optional[dict]:
    """
    Run a replication of the task with the specified model.

    Note: This would need to be run in the correct project directory
    with the correct file state for a true replication.
    """
    prompt = task.get('user_prompt_full', task.get('user_prompt', ''))

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Replication Preview")
        print("=" * 60)
        print(f"\nOriginal: {task['task_id']} ({task['model']})")
        print(f"Target model: {model}")
        print(f"\nPrompt ({len(prompt)} chars):")
        print("-" * 40)
        print(prompt[:500])
        if len(prompt) > 500:
            print(f"... [{len(prompt) - 500} more chars]")
        print("-" * 40)
        print("\nOriginal metrics:")
        print(f"  Tool calls: {len(task.get('tool_calls', []))}")
        print(f"  Files touched: {task.get('total_files_touched', 0)}")
        print(f"  Duration: {task.get('duration_seconds', 0):.1f}s")
        print(f"  Outcome: {task.get('outcome_category', '?')}")
        print("\nTo run for real, use --execute")
        return None

    # For actual execution, we'd need to:
    # 1. Navigate to the correct project directory
    # 2. Ensure file state matches (or accept differences)
    # 3. Run claude with the prompt
    # 4. Capture the session for analysis

    print(f"\nRunning replication with {model}...")
    print("NOTE: This requires being in the correct project directory")
    print("      and may produce different results due to file state changes")

    # This is a placeholder - actual execution would need more setup
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "text"]
    print(f"\nCommand: {' '.join(cmd[:4])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
        }
    except subprocess.TimeoutExpired:
        print("Replication timed out")
        return None
    except Exception as e:
        print(f"Replication failed: {e}")
        return None


def create_replication_batch(tasks: list[dict], output_dir: Path, max_tasks: int = 10):
    """Create a batch of tasks for replication."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_info = {
        'created_at': datetime.now().isoformat(),
        'tasks': [],
    }

    for i, task in enumerate(tasks[:max_tasks]):
        task_file = output_dir / f'task-{i+1:02d}.json'
        export_data = export_for_replication(task, task_file)
        batch_info['tasks'].append({
            'file': str(task_file),
            'original_task_id': task['task_id'],
            'original_model': task['model'],
        })

    # Save batch manifest
    manifest_file = output_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(batch_info, f, indent=2)

    print(f"\nCreated replication batch in {output_dir}")
    print(f"  Tasks: {len(batch_info['tasks'])}")
    print(f"  Manifest: {manifest_file}")


def select_replication_candidates(tasks: list[dict], count: int = 10) -> list[dict]:
    """Select good candidates for replication based on criteria."""
    # Filter to substantial tasks with clear outcomes
    candidates = []
    for t in tasks:
        # Skip trivial tasks
        if len(t.get('tool_calls', [])) < 3:
            continue
        # Skip unclear outcomes
        if t.get('outcome_category') not in ['satisfaction', 'dissatisfaction', 'new_task']:
            continue
        # Skip very long tasks (hard to replicate)
        if t.get('duration_seconds', 0) > 600:
            continue
        # Skip tasks that heavily depend on file state
        if len(t.get('files_read', [])) > 20:
            continue

        candidates.append(t)

    # Sort by a combination of factors
    def score(t):
        # Prefer tasks with moderate complexity
        tools = len(t.get('tool_calls', []))
        tool_score = min(tools, 20) / 20  # Cap at 20

        # Prefer clear outcomes
        outcome = t.get('outcome_category', '')
        outcome_score = 1 if outcome in ['satisfaction', 'dissatisfaction'] else 0.5

        # Prefer tasks with file changes (more to compare)
        files = t.get('total_files_touched', 0)
        file_score = min(files, 5) / 5  # Cap at 5

        return tool_score + outcome_score + file_score

    candidates.sort(key=score, reverse=True)
    return candidates[:count]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Replicate tasks for model comparison')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Export single task
    export_parser = subparsers.add_parser('export', help='Export a task for replication')
    export_parser.add_argument('task_id', help='Task ID (partial match)')
    export_parser.add_argument('--output', type=Path, default=Path('analysis/replication'))

    # Preview replication
    preview_parser = subparsers.add_parser('preview', help='Preview a replication')
    preview_parser.add_argument('task_id', help='Task ID (partial match)')

    # Create batch
    batch_parser = subparsers.add_parser('batch', help='Create replication batch')
    batch_parser.add_argument('model', choices=['opus-4-5', 'opus-4-6'], help='Source model')
    batch_parser.add_argument('--count', type=int, default=10)
    batch_parser.add_argument('--output', type=Path, default=Path('analysis/replication'))

    # Candidates
    candidates_parser = subparsers.add_parser('candidates', help='Show replication candidates')
    candidates_parser.add_argument('model', choices=['opus-4-5', 'opus-4-6'])
    candidates_parser.add_argument('--count', type=int, default=10)

    args = parser.parse_args()

    if args.command == 'export':
        task = load_task(args.task_id, args.data_dir)
        if not task:
            print(f"Task not found: {args.task_id}")
            sys.exit(1)
        args.output.mkdir(parents=True, exist_ok=True)
        output_file = args.output / f'{task["task_id"]}.json'
        export_for_replication(task, output_file)

    elif args.command == 'preview':
        task = load_task(args.task_id, args.data_dir)
        if not task:
            print(f"Task not found: {args.task_id}")
            sys.exit(1)
        run_replication(task, dry_run=True)

    elif args.command == 'batch':
        tasks_file = args.data_dir / f'tasks-deep-{args.model}.json'
        if not tasks_file.exists():
            print(f"No tasks found for {args.model}")
            sys.exit(1)
        with open(tasks_file) as f:
            tasks = json.load(f)
        candidates = select_replication_candidates(tasks, args.count)
        create_replication_batch(candidates, args.output / args.model, args.count)

    elif args.command == 'candidates':
        tasks_file = args.data_dir / f'tasks-deep-{args.model}.json'
        if not tasks_file.exists():
            print(f"No tasks found for {args.model}")
            sys.exit(1)
        with open(tasks_file) as f:
            tasks = json.load(f)
        candidates = select_replication_candidates(tasks, args.count)

        print(f"\nTop {len(candidates)} replication candidates from {args.model}:\n")
        for i, t in enumerate(candidates, 1):
            prompt = t.get('user_prompt', '')[:60].replace('\n', ' ')
            tools = len(t.get('tool_calls', []))
            outcome = t.get('outcome_category', '?')
            print(f"{i:2}. [{outcome}] {prompt}... ({tools} tools)")
            print(f"    ID: {t['task_id']}")


if __name__ == '__main__':
    main()
