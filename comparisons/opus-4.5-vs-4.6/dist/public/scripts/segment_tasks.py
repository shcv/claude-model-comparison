#!/usr/bin/env python3
"""
Task Segmentation for Opus 4.5 vs Opus 4.6 Model Comparison

Breaks sessions into discrete "tasks" (user request → completion) with
heuristic-based boundary detection.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime


@dataclass
class TaskBoundary:
    task_id: str
    session_id: str
    user_prompt: str
    tool_calls: int
    tools_used: list
    duration_seconds: float
    completion_signal: str
    appears_successful: bool


# Patterns for detecting completion signals
DONE_PATTERNS = [
    r'\bthanks?\b', r'\bthank you\b', r'\bperfect\b', r'\bgreat\b',
    r'\bawesome\b', r'\blooks good\b', r'\bnice\b', r'\bgot it\b',
    r'\bthat works\b', r'\bexactly\b'
]
DISSATISFIED_PATTERNS = [
    r'\bwrong\b', r'\btry again\b', r'\bnot what\b', r'\bincorrect\b',
    r'\bno[,.]?\s*(that|this)\b', r'\bactually\b', r"\bthat's not\b"
]

# Patterns for system/meta messages to skip
SKIP_PATTERNS = [
    r'^<local-command',
    r'^<command-name>',
    r'^<system-reminder>',
    r'^\s*$',  # Empty
]


def extract_user_text(content) -> str:
    """Extract text from user message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    texts.append(block.get('text', ''))
            elif isinstance(block, str):
                texts.append(block)
        return ' '.join(texts)
    return ''


def is_tool_result(content) -> bool:
    """Check if user message is a tool result (not human input)."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                return True
    return False


def should_skip_message(user_text: str) -> bool:
    """Check if this message should be skipped (system/meta content)."""
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, user_text.strip()):
            return True
    return False


def detect_completion_signal(user_text: str) -> Optional[str]:
    """Detect what kind of completion signal a user message represents."""
    text_lower = user_text.lower()

    # Check for satisfaction signals
    for pattern in DONE_PATTERNS:
        if re.search(pattern, text_lower):
            return 'explicit_done'

    # Check for dissatisfaction signals
    for pattern in DISSATISFIED_PATTERNS:
        if re.search(pattern, text_lower):
            return 'user_dissatisfied'

    return None


def segment_session(file_path: Path) -> list[TaskBoundary]:
    """Segment a session file into discrete tasks."""
    messages = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages.append(obj)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    tasks = []
    session_id = file_path.stem

    # Track current task state
    current_task_start_idx = None
    current_prompt = ""
    current_tool_calls = 0
    current_tools = []
    current_start_time = None
    task_counter = 0

    def finalize_task(end_idx: int, signal: str, successful: bool):
        nonlocal task_counter, current_task_start_idx, current_prompt
        nonlocal current_tool_calls, current_tools, current_start_time

        if current_task_start_idx is None or not current_prompt.strip():
            return

        # Calculate duration
        end_time = None
        for i in range(end_idx, current_task_start_idx - 1, -1):
            if i < len(messages) and 'timestamp' in messages[i]:
                end_time = messages[i]['timestamp']
                break

        duration = 0
        if current_start_time and end_time:
            try:
                start = datetime.fromisoformat(current_start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = (end - start).total_seconds()
            except ValueError:
                pass

        task_counter += 1
        tasks.append(TaskBoundary(
            task_id=f"{session_id}-task-{task_counter}",
            session_id=session_id,
            user_prompt=current_prompt[:500],  # Truncate long prompts
            tool_calls=current_tool_calls,
            tools_used=list(set(current_tools)),
            duration_seconds=round(duration, 1),
            completion_signal=signal,
            appears_successful=successful
        ))

        # Reset state
        current_task_start_idx = None
        current_prompt = ""
        current_tool_calls = 0
        current_tools = []
        current_start_time = None

    for idx, msg in enumerate(messages):
        msg_type = msg.get('type')

        if msg_type == 'user':
            content = msg.get('message', {}).get('content', [])

            # Skip tool results - these aren't human messages
            if is_tool_result(content):
                continue

            user_text = extract_user_text(content)

            # Skip system/meta messages
            if should_skip_message(user_text):
                continue

            # Check if this ends the previous task
            if current_task_start_idx is not None:
                signal = detect_completion_signal(user_text)
                if signal:
                    successful = signal == 'explicit_done'
                    finalize_task(idx - 1, signal, successful)
                else:
                    # User continuing with new request - treat as new topic
                    finalize_task(idx - 1, 'user_continues', True)

            # Start new task
            current_task_start_idx = idx
            current_prompt = user_text
            current_start_time = msg.get('timestamp')

        elif msg_type == 'assistant':
            message = msg.get('message', {})
            content = message.get('content', [])

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        current_tool_calls += 1
                        current_tools.append(block.get('name', 'unknown'))

    # Finalize any remaining task
    if current_task_start_idx is not None:
        finalize_task(len(messages) - 1, 'session_end', True)

    return tasks


def segment_all_sessions(sessions_file: Path) -> list[TaskBoundary]:
    """Segment all sessions from a sessions JSON file."""
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    all_tasks = []
    for session in sessions:
        file_path = Path(session['file_path'])
        if file_path.exists():
            tasks = segment_session(file_path)
            all_tasks.extend(tasks)

    return all_tasks


def print_summary(tasks: list[TaskBoundary], model: str):
    """Print task summary statistics."""
    if not tasks:
        print(f"\n{model.upper()}: No tasks found")
        return

    print(f"\n{model.upper()} Tasks:")
    print(f"  Total tasks: {len(tasks)}")

    # Completion signals
    signals = {}
    for t in tasks:
        signals[t.completion_signal] = signals.get(t.completion_signal, 0) + 1
    print(f"  Completion signals: {signals}")

    # Success rate
    successful = sum(1 for t in tasks if t.appears_successful)
    print(f"  Appears successful: {successful}/{len(tasks)} ({100*successful/len(tasks):.1f}%)")

    # Tool usage
    avg_tools = sum(t.tool_calls for t in tasks) / len(tasks)
    print(f"  Avg tool calls per task: {avg_tools:.1f}")

    # Duration
    durations = [t.duration_seconds for t in tasks if t.duration_seconds > 0]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"  Avg duration: {avg_duration:.0f}s ({avg_duration/60:.1f}m)")


def save_results(tasks: list[TaskBoundary], output_file: Path):
    """Save tasks to JSON file."""
    data = [asdict(t) for t in tasks]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} tasks to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Segment sessions into tasks')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory (default: data)')
    args = parser.parse_args()

    print("=" * 60)
    print("TASK SEGMENTATION")
    print("=" * 60)

    for model in ['opus-4-5', 'opus-4-6']:
        sessions_file = args.data_dir / f'sessions-{model}.json'
        if not sessions_file.exists():
            print(f"\nSkipping {model}: {sessions_file} not found")
            continue

        tasks = segment_all_sessions(sessions_file)
        print_summary(tasks, model)

        output_file = args.data_dir / f'tasks-{model}.json'
        save_results(tasks, output_file)


if __name__ == '__main__':
    main()
