#!/usr/bin/env python3
"""
Deep Task Extraction for Model Comparison

Extracts comprehensive task data including:
- Full user prompt and context
- Agent actions with file/line change metrics
- User's NEXT message to determine actual outcome
- Git-style stats: files read, written, lines changed
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileChange:
    """Tracks changes to a single file."""
    path: str
    action: str  # read, write, edit
    lines_before: int = 0
    lines_after: int = 0
    lines_added: int = 0
    lines_removed: int = 0


@dataclass
class TaskData:
    """Comprehensive task extraction."""
    task_id: str
    session_id: str
    model: str

    # User request
    user_prompt: str
    user_prompt_full: str  # Untruncated

    # Agent work
    tool_calls: list  # List of {name, input_summary}
    tool_sequence: str  # e.g., "Read→Grep→Edit→Bash"
    files_read: list
    files_written: list
    files_edited: list

    # Scale metrics
    total_files_touched: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    bash_commands: list = field(default_factory=list)

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0

    # Outcome - from USER's perspective
    next_user_message: str = ""
    outcome_category: str = ""  # new_task, continuation, satisfaction, dissatisfaction, session_end
    outcome_evidence: str = ""  # Quote from user message

    # For comparison
    project_path: str = ""
    is_meta: bool = False   # True if session is from claude-investigations (meta/self-referential)
    task_summary: str = ""  # One-line summary for listing


# Patterns for categorizing user responses
SATISFACTION_PATTERNS = [
    (r'\bthanks?\b', 'thanks'),
    (r'\bthank you\b', 'thank you'),
    (r'\bperfect\b', 'perfect'),
    (r'\bgreat\b', 'great'),
    (r'\bawesome\b', 'awesome'),
    (r'\blooks good\b', 'looks good'),
    (r'\bnice\b', 'nice'),
    (r'\bthat works\b', 'that works'),
    (r'\bexactly\b', 'exactly'),
    (r'^lgtm\b', 'lgtm'),
    (r'\bgood job\b', 'good job'),
]

DISSATISFACTION_PATTERNS = [
    (r'\bwrong\b', 'wrong'),
    (r'\btry again\b', 'try again'),
    (r'\bnot what\b', 'not what'),
    (r'\bincorrect\b', 'incorrect'),
    (r"\bthat's not\b", "that's not"),
    (r'\bno[,.]?\s+(?:that|this)\b', 'no, that/this'),
    (r'\bundo\b', 'undo'),
    (r'\brevert\b', 'revert'),
    (r'\bfix\b', 'fix (potential)'),
]

CONTINUATION_PATTERNS = [
    (r'\balso\b', 'also'),
    (r'\band\s+(?:can|could|please)\b', 'and can/could'),
    (r'\bone more\b', 'one more'),
    (r'\bactually\b', 'actually'),
    (r'\bwait\b', 'wait'),
    (r'\bbut\b', 'but'),
    (r'\bhmm\b', 'hmm'),
]

SKIP_PATTERNS = [
    r'^<local-command',
    r'^<command-name>',
    r'^<system-reminder>',
    r'^<task-notification>',
    r'^<teammate-message>',
    r'^\s*$',
]

# Patterns indicating the "user message" is actually system-generated content
SYSTEM_CONTENT_PATTERNS = [
    r'^This session is being continued from a previous conversation',
    r'^# Iteration workflow',
    r'^Implement the following plan:',  # Plan templates - not sentiment
    r'^\[Request interrupted',
    r'^<summary>',
    r'^Analysis:',  # Continuation summaries
]


def should_skip_message(text: str) -> bool:
    """Check if message should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    return False


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
    """Check if user message is a tool result."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                return True
    return False


def is_system_generated(text: str) -> bool:
    """Check if text is system-generated content, not actual user sentiment."""
    for pattern in SYSTEM_CONTENT_PATTERNS:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False


def categorize_user_response(text: str) -> tuple[str, str]:
    """Categorize user response and extract evidence."""
    # First check if this is system-generated content - don't extract sentiment
    if is_system_generated(text):
        return 'system_continuation', 'auto-generated'

    text_lower = text.lower().strip()

    # Check satisfaction first
    for pattern, evidence in SATISFACTION_PATTERNS:
        if re.search(pattern, text_lower):
            return 'satisfaction', evidence

    # Check dissatisfaction - but be more careful
    for pattern, evidence in DISSATISFACTION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            # Additional check: is this a question about fixing, not a complaint?
            # "will this fix it?" vs "that's wrong"
            context_start = max(0, match.start() - 20)
            context = text_lower[context_start:match.end() + 20]
            if re.search(r'\b(will|can|could|would|should|does|is there|any)\b.*\b(fix|wrong)\b', context):
                continue  # Skip - this is a question, not dissatisfaction
            return 'dissatisfaction', evidence

    # Check continuation
    for pattern, evidence in CONTINUATION_PATTERNS:
        if re.search(pattern, text_lower):
            return 'continuation', evidence

    # Default: assume new task if substantial text
    if len(text_lower) > 20:
        return 'new_task', 'new request'
    return 'unclear', ''


def extract_tool_info(tool_use: dict) -> dict:
    """Extract relevant info from a tool use block."""
    name = tool_use.get('name', 'unknown')
    input_data = tool_use.get('input', {})

    info = {'name': name}

    if name == 'Read':
        info['file'] = input_data.get('file_path', '')
    elif name == 'Write':
        info['file'] = input_data.get('file_path', '')
        content = input_data.get('content', '')
        info['lines'] = content.count('\n') + 1 if content else 0
    elif name == 'Edit':
        info['file'] = input_data.get('file_path', '')
        old = input_data.get('old_string', '')
        new = input_data.get('new_string', '')
        info['lines_removed'] = old.count('\n') + 1 if old else 0
        info['lines_added'] = new.count('\n') + 1 if new else 0
    elif name == 'Bash':
        cmd = input_data.get('command', '')
        info['command'] = cmd[:100]  # Truncate
    elif name == 'Grep':
        info['pattern'] = input_data.get('pattern', '')[:50]
    elif name == 'Glob':
        info['pattern'] = input_data.get('pattern', '')

    return info


def extract_tasks_from_session(file_path: Path, model: str) -> list[TaskData]:
    """Extract all tasks from a session file with full context."""
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

    # Extract project path from file location
    project_path = str(file_path.parent.name)

    # Track current task state
    current_task = None
    task_counter = 0

    def finalize_task(next_user_text: str = "", is_session_end: bool = False):
        nonlocal current_task, task_counter
        if current_task is None:
            return

        # Categorize outcome based on next user message
        if is_session_end:
            current_task.outcome_category = 'session_end'
            current_task.outcome_evidence = 'session terminated'
        elif next_user_text:
            current_task.next_user_message = next_user_text[:500]
            category, evidence = categorize_user_response(next_user_text)
            current_task.outcome_category = category
            current_task.outcome_evidence = evidence

        # Calculate totals
        current_task.total_files_touched = len(set(
            current_task.files_read + current_task.files_written + current_task.files_edited
        ))

        # Build tool sequence
        tool_names = [t['name'] for t in current_task.tool_calls]
        # Dedupe consecutive same tools
        deduped = []
        for name in tool_names:
            if not deduped or deduped[-1] != name:
                deduped.append(name)
        current_task.tool_sequence = '→'.join(deduped[:10])  # Limit length

        # Generate summary
        prompt_preview = current_task.user_prompt[:60].replace('\n', ' ')
        current_task.task_summary = f"{prompt_preview}... ({len(current_task.tool_calls)} tools, {current_task.total_files_touched} files)"

        tasks.append(current_task)
        current_task = None

    for idx, msg in enumerate(messages):
        msg_type = msg.get('type')
        timestamp = msg.get('timestamp', '')

        if msg_type == 'user':
            content = msg.get('message', {}).get('content', [])

            # Skip tool results
            if is_tool_result(content):
                continue

            user_text = extract_user_text(content)

            # Skip system messages
            if should_skip_message(user_text):
                continue

            # This user message ends the previous task (if any)
            if current_task is not None:
                finalize_task(next_user_text=user_text)

            # Start new task
            task_counter += 1
            current_task = TaskData(
                task_id=f"{session_id}-task-{task_counter}",
                session_id=session_id,
                model=model,
                user_prompt=user_text[:500],
                user_prompt_full=user_text,
                tool_calls=[],
                tool_sequence="",
                files_read=[],
                files_written=[],
                files_edited=[],
                bash_commands=[],
                start_time=timestamp,
                project_path=project_path,
            )

        elif msg_type == 'assistant' and current_task is not None:
            current_task.end_time = timestamp
            message = msg.get('message', {})
            content = message.get('content', [])

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        tool_info = extract_tool_info(block)
                        current_task.tool_calls.append(tool_info)

                        name = tool_info.get('name')
                        file_path_str = tool_info.get('file', '')

                        if name == 'Read' and file_path_str:
                            current_task.files_read.append(file_path_str)
                        elif name == 'Write' and file_path_str:
                            current_task.files_written.append(file_path_str)
                            current_task.total_lines_added += tool_info.get('lines', 0)
                        elif name == 'Edit' and file_path_str:
                            current_task.files_edited.append(file_path_str)
                            current_task.total_lines_added += tool_info.get('lines_added', 0)
                            current_task.total_lines_removed += tool_info.get('lines_removed', 0)
                        elif name == 'Bash':
                            cmd = tool_info.get('command', '')
                            if cmd:
                                current_task.bash_commands.append(cmd)

    # Finalize last task
    if current_task is not None:
        finalize_task(is_session_end=True)

    # Calculate durations
    for task in tasks:
        if task.start_time and task.end_time:
            try:
                start = datetime.fromisoformat(task.start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(task.end_time.replace('Z', '+00:00'))
                task.duration_seconds = (end - start).total_seconds()
            except ValueError:
                pass

    return tasks


def extract_all_tasks(sessions_file: Path, model: str, include_meta: bool = True) -> list[TaskData]:
    """Extract tasks from all sessions in a sessions JSON file.

    Args:
        include_meta: If False, skip sessions tagged as is_meta (default: include all)
    """
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    all_tasks = []
    meta_skipped = 0
    for session in sessions:
        is_meta = session.get('is_meta', False)
        if not include_meta and is_meta:
            meta_skipped += 1
            continue

        file_path = Path(session['file_path'])
        if file_path.exists():
            tasks = extract_tasks_from_session(file_path, model)
            # Propagate is_meta and project_path from session to tasks
            for task in tasks:
                task.is_meta = is_meta
                if not task.project_path and 'project_path' in session:
                    task.project_path = session['project_path']
            all_tasks.extend(tasks)

    if meta_skipped:
        print(f"  Skipped {meta_skipped} meta sessions")

    return all_tasks


def print_summary(tasks: list[TaskData], model: str):
    """Print extraction summary."""
    if not tasks:
        print(f"\n{model.upper()}: No tasks found")
        return

    print(f"\n{model.upper()} Tasks: {len(tasks)}")

    # Outcome distribution
    outcomes = {}
    for t in tasks:
        outcomes[t.outcome_category] = outcomes.get(t.outcome_category, 0) + 1
    print(f"  Outcomes: {outcomes}")

    # Scale metrics
    tasks_with_changes = [t for t in tasks if t.total_files_touched > 0]
    if tasks_with_changes:
        avg_files = sum(t.total_files_touched for t in tasks_with_changes) / len(tasks_with_changes)
        avg_lines_added = sum(t.total_lines_added for t in tasks_with_changes) / len(tasks_with_changes)
        print(f"  Avg files touched: {avg_files:.1f}")
        print(f"  Avg lines added: {avg_lines_added:.1f}")

    # Tool usage
    avg_tools = sum(len(t.tool_calls) for t in tasks) / len(tasks)
    print(f"  Avg tool calls: {avg_tools:.1f}")


def save_tasks(tasks: list[TaskData], output_file: Path):
    """Save tasks to JSON file."""
    data = [asdict(t) for t in tasks]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} tasks to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract comprehensive task data')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory (default: data)')
    parser.add_argument('--include-meta', action='store_true',
                        help='Include meta sessions (claude-investigations). Default: exclude')
    args = parser.parse_args()

    print("=" * 60)
    print("DEEP TASK EXTRACTION")
    print("=" * 60)
    if not args.include_meta:
        print("(Excluding meta sessions. Use --include-meta to include.)")

    for model in ['opus-4-5', 'opus-4-6']:
        sessions_file = args.data_dir / f'sessions-{model}.json'
        if not sessions_file.exists():
            print(f"\nSkipping {model}: {sessions_file} not found")
            continue

        tasks = extract_all_tasks(sessions_file, model, include_meta=args.include_meta)

        # Report meta vs non-meta counts
        meta_tasks = sum(1 for t in tasks if t.is_meta)
        real_tasks = len(tasks) - meta_tasks
        print(f"  {model}: {len(tasks)} total ({real_tasks} real, {meta_tasks} meta)")

        print_summary(tasks, model)

        output_file = args.data_dir / f'tasks-deep-{model}.json'
        save_tasks(tasks, output_file)


if __name__ == '__main__':
    main()
