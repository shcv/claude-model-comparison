#!/usr/bin/env python3
"""
Deep Task Extraction for Model Comparison

Extracts comprehensive task data including:
- Full user prompt and context
- Agent actions with file/line change metrics
- User's NEXT message to determine actual outcome
- Git-style stats: files read, written, lines changed

Canonical mode (--canonical) adds:
- Edit events with success tracking
- Token usage and cost estimates
- Behavioral signals (planning, subagents, parallelism)
- Compaction events
- Error context from tool results
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models
from session_utils import (
    SKIP_PATTERNS, SYSTEM_CONTENT_PATTERNS,
    extract_user_text, should_skip_message, is_tool_result, is_system_generated,
)


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


# ---------------------------------------------------------------------------
# Canonical task: all signals in one record
# ---------------------------------------------------------------------------

# Anthropic API pricing (per million tokens)
PRICING = {
    'opus-4-5': {'input': 15.0, 'output': 75.0, 'cache_read': 1.875, 'cache_write': 18.75},
    'opus-4-6': {'input': 15.0, 'output': 75.0, 'cache_read': 1.875, 'cache_write': 18.75},
}

# Error indicators in tool results (from analyze_edits.py)
ERROR_PATTERNS = [
    r'error', r'failed', r'FAIL', r'traceback', r'exception',
    r'SyntaxError', r'TypeError', r'NameError', r'ImportError',
    r'is_error.*true', r'tool_use_error', r'exit code [1-9]',
    r'compilation failed', r'build failed',
]


@dataclass
class CanonicalTask:
    """Single canonical task record with all signals for downstream analysis."""
    task_id: str
    session_id: str
    model: str

    # User request
    user_prompt: str
    user_prompt_full: str

    # Agent work (same as TaskData)
    tool_calls: list = field(default_factory=list)
    tool_sequence: str = ""
    files_read: list = field(default_factory=list)
    files_written: list = field(default_factory=list)
    files_edited: list = field(default_factory=list)

    # Scale metrics
    total_files_touched: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    bash_commands: list = field(default_factory=list)

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0
    msg_index_start: int = 0
    msg_index_end: int = 0

    # Outcome
    next_user_message: str = ""
    outcome_category: str = ""
    outcome_evidence: str = ""

    # Context
    project_path: str = ""
    is_meta: bool = False
    task_summary: str = ""

    # --- Edit events ---
    edit_events: list = field(default_factory=list)
    # Each: {tool_use_id, tool_name, file_path, old_string, new_string,
    #         replace_all, succeeded, msg_index, timestamp}

    # --- Token usage ---
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_chars: int = 0
    thinking_blocks: int = 0
    text_chars: int = 0
    text_blocks: int = 0
    tool_use_chars: int = 0
    edit_content_chars: int = 0
    write_content_chars: int = 0
    bash_command_chars: int = 0
    agent_prompt_chars: int = 0
    search_chars: int = 0
    estimated_cost: float = 0.0
    request_count: int = 0

    # --- Behavioral signals ---
    used_planning: bool = False
    used_subagents: bool = False
    subagent_count: int = 0
    subagent_types: list = field(default_factory=list)
    parallel_tool_messages: int = 0
    run_in_background_count: int = 0

    # --- Timing details ---
    request_timings: list = field(default_factory=list)
    # Each: {request_id, first_timestamp, last_timestamp, has_tool_use, tool_count}
    turn_duration_ms: int = 0

    # --- Compaction context ---
    compaction_events_before: list = field(default_factory=list)
    # Each: {trigger, pre_tokens, msg_index, timestamp}

    # --- Error context ---
    errors: list = field(default_factory=list)
    # Each: {msg_index, text, is_tool_error}

    # --- Data cleaning ---
    exclude_reason: str = ""
    # Non-empty string means task should be excluded from primary analysis.
    # Values: "slash_command", "system_continuation", "empty_continuation",
    #         "no_response_interrupt"
    flags: list = field(default_factory=list)
    # Informational flags (task is still included). Values:
    # "meta", "no_project", "interrupted", "post_compaction"


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

## SKIP_PATTERNS, SYSTEM_CONTENT_PATTERNS, extract_user_text, should_skip_message,
## is_tool_result, is_system_generated are imported from session_utils above.


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


def extract_canonical_from_session(file_path: Path, model: str) -> list[CanonicalTask]:
    """Extract canonical tasks from a session JSONL, collecting all signals in one pass.

    Combines logic from extract_tasks, analyze_edits, extract_tokens,
    analyze_behavior, and analyze_compaction into a single traversal.
    """
    messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    session_id = file_path.stem
    project_path = str(file_path.parent.name)

    tasks = []
    current_task: Optional[CanonicalTask] = None
    task_counter = 0

    # Token tracking: output_tokens is cumulative per request, track max per requestId
    seen_request_ids: set[str] = set()
    request_output_max: dict[str, int] = {}  # requestId -> max output_tokens

    # Request timing tracking: per-request first/last timestamps and tool counts
    request_timing_data: dict[str, dict] = {}  # requestId -> timing info

    # Edit tracking: pending edits waiting for tool_result
    pending_edits: dict[str, dict] = {}  # tool_use_id -> edit event dict

    # Compaction events seen so far (assigned to the next task that starts after them)
    pending_compactions: list[dict] = []

    def finalize_task(next_user_text: str = "", is_session_end: bool = False):
        nonlocal current_task, seen_request_ids, request_output_max, request_timing_data
        if current_task is None:
            return

        # Compile request timings
        current_task.request_timings = list(request_timing_data.values())

        # Outcome categorization
        if is_session_end:
            current_task.outcome_category = 'session_end'
            current_task.outcome_evidence = 'session terminated'
        elif next_user_text:
            current_task.next_user_message = next_user_text[:500]
            category, evidence = categorize_user_response(next_user_text)
            current_task.outcome_category = category
            current_task.outcome_evidence = evidence

        # File totals
        current_task.total_files_touched = len(set(
            current_task.files_read + current_task.files_written + current_task.files_edited
        ))

        # Tool sequence (deduped consecutive)
        tool_names = [t['name'] for t in current_task.tool_calls]
        deduped = []
        for name in tool_names:
            if not deduped or deduped[-1] != name:
                deduped.append(name)
        current_task.tool_sequence = '\u2192'.join(deduped[:10])

        # Summary
        prompt_preview = current_task.user_prompt[:60].replace('\n', ' ')
        current_task.task_summary = (
            f"{prompt_preview}... ({len(current_task.tool_calls)} tools, "
            f"{current_task.total_files_touched} files)"
        )

        # Finalize output tokens from per-request maxima
        current_task.output_tokens = sum(request_output_max.values())

        # Cost estimate
        pricing = PRICING.get(model, PRICING.get('opus-4-5', {}))
        if pricing:
            cost = 0.0
            cost += (current_task.input_tokens / 1_000_000) * pricing['input']
            cost += (current_task.output_tokens / 1_000_000) * pricing['output']
            cost += (current_task.cache_read_tokens / 1_000_000) * pricing['cache_read']
            cost += (current_task.cache_write_tokens / 1_000_000) * pricing['cache_write']
            current_task.estimated_cost = round(cost, 4)

        # Behavioral: dedupe subagent_types
        current_task.used_subagents = current_task.subagent_count > 0

        tasks.append(current_task)
        current_task = None
        seen_request_ids = set()
        request_output_max = {}
        request_timing_data = {}

    for idx, msg in enumerate(messages):
        msg_type = msg.get('type')
        timestamp = msg.get('timestamp', '')

        # --- Turn duration (system messages) ---
        if msg_type == 'system' and msg.get('subtype') == 'turn_duration':
            duration_ms = msg.get('durationMs', 0) or 0
            if current_task is not None and duration_ms > 0:
                current_task.turn_duration_ms += duration_ms
            continue

        # --- Compaction events (system messages) ---
        if msg_type == 'system' and msg.get('subtype') == 'compact_boundary':
            meta = msg.get('compactMetadata', {})
            event = {
                'trigger': meta.get('trigger', 'unknown'),
                'pre_tokens': meta.get('preTokens', 0),
                'msg_index': idx,
                'timestamp': timestamp,
            }
            if current_task is not None:
                # Compaction happened during a task — attach to current task
                current_task.compaction_events_before.append(event)
            else:
                # Compaction before any task starts — buffer for next task
                pending_compactions.append(event)
            continue

        # --- User messages ---
        if msg_type == 'user':
            content = msg.get('message', {}).get('content', [])

            # Process tool results (edit success/failure, errors)
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'tool_result':
                        tool_use_id = block.get('tool_use_id', '')
                        is_error = block.get('is_error', False)

                        # Match pending edits
                        if tool_use_id in pending_edits:
                            edit_event = pending_edits.pop(tool_use_id)
                            edit_event['succeeded'] = not is_error
                            if current_task is not None:
                                current_task.edit_events.append(edit_event)

                        # Extract error context
                        result_text = ''
                        rc = block.get('content', '')
                        if isinstance(rc, str):
                            result_text = rc
                        elif isinstance(rc, list):
                            for rb in rc:
                                if isinstance(rb, dict) and rb.get('type') == 'text':
                                    result_text += rb.get('text', '')

                        is_tool_error = bool(is_error)
                        has_error_pattern = any(
                            re.search(p, result_text, re.IGNORECASE)
                            for p in ERROR_PATTERNS
                        ) if result_text else False

                        if (is_tool_error or has_error_pattern) and current_task is not None:
                            current_task.errors.append({
                                'msg_index': idx,
                                'text': result_text[:200],
                                'is_tool_error': is_tool_error,
                            })

            # Skip tool-result-only messages for task boundary detection
            if is_tool_result(content):
                continue

            user_text = extract_user_text(content)
            if should_skip_message(user_text):
                continue

            # This user message ends the previous task
            if current_task is not None:
                current_task.msg_index_end = idx - 1
                finalize_task(next_user_text=user_text)

            # Start new task
            task_counter += 1
            current_task = CanonicalTask(
                task_id=f"{session_id}-task-{task_counter}",
                session_id=session_id,
                model=model,
                user_prompt=user_text[:500],
                user_prompt_full=user_text,
                start_time=timestamp,
                project_path=project_path,
                msg_index_start=idx,
            )

            # Attach buffered compaction events
            if pending_compactions:
                current_task.compaction_events_before = pending_compactions[:]
                pending_compactions.clear()

        # --- Assistant messages ---
        elif msg_type == 'assistant' and current_task is not None:
            current_task.end_time = timestamp
            current_task.msg_index_end = idx
            message = msg.get('message', {})
            content = message.get('content', [])
            usage = message.get('usage', {})
            request_id = msg.get('requestId', '')

            # --- Request timing tracking ---
            if request_id:
                if request_id not in request_timing_data:
                    request_timing_data[request_id] = {
                        'request_id': request_id,
                        'first_timestamp': timestamp,
                        'last_timestamp': timestamp,
                        'has_tool_use': False,
                        'tool_count': 0,
                    }
                else:
                    request_timing_data[request_id]['last_timestamp'] = timestamp

            # --- Token usage ---
            if request_id and request_id not in seen_request_ids:
                seen_request_ids.add(request_id)
                current_task.request_count += 1

                input_tokens = usage.get('input_tokens', 0) or 0
                cache_read = usage.get('cache_read_input_tokens', 0) or 0
                cache_write = usage.get('cache_creation_input_tokens', 0) or 0

                # Nested cache fields
                cache_creation = usage.get('cache_creation', {})
                if isinstance(cache_creation, dict):
                    cache_write += cache_creation.get('ephemeral_5m_input_tokens', 0) or 0
                    cache_write += cache_creation.get('ephemeral_1h_input_tokens', 0) or 0

                current_task.input_tokens += input_tokens
                current_task.cache_read_tokens += cache_read
                current_task.cache_write_tokens += cache_write

            # Track max output_tokens per request (cumulative streaming counter)
            if request_id:
                output_tokens = usage.get('output_tokens', 0) or 0
                request_output_max[request_id] = max(
                    request_output_max.get(request_id, 0), output_tokens
                )

            # --- Content block analysis ---
            if not isinstance(content, list):
                continue

            tool_uses_in_msg = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get('type', '')

                # Thinking content
                if block_type == 'thinking':
                    thinking_text = block.get('thinking', '')
                    current_task.thinking_chars += len(thinking_text)
                    current_task.thinking_blocks += 1

                # Text content
                elif block_type == 'text':
                    text = block.get('text', '')
                    current_task.text_chars += len(text)
                    current_task.text_blocks += 1

                # Tool use
                elif block_type == 'tool_use':
                    tool_uses_in_msg.append(block)
                    tool_info = extract_tool_info(block)
                    tool_info['timestamp'] = timestamp
                    tool_info['tool_use_id'] = block.get('id', '')
                    current_task.tool_calls.append(tool_info)

                    # Update request timing tool counts
                    if request_id and request_id in request_timing_data:
                        request_timing_data[request_id]['has_tool_use'] = True
                        request_timing_data[request_id]['tool_count'] += 1

                    name = tool_info.get('name')
                    file_path_str = tool_info.get('file', '')
                    inp = block.get('input', {})

                    # Tool input char breakdown
                    if isinstance(inp, dict):
                        inp_chars = sum(len(str(v)) for v in inp.values())
                        current_task.tool_use_chars += inp_chars
                        if name == 'Edit':
                            current_task.edit_content_chars += len(inp.get('old_string', '')) + len(inp.get('new_string', ''))
                        elif name in ('Write', 'NotebookEdit'):
                            current_task.write_content_chars += len(inp.get('content', '') or inp.get('new_source', ''))
                        elif name == 'Bash':
                            current_task.bash_command_chars += len(inp.get('command', ''))
                        elif name == 'Task':
                            current_task.agent_prompt_chars += len(inp.get('prompt', ''))
                        elif name in ('Grep', 'Glob'):
                            current_task.search_chars += len(inp.get('pattern', '')) + len(inp.get('path', ''))
                    tool_use_id = block.get('id', '')

                    # File tracking (same as TaskData)
                    if name == 'Read' and file_path_str:
                        current_task.files_read.append(file_path_str)
                    elif name == 'Write' and file_path_str:
                        current_task.files_written.append(file_path_str)
                        current_task.total_lines_added += tool_info.get('lines', 0)
                        # Track as pending edit
                        pending_edits[tool_use_id] = {
                            'tool_use_id': tool_use_id,
                            'tool_name': 'Write',
                            'file_path': file_path_str,
                            'old_string': '',
                            'new_string': '',  # Don't store full content
                            'replace_all': False,
                            'succeeded': False,
                            'msg_index': idx,
                            'timestamp': timestamp,
                        }
                    elif name == 'Edit' and file_path_str:
                        current_task.files_edited.append(file_path_str)
                        current_task.total_lines_added += tool_info.get('lines_added', 0)
                        current_task.total_lines_removed += tool_info.get('lines_removed', 0)
                        # Track as pending edit
                        pending_edits[tool_use_id] = {
                            'tool_use_id': tool_use_id,
                            'tool_name': 'Edit',
                            'file_path': file_path_str,
                            'old_string': inp.get('old_string', '')[:200],
                            'new_string': inp.get('new_string', '')[:200],
                            'replace_all': inp.get('replace_all', False),
                            'succeeded': False,
                            'msg_index': idx,
                            'timestamp': timestamp,
                        }
                    elif name == 'Bash':
                        cmd = tool_info.get('command', '')
                        if cmd:
                            current_task.bash_commands.append(cmd)
                        if inp.get('run_in_background', False):
                            current_task.run_in_background_count += 1

                    # Behavioral: planning
                    elif name == 'EnterPlanMode':
                        current_task.used_planning = True

                    # Behavioral: subagents
                    elif name == 'Task':
                        current_task.subagent_count += 1
                        subagent_type = inp.get('subagent_type', 'unknown')
                        current_task.subagent_types.append(subagent_type)
                        if inp.get('run_in_background', False):
                            current_task.run_in_background_count += 1

            # Parallel tool messages
            if len(tool_uses_in_msg) > 1:
                current_task.parallel_tool_messages += 1

    # Finalize last task
    if current_task is not None:
        current_task.msg_index_end = len(messages) - 1
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


# --- Data cleaning patterns ---

SLASH_COMMAND_PATTERN = re.compile(
    r'^<command-(?:name|message)>|^/\w+',
)

BARE_CONTINUATION_PROMPTS = {
    'continue', 'ok', 'yes', 'go ahead', 'proceed', 'sure',
    'sounds good', 'yep', 'yeah', 'ok.', 'yes.', 'sure.',
}


def classify_data_cleaning(task: 'CanonicalTask') -> None:
    """Set exclude_reason and flags on a canonical task based on data cleaning rules.

    Exclusion rules (task removed from primary analysis):
    1. Slash commands: user_prompt is a /command or <command-name> tag
    2. System continuations: outcome_category == 'system_continuation'
       OR user_prompt matches SYSTEM_CONTENT_PATTERNS
    3. Empty bare continuations: bare ack ("ok","continue") with 0 tools and <5s
    4. No-response interrupts: 0 tools, 0 duration, session_end outcome

    Flags (informational, task still included):
    - meta: from claude-investigations project
    - no_project: session run from home dir (no specific project)
    - interrupted: user interrupted the agent mid-work
    - post_compaction: task started after a compaction event
    """
    prompt = (task.user_prompt or '').strip()
    prompt_lower = prompt.lower().strip().rstrip('.')

    # --- Exclusion rules ---

    # 1. Slash commands
    if SLASH_COMMAND_PATTERN.match(prompt):
        task.exclude_reason = 'slash_command'
        return

    # 2. System continuations
    if task.outcome_category == 'system_continuation':
        task.exclude_reason = 'system_continuation'
        return
    if is_system_generated(prompt):
        task.exclude_reason = 'system_continuation'
        return

    # 3. Empty bare continuations (no work done)
    tool_count = len(task.tool_calls)
    duration = task.duration_seconds or 0
    if prompt_lower in BARE_CONTINUATION_PROMPTS and tool_count == 0 and duration < 5:
        task.exclude_reason = 'empty_continuation'
        return

    # 4. No-response interrupts
    if (tool_count == 0 and duration == 0
            and task.outcome_category == 'session_end'):
        task.exclude_reason = 'no_response_interrupt'
        return

    # --- Flags (non-exclusive) ---

    if task.is_meta:
        task.flags.append('meta')

    if not task.project_path or task.project_path == '-home-shcv':
        task.flags.append('no_project')

    if '[Request interrupted' in prompt:
        task.flags.append('interrupted')

    if task.compaction_events_before:
        task.flags.append('post_compaction')


# --- Plan mode continuation merging ---

_PLAN_IMPL_RE = re.compile(r'^Implement the following plan:', re.IGNORECASE)
_INTERRUPTED_RE = re.compile(r'^\[Request interrupted')


def _merge_task_into(target: CanonicalTask, source: CanonicalTask) -> None:
    """Merge source task's data into target task.

    Combines tool calls, edit events, tokens, files, duration, etc.
    Target keeps its identity (task_id, user_prompt) but gains source's work.
    """
    # Lists: extend
    target.tool_calls.extend(source.tool_calls)
    target.files_read.extend(source.files_read)
    target.files_written.extend(source.files_written)
    target.files_edited.extend(source.files_edited)
    target.bash_commands.extend(source.bash_commands)
    target.edit_events.extend(source.edit_events)
    target.request_timings.extend(source.request_timings)
    target.compaction_events_before.extend(source.compaction_events_before)
    target.errors.extend(source.errors)
    target.subagent_types.extend(source.subagent_types)

    # Scalars: sum
    target.total_lines_added += source.total_lines_added
    target.total_lines_removed += source.total_lines_removed
    target.input_tokens += source.input_tokens
    target.output_tokens += source.output_tokens
    target.cache_read_tokens += source.cache_read_tokens
    target.cache_write_tokens += source.cache_write_tokens
    target.thinking_chars += source.thinking_chars
    target.thinking_blocks += source.thinking_blocks
    target.text_chars += source.text_chars
    target.text_blocks += source.text_blocks
    target.tool_use_chars += source.tool_use_chars
    target.edit_content_chars += source.edit_content_chars
    target.write_content_chars += source.write_content_chars
    target.bash_command_chars += source.bash_command_chars
    target.agent_prompt_chars += source.agent_prompt_chars
    target.search_chars += source.search_chars
    target.estimated_cost += round(source.estimated_cost, 4)
    target.request_count += source.request_count
    target.subagent_count += source.subagent_count
    target.parallel_tool_messages += source.parallel_tool_messages
    target.run_in_background_count += source.run_in_background_count
    target.turn_duration_ms += source.turn_duration_ms

    # Booleans: OR
    target.used_planning = target.used_planning or source.used_planning
    target.used_subagents = target.used_subagents or source.used_subagents

    # Time range: extend to cover both tasks
    if source.end_time and (not target.end_time or source.end_time > target.end_time):
        target.end_time = source.end_time
    if source.start_time and (not target.start_time or source.start_time < target.start_time):
        target.start_time = source.start_time
    target.msg_index_end = max(target.msg_index_end, source.msg_index_end)

    # Recompute derived fields
    target.total_files_touched = len(set(
        target.files_read + target.files_written + target.files_edited
    ))
    if target.start_time and target.end_time:
        try:
            start = datetime.fromisoformat(target.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(target.end_time.replace('Z', '+00:00'))
            target.duration_seconds = (end - start).total_seconds()
        except ValueError:
            target.duration_seconds += source.duration_seconds

    # Recompute tool_sequence
    tool_names = [t['name'] for t in target.tool_calls]
    deduped = []
    for name in tool_names:
        if not deduped or deduped[-1] != name:
            deduped.append(name)
    target.tool_sequence = '\u2192'.join(deduped[:10])

    # Merge flags (dedupe)
    for flag in source.flags:
        if flag not in target.flags:
            target.flags.append(flag)


def merge_plan_continuations(tasks: list[CanonicalTask]) -> list[CanonicalTask]:
    """Merge plan-mode continuation chains into their parent tasks.

    When plan mode is used, the implementation phase creates a separate 'task'
    starting with "Implement the following plan:". This gets excluded as
    system_continuation but contains all the actual editing work. This function
    merges these back.

    Patterns handled:
    1. parent(used_planning) -> [Request interrupted...] -> Implement plan
       -> all merged into parent
    2. [Request interrupted...] -> Implement plan (no parent in session)
       -> interrupt absorbed, plan implementation promoted to included task
    """
    from collections import defaultdict

    session_tasks = defaultdict(list)
    for t in tasks:
        session_tasks[t.session_id].append(t)

    absorbed = set()  # task_ids absorbed into another task
    merge_count = 0

    for sid, stasks in session_tasks.items():
        stasks.sort(key=lambda t: t.msg_index_start)

        for i, task in enumerate(stasks):
            prompt = (task.user_prompt or '').strip()
            if not _PLAN_IMPL_RE.match(prompt):
                continue

            # Found a plan implementation. Scan backwards for parent/interrupts.
            parent = None
            interrupts = []

            for j in range(i - 1, -1, -1):
                prev = stasks[j]
                if prev.task_id in absorbed:
                    continue
                prev_prompt = (prev.user_prompt or '').strip()

                if _INTERRUPTED_RE.match(prev_prompt):
                    interrupts.append(prev)
                    continue

                # Real task — merge into it if it used planning
                if prev.used_planning and not prev.exclude_reason:
                    parent = prev
                break

            if parent is not None:
                # Merge interrupts and implementation into parent
                for intr in interrupts:
                    _merge_task_into(parent, intr)
                    absorbed.add(intr.task_id)
                _merge_task_into(parent, task)
                absorbed.add(task.task_id)
            else:
                # No parent in session — promote plan implementation
                for intr in interrupts:
                    _merge_task_into(task, intr)
                    absorbed.add(intr.task_id)
                task.used_planning = True
                task.exclude_reason = ''

            merge_count += 1

    result = [t for t in tasks if t.task_id not in absorbed]

    if merge_count:
        print(f"  Merged {merge_count} plan continuation(s)")

    return result


def extract_all_canonical(sessions_file: Path, model: str,
                          include_meta: bool = True) -> list[CanonicalTask]:
    """Extract canonical tasks from all sessions in a sessions JSON file."""
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    all_tasks = []
    meta_skipped = 0
    for i, session in enumerate(sessions):
        is_meta = session.get('is_meta', False)
        if not include_meta and is_meta:
            meta_skipped += 1
            continue

        fp = Path(session['file_path'])
        if fp.exists():
            tasks = extract_canonical_from_session(fp, model)
            for task in tasks:
                task.is_meta = is_meta
                if not task.project_path and 'project_path' in session:
                    task.project_path = session['project_path']
            all_tasks.extend(tasks)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sessions)}] {len(all_tasks)} tasks extracted")

    if meta_skipped:
        print(f"  Skipped {meta_skipped} meta sessions")

    # Apply data cleaning classification
    for task in all_tasks:
        classify_data_cleaning(task)

    # Merge plan-mode continuations back into parent tasks
    all_tasks = merge_plan_continuations(all_tasks)

    excluded = sum(1 for t in all_tasks if t.exclude_reason)
    if excluded:
        reasons = {}
        for t in all_tasks:
            if t.exclude_reason:
                reasons[t.exclude_reason] = reasons.get(t.exclude_reason, 0) + 1
        print(f"  Data cleaning: {excluded} excluded ({reasons})")

    return all_tasks


def print_canonical_summary(tasks: list[CanonicalTask], model: str):
    """Print extraction summary for canonical tasks."""
    if not tasks:
        print(f"\n{model.upper()}: No tasks found")
        return

    n = len(tasks)
    excluded = sum(1 for t in tasks if t.exclude_reason)
    included = n - excluded
    flagged = sum(1 for t in tasks if t.flags and not t.exclude_reason)
    print(f"\n{model.upper()} Canonical Tasks: {n} ({included} included, {excluded} excluded, {flagged} flagged)")

    # Outcome distribution
    outcomes = {}
    for t in tasks:
        outcomes[t.outcome_category] = outcomes.get(t.outcome_category, 0) + 1
    print(f"  Outcomes: {outcomes}")

    # Token summary
    total_cost = sum(t.estimated_cost for t in tasks)
    avg_input = sum(t.input_tokens for t in tasks) / n if n else 0
    avg_output = sum(t.output_tokens for t in tasks) / n if n else 0
    print(f"  Avg input tokens: {avg_input:,.0f}, Avg output tokens: {avg_output:,.0f}")
    print(f"  Total estimated cost: ${total_cost:.2f}")

    # Edit events
    total_edits = sum(len(t.edit_events) for t in tasks)
    succeeded = sum(1 for t in tasks for e in t.edit_events if e.get('succeeded'))
    print(f"  Edit events: {total_edits} ({succeeded} succeeded)")

    # Behavioral
    planning = sum(1 for t in tasks if t.used_planning)
    subagents = sum(1 for t in tasks if t.used_subagents)
    parallel = sum(1 for t in tasks if t.parallel_tool_messages > 0)
    print(f"  Planning: {planning}, Subagents: {subagents}, Parallel: {parallel}")

    # Compaction
    compacted = sum(1 for t in tasks if t.compaction_events_before)
    print(f"  Tasks with prior compaction: {compacted}")

    # Errors
    errored = sum(1 for t in tasks if t.errors)
    total_errors = sum(len(t.errors) for t in tasks)
    print(f"  Tasks with errors: {errored}, Total errors: {total_errors}")


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
    parser.add_argument('--include-meta', action='store_true', default=True,
                        help='Include meta sessions (claude-investigations). Default: include')
    parser.add_argument('--exclude-meta', action='store_false', dest='include_meta',
                        help='Exclude meta sessions (claude-investigations)')
    parser.add_argument('--canonical', action='store_true',
                        help='Produce canonical task files with all signals')
    args = parser.parse_args()

    print("=" * 60)
    if args.canonical:
        print("CANONICAL TASK EXTRACTION")
    else:
        print("DEEP TASK EXTRACTION")
    print("=" * 60)
    if not args.include_meta:
        print("(Excluding meta sessions. Use --include-meta to include.)")
    else:
        print("(Including meta sessions. Use --exclude-meta to exclude.)")

    models = discover_models(args.data_dir)
    if not models:
        print(f"No sessions-*.json files found in {args.data_dir}")
        return

    print(f"Found {len(models)} model(s): {', '.join(models)}")

    for model in models:
        sessions_file = args.data_dir / f'sessions-{model}.json'
        if not sessions_file.exists():
            print(f"\nSkipping {model}: {sessions_file} not found")
            continue

        if args.canonical:
            tasks = extract_all_canonical(
                sessions_file, model, include_meta=args.include_meta
            )

            meta_tasks = sum(1 for t in tasks if t.is_meta)
            real_tasks = len(tasks) - meta_tasks
            print(f"  {model}: {len(tasks)} total ({real_tasks} real, {meta_tasks} meta)")

            print_canonical_summary(tasks, model)

            output_file = args.data_dir / f'tasks-canonical-{model}.json'
            save_tasks(tasks, output_file)
        else:
            tasks = extract_all_tasks(
                sessions_file, model, include_meta=args.include_meta
            )

            meta_tasks = sum(1 for t in tasks if t.is_meta)
            real_tasks = len(tasks) - meta_tasks
            print(f"  {model}: {len(tasks)} total ({real_tasks} real, {meta_tasks} meta)")

            print_summary(tasks, model)

            output_file = args.data_dir / f'tasks-canonical-{model}.json'
            save_tasks(tasks, output_file)


if __name__ == '__main__':
    main()
