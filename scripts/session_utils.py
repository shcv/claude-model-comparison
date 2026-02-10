#!/usr/bin/env python3
"""Shared JSONL parsing utilities for session analysis."""

import json
import re
from pathlib import Path


# Patterns indicating system-generated user messages to skip
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
    r'^Implement the following plan:',
    r'^\[Request interrupted',
    r'^<summary>',
    r'^Analysis:',
]


def extract_user_text(content):
    """Extract text from user message content.

    Handles both string content and list-of-blocks content.
    """
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


def should_skip_message(text):
    """Check if message should be skipped based on SKIP_PATTERNS."""
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    return False


def is_tool_result(content):
    """Check if user message content is a tool result."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                return True
    return False


def is_system_generated(text):
    """Check if text is system-generated content, not actual user sentiment."""
    for pattern in SYSTEM_CONTENT_PATTERNS:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False


def iter_messages(file_path):
    """Read a JSONL session file, yielding parsed message dicts.

    Skips blank lines and JSON decode errors.
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_tasks(messages):
    """Canonical task boundary iterator.

    Yields (user_prompt, task_messages) tuples based on the extract_tasks.py
    heuristic: each non-skipped, non-tool-result user message starts a new task.

    Args:
        messages: iterable of parsed message dicts (from iter_messages or a list)

    Yields:
        (user_prompt, task_messages) where user_prompt is the text that started
        the task and task_messages is the list of all messages in that task
        (including the initial user message).
    """
    current_prompt = None
    current_messages = []

    for msg in messages:
        msg_type = msg.get('type')

        if msg_type == 'user':
            content = msg.get('message', {}).get('content', [])

            # Skip tool results
            if is_tool_result(content):
                if current_prompt is not None:
                    current_messages.append(msg)
                continue

            user_text = extract_user_text(content)

            # Skip system messages
            if should_skip_message(user_text):
                if current_prompt is not None:
                    current_messages.append(msg)
                continue

            # New real user message: yield previous task if any
            if current_prompt is not None:
                yield current_prompt, current_messages

            current_prompt = user_text
            current_messages = [msg]
        else:
            if current_prompt is not None:
                current_messages.append(msg)

    # Yield final task
    if current_prompt is not None:
        yield current_prompt, current_messages
