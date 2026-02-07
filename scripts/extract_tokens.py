#!/usr/bin/env python3
"""
Token Usage Extraction for Model Comparison

Extracts per-task token usage from session JSONL files:
- Input/output/cache tokens from usage fields
- Thinking content (character count, estimated tokens)
- Text output content (character count)
- Cost estimates based on Anthropic API pricing

Outputs per-task token data that joins with classification data.
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Anthropic API pricing (per million tokens)
# https://docs.anthropic.com/en/docs/about-claude/models
PRICING = {
    'opus-4-5': {
        'input': 15.0,
        'output': 75.0,
        'cache_read': 1.875,
        'cache_write': 18.75,
    },
    'opus-4-6': {
        'input': 15.0,
        'output': 75.0,
        'cache_read': 1.875,
        'cache_write': 18.75,
    },
}

# Model alias resolution (same as collect_sessions.py)
MODEL_ALIASES = {
    'claude-fudge-eap-cc': 'opus-4-6',
}

# Patterns indicating system-generated user messages (same as extract_tasks.py)
import re
SKIP_PATTERNS = [
    r'^<local-command',
    r'^<command-name>',
    r'^<system-reminder>',
    r'^<task-notification>',
    r'^<teammate-message>',
    r'^\s*$',
]


def should_skip_message(text: str) -> bool:
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    return False


def extract_user_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                texts.append(block.get('text', ''))
            elif isinstance(block, str):
                texts.append(block)
        return ' '.join(texts)
    return ''


def is_tool_result(content) -> bool:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                return True
    return False


@dataclass
class TaskTokens:
    """Token usage data for a single task."""
    task_id: str
    session_id: str
    model: str
    project_path: str = ""

    # API-reported token counts (aggregated across all requests in task)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Content analysis
    thinking_chars: int = 0
    thinking_blocks: int = 0
    text_chars: int = 0
    text_blocks: int = 0
    tool_use_blocks: int = 0

    # Estimated token breakdown
    estimated_thinking_tokens: int = 0  # thinking_chars / 3
    estimated_text_tokens: int = 0  # text_chars / 3.5

    # Request-level data
    request_count: int = 0

    # Cost estimate (USD)
    estimated_cost: float = 0.0

    # Thinking metadata (from user messages)
    thinking_enabled: bool = False
    max_thinking_tokens: int = 0


def extract_tokens_from_session(file_path: Path, model: str) -> list[TaskTokens]:
    """Extract token usage per task from a session JSONL file."""
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
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []

    session_id = file_path.stem
    project_path = str(file_path.parent.name)
    tasks = []
    current_task = None
    task_counter = 0
    seen_request_ids = set()

    # Track thinking config from user messages
    current_thinking_enabled = False
    current_max_thinking = 0

    for msg in messages:
        msg_type = msg.get('type')

        if msg_type == 'user':
            content = msg.get('message', {}).get('content', [])

            # Check for thinking metadata
            thinking_meta = msg.get('thinkingMetadata', {})
            if thinking_meta:
                disabled = thinking_meta.get('disabled', False)
                current_thinking_enabled = not disabled
                current_max_thinking = thinking_meta.get('maxThinkingTokens', 0)

            if is_tool_result(content):
                continue

            user_text = extract_user_text(content)
            if should_skip_message(user_text):
                continue

            # Finalize previous task
            if current_task is not None:
                tasks.append(current_task)

            # Start new task
            task_counter += 1
            current_task = TaskTokens(
                task_id=f"{session_id}-task-{task_counter}",
                session_id=session_id,
                model=model,
                project_path=project_path,
                thinking_enabled=current_thinking_enabled,
                max_thinking_tokens=current_max_thinking,
            )
            seen_request_ids = set()

        elif msg_type == 'assistant' and current_task is not None:
            message = msg.get('message', {})
            content = message.get('content', [])
            usage = message.get('usage', {})
            request_id = msg.get('requestId', '')

            # Track unique requests
            if request_id and request_id not in seen_request_ids:
                seen_request_ids.add(request_id)
                current_task.request_count += 1

                # Usage is cumulative per request; take the values from each
                # new request (they represent that request's totals)
                input_tokens = usage.get('input_tokens', 0) or 0
                output_tokens = usage.get('output_tokens', 0) or 0
                cache_read = usage.get('cache_read_input_tokens', 0) or 0
                cache_write = usage.get('cache_creation_input_tokens', 0) or 0

                # Also check nested cache fields
                cache_creation = usage.get('cache_creation', {})
                if isinstance(cache_creation, dict):
                    cache_write += cache_creation.get('ephemeral_5m_input_tokens', 0) or 0
                    cache_write += cache_creation.get('ephemeral_1h_input_tokens', 0) or 0

                current_task.total_input_tokens += input_tokens
                current_task.cache_read_tokens += cache_read
                current_task.cache_write_tokens += cache_write

            # For output_tokens, we need the LAST value per requestId
            # (it's a cumulative streaming counter)
            # Store per-request and take max later
            if request_id:
                output_tokens = usage.get('output_tokens', 0) or 0
                # We'll track max output_tokens per request
                if not hasattr(current_task, '_request_output'):
                    current_task._request_output = {}
                current_task._request_output[request_id] = max(
                    current_task._request_output.get(request_id, 0),
                    output_tokens
                )

            # Analyze content blocks
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get('type', '')

                    if block_type == 'thinking':
                        thinking_text = block.get('thinking', '')
                        current_task.thinking_chars += len(thinking_text)
                        current_task.thinking_blocks += 1

                    elif block_type == 'text':
                        text = block.get('text', '')
                        current_task.text_chars += len(text)
                        current_task.text_blocks += 1

                    elif block_type == 'tool_use':
                        current_task.tool_use_blocks += 1

    # Finalize last task
    if current_task is not None:
        tasks.append(current_task)

    # Post-process: compute output tokens from per-request maxima
    for task in tasks:
        if hasattr(task, '_request_output'):
            task.total_output_tokens = sum(task._request_output.values())
            del task._request_output

        # Estimate thinking vs text tokens
        task.estimated_thinking_tokens = task.thinking_chars // 3
        task.estimated_text_tokens = task.text_chars // 4  # text is slightly more token-dense

        # Cost estimate
        pricing = PRICING.get(model, PRICING['opus-4-5'])
        cost = 0.0
        cost += (task.total_input_tokens / 1_000_000) * pricing['input']
        cost += (task.total_output_tokens / 1_000_000) * pricing['output']
        cost += (task.cache_read_tokens / 1_000_000) * pricing['cache_read']
        cost += (task.cache_write_tokens / 1_000_000) * pricing['cache_write']
        task.estimated_cost = round(cost, 4)

    return tasks


def extract_all_tokens(sessions_file: Path, model: str, include_meta: bool = False) -> list[TaskTokens]:
    """Extract tokens from all sessions."""
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    all_tasks = []
    meta_skipped = 0

    for i, session in enumerate(sessions):
        is_meta = session.get('is_meta', False)
        if not include_meta and is_meta:
            meta_skipped += 1
            continue

        file_path = Path(session['file_path'])
        if file_path.exists():
            tasks = extract_tokens_from_session(file_path, model)
            all_tasks.extend(tasks)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sessions)}] {len(all_tasks)} tasks extracted")

    if meta_skipped:
        print(f"  Skipped {meta_skipped} meta sessions")

    return all_tasks


def compute_aggregates(tasks: list[TaskTokens], classifications: dict) -> dict:
    """Compute aggregate token statistics, broken down by complexity and type."""
    results = {
        'overall': {},
        'by_complexity': {},
        'by_type': {},
    }

    def stats_for_group(group: list[TaskTokens]) -> dict:
        if not group:
            return {}
        n = len(group)
        total_input = sum(t.total_input_tokens for t in group)
        total_output = sum(t.total_output_tokens for t in group)
        total_thinking_chars = sum(t.thinking_chars for t in group)
        total_text_chars = sum(t.text_chars for t in group)
        total_cost = sum(t.estimated_cost for t in group)
        tasks_with_thinking = sum(1 for t in group if t.thinking_blocks > 0)

        return {
            'count': n,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_cost_usd': round(total_cost, 2),
            'avg_input_tokens': round(total_input / n),
            'avg_output_tokens': round(total_output / n),
            'avg_cost_usd': round(total_cost / n, 4),
            'avg_thinking_chars': round(total_thinking_chars / n),
            'avg_text_chars': round(total_text_chars / n),
            'thinking_ratio': round(tasks_with_thinking / n, 3),
            'avg_thinking_chars_when_used': round(
                total_thinking_chars / tasks_with_thinking
            ) if tasks_with_thinking else 0,
            'estimated_thinking_tokens': sum(t.estimated_thinking_tokens for t in group),
            'estimated_text_tokens': sum(t.estimated_text_tokens for t in group),
            'avg_requests_per_task': round(sum(t.request_count for t in group) / n, 1),
            'cache_read_tokens': sum(t.cache_read_tokens for t in group),
            'cache_write_tokens': sum(t.cache_write_tokens for t in group),
        }

    # Overall
    results['overall'] = stats_for_group(tasks)

    # By complexity
    complexity_groups = defaultdict(list)
    for t in tasks:
        cls = classifications.get(t.task_id, {})
        complexity = cls.get('complexity', 'unknown')
        complexity_groups[complexity].append(t)

    for complexity, group in sorted(complexity_groups.items()):
        results['by_complexity'][complexity] = stats_for_group(group)

    # By type
    type_groups = defaultdict(list)
    for t in tasks:
        cls = classifications.get(t.task_id, {})
        task_type = cls.get('type', 'unknown')
        type_groups[task_type].append(t)

    for task_type, group in sorted(type_groups.items()):
        results['by_type'][task_type] = stats_for_group(group)

    return results


def load_classifications(classified_file: Path) -> dict:
    """Load task classifications into a dict keyed by task_id."""
    if not classified_file.exists():
        return {}
    with open(classified_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    result = {}
    for t in tasks:
        task_id = t.get('task_id', '')
        cls = t.get('classification', {})
        result[task_id] = {
            'complexity': cls.get('complexity', 'unknown'),
            'type': cls.get('type', 'unknown'),
        }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract token usage from session logs')
    parser.add_argument('--dir', type=Path, default=None,
                        help='Comparison directory containing data/ and analysis/')
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--analysis-dir', type=Path, default=None)
    args = parser.parse_args()

    base = args.dir or Path(__file__).parent.parent
    data_dir = args.data_dir or base / 'data'
    analysis_dir = args.analysis_dir or base / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    all_results = {}

    for model in ['opus-4-5', 'opus-4-6']:
        sessions_file = data_dir / f'sessions-{model}.json'
        classified_file = data_dir / f'tasks-classified-{model}.json'

        if not sessions_file.exists():
            print(f"Skipping {model}: {sessions_file} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting tokens for {model}")
        print(f"{'='*60}")

        # Extract tokens
        tasks = extract_all_tokens(sessions_file, model)
        print(f"  Total tasks: {len(tasks)}")

        # Save raw per-task token data
        raw_file = data_dir / f'tokens-{model}.json'
        raw_data = [asdict(t) for t in tasks]
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2)
        print(f"  Saved raw data to {raw_file}")

        # Load classifications and compute aggregates
        classifications = load_classifications(classified_file)
        print(f"  Loaded {len(classifications)} classifications")

        aggregates = compute_aggregates(tasks, classifications)
        all_results[model] = aggregates

        # Print summary
        overall = aggregates['overall']
        print(f"\n  Overall:")
        print(f"    Tasks: {overall['count']}")
        print(f"    Avg input tokens/task: {overall['avg_input_tokens']:,}")
        print(f"    Avg output tokens/task: {overall['avg_output_tokens']:,}")
        print(f"    Avg cost/task: ${overall['avg_cost_usd']:.4f}")
        print(f"    Total cost: ${overall['total_cost_usd']:.2f}")
        print(f"    Thinking ratio: {overall['thinking_ratio']:.1%}")
        print(f"    Avg thinking chars (when used): {overall['avg_thinking_chars_when_used']:,}")
        print(f"    Avg text chars: {overall['avg_text_chars']:,}")

        print(f"\n  By complexity:")
        for c, stats in sorted(aggregates['by_complexity'].items()):
            print(f"    {c}: n={stats['count']}, "
                  f"avg_out={stats['avg_output_tokens']:,}, "
                  f"avg_cost=${stats['avg_cost_usd']:.4f}, "
                  f"thinking={stats['thinking_ratio']:.0%}")

        print(f"\n  By type:")
        for t, stats in sorted(aggregates['by_type'].items()):
            if stats['count'] >= 5:  # Only show types with enough data
                print(f"    {t}: n={stats['count']}, "
                      f"avg_out={stats['avg_output_tokens']:,}, "
                      f"avg_cost=${stats['avg_cost_usd']:.4f}, "
                      f"thinking={stats['thinking_ratio']:.0%}")

    # Save aggregate results
    output_file = analysis_dir / 'token-analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregate analysis to {output_file}")


if __name__ == '__main__':
    main()
