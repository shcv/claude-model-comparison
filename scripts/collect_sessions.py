#!/usr/bin/env python3
"""
Session Collector for Opus 4.5 vs Opus 4.6 Model Comparison

Scans ~/.claude/projects/ for session files, extracts model information,
and groups sessions by dominant model (opus-4-5 vs opus-4-6).
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Model name mapping
MODEL_NAMES = {
    'opus-4-5': 'Opus 4.5',
    'opus-4-6': 'Opus 4.6',
}
MODEL_IDS = {
    'opus-4-5': 'claude-opus-4-5-20251101',
    'opus-4-6': 'claude-opus-4-6',
}
# EAP codenames that map to the same model
MODEL_ALIASES = {
    'claude-fudge-eap-cc': 'opus-4-6',
}


@dataclass
class SessionMetadata:
    session_id: str
    file_path: str
    model: str              # Dominant model
    model_counts: dict      # All models with counts
    start_time: str
    end_time: str
    duration_minutes: float
    user_message_count: int
    assistant_message_count: int
    tool_call_count: int
    tools_used: dict        # Tool -> count
    project_path: str = ""  # Decoded project path (e.g., /home/shcv/projects/grocery-list)
    is_meta: bool = False   # True if project is claude-investigations (meta/self-referential)


def decode_project_dir(dir_name: str) -> str:
    """Return the raw project directory name as an identifier.

    Claude project dirs use the encoded path format (e.g.,
    -home-shcv-projects-grocery-list). Since dash is ambiguous
    (path separator vs literal), we keep the raw name for matching.
    """
    return dir_name


def is_meta_project(dir_name: str) -> bool:
    """Check if a project is meta/self-referential (this analysis itself).

    Matches on the raw encoded dir name, so 'claude-investigations'
    will match -home-shcv-projects-claude-investigations.
    """
    return 'claude-investigations' in dir_name


def parse_session_file(file_path: Path) -> Optional[SessionMetadata]:
    """Parse a JSONL session file and extract metadata."""
    model_counts = defaultdict(int)
    tool_counts = defaultdict(int)
    user_messages = 0
    assistant_messages = 0
    tool_calls = 0
    timestamps = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract timestamp
                if 'timestamp' in obj:
                    timestamps.append(obj['timestamp'])

                # Check message type and extract model
                msg_type = obj.get('type')

                if msg_type == 'user':
                    user_messages += 1
                elif msg_type == 'assistant':
                    assistant_messages += 1
                    # Model is in message.model
                    message = obj.get('message', {})
                    model = message.get('model', '')
                    if model:
                        # Resolve EAP codenames to canonical model IDs
                        model = MODEL_ALIASES.get(model, model)
                        model_counts[model] += 1

                    # Count tool use blocks
                    content = message.get('content', [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'tool_use':
                                tool_calls += 1
                                tool_name = block.get('name', 'unknown')
                                tool_counts[tool_name] += 1

        # Skip empty or minimal sessions
        if not timestamps or assistant_messages < 1:
            return None

        # Determine dominant model
        if not model_counts:
            return None

        dominant_model = max(model_counts.items(), key=lambda x: x[1])[0]

        # Calculate duration
        timestamps.sort()
        try:
            start = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
            end = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            duration = (end - start).total_seconds() / 60
        except (ValueError, IndexError):
            duration = 0

        session_id = file_path.stem

        return SessionMetadata(
            session_id=session_id,
            file_path=str(file_path),
            model=dominant_model,
            model_counts=dict(model_counts),
            start_time=timestamps[0] if timestamps else '',
            end_time=timestamps[-1] if timestamps else '',
            duration_minutes=round(duration, 2),
            user_message_count=user_messages,
            assistant_message_count=assistant_messages,
            tool_call_count=tool_calls,
            tools_used=dict(tool_counts)
        )
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def collect_sessions(days_back: int = 7, exclude_projects: list[str] = None) -> dict:
    """Collect sessions from the last N days, grouped by model family.

    Args:
        days_back: How many days back to scan
        exclude_projects: List of substrings to exclude from project paths
                         (default: ["model-comparison"] to exclude this analysis)
    """
    if exclude_projects is None:
        exclude_projects = ['model-comparison']

    claude_dir = Path.home() / '.claude' / 'projects'
    cutoff = datetime.now() - timedelta(days=days_back)

    sessions = defaultdict(list)
    excluded_count = 0
    meta_count = 0

    # Build reverse lookup: model_id -> canonical key
    model_id_to_key = {}
    for key, model_id in MODEL_IDS.items():
        model_id_to_key[model_id] = key
    for alias, key in MODEL_ALIASES.items():
        model_id_to_key[alias] = key

    # Walk through all project directories
    for project_dir in claude_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_path = decode_project_dir(project_dir.name)
        meta = is_meta_project(project_path)

        # Check exclusion list
        if any(excl in project_path for excl in exclude_projects):
            excluded_count += len(list(project_dir.glob('*.jsonl')))
            continue

        for session_file in project_dir.glob('*.jsonl'):
            # Skip agent subagent files
            if session_file.name.startswith('agent-'):
                continue

            # Check file modification time
            mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
            if mtime < cutoff:
                continue

            metadata = parse_session_file(session_file)
            if not metadata:
                continue

            # Populate new fields
            metadata.project_path = project_path
            metadata.is_meta = meta
            if meta:
                meta_count += 1

            # Categorize by model using reverse lookup
            canonical = model_id_to_key.get(metadata.model)
            if canonical is None:
                sessions['other'].append(metadata)
            else:
                sessions[canonical].append(metadata)

    if excluded_count:
        print(f"  Excluded {excluded_count} session files from: {', '.join(exclude_projects)}")
    if meta_count:
        print(f"  Tagged {meta_count} sessions as meta (claude-investigations)")

    return dict(sessions)


def print_summary(sessions: dict):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SESSION COLLECTION SUMMARY")
    print("=" * 60)

    for model_type, session_list in sessions.items():
        if not session_list:
            print(f"\n{model_type.upper()}: No sessions found")
            continue

        total_duration = sum(s.duration_minutes for s in session_list)
        total_tools = sum(s.tool_call_count for s in session_list)
        total_user = sum(s.user_message_count for s in session_list)
        total_assistant = sum(s.assistant_message_count for s in session_list)
        meta_sessions = sum(1 for s in session_list if s.is_meta)

        # Aggregate tool usage
        all_tools = defaultdict(int)
        for s in session_list:
            for tool, count in s.tools_used.items():
                all_tools[tool] += count

        # Aggregate project paths
        projects = defaultdict(int)
        for s in session_list:
            proj = s.project_path.rsplit('/', 1)[-1] if s.project_path else 'unknown'
            projects[proj] += 1

        print(f"\n{model_type.upper()}:")
        print(f"  Sessions: {len(session_list)} ({meta_sessions} meta)")
        print(f"  Total duration: {total_duration:.1f} minutes")
        print(f"  User messages: {total_user}")
        print(f"  Assistant messages: {total_assistant}")
        print(f"  Tool calls: {total_tools}")
        print(f"  Projects: {dict(sorted(projects.items(), key=lambda x: -x[1]))}")
        print(f"  Top tools: {dict(sorted(all_tools.items(), key=lambda x: -x[1])[:5])}")


def save_results(sessions: dict, output_dir: Path):
    """Save session data to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_type, session_list in sessions.items():
        if model_type == 'other':
            continue
        output_file = output_dir / f'sessions-{model_type}.json'
        data = [asdict(s) for s in session_list]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} sessions to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Collect Claude Code sessions by model')
    parser.add_argument('--days', type=int, default=90, help='Days back to scan (default: 90)')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Output directory for session files (default: data)')
    parser.add_argument('--exclude-projects', nargs='*', default=['model-comparison'],
                        help='Exclude projects containing these substrings (default: model-comparison)')
    args = parser.parse_args()

    print(f"Scanning sessions from the last {args.days} days...")
    if args.exclude_projects:
        print(f"Excluding projects matching: {', '.join(args.exclude_projects)}")
    sessions = collect_sessions(days_back=args.days, exclude_projects=args.exclude_projects)

    print_summary(sessions)
    save_results(sessions, args.data_dir)


if __name__ == '__main__':
    main()
