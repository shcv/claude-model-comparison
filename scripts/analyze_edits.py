#!/usr/bin/env python3
"""Edit Timeline Analysis: Detect when an agent rewrites its own code.

For each session, builds a per-file content tracker, then detects when a later
edit's old_string overlaps with an earlier edit's new_string on the same file.
This catches self-corrections, error recovery, and user-directed corrections.

Overlap detection uses range tracking rather than pairwise comparison:
each successful Edit/Write updates a per-file content map tracking which edit
"owns" each piece of content. When a new edit removes content, we check which
prior edit placed it.

Output (to analysis/):
  - edit-overlaps-{model}.json: per-overlap records
  - edit-metrics-{model}.json: per-task aggregated metrics
  - edit-analysis.json: cross-model comparison summary

Usage:
    python scripts/analyze_edits.py \\
      --data-dir comparisons/opus-4.5-vs-4.6/data \\
      --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# JSONL parsing helpers (shared patterns with extract_tasks.py)
# ---------------------------------------------------------------------------

def extract_user_text(content) -> str:
    """Extract text from message content."""
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
    """Check if user message is a tool result."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                return True
    return False


SKIP_PATTERNS = [
    r'^<local-command', r'^<command-name>', r'^<system-reminder>',
    r'^<task-notification>', r'^<teammate-message>', r'^\s*$',
]

SYSTEM_CONTENT_PATTERNS = [
    r'^This session is being continued from a previous conversation',
    r'^# Iteration workflow', r'^Implement the following plan:',
    r'^\[Request interrupted', r'^<summary>', r'^Analysis:',
]


def should_skip_message(text: str) -> bool:
    return any(re.match(p, text.strip()) for p in SKIP_PATTERNS)


def is_system_generated(text: str) -> bool:
    return any(re.match(p, text.strip(), re.IGNORECASE) for p in SYSTEM_CONTENT_PATTERNS)


# Dissatisfaction signals in user messages
DISSATISFACTION_SIGNALS = [
    r'\bwrong\b', r'\btry again\b', r'\bnot what\b', r'\bincorrect\b',
    r"\bthat's not\b", r'\bno[,.]?\s+(?:that|this)\b', r'\bundo\b',
    r'\brevert\b', r'\bfix (?:that|this|it)\b', r'\bbreak\b', r'\bbroke\b',
]

# Error indicators in tool results
ERROR_PATTERNS = [
    r'error', r'failed', r'FAIL', r'traceback', r'exception',
    r'SyntaxError', r'TypeError', r'NameError', r'ImportError',
    r'is_error.*true', r'tool_use_error', r'exit code [1-9]',
    r'compilation failed', r'build failed',
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EditEvent:
    """A single Edit or Write tool call."""
    index: int              # position in session message stream
    tool_use_id: str
    tool_name: str          # 'Edit' or 'Write'
    file_path: str
    old_string: str         # empty for Write
    new_string: str
    replace_all: bool
    succeeded: bool
    task_id: str            # which task this belongs to
    timestamp: str


@dataclass
class Overlap:
    """A detected overlap where edit B rewrites content from edit A."""
    file_path: str
    edit_a_index: int
    edit_b_index: int
    edit_a_task: str
    edit_b_task: str
    tier: str               # 'exact', 'containment', 'line_overlap'
    classification: str     # 'self_correction', 'error_recovery', 'user_directed', 'iterative_refinement'
    overlap_ratio: float    # 0-1, how much of A's content was overwritten
    context: str            # brief description


@dataclass
class TaskEditMetrics:
    """Edit metrics for a single task."""
    task_id: str
    session_id: str
    edit_count: int = 0
    write_count: int = 0
    failed_edit_count: int = 0
    overlap_count: int = 0
    self_corrections: int = 0
    error_recoveries: int = 0
    user_corrections: int = 0
    iterative_refinements: int = 0
    max_chain_depth: int = 0
    files_edited: int = 0
    replace_all_count: int = 0
    write_write_same_file: int = 0
    edit_write_undo: int = 0
    bash_file_mod_count: int = 0
    has_subagent_edits: bool = False
    triage_score: float = 0.0


# ---------------------------------------------------------------------------
# Per-file content tracker (range-based overlap detection)
# ---------------------------------------------------------------------------

class FileContentTracker:
    """Tracks which edit placed each piece of content in a file.

    Instead of comparing edit pairs, we maintain a list of "content chunks"
    that were placed by prior edits. When a new edit's old_string matches
    content from a prior chunk, that's an overlap.

    Each chunk is: (content_text, edit_index, edit_event)
    """

    def __init__(self):
        # List of (content, edit_index, edit_event) — most recent last
        self.chunks: list[tuple[str, int, EditEvent]] = []

    def add_content(self, content: str, edit: EditEvent):
        """Record that an edit placed this content."""
        if len(content) < 20:
            return
        self.chunks.append((content, edit.index, edit))

    def find_overlaps(self, old_string: str, new_edit: EditEvent) -> list[tuple[EditEvent, str, float]]:
        """Check if old_string overlaps with any tracked content.

        Returns list of (prior_edit, tier, overlap_ratio).
        """
        if len(old_string) < 20:
            return []

        results = []
        for content, _, prior_edit in self.chunks:
            if prior_edit.index >= new_edit.index:
                continue

            tier, ratio = _compute_overlap(content, old_string)
            if tier:
                results.append((prior_edit, tier, ratio))

        return results

    def remove_content(self, old_string: str):
        """Remove chunks whose content is being overwritten."""
        # Remove chunks that are fully contained in old_string
        self.chunks = [
            (c, idx, e) for c, idx, e in self.chunks
            if c not in old_string
        ]


def _compute_overlap(a_new: str, b_old: str) -> tuple[str | None, float]:
    """Compute overlap tier between A's new_string and B's old_string.

    Returns (tier, ratio) or (None, 0) if no overlap.
    """
    # Tier 1: Exact match
    if a_new == b_old:
        return 'exact', 1.0

    # Tier 2: Containment — the contained string must be >=40 chars AND
    # represent at least 30% of the larger string (to avoid flagging cases
    # where a small prior edit happens to be a substring of a much larger region)
    if len(a_new) >= 40 and a_new in b_old:
        ratio = len(a_new) / len(b_old)
        if ratio >= 0.3:
            return 'containment', ratio
    if len(b_old) >= 40 and b_old in a_new:
        ratio = len(b_old) / len(a_new)
        if ratio >= 0.3:
            return 'containment', ratio

    # Tier 3: Line overlap via Jaccard coefficient
    a_lines = _nontrivial_lines(a_new)
    b_lines = _nontrivial_lines(b_old)

    if not a_lines or not b_lines:
        return None, 0

    intersection = a_lines & b_lines
    if not intersection:
        return None, 0

    union = a_lines | b_lines
    jaccard = len(intersection) / len(union)
    coverage = len(intersection) / len(b_lines)  # how much of B was from A

    if jaccard > 0.3 or coverage > 0.5:
        return 'line_overlap', max(jaccard, coverage)

    return None, 0


def _nontrivial_lines(text: str) -> set[str]:
    """Extract non-trivial lines (>15 chars, stripped) as a set."""
    return {
        line.strip()
        for line in text.split('\n')
        if len(line.strip()) > 15
    }


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------

def parse_session(file_path: Path, model: str) -> tuple[list[EditEvent], list[dict]]:
    """Parse a session JSONL file and extract edit events + context.

    Returns (edit_events, context_events) where context_events track
    user messages, tool errors, and task boundaries.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            messages = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return [], []

    session_id = file_path.stem
    edits = []
    context = []  # user messages, errors, task boundaries
    task_counter = 1
    current_task_id = f"{session_id}-task-1"
    msg_index = 0

    # Map tool_use_id -> EditEvent for matching results
    pending_edits: dict[str, EditEvent] = {}

    for msg in messages:
        msg_type = msg.get('type')
        timestamp = msg.get('timestamp', '')

        if msg_type == 'user':
            content = msg.get('message', {}).get('content', [])

            # Check tool results for edit success/failure and errors
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    if block.get('type') == 'tool_result':
                        tool_use_id = block.get('tool_use_id', '')
                        is_error = block.get('is_error', False)

                        # Match to pending edit
                        if tool_use_id in pending_edits:
                            edit = pending_edits.pop(tool_use_id)
                            edit.succeeded = not is_error
                            if edit.succeeded:
                                edits.append(edit)

                        # Track errors for classification
                        result_text = ''
                        rc = block.get('content', '')
                        if isinstance(rc, str):
                            result_text = rc
                        elif isinstance(rc, list):
                            for rb in rc:
                                if isinstance(rb, dict) and rb.get('type') == 'text':
                                    result_text += rb.get('text', '')

                        if is_error or any(re.search(p, result_text, re.IGNORECASE)
                                           for p in ERROR_PATTERNS):
                            context.append({
                                'type': 'error',
                                'index': msg_index,
                                'task_id': current_task_id,
                                'text': result_text[:200],
                            })

            # Check for real user messages (task boundaries)
            if not is_tool_result(content):
                text = extract_user_text(content)
                if text.strip() and not should_skip_message(text) and not is_system_generated(text):
                    task_counter += 1
                    current_task_id = f"{session_id}-task-{task_counter}"
                    context.append({
                        'type': 'user_message',
                        'index': msg_index,
                        'task_id': current_task_id,
                        'text': text[:500],
                    })

        elif msg_type == 'assistant':
            content = msg.get('message', {}).get('content', [])
            if not isinstance(content, list):
                msg_index += 1
                continue

            for block in content:
                if not isinstance(block, dict) or block.get('type') != 'tool_use':
                    continue

                name = block.get('name', '')
                inp = block.get('input', {})
                tool_use_id = block.get('id', '')

                if name == 'Edit':
                    edit = EditEvent(
                        index=msg_index,
                        tool_use_id=tool_use_id,
                        tool_name='Edit',
                        file_path=inp.get('file_path', ''),
                        old_string=inp.get('old_string', ''),
                        new_string=inp.get('new_string', ''),
                        replace_all=inp.get('replace_all', False),
                        succeeded=False,  # set when we see the result
                        task_id=current_task_id,
                        timestamp=timestamp,
                    )
                    pending_edits[tool_use_id] = edit

                elif name == 'Write':
                    edit = EditEvent(
                        index=msg_index,
                        tool_use_id=tool_use_id,
                        tool_name='Write',
                        file_path=inp.get('file_path', ''),
                        old_string='',
                        new_string=inp.get('content', ''),
                        replace_all=False,
                        succeeded=False,
                        task_id=current_task_id,
                        timestamp=timestamp,
                    )
                    pending_edits[tool_use_id] = edit

                elif name == 'Bash':
                    cmd = inp.get('command', '')
                    if re.search(r'\b(sed|tee|>>|>\s)\b', cmd):
                        context.append({
                            'type': 'bash_file_mod',
                            'index': msg_index,
                            'task_id': current_task_id,
                            'command': cmd[:100],
                        })

                elif name == 'Task':
                    context.append({
                        'type': 'subagent',
                        'index': msg_index,
                        'task_id': current_task_id,
                    })

        msg_index += 1

    return edits, context


def classify_overlap(edit_a: EditEvent, edit_b: EditEvent,
                     context: list[dict], chain_depth: int) -> str:
    """Classify the nature of an overlap.

    - self_correction: same task, no intervening user prompt
    - error_recovery: same task, error between edits
    - user_directed: different task, dissatisfaction signal between
    - iterative_refinement: chain depth > 3 or low overlap
    """
    if chain_depth > 3:
        return 'iterative_refinement'

    same_task = edit_a.task_id == edit_b.task_id

    # Check for errors between the two edits
    errors_between = [
        c for c in context
        if c['type'] == 'error'
        and c['index'] > edit_a.index
        and c['index'] < edit_b.index
    ]

    if same_task:
        if errors_between:
            return 'error_recovery'
        return 'self_correction'

    # Different tasks — check for user dissatisfaction between them
    user_msgs_between = [
        c for c in context
        if c['type'] == 'user_message'
        and c['index'] > edit_a.index
        and c['index'] < edit_b.index
    ]

    for um in user_msgs_between:
        text_lower = um['text'].lower()
        if any(re.search(p, text_lower) for p in DISSATISFACTION_SIGNALS):
            return 'user_directed'

    return 'iterative_refinement'


def analyze_session(file_path: Path, model: str) -> tuple[list[Overlap], dict[str, TaskEditMetrics]]:
    """Analyze a single session for edit overlaps.

    Returns (overlaps, task_metrics_dict).
    """
    edits, context = parse_session(file_path, model)
    session_id = file_path.stem

    if not edits:
        return [], {}

    # Per-file content trackers
    file_trackers: dict[str, FileContentTracker] = defaultdict(FileContentTracker)

    # Track chain depth per edit (how many times this content has been rewritten)
    chain_depth: dict[int, int] = defaultdict(int)  # edit_index -> depth

    # Per-task metrics
    task_metrics: dict[str, TaskEditMetrics] = {}
    overlaps = []

    # Track Write→Write on same file within same task
    task_file_writes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Track subagent tasks for flagging
    subagent_tasks = {c['task_id'] for c in context if c['type'] == 'subagent'}

    for edit in edits:
        # Ensure task metrics exist
        if edit.task_id not in task_metrics:
            task_metrics[edit.task_id] = TaskEditMetrics(
                task_id=edit.task_id,
                session_id=session_id,
            )
        tm = task_metrics[edit.task_id]

        # Count edits
        if edit.tool_name == 'Edit':
            tm.edit_count += 1
            if edit.replace_all:
                tm.replace_all_count += 1
        elif edit.tool_name == 'Write':
            tm.write_count += 1
            task_file_writes[edit.task_id][edit.file_path] += 1

        # Count bash file mods
        bash_mods = [c for c in context
                     if c['type'] == 'bash_file_mod' and c['task_id'] == edit.task_id]
        tm.bash_file_mod_count = len(bash_mods)

        # Flag subagent tasks
        if edit.task_id in subagent_tasks:
            tm.has_subagent_edits = True

        tracker = file_trackers[edit.file_path]

        # Skip replace_all edits (typically variable renames)
        if edit.replace_all:
            tracker.add_content(edit.new_string, edit)
            continue

        # Skip tiny edits
        if max(len(edit.old_string), len(edit.new_string)) < 20:
            tracker.add_content(edit.new_string, edit)
            continue

        # For Edit tool: check if old_string overlaps with prior content
        if edit.tool_name == 'Edit' and edit.old_string:
            matches = tracker.find_overlaps(edit.old_string, edit)

            for prior_edit, tier, ratio in matches:
                depth = chain_depth[prior_edit.index] + 1
                chain_depth[edit.index] = max(chain_depth[edit.index], depth)

                classification = classify_overlap(prior_edit, edit, context, depth)

                overlap = Overlap(
                    file_path=edit.file_path,
                    edit_a_index=prior_edit.index,
                    edit_b_index=edit.index,
                    edit_a_task=prior_edit.task_id,
                    edit_b_task=edit.task_id,
                    tier=tier,
                    classification=classification,
                    overlap_ratio=ratio,
                    context=f"{prior_edit.tool_name}@{prior_edit.index} -> {edit.tool_name}@{edit.index}",
                )
                overlaps.append(overlap)

                # Update task metrics
                tm.overlap_count += 1
                if classification == 'self_correction':
                    tm.self_corrections += 1
                elif classification == 'error_recovery':
                    tm.error_recoveries += 1
                elif classification == 'user_directed':
                    tm.user_corrections += 1
                elif classification == 'iterative_refinement':
                    tm.iterative_refinements += 1

                tm.max_chain_depth = max(tm.max_chain_depth, depth)

            # Remove overwritten content, then add new
            tracker.remove_content(edit.old_string)

        # For Write tool: check if this is a Write→Write on same file
        elif edit.tool_name == 'Write':
            # Check if a prior edit on this file had content we're overwriting
            # (Write replaces entire file, so any prior content on this file is gone)
            prior_chunks = tracker.chunks[:]
            if prior_chunks:
                # The Write overwrites everything — find the most significant prior edit
                best_prior = max(prior_chunks, key=lambda c: len(c[0]))
                prior_edit = best_prior[2]

                # Only flag if both are in same task (Write→Write pattern)
                if prior_edit.task_id == edit.task_id:
                    if prior_edit.tool_name == 'Write':
                        tm.write_write_same_file += 1
                    elif prior_edit.tool_name == 'Edit':
                        # Edit→Write where Write doesn't contain Edit's new_string = undo
                        if prior_edit.new_string not in edit.new_string:
                            tm.edit_write_undo += 1

            # Write replaces all tracked content for this file
            tracker.chunks.clear()

        # Track new content
        tracker.add_content(edit.new_string, edit)

    # Compute files_edited per task
    task_files: dict[str, set[str]] = defaultdict(set)
    for edit in edits:
        task_files[edit.task_id].add(edit.file_path)
    for tid, files in task_files.items():
        if tid in task_metrics:
            task_metrics[tid].files_edited = len(files)

    # Compute triage scores
    for tm in task_metrics.values():
        total = tm.edit_count + tm.write_count
        if total > 0:
            tm.triage_score = (
                tm.self_corrections * 3
                + tm.error_recoveries * 2
                + tm.user_corrections * 5
                + tm.max_chain_depth
            ) / total

    return overlaps, task_metrics


# ---------------------------------------------------------------------------
# Model-level aggregation
# ---------------------------------------------------------------------------

def analyze_model(sessions_file: Path, model: str, analysis_dir: Path):
    """Analyze all sessions for a model. Write per-overlap and per-task output."""
    with open(sessions_file, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    all_overlaps = []
    all_task_metrics = []
    total_edits = 0
    total_writes = 0
    total_failed = 0
    sessions_processed = 0

    meta_skipped = 0
    for session in sessions:
        if session.get('is_meta', False):
            meta_skipped += 1
            continue

        file_path = Path(session['file_path'])
        if not file_path.exists():
            continue

        overlaps, task_metrics = analyze_session(file_path, model)
        sessions_processed += 1

        for o in overlaps:
            all_overlaps.append(asdict(o))

        for tm in task_metrics.values():
            total_edits += tm.edit_count
            total_writes += tm.write_count
            total_failed += tm.failed_edit_count
            all_task_metrics.append(asdict(tm))

    # Write per-overlap records
    overlaps_path = analysis_dir / f'edit-overlaps-{model}.json'
    with open(overlaps_path, 'w') as f:
        json.dump(all_overlaps, f, indent=2)
    print(f"  {len(all_overlaps)} overlaps -> {overlaps_path}")

    # Write per-task metrics
    metrics_path = analysis_dir / f'edit-metrics-{model}.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_task_metrics, f, indent=2)
    print(f"  {len(all_task_metrics)} task records -> {metrics_path}")

    # Compute summary stats
    overlap_tasks = [m for m in all_task_metrics if m['overlap_count'] > 0]
    summary = {
        'model': model,
        'sessions_processed': sessions_processed,
        'total_tasks': len(all_task_metrics),
        'total_edits': total_edits,
        'total_writes': total_writes,
        'total_overlaps': len(all_overlaps),
        'rewrite_rate': len(all_overlaps) / max(total_edits, 1),
        'tasks_with_overlaps': len(overlap_tasks),
        'by_tier': {
            'exact': sum(1 for o in all_overlaps if o['tier'] == 'exact'),
            'containment': sum(1 for o in all_overlaps if o['tier'] == 'containment'),
            'line_overlap': sum(1 for o in all_overlaps if o['tier'] == 'line_overlap'),
        },
        'by_classification': {
            'self_correction': sum(1 for o in all_overlaps if o['classification'] == 'self_correction'),
            'error_recovery': sum(1 for o in all_overlaps if o['classification'] == 'error_recovery'),
            'user_directed': sum(1 for o in all_overlaps if o['classification'] == 'user_directed'),
            'iterative_refinement': sum(1 for o in all_overlaps if o['classification'] == 'iterative_refinement'),
        },
        'triage_top_10': sorted(
            [{'task_id': m['task_id'], 'score': m['triage_score'],
              'self_corrections': m['self_corrections'],
              'error_recoveries': m['error_recoveries'],
              'user_corrections': m['user_corrections'],
              'chain_depth': m['max_chain_depth']}
             for m in all_task_metrics if m['triage_score'] > 0],
            key=lambda x: x['score'], reverse=True
        )[:10],
    }

    if meta_skipped:
        print(f"  Skipped {meta_skipped} meta sessions")
    print(f"\n  {model} summary:")
    print(f"    Edits: {total_edits}, Writes: {total_writes}")
    print(f"    Overlaps: {len(all_overlaps)} ({summary['rewrite_rate']:.1%} rewrite rate)")
    print(f"    By tier: {summary['by_tier']}")
    print(f"    By class: {summary['by_classification']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze edit timelines for self-correction detection')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory with sessions-*.json files')
    parser.add_argument('--analysis-dir', type=Path, default=None,
                        help='Output directory for analysis results (default: data/../analysis)')
    args = parser.parse_args()

    data_dir = args.data_dir
    analysis_dir = args.analysis_dir or data_dir.parent / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Discover models from session files
    session_files = sorted(data_dir.glob('sessions-*.json'))
    if not session_files:
        print(f"Error: No sessions-*.json files found in {data_dir}")
        return

    models = {}
    for sf in session_files:
        # sessions-opus-4-5.json -> opus-4-5
        model_name = sf.stem.replace('sessions-', '')
        models[model_name] = sf

    print(f"Found {len(models)} model(s): {', '.join(models.keys())}")

    summaries = {}
    for model, sf in models.items():
        print(f"\nAnalyzing {model}...")
        summaries[model] = analyze_model(sf, model, analysis_dir)

    # Write cross-model summary
    output_path = analysis_dir / 'edit-analysis.json'
    with open(output_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCross-model summary -> {output_path}")


if __name__ == '__main__':
    main()
