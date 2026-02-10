#!/usr/bin/env python3
"""Edit Timeline Analysis: Detect when an agent rewrites its own code.

For each session, builds a per-file content tracker, then detects when a later
edit's old_string overlaps with an earlier edit's new_string on the same file.
This catches self-corrections, error recovery, and user-directed corrections.

Overlap detection uses range tracking rather than pairwise comparison:
each successful Edit/Write updates a per-file content map tracking which edit
"owns" each piece of content. When a new edit removes content, we check which
prior edit placed it.

Reads edit events from canonical task files (tasks-canonical-{model}.json)
instead of raw JSONL session data.

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
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models


# Dissatisfaction signals in user messages
DISSATISFACTION_SIGNALS = [
    r'\bwrong\b', r'\btry again\b', r'\bnot what\b', r'\bincorrect\b',
    r"\bthat's not\b", r'\bno[,.]?\s+(?:that|this)\b', r'\bundo\b',
    r'\brevert\b', r'\bfix (?:that|this|it)\b', r'\bbreak\b', r'\bbroke\b',
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
        # List of (content, edit_index, edit_event) -- most recent last
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

    # Tier 2: Containment -- the contained string must be >=40 chars AND
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
# Session analysis from canonical tasks
# ---------------------------------------------------------------------------

def build_session_data(tasks: list[dict]) -> tuple[list[EditEvent], list[dict]]:
    """Build edit events and context from canonical task records for one session.

    Tasks should be pre-sorted by msg_index_start within the session.
    Returns (edit_events, context_events).
    """
    edits = []
    context = []

    for task in tasks:
        task_id = task['task_id']

        # Add user message as context (task boundary)
        user_prompt = task.get('user_prompt', '') or ''
        if user_prompt.strip():
            context.append({
                'type': 'user_message',
                'index': task.get('msg_index_start', 0),
                'task_id': task_id,
                'text': user_prompt[:500],
            })

        # Add errors as context
        for err in task.get('errors', []):
            context.append({
                'type': 'error',
                'index': err.get('msg_index', 0),
                'task_id': task_id,
                'text': err.get('text', '')[:200],
            })

        # Track subagent usage for flagging
        if task.get('used_subagents'):
            context.append({
                'type': 'subagent',
                'index': task.get('msg_index_start', 0),
                'task_id': task_id,
            })

        # Track bash file modifications
        for cmd in task.get('bash_commands', []):
            if re.search(r'\b(sed|tee|>>|>\s)\b', cmd):
                context.append({
                    'type': 'bash_file_mod',
                    'index': task.get('msg_index_start', 0),
                    'task_id': task_id,
                    'command': cmd[:100],
                })

        # Build EditEvent objects from canonical edit_events
        for ee in task.get('edit_events', []):
            if not ee.get('succeeded', False):
                continue

            edit = EditEvent(
                index=ee.get('msg_index', 0),
                tool_use_id=ee.get('tool_use_id', ''),
                tool_name=ee.get('tool_name', 'Edit'),
                file_path=ee.get('file_path', ''),
                old_string=ee.get('old_string', ''),
                new_string=ee.get('new_string', ''),
                replace_all=ee.get('replace_all', False),
                succeeded=True,
                task_id=task_id,
                timestamp=ee.get('timestamp', ''),
            )
            edits.append(edit)

    # Sort edits by msg_index for correct ordering
    edits.sort(key=lambda e: e.index)

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

    # Different tasks -- check for user dissatisfaction between them
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


def analyze_session_from_tasks(tasks: list[dict]) -> tuple[list[Overlap], dict[str, TaskEditMetrics]]:
    """Analyze a single session's canonical tasks for edit overlaps.

    Returns (overlaps, task_metrics_dict).
    """
    if not tasks:
        return [], {}

    session_id = tasks[0]['session_id']
    edits, context = build_session_data(tasks)

    if not edits:
        return [], {}

    # Per-file content trackers
    file_trackers: dict[str, FileContentTracker] = defaultdict(FileContentTracker)

    # Track chain depth per edit (how many times this content has been rewritten)
    chain_depth: dict[int, int] = defaultdict(int)  # edit_index -> depth

    # Per-task metrics
    task_metrics: dict[str, TaskEditMetrics] = {}
    overlaps = []

    # Track Write->Write on same file within same task
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

        # For Write tool: check if this is a Write->Write on same file
        elif edit.tool_name == 'Write':
            prior_chunks = tracker.chunks[:]
            if prior_chunks:
                best_prior = max(prior_chunks, key=lambda c: len(c[0]))
                prior_edit = best_prior[2]

                if prior_edit.task_id == edit.task_id:
                    if prior_edit.tool_name == 'Write':
                        tm.write_write_same_file += 1
                    elif prior_edit.tool_name == 'Edit':
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

def analyze_model(data_dir: Path, model: str, analysis_dir: Path):
    """Analyze all canonical tasks for a model. Write per-overlap and per-task output."""
    canonical_path = data_dir / f'tasks-canonical-{model}.json'
    if not canonical_path.exists():
        print(f"Error: {canonical_path} not found")
        return {}

    with open(canonical_path, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)

    # Group tasks by session_id
    sessions: dict[str, list[dict]] = defaultdict(list)
    meta_skipped = 0
    for task in all_tasks:
        if task.get('is_meta', False):
            meta_skipped += 1
            continue
        sessions[task['session_id']].append(task)

    # Sort tasks within each session by msg_index_start
    for sid in sessions:
        sessions[sid].sort(key=lambda t: t.get('msg_index_start', 0))

    all_overlaps = []
    all_task_metrics = []
    total_edits = 0
    total_writes = 0
    total_failed = 0
    sessions_processed = 0

    for sid, session_tasks in sessions.items():
        overlaps, task_metrics = analyze_session_from_tasks(session_tasks)
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


def compute_complexity_bins(model: str, data_dir: Path, analysis_dir: Path):
    """Join edit-metrics with tasks-classified to compute per-bin edit accuracy.

    Returns a dict with by_complexity, by_duration_tercile, and coverage keys,
    or None if classified tasks aren't available.
    """
    classified_path = data_dir / f'tasks-classified-{model}.json'
    metrics_path = analysis_dir / f'edit-metrics-{model}.json'

    if not classified_path.exists() or not metrics_path.exists():
        return None

    with open(classified_path) as f:
        classified = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Build lookup: task_id -> classification + duration
    task_info = {}
    for t in classified:
        task_info[t['task_id']] = {
            'complexity': t.get('classification', {}).get('complexity', 'unknown'),
            'duration_seconds': t.get('duration_seconds', 0),
        }

    # Join metrics with classified tasks
    matched = []
    for m in metrics:
        tid = m['task_id']
        if tid in task_info:
            matched.append({**m, **task_info[tid]})

    total = len(metrics)
    n_matched = len(matched)
    if n_matched == 0:
        return None

    print(f"    Complexity bins: {n_matched}/{total} tasks matched ({n_matched/total*100:.1f}%)")

    # --- By complexity ---
    COMPLEXITY_ORDER = ['trivial', 'simple', 'moderate', 'complex', 'major']
    bins_by_complexity = defaultdict(list)
    for m in matched:
        bins_by_complexity[m['complexity']].append(m)

    # Collapse complex+major if either has <10 tasks
    complex_n = len(bins_by_complexity.get('complex', []))
    major_n = len(bins_by_complexity.get('major', []))
    collapse = complex_n < 10 or major_n < 10
    if collapse and ('complex' in bins_by_complexity or 'major' in bins_by_complexity):
        merged = bins_by_complexity.pop('complex', []) + bins_by_complexity.pop('major', [])
        if merged:
            bins_by_complexity['complex+'] = merged

    by_complexity = {}
    for level in COMPLEXITY_ORDER + ['complex+']:
        tasks = bins_by_complexity.get(level)
        if tasks is None:
            continue
        if collapse and level in ('complex', 'major'):
            continue
        by_complexity[level] = _compute_bin_rates(tasks)

    # --- By duration tercile ---
    durations = sorted(m['duration_seconds'] for m in matched if m['duration_seconds'] > 0)
    if len(durations) >= 3:
        t1 = durations[len(durations) // 3]
        t2 = durations[2 * len(durations) // 3]
    else:
        t1 = t2 = 0

    short, medium, long_ = [], [], []
    for m in matched:
        d = m['duration_seconds']
        if d <= t1:
            short.append(m)
        elif d <= t2:
            medium.append(m)
        else:
            long_.append(m)

    by_duration = {}
    if short:
        by_duration['short'] = _compute_bin_rates(short)
        by_duration['short']['range'] = f"0-{t1:.0f}s"
    if medium:
        by_duration['medium'] = _compute_bin_rates(medium)
        by_duration['medium']['range'] = f"{t1:.0f}-{t2:.0f}s"
    if long_:
        by_duration['long'] = _compute_bin_rates(long_)
        by_duration['long']['range'] = f"{t2:.0f}s+"

    return {
        'by_complexity': by_complexity,
        'by_duration_tercile': by_duration,
        'coverage': {
            'matched': n_matched,
            'total': total,
            'pct': round(n_matched / total * 100, 1),
        },
    }


def _compute_bin_rates(tasks: list) -> dict:
    """Compute edit accuracy rates for a bin of tasks."""
    n = len(tasks)
    edits = sum(t['edit_count'] for t in tasks)
    writes = sum(t['write_count'] for t in tasks)
    total_ops = edits + writes
    sc = sum(t['self_corrections'] for t in tasks)
    er = sum(t['error_recoveries'] for t in tasks)
    uc = sum(t['user_corrections'] for t in tasks)
    ir = sum(t['iterative_refinements'] for t in tasks)

    def rate(count):
        return round(count / total_ops, 4) if total_ops > 0 else 0

    return {
        'n': n,
        'edits': edits,
        'writes': writes,
        'self_correction_rate': rate(sc),
        'error_recovery_rate': rate(er),
        'user_correction_rate': rate(uc),
        'iterative_refinement_rate': rate(ir),
        'self_corrections': sc,
        'error_recoveries': er,
        'user_corrections': uc,
        'iterative_refinements': ir,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze edit timelines for self-correction detection')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory with tasks-canonical-*.json files')
    parser.add_argument('--analysis-dir', type=Path, default=None,
                        help='Output directory for analysis results (default: data/../analysis)')
    args = parser.parse_args()

    data_dir = args.data_dir
    analysis_dir = args.analysis_dir or data_dir.parent / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Discover models from canonical task files
    models = discover_models(data_dir, prefix="tasks-canonical")
    if not models:
        # Fall back to session file discovery
        models = discover_models(data_dir)
    if not models:
        print(f"Error: No tasks-canonical-*.json or sessions-*.json files found in {data_dir}")
        return

    print(f"Found {len(models)} model(s): {', '.join(models)}")

    summaries = {}
    for model in models:
        print(f"\nAnalyzing {model}...")
        summaries[model] = analyze_model(data_dir, model, analysis_dir)

    # Compute complexity-binned accuracy
    for model in summaries:
        print(f"\n  Computing complexity bins for {model}...")
        bins = compute_complexity_bins(model, data_dir, analysis_dir)
        if bins:
            summaries[model]['complexity_bins'] = bins

    # Write cross-model summary
    output_path = analysis_dir / 'edit-analysis.json'
    with open(output_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCross-model summary -> {output_path}")


if __name__ == '__main__':
    main()
