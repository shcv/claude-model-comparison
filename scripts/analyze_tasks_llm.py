#!/usr/bin/env python3
"""
LLM-Powered Task Analysis for Model Comparison

Extends cc-summarize's approach with a different summary prompt that considers
both user messages (initial request + follow-up) to provide:
- Work type characterization (beyond heuristic classification)
- Execution quality assessment
- User sentiment inference (from next_user_message)
- Follow-up pattern analysis
- Summary statistics

Uses claude -p for non-interactive analysis.
"""

import json
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
from datetime import datetime
import argparse
import sys
import threading


@dataclass
class TaskAnalysis:
    """Rich qualitative analysis of a task."""
    task_id: str
    model: str

    # Basic stats (from existing data)
    tool_calls: int
    files_touched: int
    lines_added: int
    lines_removed: int
    duration_seconds: float

    # LLM-generated analysis - core
    work_category: str  # What kind of work was this really?
    execution_quality: str  # How well was it executed?
    user_sentiment: str  # Inferred sentiment
    sentiment_confidence: str  # low/medium/high
    follow_up_pattern: str  # What happened next / what to expect?
    summary: str  # One-paragraph summary

    # LLM-generated analysis - extended dimensions
    task_completion: str = ""  # complete|partial|interrupted|failed
    scope_management: str = ""  # focused|appropriate|expanded|over_engineered
    communication_quality: str = ""  # clear|adequate|verbose|unclear
    error_recovery: str = ""  # none_needed|recovered|struggled|failed
    iteration_required: str = ""  # one_shot|minor|significant
    alignment_score: int = 0  # 1-5 how well agent matched user intent

    # Derived metrics
    lines_per_minute: float = 0.0
    tools_per_file: float = 0.0
    autonomy_level: str = ""  # high/medium/low


ANALYSIS_PROMPT_TEMPLATE = """You are analyzing a Claude Code task interaction to provide a rich qualitative summary.

## Task Data

**Task ID:** {task_id}
**Model:** {model}
**Project:** {project_path}

### User's Initial Request:
"{user_prompt}"

### Work Done:
- **Tool sequence:** {tool_sequence}
- **Files:** {files_touched} touched ({files_written} written, {files_edited} edited, {files_read} read)
- **Lines:** +{lines_added}/-{lines_removed}
- **Tool calls:** {tool_count}
- **Duration:** {duration:.0f} seconds
- **Key tools:** {key_tools}

### User's Next Message:
"{next_user_message}"

### Outcome Category (heuristic):
{outcome_category} - "{outcome_evidence}"

### Heuristic Classification:
- Type: {classification_type}
- Complexity: {complexity}

## Analysis Required

Return a JSON object with exactly these fields:

{{
  "work_category": "1-2 sentence characterization of what kind of work this really was",
  "execution_quality": "1-2 sentence assessment of how well the task was executed",
  "user_sentiment": "inferred user sentiment based on next_user_message and outcome",
  "sentiment_confidence": "low|medium|high",
  "follow_up_pattern": "what happened next or what would be expected to happen",
  "autonomy_level": "high|medium|low - how independently did the agent work?",
  "task_completion": "complete|partial|interrupted|failed - did the agent finish the requested work?",
  "scope_management": "focused|appropriate|expanded|over_engineered - did agent stay on track or add unnecessary work?",
  "communication_quality": "clear|adequate|verbose|unclear - how well did agent explain its work?",
  "error_recovery": "none_needed|recovered|struggled|failed - how did agent handle any errors encountered?",
  "iteration_required": "one_shot|minor|significant - how much back-and-forth was needed?",
  "alignment_score": "1-5 integer - how well did agent match the user's actual intent? (5=perfect alignment)",
  "summary": "2-3 sentence overall summary combining all aspects"
}}

Consider:
- Empty next_user_message with session_end often indicates satisfaction for simple tasks
- "sounds good" type messages indicate approval of prior work
- Technical corrections may be refinement, not dissatisfaction
- Questions in follow-up may indicate confusion or normal iteration
- scope_management: "focused"=minimal, "appropriate"=matched request, "expanded"=added useful extras, "over_engineered"=unnecessary complexity
- alignment_score: 5=exactly what user wanted, 4=minor misalignment, 3=partially aligned, 2=significant gap, 1=missed intent

Return ONLY the JSON object, no other text."""


def build_analysis_prompt(task: dict) -> str:
    """Build the analysis prompt for a task."""
    classification = task.get('classification', {})

    # Extract key tools (deduplicated, limited)
    tool_calls = task.get('tool_calls', [])
    tool_names = [t.get('name', 'unknown') for t in tool_calls if isinstance(t, dict)]
    unique_tools = list(dict.fromkeys(tool_names))[:5]  # First 5 unique
    key_tools = ', '.join(unique_tools) if unique_tools else 'none'

    return ANALYSIS_PROMPT_TEMPLATE.format(
        task_id=task.get('task_id', 'unknown'),
        model=task.get('model', 'unknown'),
        project_path=task.get('project_path', 'unknown'),
        user_prompt=task.get('user_prompt', '')[:500],
        tool_sequence=task.get('tool_sequence', 'none'),
        files_touched=task.get('total_files_touched', 0),
        files_written=len(task.get('files_written', [])),
        files_edited=len(task.get('files_edited', [])),
        files_read=len(task.get('files_read', [])),
        lines_added=task.get('total_lines_added', 0),
        lines_removed=task.get('total_lines_removed', 0),
        tool_count=len(tool_calls),
        duration=task.get('duration_seconds', 0),
        key_tools=key_tools,
        next_user_message=task.get('next_user_message', '(session ended)')[:300],
        outcome_category=task.get('outcome_category', 'unknown'),
        outcome_evidence=task.get('outcome_evidence', ''),
        classification_type=classification.get('type', 'unknown'),
        complexity=classification.get('complexity', 'unknown'),
    )


def analyze_task_with_llm(task: dict, model: str = "haiku") -> Optional[TaskAnalysis]:
    """Analyze a single task using Claude."""
    prompt = build_analysis_prompt(task)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  Error: {result.stderr[:200]}", file=sys.stderr)
            return None

        response = result.stdout.strip()

        # Extract JSON from response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start < 0 or end <= start:
            print(f"  No JSON found in response", file=sys.stderr)
            return None

        data = json.loads(response[start:end])

        # Calculate derived metrics
        duration = task.get('duration_seconds', 0) or 1
        lines_total = task.get('total_lines_added', 0) + task.get('total_lines_removed', 0)
        files = task.get('total_files_touched', 0) or 1
        tool_count = len(task.get('tool_calls', []))

        # Parse alignment_score as int
        alignment = data.get('alignment_score', 3)
        if isinstance(alignment, str):
            try:
                alignment = int(alignment)
            except ValueError:
                alignment = 3

        return TaskAnalysis(
            task_id=task.get('task_id', 'unknown'),
            model=task.get('model', 'unknown'),
            tool_calls=tool_count,
            files_touched=task.get('total_files_touched', 0),
            lines_added=task.get('total_lines_added', 0),
            lines_removed=task.get('total_lines_removed', 0),
            duration_seconds=duration,
            work_category=data.get('work_category', ''),
            execution_quality=data.get('execution_quality', ''),
            user_sentiment=data.get('user_sentiment', ''),
            sentiment_confidence=data.get('sentiment_confidence', 'low'),
            follow_up_pattern=data.get('follow_up_pattern', ''),
            autonomy_level=data.get('autonomy_level', 'medium'),
            task_completion=data.get('task_completion', 'complete'),
            scope_management=data.get('scope_management', 'appropriate'),
            communication_quality=data.get('communication_quality', 'adequate'),
            error_recovery=data.get('error_recovery', 'none_needed'),
            iteration_required=data.get('iteration_required', 'one_shot'),
            alignment_score=alignment,
            summary=data.get('summary', ''),
            lines_per_minute=lines_total / (duration / 60) if duration > 0 else 0,
            tools_per_file=tool_count / files if files > 0 else 0,
        )

    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing task", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def get_cache_path(cache_dir: Path, task_id: str, prompt_hash: str) -> Path:
    """Get cache file path for a task analysis."""
    return cache_dir / f"{task_id}_{prompt_hash[:8]}.json"


def _analyze_single_task(
    task: dict,
    cache_dir: Optional[Path],
    model: str,
    skip_cached: bool,
    progress_lock: threading.Lock,
    progress_counter: list,
    total: int
) -> Optional[TaskAnalysis]:
    """Analyze a single task (for parallel execution)."""
    task_id = task.get('task_id', 'unknown')

    # Check cache
    prompt = build_analysis_prompt(task)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    if cache_dir and skip_cached:
        cache_path = get_cache_path(cache_dir, task_id, prompt_hash)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                with progress_lock:
                    progress_counter[0] += 1
                    print(f"[{progress_counter[0]}/{total}] {task_id[:20]}... (cached)")
                return TaskAnalysis(**cached)
            except Exception:
                pass

    # Analyze with LLM
    analysis = analyze_task_with_llm(task, model)

    with progress_lock:
        progress_counter[0] += 1
        status = "done" if analysis else "failed"
        print(f"[{progress_counter[0]}/{total}] {task_id[:20]}... {status}")

    if analysis and cache_dir:
        cache_path = get_cache_path(cache_dir, task_id, prompt_hash)
        with open(cache_path, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)

    return analysis


def analyze_tasks(
    tasks: list[dict],
    cache_dir: Optional[Path] = None,
    model: str = "haiku",
    limit: Optional[int] = None,
    skip_cached: bool = True,
    max_workers: int = 8
) -> list[TaskAnalysis]:
    """Analyze multiple tasks in parallel, with optional caching."""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_process = tasks[:limit] if limit else tasks
    total = len(tasks_to_process)

    print(f"Analyzing {total} tasks with {max_workers} workers...")

    results = []
    progress_lock = threading.Lock()
    progress_counter = [0]  # Use list to allow mutation in closure

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _analyze_single_task,
                task, cache_dir, model, skip_cached,
                progress_lock, progress_counter, total
            ): task
            for task in tasks_to_process
        }

        for future in as_completed(futures):
            try:
                analysis = future.result()
                if analysis:
                    results.append(analysis)
            except Exception as e:
                print(f"  Worker error: {e}", file=sys.stderr)

    return results


def _print_distribution(analyses: list, attr: str, title: str, order: list = None):
    """Helper to print a distribution of values."""
    dist = {}
    for a in analyses:
        val = getattr(a, attr, 'unknown').lower()
        dist[val] = dist.get(val, 0) + 1

    print(f"\n{title}:")
    items = order if order else sorted(dist.keys(), key=lambda x: -dist.get(x, 0))
    for key in items:
        count = dist.get(key, 0)
        if count == 0 and order:
            continue  # Skip empty ordered items
        pct = 100 * count / len(analyses) if analyses else 0
        bar = '█' * int(pct / 2)
        print(f"  {key:<18} {count:3d} ({pct:5.1f}%) {bar}")


def print_analysis_summary(analyses: list[TaskAnalysis], model_name: str):
    """Print summary of analyses."""
    if not analyses:
        print(f"\n{model_name.upper()}: No analyses")
        return

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} TASK ANALYSIS SUMMARY ({len(analyses)} tasks)")
    print('='*60)

    # Sentiment distribution (categorized)
    sentiments = {}
    for a in analyses:
        sent = a.user_sentiment.lower()
        if 'satisfied' in sent or 'positive' in sent:
            key = 'satisfied'
        elif 'dissatisfied' in sent or 'negative' in sent or 'frustrated' in sent:
            key = 'dissatisfied'
        elif 'neutral' in sent or 'unclear' in sent:
            key = 'neutral'
        elif 'collaborative' in sent or 'iterative' in sent or 'refinement' in sent:
            key = 'collaborative'
        else:
            key = 'other'
        sentiments[key] = sentiments.get(key, 0) + 1

    print("\nUser Sentiment:")
    for sent, count in sorted(sentiments.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(analyses)
        bar = '█' * int(pct / 2)
        print(f"  {sent:<18} {count:3d} ({pct:5.1f}%) {bar}")

    # Task Completion
    _print_distribution(analyses, 'task_completion', 'Task Completion',
                        ['complete', 'partial', 'interrupted', 'failed'])

    # Alignment Score
    scores = [a.alignment_score for a in analyses if a.alignment_score > 0]
    if scores:
        avg_score = sum(scores) / len(scores)
        score_dist = {}
        for s in scores:
            score_dist[s] = score_dist.get(s, 0) + 1
        print(f"\nAlignment Score (avg: {avg_score:.2f}/5):")
        for score in [5, 4, 3, 2, 1]:
            count = score_dist.get(score, 0)
            pct = 100 * count / len(scores) if scores else 0
            bar = '█' * int(pct / 2)
            print(f"  {score:<18} {count:3d} ({pct:5.1f}%) {bar}")

    # Autonomy
    _print_distribution(analyses, 'autonomy_level', 'Autonomy Level',
                        ['high', 'medium', 'low'])

    # Scope Management
    _print_distribution(analyses, 'scope_management', 'Scope Management',
                        ['focused', 'appropriate', 'expanded', 'over_engineered'])

    # Iteration Required
    _print_distribution(analyses, 'iteration_required', 'Iteration Required',
                        ['one_shot', 'minor', 'significant'])

    # Communication Quality
    _print_distribution(analyses, 'communication_quality', 'Communication Quality',
                        ['clear', 'adequate', 'verbose', 'unclear'])

    # Error Recovery
    _print_distribution(analyses, 'error_recovery', 'Error Recovery',
                        ['none_needed', 'recovered', 'struggled', 'failed'])

    # Efficiency Metrics
    lpm = [a.lines_per_minute for a in analyses if a.lines_per_minute > 0]
    tpf = [a.tools_per_file for a in analyses if a.tools_per_file > 0]
    durations = [a.duration_seconds for a in analyses if a.duration_seconds > 0]

    print("\nEfficiency Metrics:")
    if lpm:
        print(f"  Avg lines/minute:    {sum(lpm)/len(lpm):.1f}")
    if tpf:
        print(f"  Avg tools/file:      {sum(tpf)/len(tpf):.1f}")
    if durations:
        print(f"  Avg duration:        {sum(durations)/len(durations):.0f}s")
    print(f"  Avg tool calls:      {sum(a.tool_calls for a in analyses)/len(analyses):.1f}")


def save_analyses(analyses: list[TaskAnalysis], output_file: Path):
    """Save analyses to JSON file."""
    data = [asdict(a) for a in analyses]
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data)} analyses to {output_file}")


def _calc_distribution(analyses: list, attr: str) -> dict:
    """Calculate distribution of an attribute."""
    dist = {}
    for a in analyses:
        val = getattr(a, attr, 'unknown')
        if isinstance(val, str):
            val = val.lower()
        dist[val] = dist.get(val, 0) + 1
    return dist


def _categorize_sentiment(sentiment: str) -> str:
    """Categorize sentiment into standard buckets."""
    sent = sentiment.lower()
    if 'satisfied' in sent or 'positive' in sent:
        return 'satisfied'
    elif 'dissatisfied' in sent or 'negative' in sent or 'frustrated' in sent:
        return 'dissatisfied'
    elif 'neutral' in sent or 'unclear' in sent:
        return 'neutral'
    elif 'collaborative' in sent or 'iterative' in sent or 'refinement' in sent:
        return 'collaborative'
    return 'other'


def generate_org_report(
    opus_4_5_analyses: list[TaskAnalysis],
    opus_4_6_analyses: list[TaskAnalysis],
    output_file: Path
):
    """Generate comprehensive org-mode comparison report."""
    from datetime import datetime

    lines = []
    lines.append("#+TITLE: LLM-Powered Task Analysis: Opus 4.5 vs Opus 4.6")
    lines.append(f"#+DATE: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("#+OPTIONS: toc:2 num:t")
    lines.append("")

    # Executive Summary
    lines.append("* Executive Summary")
    lines.append("")

    opus_4_5_n = len(opus_4_5_analyses)
    opus_4_6_n = len(opus_4_6_analyses)

    # Calculate key metrics
    def avg_alignment(analyses):
        scores = [a.alignment_score for a in analyses if a.alignment_score > 0]
        return sum(scores) / len(scores) if scores else 0

    def pct(analyses, attr, values):
        dist = _calc_distribution(analyses, attr)
        total = sum(dist.values())
        matched = sum(dist.get(v, 0) for v in values)
        return 100 * matched / total if total else 0

    def sentiment_pct(analyses, categories):
        cats = {}
        for a in analyses:
            cat = _categorize_sentiment(a.user_sentiment)
            cats[cat] = cats.get(cat, 0) + 1
        total = sum(cats.values())
        matched = sum(cats.get(c, 0) for c in categories)
        return 100 * matched / total if total else 0

    opus_4_5_align = avg_alignment(opus_4_5_analyses)
    opus_4_6_align = avg_alignment(opus_4_6_analyses)

    opus_4_5_satisfied = sentiment_pct(opus_4_5_analyses, ['satisfied'])
    opus_4_6_satisfied = sentiment_pct(opus_4_6_analyses, ['satisfied'])

    opus_4_5_complete = pct(opus_4_5_analyses, 'task_completion', ['complete'])
    opus_4_6_complete = pct(opus_4_6_analyses, 'task_completion', ['complete'])

    opus_4_5_high_auto = pct(opus_4_5_analyses, 'autonomy_level', ['high'])
    opus_4_6_high_auto = pct(opus_4_6_analyses, 'autonomy_level', ['high'])

    lines.append(f"Analysis of {opus_4_5_n} Opus 4.5 tasks and {opus_4_6_n} Opus 4.6 tasks using LLM-powered")
    lines.append("qualitative assessment of the full task arc: user request → agent work → user response.")
    lines.append("")
    lines.append("** Key Findings")
    lines.append("")
    lines.append("| Metric | Opus 4.5 | Opus 4.6 | Winner |")
    lines.append("|--------|----------|----------|--------|")
    lines.append(f"| Alignment Score (1-5) | {opus_4_5_align:.2f} | {opus_4_6_align:.2f} | {'Opus 4.6' if opus_4_6_align > opus_4_5_align else 'Opus 4.5' if opus_4_5_align > opus_4_6_align else 'Tie'} |")
    lines.append(f"| User Satisfied % | {opus_4_5_satisfied:.1f}% | {opus_4_6_satisfied:.1f}% | {'Opus 4.6' if opus_4_6_satisfied > opus_4_5_satisfied else 'Opus 4.5' if opus_4_5_satisfied > opus_4_6_satisfied else 'Tie'} |")
    lines.append(f"| Task Complete % | {opus_4_5_complete:.1f}% | {opus_4_6_complete:.1f}% | {'Opus 4.6' if opus_4_6_complete > opus_4_5_complete else 'Opus 4.5' if opus_4_5_complete > opus_4_6_complete else 'Tie'} |")
    lines.append(f"| High Autonomy % | {opus_4_5_high_auto:.1f}% | {opus_4_6_high_auto:.1f}% | {'Opus 4.6' if opus_4_6_high_auto > opus_4_5_high_auto else 'Opus 4.5' if opus_4_5_high_auto > opus_4_6_high_auto else 'Tie'} |")
    lines.append("")

    # Detailed comparisons
    lines.append("* Detailed Comparison")
    lines.append("")

    def comparison_table(title: str, attr: str, order: list):
        """Generate comparison table for an attribute."""
        lines.append(f"** {title}")
        lines.append("")
        opus_4_5_dist = _calc_distribution(opus_4_5_analyses, attr)
        opus_4_6_dist = _calc_distribution(opus_4_6_analyses, attr)

        lines.append(f"| {title} | Opus 4.5 | Opus 4.5 % | Opus 4.6 | Opus 4.6 % |")
        lines.append("|--------|----------|------------|----------|------------|")
        for val in order:
            o_count = opus_4_5_dist.get(val, 0)
            f_count = opus_4_6_dist.get(val, 0)
            o_pct = 100 * o_count / opus_4_5_n if opus_4_5_n else 0
            f_pct = 100 * f_count / opus_4_6_n if opus_4_6_n else 0
            lines.append(f"| {val} | {o_count} | {o_pct:.1f}% | {f_count} | {f_pct:.1f}% |")
        lines.append("")

    # Sentiment (special handling)
    lines.append("** User Sentiment")
    lines.append("")
    opus_4_5_sent = {}
    opus_4_6_sent = {}
    for a in opus_4_5_analyses:
        cat = _categorize_sentiment(a.user_sentiment)
        opus_4_5_sent[cat] = opus_4_5_sent.get(cat, 0) + 1
    for a in opus_4_6_analyses:
        cat = _categorize_sentiment(a.user_sentiment)
        opus_4_6_sent[cat] = opus_4_6_sent.get(cat, 0) + 1

    lines.append("| Sentiment | Opus 4.5 | Opus 4.5 % | Opus 4.6 | Opus 4.6 % |")
    lines.append("|-----------|----------|------------|----------|------------|")
    for cat in ['satisfied', 'collaborative', 'neutral', 'dissatisfied', 'other']:
        o_count = opus_4_5_sent.get(cat, 0)
        f_count = opus_4_6_sent.get(cat, 0)
        o_pct = 100 * o_count / opus_4_5_n if opus_4_5_n else 0
        f_pct = 100 * f_count / opus_4_6_n if opus_4_6_n else 0
        lines.append(f"| {cat} | {o_count} | {o_pct:.1f}% | {f_count} | {f_pct:.1f}% |")
    lines.append("")

    comparison_table("Task Completion", "task_completion",
                     ['complete', 'partial', 'interrupted', 'failed'])

    # Alignment Score distribution
    lines.append("** Alignment Score Distribution")
    lines.append("")
    opus_4_5_scores = {}
    opus_4_6_scores = {}
    for a in opus_4_5_analyses:
        if a.alignment_score > 0:
            opus_4_5_scores[a.alignment_score] = opus_4_5_scores.get(a.alignment_score, 0) + 1
    for a in opus_4_6_analyses:
        if a.alignment_score > 0:
            opus_4_6_scores[a.alignment_score] = opus_4_6_scores.get(a.alignment_score, 0) + 1

    opus_4_5_total = sum(opus_4_5_scores.values())
    opus_4_6_total = sum(opus_4_6_scores.values())

    lines.append("| Score | Opus 4.5 | Opus 4.5 % | Opus 4.6 | Opus 4.6 % | Interpretation |")
    lines.append("|-------|----------|------------|----------|------------|----------------|")
    interp = {5: "Perfect alignment", 4: "Minor misalignment", 3: "Partial alignment",
              2: "Significant gap", 1: "Missed intent"}
    for score in [5, 4, 3, 2, 1]:
        o_count = opus_4_5_scores.get(score, 0)
        f_count = opus_4_6_scores.get(score, 0)
        o_pct = 100 * o_count / opus_4_5_total if opus_4_5_total else 0
        f_pct = 100 * f_count / opus_4_6_total if opus_4_6_total else 0
        lines.append(f"| {score} | {o_count} | {o_pct:.1f}% | {f_count} | {f_pct:.1f}% | {interp[score]} |")
    lines.append("")

    comparison_table("Autonomy Level", "autonomy_level", ['high', 'medium', 'low'])
    comparison_table("Scope Management", "scope_management",
                     ['focused', 'appropriate', 'expanded', 'over_engineered'])
    comparison_table("Iteration Required", "iteration_required",
                     ['one_shot', 'minor', 'significant'])
    # Communication quality omitted - weak signal without full agent response text
    comparison_table("Error Recovery", "error_recovery",
                     ['none_needed', 'recovered', 'struggled', 'failed'])

    # Efficiency Metrics
    lines.append("** Efficiency Metrics")
    lines.append("")

    def avg_metric(analyses, attr):
        vals = [getattr(a, attr) for a in analyses if getattr(a, attr, 0) > 0]
        return sum(vals) / len(vals) if vals else 0

    lines.append("| Metric | Opus 4.5 | Opus 4.6 |")
    lines.append("|--------|----------|----------|")
    lines.append(f"| Avg Lines/Minute | {avg_metric(opus_4_5_analyses, 'lines_per_minute'):.1f} | {avg_metric(opus_4_6_analyses, 'lines_per_minute'):.1f} |")
    lines.append(f"| Avg Tools/File | {avg_metric(opus_4_5_analyses, 'tools_per_file'):.1f} | {avg_metric(opus_4_6_analyses, 'tools_per_file'):.1f} |")
    lines.append(f"| Avg Duration (s) | {avg_metric(opus_4_5_analyses, 'duration_seconds'):.0f} | {avg_metric(opus_4_6_analyses, 'duration_seconds'):.0f} |")
    lines.append(f"| Avg Tool Calls | {sum(a.tool_calls for a in opus_4_5_analyses)/opus_4_5_n:.1f} | {sum(a.tool_calls for a in opus_4_6_analyses)/opus_4_6_n:.1f} |")
    lines.append("")

    # Notable Examples section
    lines.append("* Notable Examples")
    lines.append("")

    def find_examples(analyses, condition, description, n=3):
        """Find examples matching a condition."""
        matching = [a for a in analyses if condition(a)]
        if not matching:
            return []
        return matching[:n]

    lines.append("** High Alignment Examples (Score 5)")
    lines.append("")
    for model, analyses in [('Opus 4.5', opus_4_5_analyses), ('Opus 4.6', opus_4_6_analyses)]:
        examples = find_examples(analyses, lambda a: a.alignment_score == 5, "perfect alignment")
        if examples:
            lines.append(f"*** {model}")
            for ex in examples[:2]:
                lines.append(f"- *{ex.task_id[:30]}*: {ex.summary[:200]}...")
            lines.append("")

    lines.append("** Low Alignment Examples (Score 1-2)")
    lines.append("")
    for model, analyses in [('Opus 4.5', opus_4_5_analyses), ('Opus 4.6', opus_4_6_analyses)]:
        examples = find_examples(analyses, lambda a: a.alignment_score <= 2 and a.alignment_score > 0, "low alignment")
        if examples:
            lines.append(f"*** {model}")
            for ex in examples[:2]:
                lines.append(f"- *{ex.task_id[:30]}*: {ex.summary[:200]}...")
            lines.append("")

    # Methodology
    lines.append("* Methodology")
    lines.append("")
    lines.append("** Analysis Approach")
    lines.append("")
    lines.append("Each task was analyzed using Claude (haiku) with a structured prompt that considers:")
    lines.append("1. The user's initial request")
    lines.append("2. The agent's work (tools used, files touched, duration)")
    lines.append("3. The user's next message (the ground truth for satisfaction)")
    lines.append("")
    lines.append("** Dimensions Analyzed")
    lines.append("")
    lines.append("| Dimension | Values | Description |")
    lines.append("|-----------|--------|-------------|")
    lines.append("| Alignment Score | 1-5 | How well agent matched user intent |")
    lines.append("| User Sentiment | satisfied/collaborative/neutral/dissatisfied | Inferred from next message |")
    lines.append("| Task Completion | complete/partial/interrupted/failed | Did agent finish the work? |")
    lines.append("| Autonomy Level | high/medium/low | Independence of agent work |")
    lines.append("| Scope Management | focused/appropriate/expanded/over_engineered | Did agent stay on track? |")
    lines.append("| Iteration Required | one_shot/minor/significant | Back-and-forth needed |")
    lines.append("| Error Recovery | none_needed/recovered/struggled/failed | Handling of errors |")
    lines.append("")

    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='LLM-powered task analysis for model comparison'
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory (default: data)')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis'),
                        help='Output directory (default: analysis)')
    parser.add_argument('--model', choices=['opus-4-5', 'opus-4-6', 'both'], default='both',
                        help='Which model to analyze')
    parser.add_argument('--llm', default='haiku',
                        help='LLM to use for analysis (default: haiku)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks to analyze')
    parser.add_argument('--no-cache', action='store_true',
                        help='Skip cache and re-analyze all tasks')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N random tasks to analyze')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--report', action='store_true',
                        help='Generate org-mode comparison report')
    parser.add_argument('--report-only', action='store_true',
                        help='Only generate report from existing analysis files')
    args = parser.parse_args()

    print("=" * 60)
    print("LLM-POWERED TASK ANALYSIS")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'llm-cache'

    all_analyses = {}

    # If report-only, load from existing files
    if args.report_only:
        for model_name in ['opus-4-5', 'opus-4-6']:
            analysis_file = args.output_dir / f'llm-analysis-{model_name}.json'
            if analysis_file.exists():
                with open(analysis_file) as f:
                    data = json.load(f)
                all_analyses[model_name] = [TaskAnalysis(**d) for d in data]
                print(f"Loaded {len(all_analyses[model_name])} {model_name} analyses from cache")
    else:
        # Run analysis
        models = ['opus-4-5', 'opus-4-6'] if args.model == 'both' else [args.model]

        for model_name in models:
            tasks_file = args.data_dir / f'tasks-classified-{model_name}.json'
            if not tasks_file.exists():
                print(f"\nSkipping {model_name}: {tasks_file} not found")
                continue

            with open(tasks_file) as f:
                tasks = json.load(f)

            print(f"\n{model_name.upper()}: {len(tasks)} tasks loaded")

            # Sample if requested
            if args.sample:
                import random
                tasks = random.sample(tasks, min(args.sample, len(tasks)))
                print(f"  Sampled {len(tasks)} tasks")

            # Analyze
            analyses = analyze_tasks(
                tasks,
                cache_dir=cache_dir,
                model=args.llm,
                limit=args.limit,
                skip_cached=not args.no_cache,
                max_workers=args.workers
            )

            all_analyses[model_name] = analyses

            # Print summary
            print_analysis_summary(analyses, model_name)

            # Save results
            output_file = args.output_dir / f'llm-analysis-{model_name}.json'
            save_analyses(analyses, output_file)

    # Generate report if requested
    if args.report or args.report_only:
        opus_4_5_analyses = all_analyses.get('opus-4-5', [])
        opus_4_6_analyses = all_analyses.get('opus-4-6', [])

        if opus_4_5_analyses and opus_4_6_analyses:
            report_file = args.output_dir / 'llm-comparison-report.org'
            generate_org_report(opus_4_5_analyses, opus_4_6_analyses, report_file)
        else:
            print("\nCannot generate report: need both opus-4-5 and opus-4-6 analyses")


if __name__ == '__main__':
    main()
