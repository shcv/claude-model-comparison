#!/usr/bin/env python3
"""Behavioral Analysis: Planning, Subagents, and Parallelization

Extracts and compares behavioral patterns from canonical task data:
- Subagent (Task tool) usage patterns
- EnterPlanMode usage
- Parallel tool calls
- Autonomous vs user-directed behavior
- Implementation delegation scope
"""

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models, load_canonical_tasks


@dataclass
class BehaviorMetrics:
    """Aggregated behavioral metrics for a model."""
    model: str
    total_tasks: int = 0
    total_sessions: int = 0

    # Subagent usage
    tasks_with_subagents: int = 0
    total_subagent_calls: int = 0
    subagent_types: dict = field(default_factory=dict)
    subagent_user_requested: int = 0
    subagent_autonomous: int = 0

    # General-purpose breakdown
    gp_total: int = 0
    gp_user_requested: int = 0
    gp_autonomous: int = 0
    gp_create_module: int = 0
    gp_fix_targeted: int = 0
    gp_other: int = 0

    # Planning
    tasks_with_planning: int = 0
    planning_user_requested: int = 0
    planning_autonomous: int = 0

    # Parallel tool calls
    messages_with_tools: int = 0
    messages_with_parallel: int = 0
    max_parallel_tools: int = 0

    # Background execution
    foreground_tasks: int = 0
    background_tasks: int = 0

    # Directive compliance - "parallel" directive
    parallel_directive_tasks: int = 0
    parallel_directive_agents: int = 0
    parallel_directive_background: int = 0  # True parallel
    parallel_directive_per_task: list = field(default_factory=list)

    # Directive compliance - "use agents" directive (non-parallel)
    useagents_directive_tasks: int = 0
    useagents_directive_agents: int = 0
    useagents_directive_background: int = 0
    useagents_directive_per_task: list = field(default_factory=list)

    # Subagent success rates (cross-referenced with LLM analysis)
    subagent_task_satisfaction: dict = field(default_factory=dict)
    subagent_prompt_lengths: list = field(default_factory=list)
    avg_subagent_prompt_length: float = 0.0

    # Tool sequence n-grams
    tool_2grams: dict = field(default_factory=dict)
    tool_3grams: dict = field(default_factory=dict)
    common_2grams: list = field(default_factory=list)
    common_3grams: list = field(default_factory=list)


AGENT_KEYWORDS = [
    'subagent', 'agent', 'parallel', 'delegate', 'spawn',
    'dispatch', 'concurrent', 'in parallel'
]

PLANNING_KEYWORDS = [
    'plan', 'approach', 'strategy', 'design', 'how should',
    "what's the best way", 'think through'
]


def is_user_requested_agents(user_prompt: str) -> bool:
    """Check if user explicitly requested agent usage."""
    prompt_lower = user_prompt.lower()
    return any(kw in prompt_lower for kw in AGENT_KEYWORDS)


def is_user_requested_planning(user_prompt: str) -> bool:
    """Check if user explicitly requested planning."""
    prompt_lower = user_prompt.lower()
    return any(kw in prompt_lower for kw in PLANNING_KEYWORDS)


def classify_agent_directive(prompt: str) -> str:
    """Classify the type of agent directive in the prompt.

    Returns: 'parallel', 'use_agents', or None
    """
    prompt_lower = prompt.lower()

    # Check for explicit parallel
    if 'parallel' in prompt_lower or 'concurrently' in prompt_lower or 'simultaneously' in prompt_lower:
        return 'parallel'

    # Check for general agent/subagent usage (but not parallel)
    if re.search(r'\b(subagents?|agents?)\b.*\b(to|for)\b', prompt_lower):
        return 'use_agents'
    if re.search(r'\buse\b.*\b(subagents?|agents?)\b', prompt_lower):
        return 'use_agents'
    if re.search(r'\bspawn\b|\bdelegate\b|\bdispatch\b', prompt_lower):
        return 'use_agents'

    return None


def compute_subagent_success(data_dir: Path, model: str, metrics: BehaviorMetrics):
    """Cross-reference tasks-canonical with llm-analysis to compute subagent success rates."""
    analysis_dir = data_dir.parent / 'analysis'
    llm_path = analysis_dir / f'llm-analysis-{model}.json'

    deep_tasks = load_canonical_tasks(data_dir, model)
    if not deep_tasks or not llm_path.exists():
        print(f"  Skipping subagent success: missing tasks or {llm_path}")
        return

    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_analyses = json.load(f)

    # Index LLM analysis by task_id
    llm_by_task = {a['task_id']: a for a in llm_analyses}

    satisfaction = {
        'with_subagents': {'satisfied': 0, 'neutral': 0, 'dissatisfied': 0, 'total': 0,
                           'scores': [], 'completions': Counter()},
        'without_subagents': {'satisfied': 0, 'neutral': 0, 'dissatisfied': 0, 'total': 0,
                              'scores': [], 'completions': Counter()},
    }

    for task in deep_tasks:
        task_id = task.get('task_id', '')
        tool_calls = task.get('tool_calls', [])
        tool_names = [tc.get('name', '') for tc in tool_calls]

        has_subagent = 'Task' in tool_names
        category = 'with_subagents' if has_subagent else 'without_subagents'

        llm = llm_by_task.get(task_id)
        if llm:
            satisfaction[category]['total'] += 1
            sentiment = llm.get('normalized_user_sentiment', 'neutral')
            if sentiment == 'satisfied':
                satisfaction[category]['satisfied'] += 1
            elif sentiment in ('dissatisfied', 'frustrated'):
                satisfaction[category]['dissatisfied'] += 1
            else:
                satisfaction[category]['neutral'] += 1

            score = llm.get('alignment_score')
            if score is not None:
                satisfaction[category]['scores'].append(score)

            completion = llm.get('task_completion', 'unknown')
            if ' - ' in completion:
                completion = completion.split(' - ')[0].strip()
            satisfaction[category]['completions'][completion] += 1

    metrics.subagent_task_satisfaction = {
        k: {
            'satisfied': v['satisfied'],
            'neutral': v['neutral'],
            'dissatisfied': v['dissatisfied'],
            'total': v['total'],
            'avg_alignment': sum(v['scores']) / len(v['scores']) if v['scores'] else 0,
            'completions': dict(v['completions']),
        }
        for k, v in satisfaction.items()
    }


def compute_tool_ngrams(data_dir: Path, model: str, metrics: BehaviorMetrics):
    """Extract 2-gram and 3-gram tool call sequences from tasks-canonical data."""
    deep_tasks = load_canonical_tasks(data_dir, model)
    if not deep_tasks:
        print(f"  Skipping n-grams: no tasks found for {model}")
        return

    bigrams = Counter()
    trigrams = Counter()

    for task in deep_tasks:
        tool_calls = task.get('tool_calls', [])
        names = [tc.get('name', '') for tc in tool_calls if tc.get('name')]

        if len(names) < 2:
            continue

        for i in range(len(names) - 1):
            bigram = f"{names[i]}->{names[i+1]}"
            bigrams[bigram] += 1

        if len(names) >= 3:
            for i in range(len(names) - 2):
                trigram = f"{names[i]}->{names[i+1]}->{names[i+2]}"
                trigrams[trigram] += 1

    metrics.tool_2grams = dict(bigrams)
    metrics.tool_3grams = dict(trigrams)
    metrics.common_2grams = bigrams.most_common(10)
    metrics.common_3grams = trigrams.most_common(10)


def analyze_canonical_tasks(tasks: list[dict], metrics: BehaviorMetrics):
    """Aggregate behavioral metrics from canonical task records."""
    session_ids = set()

    for task in tasks:
        if task.get('is_meta', False):
            continue

        metrics.total_tasks += 1
        session_ids.add(task['session_id'])

        user_prompt = task.get('user_prompt', '') or ''

        # Subagent usage
        if task.get('used_subagents'):
            metrics.tasks_with_subagents += 1
            count = task.get('subagent_count', 0)
            metrics.total_subagent_calls += count

            # Track subagent types
            for st in task.get('subagent_types', []):
                metrics.subagent_types[st] = metrics.subagent_types.get(st, 0) + 1

            # Background vs foreground from run_in_background_count
            bg = task.get('run_in_background_count', 0)
            fg = count - bg
            metrics.background_tasks += bg
            metrics.foreground_tasks += fg

            # General-purpose breakdown
            gp_count = sum(1 for st in task.get('subagent_types', [])
                           if st == 'general-purpose')
            metrics.gp_total += gp_count

            # User-requested vs autonomous (task-level classification)
            if is_user_requested_agents(user_prompt):
                metrics.subagent_user_requested += count
                metrics.gp_user_requested += gp_count
            else:
                metrics.subagent_autonomous += count
                metrics.gp_autonomous += gp_count

            # Directive compliance
            directive = classify_agent_directive(user_prompt)
            if directive:
                task_info = {
                    'agents': count,
                    'background': bg,
                }
                if directive == 'parallel':
                    metrics.parallel_directive_tasks += 1
                    metrics.parallel_directive_agents += count
                    metrics.parallel_directive_background += bg
                    metrics.parallel_directive_per_task.append(task_info)
                elif directive == 'use_agents':
                    metrics.useagents_directive_tasks += 1
                    metrics.useagents_directive_agents += count
                    metrics.useagents_directive_background += bg
                    metrics.useagents_directive_per_task.append(task_info)

        # Planning
        if task.get('used_planning'):
            metrics.tasks_with_planning += 1
            if is_user_requested_planning(user_prompt):
                metrics.planning_user_requested += 1
            else:
                metrics.planning_autonomous += 1

        # Parallel tool calls: parallel_tool_messages is the count of
        # assistant messages with >1 tool_use block
        ptm = task.get('parallel_tool_messages', 0)
        if ptm > 0:
            metrics.messages_with_parallel += ptm

        # Count total tool-bearing messages from tool_calls length as approximation
        # (each tool call is one invocation; messages_with_tools counts messages)
        tc_count = len(task.get('tool_calls', []))
        if tc_count > 0:
            # We don't have exact messages_with_tools from canonical;
            # approximate: at least 1 message per task with tools
            metrics.messages_with_tools += max(tc_count - ptm, 0) + ptm

    metrics.total_sessions = len(session_ids)


def analyze_model(data_dir: Path, model: str) -> BehaviorMetrics:
    """Analyze all canonical tasks for a model."""
    tasks = load_canonical_tasks(data_dir, model)
    if not tasks:
        print(f"Error: no tasks found for {model}")
        return BehaviorMetrics(model=model)

    metrics = BehaviorMetrics(model=model)
    analyze_canonical_tasks(tasks, metrics)

    # Cross-reference with tasks-canonical and llm-analysis
    compute_subagent_success(data_dir, model, metrics)
    compute_tool_ngrams(data_dir, model, metrics)

    return metrics


def print_comparison(opus_4_5: BehaviorMetrics, opus_4_6: BehaviorMetrics):
    """Print comparison tables."""

    def pct(n, total):
        return f"{100*n/total:.1f}%" if total > 0 else "N/A"

    print("=" * 70)
    print("BEHAVIORAL ANALYSIS: OPUS 4.5 vs OPUS 4.6")
    print("=" * 70)

    print("\n## Overview")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Total sessions':<35} {opus_4_5.total_sessions:>15} {opus_4_6.total_sessions:>15}")
    print(f"{'Total tasks':<35} {opus_4_5.total_tasks:>15} {opus_4_6.total_tasks:>15}")

    print("\n## Subagent Usage")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Tasks with subagents':<35} {opus_4_5.tasks_with_subagents:>15} {opus_4_6.tasks_with_subagents:>15}")
    print(f"{'  % of tasks':<35} {pct(opus_4_5.tasks_with_subagents, opus_4_5.total_tasks):>15} {pct(opus_4_6.tasks_with_subagents, opus_4_6.total_tasks):>15}")
    print(f"{'Total subagent calls':<35} {opus_4_5.total_subagent_calls:>15} {opus_4_6.total_subagent_calls:>15}")
    print(f"{'User requested':<35} {opus_4_5.subagent_user_requested:>15} {opus_4_6.subagent_user_requested:>15}")
    print(f"{'Autonomous':<35} {opus_4_5.subagent_autonomous:>15} {opus_4_6.subagent_autonomous:>15}")
    print(f"{'  % autonomous':<35} {pct(opus_4_5.subagent_autonomous, opus_4_5.total_subagent_calls):>15} {pct(opus_4_6.subagent_autonomous, opus_4_6.total_subagent_calls):>15}")

    print("\n## Subagent Types")
    print(f"{'Type':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    all_types = set(opus_4_5.subagent_types.keys()) | set(opus_4_6.subagent_types.keys())
    for t in sorted(all_types):
        o_count = opus_4_5.subagent_types.get(t, 0)
        f_count = opus_4_6.subagent_types.get(t, 0)
        o_pct = pct(o_count, opus_4_5.total_subagent_calls)
        f_pct = pct(f_count, opus_4_6.total_subagent_calls)
        print(f"{t:<35} {o_count:>7} ({o_pct:>5}) {f_count:>7} ({f_pct:>5})")

    print("\n## General-Purpose Breakdown")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Total GP calls':<35} {opus_4_5.gp_total:>15} {opus_4_6.gp_total:>15}")
    print(f"{'User requested':<35} {opus_4_5.gp_user_requested:>15} {opus_4_6.gp_user_requested:>15}")
    print(f"{'Autonomous':<35} {opus_4_5.gp_autonomous:>15} {opus_4_6.gp_autonomous:>15}")
    print(f"{'  % autonomous':<35} {pct(opus_4_5.gp_autonomous, opus_4_5.gp_total):>15} {pct(opus_4_6.gp_autonomous, opus_4_6.gp_total):>15}")

    print("\n## Planning (EnterPlanMode)")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Tasks with planning':<35} {opus_4_5.tasks_with_planning:>15} {opus_4_6.tasks_with_planning:>15}")
    print(f"{'  % of tasks':<35} {pct(opus_4_5.tasks_with_planning, opus_4_5.total_tasks):>15} {pct(opus_4_6.tasks_with_planning, opus_4_6.total_tasks):>15}")
    print(f"{'User requested':<35} {opus_4_5.planning_user_requested:>15} {opus_4_6.planning_user_requested:>15}")
    print(f"{'Autonomous':<35} {opus_4_5.planning_autonomous:>15} {opus_4_6.planning_autonomous:>15}")

    print("\n## Parallel Tool Calls")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Messages with tools':<35} {opus_4_5.messages_with_tools:>15} {opus_4_6.messages_with_tools:>15}")
    print(f"{'Messages with parallel (>1)':<35} {opus_4_5.messages_with_parallel:>15} {opus_4_6.messages_with_parallel:>15}")
    print(f"{'  % parallel':<35} {pct(opus_4_5.messages_with_parallel, opus_4_5.messages_with_tools):>15} {pct(opus_4_6.messages_with_parallel, opus_4_6.messages_with_tools):>15}")
    print(f"{'Max parallel tools':<35} {opus_4_5.max_parallel_tools:>15} {opus_4_6.max_parallel_tools:>15}")

    print("\n## Background Execution")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Foreground Task calls':<35} {opus_4_5.foreground_tasks:>15} {opus_4_6.foreground_tasks:>15}")
    print(f"{'Background Task calls':<35} {opus_4_5.background_tasks:>15} {opus_4_6.background_tasks:>15}")
    print(f"{'  % background':<35} {pct(opus_4_5.background_tasks, opus_4_5.total_subagent_calls):>15} {pct(opus_4_6.background_tasks, opus_4_6.total_subagent_calls):>15}")

    print("\n## Directive Compliance: User-Directed Agent Usage")

    print(f"\n### 'Parallel' Directive")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Tasks':<35} {opus_4_5.parallel_directive_tasks:>15} {opus_4_6.parallel_directive_tasks:>15}")
    print(f"{'Total agents':<35} {opus_4_5.parallel_directive_agents:>15} {opus_4_6.parallel_directive_agents:>15}")
    print(f"{'Background (true parallel)':<35} {opus_4_5.parallel_directive_background:>15} {opus_4_6.parallel_directive_background:>15}")
    print(f"{'  % actually parallel':<35} {pct(opus_4_5.parallel_directive_background, opus_4_5.parallel_directive_agents):>15} {pct(opus_4_6.parallel_directive_background, opus_4_6.parallel_directive_agents):>15}")

    print(f"\n### 'Use Agents' Directive (non-parallel)")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    print(f"{'Tasks':<35} {opus_4_5.useagents_directive_tasks:>15} {opus_4_6.useagents_directive_tasks:>15}")
    print(f"{'Total agents':<35} {opus_4_5.useagents_directive_agents:>15} {opus_4_6.useagents_directive_agents:>15}")
    print(f"{'Background':<35} {opus_4_5.useagents_directive_background:>15} {opus_4_6.useagents_directive_background:>15}")
    print(f"{'  % background':<35} {pct(opus_4_5.useagents_directive_background, opus_4_5.useagents_directive_agents):>15} {pct(opus_4_6.useagents_directive_background, opus_4_6.useagents_directive_agents):>15}")

    # Summary
    total_opus_4_5_directed = opus_4_5.parallel_directive_tasks + opus_4_5.useagents_directive_tasks
    total_opus_4_6_directed = opus_4_6.parallel_directive_tasks + opus_4_6.useagents_directive_tasks
    print(f"\n### Summary")
    print(f"{'Total user-directed agent tasks':<35} {total_opus_4_5_directed:>15} {total_opus_4_6_directed:>15}")

    # Subagent success rates
    print("\n## Subagent Success Rates (cross-ref with LLM analysis)")
    print(f"{'Metric':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)

    for category in ['with_subagents', 'without_subagents']:
        label = category.replace('_', ' ').title()
        o_data = opus_4_5.subagent_task_satisfaction.get(category, {})
        f_data = opus_4_6.subagent_task_satisfaction.get(category, {})
        o_total = o_data.get('total', 0)
        f_total = f_data.get('total', 0)

        print(f"\n  {label}:")
        print(f"{'    Tasks analyzed':<35} {o_total:>15} {f_total:>15}")
        print(f"{'    Satisfied':<35} {o_data.get('satisfied', 0):>15} {f_data.get('satisfied', 0):>15}")
        print(f"{'    Neutral':<35} {o_data.get('neutral', 0):>15} {f_data.get('neutral', 0):>15}")
        print(f"{'    Dissatisfied':<35} {o_data.get('dissatisfied', 0):>15} {f_data.get('dissatisfied', 0):>15}")
        print(f"{'    Avg alignment score':<35} {o_data.get('avg_alignment', 0):>15.2f} {f_data.get('avg_alignment', 0):>15.2f}")
        print(f"{'    % satisfied':<35} {pct(o_data.get('satisfied', 0), o_total):>15} {pct(f_data.get('satisfied', 0), f_total):>15}")

        o_comp = o_data.get('completions', {})
        f_comp = f_data.get('completions', {})
        all_comp = sorted(set(o_comp.keys()) | set(f_comp.keys()))
        for c in all_comp:
            print(f"{'    completion: ' + c:<35} {o_comp.get(c, 0):>15} {f_comp.get(c, 0):>15}")

    print(f"\n{'Avg subagent prompt length':<35} {opus_4_5.avg_subagent_prompt_length:>15.0f} {opus_4_6.avg_subagent_prompt_length:>15.0f}")
    o_prompts = opus_4_5.subagent_prompt_lengths
    f_prompts = opus_4_6.subagent_prompt_lengths
    print(f"{'Subagent prompts recorded':<35} {len(o_prompts):>15} {len(f_prompts):>15}")
    if o_prompts:
        print(f"{'  Min/Max length (Opus 4.5)':<35} {min(o_prompts):>7} / {max(o_prompts):>6}")
    if f_prompts:
        print(f"{'  Min/Max length (Opus 4.6)':<35} {min(f_prompts):>7} / {max(f_prompts):>6}")

    # Tool sequence n-grams
    print("\n## Tool Sequence Patterns (2-grams)")
    print(f"{'Rank':<6} {'Opus 4.5':<35} {'Opus 4.6':<35}")
    print("-" * 76)
    o_2g = opus_4_5.common_2grams
    f_2g = opus_4_6.common_2grams
    for i in range(10):
        o_entry = f"{o_2g[i][0]} ({o_2g[i][1]})" if i < len(o_2g) else ""
        f_entry = f"{f_2g[i][0]} ({f_2g[i][1]})" if i < len(f_2g) else ""
        print(f"{i+1:<6} {o_entry:<35} {f_entry:<35}")

    print("\n## Tool Sequence Patterns (3-grams)")
    print(f"{'Rank':<6} {'Opus 4.5':<45} {'Opus 4.6':<45}")
    print("-" * 96)
    o_3g = opus_4_5.common_3grams
    f_3g = opus_4_6.common_3grams
    for i in range(5):
        o_entry = f"{o_3g[i][0]} ({o_3g[i][1]})" if i < len(o_3g) else ""
        f_entry = f"{f_3g[i][0]} ({f_3g[i][1]})" if i < len(f_3g) else ""
        print(f"{i+1:<6} {o_entry:<45} {f_entry:<45}")

    # Highlight specific patterns of interest
    print("\n## Key Pattern Frequencies")
    print(f"{'Pattern':<35} {'Opus 4.5':>15} {'Opus 4.6':>15}")
    print("-" * 65)
    key_patterns_2 = ['Read->Edit', 'Read->Grep', 'Grep->Read', 'Glob->Read']
    key_patterns_3 = ['Read->Grep->Edit', 'Read->Grep->Read', 'Glob->Read->Edit',
                       'EnterPlanMode->Read->Edit']
    for pat in key_patterns_2:
        o_count = opus_4_5.tool_2grams.get(pat, 0)
        f_count = opus_4_6.tool_2grams.get(pat, 0)
        print(f"{pat:<35} {o_count:>15} {f_count:>15}")
    for pat in key_patterns_3:
        o_count = opus_4_5.tool_3grams.get(pat, 0)
        f_count = opus_4_6.tool_3grams.get(pat, 0)
        print(f"{pat:<35} {o_count:>15} {f_count:>15}")


def generate_org_report(opus_4_5: BehaviorMetrics, opus_4_6: BehaviorMetrics, output_path: Path):
    """Generate org-mode report."""

    def pct(n, total):
        return f"{100*n/total:.1f}%" if total > 0 else "N/A"

    lines = [
        "#+TITLE: Behavioral Analysis: Opus 4.5 vs Opus 4.6",
        f"#+DATE: {__import__('datetime').date.today()}",
        "#+OPTIONS: toc:2 num:t",
        "",
        "* Executive Summary",
        "",
        "| Metric | Opus 4.5 | Opus 4.6 |",
        "|--------|----------|----------|",
        f"| Tasks with subagents | {pct(opus_4_5.tasks_with_subagents, opus_4_5.total_tasks)} | {pct(opus_4_6.tasks_with_subagents, opus_4_6.total_tasks)} |",
        f"| Tasks with planning | {pct(opus_4_5.tasks_with_planning, opus_4_5.total_tasks)} | {pct(opus_4_6.tasks_with_planning, opus_4_6.total_tasks)} |",
        f"| Autonomous subagent % | {pct(opus_4_5.subagent_autonomous, opus_4_5.total_subagent_calls)} | {pct(opus_4_6.subagent_autonomous, opus_4_6.total_subagent_calls)} |",
        f"| General-purpose calls | {opus_4_5.gp_total} | {opus_4_6.gp_total} |",
        f"| Parallel tool messages | {opus_4_5.messages_with_parallel} | {opus_4_6.messages_with_parallel} |",
        "",
        "* Subagent Types",
        "",
        "| Type | Opus 4.5 | Opus 4.5 % | Opus 4.6 | Opus 4.6 % |",
        "|------|----------|------------|----------|------------|",
    ]

    all_types = sorted(set(opus_4_5.subagent_types.keys()) | set(opus_4_6.subagent_types.keys()))
    for t in all_types:
        o_count = opus_4_5.subagent_types.get(t, 0)
        f_count = opus_4_6.subagent_types.get(t, 0)
        lines.append(f"| {t} | {o_count} | {pct(o_count, opus_4_5.total_subagent_calls)} | {f_count} | {pct(f_count, opus_4_6.total_subagent_calls)} |")

    lines.extend([
        "",
        "* General-Purpose Scope",
        "",
        "| Scope | Opus 4.5 | Opus 4.6 |",
        "|-------|----------|----------|",
        f"| Total GP | {opus_4_5.gp_total} | {opus_4_6.gp_total} |",
        f"| User requested | {opus_4_5.gp_user_requested} | {opus_4_6.gp_user_requested} |",
        f"| Autonomous | {opus_4_5.gp_autonomous} | {opus_4_6.gp_autonomous} |",
        "",
    ])

    # Subagent success rates
    lines.extend([
        "* Subagent Success Rates",
        "",
    ])

    for category in ['with_subagents', 'without_subagents']:
        label = category.replace('_', ' ').title()
        o_data = opus_4_5.subagent_task_satisfaction.get(category, {})
        f_data = opus_4_6.subagent_task_satisfaction.get(category, {})
        o_total = o_data.get('total', 0)
        f_total = f_data.get('total', 0)

        lines.extend([
            f"** {label}",
            "",
            "| Metric | Opus 4.5 | Opus 4.6 |",
            "|--------|----------|----------|",
            f"| Tasks analyzed | {o_total} | {f_total} |",
            f"| Satisfied | {o_data.get('satisfied', 0)} | {f_data.get('satisfied', 0)} |",
            f"| Neutral | {o_data.get('neutral', 0)} | {f_data.get('neutral', 0)} |",
            f"| Dissatisfied | {o_data.get('dissatisfied', 0)} | {f_data.get('dissatisfied', 0)} |",
            f"| Avg alignment score | {o_data.get('avg_alignment', 0):.2f} | {f_data.get('avg_alignment', 0):.2f} |",
            f"| % satisfied | {pct(o_data.get('satisfied', 0), o_total)} | {pct(f_data.get('satisfied', 0), f_total)} |",
            "",
        ])

    lines.extend([
        "** Subagent Prompt Length",
        "",
        "| Metric | Opus 4.5 | Opus 4.6 |",
        "|--------|----------|----------|",
        f"| Avg prompt length (chars) | {opus_4_5.avg_subagent_prompt_length:.0f} | {opus_4_6.avg_subagent_prompt_length:.0f} |",
        f"| Prompts recorded | {len(opus_4_5.subagent_prompt_lengths)} | {len(opus_4_6.subagent_prompt_lengths)} |",
        "",
    ])

    # Tool n-grams
    lines.extend([
        "* Tool Sequence Patterns",
        "",
        "** Top 10 Tool 2-grams",
        "",
        "| Rank | Opus 4.5 | Count | Opus 4.6 | Count |",
        "|------|----------|-------|----------|-------|",
    ])

    o_2g = opus_4_5.common_2grams
    f_2g = opus_4_6.common_2grams
    for i in range(10):
        o_name = o_2g[i][0] if i < len(o_2g) else ""
        o_count = o_2g[i][1] if i < len(o_2g) else ""
        f_name = f_2g[i][0] if i < len(f_2g) else ""
        f_count = f_2g[i][1] if i < len(f_2g) else ""
        lines.append(f"| {i+1} | {o_name} | {o_count} | {f_name} | {f_count} |")

    lines.extend([
        "",
        "** Top 5 Tool 3-grams",
        "",
        "| Rank | Opus 4.5 | Count | Opus 4.6 | Count |",
        "|------|----------|-------|----------|-------|",
    ])

    o_3g = opus_4_5.common_3grams
    f_3g = opus_4_6.common_3grams
    for i in range(5):
        o_name = o_3g[i][0] if i < len(o_3g) else ""
        o_count = o_3g[i][1] if i < len(o_3g) else ""
        f_name = f_3g[i][0] if i < len(f_3g) else ""
        f_count = f_3g[i][1] if i < len(f_3g) else ""
        lines.append(f"| {i+1} | {o_name} | {o_count} | {f_name} | {f_count} |")

    lines.extend([
        "",
        "** Key Pattern Frequencies",
        "",
        "| Pattern | Opus 4.5 | Opus 4.6 |",
        "|---------|----------|----------|",
    ])

    key_patterns = [
        'Read->Edit', 'Read->Grep', 'Grep->Read', 'Glob->Read',
        'Read->Grep->Edit', 'Read->Grep->Read', 'Glob->Read->Edit',
        'EnterPlanMode->Read->Edit',
    ]
    for pat in key_patterns:
        if '->' in pat and pat.count('->') == 1:
            o_count = opus_4_5.tool_2grams.get(pat, 0)
            f_count = opus_4_6.tool_2grams.get(pat, 0)
        else:
            o_count = opus_4_5.tool_3grams.get(pat, 0)
            f_count = opus_4_6.tool_3grams.get(pat, 0)
        lines.append(f"| {pat} | {o_count} | {f_count} |")

    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nOrg report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze behavioral patterns')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='Data directory (default: data)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output JSON file for metrics')
    parser.add_argument('--report', action='store_true',
                        help='Generate org-mode report')
    args = parser.parse_args()

    models = discover_models(args.data_dir, prefix="tasks-canonical")
    if len(models) < 2:
        # Fall back to session file discovery
        models = discover_models(args.data_dir)
    if len(models) < 2:
        print(f"Error: Need at least 2 models in {args.data_dir}, found: {models}")
        return

    model_metrics = {}
    for model in models:
        print(f"Analyzing {model} from canonical tasks...")
        model_metrics[model] = analyze_model(args.data_dir, model)

    if len(model_metrics) >= 2:
        model_names = list(model_metrics.keys())
        print_comparison(model_metrics[model_names[0]], model_metrics[model_names[1]])

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({m: asdict(v) for m, v in model_metrics.items()}, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")

    if args.report and len(model_metrics) >= 2:
        model_names = list(model_metrics.keys())
        report_path = args.data_dir.parent / 'analysis' / 'behavior-report.org'
        report_path.parent.mkdir(exist_ok=True)
        generate_org_report(model_metrics[model_names[0]], model_metrics[model_names[1]], report_path)


if __name__ == '__main__':
    main()
