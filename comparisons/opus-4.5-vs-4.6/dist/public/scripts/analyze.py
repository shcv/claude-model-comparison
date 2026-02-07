#!/usr/bin/env python3
"""
Analysis Orchestrator for Opus 4.5 vs Opus 4.6 Model Comparison

Uses Claude Code SDK to:
1. Sample tasks from each model
2. Score individual tasks with Haiku (fast/cheap)
3. Generate comparison report with Sonnet (higher quality synthesis)
"""

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


# Configuration
HAIKU_MODEL = "haiku"
SONNET_MODEL = "sonnet"
TASKS_PER_MODEL = 20  # Sample size for detailed analysis
OUTPUT_DIR = Path("analysis")


@dataclass
class TaskScore:
    task_id: str
    model: str
    efficiency: int
    correctness: int
    communication: int
    autonomy: int
    friction: int
    overall: float
    strengths: list
    weaknesses: list
    notes: str


def run_claude(prompt: str, model: str = HAIKU_MODEL, max_tokens: int = 2000) -> Optional[str]:
    """Run Claude via the SDK CLI and return the response."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"Claude error: {result.stderr}", file=sys.stderr)
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("Claude request timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Claude CLI not found. Is claude-code installed?", file=sys.stderr)
        return None


# Model name mapping
MODEL_NAMES = {
    'opus-4-5': 'Opus 4.5',
    'opus-4-6': 'Opus 4.6',
}
MODEL_IDS = {
    'opus-4-5': 'claude-opus-4-5-20251101',
    'opus-4-6': 'claude-opus-4-6',
}


def load_tasks(model: str) -> list[dict]:
    """Load tasks for a given model."""
    tasks_file = Path(f"data/tasks-{model}.json")
    if not tasks_file.exists():
        return []
    with open(tasks_file) as f:
        return json.load(f)


def sample_tasks(tasks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n tasks, preferring tasks with actual tool usage."""
    random.seed(seed)

    # Filter to tasks with meaningful content
    meaningful = [t for t in tasks if t['tool_calls'] > 0 and len(t['user_prompt']) > 20]

    # Stratify by completion signal to get variety
    by_signal = {}
    for t in meaningful:
        signal = t['completion_signal']
        by_signal.setdefault(signal, []).append(t)

    # Sample proportionally from each signal type
    sampled = []
    for signal, signal_tasks in by_signal.items():
        proportion = len(signal_tasks) / len(meaningful)
        count = max(1, int(n * proportion))
        sampled.extend(random.sample(signal_tasks, min(count, len(signal_tasks))))

    # Fill remaining slots randomly if needed
    if len(sampled) < n:
        remaining = [t for t in meaningful if t not in sampled]
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled[:n]


def score_task(task: dict, model: str) -> Optional[TaskScore]:
    """Score a single task using Haiku."""
    prompt = f"""Score this Claude Code task on 5 dimensions (1-5 scale each).

## Task Context
- Model: {model}
- User Request: {task['user_prompt'][:500]}
- Tool Calls: {task['tool_calls']}
- Tools Used: {', '.join(task['tools_used']) if task['tools_used'] else 'None'}
- Duration: {task['duration_seconds']}s
- Completion Signal: {task['completion_signal']}
- Appears Successful: {task['appears_successful']}

## Scoring Dimensions
1. **Efficiency** (1-5): Direct path vs wandering (based on tool count relative to task complexity)
2. **Correctness** (1-5): Infer from completion signal and success flag
3. **Communication** (1-5): Base on typical model behavior (limited info available)
4. **Autonomy** (1-5): Infer from tool diversity and completion without dissatisfaction
5. **Friction** (1-5, higher=better): Smooth interaction based on completion signal

Return ONLY valid JSON in this exact format:
{{"efficiency": N, "correctness": N, "communication": N, "autonomy": N, "friction": N, "overall": N.N, "strengths": ["..."], "weaknesses": ["..."], "notes": "..."}}"""

    response = run_claude(prompt, model=HAIKU_MODEL)
    if not response:
        return None

    # Extract JSON from response
    try:
        # Find JSON in response (may have surrounding text)
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            return TaskScore(
                task_id=task['task_id'],
                model=model,
                efficiency=data.get('efficiency', 3),
                correctness=data.get('correctness', 3),
                communication=data.get('communication', 3),
                autonomy=data.get('autonomy', 3),
                friction=data.get('friction', 3),
                overall=data.get('overall', 3.0),
                strengths=data.get('strengths', []),
                weaknesses=data.get('weaknesses', []),
                notes=data.get('notes', '')
            )
    except json.JSONDecodeError as e:
        print(f"Failed to parse score response: {e}", file=sys.stderr)
        return None

    return None


def aggregate_scores(scores: list[TaskScore]) -> dict:
    """Aggregate scores into summary statistics."""
    if not scores:
        return {}

    def avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0

    return {
        'count': len(scores),
        'efficiency': avg([s.efficiency for s in scores]),
        'correctness': avg([s.correctness for s in scores]),
        'communication': avg([s.communication for s in scores]),
        'autonomy': avg([s.autonomy for s in scores]),
        'friction': avg([s.friction for s in scores]),
        'overall': avg([s.overall for s in scores]),
        'common_strengths': get_common_items([s.strengths for s in scores]),
        'common_weaknesses': get_common_items([s.weaknesses for s in scores]),
    }


def get_common_items(lists: list[list], top_n: int = 5) -> list[str]:
    """Find most common items across lists of strings."""
    counts = {}
    for lst in lists:
        for item in lst:
            item_lower = item.lower().strip()
            counts[item_lower] = counts.get(item_lower, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    return [item for item, _ in sorted_items[:top_n]]


def generate_comparison_report(
    opus_4_5_scores: list[TaskScore],
    opus_4_6_scores: list[TaskScore],
    opus_4_5_tasks: list[dict],
    opus_4_6_tasks: list[dict]
) -> str:
    """Generate final comparison report using Sonnet."""
    opus_4_5_agg = aggregate_scores(opus_4_5_scores)
    opus_4_6_agg = aggregate_scores(opus_4_6_scores)

    # Calculate completion signal distributions
    def signal_dist(tasks):
        dist = {}
        for t in tasks:
            sig = t['completion_signal']
            dist[sig] = dist.get(sig, 0) + 1
        return dist

    opus_4_5_signals = signal_dist(opus_4_5_tasks)
    opus_4_6_signals = signal_dist(opus_4_6_tasks)

    prompt = f"""Generate a comprehensive comparison report between two Claude models based on task analysis data.

## Models
- **Opus 4.5** (claude-opus-4-5-20251101): Standard flagship model
- **Opus 4.6** (claude-opus-4-6): Latest flagship model

## Overall Statistics

### Opus 4.5 (from {len(opus_4_5_tasks)} total tasks)
- Sampled & scored: {opus_4_5_agg.get('count', 0)} tasks
- Completion signals: {json.dumps(opus_4_5_signals)}
- Success rate: {sum(1 for t in opus_4_5_tasks if t['appears_successful']) / len(opus_4_5_tasks) * 100:.1f}%
- Avg tool calls/task: {sum(t['tool_calls'] for t in opus_4_5_tasks) / len(opus_4_5_tasks):.1f}
- Avg duration: {sum(t['duration_seconds'] for t in opus_4_5_tasks) / len(opus_4_5_tasks):.0f}s

### Opus 4.6 (from {len(opus_4_6_tasks)} total tasks)
- Sampled & scored: {opus_4_6_agg.get('count', 0)} tasks
- Completion signals: {json.dumps(opus_4_6_signals)}
- Success rate: {sum(1 for t in opus_4_6_tasks if t['appears_successful']) / len(opus_4_6_tasks) * 100:.1f}%
- Avg tool calls/task: {sum(t['tool_calls'] for t in opus_4_6_tasks) / len(opus_4_6_tasks):.1f}
- Avg duration: {sum(t['duration_seconds'] for t in opus_4_6_tasks) / len(opus_4_6_tasks):.0f}s

## Scored Sample Analysis

### Opus 4.5 Scores (n={opus_4_5_agg.get('count', 0)})
- Efficiency: {opus_4_5_agg.get('efficiency', 'N/A')}/5
- Correctness: {opus_4_5_agg.get('correctness', 'N/A')}/5
- Communication: {opus_4_5_agg.get('communication', 'N/A')}/5
- Autonomy: {opus_4_5_agg.get('autonomy', 'N/A')}/5
- Friction (higher=better): {opus_4_5_agg.get('friction', 'N/A')}/5
- Overall: {opus_4_5_agg.get('overall', 'N/A')}/5
- Common strengths: {opus_4_5_agg.get('common_strengths', [])}
- Common weaknesses: {opus_4_5_agg.get('common_weaknesses', [])}

### Opus 4.6 Scores (n={opus_4_6_agg.get('count', 0)})
- Efficiency: {opus_4_6_agg.get('efficiency', 'N/A')}/5
- Correctness: {opus_4_6_agg.get('correctness', 'N/A')}/5
- Communication: {opus_4_6_agg.get('communication', 'N/A')}/5
- Autonomy: {opus_4_6_agg.get('autonomy', 'N/A')}/5
- Friction (higher=better): {opus_4_6_agg.get('friction', 'N/A')}/5
- Overall: {opus_4_6_agg.get('overall', 'N/A')}/5
- Common strengths: {opus_4_6_agg.get('common_strengths', [])}
- Common weaknesses: {opus_4_6_agg.get('common_weaknesses', [])}

## Instructions

Write a detailed comparison report in Markdown format with:
1. Executive Summary (2-3 sentences)
2. Key Findings (3-5 bullet points)
3. Detailed Comparison by dimension
4. Model Profiles (strengths/weaknesses for each)
5. Recommendations (when to prefer each model)
6. Caveats and limitations of this analysis

Be analytical and balanced. Note that scores are based on limited metadata - actual task content wasn't deeply analyzed.

IMPORTANT: Write the actual report content directly. Do NOT write a summary of what the report would contain. Start with "## Executive Summary" and write the full report."""

    response = run_claude(prompt, model=SONNET_MODEL, max_tokens=4000)
    return response or "Failed to generate comparison report."


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze model comparison data')
    parser.add_argument('--sample-size', type=int, default=TASKS_PER_MODEL,
                        help=f'Tasks to sample per model (default: {TASKS_PER_MODEL})')
    parser.add_argument('--skip-scoring', action='store_true',
                        help='Skip individual task scoring, use existing scores')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    print("=" * 60)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 60)

    # Load all tasks
    opus_4_5_tasks = load_tasks('opus-4-5')
    opus_4_6_tasks = load_tasks('opus-4-6')
    print(f"\nLoaded {len(opus_4_5_tasks)} Opus 4.5 tasks, {len(opus_4_6_tasks)} Opus 4.6 tasks")

    opus_4_5_scores = []
    opus_4_6_scores = []
    scores_file = args.output / 'scores-latest.json'

    if args.skip_scoring and scores_file.exists():
        print("\nLoading existing scores...")
        with open(scores_file) as f:
            saved = json.load(f)
            opus_4_5_scores = [TaskScore(**s) for s in saved.get('opus-4-5', [])]
            opus_4_6_scores = [TaskScore(**s) for s in saved.get('opus-4-6', [])]
    else:
        # Sample and score tasks
        print(f"\nSampling {args.sample_size} tasks per model...")

        opus_4_5_sample = sample_tasks(opus_4_5_tasks, args.sample_size)
        opus_4_6_sample = sample_tasks(opus_4_6_tasks, args.sample_size)

        print(f"\nScoring {len(opus_4_5_sample)} Opus 4.5 tasks with Haiku...")
        for i, task in enumerate(opus_4_5_sample):
            print(f"  [{i+1}/{len(opus_4_5_sample)}] {task['task_id'][:20]}...", end=" ")
            score = score_task(task, 'opus-4-5')
            if score:
                opus_4_5_scores.append(score)
                print(f"overall={score.overall}")
            else:
                print("FAILED")

        print(f"\nScoring {len(opus_4_6_sample)} Opus 4.6 tasks with Haiku...")
        for i, task in enumerate(opus_4_6_sample):
            print(f"  [{i+1}/{len(opus_4_6_sample)}] {task['task_id'][:20]}...", end=" ")
            score = score_task(task, 'opus-4-6')
            if score:
                opus_4_6_scores.append(score)
                print(f"overall={score.overall}")
            else:
                print("FAILED")

        # Save scores
        with open(scores_file, 'w') as f:
            json.dump({
                'opus-4-5': [s.__dict__ for s in opus_4_5_scores],
                'opus-4-6': [s.__dict__ for s in opus_4_6_scores],
                'timestamp': timestamp
            }, f, indent=2)
        print(f"\nSaved scores to {scores_file}")

    # Generate comparison report
    print("\nGenerating comparison report with Sonnet...")
    report = generate_comparison_report(opus_4_5_scores, opus_4_6_scores, opus_4_5_tasks, opus_4_6_tasks)

    report_file = args.output / f'comparison-{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write(f"# Model Comparison Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(report)
    print(f"Saved report to {report_file}")

    # Also save as latest
    latest_file = args.output / 'comparison-latest.md'
    with open(latest_file, 'w') as f:
        f.write(f"# Model Comparison Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(report)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    opus_4_5_agg = aggregate_scores(opus_4_5_scores)
    opus_4_6_agg = aggregate_scores(opus_4_6_scores)

    print(f"\nOpus 4.5 (n={opus_4_5_agg.get('count', 0)}): overall={opus_4_5_agg.get('overall', 'N/A')}/5")
    print(f"Opus 4.6 (n={opus_4_6_agg.get('count', 0)}): overall={opus_4_6_agg.get('overall', 'N/A')}/5")

    print(f"\nFull report: {report_file}")


if __name__ == '__main__':
    main()
