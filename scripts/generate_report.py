#!/usr/bin/env python3
"""
Generate an HTML report using Claude Code SDK.

Reads data files + style template, constructs a prompt, and calls
`claude -p` to generate the final report HTML.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output dist/public/report.html
    python scripts/generate_report.py --model claude-opus-4-6
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
        return {}


def _condense_stat_tests(data: dict) -> dict:
    """Extract key results from stat-tests.json without the full category bloat."""
    condensed = {'metadata': data.get('metadata', {}), 'significant_findings': [], 'non_significant': []}

    for section_key in ['overall'] + [k for k in data if k.startswith('complexity_')]:
        section = data.get(section_key, {})
        label = section.get('label', section_key)

        for test_type in ['chi_square', 'mann_whitney', 'proportions']:
            for test in section.get(test_type, []):
                entry = {
                    'section': label,
                    'test': test.get('test', ''),
                    'field': test.get('field', ''),
                    'p_value': test.get('p_value'),
                    'significant_p05': test.get('significant_p05', False),
                }
                # Add effect size info
                for key in ['cohens_h', 'cohens_d', 'cramers_v', 'effect_size_label',
                            'proportion_a', 'proportion_b', 'ci_a', 'ci_b',
                            'median_a', 'median_b']:
                    if key in test:
                        entry[key] = test[key]

                if test.get('significant_p05'):
                    condensed['significant_findings'].append(entry)
                else:
                    condensed['non_significant'].append(entry)

    # Trim non-significant to just counts per section
    non_sig_summary = {}
    for item in condensed['non_significant']:
        sec = item['section']
        non_sig_summary[sec] = non_sig_summary.get(sec, 0) + 1
    condensed['non_significant_summary'] = non_sig_summary
    condensed['non_significant_count'] = len(condensed['non_significant'])
    del condensed['non_significant']

    return condensed


def build_data_summary(base_dir: Path) -> str:
    """Collect and format all data files into a single context string."""
    sections = []

    # Aggregate stats
    agg = load_json(base_dir / 'dist' / 'public' / 'aggregate-stats.json')
    if agg:
        sections.append("## Aggregate Statistics\n```json\n" +
                        json.dumps(agg, indent=2) + "\n```")

    # Behavior metrics
    beh = load_json(base_dir / 'analysis' / 'behavior-metrics.json')
    if beh:
        sections.append("## Behavioral Metrics\n```json\n" +
                        json.dumps(beh, indent=2) + "\n```")

    # Dissatisfaction audit summary
    dis = load_json(base_dir / 'analysis' / 'dissatisfaction-audit.json')
    if dis and 'summary' in dis:
        sections.append("## Dissatisfaction Audit Summary\n```json\n" +
                        json.dumps(dis['summary'], indent=2) + "\n```")

    # Scores summary (aggregate, not per-task)
    scores = load_json(base_dir / 'analysis' / 'scores-latest.json')
    if scores:
        summary = {}
        for model_key, task_list in scores.items():
            if not isinstance(task_list, list):
                continue
            n = len(task_list)
            if n == 0:
                continue
            avg = lambda field: round(sum(t.get(field, 0) for t in task_list) / n, 2)
            summary[model_key] = {
                'task_count': n,
                'avg_efficiency': avg('efficiency'),
                'avg_correctness': avg('correctness'),
                'avg_communication': avg('communication'),
                'avg_autonomy': avg('autonomy'),
                'avg_friction': avg('friction'),
                'avg_overall': avg('overall'),
            }
        if summary:
            sections.append("## LLM Quality Scores (Averages)\n```json\n" +
                            json.dumps(summary, indent=2) + "\n```")

    # Matched pairs (if exists)
    pairs = load_json(base_dir / 'analysis' / 'matched-pairs.json')
    if pairs and 'summary' in pairs:
        sections.append("## Matched Pairs Summary\n```json\n" +
                        json.dumps(pairs['summary'], indent=2) + "\n```")

    # Statistical significance tests (condensed summary only)
    stat_tests = load_json(base_dir / 'analysis' / 'stat-tests.json')
    if stat_tests:
        summary = _condense_stat_tests(stat_tests)
        sections.append("## Statistical Significance Tests\n```json\n" +
                        json.dumps(summary, indent=2) + "\n```")

    # Session-level analysis
    session_analysis = load_json(base_dir / 'analysis' / 'session-analysis.json')
    if session_analysis:
        sections.append("## Session-Level Analysis\n```json\n" +
                        json.dumps(session_analysis, indent=2) + "\n```")

    # Key narrative findings from the org report (condensed, not full text)
    sections.append("""## Key Narrative Findings

These findings come from the detailed org-mode analysis report and should inform the HTML report's prose:

### Planning: Correlation vs Causation
- Opus 4.6 uses planning mode 10x more frequently (11.5% vs 1.1% of tasks)
- But planning only improves alignment scores by +0.04
- The planning-quality correlation may reflect a common cause (model thoroughness) rather than direct planning-to-outcome effect
- Planning may primarily benefit truly complex tasks, not provide uniform advantage

### Refactoring: A False Signal
- Original claim: Opus 4.6 has higher refactoring dissatisfaction (9.1% vs 5.9%)
- After audit: ALL flagged Opus 4.6 refactoring dissatisfaction (2 of 2) were false positives ("fix" in task requests)
- Opus 4.5's 2 flagged cases are unaudited but follow the same false-positive pattern
- With only 22-34 tasks per model, a single misclassification shifts rates by 3-4.5pp
- Corrected: both models show ~0% true refactoring dissatisfaction
- Behavioral observation stands: Opus 4.6 does explore before refactoring (adds overhead but no quality difference)
- DO NOT recommend routing refactoring to Opus 4.5 — the data does not support it

### Invisible Subagent Phenomenon
- Opus 4.6 uses MORE subagents overall (186 vs 169) but 72% are lightweight Explore agents
- Only 16% are general-purpose (implementation-capable)
- Subagents feel "invisible" because they're quiet research agents, not large implementation workers
- Opus 4.5 inverts: 50% general-purpose, 32% Explore

### Dissatisfaction False Positive Correction
- Original rates: Opus 4.5=7.1%, Opus 4.6=8.7%
- Corrected rates: Opus 4.5=3.7%, Opus 4.6=1.3%
- False positive rates: 73% (Opus 4.5), 93% (Opus 4.6)
- Higher FP for Opus 4.6 due to heavier subagent delegation generating "fix" keywords

### Complexity Crossover
- Trivial tasks: Opus 4.5 leads by +7pp in satisfaction
- Complex tasks: Opus 4.6 leads by +11pp in satisfaction
- Crossover occurs around "simple" complexity level
""")


    return "\n\n".join(sections)


def build_prompt(base_dir: Path) -> str:
    """Construct the full prompt for Claude."""
    # Load the generation instructions
    prompt_path = base_dir / 'prompts' / 'report-generation.md'
    with open(prompt_path) as f:
        instructions = f.read()

    # Load the style reference
    style_path = Path(__file__).resolve().parent / 'report-style.html'
    with open(style_path) as f:
        style_ref = f.read()

    # Build data summary
    data = build_data_summary(base_dir)

    prompt = f"""{instructions}

## Style Reference

The following HTML file shows the complete CSS and all available components.
Copy the CSS exactly. Use the HTML patterns as templates for your content.

```html
{style_ref}
```

## Data

The following data should drive all numbers in the report.
Calculate percentages and comparisons from these raw values.

{data}

## Task

Generate the complete HTML report now. Output ONLY the HTML, starting with `<!DOCTYPE html>` and ending with `</html>`. No markdown fences, no commentary.
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report via Claude SDK")
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file path (default: dist/public/report.html)',
    )
    parser.add_argument(
        '--model', '-m',
        default='claude-opus-4-6',
        help='Model to use for generation (default: claude-opus-4-6)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the prompt instead of calling Claude',
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=None,
        help='Base directory of the model-comparison project',
    )
    args = parser.parse_args()

    # Find base directory
    if args.base_dir:
        base_dir = args.base_dir
    else:
        # Try relative to script location
        base_dir = Path(__file__).resolve().parent.parent

    script_dir = Path(__file__).resolve().parent
    if not (script_dir / 'report-style.html').exists():
        print(f"Error: Cannot find report-style.html in {script_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (base_dir / 'dist' / 'public' / 'report.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(base_dir)

    if args.dry_run:
        print(prompt)
        print(f"\n--- Prompt length: {len(prompt)} chars ---", file=sys.stderr)
        return

    print(f"Generating report with {args.model}...", file=sys.stderr)
    print(f"Prompt length: {len(prompt)} chars", file=sys.stderr)

    # Write prompt to a temp file to avoid CLI argument length issues
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write(prompt)
        tmp_path = tmp.name

    try:
        # Pipe via stdin; disable tools so output goes to stdout
        with open(tmp_path) as f:
            result = subprocess.run(
                ['claude', '-p', '--model', args.model, '--tools', ''],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=600,
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"Error: claude exited with code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    html = result.stdout.strip()

    # Strip any preamble before <!DOCTYPE html>
    doctype_idx = html.find('<!DOCTYPE html>')
    if doctype_idx < 0:
        doctype_idx = html.find('<!doctype html>')
    if doctype_idx > 0:
        html = html[doctype_idx:]

    # Strip anything after </html>
    end_idx = html.rfind('</html>')
    if end_idx > 0:
        html = html[:end_idx + len('</html>')]

    # Strip markdown fences if the model wrapped the output
    if html.startswith('```'):
        lines = html.split('\n')
        if lines[-1].strip() == '```':
            lines = lines[1:-1]
        elif lines[0].startswith('```'):
            lines = lines[1:]
        html = '\n'.join(lines)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Report written to {output_path}", file=sys.stderr)
    print(f"Output size: {len(html)} chars", file=sys.stderr)


if __name__ == '__main__':
    main()
