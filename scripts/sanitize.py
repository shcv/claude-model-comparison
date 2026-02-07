#!/usr/bin/env python3
"""
Data Sanitization for Publication

Produces two output tiers:
- dist/private/: Full data with personal paths anonymized
- dist/public/: Aggregate statistics only, no raw task data

Sanitization:
- /home/shcv/ → /home/user/
- Project paths → project-01, project-02, etc.
- Session file_path fields stripped (absolute paths to JSONL files)
"""

import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Model name mapping
MODEL_NAMES = {
    'opus-4-5': 'Opus 4.5',
    'opus-4-6': 'Opus 4.6',
}


def build_project_mapping(data_dir: Path) -> dict[str, str]:
    """Scan all JSON data files to find unique project paths and assign labels."""
    project_paths = set()

    for json_file in sorted(data_dir.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    pp = item.get('project_path', '')
                    if pp:
                        project_paths.add(pp)

    # Sort for deterministic ordering
    sorted_paths = sorted(project_paths)
    mapping = {}
    for i, path in enumerate(sorted_paths, 1):
        mapping[path] = f'project-{i:02d}'

    return mapping


def sanitize_value(value: Any, project_mapping: dict[str, str]) -> Any:
    """Recursively sanitize a value."""
    if isinstance(value, str):
        # Replace home directory
        value = value.replace('/home/shcv/', '/home/user/')
        value = value.replace('/home/shcv', '/home/user')

        # Replace project paths
        for original, anonymized in project_mapping.items():
            value = value.replace(original, anonymized)

        return value

    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            # Strip file_path fields with absolute session paths
            if k == 'file_path' and isinstance(v, str) and v.startswith('/'):
                result[k] = '[redacted]'
            else:
                result[k] = sanitize_value(v, project_mapping)
        return result

    elif isinstance(value, list):
        return [sanitize_value(item, project_mapping) for item in value]

    return value


def sanitize_json_file(input_path: Path, output_path: Path,
                       project_mapping: dict[str, str]) -> None:
    """Sanitize a single JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    sanitized = sanitize_value(data, project_mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sanitized, f, indent=2)


def generate_aggregate_stats(data_dir: Path, analysis_dir: Path) -> dict:
    """Generate aggregate statistics without per-task data."""
    stats = {}

    for model in ['opus-4-5', 'opus-4-6']:
        model_stats = {'display_name': MODEL_NAMES.get(model, model)}

        # Session stats
        sessions_file = data_dir / f'sessions-{model}.json'
        if sessions_file.exists():
            with open(sessions_file) as f:
                sessions = json.load(f)
            model_stats['session_count'] = len(sessions)
            model_stats['total_duration_minutes'] = sum(
                s.get('duration_minutes', 0) for s in sessions
            )

        # Task stats (from classified)
        tasks_file = data_dir / f'tasks-classified-{model}.json'
        if tasks_file.exists():
            with open(tasks_file) as f:
                tasks = json.load(f)
            model_stats['task_count'] = len(tasks)

            # Outcome distribution
            outcomes = {}
            for t in tasks:
                cat = t.get('outcome_category', 'unknown')
                outcomes[cat] = outcomes.get(cat, 0) + 1
            model_stats['outcome_distribution'] = outcomes

            # Complexity distribution
            complexities = {}
            for t in tasks:
                c = t.get('classification', {}).get('complexity', 'unknown')
                complexities[c] = complexities.get(c, 0) + 1
            model_stats['complexity_distribution'] = complexities

            # Type distribution
            types = {}
            for t in tasks:
                tp = t.get('classification', {}).get('type', 'unknown')
                types[tp] = types.get(tp, 0) + 1
            model_stats['type_distribution'] = types

            # Aggregate metrics
            with_changes = [t for t in tasks if t.get('total_files_touched', 0) > 0]
            if with_changes:
                model_stats['avg_tools_per_task'] = round(
                    sum(len(t.get('tool_calls', [])) for t in with_changes) / len(with_changes), 1
                )
                model_stats['avg_files_per_task'] = round(
                    sum(t['total_files_touched'] for t in with_changes) / len(with_changes), 1
                )
                model_stats['avg_lines_added'] = round(
                    sum(t.get('total_lines_added', 0) for t in with_changes) / len(with_changes), 1
                )

        # LLM analysis stats
        llm_file = analysis_dir / f'llm-analysis-{model}.json'
        if llm_file.exists():
            with open(llm_file) as f:
                analyses = json.load(f)

            scores = [a.get('alignment_score', 0) for a in analyses if a.get('alignment_score', 0) > 0]
            if scores:
                model_stats['avg_alignment_score'] = round(sum(scores) / len(scores), 2)

            # Sentiment distribution
            sentiments = {}
            for a in analyses:
                sent = a.get('user_sentiment', '').lower()
                if 'satisfied' in sent or 'positive' in sent:
                    key = 'satisfied'
                elif 'dissatisfied' in sent or 'negative' in sent:
                    key = 'dissatisfied'
                elif 'neutral' in sent:
                    key = 'neutral'
                else:
                    key = 'other'
                sentiments[key] = sentiments.get(key, 0) + 1
            model_stats['sentiment_distribution'] = sentiments

            # Completion distribution
            completions = {}
            for a in analyses:
                c = a.get('task_completion', 'unknown').lower()
                completions[c] = completions.get(c, 0) + 1
            model_stats['completion_distribution'] = completions

        stats[model] = model_stats

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sanitize data for publication')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--analysis-dir', type=Path, default=Path('analysis'))
    parser.add_argument('--output-dir', type=Path, default=Path('dist'))
    args = parser.parse_args()

    base_dir = Path('.')

    # Build project path mapping
    print("Building project path mapping...")
    project_mapping = build_project_mapping(args.data_dir)
    print(f"  Found {len(project_mapping)} unique project paths")

    # Save mapping (gitignored)
    mapping_file = base_dir / 'sanitize-mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(project_mapping, f, indent=2)
    print(f"  Mapping saved to {mapping_file}")

    # --- Private tier: full data, paths sanitized ---
    private_dir = args.output_dir / 'private'
    print(f"\nGenerating Private tier → {private_dir}/")

    # Sanitize data/ files
    for json_file in sorted(args.data_dir.glob('*.json')):
        out_file = private_dir / 'data' / json_file.name
        sanitize_json_file(json_file, out_file, project_mapping)
        print(f"  Sanitized {json_file.name}")

    # Sanitize analysis/ JSON files
    for json_file in sorted(args.analysis_dir.glob('*.json')):
        out_file = private_dir / 'analysis' / json_file.name
        sanitize_json_file(json_file, out_file, project_mapping)
        print(f"  Sanitized {json_file.name}")

    # Copy analysis org/md reports (sanitize text in them too)
    for report_file in sorted(args.analysis_dir.glob('*.*')):
        if report_file.suffix in ('.org', '.md'):
            out_file = private_dir / 'analysis' / report_file.name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            content = report_file.read_text()
            content = content.replace('/home/shcv/', '/home/user/')
            for original, anonymized in project_mapping.items():
                content = content.replace(original, anonymized)
            out_file.write_text(content)
            print(f"  Sanitized {report_file.name}")

    # Copy scripts
    src_scripts_dir = base_dir / 'scripts'
    out_scripts_dir = private_dir / 'scripts'
    out_scripts_dir.mkdir(parents=True, exist_ok=True)
    for py_file in sorted(src_scripts_dir.glob('*.py')):
        shutil.copy2(py_file, out_scripts_dir / py_file.name)
    print(f"  Copied {len(list(src_scripts_dir.glob('*.py')))} scripts")

    # Copy methodology
    if (base_dir / 'analysis-process.org').exists():
        shutil.copy2(base_dir / 'analysis-process.org', private_dir / 'analysis-process.org')

    # Copy prompts
    prompts_dir = base_dir / 'prompts'
    if prompts_dir.exists():
        out_prompts = private_dir / 'prompts'
        if out_prompts.exists():
            shutil.rmtree(out_prompts)
        shutil.copytree(prompts_dir, out_prompts)
        print(f"  Copied prompts/")

    # --- Public tier: aggregate only ---
    public_dir = args.output_dir / 'public'
    print(f"\nGenerating Public tier → {public_dir}/")
    public_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate statistics
    print("  Computing aggregate statistics...")
    aggregate = generate_aggregate_stats(args.data_dir, args.analysis_dir)
    with open(public_dir / 'aggregate-stats.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"  Saved aggregate-stats.json")

    # Copy methodology doc
    if (base_dir / 'analysis-process.org').exists():
        shutil.copy2(base_dir / 'analysis-process.org', public_dir / 'analysis-process.org')
        print(f"  Copied analysis-process.org")

    # Copy report if it exists
    if (base_dir / 'report.org').exists():
        shutil.copy2(base_dir / 'report.org', public_dir / 'report.org')
        print(f"  Copied report.org")

    # Copy scripts as reference implementation
    public_scripts = public_dir / 'scripts'
    public_scripts.mkdir(parents=True, exist_ok=True)
    for py_file in sorted(src_scripts_dir.glob('*.py')):
        if py_file.name != 'sanitize.py':
            shutil.copy2(py_file, public_scripts / py_file.name)
    print(f"  Copied analysis scripts")

    # --- Verification ---
    print("\n--- Verification ---")

    # Check no /home/shcv/ in private output
    violations = 0
    for out_file in private_dir.rglob('*'):
        if out_file.is_file() and out_file.suffix in ('.json', '.org', '.md'):
            content = out_file.read_text()
            if '/home/shcv/' in content:
                print(f"  WARNING: /home/shcv/ found in {out_file}")
                violations += 1

    if violations == 0:
        print("  No /home/shcv/ references in private tier")
    else:
        print(f"  {violations} files still contain /home/shcv/")

    # Check public tier has no raw task data
    public_json_files = list(public_dir.rglob('*.json'))
    for jf in public_json_files:
        with open(jf) as f:
            content = f.read()
        if '"task_id"' in content and '"user_prompt"' in content:
            print(f"  WARNING: {jf} may contain raw task data")

    print(f"\nDone. Private tier: {private_dir}, Public tier: {public_dir}")


if __name__ == '__main__':
    main()
