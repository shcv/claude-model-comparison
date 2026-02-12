#!/usr/bin/env python3
"""
Task Classification for Model Comparison

Categorizes tasks by:
1. Purpose/Type: investigation, bugfix, feature, greenfield, refactor, sysadmin, docs, continuation, port
2. Complexity: trivial, simple, moderate, complex, major
3. Domain: frontend, backend, devops, data, mixed
"""

import hashlib
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models, load_canonical_tasks


# Task type patterns (checked in order, first match wins for primary type)
TYPE_PATTERNS = {
    'continuation': [
        r'^continue\.?\.?\.?$',
        r'^\[request interrupted',
        r'^ok\.?$',
        r'^yes\.?$',
        r'^no\.?$',
        r'^go ahead',
        r'^proceed',
        r'^<task-notification>',
        r'^<teammate-message>',
        r'^This session is being continued',
        r'^sure\.?$',
        r'^sounds good',
        r'^yep',
        r'^yeah',
    ],
    'investigation': [
        r'\b(investigate|explore|understand|explain|what is|how does|where is|find out|look at|analyze|research|review|compare|check)\b',
        r'\b(show me|tell me about|describe|summarize|list)\b',
        r'^(what|where|how|why|can you|is there)\b',
        r'\b(deep investigation|deep dive)\b',
    ],
    'bugfix': [
        r'\b(fix|bug|broken|error|issue|wrong|doesn\'t work|not working|fails|crash|exception)\b',
        r'\b(debug|troubleshoot|diagnose|resolve)\b',
    ],
    'refactor': [
        r'\b(refactor|restructure|reorganize|clean ?up|simplify|rename|move|extract|consolidate)\b',
        r'\b(split|merge|dedupe|deduplicate|factor out|remove|removing)\b',
    ],
    'port': [
        r'\b(port|convert|migrate|translate|rewrite in|switch to)\b',
        r'\b(from .+ to .+)\b',
    ],
    'greenfield': [
        r'\b(create|new|implement from scratch|build|scaffold|bootstrap|initialize|setup|set up)\b',
        r'^(make|write|create|build|implement)\s+(a|an|the|new)\b',
    ],
    'feature': [
        r'\b(add|implement|feature|enhance|extend|support|enable|allow|include)\b',
        r'\b(integrate|hook up|wire up|connect)\b',
        r'implement the following plan',
        r'\bshould be\b',  # "the visual should be X"
        r'\buse .+ instead of\b',  # "use X instead of Y"
        r'\bmake it\b',  # "make it look like..."
        r'\bswap\b',  # "swap the order"
    ],
    'sysadmin': [
        r'\b(git|commit|push|pull|merge|rebase|branch|deploy|install|configure|docker|npm|pip|chmod|chown)\b',
        r'\b(server|service|daemon|cron|systemd|nginx|apache)\b',
        r'^(run|execute|start|stop|restart|try|test|build)\b',
        r'\blet\'s try\b',
        r'\brun \d+ instance',
        r'\bnot working\b',
        r'\bdoesn\'t seem to be\b',
    ],
    'docs': [
        r'\b(document|readme|comment|docstring|jsdoc|explain in|write up|changelog)\b',
        r'\b(update docs|add documentation)\b',
    ],
}

# Complexity indicators
COMPLEXITY_SIGNALS = {
    'trivial': {
        'max_tools': 3,
        'max_files': 1,
        'max_lines': 20,
        'patterns': [r'\b(quick|simple|just|only)\b'],
    },
    'simple': {
        'max_tools': 10,
        'max_files': 3,
        'max_lines': 100,
        'patterns': [r'\b(small|minor|single)\b'],
    },
    'moderate': {
        'max_tools': 30,
        'max_files': 10,
        'max_lines': 500,
        'patterns': [],
    },
    'complex': {
        'max_tools': 80,
        'max_files': 25,
        'max_lines': 2000,
        'patterns': [r'\b(comprehensive|full|complete|entire)\b'],
    },
    'major': {
        'max_tools': float('inf'),
        'max_files': float('inf'),
        'max_lines': float('inf'),
        'patterns': [r'\b(overhaul|rewrite|major|massive|huge)\b'],
    },
}

# Domain detection based on file extensions and paths
DOMAIN_PATTERNS = {
    'frontend': [r'\.(tsx?|jsx?|vue|svelte|css|scss|html)$', r'(components?|pages?|views?|ui)/'],
    'backend': [r'\.(py|go|rs|java|rb|php)$', r'(api|server|services?|handlers?)/'],
    'devops': [r'(docker|kubernetes|terraform|ansible|\.ya?ml$|Makefile|\.sh$)', r'(deploy|infra|ci)/'],
    'data': [r'\.(sql|csv|json|parquet)$', r'(data|models?|schemas?|migrations?)/'],
    'docs': [r'\.(md|rst|txt|org)$', r'(docs?|readme)/i'],
}


@dataclass
class TaskClassification:
    task_id: str
    primary_type: str
    secondary_types: list
    complexity: str
    domain: str
    confidence: float  # 0-1, how confident we are in classification
    signals: list  # What patterns/signals led to this classification


def classify_type(prompt: str, tool_sequence: str = "") -> tuple[str, list, list]:
    """Classify task type from prompt and tools."""
    prompt_lower = prompt.lower()
    matched_types = []
    signals = []

    for task_type, patterns in TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                matched_types.append(task_type)
                signals.append(f"{task_type}: matched '{pattern}'")
                break  # Only count each type once

    # If we have "Implement the following plan:" it's likely feature or greenfield
    if re.search(r'implement the following plan', prompt_lower):
        if 'greenfield' not in matched_types and 'feature' not in matched_types:
            matched_types.insert(0, 'feature')
            signals.append("feature: 'implement the following plan' pattern")

    # Default fallback
    if not matched_types:
        matched_types = ['unknown']
        signals.append("unknown: no patterns matched")

    primary = matched_types[0]
    secondary = matched_types[1:] if len(matched_types) > 1 else []

    return primary, secondary, signals


def classify_complexity(task: dict) -> tuple[str, list]:
    """Classify task complexity from metrics."""
    tools = len(task.get('tool_calls', []))
    files = task.get('total_files_touched', 0)
    lines = task.get('total_lines_added', 0) + task.get('total_lines_removed', 0)
    prompt = task.get('user_prompt', '').lower()
    signals = []

    # Check pattern-based complexity first
    for level in ['trivial', 'major', 'complex']:  # Check extremes first
        for pattern in COMPLEXITY_SIGNALS[level]['patterns']:
            if re.search(pattern, prompt):
                signals.append(f"{level}: prompt matched '{pattern}'")

    # Metric-based classification
    for level in ['trivial', 'simple', 'moderate', 'complex', 'major']:
        thresholds = COMPLEXITY_SIGNALS[level]
        if (tools <= thresholds['max_tools'] and
            files <= thresholds['max_files'] and
            lines <= thresholds['max_lines']):
            signals.append(f"{level}: tools={tools}, files={files}, lines={lines}")
            return level, signals

    return 'major', signals


def classify_domain(task: dict) -> str:
    """Classify domain from files touched."""
    files = (task.get('files_read', []) +
             task.get('files_written', []) +
             task.get('files_edited', []))

    domain_scores = {d: 0 for d in DOMAIN_PATTERNS}

    for file_path in files:
        for domain, patterns in DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    domain_scores[domain] += 1
                    break

    if not any(domain_scores.values()):
        return 'unknown'

    # Return highest scoring domain, or 'mixed' if close
    sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
    if sorted_domains[0][1] == 0:
        return 'unknown'
    if len(sorted_domains) > 1 and sorted_domains[1][1] >= sorted_domains[0][1] * 0.7:
        return 'mixed'
    return sorted_domains[0][0]


def classify_task(task: dict) -> TaskClassification:
    """Fully classify a task."""
    primary_type, secondary_types, type_signals = classify_type(
        task.get('user_prompt', ''),
        task.get('tool_sequence', '')
    )
    complexity, complexity_signals = classify_complexity(task)
    domain = classify_domain(task)

    # Calculate confidence based on signal strength
    confidence = min(1.0, len(type_signals) * 0.3)
    if primary_type == 'unknown':
        confidence *= 0.5

    return TaskClassification(
        task_id=task['task_id'],
        primary_type=primary_type,
        secondary_types=secondary_types,
        complexity=complexity,
        domain=domain,
        confidence=confidence,
        signals=type_signals + complexity_signals,
    )


def classify_with_llm(task: dict, model: str = "haiku") -> Optional[TaskClassification]:
    """Use LLM for more accurate classification of ambiguous tasks."""
    prompt = f"""Classify this coding task. Return ONLY a JSON object, no other text.

Task prompt: {task.get('user_prompt', '')[:500]}

Tool calls: {len(task.get('tool_calls', []))}
Files touched: {task.get('total_files_touched', 0)}
Lines added/removed: {task.get('total_lines_added', 0)}/{task.get('total_lines_removed', 0)}
Duration: {task.get('duration_seconds', 0):.0f}s

Categories:
- Type: investigation, bugfix, refactor, port, greenfield, feature, sysadmin, docs, continuation
- Complexity: trivial, simple, moderate, complex, major
- Domain: frontend, backend, devops, data, docs, mixed, unknown

Return JSON: {{"type": "...", "secondary_types": [...], "complexity": "...", "domain": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/tmp"
        )
        if result.returncode != 0:
            return None

        response = result.stdout.strip()
        # Find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            return TaskClassification(
                task_id=task['task_id'],
                primary_type=data.get('type', 'unknown'),
                secondary_types=data.get('secondary_types', []),
                complexity=data.get('complexity', 'moderate'),
                domain=data.get('domain', 'unknown'),
                confidence=data.get('confidence', 0.7),
                signals=[f"LLM: {data.get('reasoning', 'no reasoning')}"],
            )
    except Exception as e:
        print(f"LLM classification failed: {e}")
    return None


LLM_COMPLEXITY_PROMPT = """Rate the conceptual complexity of this coding task (not the amount of code produced):

- trivial: Quick answer, single change, obvious fix
- simple: Clear task, limited scope, straightforward approach
- moderate: Requires understanding context, multiple considerations
- complex: Architectural decisions, multi-system impact, tricky trade-offs
- major: Large-scope redesign, many interacting concerns

User request: {prompt}

Return ONLY a JSON object: {{"complexity": "...", "reasoning": "one sentence explanation"}}"""


def classify_complexity_llm(task: dict, llm_model: str = "haiku",
                            cache_dir: Optional[Path] = None) -> Optional[str]:
    """Use LLM to classify conceptual complexity from prompt text alone.

    Returns the complexity level string, or None on failure.
    """
    prompt_text = task.get('user_prompt', '')[:500]
    if not prompt_text.strip():
        return None

    full_prompt = LLM_COMPLEXITY_PROMPT.format(prompt=prompt_text)

    # Check cache
    prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()[:12]
    task_id = task.get('task_id', 'unknown')
    if cache_dir:
        cache_file = cache_dir / f"{task_id}_{prompt_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                return cached.get('complexity')
            except Exception:
                pass

    try:
        result = subprocess.run(
            ["claude", "-p", full_prompt, "--model", llm_model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/tmp",
        )
        if result.returncode != 0:
            return None

        response = result.stdout.strip()
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            complexity = data.get('complexity', '').lower().strip()
            valid = {'trivial', 'simple', 'moderate', 'complex', 'major'}
            if complexity not in valid:
                return None

            # Cache result
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"{task_id}_{prompt_hash}.json"
                with open(cache_file, 'w') as f:
                    json.dump({'complexity': complexity, 'reasoning': data.get('reasoning', '')}, f)

            return complexity
    except Exception:
        pass
    return None


def _reclassify_single(task: dict, llm_model: str, cache_dir: Optional[Path],
                       lock: threading.Lock, counter: list, total: int) -> tuple[str, Optional[str]]:
    """Reclassify a single task's complexity (for parallel execution)."""
    task_id = task.get('task_id', 'unknown')
    result = classify_complexity_llm(task, llm_model, cache_dir)

    with lock:
        counter[0] += 1
        status = result or 'failed'
        print(f"  [{counter[0]}/{total}] {task_id[:25]}... {status}")

    return task_id, result


def reclassify_complexity_batch(tasks: list[dict], llm_model: str = "haiku",
                                cache_dir: Optional[Path] = None,
                                max_workers: int = 8) -> dict[str, str]:
    """Reclassify complexity for all tasks using LLM. Returns {task_id: complexity}."""
    print(f"Reclassifying {len(tasks)} tasks with LLM ({max_workers} workers)...")

    lock = threading.Lock()
    counter = [0]
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_reclassify_single, task, llm_model, cache_dir, lock, counter, len(tasks)): task
            for task in tasks
        }
        for future in as_completed(futures):
            try:
                task_id, complexity = future.result()
                if complexity:
                    results[task_id] = complexity
            except Exception as e:
                print(f"  Worker error: {e}")

    return results


def load_and_classify(data_dir: Path, model: str, use_llm: bool = False) -> list[dict]:
    """Load tasks and add classifications."""
    tasks = load_canonical_tasks(data_dir, model)

    classified = []
    for task in tasks:
        # Use heuristic classification
        classification = classify_task(task)

        # Optionally enhance with LLM for low-confidence cases
        if use_llm and classification.confidence < 0.5:
            llm_class = classify_with_llm(task)
            if llm_class:
                classification = llm_class

        # Add classification to task
        task['classification'] = {
            'type': classification.primary_type,
            'secondary_types': classification.secondary_types,
            'complexity': classification.complexity,
            'domain': classification.domain,
            'confidence': classification.confidence,
            'signals': classification.signals,
        }
        classified.append(task)

    return classified


def print_distribution(tasks: list[dict], model: str):
    """Print classification distribution."""
    print(f"\n{model.upper()} Classification ({len(tasks)} tasks)")
    print("=" * 50)

    # Type distribution
    types = {}
    for t in tasks:
        task_type = t.get('classification', {}).get('type', 'unknown')
        types[task_type] = types.get(task_type, 0) + 1

    print("\nBy Type:")
    for task_type, count in sorted(types.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(tasks)
        bar = '█' * int(pct / 2)
        print(f"  {task_type:<15} {count:3d} ({pct:5.1f}%) {bar}")

    # Complexity distribution
    complexities = {}
    for t in tasks:
        complexity = t.get('classification', {}).get('complexity', 'unknown')
        complexities[complexity] = complexities.get(complexity, 0) + 1

    print("\nBy Complexity:")
    for level in ['trivial', 'simple', 'moderate', 'complex', 'major']:
        count = complexities.get(level, 0)
        pct = 100 * count / len(tasks) if tasks else 0
        bar = '█' * int(pct / 2)
        print(f"  {level:<15} {count:3d} ({pct:5.1f}%) {bar}")

    # Domain distribution
    domains = {}
    for t in tasks:
        domain = t.get('classification', {}).get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1

    print("\nBy Domain:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(tasks)
        print(f"  {domain:<15} {count:3d} ({pct:5.1f}%)")


def save_classified(tasks: list[dict], output_file: Path):
    """Save classified tasks."""
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"\nSaved {len(tasks)} classified tasks to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Classify tasks by type and complexity')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for low-confidence cases')
    parser.add_argument('--model', default='both',
                        help='Model to classify (or "both" for all discovered models)')
    parser.add_argument('--reclassify-complexity', action='store_true',
                        help='Add LLM-based conceptual complexity (llm_complexity field)')
    parser.add_argument('--llm-model', default='haiku',
                        help='LLM model for complexity reclassification (default: haiku)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel workers for LLM reclassification (default: 2)')
    args = parser.parse_args()

    print("=" * 60)
    print("TASK CLASSIFICATION")
    print("=" * 60)

    models = discover_models(args.data_dir, prefix="tasks-canonical") if args.model == 'both' else [args.model]

    for model in models:
        tasks = load_and_classify(args.data_dir, model, use_llm=args.use_llm)
        if not tasks:
            print(f"\nNo tasks found for {model}")
            continue

        # LLM-based conceptual complexity reclassification
        if args.reclassify_complexity:
            cache_dir = args.data_dir / 'llm-complexity-cache'
            results = reclassify_complexity_batch(
                tasks, llm_model=args.llm_model,
                cache_dir=cache_dir, max_workers=args.workers
            )
            # Apply results
            applied = 0
            for task in tasks:
                tid = task.get('task_id')
                if tid in results:
                    task['classification']['llm_complexity'] = results[tid]
                    applied += 1
            print(f"\n  Applied LLM complexity to {applied}/{len(tasks)} tasks")

            # Show divergence between size-based and conceptual complexity
            divergent = 0
            for task in tasks:
                cls = task.get('classification', {})
                if cls.get('llm_complexity') and cls['llm_complexity'] != cls['complexity']:
                    divergent += 1
            print(f"  Divergent classifications: {divergent} ({100*divergent/len(tasks):.1f}%)")

        print_distribution(tasks, model)

        output_file = args.data_dir / f'tasks-classified-{model}.json'
        save_classified(tasks, output_file)


if __name__ == '__main__':
    main()
