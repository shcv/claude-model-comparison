#!/usr/bin/env python3
"""Backfill task_type into existing annotation cache files.

Reads all cache files in annotation-cache/, finds entries missing task_type,
calls a lightweight LLM classifier in parallel, and updates the cache files.

Usage:
    python scripts/backfill_task_type.py --dir comparisons/opus-4.5-vs-4.6 [--workers 16]
"""

import json
import glob
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VALID_TASK_TYPES = {
    "investigation", "bugfix", "feature", "greenfield", "refactor",
    "sysadmin", "docs", "continuation", "port",
}

TASK_TYPE_PROMPT = """Classify this Claude Code task by its primary purpose. Choose exactly ONE from:
- investigation: Research, exploration, understanding, "what is", "how does", reviewing code
- bugfix: Fixing errors, debugging, resolving issues, "doesn't work", troubleshooting
- feature: Adding capability to existing code, enhancing, extending, integrating
- greenfield: Creating something new from scratch, scaffolding, bootstrapping
- refactor: Restructuring existing code without changing behavior, renaming, cleanup
- sysadmin: Git operations, deployment, configuration, running commands, testing
- docs: Documentation, READMEs, comments, changelogs
- continuation: Minimal response (ok, yes, go ahead), session handoff, not a real task
- port: Migrating between technologies, converting formats

Very short prompts (<5 words) that aren't actionable should be "continuation".

**User prompt:** "{user_prompt}"
**Work summary:** {summary}

Return ONLY a JSON object: {{"task_type": "<type>", "confidence": "high|medium|low"}}"""


def classify_single(cache_path, task_data, llm_model, lock, counter, total):
    """Classify a single task and update its cache file."""
    # Use summary and work_category from the cached annotation for context
    summary = task_data.get('summary', '')
    work_cat = task_data.get('work_category', '')
    context = summary or work_cat or 'no context'

    # We need the user prompt — extract task_id from filename to find canonical task
    # But we don't have it readily. Use the summary/work_category as proxy.
    # Actually, we need the original user prompt. Let's read it from canonical tasks.
    task_id = cache_path.stem.rsplit('_', 1)[0]  # Remove hash suffix

    prompt = TASK_TYPE_PROMPT.format(
        user_prompt=context[:500],
        summary=summary[:200],
    )

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", llm_model, "--output-format", "text"],
            capture_output=True, text=True, timeout=30, cwd="/tmp",
        )
        if result.returncode == 0:
            response = result.stdout.strip()
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                task_type = parsed.get('task_type', '').lower().strip()
                confidence = parsed.get('confidence', 'medium')
                if task_type in VALID_TASK_TYPES:
                    task_data['task_type'] = task_type
                    task_data['task_type_confidence'] = confidence
                    with open(cache_path, 'w') as f:
                        json.dump(task_data, f, indent=2)
                    with lock:
                        counter[0] += 1
                        if counter[0] % 50 == 0:
                            print(f"  [{counter[0]}/{total}] classified...")
                    return True
    except Exception:
        pass

    with lock:
        counter[1] += 1
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--model', default='haiku')
    args = parser.parse_args()

    comparison_dir = Path(args.dir)
    cache_dir = comparison_dir / 'analysis' / 'annotation-cache'

    # Find all cache files missing task_type
    needs_backfill = []
    has_type = 0
    for f in sorted(glob.glob(str(cache_dir / '*.json'))):
        path = Path(f)
        with open(path) as fh:
            data = json.load(fh)
        if data.get('task_type'):
            has_type += 1
        else:
            needs_backfill.append((path, data))

    total = len(needs_backfill)
    print(f"Cache files: {has_type + total} total, {has_type} have task_type, {total} need backfill")

    if total == 0:
        print("Nothing to backfill!")
        return

    # But we need user prompts for classification. Let's load canonical tasks for context.
    data_dir = comparison_dir / 'data'
    task_prompts = {}
    for f in sorted(data_dir.glob('tasks-canonical-*.json')):
        with open(f) as fh:
            tasks = json.load(fh)
        for t in tasks:
            task_prompts[t['task_id']] = t.get('user_prompt', '')[:500]

    # Enrich cache data with user prompts for better classification
    enriched = []
    for path, data in needs_backfill:
        task_id = path.stem.rsplit('_', 1)[0]
        user_prompt = task_prompts.get(task_id, '')
        if user_prompt:
            data['_user_prompt'] = user_prompt
        enriched.append((path, data))

    # Override the prompt to use user_prompt when available
    global TASK_TYPE_PROMPT
    TASK_TYPE_PROMPT = """Classify this Claude Code task by its primary purpose. Choose exactly ONE from:
- investigation: Research, exploration, understanding, "what is", "how does", reviewing code
- bugfix: Fixing errors, debugging, resolving issues, "doesn't work", troubleshooting
- feature: Adding capability to existing code, enhancing, extending, integrating
- greenfield: Creating something new from scratch, scaffolding, bootstrapping
- refactor: Restructuring existing code without changing behavior, renaming, cleanup
- sysadmin: Git operations, deployment, configuration, running commands, testing
- docs: Documentation, READMEs, comments, changelogs
- continuation: Minimal response (ok, yes, go ahead), session handoff, not a real task
- port: Migrating between technologies, converting formats

Very short prompts (<5 words) that aren't actionable should be "continuation".

**User prompt:** "{user_prompt}"
**Work done:** {summary}

Return ONLY a JSON object: {{"task_type": "<type>", "confidence": "high|medium|low"}}"""

    lock = threading.Lock()
    counter = [0, 0]  # [success, failure]

    print(f"Backfilling {total} tasks with {args.workers} workers...")

    def classify_enriched(item):
        path, data = item
        user_prompt = data.pop('_user_prompt', data.get('summary', ''))
        summary = data.get('work_category', data.get('summary', ''))

        prompt = TASK_TYPE_PROMPT.format(
            user_prompt=user_prompt[:500],
            summary=summary[:200],
        )

        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", args.model, "--output-format", "text"],
                capture_output=True, text=True, timeout=30, cwd="/tmp",
            )
            if result.returncode == 0:
                response = result.stdout.strip()
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(response[start:end])
                    task_type = parsed.get('task_type', '').lower().strip()
                    confidence = parsed.get('confidence', 'medium')
                    if task_type in VALID_TASK_TYPES:
                        data['task_type'] = task_type
                        data['task_type_confidence'] = confidence
                        with open(path, 'w') as f:
                            json.dump(data, f, indent=2)
                        with lock:
                            counter[0] += 1
                            if counter[0] % 100 == 0:
                                print(f"  [{counter[0]}/{total}] classified...")
                        return True
        except Exception:
            pass

        with lock:
            counter[1] += 1
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(classify_enriched, item) for item in enriched]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)

    print(f"\nDone! Classified: {counter[0]}, Failed: {counter[1]}")


if __name__ == '__main__':
    main()
