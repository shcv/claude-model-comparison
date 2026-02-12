#!/usr/bin/env python3
"""Evaluate task type classification quality.

Compares regex-based classification with LLM classification on a sample of tasks.
Reports agreement rates, unknown resolution, and confusion patterns.

Usage:
    python scripts/eval_task_types.py --dir comparisons/opus-4.5-vs-4.6 [--n-sample 50] [--model haiku]
"""

import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID_TYPES = [
    "investigation", "bugfix", "feature", "greenfield", "refactor",
    "sysadmin", "docs", "continuation", "port",
]
# Note: "simple" was removed as a task type — it was a catch-all that matched
# generic words (change, update, modify) which are better classified by the LLM.

LLM_PROMPT = """Classify this Claude Code task by its primary purpose. Choose exactly ONE from:
- investigation: Research, exploration, understanding, "what is", "how does", reviewing code
- bugfix: Fixing errors, debugging, resolving issues, "doesn't work", troubleshooting
- feature: Adding capability to existing code, enhancing, extending, integrating
- greenfield: Creating something new from scratch, scaffolding, bootstrapping
- refactor: Restructuring existing code without changing behavior, renaming, cleanup
- sysadmin: Git operations, deployment, configuration, running commands, testing
- docs: Documentation, READMEs, comments, changelogs
- continuation: Minimal response (ok, yes, go ahead), session handoff, not a real task
- port: Migrating between technologies, converting formats

If the task doesn't fit any category, choose the closest match. Very short prompts
(<5 words) that aren't actionable should be "continuation".

## Task

**User prompt:** "{user_prompt}"
**Tools used:** {tool_count}
**Files touched:** {files_touched}
**Tool sequence (first 10):** {tool_sequence}

Return a JSON object:
{{"task_type": "<type>", "confidence": "high|medium|low", "reasoning": "<1 sentence>"}}

Return ONLY the JSON object."""


def call_llm(prompt, model="claude-haiku-4-5-20251001"):
    """Call Claude via the SDK."""
    try:
        result = subprocess.run(
            ["claude", "--model", model, "-p", prompt, "--output-format", "json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        text = data.get("result", "")
        # Extract JSON from response
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Just show samples, don't call LLM")
    args = parser.parse_args()

    comparison_dir = Path(args.dir)
    data_dir = comparison_dir / "data"

    # Load classified tasks from both models
    all_tasks = []
    for f in sorted(data_dir.glob("tasks-classified-*.json")):
        tasks = json.load(open(f))
        for t in tasks:
            t["_source_file"] = f.name
        all_tasks.extend(tasks)

    # Split into classified vs unknown
    unknowns = [t for t in all_tasks if t.get("classification", {}).get("type") == "unknown"]
    classified = [t for t in all_tasks if t.get("classification", {}).get("type") not in ("unknown", None)]

    print(f"Total tasks: {len(all_tasks)}")
    print(f"  Classified: {len(classified)} ({len(classified)/len(all_tasks)*100:.1f}%)")
    print(f"  Unknown: {len(unknowns)} ({len(unknowns)/len(all_tasks)*100:.1f}%)")

    # Type distribution for classified
    type_dist = Counter(t["classification"]["type"] for t in classified)
    print(f"\nRegex type distribution (classified only):")
    for k, v in type_dist.most_common():
        print(f"  {k}: {v}")

    # Sample
    random.seed(args.seed)
    n_unknown = min(args.n_sample, len(unknowns))
    n_classified = min(args.n_sample, len(classified))
    sample_unknowns = random.sample(unknowns, n_unknown)
    sample_classified = random.sample(classified, n_classified)

    if args.dry_run:
        print(f"\n=== Would sample {n_unknown} unknowns + {n_classified} classified ===")
        for t in sample_unknowns[:5]:
            print(f"\n  Unknown: \"{t['user_prompt'][:150]}\"")
        for t in sample_classified[:5]:
            rt = t["classification"]["type"]
            print(f"\n  Classified ({rt}): \"{t['user_prompt'][:150]}\"")
        return

    # Evaluate unknowns
    print(f"\n{'='*60}")
    print(f"Evaluating {n_unknown} unknown tasks with LLM...")
    print(f"{'='*60}")

    unknown_results = []
    for i, t in enumerate(sample_unknowns):
        prompt = LLM_PROMPT.format(
            user_prompt=t["user_prompt"][:500],
            tool_count=len(t.get("tool_calls", [])),
            files_touched=t.get("total_files_touched", 0),
            tool_sequence=", ".join(t.get("tool_sequence", "").split(",")[:10]),
        )
        result = call_llm(prompt, args.model)
        if result:
            llm_type = result.get("task_type", "unknown")
            confidence = result.get("confidence", "unknown")
            unknown_results.append({
                "task_id": t["task_id"],
                "regex_type": "unknown",
                "llm_type": llm_type,
                "confidence": confidence,
                "prompt": t["user_prompt"][:200],
            })
            print(f"  [{i+1}/{n_unknown}] unknown → {llm_type} ({confidence})")
        else:
            print(f"  [{i+1}/{n_unknown}] LLM failed")

    # Evaluate classified tasks
    print(f"\n{'='*60}")
    print(f"Evaluating {n_classified} classified tasks with LLM...")
    print(f"{'='*60}")

    classified_results = []
    for i, t in enumerate(sample_classified):
        regex_type = t["classification"]["type"]
        prompt = LLM_PROMPT.format(
            user_prompt=t["user_prompt"][:500],
            tool_count=len(t.get("tool_calls", [])),
            files_touched=t.get("total_files_touched", 0),
            tool_sequence=", ".join(t.get("tool_sequence", "").split(",")[:10]),
        )
        result = call_llm(prompt, args.model)
        if result:
            llm_type = result.get("task_type", "unknown")
            confidence = result.get("confidence", "unknown")
            agree = llm_type == regex_type
            classified_results.append({
                "task_id": t["task_id"],
                "regex_type": regex_type,
                "llm_type": llm_type,
                "confidence": confidence,
                "agree": agree,
                "prompt": t["user_prompt"][:200],
            })
            marker = "OK" if agree else "DISAGREE"
            print(f"  [{i+1}/{n_classified}] {regex_type} → {llm_type} ({confidence}) [{marker}]")
        else:
            print(f"  [{i+1}/{n_classified}] LLM failed")

    # Report
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Unknown resolution
    if unknown_results:
        resolved = Counter(r["llm_type"] for r in unknown_results)
        still_unknown = sum(1 for r in unknown_results if r["llm_type"] == "unknown")
        print(f"\nUnknown task resolution ({len(unknown_results)} sampled):")
        for k, v in resolved.most_common():
            print(f"  → {k}: {v} ({v/len(unknown_results)*100:.1f}%)")
        print(f"  Resolution rate: {(len(unknown_results)-still_unknown)/len(unknown_results)*100:.1f}%")

    # Agreement on classified tasks
    if classified_results:
        agree_count = sum(1 for r in classified_results if r["agree"])
        total = len(classified_results)
        print(f"\nRegex vs LLM agreement ({total} sampled):")
        print(f"  Agreement: {agree_count}/{total} ({agree_count/total*100:.1f}%)")

        # Confusion matrix
        disagree = [r for r in classified_results if not r["agree"]]
        if disagree:
            print(f"\n  Disagreements ({len(disagree)}):")
            confusion = defaultdict(int)
            for r in disagree:
                confusion[(r["regex_type"], r["llm_type"])] += 1
            for (regex, llm), count in sorted(confusion.items(), key=lambda x: -x[1]):
                print(f"    regex={regex} → llm={llm}: {count}")

    # Confidence distribution
    if unknown_results:
        conf = Counter(r["confidence"] for r in unknown_results)
        print(f"\nLLM confidence on unknowns: {dict(conf)}")
    if classified_results:
        conf = Counter(r["confidence"] for r in classified_results)
        print(f"LLM confidence on classified: {dict(conf)}")

    # Save results
    output = {
        "unknown_results": unknown_results,
        "classified_results": classified_results,
        "summary": {
            "unknown_sampled": len(unknown_results),
            "classified_sampled": len(classified_results),
            "agreement_rate": agree_count / total if classified_results else None,
            "unknown_resolution_rate": (len(unknown_results) - still_unknown) / len(unknown_results) if unknown_results else None,
        }
    }
    output_path = comparison_dir / "analysis" / "task-type-eval.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
