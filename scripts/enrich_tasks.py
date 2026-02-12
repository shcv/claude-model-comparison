#!/usr/bin/env python3
"""Enrich annotated tasks by joining token, edit-metric, and classification data.

Produces a single enriched file per model with all per-task fields needed for
statistical testing and report generation.

Usage:
    python scripts/enrich_tasks.py --data-dir data --analysis-dir analysis
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_model_pair


def load_json(path):
    """Load JSON file, return empty list if missing."""
    if not Path(path).exists():
        print(f"  Warning: {path} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


def index_by_task_id(records):
    """Index a list of dicts by task_id."""
    return {r["task_id"]: r for r in records if "task_id" in r}


# Fields to copy from token records
TOKEN_FIELDS = [
    "total_input_tokens", "total_output_tokens",
    "cache_read_tokens", "cache_write_tokens",
    "thinking_chars", "thinking_blocks",
    "text_chars", "text_blocks",
    "tool_use_chars", "edit_content_chars", "write_content_chars",
    "bash_command_chars",
    "request_count", "estimated_cost",
]

# Fields to copy from edit-metric records (default to 0 for tasks without edits)
EDIT_FIELDS = [
    "edit_count", "write_count", "failed_edit_count",
    "overlap_count", "self_corrections", "error_recoveries",
    "user_corrections", "iterative_refinements", "max_chain_depth",
    "triage_score",
]


def enrich_model(data_dir, analysis_dir, model):
    """Enrich annotated tasks for a single model.

    Joins from:
    - analysis/tasks-annotated-{model}.json (base)
    - data/tokens-{model}.json
    - analysis/edit-metrics-{model}.json
    - data/tasks-classified-{model}.json
    """
    data_dir = Path(data_dir)
    analysis_dir = Path(analysis_dir)

    # Load base annotated tasks
    annotated_path = analysis_dir / f"tasks-annotated-{model}.json"
    annotated = load_json(annotated_path)
    if not annotated:
        print(f"  No annotated tasks for {model}, skipping")
        return []

    # Load and index join sources
    tokens = load_json(data_dir / f"tokens-{model}.json")
    tokens_by_id = index_by_task_id(tokens)

    edits = load_json(analysis_dir / f"edit-metrics-{model}.json")
    edits_by_id = index_by_task_id(edits)

    classified = load_json(data_dir / f"tasks-classified-{model}.json")
    classified_by_id = index_by_task_id(classified)

    print(f"  {model}: {len(annotated)} annotated, "
          f"{len(tokens_by_id)} tokens, "
          f"{len(edits_by_id)} edit-metrics, "
          f"{len(classified_by_id)} classified")

    # Join
    join_stats = {"token_hits": 0, "edit_hits": 0, "class_hits": 0}
    enriched = []

    for task in annotated:
        rec = dict(task)  # shallow copy
        tid = rec.get("task_id", "")

        # Merge token fields (None if missing)
        tok = tokens_by_id.get(tid)
        if tok:
            join_stats["token_hits"] += 1
            for field in TOKEN_FIELDS:
                rec.setdefault(field, tok.get(field))
        else:
            for field in TOKEN_FIELDS:
                rec.setdefault(field, None)

        # Merge edit fields (0 if missing — no edits means zero counts)
        edit = edits_by_id.get(tid)
        if edit:
            join_stats["edit_hits"] += 1
            for field in EDIT_FIELDS:
                rec.setdefault(field, edit.get(field, 0))
        else:
            for field in EDIT_FIELDS:
                rec.setdefault(field, 0)

        # Merge task_type from classification
        cl = classified_by_id.get(tid)
        if cl:
            join_stats["class_hits"] += 1
            classification = cl.get("classification", {})
            rec.setdefault("task_type", classification.get("type"))
        else:
            rec.setdefault("task_type", None)

        # Compute derived fields
        duration = rec.get("duration_seconds") or 0
        cost = rec.get("estimated_cost")
        req_count = rec.get("request_count")
        total_output = rec.get("total_output_tokens")
        total_input = rec.get("total_input_tokens") or 0
        cache_read = rec.get("cache_read_tokens") or 0
        thinking = rec.get("thinking_chars") or 0
        text = rec.get("text_chars") or 0
        edit_count = rec.get("edit_count") or 0
        write_count = rec.get("write_count") or 0
        overlap_count = rec.get("overlap_count") or 0

        rec["cost_per_minute"] = (
            cost / (duration / 60) if cost and duration > 0 else None
        )
        rec["output_per_request"] = (
            total_output / req_count if total_output and req_count and req_count > 0 else None
        )
        rec["cache_hit_rate"] = (
            cache_read / (cache_read + total_input)
            if (cache_read + total_input) > 0 else None
        )
        rec["thinking_fraction"] = (
            thinking / (thinking + text)
            if (thinking + text) > 0 else None
        )
        rec["has_edits"] = edit_count > 0
        rec["has_overlaps"] = overlap_count > 0
        rec["rewrite_rate"] = (
            overlap_count / max(edit_count + write_count, 1)
        )

        enriched.append(rec)

    print(f"  Joins: tokens={join_stats['token_hits']}/{len(annotated)}, "
          f"edits={join_stats['edit_hits']}/{len(annotated)}, "
          f"classified={join_stats['class_hits']}/{len(annotated)}")

    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Enrich annotated tasks with token, edit, and classification data")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to data directory")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Path to analysis directory (default: data-dir/../analysis)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    analysis_dir = (args.analysis_dir or data_dir.parent / "analysis").resolve()

    models = list(discover_model_pair(data_dir))
    print(f"Enriching tasks for models: {models}")

    for model in models:
        enriched = enrich_model(data_dir, analysis_dir, model)
        if not enriched:
            continue

        output_path = analysis_dir / f"tasks-enriched-{model}.json"
        with open(output_path, "w") as f:
            json.dump(enriched, f, indent=2)
            f.write("\n")
        print(f"  Wrote {len(enriched)} enriched tasks to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
