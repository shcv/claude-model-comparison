#!/usr/bin/env python3
"""Per-section quality review for model comparison reports.

Runs parallel per-section reviews and an optional holistic cross-section
review using LLM calls. Produces a review-summary.json with corrections
categorized by type and severity.

Usage:
    python scripts/review_report.py --dir comparisons/opus-4.5-vs-4.6
    python scripts/review_report.py --dir comparisons/opus-4.5-vs-4.6 --apply
    python scripts/review_report.py --dir comparisons/opus-4.5-vs-4.6 --sections cost,thinking
    python scripts/review_report.py --dir comparisons/opus-4.5-vs-4.6 --skip-holistic
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import load_comparison_config
import subprocess as _subprocess
from update_sections import (
    build_annotated_template,
    call_claude_sdk as _call_claude_sdk,
    strip_code_fences,
    content_hash,
    decompose_document,
    apply_corrections,
    parse_corrections as _parse_corrections,
    load_spec,
    load_data_sources,
    resolve_key_metrics,
)

# Module-level verbose flag, set by main()
_verbose = False


def call_claude_sdk(prompt: str, model: str = "sonnet") -> str | None:
    """Wrapper around update_sections.call_claude_sdk with verbose error output."""
    if not _verbose:
        return _call_claude_sdk(prompt, model)

    # Verbose mode: run directly so we can capture full output
    try:
        result = _subprocess.run(
            ["claude", "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  Claude SDK exit code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"  stderr: {result.stderr.rstrip()}", file=sys.stderr)
            if result.stdout:
                print(f"  stdout ({len(result.stdout)} chars): {result.stdout[:500]}", file=sys.stderr)
            return None
        raw = strip_code_fences(result.stdout.strip())
        if _verbose:
            print(f"  LLM response ({len(raw)} chars): {raw[:300]}{'...' if len(raw) > 300 else ''}", file=sys.stderr)
        return raw
    except FileNotFoundError:
        print("  Error: 'claude' CLI not found", file=sys.stderr)
        return None
    except _subprocess.TimeoutExpired:
        print("  Error: Claude SDK timed out after 600s", file=sys.stderr)
        return None


def parse_corrections(raw: str) -> list[dict] | None:
    """Extract corrections from <corrections>...</corrections> tags.

    Falls back to bare JSON parsing if tags aren't present.
    """
    # Try extracting from <corrections> tags first
    m = re.search(r'<corrections>(.*?)</corrections>', raw, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        result = _parse_corrections(inner)
        if result is not None:
            return result

    # Fall back to raw parsing
    return _parse_corrections(raw)


SECTION_REVIEW_PROMPT = """\
You are reviewing a section of a model comparison report for quality.

## Section intent
{section_intent}

## What to focus on
{section_focus}

## General checks
1. **Prose quality**: Awkward phrasing, unclear antecedents, passive voice overuse,
   inconsistent tone, redundant statements.
2. **Factual accuracy**: Numbers in prose that don't match the data provided below.
   GENERATED-TABLE blocks are data-driven and correct — only check prose text.
3. **Internal consistency**: Claims that contradict each other within this section,
   or conclusions that don't follow from the data shown.

## Section content
---
{section_content}
---

## Key metrics for this section
{metrics_json}

Wrap your response in <corrections>...</corrections> tags containing a JSON array.
Each correction:
{{
  "old": "exact text to replace (must match exactly)",
  "new": "replacement text",
  "reason": "explanation",
  "category": "prose|factual|consistency",
  "severity": "low|medium|high"
}}

If no corrections are needed, return: <corrections>[]</corrections>
"""

HOLISTIC_REVIEW_PROMPT = """\
You are reviewing a complete model comparison report for cross-section quality.
The report compares {display_a} vs {display_b}. Check for:

1. **Cross-section consistency**: Numbers or claims in one section that contradict
   another section. For example, a cost figure cited differently in the summary
   vs the cost section.
2. **Narrative flow**: Abrupt transitions, sections that assume context not yet
   introduced, or conclusions that appear before their supporting evidence.
3. **Terminology consistency**: The same concept described with different terms
   in different sections (e.g., "task completion" vs "completion rate").
4. **Redundancy**: The same point made in multiple sections without adding value.

Do NOT flag issues within GENERATED-TABLE blocks — those are data-driven and correct.

Full report:
---
{document}
---

Wrap your response in <corrections>...</corrections> tags containing a JSON array.
Each correction:
{{
  "old": "exact text to replace (must match exactly)",
  "new": "replacement text",
  "reason": "explanation",
  "category": "cross-consistency|flow|terminology|redundancy",
  "severity": "low|medium|high",
  "sections": ["section-id-1", "section-id-2"]
}}

If no corrections are needed, return: <corrections>[]</corrections>
"""


def extract_sections(annotated_html: str) -> dict[str, str]:
    """Extract <section id="...">...</section> blocks from annotated HTML."""
    sections = {}
    for m in re.finditer(
        r'<section\s+id="([^"]+)">(.*?)</section>',
        annotated_html,
        re.DOTALL,
    ):
        sections[m.group(1)] = m.group(0)
    return sections


def find_section_specs(section_content: str, specs_dir: Path) -> list[Path]:
    """Find spec files relevant to a section by checking expansion markers."""
    # Extract expansion names from BEGIN-EXPANSION markers in the section
    expansion_names = re.findall(
        r'<!-- BEGIN-EXPANSION: (\S+) -->', section_content
    )

    # Also check for expand markers (pre-annotation)
    expansion_names.extend(re.findall(
        r'<!-- expand: (\S+) -->', section_content
    ))

    # Map expansion names to spec files by checking which specs define tables
    # matching these expansions
    relevant_specs = set()
    for spec_path in specs_dir.glob("*.json"):
        try:
            with open(spec_path) as f:
                spec = json.load(f)
            # Check if any table in this spec matches an expansion in the section
            for table_def in spec.get("tables", {}).values():
                exp_name = table_def.get("expansion")
                if exp_name and exp_name in expansion_names:
                    relevant_specs.add(spec_path)
                    break
            # Also match by spec name vs section id
            if spec_path.stem in expansion_names or any(
                spec_path.stem in en for en in expansion_names
            ):
                relevant_specs.add(spec_path)
        except (json.JSONDecodeError, KeyError):
            continue

    return list(relevant_specs)


def load_review_cache(cache_dir: Path) -> dict:
    """Load the review cache."""
    cache_file = cache_dir / "cache.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"sections": {}, "holistic": {}}
    return {"sections": {}, "holistic": {}}


def save_review_cache(cache_dir: Path, cache: dict):
    """Save the review cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "cache.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
        f.write("\n")


def _load_section_review_guidance(section_id: str, spec_files: list[Path],
                                   specs_dir: Path) -> tuple[str, str]:
    """Load review intent and focus for a section from its spec files.

    Tries spec files matched by content first, then falls back to a spec
    whose section_id matches. Returns (intent, focus_text).
    """
    # Check matched spec files for review guidance
    for spec_path in spec_files:
        try:
            with open(spec_path) as f:
                spec = json.load(f)
            review = spec.get("review", {})
            if review.get("intent"):
                intent = review["intent"]
                focus_items = review.get("focus", [])
                focus_text = "\n".join(f"- {f}" for f in focus_items) if focus_items else "(none)"
                return intent, focus_text
        except (json.JSONDecodeError, OSError):
            continue

    # Fall back to spec matching section_id by name
    direct_spec = specs_dir / f"{section_id}.json"
    if direct_spec.exists() and direct_spec not in spec_files:
        try:
            with open(direct_spec) as f:
                spec = json.load(f)
            review = spec.get("review", {})
            if review.get("intent"):
                intent = review["intent"]
                focus_items = review.get("focus", [])
                focus_text = "\n".join(f"- {f}" for f in focus_items) if focus_items else "(none)"
                return intent, focus_text
        except (json.JSONDecodeError, OSError):
            pass

    return "(No specific intent defined for this section.)", "(General quality review)"


def review_section(section_id: str, section_content: str,
                   specs_dir: Path, comparison_dir: Path,
                   config: dict, model: str,
                   cache: dict) -> tuple[str, list[dict], bool]:
    """Review a single section. Returns (section_id, corrections, from_cache)."""
    # Collect metrics from relevant specs
    spec_files = find_section_specs(section_content, specs_dir)

    # Also try direct spec match by section_id
    direct_spec = specs_dir / f"{section_id}.json"
    all_spec_files = list(spec_files)
    if direct_spec.exists() and direct_spec not in all_spec_files:
        all_spec_files.append(direct_spec)

    all_metrics = {}
    for spec_path in all_spec_files:
        spec = load_spec(spec_path)
        data = load_data_sources(spec, comparison_dir)
        key_metrics = spec.get("prose", {}).get("key_metrics", [])
        metrics = resolve_key_metrics(key_metrics, data, config)
        all_metrics.update(metrics)

    # Load section-specific review guidance
    section_intent, section_focus = _load_section_review_guidance(
        section_id, all_spec_files, specs_dir
    )

    # Compute cache key (includes guidance so cache invalidates on spec changes)
    metrics_json = json.dumps(all_metrics, indent=2, default=str)
    cache_key = content_hash(section_content + metrics_json + section_intent)

    # Check cache
    section_cache = cache.get("sections", {}).get(section_id, {})
    if cache_key in section_cache:
        return section_id, section_cache[cache_key].get("corrections", []), True

    # Call LLM
    prompt = SECTION_REVIEW_PROMPT.format(
        section_intent=section_intent,
        section_focus=section_focus,
        section_content=section_content,
        metrics_json=metrics_json if all_metrics else "(No quantitative metrics for this section — review prose quality and consistency only.)",
    )
    raw = call_claude_sdk(prompt, model=model)
    if raw is None:
        print(f"  [{section_id}] LLM call failed", file=sys.stderr)
        return section_id, [], False

    corrections = parse_corrections(raw)
    if corrections is None:
        print(f"  [{section_id}] Failed to parse corrections", file=sys.stderr)
        return section_id, [], False

    # Add section_id to each correction
    for c in corrections:
        c["section"] = section_id

    # Update cache
    cache.setdefault("sections", {}).setdefault(section_id, {})[cache_key] = {
        "corrections": corrections
    }

    return section_id, corrections, False


def review_holistic(annotated_html: str, config: dict, model: str,
                    cache: dict) -> list[dict]:
    """Run holistic cross-section review. Returns corrections list."""
    cache_key = content_hash(annotated_html)

    holistic_cache = cache.get("holistic", {})
    if cache_key in holistic_cache:
        print("  [holistic] Using cached results")
        return holistic_cache[cache_key].get("corrections", [])

    prompt = HOLISTIC_REVIEW_PROMPT.format(
        document=annotated_html,
        display_a=config["display_a"],
        display_b=config["display_b"],
    )
    raw = call_claude_sdk(prompt, model=model)
    if raw is None:
        print("  [holistic] LLM call failed", file=sys.stderr)
        return []

    corrections = parse_corrections(raw)
    if corrections is None:
        print("  [holistic] Failed to parse corrections", file=sys.stderr)
        return []

    # Mark as holistic
    for c in corrections:
        c.setdefault("category", "cross-consistency")

    cache.setdefault("holistic", {})[cache_key] = {
        "corrections": corrections
    }

    return corrections


def print_corrections(corrections: list[dict]):
    """Print corrections in a readable format."""
    if not corrections:
        print("  No corrections found.")
        return

    for i, c in enumerate(corrections):
        severity = c.get("severity", "?")
        category = c.get("category", "?")
        section = c.get("section", c.get("sections", "?"))
        reason = c.get("reason", "")
        old = c.get("old", "")[:80]
        print(f"  [{i+1}] {severity:6s} {category:18s} {section}")
        print(f"       {reason}")
        print(f"       old: {old}...")
        print()


def write_summary(report_dir: Path, all_corrections: list[dict]):
    """Write review-summary.json with totals by category and severity."""
    by_category = {}
    by_severity = {}
    for c in all_corrections:
        cat = c.get("category", "unknown")
        sev = c.get("severity", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
        by_severity[sev] = by_severity.get(sev, 0) + 1

    summary = {
        "total_corrections": len(all_corrections),
        "by_category": by_category,
        "by_severity": by_severity,
        "corrections": all_corrections,
    }

    summary_path = report_dir / "review-summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\n  Wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-section quality review for model comparison reports")
    parser.add_argument("--dir", type=Path, required=True,
                        help="Comparison directory")
    parser.add_argument("--model", type=str, default="opus",
                        help="Claude model for review (default: opus)")
    parser.add_argument("--sections", type=str, default=None,
                        help="Comma-separated section IDs to review")
    parser.add_argument("--apply", action="store_true",
                        help="Apply corrections to template/expansions (default: show only)")
    parser.add_argument("--skip-holistic", action="store_true",
                        help="Skip cross-section holistic review")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers for per-section review")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full LLM request/response details")
    args = parser.parse_args()

    global _verbose
    _verbose = args.verbose

    comparison_dir = args.dir.resolve()
    report_dir = comparison_dir / "report"
    expansions_dir = report_dir / "expansions"
    specs_dir = report_dir / "specs"
    cache_dir = report_dir / ".review-cache"

    config = load_comparison_config(comparison_dir)

    # Load template and build annotated version
    template_path = report_dir / "report.html"
    if not template_path.exists():
        print(f"Error: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    template_html = template_path.read_text()
    annotated_html, expansion_names = build_annotated_template(
        template_html, expansions_dir
    )

    # Extract sections
    sections = extract_sections(annotated_html)
    if not sections:
        print("Error: no <section id=...> blocks found in template",
              file=sys.stderr)
        sys.exit(1)

    # Filter to requested sections
    if args.sections:
        requested = [s.strip() for s in args.sections.split(",")]
        missing = [s for s in requested if s not in sections]
        if missing:
            print(f"Warning: sections not found: {', '.join(missing)}",
                  file=sys.stderr)
        sections = {k: v for k, v in sections.items() if k in requested}

    print(f"Reviewing {len(sections)} sections: {', '.join(sections.keys())}")
    print(f"Model: {args.model}")

    cache = load_review_cache(cache_dir)
    all_corrections = []

    # Phase 1: Per-section parallel review
    print(f"\n--- Phase 1: Per-section review ---")
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for section_id, section_content in sections.items():
            future = pool.submit(
                review_section, section_id, section_content,
                specs_dir, comparison_dir, config, args.model, cache
            )
            futures[future] = section_id

        for future in as_completed(futures):
            section_id = futures[future]
            try:
                sid, corrections, from_cache = future.result()
                status = "cached" if from_cache else f"{len(corrections)} corrections"
                print(f"  [{sid}] {status}")
                all_corrections.extend(corrections)
            except Exception as e:
                print(f"  [{section_id}] Error: {e}", file=sys.stderr)

    save_review_cache(cache_dir, cache)

    # Phase 2: Holistic cross-section review
    if not args.skip_holistic and len(sections) > 1:
        print(f"\n--- Phase 2: Holistic cross-section review ---")
        holistic_corrections = review_holistic(
            annotated_html, config, args.model, cache
        )
        print(f"  {len(holistic_corrections)} cross-section corrections")
        all_corrections.extend(holistic_corrections)
        save_review_cache(cache_dir, cache)

    # Output
    print(f"\n--- Results: {len(all_corrections)} total corrections ---")

    # Default: show corrections. --apply: apply them instead.
    if not args.apply:
        print_corrections(all_corrections)

    if args.apply and all_corrections:
        print(f"\n--- Applying {len(all_corrections)} corrections ---")
        doc, n_applied, direction_changes = apply_corrections(
            annotated_html, all_corrections
        )
        print(f"  Applied {n_applied}/{len(all_corrections)} corrections")

        if direction_changes:
            print(f"  DIRECTION CHANGES: {len(direction_changes)}")
            for dc in direction_changes:
                print(f"    - {dc}")

        # Decompose back to template + expansions
        new_template, new_expansions = decompose_document(doc, expansion_names)
        template_path.write_text(new_template)
        print(f"  Updated {template_path}")
        for name, content in new_expansions.items():
            exp_path = expansions_dir / f"{name}.html"
            exp_path.write_text(content)
        print(f"  Updated {len(new_expansions)} expansion files")

    # Always write summary
    write_summary(report_dir, all_corrections)

    # Print category/severity breakdown
    by_cat = {}
    by_sev = {}
    for c in all_corrections:
        cat = c.get("category", "unknown")
        sev = c.get("severity", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_sev[sev] = by_sev.get(sev, 0) + 1

    if by_cat:
        print(f"\n  By category: {', '.join(f'{k}={v}' for k, v in sorted(by_cat.items()))}")
    if by_sev:
        print(f"  By severity: {', '.join(f'{k}={v}' for k, v in sorted(by_sev.items()))}")


if __name__ == "__main__":
    main()
