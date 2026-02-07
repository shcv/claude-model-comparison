#!/usr/bin/env python3
"""
Analyze replication results against baseline.

Usage:
    python analyze_replication.py results/task-name_*.json
    python analyze_replication.py results/  # analyze all results in directory
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    """Load result JSON files."""
    results = []
    for path in paths:
        if path.is_dir():
            # Load all JSON files in directory
            for json_file in sorted(path.glob("*.json")):
                if json_file.name.endswith("_session.jsonl"):
                    continue  # Skip session files
                try:
                    with open(json_file) as f:
                        results.append(json.load(f))
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)
        else:
            try:
                with open(path) as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
    return results


def compute_efficiency_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Compute derived efficiency metrics."""
    orig = result.get("original_metrics", {})
    rep = result.get("replication_metrics", {})

    metrics = {}

    # Tools per 100 lines added
    orig_lines = orig.get("lines_added", 0)
    rep_tools = rep.get("total_tools", 0)
    orig_tools = orig.get("tools", 0)

    if orig_lines > 0:
        metrics["orig_tools_per_100_lines"] = (orig_tools / orig_lines) * 100
        if rep_tools > 0:
            metrics["rep_tools_per_100_lines"] = (rep_tools / orig_lines) * 100

    # Duration per tool
    orig_duration = orig.get("duration_seconds", 0)
    rep_duration = rep.get("duration_seconds", 0)

    if orig_tools > 0:
        metrics["orig_seconds_per_tool"] = orig_duration / orig_tools
    if rep_tools > 0:
        metrics["rep_seconds_per_tool"] = rep_duration / rep_tools

    # Tool efficiency ratio (lower is better)
    if orig_tools > 0 and rep_tools > 0:
        metrics["tool_ratio"] = rep_tools / orig_tools

    # Duration efficiency ratio
    if orig_duration > 0 and rep_duration > 0:
        metrics["duration_ratio"] = rep_duration / orig_duration

    return metrics


def format_comparison_table(results: list[dict[str, Any]]) -> str:
    """Format a comparison table for multiple results."""
    if not results:
        return "No results to display."

    lines = []
    lines.append("=" * 80)
    lines.append("REPLICATION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    for result in results:
        task = result.get("task", "unknown")
        orig_model = result.get("original_model", "unknown")
        rep_model = result.get("model", "unknown")
        timestamp = result.get("timestamp", "unknown")

        lines.append(f"Task: {task}")
        lines.append(f"Original Model: {orig_model}")
        lines.append(f"Replication Model: {rep_model}")
        lines.append(f"Timestamp: {timestamp}")
        lines.append("")

        # Verification status
        verification = result.get("verification", {})
        success = verification.get("overall_success", False)
        status = "PASSED" if success else "FAILED"
        lines.append(f"Verification: {status}")

        if not success:
            tests_passed = verification.get("tests_passed", False)
            patterns_matched = verification.get("patterns_matched", False)
            if not tests_passed:
                lines.append("  - Some test commands failed")
            if not patterns_matched:
                lines.append("  - Some success patterns not matched")
        lines.append("")

        # Metrics comparison
        orig = result.get("original_metrics", {})
        rep = result.get("replication_metrics", {})

        lines.append("Metrics Comparison:")
        lines.append("-" * 60)
        lines.append(f"{'Metric':<25} {'Original':>12} {'Replication':>12} {'Delta':>10}")
        lines.append("-" * 60)

        # Tools
        orig_tools = orig.get("tools", "N/A")
        rep_tools = rep.get("total_tools", "N/A")
        if isinstance(orig_tools, (int, float)) and isinstance(rep_tools, (int, float)):
            delta = rep_tools - orig_tools
            lines.append(f"{'Total Tools':<25} {orig_tools:>12} {rep_tools:>12} {delta:>+10}")
        else:
            lines.append(f"{'Total Tools':<25} {str(orig_tools):>12} {str(rep_tools):>12} {'N/A':>10}")

        # Duration
        orig_dur = orig.get("duration_seconds", "N/A")
        rep_dur = rep.get("duration_seconds", "N/A")
        if isinstance(orig_dur, (int, float)) and isinstance(rep_dur, (int, float)):
            delta = rep_dur - orig_dur
            lines.append(f"{'Duration (s)':<25} {orig_dur:>12.1f} {rep_dur:>12.1f} {delta:>+10.1f}")
        else:
            lines.append(f"{'Duration (s)':<25} {str(orig_dur):>12} {str(rep_dur):>12} {'N/A':>10}")

        # Files/Lines (original only)
        if "files_touched" in orig:
            lines.append(f"{'Files Touched (orig)':<25} {orig['files_touched']:>12} {'N/A':>12} {'N/A':>10}")
        if "lines_added" in orig:
            lines.append(f"{'Lines Added (orig)':<25} {orig['lines_added']:>12} {'N/A':>12} {'N/A':>10}")
        if "lines_removed" in orig:
            lines.append(f"{'Lines Removed (orig)':<25} {orig['lines_removed']:>12} {'N/A':>12} {'N/A':>10}")

        lines.append("")

        # Efficiency metrics
        efficiency = compute_efficiency_metrics(result)
        if efficiency:
            lines.append("Efficiency Metrics:")
            lines.append("-" * 60)

            if "orig_tools_per_100_lines" in efficiency:
                orig_eff = efficiency["orig_tools_per_100_lines"]
                rep_eff = efficiency.get("rep_tools_per_100_lines", 0)
                lines.append(f"{'Tools per 100 lines':<25} {orig_eff:>12.2f} {rep_eff:>12.2f}")

            if "orig_seconds_per_tool" in efficiency:
                orig_eff = efficiency["orig_seconds_per_tool"]
                rep_eff = efficiency.get("rep_seconds_per_tool", 0)
                lines.append(f"{'Seconds per tool':<25} {orig_eff:>12.2f} {rep_eff:>12.2f}")

            if "tool_ratio" in efficiency:
                ratio = efficiency["tool_ratio"]
                pct = (ratio - 1) * 100
                interpretation = "fewer" if pct < 0 else "more"
                lines.append(f"{'Tool ratio':<25} {ratio:>12.2f}x ({abs(pct):.1f}% {interpretation} tools)")

            if "duration_ratio" in efficiency:
                ratio = efficiency["duration_ratio"]
                pct = (ratio - 1) * 100
                interpretation = "faster" if pct < 0 else "slower"
                lines.append(f"{'Duration ratio':<25} {ratio:>12.2f}x ({abs(pct):.1f}% {interpretation})")

            lines.append("")

        # Tool breakdown
        tool_breakdown = rep.get("tool_breakdown", {})
        if tool_breakdown:
            lines.append("Tool Breakdown (Replication):")
            lines.append("-" * 40)
            for tool, count in sorted(tool_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"  {tool:<30} {count:>6}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("")

    return "\n".join(lines)


def format_summary_table(results: list[dict[str, Any]]) -> str:
    """Format a summary table for all results."""
    if not results:
        return "No results to summarize."

    lines = []
    lines.append("")
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(
        f"{'Task':<30} {'Model':>10} {'Tools':>8} {'Time':>8} "
        f"{'T-Ratio':>8} {'D-Ratio':>8} {'Status':>8}"
    )
    lines.append("-" * 100)

    for result in results:
        task = result.get("task", "unknown")[:28]
        model = result.get("model", "unknown")
        if "opus-4-6" in model.lower():
            model = "Opus 4.6"
        elif "opus" in model.lower():
            model = "Opus 4.5"
        elif "sonnet" in model.lower():
            model = "Sonnet"

        orig = result.get("original_metrics", {})
        rep = result.get("replication_metrics", {})

        orig_tools = orig.get("tools", 0)
        rep_tools = rep.get("total_tools", 0)
        orig_dur = orig.get("duration_seconds", 0)
        rep_dur = rep.get("duration_seconds", 0)

        t_ratio = rep_tools / orig_tools if orig_tools > 0 else 0
        d_ratio = rep_dur / orig_dur if orig_dur > 0 else 0

        success = result.get("verification", {}).get("overall_success", False)
        status = "PASS" if success else "FAIL"

        lines.append(
            f"{task:<30} {model:>10} {rep_tools:>8} {rep_dur:>7.0f}s "
            f"{t_ratio:>7.2f}x {d_ratio:>7.2f}x {status:>8}"
        )

    lines.append("=" * 100)

    # Aggregate stats
    passed = sum(1 for r in results if r.get("verification", {}).get("overall_success", False))
    total = len(results)
    lines.append(f"Total: {passed}/{total} passed")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze replication results")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Result JSON files or directories to analyze",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only summary table",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    results = load_results(args.paths)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {
            "results": results,
            "efficiency_metrics": [compute_efficiency_metrics(r) for r in results],
        }
        print(json.dumps(output, indent=2))
    elif args.summary:
        print(format_summary_table(results))
    else:
        print(format_comparison_table(results))
        print(format_summary_table(results))


if __name__ == "__main__":
    main()
