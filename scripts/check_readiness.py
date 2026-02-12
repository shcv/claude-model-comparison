#!/usr/bin/env python3
"""Check readiness of deferred report sections for promotion.

Standalone diagnostic — not a pipeline step. Reads analysis JSONs and
reports sample sizes for each area currently in Future Investigations,
flagging which areas have enough data to promote to full sections.
"""

import argparse
import json
import sys
from pathlib import Path

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from models import discover_model_pair


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_edit_accuracy(analysis_dir, model_b):
    """Check edit accuracy readiness: editing tasks and overlapping edits."""
    path = analysis_dir / "edit-analysis.json"
    if not path.exists():
        return [("4.6 editing tasks", 0, 50, False),
                ("4.6 overlapping edits", 0, 30, False)]

    data = load_json(path)
    b = data.get(model_b, {})
    editing_tasks = b.get("total_tasks", 0)
    overlaps = b.get("total_overlaps", 0)

    return [
        ("4.6 editing tasks", editing_tasks, 50, editing_tasks >= 50),
        ("4.6 overlapping edits", overlaps, 30, overlaps >= 30),
    ]


def check_quality(analysis_dir):
    """Check quality readiness: any metric surviving Bonferroni."""
    path = analysis_dir / "stat-tests.json"
    if not path.exists():
        return [("Any quality metric surviving Bonferroni", "no data", "p<0.00043", False)]

    data = load_json(path)
    bonferroni = 0.05 / 115  # 115 tests

    # Check quality-related fields
    quality_fields = {"alignment_score", "satisfied_rate", "dissatisfied_rate",
                      "neutral_rate", "complete_rate", "partial_rate",
                      "failed_rate", "interrupted_rate"}

    best_p = 1.0
    best_field = "none"
    for test_type in ["mann_whitney", "chi_square", "proportions"]:
        for item in data.get("overall", {}).get(test_type, []):
            if isinstance(item, dict) and item.get("field") in quality_fields:
                p = item.get("p_value", 1)
                if p < best_p:
                    best_p = p
                    best_field = item["field"]

    survived = best_p < bonferroni
    current = f"p={best_p:.6f} ({best_field})" if best_p < 1 else "no quality tests"
    return [("Any quality metric surviving Bonferroni", current, f"p<{bonferroni:.5f}", survived)]


def check_planning(analysis_dir, model_b):
    """Check planning readiness: 4.6 planned tasks."""
    path = analysis_dir / "planning-analysis.json"
    if not path.exists():
        return [("4.6 planned tasks", 0, 30, False)]

    data = load_json(path)
    planned = data.get(model_b, {}).get("total_planned", 0)
    return [("4.6 planned tasks", planned, 30, planned >= 30)]


def check_compaction(analysis_dir, model_a, model_b):
    """Check compaction readiness: total events across both models."""
    path = analysis_dir / "compaction-analysis.json"
    if not path.exists():
        return [("Total compaction events", 0, 10, False)]

    data = load_json(path)
    events_a = data.get(model_a, {}).get("sessions_with_compaction", 0)
    events_b = data.get(model_b, {}).get("sessions_with_compaction", 0)
    total = events_a + events_b
    return [("Total compaction events", total, 10, total >= 10)]


def check_session_dynamics(analysis_dir):
    """Check session dynamics: any session test surviving Bonferroni."""
    path = analysis_dir / "stat-tests.json"
    if not path.exists():
        return [("Any session test surviving Bonferroni", "no data", "true", False)]

    data = load_json(path)
    bonferroni = 0.05 / 115

    session_fields = {"duration_seconds", "session_length", "warmup_effect",
                      "research_ratio", "implementation_ratio", "front_load"}

    best_p = 1.0
    best_field = "none"
    for test_type in ["mann_whitney", "chi_square", "proportions"]:
        for item in data.get("overall", {}).get(test_type, []):
            if isinstance(item, dict) and item.get("field") in session_fields:
                p = item.get("p_value", 1)
                if p < best_p:
                    best_p = p
                    best_field = item["field"]

    survived = best_p < bonferroni
    current = f"p={best_p:.6f} ({best_field})" if best_p < 1 else "no session tests"
    return [("Any session test surviving Bonferroni", current, f"p<{bonferroni:.5f}", survived)]


def main():
    parser = argparse.ArgumentParser(description="Check readiness of deferred report sections")
    parser.add_argument("--dir", required=True, help="Comparison directory")
    args = parser.parse_args()

    comp_dir = Path(args.dir)
    analysis_dir = comp_dir / "analysis"
    data_dir = comp_dir / "data"

    model_a, model_b = discover_model_pair(data_dir)

    print("=" * 70)
    print("  FUTURE INVESTIGATIONS — READINESS CHECK")
    print("=" * 70)
    print()

    areas = [
        ("Edit Accuracy", check_edit_accuracy(analysis_dir, model_b)),
        ("Quality & Satisfaction", check_quality(analysis_dir)),
        ("Planning", check_planning(analysis_dir, model_b)),
        ("Compaction", check_compaction(analysis_dir, model_a, model_b)),
        ("Session Dynamics", check_session_dynamics(analysis_dir)),
    ]

    any_ready = False
    for area_name, checks in areas:
        print(f"  {area_name}")
        for metric, current, threshold, passed in checks:
            status = "\033[32mREADY\033[0m" if passed else "\033[33mNOT READY\033[0m"
            print(f"    {status}  {metric}: {current} (need: {threshold})")
            if passed:
                any_ready = True
        print()

    print("=" * 70)
    if any_ready:
        print("  Some areas are ready for promotion to full sections.")
    else:
        print("  No areas ready for promotion yet. Continue collecting data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
