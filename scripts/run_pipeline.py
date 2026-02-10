#!/usr/bin/env python3
"""Pipeline orchestrator for model comparison analysis.

Runs all analysis steps in sequence. Steps can be selected individually
or run from a given starting point.

Usage:
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --steps collect,extract
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --from classify
"""

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    ("collect",  "Collect sessions"),
    ("extract",  "Extract tasks"),
    ("classify", "Classify tasks"),
    ("analyze",  "Analyze behavior"),
    ("tokens",   "Extract tokens"),
    ("stats",    "Run statistical tests"),
    ("edits",    "Analyze edits"),
    ("planning", "Analyze planning by complexity"),
    ("compaction", "Analyze compaction"),
    ("report",     "Build report"),
]

STEP_NAMES = [s[0] for s in STEPS]


def build_command(step, data_dir, analysis_dir):
    """Return the command list for a given step."""
    scripts_dir = Path(__file__).resolve().parent
    if step == "collect":
        return [sys.executable, str(scripts_dir / "collect_sessions.py"),
                "--data-dir", str(data_dir)]
    elif step == "extract":
        return [sys.executable, str(scripts_dir / "extract_tasks.py"),
                "--data-dir", str(data_dir)]
    elif step == "classify":
        return [sys.executable, str(scripts_dir / "classify_tasks.py"),
                "--data-dir", str(data_dir)]
    elif step == "analyze":
        return [sys.executable, str(scripts_dir / "analyze_behavior.py"),
                "--data-dir", str(data_dir)]
    elif step == "tokens":
        # extract_tokens.py uses --dir pointing to the comparison root
        comparison_dir = data_dir.parent
        return [sys.executable, str(scripts_dir / "extract_tokens.py"),
                "--dir", str(comparison_dir)]
    elif step == "stats":
        return [sys.executable, str(scripts_dir / "stat_tests.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "edits":
        return [sys.executable, str(scripts_dir / "analyze_edits.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "planning":
        return [sys.executable, str(scripts_dir / "planning_analysis.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "compaction":
        return [sys.executable, str(scripts_dir / "analyze_compaction.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "report":
        comparison_dir = data_dir.parent
        return [sys.executable, str(scripts_dir / "build_report.py"),
                "--dir", str(comparison_dir)]
    else:
        raise ValueError(f"Unknown step: {step}")


def run_step(name, label, data_dir, analysis_dir):
    """Run a single pipeline step. Returns True on success."""
    cmd = build_command(name, data_dir, analysis_dir)
    sys.stdout.flush()
    print(f"\n{'='*60}")
    print(f"  [{name}] {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"  [{name}] SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [{name}] FAILED (exit code {e.returncode})", file=sys.stderr)
        return False
    except FileNotFoundError as e:
        print(f"  [{name}] FAILED ({e})", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the model comparison analysis pipeline")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to the data directory")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Path to analysis output directory (default: data-dir/../analysis)")
    parser.add_argument("--steps", type=str, default=None,
                        help=f"Comma-separated list of steps to run: {','.join(STEP_NAMES)}")
    parser.add_argument("--from", dest="from_step", type=str, default=None,
                        help=f"Start from this step: {','.join(STEP_NAMES)}")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    analysis_dir = (args.analysis_dir or data_dir.parent / "analysis").resolve()

    # Determine which steps to run
    if args.steps and args.from_step:
        print("Error: --steps and --from are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if args.steps:
        selected = [s.strip() for s in args.steps.split(",")]
        for s in selected:
            if s not in STEP_NAMES:
                print(f"Error: unknown step '{s}'. Valid steps: {', '.join(STEP_NAMES)}",
                      file=sys.stderr)
                sys.exit(1)
        steps_to_run = [(name, label) for name, label in STEPS if name in selected]
    elif args.from_step:
        if args.from_step not in STEP_NAMES:
            print(f"Error: unknown step '{args.from_step}'. Valid steps: {', '.join(STEP_NAMES)}",
                  file=sys.stderr)
            sys.exit(1)
        start_idx = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = list(STEPS)

    print(f"Data dir:     {data_dir}")
    print(f"Analysis dir: {analysis_dir}")
    print(f"Steps:        {', '.join(s[0] for s in steps_to_run)}")

    succeeded = 0
    failed = 0
    for name, label in steps_to_run:
        if run_step(name, label, data_dir, analysis_dir):
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {succeeded} succeeded, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
