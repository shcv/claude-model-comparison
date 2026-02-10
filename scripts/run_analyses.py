#!/usr/bin/env python3
"""Run all analysis scripts in sequence.

Coordinator script that runs analyze_behavior, analyze_edits,
planning_analysis, and analyze_compaction. Replaces individual
pipeline steps with a single "analyze" step.

Usage:
    python scripts/run_analyses.py \
      --data-dir comparisons/opus-4.5-vs-4.6/data \
      --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
"""

import argparse
import subprocess
import sys
from pathlib import Path


ANALYSES = [
    ("analyze_behavior.py", "Behavioral analysis",
     lambda data_dir, analysis_dir: [
         sys.executable, str(Path(__file__).resolve().parent / "analyze_behavior.py"),
         "--data-dir", str(data_dir),
         "--output", str(analysis_dir / "behavior-metrics.json"),
     ]),
    ("analyze_edits.py", "Edit overlap analysis",
     lambda data_dir, analysis_dir: [
         sys.executable, str(Path(__file__).resolve().parent / "analyze_edits.py"),
         "--data-dir", str(data_dir),
         "--analysis-dir", str(analysis_dir),
     ]),
    ("planning_analysis.py", "Planning by complexity",
     lambda data_dir, analysis_dir: [
         sys.executable, str(Path(__file__).resolve().parent / "planning_analysis.py"),
         "--data-dir", str(data_dir),
         "--analysis-dir", str(analysis_dir),
     ]),
    ("analyze_compaction.py", "Compaction analysis",
     lambda data_dir, analysis_dir: [
         sys.executable, str(Path(__file__).resolve().parent / "analyze_compaction.py"),
         "--data-dir", str(data_dir),
         "--analysis-dir", str(analysis_dir),
     ]),
    ("analyze_timing.py", "Timing analysis",
     lambda data_dir, analysis_dir: [
         sys.executable, str(Path(__file__).resolve().parent / "analyze_timing.py"),
         "--data-dir", str(data_dir),
         "--analysis-dir", str(analysis_dir),
     ]),
]


def main():
    parser = argparse.ArgumentParser(description='Run all analysis scripts')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Data directory with tasks-canonical-*.json files')
    parser.add_argument('--analysis-dir', type=Path, default=None,
                        help='Output directory for analysis results (default: data/../analysis)')
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    analysis_dir = (args.analysis_dir or data_dir.parent / 'analysis').resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0

    for script_name, label, cmd_builder in ANALYSES:
        cmd = cmd_builder(data_dir, analysis_dir)
        print(f"\n{'='*60}")
        print(f"  [{script_name}] {label}")
        print(f"  {' '.join(cmd)}")
        print(f"{'='*60}", flush=True)

        try:
            result = subprocess.run(cmd, check=True)
            print(f"  [{script_name}] SUCCESS")
            succeeded += 1
        except subprocess.CalledProcessError as e:
            print(f"  [{script_name}] FAILED (exit code {e.returncode})", file=sys.stderr)
            failed += 1
        except FileNotFoundError as e:
            print(f"  [{script_name}] FAILED ({e})", file=sys.stderr)
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Analyses complete: {succeeded} succeeded, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
