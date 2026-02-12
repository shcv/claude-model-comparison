#!/usr/bin/env python3
"""Pipeline orchestrator for model comparison analysis.

Runs all analysis steps in sequence with staleness detection and manifest
tracking. Steps can be selected individually or run from a given starting point.

Usage:
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --steps collect,extract
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --from classify
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --check-stale
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --check-consistency
    python scripts/run_pipeline.py --data-dir comparisons/opus-4.5-vs-4.6/data --force
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STEPS = [
    ("collect",   "Collect sessions"),
    ("extract",   "Extract tasks"),
    ("classify",  "Classify tasks"),
    ("annotate",  "Annotate tasks (LLM + signals)"),
    ("analyze",   "Run all analyses (behavior, edits, planning, compaction, timing)"),
    ("tokens",    "Extract tokens"),
    ("enrich",    "Enrich tasks (join sources)"),
    ("stats",     "Run statistical tests"),
    ("findings",  "Generate findings registry"),
    ("dataset",   "Generate dataset overview"),
    ("update",    "Update report sections"),
    ("report",    "Build report"),
]

STEP_NAMES = [s[0] for s in STEPS]

# Old step names that were removed in the pipeline consolidation.
# Maps old name -> suggestion for the replacement.
OLD_STEP_NAMES = {
    "llm_analyze": "annotate",
    "normalize": "annotate",
    "edits": "analyze",
    "planning": "analyze",
    "compaction": "analyze",
}


def _file_hash(path):
    """Return SHA-256 hex digest of a file's contents, or None if missing."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (FileNotFoundError, IsADirectoryError):
        return None


def _glob_hashes(directory, pattern):
    """Hash all files matching a glob pattern in a directory.

    Returns dict mapping relative filename -> hash.
    """
    d = Path(directory)
    if not d.is_dir():
        return {}
    result = {}
    for f in sorted(d.glob(pattern)):
        if f.is_file():
            result[f.name] = _file_hash(f)
    return result


def get_step_io(step, data_dir, analysis_dir, comparison_dir):
    """Return (inputs_dict, outputs_dict) for a pipeline step.

    Each dict maps a label -> file path (or a dict of filename -> path for
    glob patterns). The manifest stores hashes of these files.
    """
    report_dir = comparison_dir / "report"
    expansions_dir = report_dir / "expansions"

    if step == "collect":
        # No file inputs (reads session dirs); outputs are session files
        return (
            {},
            _glob_hashes(data_dir, "sessions-*.json"),
        )
    elif step == "extract":
        return (
            _glob_hashes(data_dir, "sessions-*.json"),
            _glob_hashes(data_dir, "tasks-canonical-*.json"),
        )
    elif step == "classify":
        return (
            _glob_hashes(data_dir, "tasks-canonical-*.json"),
            _glob_hashes(data_dir, "tasks-classified-*.json"),
        )
    elif step == "annotate":
        return (
            _glob_hashes(data_dir, "tasks-canonical-*.json"),
            _glob_hashes(analysis_dir, "tasks-annotated-*.json"),
        )
    elif step == "analyze":
        inputs = _glob_hashes(data_dir, "tasks-canonical-*.json")
        outputs = {}
        for name in ["behavior-metrics.json", "edit-analysis.json",
                      "planning-analysis.json", "compaction-analysis.json",
                      "timing-analysis.json"]:
            h = _file_hash(analysis_dir / name)
            if h:
                outputs[name] = h
        return (inputs, outputs)
    elif step == "tokens":
        inputs = _glob_hashes(data_dir, "tasks-canonical-*.json")
        h = _file_hash(analysis_dir / "token-analysis.json")
        outputs = {"token-analysis.json": h} if h else {}
        return (inputs, outputs)
    elif step == "enrich":
        inputs = {}
        inputs.update(_glob_hashes(analysis_dir, "tasks-annotated-*.json"))
        inputs.update(_glob_hashes(data_dir, "tokens-*.json"))
        inputs.update(_glob_hashes(analysis_dir, "edit-metrics-*.json"))
        inputs.update(_glob_hashes(data_dir, "tasks-classified-*.json"))
        outputs = _glob_hashes(analysis_dir, "tasks-enriched-*.json")
        return (inputs, outputs)
    elif step == "stats":
        # Prefer enriched, fall back to annotated
        inputs = _glob_hashes(analysis_dir, "tasks-enriched-*.json")
        if not inputs:
            inputs = _glob_hashes(analysis_dir, "tasks-annotated-*.json")
        h = _file_hash(analysis_dir / "stat-tests.json")
        outputs = {"stat-tests.json": h} if h else {}
        return (inputs, outputs)
    elif step == "findings":
        h_in = _file_hash(analysis_dir / "stat-tests.json")
        inputs = {"stat-tests.json": h_in} if h_in else {}
        h_out = _file_hash(analysis_dir / "findings.json")
        outputs = {"findings.json": h_out} if h_out else {}
        return (inputs, outputs)
    elif step == "dataset":
        # Depends on all analysis outputs
        inputs = {}
        for name in ["behavior-metrics.json", "edit-analysis.json",
                      "planning-analysis.json", "compaction-analysis.json",
                      "token-analysis.json", "stat-tests.json",
                      "findings.json"]:
            h = _file_hash(analysis_dir / name)
            if h:
                inputs[name] = h
        h = _file_hash(analysis_dir / "dataset-overview.json")
        outputs = {"dataset-overview.json": h} if h else {}
        return (inputs, outputs)
    elif step == "update":
        # Template + all analysis files -> expansions
        inputs = {}
        template = report_dir / "report.html"
        h = _file_hash(template)
        if h:
            inputs["report.html"] = h
        for name in ["behavior-metrics.json", "edit-analysis.json",
                      "planning-analysis.json", "compaction-analysis.json",
                      "token-analysis.json", "stat-tests.json",
                      "dataset-overview.json", "findings.json"]:
            h = _file_hash(analysis_dir / name)
            if h:
                inputs[name] = h
        outputs = _glob_hashes(expansions_dir, "*.html")
        return (inputs, outputs)
    elif step == "report":
        inputs = {}
        template = report_dir / "report.html"
        h = _file_hash(template)
        if h:
            inputs["report.html"] = h
        for name, fh in _glob_hashes(expansions_dir, "*.html").items():
            inputs[f"expansions/{name}"] = fh
        dist_report = comparison_dir / "dist" / "public" / "report.html"
        h = _file_hash(dist_report)
        outputs = {"dist/public/report.html": h} if h else {}
        return (inputs, outputs)
    else:
        return ({}, {})


# -- Manifest --

def load_manifest(data_dir):
    """Load the pipeline manifest from data_dir, or return empty structure."""
    manifest_path = Path(data_dir) / "pipeline-manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"steps": {}}


def save_manifest(data_dir, manifest):
    """Write the pipeline manifest to data_dir."""
    manifest_path = Path(data_dir) / "pipeline-manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def is_step_stale(step, manifest, data_dir, analysis_dir, comparison_dir):
    """Check if a step's inputs have changed since its last recorded run.

    Returns (stale: bool, reason: str).
    """
    step_record = manifest.get("steps", {}).get(step)
    if not step_record:
        return True, "never run"

    current_inputs, _ = get_step_io(step, data_dir, analysis_dir, comparison_dir)

    recorded_inputs = step_record.get("inputs", {})

    if current_inputs != recorded_inputs:
        # Find what changed
        changed = []
        all_keys = set(current_inputs) | set(recorded_inputs)
        for k in sorted(all_keys):
            if k not in current_inputs:
                changed.append(f"{k} removed")
            elif k not in recorded_inputs:
                changed.append(f"{k} added")
            elif current_inputs[k] != recorded_inputs[k]:
                changed.append(f"{k} modified")
        return True, ", ".join(changed[:3]) + ("..." if len(changed) > 3 else "")

    return False, "inputs unchanged"


def record_step(step, manifest, data_dir, analysis_dir, comparison_dir):
    """Record a step's completion in the manifest with current file hashes."""
    inputs, outputs = get_step_io(step, data_dir, analysis_dir, comparison_dir)
    manifest.setdefault("steps", {})[step] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
    }


# -- Commands --

def build_command(step, data_dir, analysis_dir):
    """Return the command list for a given step."""
    scripts_dir = Path(__file__).resolve().parent
    comparison_dir = data_dir.parent

    if step == "collect":
        return [sys.executable, str(scripts_dir / "collect_sessions.py"),
                "--data-dir", str(data_dir)]
    elif step == "extract":
        return [sys.executable, str(scripts_dir / "extract_tasks.py"),
                "--data-dir", str(data_dir), "--canonical"]
    elif step == "classify":
        return [sys.executable, str(scripts_dir / "classify_tasks.py"),
                "--data-dir", str(data_dir)]
    elif step == "annotate":
        return [sys.executable, str(scripts_dir / "annotate_tasks.py"),
                "--data-dir", str(data_dir),
                "--output-dir", str(analysis_dir)]
    elif step == "analyze":
        return [sys.executable, str(scripts_dir / "run_analyses.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "tokens":
        return [sys.executable, str(scripts_dir / "extract_tokens.py"),
                "--dir", str(comparison_dir)]
    elif step == "enrich":
        return [sys.executable, str(scripts_dir / "enrich_tasks.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "stats":
        return [sys.executable, str(scripts_dir / "stat_tests.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "findings":
        return [sys.executable, str(scripts_dir / "generate_findings.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "dataset":
        return [sys.executable, str(scripts_dir / "analyze_dataset.py"),
                "--data-dir", str(data_dir),
                "--analysis-dir", str(analysis_dir)]
    elif step == "update":
        return [sys.executable, str(scripts_dir / "update_sections.py"),
                "--dir", str(comparison_dir)]
    elif step == "report":
        return [sys.executable, str(scripts_dir / "build_report.py"),
                "--dir", str(comparison_dir)]
    else:
        raise ValueError(f"Unknown step: {step}")


def run_step(name, label, data_dir, analysis_dir, comparison_dir,
             manifest, force=False):
    """Run a single pipeline step with staleness detection.

    Returns: "success", "skipped", or "failed".
    """
    # Check staleness
    if not force:
        stale, reason = is_step_stale(name, manifest, data_dir, analysis_dir,
                                      comparison_dir)
        if not stale:
            print(f"\n  [{name}] SKIPPED: {reason}")
            return "skipped"

    cmd = build_command(name, data_dir, analysis_dir)
    sys.stdout.flush()
    print(f"\n{'='*60}")
    print(f"  [{name}] {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)

    try:
        subprocess.run(cmd, check=True)
        print(f"  [{name}] SUCCESS")
        record_step(name, manifest, data_dir, analysis_dir, comparison_dir)
        save_manifest(data_dir, manifest)
        return "success"
    except subprocess.CalledProcessError as e:
        print(f"  [{name}] FAILED (exit code {e.returncode})", file=sys.stderr)
        return "failed"
    except FileNotFoundError as e:
        print(f"  [{name}] FAILED ({e})", file=sys.stderr)
        return "failed"


# -- Consistency checks --

def check_consistency(data_dir, analysis_dir):
    """Verify all analysis files reference the same task counts per model.

    Returns True if consistent, False if mismatches found.
    """
    # Discover models from canonical task files (count only included tasks)
    canonical_counts = {}
    for f in sorted(Path(data_dir).glob("tasks-canonical-*.json")):
        model = f.stem.replace("tasks-canonical-", "")
        with open(f) as fh:
            tasks = json.load(fh)
        total = len(tasks)
        included = sum(1 for t in tasks if not t.get('exclude_reason'))
        canonical_counts[model] = included

    if not canonical_counts:
        print("  No canonical task files found.")
        return True

    print(f"\n  Canonical task counts (included):")
    for model, count in canonical_counts.items():
        print(f"    {model}: {count} tasks")

    # Check analysis files: counts should be > 0 and <= canonical (some scripts
    # apply additional filters like is_meta exclusion)
    mismatches = []

    def check_count(label, n, model):
        expected = canonical_counts[model]
        if n is not None and n > 0 and n > expected:
            mismatches.append(f"{label}: {n} (exceeds canonical {expected})")
        elif n is not None and n == 0:
            mismatches.append(f"{label}: 0 tasks (expected > 0)")

    # behavior-metrics.json
    bm_path = analysis_dir / "behavior-metrics.json"
    if bm_path.exists():
        with open(bm_path) as f:
            bm = json.load(f)
        for model in canonical_counts:
            total = bm.get(model, {}).get("total_tasks")
            check_count(f"behavior-metrics.json[{model}].total_tasks", total, model)

    # token-analysis.json — operates on JSONL sessions, may include excluded tasks
    # Only check for zero counts, not upper bounds
    ta_path = analysis_dir / "token-analysis.json"
    if ta_path.exists():
        with open(ta_path) as f:
            ta = json.load(f)
        for model in canonical_counts:
            n = ta.get(model, {}).get("overall", {}).get("count")
            if n is not None and n == 0:
                mismatches.append(f"token-analysis.json[{model}].overall.count: 0 tasks")

    # stat-tests.json
    st_path = analysis_dir / "stat-tests.json"
    if st_path.exists():
        with open(st_path) as f:
            st = json.load(f)
        sizes = st.get("metadata", {}).get("sample_sizes", {})
        for model in canonical_counts:
            n = sizes.get(model)
            check_count(f"stat-tests.json.metadata.sample_sizes.{model}", n, model)

    if mismatches:
        print(f"\n  CONSISTENCY ERRORS:")
        for m in mismatches:
            print(f"    {m}")
        return False
    else:
        print(f"\n  All analysis files consistent with canonical counts.")
        return True


# -- Stale check --

def print_stale_status(steps_to_check, manifest, data_dir, analysis_dir,
                       comparison_dir):
    """Print staleness status for each step without running anything."""
    print(f"\nPipeline staleness report:")
    print(f"{'='*60}")
    for name, label in steps_to_check:
        stale, reason = is_step_stale(name, manifest, data_dir, analysis_dir,
                                      comparison_dir)
        if stale:
            print(f"  STALE:   {name:12s} ({reason})")
        else:
            print(f"  CURRENT: {name:12s}")
    print(f"{'='*60}")


def resolve_step_name(name):
    """Resolve a step name, handling old names with helpful errors."""
    if name in STEP_NAMES:
        return name
    if name in OLD_STEP_NAMES:
        replacement = OLD_STEP_NAMES[name]
        print(f"Note: step '{name}' was consolidated into '{replacement}'",
              file=sys.stderr)
        return replacement
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run the model comparison analysis pipeline")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to the data directory")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Path to analysis output directory "
                             "(default: data-dir/../analysis)")
    parser.add_argument("--steps", type=str, default=None,
                        help=f"Comma-separated list of steps to run: "
                             f"{','.join(STEP_NAMES)}")
    parser.add_argument("--from", dest="from_step", type=str, default=None,
                        help=f"Start from this step: {','.join(STEP_NAMES)}")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if inputs unchanged")
    parser.add_argument("--check-stale", action="store_true",
                        help="Report staleness status without running steps")
    parser.add_argument("--check-consistency", action="store_true",
                        help="Verify task counts are consistent across "
                             "analysis files")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    analysis_dir = (args.analysis_dir or data_dir.parent / "analysis").resolve()
    comparison_dir = data_dir.parent

    # Determine which steps to consider
    if args.steps and args.from_step:
        print("Error: --steps and --from are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)

    if args.steps:
        selected = []
        for s in args.steps.split(","):
            s = s.strip()
            resolved = resolve_step_name(s)
            if resolved is None:
                print(f"Error: unknown step '{s}'. "
                      f"Valid steps: {', '.join(STEP_NAMES)}",
                      file=sys.stderr)
                sys.exit(1)
            selected.append(resolved)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in selected:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        steps_to_run = [(name, label) for name, label in STEPS
                        if name in unique]
    elif args.from_step:
        resolved = resolve_step_name(args.from_step)
        if resolved is None:
            print(f"Error: unknown step '{args.from_step}'. "
                  f"Valid steps: {', '.join(STEP_NAMES)}", file=sys.stderr)
            sys.exit(1)
        start_idx = STEP_NAMES.index(resolved)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = list(STEPS)

    manifest = load_manifest(data_dir)

    # Handle --check-consistency
    if args.check_consistency:
        ok = check_consistency(data_dir, analysis_dir)
        sys.exit(0 if ok else 1)

    # Handle --check-stale
    if args.check_stale:
        print_stale_status(steps_to_run, manifest, data_dir, analysis_dir,
                           comparison_dir)
        sys.exit(0)

    # Normal pipeline run
    print(f"Data dir:     {data_dir}")
    print(f"Analysis dir: {analysis_dir}")
    print(f"Steps:        {', '.join(s[0] for s in steps_to_run)}")
    if args.force:
        print(f"Mode:         FORCE (ignoring staleness)")

    succeeded = 0
    skipped = 0
    failed = 0
    for name, label in steps_to_run:
        result = run_step(name, label, data_dir, analysis_dir, comparison_dir,
                          manifest, force=args.force)
        if result == "success":
            succeeded += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    parts = [f"{succeeded} succeeded"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    print(f"  Pipeline complete: {', '.join(parts)}")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
