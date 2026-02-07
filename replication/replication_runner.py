#!/usr/bin/env python3
"""
Replication runner for comparing model performance against baseline tasks.

Usage:
    python replication_runner.py task-config.json --output-dir results/
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def setup_worktree(repo_path: Path, commit: str, work_dir: Path) -> bool:
    """Create a git worktree at the specified commit.

    Args:
        repo_path: Path to the git repository
        commit: Git commit hash or ref
        work_dir: Directory for the worktree

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create worktree
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(work_dir), commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error creating worktree: {result.stderr}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception creating worktree: {e}", file=sys.stderr)
        return False


def cleanup_worktree(repo_path: Path, work_dir: Path) -> None:
    """Remove a git worktree."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(work_dir)],
            cwd=repo_path,
            capture_output=True,
        )
    except Exception:
        pass  # Best effort cleanup


def setup_greenfield(work_dir: Path, context_sources: list[dict]) -> bool:
    """Set up a greenfield directory with optional context files.

    Args:
        work_dir: Directory to create
        context_sources: List of {src, dst} mappings for files/dirs to copy

    Returns:
        True if successful, False otherwise
    """
    try:
        work_dir.mkdir(parents=True, exist_ok=True)

        for source in context_sources:
            src = expand_path(source["src"])
            dst = work_dir / source["dst"]

            if not src.exists():
                print(f"Warning: Source does not exist: {src}", file=sys.stderr)
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        return True
    except Exception as e:
        print(f"Exception setting up greenfield: {e}", file=sys.stderr)
        return False


def run_claude(
    work_dir: Path, prompt: str, model: str, timeout: int = 1800
) -> dict[str, Any]:
    """Run claude -p in the specified directory.

    Args:
        work_dir: Working directory for claude
        prompt: The prompt to send
        model: Model identifier
        timeout: Timeout in seconds (default 30 minutes)

    Returns:
        Dict with stdout, stderr, exit_code, duration
    """
    start_time = time.time()

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model, "--dangerously-skip-permissions"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start_time

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "duration_seconds": duration,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "stdout": "",
            "stderr": f"Timeout after {timeout} seconds",
            "exit_code": -1,
            "duration_seconds": duration,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "duration_seconds": duration,
        }


def find_session_file(work_dir: Path) -> Path | None:
    """Find the most recent session file matching the work directory.

    Claude Code stores sessions in ~/.claude/projects/ with the path
    encoded as -home-user-path-to-project/

    Args:
        work_dir: The working directory used for the claude run

    Returns:
        Path to the session file, or None if not found
    """
    claude_projects = Path.home() / ".claude" / "projects"

    if not claude_projects.exists():
        return None

    # Convert work_dir to the encoded format Claude uses
    # /home/user/foo/bar -> -home-user-foo-bar
    # Claude also converts underscores to dashes
    work_dir_str = str(work_dir.resolve())
    encoded_path = work_dir_str.replace("/", "-").replace("_", "-")

    project_dir = claude_projects / encoded_path

    if not project_dir.exists():
        # Try to find a matching directory
        for candidate in claude_projects.iterdir():
            if candidate.is_dir() and work_dir_str.replace("/", "-") in str(candidate):
                project_dir = candidate
                break
        else:
            return None

    # Find the most recent .jsonl file
    jsonl_files = list(project_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Return most recently modified
    return max(jsonl_files, key=lambda p: p.stat().st_mtime)


def run_tests(work_dir: Path, test_commands: list[str]) -> list[dict[str, Any]]:
    """Execute verification commands.

    Args:
        work_dir: Directory to run commands in
        test_commands: List of shell commands to run

    Returns:
        List of result dicts with command, stdout, stderr, exit_code
    """
    results = []

    for cmd in test_commands:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            results.append({
                "command": cmd,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            })
        except subprocess.TimeoutExpired:
            results.append({
                "command": cmd,
                "stdout": "",
                "stderr": "Timeout",
                "exit_code": -1,
            })
        except Exception as e:
            results.append({
                "command": cmd,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
            })

    return results


def collect_metrics(session_file: Path | None) -> dict[str, Any]:
    """Extract metrics from a Claude session file.

    Args:
        session_file: Path to the .jsonl session file

    Returns:
        Dict with tool_count, tool_breakdown, etc.
    """
    if session_file is None or not session_file.exists():
        return {"error": "No session file found"}

    tool_counts: dict[str, int] = {}
    total_tools = 0
    message_count = 0

    try:
        with open(session_file) as f:
            for line in f:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message_count += 1

                # Count tool uses
                if msg.get("type") == "assistant":
                    content = msg.get("message", {}).get("content", [])
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                            total_tools += 1

        return {
            "total_tools": total_tools,
            "tool_breakdown": tool_counts,
            "message_count": message_count,
            "session_file": str(session_file),
        }
    except Exception as e:
        return {"error": str(e)}


def check_success_patterns(
    test_results: list[dict], patterns: list[str]
) -> dict[str, bool]:
    """Check if success patterns appear in test output.

    Args:
        test_results: Results from run_tests()
        patterns: List of patterns to search for

    Returns:
        Dict mapping pattern to whether it was found
    """
    combined_output = ""
    for result in test_results:
        combined_output += result.get("stdout", "") + result.get("stderr", "")

    return {pattern: pattern in combined_output for pattern in patterns}


def main():
    parser = argparse.ArgumentParser(
        description="Run replication task and compare against baseline"
    )
    parser.add_argument("task_config", help="Path to task config JSON file")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files (default: results)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-5-20251101",
        help="Model to use (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the working directory after completion",
    )
    args = parser.parse_args()

    # Load task config
    config_path = Path(args.task_config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # Load prompt
    prompt_path = config_path.parent / config["prompt_file"]
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)

    with open(prompt_path) as f:
        prompt = f.read()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config['name']}_{timestamp}"

    print(f"Starting replication: {config['name']}")
    print(f"Run ID: {run_id}")
    print(f"Model: {args.model}")

    # Set up working directory
    setup_config = config["setup"]
    work_dir = Path(tempfile.mkdtemp(prefix=f"replication_{config['name']}_"))

    print(f"Working directory: {work_dir}")

    repo_path = None
    try:
        if setup_config["type"] == "worktree":
            repo_path = expand_path(setup_config["repo"])
            commit = setup_config["commit"]
            print(f"Setting up worktree from {repo_path} at {commit}")
            if not setup_worktree(repo_path, commit, work_dir):
                print("Failed to set up worktree", file=sys.stderr)
                sys.exit(1)
        elif setup_config["type"] == "greenfield":
            context_sources = setup_config.get("context_sources", [])
            print(f"Setting up greenfield with {len(context_sources)} context sources")
            if not setup_greenfield(work_dir, context_sources):
                print("Failed to set up greenfield", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Unknown setup type: {setup_config['type']}", file=sys.stderr)
            sys.exit(1)

        # Run claude
        print(f"Running claude with model {args.model}...")
        claude_result = run_claude(work_dir, prompt, args.model)

        print(f"Claude completed in {claude_result['duration_seconds']:.1f}s")
        print(f"Exit code: {claude_result['exit_code']}")

        # Find session file
        session_file = find_session_file(work_dir)
        if session_file:
            print(f"Session file: {session_file}")

        # Collect metrics
        metrics = collect_metrics(session_file)

        # Run verification tests
        verification = config.get("verification", {})
        test_commands = verification.get("commands", [])
        print(f"Running {len(test_commands)} verification commands...")
        test_results = run_tests(work_dir, test_commands)

        # Check success patterns
        success_patterns = verification.get("success_patterns", [])
        pattern_results = check_success_patterns(test_results, success_patterns)

        # Determine overall success
        tests_passed = all(r["exit_code"] == 0 for r in test_results)
        patterns_matched = all(pattern_results.values()) if pattern_results else True
        overall_success = tests_passed and patterns_matched

        # Copy session file to output
        if session_file and session_file.exists():
            session_copy = output_dir / f"{run_id}_session.jsonl"
            shutil.copy2(session_file, session_copy)

        # Build result
        result = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "task": config["name"],
            "description": config.get("description", ""),
            "model": args.model,
            "original_model": config.get("original_model", "unknown"),
            "original_metrics": config.get("original_metrics", {}),
            "replication_metrics": {
                "duration_seconds": claude_result["duration_seconds"],
                "exit_code": claude_result["exit_code"],
                **metrics,
            },
            "verification": {
                "tests": test_results,
                "patterns": pattern_results,
                "tests_passed": tests_passed,
                "patterns_matched": patterns_matched,
                "overall_success": overall_success,
            },
            "work_dir": str(work_dir),
            "session_file": str(session_file) if session_file else None,
        }

        # Save result
        result_path = output_dir / f"{run_id}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nResults saved to: {result_path}")
        print(f"Overall success: {overall_success}")

        # Print comparison
        if "original_metrics" in config:
            orig = config["original_metrics"]
            rep = result["replication_metrics"]
            print("\n=== Comparison ===")
            print(f"{'Metric':<20} {'Original':>12} {'Replication':>12} {'Delta':>12}")
            print("-" * 56)

            if "tools" in orig and "total_tools" in rep:
                delta = rep["total_tools"] - orig["tools"]
                print(f"{'Tools':<20} {orig['tools']:>12} {rep['total_tools']:>12} {delta:>+12}")

            if "duration_seconds" in orig and "duration_seconds" in rep:
                delta = rep["duration_seconds"] - orig["duration_seconds"]
                print(
                    f"{'Duration (s)':<20} {orig['duration_seconds']:>12.1f} "
                    f"{rep['duration_seconds']:>12.1f} {delta:>+12.1f}"
                )

    finally:
        # Cleanup
        if not args.keep_workdir:
            if repo_path and setup_config["type"] == "worktree":
                cleanup_worktree(repo_path, work_dir)
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            print(f"Cleaned up working directory")
        else:
            print(f"Keeping working directory: {work_dir}")


if __name__ == "__main__":
    main()
