#!/usr/bin/env python3
"""Model discovery and configuration for comparison pipelines."""

import json
import re
from pathlib import Path


def discover_models(data_dir, prefix="sessions"):
    """Discover model names from session files in a data directory.

    Globs for {prefix}-*.json and extracts model names.
    E.g., "sessions-opus-4-5.json" -> "opus-4-5".

    Returns sorted list of model name strings.
    """
    data_dir = Path(data_dir)
    pattern = f"{prefix}-*.json"
    files = sorted(data_dir.glob(pattern))
    models = []
    for f in files:
        # sessions-opus-4-5.json -> opus-4-5
        name = f.stem.replace(f"{prefix}-", "", 1)
        models.append(name)
    return sorted(models)


def discover_model_pair(data_dir, prefix="sessions"):
    """Discover exactly two models from session files.

    Raises ValueError if != 2 models found.
    Returns (model_a, model_b) tuple.
    """
    models = discover_models(data_dir, prefix)
    if len(models) != 2:
        raise ValueError(
            f"Expected exactly 2 models in {data_dir}, found {len(models)}: {models}"
        )
    return (models[0], models[1])


def load_canonical_tasks(data_dir, model, include_excluded=False):
    """Load canonical tasks for a model, filtering excluded tasks by default.

    Args:
        data_dir: Path to data directory
        model: Model name (e.g., "opus-4-5")
        include_excluded: If True, return all tasks including excluded ones

    Returns list of task dicts. Empty list if file not found.
    """
    data_dir = Path(data_dir)
    tasks_file = data_dir / f'tasks-canonical-{model}.json'
    if not tasks_file.exists():
        return []

    with open(tasks_file) as f:
        tasks = json.load(f)

    if not include_excluded:
        tasks = [t for t in tasks if not t.get('exclude_reason')]

    return tasks


def load_comparison_config(comparison_dir):
    """Parse comparison directory name to extract model configuration.

    The directory name format is like "opus-4.5-vs-4.6".

    Returns dict with keys: model_a, model_b, display_a, display_b, family.
    Also merges any optional models.json from the data dir.
    """
    comparison_dir = Path(comparison_dir)
    name = comparison_dir.name
    m = re.match(r'(\w+)-([\d.]+)-vs-([\d.]+)', name)
    if not m:
        raise ValueError(
            f"Cannot parse model names from directory '{name}'. "
            "Expected format: <family>-<version>-vs-<version>"
        )
    family = m.group(1)
    ver_a = m.group(2)
    ver_b = m.group(3)
    config = {
        "model_a": f"{family}-{ver_a.replace('.', '-')}",
        "model_b": f"{family}-{ver_b.replace('.', '-')}",
        "display_a": ver_a,
        "display_b": ver_b,
        "family": family,
    }

    # Merge optional models.json
    models_json = comparison_dir / "data" / "models.json"
    if models_json.exists():
        with open(models_json) as f:
            overrides = json.load(f)
        config.update(overrides)

    return config
