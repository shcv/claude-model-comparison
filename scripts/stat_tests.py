#!/usr/bin/env python3
"""Statistical significance testing for model comparison data.

Runs Chi-square tests, Mann-Whitney U tests, calculates confidence intervals
(Wilson score) and effect sizes (Cohen's h, Cohen's d) for comparing two models.
Tests are run overall and across configurable cross-cutting dimensions.

Supports config-driven measurements (via --config), FDR correction
(Benjamini-Hochberg), and theme tagging for downstream findings generation.

Usage:
    python scripts/stat_tests.py --data-dir data --analysis-dir analysis
    python scripts/stat_tests.py --data-dir data --analysis-dir analysis --config scripts/analysis_config.json
    python scripts/stat_tests.py --data-dir data --analysis-dir analysis --sensitivity
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_model_pair, load_canonical_tasks


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


MODELS = ["opus-4-5", "opus-4-6"]  # set dynamically in main()
COMPLEXITY_BINS = ["trivial", "simple", "moderate", "complex"]

# Default field lists (used when no --config provided)
DEFAULT_CATEGORICAL_FIELDS = {
    "task_completion": None,
    "normalized_user_sentiment": None,
    "scope_management": None,
    "iteration_required": None,
    "error_recovery": None,
    "communication_quality": None,
    "normalized_execution_quality": None,
    "autonomy_level": None,
}

DEFAULT_PROPORTION_FIELDS = {
    "satisfaction_rate": lambda rec: rec.get("normalized_user_sentiment") == "satisfied",
    "dissatisfaction_rate": lambda rec: rec.get("normalized_user_sentiment") == "dissatisfied",
    "complete_rate": lambda rec: rec.get("task_completion") == "complete",
    "failed_rate": lambda rec: rec.get("task_completion") == "failed",
    "scope_expanded_rate": lambda rec: rec.get("scope_management") == "expanded",
    "one_shot_rate": lambda rec: rec.get("iteration_required") == "one_shot",
    "good_execution_rate": lambda rec: rec.get("normalized_execution_quality") == "good",
}

DEFAULT_CONTINUOUS_FIELDS = [
    "alignment_score",
    "duration_seconds",
    "tool_calls",
    "files_touched",
    "lines_added",
    "lines_removed",
    "lines_per_minute",
    "tools_per_file",
]

# Module-level field lists (overridden by config if provided)
CATEGORICAL_FIELDS = dict(DEFAULT_CATEGORICAL_FIELDS)
PROPORTION_FIELDS = dict(DEFAULT_PROPORTION_FIELDS)
CONTINUOUS_FIELDS = list(DEFAULT_CONTINUOUS_FIELDS)

# Cross-cuts config (default: overall + complexity)
CROSS_CUTS = None  # Set from config or defaults

# Theme mapping (field -> theme name)
THEME_MAP = {}


def load_analysis_config(config_path):
    """Load analysis config and populate module-level field lists."""
    global CATEGORICAL_FIELDS, PROPORTION_FIELDS, CONTINUOUS_FIELDS
    global CROSS_CUTS, THEME_MAP

    with open(config_path) as f:
        config = json.load(f)

    measurements = config.get("measurements", {})

    # Categorical fields
    cat_list = measurements.get("categorical", [])
    CATEGORICAL_FIELDS = {f: None for f in cat_list}

    # Continuous fields
    CONTINUOUS_FIELDS = measurements.get("continuous", [])

    # Proportion fields: build lambda predicates from declarative specs
    prop_specs = measurements.get("proportions", {})
    PROPORTION_FIELDS = {}
    for name, spec in prop_specs.items():
        field = spec["field"]
        value = spec["value"]
        # Capture field/value in closure
        PROPORTION_FIELDS[name] = _make_predicate(field, value)

    # Cross-cuts
    CROSS_CUTS = config.get("cross_cuts")

    # Theme map: field -> theme
    themes = config.get("themes", {})
    THEME_MAP.clear()
    for theme_name, fields in themes.items():
        for f in fields:
            THEME_MAP[f] = theme_name

    return config


def _make_predicate(field, value):
    """Create a predicate function for proportion testing."""
    def predicate(rec):
        return rec.get(field) == value
    return predicate


def load_data(data_dir, analysis_dir):
    """Load task data for both models.

    Prefers tasks-enriched (output of enrich_tasks.py), then tasks-annotated,
    then falls back to llm-analysis + tasks-classified.

    Enriches records with project_path from canonical tasks for sensitivity analysis.
    """
    data = {}
    for model in MODELS:
        # Try enriched first
        enriched_path = Path(analysis_dir) / f"tasks-enriched-{model}.json"
        if enriched_path.exists():
            print(f"  Loading enriched tasks: {enriched_path}")
            with open(enriched_path) as f:
                records = json.load(f)
            # Ensure project_path from canonical
            canonical = load_canonical_tasks(data_dir, model)
            canonical_by_id = {t['task_id']: t for t in canonical}
            for rec in records:
                tid = rec.get('task_id', '')
                if tid in canonical_by_id:
                    ct = canonical_by_id[tid]
                    rec.setdefault('project_path', ct.get('project_path', ''))
                    rec.setdefault('session_id', ct.get('session_id', ''))
            data[model] = records
            continue

        # Try annotated
        annotated_path = Path(analysis_dir) / f"tasks-annotated-{model}.json"
        if annotated_path.exists():
            print(f"  Loading annotated tasks: {annotated_path}")
            with open(annotated_path) as f:
                records = json.load(f)
            canonical = load_canonical_tasks(data_dir, model)
            canonical_by_id = {t['task_id']: t for t in canonical}
            for rec in records:
                tid = rec.get('task_id', '')
                if tid in canonical_by_id:
                    ct = canonical_by_id[tid]
                    rec.setdefault('project_path', ct.get('project_path', ''))
                    rec.setdefault('session_id', ct.get('session_id', ''))
            data[model] = records
            continue

        # Fallback: merge llm-analysis + tasks-classified
        analysis_path = Path(analysis_dir) / f"llm-analysis-{model}.json"
        classified_path = Path(data_dir) / f"tasks-classified-{model}.json"

        print(f"  Falling back to llm-analysis + tasks-classified for {model}")
        with open(analysis_path) as f:
            analysis = json.load(f)
        with open(classified_path) as f:
            classified = json.load(f)

        classified_by_id = {t["task_id"]: t for t in classified}
        canonical = load_canonical_tasks(data_dir, model)
        canonical_by_id = {t['task_id']: t for t in canonical}
        for rec in analysis:
            tid = rec["task_id"]
            if tid in classified_by_id:
                cl = classified_by_id[tid].get("classification", {})
                rec["complexity"] = cl.get("complexity", "unknown")
            else:
                rec["complexity"] = "unknown"
            if tid in canonical_by_id:
                ct = canonical_by_id[tid]
                rec.setdefault('project_path', ct.get('project_path', ''))
                rec.setdefault('session_id', ct.get('session_id', ''))

        data[model] = analysis

    return data


def get_values(records, field):
    """Extract non-None values for a field from records."""
    return [r[field] for r in records if r.get(field) is not None]


def get_bool_counts(records, predicate):
    """Count True/False for a predicate across records."""
    results = [predicate(r) for r in records]
    return sum(results), len(results) - sum(results)


def wilson_score_interval(successes, total, z=1.96):
    """Calculate Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (p, lower, upper)


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(vals1, vals2):
    """Cohen's d effect size for two samples."""
    n1, n2 = len(vals1), len(vals2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    mean1, mean2 = np.mean(vals1), np.mean(vals2)
    var1, var2 = np.var(vals1, ddof=1), np.var(vals2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def effect_size_label(d):
    """Interpret effect size magnitude."""
    ad = abs(d)
    if math.isnan(ad):
        return "n/a"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def chi_square_test(data_a, data_b, field):
    """Run Chi-square test on a categorical field between two groups."""
    counts_a = Counter(r.get(field) for r in data_a if r.get(field) is not None)
    counts_b = Counter(r.get(field) for r in data_b if r.get(field) is not None)

    all_categories = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    if len(all_categories) < 2:
        return None

    observed = np.array([
        [counts_a.get(c, 0) for c in all_categories],
        [counts_b.get(c, 0) for c in all_categories],
    ])

    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    if total == 0:
        return None

    expected = np.outer(row_totals, col_totals) / total
    low_expected = bool(np.any(expected < 5))

    try:
        chi2, p_value, dof, _ = stats.chi2_contingency(observed)
    except ValueError:
        return None

    min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / (total * min_dim)) if total * min_dim > 0 else 0.0

    return {
        "test": "chi-square",
        "field": field,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
        "effect_size": round(cramers_v, 4),
        "effect_size_label": effect_size_label(cramers_v),
        "significant_p05": p_value < 0.05,
        "low_expected_warning": low_expected,
        "categories": all_categories,
        "counts": {
            MODELS[0]: {c: counts_a.get(c, 0) for c in all_categories},
            MODELS[1]: {c: counts_b.get(c, 0) for c in all_categories},
        },
        "n": {MODELS[0]: sum(counts_a.values()), MODELS[1]: sum(counts_b.values())},
    }


def bootstrap_ci(values, stat_func=np.mean, n_boot=5000, alpha=0.05):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(42)
    arr = np.array(values)
    n = len(arr)
    if n < 3:
        return (float('nan'), float('nan'))
    boot_stats = np.array([
        stat_func(rng.choice(arr, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (round(lo, 3), round(hi, 3))


def mann_whitney_test(data_a, data_b, field):
    """Run Mann-Whitney U test on a continuous field."""
    vals_a = [r[field] for r in data_a if r.get(field) is not None and not (isinstance(r[field], float) and math.isnan(r[field]))]
    vals_b = [r[field] for r in data_b if r.get(field) is not None and not (isinstance(r[field], float) and math.isnan(r[field]))]

    if len(vals_a) < 3 or len(vals_b) < 3:
        return None

    try:
        u_stat, p_value = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    except ValueError:
        return None

    d = cohens_d(vals_a, vals_b)

    ci_a = bootstrap_ci(vals_a, np.mean)
    ci_b = bootstrap_ci(vals_b, np.mean)
    ci_med_a = bootstrap_ci(vals_a, np.median)
    ci_med_b = bootstrap_ci(vals_b, np.median)

    return {
        "test": "mann-whitney-u",
        "field": field,
        "u_statistic": round(float(u_stat), 2),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(d, 4) if not math.isnan(d) else None,
        "effect_size": round(abs(d), 4) if not math.isnan(d) else None,
        "effect_size_label": effect_size_label(d),
        "significant_p05": p_value < 0.05,
        "n": {MODELS[0]: len(vals_a), MODELS[1]: len(vals_b)},
        "median": {MODELS[0]: round(float(np.median(vals_a)), 3), MODELS[1]: round(float(np.median(vals_b)), 3)},
        "mean": {MODELS[0]: round(float(np.mean(vals_a)), 3), MODELS[1]: round(float(np.mean(vals_b)), 3)},
        "std": {MODELS[0]: round(float(np.std(vals_a, ddof=1)), 3), MODELS[1]: round(float(np.std(vals_b, ddof=1)), 3)},
        "mean_ci": {MODELS[0]: list(ci_a), MODELS[1]: list(ci_b)},
        "median_ci": {MODELS[0]: list(ci_med_a), MODELS[1]: list(ci_med_b)},
    }


def proportion_test(data_a, data_b, name, predicate):
    """Test difference in proportions with Wilson CI and Cohen's h."""
    yes_a, no_a = get_bool_counts(data_a, predicate)
    yes_b, no_b = get_bool_counts(data_b, predicate)
    n_a = yes_a + no_a
    n_b = yes_b + no_b

    if n_a < 3 or n_b < 3:
        return None

    p_a, lo_a, hi_a = wilson_score_interval(yes_a, n_a)
    p_b, lo_b, hi_b = wilson_score_interval(yes_b, n_b)

    p_pool = (yes_a + yes_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if p_pool > 0 and p_pool < 1 else 0
    z = (p_a - p_b) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    h = cohens_h(p_a, p_b)

    return {
        "test": "two-proportion-z",
        "field": name,
        "z_statistic": round(z, 4),
        "p_value": round(p_value, 6),
        "cohens_h": round(h, 4),
        "effect_size": round(abs(h), 4),
        "effect_size_label": effect_size_label(h),
        "significant_p05": p_value < 0.05,
        MODELS[0]: {
            "proportion": round(p_a, 4),
            "ci_lower": round(lo_a, 4),
            "ci_upper": round(hi_a, 4),
            "count": yes_a,
            "n": n_a,
        },
        MODELS[1]: {
            "proportion": round(p_b, 4),
            "ci_lower": round(lo_b, 4),
            "ci_upper": round(hi_b, 4),
            "count": yes_b,
            "n": n_b,
        },
    }


def run_all_tests(data_a, data_b, label="overall"):
    """Run all statistical tests on two groups of records."""
    results = {"label": label, "chi_square": [], "mann_whitney": [], "proportions": []}

    for field in CATEGORICAL_FIELDS:
        result = chi_square_test(data_a, data_b, field)
        if result:
            results["chi_square"].append(result)

    for field in CONTINUOUS_FIELDS:
        result = mann_whitney_test(data_a, data_b, field)
        if result:
            results["mann_whitney"].append(result)

    for name, predicate in PROPORTION_FIELDS.items():
        result = proportion_test(data_a, data_b, name, predicate)
        if result:
            results["proportions"].append(result)

    return results


def count_tests(results_list):
    """Count total number of tests across all result groups."""
    total = 0
    for r in results_list:
        total += len(r["chi_square"]) + len(r["mann_whitney"]) + len(r["proportions"])
    return total


def flatten_tests(results_list):
    """Flatten all test results from grouped format into a single list.

    Each test gets a 'cross_cut' field from its group label.
    """
    flat = []
    for group in results_list:
        label = group["label"]
        for test_list in [group["chi_square"], group["mann_whitney"], group["proportions"]]:
            for t in test_list:
                t["cross_cut"] = label
                # Add theme from THEME_MAP
                field = t.get("field", "")
                t["theme"] = THEME_MAP.get(field)
                flat.append(t)
    return flat


def apply_fdr_correction(flat_tests, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction to all tests.

    Adds p_adjusted and fdr_significant fields to each test dict.
    """
    if not flat_tests:
        return

    p_values = np.array([t["p_value"] for t in flat_tests])

    # scipy's false_discovery_control returns adjusted p-values
    adjusted = false_discovery_control(p_values, method='bh')

    for t, adj_p in zip(flat_tests, adjusted):
        t["p_adjusted"] = round(float(adj_p), 8)
        t["fdr_significant"] = bool(adj_p < alpha)


def apply_bonferroni(flat_tests):
    """Apply Bonferroni correction to all tests."""
    n = len(flat_tests)
    if n == 0:
        return 0.05
    threshold = 0.05 / n
    for t in flat_tests:
        t["bonferroni_significant"] = t["p_value"] < threshold
    return threshold


def run_cross_cut_tests(data, cross_cuts=None):
    """Run tests across all configured cross-cuts.

    Returns (all_results_grouped, flat_tests).
    """
    all_results = []

    if cross_cuts is None:
        # Default: overall + complexity bins
        overall = run_all_tests(data[MODELS[0]], data[MODELS[1]], label="overall")
        all_results.append(overall)

        for cbin in COMPLEXITY_BINS:
            subset_a = [r for r in data[MODELS[0]] if r.get("complexity") == cbin]
            subset_b = [r for r in data[MODELS[1]] if r.get("complexity") == cbin]
            if len(subset_a) >= 3 and len(subset_b) >= 3:
                print(f"Running tests for complexity={cbin} ({len(subset_a)} vs {len(subset_b)})...")
                result = run_all_tests(subset_a, subset_b, label=f"complexity:{cbin}")
                all_results.append(result)
            else:
                print(f"Skipping complexity={cbin} (insufficient data: {len(subset_a)} vs {len(subset_b)})")
    else:
        for cc in cross_cuts:
            name = cc["name"]
            field = cc.get("field")
            values = cc.get("values")
            min_n = cc.get("min_n", 3)

            if cc.get("filter") is None and field is None:
                # Overall: no filtering
                print(f"Running tests for {name}...")
                result = run_all_tests(data[MODELS[0]], data[MODELS[1]], label=name)
                all_results.append(result)
            elif values:
                # Per-value slices
                for val in values:
                    subset_a = [r for r in data[MODELS[0]] if r.get(field) == val]
                    subset_b = [r for r in data[MODELS[1]] if r.get(field) == val]
                    if len(subset_a) >= min_n and len(subset_b) >= min_n:
                        label = f"{name}:{val}"
                        print(f"Running tests for {label} ({len(subset_a)} vs {len(subset_b)})...")
                        result = run_all_tests(subset_a, subset_b, label=label)
                        all_results.append(result)
                    else:
                        print(f"Skipping {name}:{val} (insufficient data: {len(subset_a)} vs {len(subset_b)})")

    return all_results


def regroup_results(flat_tests):
    """Re-group flat test results back into the grouped format for backward compat.

    Returns (overall_group, by_complexity_list, by_cross_cut_dict).
    """
    overall = {"label": "overall", "chi_square": [], "mann_whitney": [], "proportions": []}
    by_complexity = {}
    by_cross_cut = {}

    for t in flat_tests:
        cc = t.get("cross_cut", "overall")

        if cc == "overall":
            target = overall
        elif cc.startswith("complexity:"):
            if cc not in by_complexity:
                by_complexity[cc] = {"label": cc, "chi_square": [], "mann_whitney": [], "proportions": []}
            target = by_complexity[cc]
        else:
            if cc not in by_cross_cut:
                by_cross_cut[cc] = {"label": cc, "chi_square": [], "mann_whitney": [], "proportions": []}
            target = by_cross_cut[cc]

        test_type = t.get("test", "")
        if test_type == "chi-square":
            target["chi_square"].append(t)
        elif test_type == "mann-whitney-u":
            target["mann_whitney"].append(t)
        elif test_type == "two-proportion-z":
            target["proportions"].append(t)

    return overall, list(by_complexity.values()), by_cross_cut


def format_summary(all_results, total_tests, fdr_significant_count=None):
    """Format a human-readable summary table."""
    lines = []
    lines.append("=" * 100)
    lines.append("STATISTICAL SIGNIFICANCE TESTING SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Models compared: {MODELS[0]} vs {MODELS[1]}")
    lines.append(f"Total tests run: {total_tests}")
    bonferroni = 0.05 / total_tests if total_tests > 0 else 0.05
    lines.append(f"Bonferroni-corrected threshold: p < {bonferroni:.6f} (0.05 / {total_tests})")
    if fdr_significant_count is not None:
        lines.append(f"FDR significant (BH, alpha=0.05): {fdr_significant_count}")
    lines.append("")

    for group in all_results:
        label = group["label"].upper()
        lines.append("-" * 100)
        lines.append(f"  {label}")
        lines.append("-" * 100)

        sig_results = []
        all_tests_in_group = []

        # Proportion tests
        if group["proportions"]:
            lines.append("")
            lines.append(f"  {'Proportion':<28} {'Opus4.5':>10} {'Opus4.6':>10} {'Diff':>8} {'p-value':>10} {'p-adj':>10} {'h':>7} {'Effect':>12} {'Sig':>5}")
            lines.append(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*12} {'-'*5}")
            for t in group["proportions"]:
                p_a = t[MODELS[0]]["proportion"]
                p_b = t[MODELS[1]]["proportion"]
                diff = p_a - p_b
                p_adj = t.get("p_adjusted")
                p_adj_str = f"{p_adj:.4f}" if p_adj is not None else "   n/a"
                fdr = "†" if t.get("fdr_significant") else ""
                sig = "*" if t["significant_p05"] else ""
                bonf = "**" if t.get("bonferroni_significant") else sig
                mark = bonf + fdr
                lines.append(
                    f"  {t['field']:<28} {p_a:>9.1%} {p_b:>9.1%} {diff:>+7.1%} {t['p_value']:>10.4f} {p_adj_str:>10} {t['cohens_h']:>+7.3f} {t['effect_size_label']:>12} {mark:>5}"
                )
                all_tests_in_group.append(t)
                if t["significant_p05"]:
                    sig_results.append(t)

        # Mann-Whitney tests
        if group["mann_whitney"]:
            lines.append("")
            lines.append(f"  {'Continuous':<28} {'Med 4.5':>10} {'Med 4.6':>10} {'U':>10} {'p-value':>10} {'p-adj':>10} {'d':>7} {'Effect':>12} {'Sig':>5}")
            lines.append(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*12} {'-'*5}")
            for t in group["mann_whitney"]:
                med_a = t["median"][MODELS[0]]
                med_b = t["median"][MODELS[1]]
                d_val = t["cohens_d"] if t["cohens_d"] is not None else float("nan")
                p_adj = t.get("p_adjusted")
                p_adj_str = f"{p_adj:.4f}" if p_adj is not None else "   n/a"
                fdr = "†" if t.get("fdr_significant") else ""
                sig = "*" if t["significant_p05"] else ""
                bonf = "**" if t.get("bonferroni_significant") else sig
                mark = bonf + fdr
                d_str = f"{d_val:>+7.3f}" if not math.isnan(d_val) else "    n/a"
                lines.append(
                    f"  {t['field']:<28} {med_a:>10.2f} {med_b:>10.2f} {t['u_statistic']:>10.0f} {t['p_value']:>10.4f} {p_adj_str:>10} {d_str} {t['effect_size_label']:>12} {mark:>5}"
                )
                all_tests_in_group.append(t)
                if t["significant_p05"]:
                    sig_results.append(t)

        # Chi-square tests
        if group["chi_square"]:
            lines.append("")
            lines.append(f"  {'Categorical':<28} {'chi2':>10} {'dof':>5} {'p-value':>10} {'p-adj':>10} {'V':>7} {'Effect':>12} {'Sig':>5} {'Warn':>5}")
            lines.append(f"  {'-'*28} {'-'*10} {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*12} {'-'*5} {'-'*5}")
            for t in group["chi_square"]:
                p_adj = t.get("p_adjusted")
                p_adj_str = f"{p_adj:.4f}" if p_adj is not None else "   n/a"
                fdr = "†" if t.get("fdr_significant") else ""
                sig = "*" if t["significant_p05"] else ""
                bonf = "**" if t.get("bonferroni_significant") else sig
                mark = bonf + fdr
                warn = "low" if t["low_expected_warning"] else ""
                lines.append(
                    f"  {t['field']:<28} {t['chi2']:>10.2f} {t['dof']:>5} {t['p_value']:>10.4f} {p_adj_str:>10} {t['cramers_v']:>7.3f} {t['effect_size_label']:>12} {mark:>5} {warn:>5}"
                )
                all_tests_in_group.append(t)
                if t["significant_p05"]:
                    sig_results.append(t)

        # Summary for this group
        lines.append("")
        n_sig = len(sig_results)
        n_bonf = sum(1 for t in all_tests_in_group if t.get("bonferroni_significant"))
        n_fdr = sum(1 for t in all_tests_in_group if t.get("fdr_significant"))
        lines.append(f"  Significant at p<0.05: {n_sig}/{len(all_tests_in_group)}")
        lines.append(f"  Significant after Bonferroni: {n_bonf}/{len(all_tests_in_group)}")
        lines.append(f"  Significant after FDR (BH): {n_fdr}/{len(all_tests_in_group)}")
        lines.append("")

    # Legend
    lines.append("=" * 100)
    lines.append("LEGEND")
    lines.append("  *  = significant at p < 0.05 (uncorrected)")
    lines.append(f"  ** = significant at p < {bonferroni:.6f} (Bonferroni-corrected for {total_tests} tests)")
    lines.append("  †  = significant after FDR correction (Benjamini-Hochberg, alpha=0.05)")
    lines.append("  Effect sizes: |d/h| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")
    lines.append("  Warn 'low': some expected cell frequencies < 5 (Chi-square may be unreliable)")
    lines.append("=" * 100)

    return "\n".join(lines)


def find_overlapping_projects(data_dir):
    """Find projects that appear in both models' canonical task data."""
    projects = {}
    for model in MODELS:
        tasks = load_canonical_tasks(data_dir, model)
        model_projects = set()
        for t in tasks:
            pp = t.get('project_path', '')
            if pp:
                model_projects.add(pp)
        projects[model] = model_projects

    overlapping = projects[MODELS[0]] & projects[MODELS[1]]
    return sorted(overlapping)


def filter_to_projects(records, project_paths):
    """Filter records to only those from specified projects."""
    project_set = set(project_paths)
    return [r for r in records if r.get('project_path', '') in project_set]


def run_test_suite(data, label_prefix=""):
    """Run full + per-complexity test suite on data dict. Returns (all_results, total_tests)."""
    prefix = f"{label_prefix}: " if label_prefix else ""

    overall = run_all_tests(data[MODELS[0]], data[MODELS[1]], label=f"{prefix}overall")

    complexity_results = []
    for cbin in COMPLEXITY_BINS:
        subset_a = [r for r in data[MODELS[0]] if r.get("complexity") == cbin]
        subset_b = [r for r in data[MODELS[1]] if r.get("complexity") == cbin]
        if len(subset_a) >= 3 and len(subset_b) >= 3:
            result = run_all_tests(subset_a, subset_b, label=f"{prefix}complexity: {cbin}")
            complexity_results.append(result)

    all_results = [overall] + complexity_results
    total_tests = count_tests(all_results)

    bonferroni_threshold = 0.05 / total_tests if total_tests > 0 else 0.05
    for group in all_results:
        for test_list in [group["chi_square"], group["mann_whitney"], group["proportions"]]:
            for t in test_list:
                t["bonferroni_significant"] = t["p_value"] < bonferroni_threshold

    return all_results, total_tests, bonferroni_threshold


def compare_sensitivity(full_results, restricted_results, full_n, restricted_n):
    """Compare full vs restricted results, highlighting divergences."""
    lines = []
    lines.append("=" * 90)
    lines.append("SENSITIVITY ANALYSIS: Full Dataset vs Overlapping Projects Only")
    lines.append("=" * 90)
    lines.append(f"Full dataset:          {full_n[MODELS[0]]:>5} vs {full_n[MODELS[1]]:>5} tasks")
    lines.append(f"Overlapping projects:  {restricted_n[MODELS[0]]:>5} vs {restricted_n[MODELS[1]]:>5} tasks")
    lines.append("")

    full_overall = full_results[0]
    restricted_overall = restricted_results[0]

    full_props = {t["field"]: t for t in full_overall.get("proportions", [])}
    rest_props = {t["field"]: t for t in restricted_overall.get("proportions", [])}

    all_fields = sorted(set(full_props.keys()) | set(rest_props.keys()))

    lines.append(f"  {'Metric':<28} {'Full Diff':>10} {'Full p':>10} {'Restr Diff':>10} {'Restr p':>10} {'Agreement':>10}")
    lines.append(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for field in all_fields:
        fp = full_props.get(field)
        rp = rest_props.get(field)
        if not fp or not rp:
            continue
        f_diff = fp[MODELS[0]]["proportion"] - fp[MODELS[1]]["proportion"]
        r_diff = rp[MODELS[0]]["proportion"] - rp[MODELS[1]]["proportion"]
        f_sig = fp["significant_p05"]
        r_sig = rp["significant_p05"]

        same_direction = (f_diff > 0) == (r_diff > 0) if f_diff != 0 and r_diff != 0 else True
        if f_sig == r_sig and same_direction:
            agreement = "agree"
        elif same_direction and not (f_sig and r_sig):
            agreement = "weak"
        else:
            agreement = "DIVERGE"

        f_sig_str = "*" if f_sig else ""
        r_sig_str = "*" if r_sig else ""
        lines.append(
            f"  {field:<28} {f_diff:>+9.1%}{f_sig_str} {fp['p_value']:>10.4f} {r_diff:>+9.1%}{r_sig_str} {rp['p_value']:>10.4f} {agreement:>10}"
        )

    lines.append("")
    lines.append("Legend: agree=same conclusion, weak=same direction but significance differs, DIVERGE=opposite")
    lines.append("=" * 90)
    return "\n".join(lines)


def main():
    global MODELS
    parser = argparse.ArgumentParser(description="Statistical significance tests for model comparison")
    parser.add_argument("--data-dir", default="data", help="Directory containing classified task JSONs")
    parser.add_argument("--analysis-dir", default="analysis", help="Directory containing LLM analysis JSONs")
    parser.add_argument("--output", default=None, help="Output JSON path (default: analysis-dir/stat-tests.json)")
    parser.add_argument("--config", default=None,
                        help="Path to analysis_config.json (default: scripts/analysis_config.json if it exists)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis: full dataset vs overlapping projects only")
    parser.add_argument("--overlapping-only", action="store_true",
                        help="Restrict analysis to overlapping projects only")
    args = parser.parse_args()

    output_path = args.output or str(Path(args.analysis_dir) / "stat-tests.json")

    MODELS = list(discover_model_pair(args.data_dir))

    # Load config if available
    config_path = args.config
    analysis_config = None
    if config_path is None:
        # Auto-detect
        default_config = Path(__file__).resolve().parent / "analysis_config.json"
        if default_config.exists():
            config_path = str(default_config)

    if config_path:
        print(f"Loading analysis config from {config_path}")
        analysis_config = load_analysis_config(config_path)
        fdr_alpha = analysis_config.get("correction", {}).get("alpha", 0.05)
    else:
        print("No analysis config found, using default field lists")
        fdr_alpha = 0.05

    print(f"Loading data from {args.data_dir}/ and {args.analysis_dir}/...")
    data = load_data(args.data_dir, args.analysis_dir)

    for model in MODELS:
        print(f"  {model}: {len(data[model])} tasks")

    # Filter to overlapping projects if requested
    if args.overlapping_only:
        overlapping = find_overlapping_projects(args.data_dir)
        print(f"\nRestricting to {len(overlapping)} overlapping projects:")
        for p in overlapping:
            print(f"  {p}")
        data = {model: filter_to_projects(data[model], overlapping) for model in MODELS}
        for model in MODELS:
            print(f"  {model}: {len(data[model])} tasks after filtering")

    # Standard mode with config-driven cross-cuts
    print("\nRunning tests across cross-cuts...")
    cross_cuts = CROSS_CUTS if analysis_config else None
    all_results = run_cross_cut_tests(data, cross_cuts)
    total_tests = count_tests(all_results)

    # Flatten, apply corrections
    flat_tests = flatten_tests(all_results)
    bonferroni_threshold = apply_bonferroni(flat_tests)
    apply_fdr_correction(flat_tests, fdr_alpha)

    fdr_significant_count = sum(1 for t in flat_tests if t.get("fdr_significant"))
    bonferroni_count = sum(1 for t in flat_tests if t.get("bonferroni_significant"))

    # Re-group for backward-compatible output
    overall, by_complexity, by_cross_cut = regroup_results(flat_tests)

    # Build output
    output = {
        "metadata": {
            "models": MODELS,
            "total_tests": total_tests,
            "correction_method": "benjamini-hochberg",
            "fdr_alpha": fdr_alpha,
            "fdr_significant_count": fdr_significant_count,
            "bonferroni_threshold": round(bonferroni_threshold, 8),
            "bonferroni_significant_count": bonferroni_count,
            "sample_sizes": {model: len(data[model]) for model in MODELS},
        },
        "overall": overall,
        "by_complexity": by_complexity,
    }

    # Add cross-cut results if any exist beyond complexity
    if by_cross_cut:
        output["by_cross_cut"] = by_cross_cut

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults written to {output_path}")

    # Print summary table
    summary = format_summary(all_results, total_tests, fdr_significant_count)
    print("\n" + summary)

    # Sensitivity side-output: compare full vs overlapping-projects-only
    if args.sensitivity:
        overlapping = find_overlapping_projects(args.data_dir)
        if overlapping:
            restricted_data = {
                model: filter_to_projects(data[model], overlapping) for model in MODELS
            }
            rest_n = {model: len(restricted_data[model]) for model in MODELS}
            if all(n >= 3 for n in rest_n.values()):
                print(f"\n--- Sensitivity: overlapping projects ({len(overlapping)}) ---")
                for model in MODELS:
                    print(f"  {model}: {rest_n[model]} tasks (from {len(data[model])})")

                rest_results, rest_total, rest_bonf = run_test_suite(
                    restricted_data, "restricted")

                sensitivity_output = {
                    "metadata": {
                        "models": MODELS,
                        "analysis_type": "sensitivity",
                        "overlapping_projects": overlapping,
                    },
                    "full": {
                        "metadata": {
                            "total_tests": total_tests,
                            "bonferroni_threshold": round(bonferroni_threshold, 8),
                            "sample_sizes": {model: len(data[model]) for model in MODELS},
                        },
                        "overall": overall,
                        "by_complexity": by_complexity,
                    },
                    "restricted": {
                        "metadata": {
                            "total_tests": rest_total,
                            "bonferroni_threshold": round(rest_bonf, 8),
                            "sample_sizes": rest_n,
                        },
                        "overall": rest_results[0],
                        "by_complexity": rest_results[1:],
                    },
                }
                sensitivity_path = str(Path(args.analysis_dir) / "sensitivity-analysis.json")
                with open(sensitivity_path, "w") as f:
                    json.dump(sensitivity_output, f, indent=2, cls=NumpyEncoder)

                full_n = {model: len(data[model]) for model in MODELS}
                comparison = compare_sensitivity(
                    [overall] + by_complexity, rest_results, full_n, rest_n)
                print("\n" + comparison)
                print(f"Sensitivity written to {sensitivity_path}")


if __name__ == "__main__":
    main()
