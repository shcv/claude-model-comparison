#!/usr/bin/env python3
"""Statistical significance testing for model comparison data.

Runs Chi-square tests, Mann-Whitney U tests, calculates confidence intervals
(Wilson score) and effect sizes (Cohen's h, Cohen's d) for comparing two models.
Tests are run both overall and per-complexity-bin.

Usage:
    python scripts/stat_tests.py --data-dir data --analysis-dir analysis
    python scripts/stat_tests.py --data-dir data --analysis-dir analysis --output analysis/stat-tests.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

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

# Categorical fields from LLM analysis
CATEGORICAL_FIELDS = {
    "task_completion": None,       # all observed values
    "normalized_user_sentiment": None,
    "scope_management": None,
    "iteration_required": None,
    "error_recovery": None,
    "communication_quality": None,
    "normalized_execution_quality": None,
    "autonomy_level": None,
}

# Binary proportion fields (derived from categorical)
PROPORTION_FIELDS = {
    "satisfaction_rate": lambda rec: rec.get("normalized_user_sentiment") == "satisfied",
    "dissatisfaction_rate": lambda rec: rec.get("normalized_user_sentiment") == "dissatisfied",
    "complete_rate": lambda rec: rec.get("task_completion") == "complete",
    "failed_rate": lambda rec: rec.get("task_completion") == "failed",
    "scope_expanded_rate": lambda rec: rec.get("scope_management") == "expanded",
    "one_shot_rate": lambda rec: rec.get("iteration_required") == "one_shot",
    "good_execution_rate": lambda rec: rec.get("normalized_execution_quality") == "good",
}

# Continuous fields from LLM analysis
CONTINUOUS_FIELDS = [
    "alignment_score",
    "duration_seconds",
    "tool_calls",
    "files_touched",
    "lines_added",
    "lines_removed",
    "lines_per_minute",
    "tools_per_file",
]


def load_data(data_dir, analysis_dir):
    """Load task data for both models.

    Prefers tasks-annotated-{model}.json (output of annotate_tasks.py) which
    already contains merged LLM analysis + classification + canonical signals.
    Falls back to merging llm-analysis + tasks-classified if annotated files
    don't exist.

    Enriches records with project_path from canonical tasks for sensitivity analysis.
    """
    data = {}
    for model in MODELS:
        annotated_path = Path(data_dir) / f"tasks-annotated-{model}.json"

        if annotated_path.exists():
            print(f"  Loading annotated tasks: {annotated_path}")
            with open(annotated_path) as f:
                records = json.load(f)
            # Enrich with project_path from canonical tasks
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

        # Build lookup from classified tasks by task_id
        classified_by_id = {t["task_id"]: t for t in classified}

        # Enrich with complexity from classified and project_path from canonical
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
    """Run Chi-square test on a categorical field between two groups.

    Returns dict with test results or None if insufficient data.
    """
    counts_a = Counter(r.get(field) for r in data_a if r.get(field) is not None)
    counts_b = Counter(r.get(field) for r in data_b if r.get(field) is not None)

    all_categories = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    if len(all_categories) < 2:
        return None

    observed = np.array([
        [counts_a.get(c, 0) for c in all_categories],
        [counts_b.get(c, 0) for c in all_categories],
    ])

    # Check minimum expected frequency
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    if total == 0:
        return None

    expected = np.outer(row_totals, col_totals) / total
    if np.any(expected < 5):
        low_expected = True
    else:
        low_expected = False

    try:
        chi2, p_value, dof, _ = stats.chi2_contingency(observed)
    except ValueError:
        return None

    # Cramers V as effect size
    min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / (total * min_dim)) if total * min_dim > 0 else 0.0

    return {
        "test": "chi-square",
        "field": field,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
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

    # Two-proportion z-test
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

    # Chi-square tests
    for field in CATEGORICAL_FIELDS:
        result = chi_square_test(data_a, data_b, field)
        if result:
            results["chi_square"].append(result)

    # Mann-Whitney U tests
    for field in CONTINUOUS_FIELDS:
        result = mann_whitney_test(data_a, data_b, field)
        if result:
            results["mann_whitney"].append(result)

    # Proportion tests
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


def format_summary(all_results, total_tests):
    """Format a human-readable summary table."""
    lines = []
    lines.append("=" * 90)
    lines.append("STATISTICAL SIGNIFICANCE TESTING SUMMARY")
    lines.append("=" * 90)
    lines.append(f"Models compared: {MODELS[0]} vs {MODELS[1]}")
    lines.append(f"Total tests run: {total_tests}")
    bonferroni = 0.05 / total_tests if total_tests > 0 else 0.05
    lines.append(f"Bonferroni-corrected threshold: p < {bonferroni:.6f} (0.05 / {total_tests})")
    lines.append("")

    for group in all_results:
        label = group["label"].upper()
        lines.append("-" * 90)
        lines.append(f"  {label}")
        lines.append("-" * 90)

        sig_results = []
        all_tests_in_group = []

        # Proportion tests
        if group["proportions"]:
            lines.append("")
            lines.append(f"  {'Proportion':<28} {'Opus4.5':>10} {'Opus4.6':>10} {'Diff':>8} {'p-value':>10} {'h':>7} {'Effect':>12} {'Sig':>5}")
            lines.append(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*7} {'-'*12} {'-'*5}")
            for t in group["proportions"]:
                p_a = t[MODELS[0]]["proportion"]
                p_b = t[MODELS[1]]["proportion"]
                diff = p_a - p_b
                sig = "*" if t["significant_p05"] else ""
                bonf = "**" if t["p_value"] < bonferroni else sig
                lines.append(
                    f"  {t['field']:<28} {p_a:>9.1%} {p_b:>9.1%} {diff:>+7.1%} {t['p_value']:>10.4f} {t['cohens_h']:>+7.3f} {t['effect_size_label']:>12} {bonf:>5}"
                )
                all_tests_in_group.append(t)
                if t["significant_p05"]:
                    sig_results.append(t)

        # Mann-Whitney tests
        if group["mann_whitney"]:
            lines.append("")
            lines.append(f"  {'Continuous':<28} {'Med 4.5':>10} {'Med 4.6':>10} {'U':>10} {'p-value':>10} {'d':>7} {'Effect':>12} {'Sig':>5}")
            lines.append(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*12} {'-'*5}")
            for t in group["mann_whitney"]:
                med_a = t["median"][MODELS[0]]
                med_b = t["median"][MODELS[1]]
                d_val = t["cohens_d"] if t["cohens_d"] is not None else float("nan")
                sig = "*" if t["significant_p05"] else ""
                bonf = "**" if t["p_value"] < bonferroni else sig
                d_str = f"{d_val:>+7.3f}" if not math.isnan(d_val) else "    n/a"
                lines.append(
                    f"  {t['field']:<28} {med_a:>10.2f} {med_b:>10.2f} {t['u_statistic']:>10.0f} {t['p_value']:>10.4f} {d_str} {t['effect_size_label']:>12} {bonf:>5}"
                )
                all_tests_in_group.append(t)
                if t["significant_p05"]:
                    sig_results.append(t)

        # Chi-square tests
        if group["chi_square"]:
            lines.append("")
            lines.append(f"  {'Categorical':<28} {'chi2':>10} {'dof':>5} {'p-value':>10} {'V':>7} {'Effect':>12} {'Sig':>5} {'Warn':>5}")
            lines.append(f"  {'-'*28} {'-'*10} {'-'*5} {'-'*10} {'-'*7} {'-'*12} {'-'*5} {'-'*5}")
            for t in group["chi_square"]:
                sig = "*" if t["significant_p05"] else ""
                bonf = "**" if t["p_value"] < bonferroni else sig
                warn = "low" if t["low_expected_warning"] else ""
                lines.append(
                    f"  {t['field']:<28} {t['chi2']:>10.2f} {t['dof']:>5} {t['p_value']:>10.4f} {t['cramers_v']:>7.3f} {t['effect_size_label']:>12} {bonf:>5} {warn:>5}"
                )
                all_tests_in_group.append(t)
                if t["significant_p05"]:
                    sig_results.append(t)

        # Summary for this group
        lines.append("")
        n_sig = len(sig_results)
        n_bonf = sum(1 for t in all_tests_in_group if t["p_value"] < bonferroni)
        lines.append(f"  Significant at p<0.05: {n_sig}/{len(all_tests_in_group)}")
        lines.append(f"  Significant after Bonferroni: {n_bonf}/{len(all_tests_in_group)}")
        lines.append("")

    # Legend
    lines.append("=" * 90)
    lines.append("LEGEND")
    lines.append("  *  = significant at p < 0.05 (uncorrected)")
    lines.append(f"  ** = significant at p < {bonferroni:.6f} (Bonferroni-corrected for {total_tests} tests)")
    lines.append("  Effect sizes: |d/h| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large")
    lines.append("  Warn 'low': some expected cell frequencies < 5 (Chi-square may be unreliable)")
    lines.append("=" * 90)

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

    # Compare overall proportion tests (most interpretable)
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

        # Agreement: both sig same direction, both non-sig, or divergent
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
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis: full dataset vs overlapping projects only")
    parser.add_argument("--overlapping-only", action="store_true",
                        help="Restrict analysis to overlapping projects only")
    args = parser.parse_args()

    output_path = args.output or str(Path(args.analysis_dir) / "stat-tests.json")

    MODELS = list(discover_model_pair(args.data_dir))

    print(f"Loading data from {args.data_dir}/ and {args.analysis_dir}/...")
    data = load_data(args.data_dir, args.analysis_dir)

    for model in MODELS:
        print(f"  {model}: {len(data[model])} tasks")

    # Sensitivity mode: run both full and restricted, then compare
    if args.sensitivity:
        overlapping = find_overlapping_projects(args.data_dir)
        print(f"\nOverlapping projects ({len(overlapping)}):")
        for p in overlapping:
            print(f"  {p}")

        restricted_data = {
            model: filter_to_projects(data[model], overlapping) for model in MODELS
        }
        print(f"\nRestricted dataset:")
        for model in MODELS:
            print(f"  {model}: {len(restricted_data[model])} tasks (from {len(data[model])})")

        print("\n--- Full dataset tests ---")
        full_results, full_total, full_bonf = run_test_suite(data, "full")
        print(f"  {full_total} tests run")

        print("\n--- Overlapping projects tests ---")
        rest_results, rest_total, rest_bonf = run_test_suite(restricted_data, "restricted")
        print(f"  {rest_total} tests run")

        # Build combined output
        output = {
            "metadata": {
                "models": MODELS,
                "analysis_type": "sensitivity",
                "overlapping_projects": overlapping,
            },
            "full": {
                "metadata": {
                    "total_tests": full_total,
                    "bonferroni_threshold": round(full_bonf, 8),
                    "sample_sizes": {model: len(data[model]) for model in MODELS},
                },
                "overall": full_results[0],
                "by_complexity": full_results[1:],
            },
            "restricted": {
                "metadata": {
                    "total_tests": rest_total,
                    "bonferroni_threshold": round(rest_bonf, 8),
                    "sample_sizes": {model: len(restricted_data[model]) for model in MODELS},
                },
                "overall": rest_results[0],
                "by_complexity": rest_results[1:],
            },
        }

        sensitivity_path = str(Path(args.analysis_dir) / "sensitivity-analysis.json")
        with open(sensitivity_path, "w") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f"\nSensitivity results written to {sensitivity_path}")

        full_n = {model: len(data[model]) for model in MODELS}
        rest_n = {model: len(restricted_data[model]) for model in MODELS}
        comparison = compare_sensitivity(full_results, rest_results, full_n, rest_n)
        print("\n" + comparison)

        # Also write the full-dataset results to the standard output path
        standard_output = {
            "metadata": {
                "models": MODELS,
                "total_tests": full_total,
                "bonferroni_threshold": round(full_bonf, 8),
                "sample_sizes": {model: len(data[model]) for model in MODELS},
            },
            "overall": full_results[0],
            "by_complexity": full_results[1:],
        }
        with open(output_path, "w") as f:
            json.dump(standard_output, f, indent=2, cls=NumpyEncoder)
        print(f"Standard results also written to {output_path}")
        return

    # Filter to overlapping projects if requested
    if args.overlapping_only:
        overlapping = find_overlapping_projects(args.data_dir)
        print(f"\nRestricting to {len(overlapping)} overlapping projects:")
        for p in overlapping:
            print(f"  {p}")
        data = {model: filter_to_projects(data[model], overlapping) for model in MODELS}
        for model in MODELS:
            print(f"  {model}: {len(data[model])} tasks after filtering")

    # Standard mode
    print("\nRunning overall tests...")
    overall = run_all_tests(data[MODELS[0]], data[MODELS[1]], label="overall")

    complexity_results = []
    for cbin in COMPLEXITY_BINS:
        subset_a = [r for r in data[MODELS[0]] if r.get("complexity") == cbin]
        subset_b = [r for r in data[MODELS[1]] if r.get("complexity") == cbin]
        if len(subset_a) >= 3 and len(subset_b) >= 3:
            print(f"Running tests for complexity={cbin} ({len(subset_a)} vs {len(subset_b)})...")
            result = run_all_tests(subset_a, subset_b, label=f"complexity: {cbin}")
            complexity_results.append(result)
        else:
            print(f"Skipping complexity={cbin} (insufficient data: {len(subset_a)} vs {len(subset_b)})")

    all_results = [overall] + complexity_results
    total_tests = count_tests(all_results)

    # Add Bonferroni info to each test result
    bonferroni_threshold = 0.05 / total_tests if total_tests > 0 else 0.05
    for group in all_results:
        for test_list in [group["chi_square"], group["mann_whitney"], group["proportions"]]:
            for t in test_list:
                t["bonferroni_significant"] = t["p_value"] < bonferroni_threshold

    # Build output
    output = {
        "metadata": {
            "models": MODELS,
            "total_tests": total_tests,
            "bonferroni_threshold": round(bonferroni_threshold, 8),
            "sample_sizes": {model: len(data[model]) for model in MODELS},
        },
        "overall": overall,
        "by_complexity": complexity_results,
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults written to {output_path}")

    # Print summary table
    summary = format_summary(all_results, total_tests)
    print("\n" + summary)


if __name__ == "__main__":
    main()
