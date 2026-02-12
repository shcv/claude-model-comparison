#!/usr/bin/env python3
"""Generate a ranked findings registry from stat-tests.json.

Reads statistical test results, ranks by effect size, groups by theme,
and produces deterministic narrative text (no LLM).

Usage:
    python scripts/generate_findings.py --data-dir data --analysis-dir analysis
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_model_pair


def load_config(config_path):
    """Load analysis config for theme definitions."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    # Auto-detect
    default = Path(__file__).resolve().parent / "analysis_config.json"
    if default.exists():
        with open(default) as f:
            return json.load(f)
    return {}


def extract_flat_tests(stat_tests):
    """Extract a flat list of all tests from the stat-tests.json structure."""
    flat = []
    models = stat_tests.get("metadata", {}).get("models", [])

    def process_group(group, cross_cut_label=None):
        label = cross_cut_label or group.get("label", "overall")
        for test_type_key in ["chi_square", "mann_whitney", "proportions"]:
            for t in group.get(test_type_key, []):
                t.setdefault("cross_cut", label)
                flat.append(t)

    # Overall
    overall = stat_tests.get("overall")
    if overall:
        process_group(overall, "overall")

    # By complexity
    for group in stat_tests.get("by_complexity", []):
        process_group(group)

    # By cross-cut (new structure)
    for cc_name, group in stat_tests.get("by_cross_cut", {}).items():
        process_group(group, cc_name)

    return flat, models


def get_effect_size(test):
    """Extract normalized effect size from a test result."""
    es = test.get("effect_size")
    if es is not None:
        return es
    # Fallback: compute from test-specific fields
    test_type = test.get("test", "")
    if test_type == "mann-whitney-u":
        d = test.get("cohens_d")
        return abs(d) if d is not None else 0
    elif test_type == "two-proportion-z":
        h = test.get("cohens_h")
        return abs(h) if h is not None else 0
    elif test_type == "chi-square":
        return test.get("cramers_v", 0)
    return 0


def get_direction(test, models):
    """Determine which model is higher/different."""
    test_type = test.get("test", "")
    if len(models) < 2:
        return None

    ma, mb = models[0], models[1]

    if test_type == "mann-whitney-u":
        med_a = test.get("median", {}).get(ma)
        med_b = test.get("median", {}).get(mb)
        if med_a is not None and med_b is not None:
            if med_b > med_a:
                return f"{mb} higher"
            elif med_a > med_b:
                return f"{ma} higher"
            return "equal"

    elif test_type == "two-proportion-z":
        p_a = test.get(ma, {}).get("proportion")
        p_b = test.get(mb, {}).get("proportion")
        if p_a is not None and p_b is not None:
            if p_b > p_a:
                return f"{mb} higher"
            elif p_a > p_b:
                return f"{ma} higher"
            return "equal"

    elif test_type == "chi-square":
        return "distributions differ"

    return None


def generate_narrative(test, models, direction):
    """Generate deterministic narrative text for a finding."""
    test_type = test.get("test", "")
    field = test.get("field", "")
    field_display = field.replace("_", " ")

    if len(models) < 2:
        return f"{field_display}: significant difference found"

    ma, mb = models[0], models[1]
    ma_display = ma.replace("-", " ").title()
    mb_display = mb.replace("-", " ").title()

    if test_type == "mann-whitney-u":
        d = test.get("cohens_d")
        med_a = test.get("median", {}).get(ma)
        med_b = test.get("median", {}).get(mb)
        d_str = f"d={abs(d):.2f}" if d is not None else ""

        if med_a is not None and med_b is not None:
            if med_b > med_a:
                return f"{mb_display} has higher {field_display} (median {med_b:.1f} vs {med_a:.1f}, {d_str})"
            elif med_a > med_b:
                return f"{ma_display} has higher {field_display} (median {med_a:.1f} vs {med_b:.1f}, {d_str})"
        return f"No significant difference in {field_display}"

    elif test_type == "two-proportion-z":
        h = test.get("cohens_h")
        p_a = test.get(ma, {}).get("proportion")
        p_b = test.get(mb, {}).get("proportion")
        h_str = f"h={abs(h):.2f}" if h is not None else ""

        if p_a is not None and p_b is not None:
            if p_b > p_a:
                return f"{field_display}: {mb_display} {p_b:.1%} vs {ma_display} {p_a:.1%} ({h_str})"
            elif p_a > p_b:
                return f"{field_display}: {ma_display} {p_a:.1%} vs {mb_display} {p_b:.1%} ({h_str})"
        return f"No significant difference in {field_display}"

    elif test_type == "chi-square":
        v = test.get("cramers_v", 0)
        label = test.get("effect_size_label", "")
        return f"{field_display} distributions differ (V={v:.2f}, {label})"

    return f"{field_display}: significant difference found"


def build_findings(flat_tests, models, config):
    """Build the findings registry from flat test results."""
    themes = config.get("themes", {})

    # Build theme map
    theme_map = {}
    for theme_name, fields in themes.items():
        for f in fields:
            theme_map[f] = theme_name

    findings = []
    for t in flat_tests:
        field = t.get("field", "")
        es = get_effect_size(t)
        direction = get_direction(t, models)
        narrative = generate_narrative(t, models, direction)
        theme = t.get("theme") or theme_map.get(field)

        finding = {
            "field": field,
            "test_type": t.get("test", ""),
            "cross_cut": t.get("cross_cut", "overall"),
            "theme": theme,
            "p_value": t.get("p_value"),
            "p_adjusted": t.get("p_adjusted"),
            "fdr_significant": t.get("fdr_significant", False),
            "bonferroni_significant": t.get("bonferroni_significant", False),
            "effect_size": round(es, 4) if es is not None else None,
            "effect_label": t.get("effect_size_label", "n/a"),
            "direction": direction,
            "narrative": narrative,
        }

        # Add model-specific values
        if t.get("test") == "mann-whitney-u":
            finding["values"] = {
                m: {"median": t.get("median", {}).get(m),
                    "mean": t.get("mean", {}).get(m)}
                for m in models
            }
        elif t.get("test") == "two-proportion-z":
            finding["values"] = {
                m: {"proportion": t.get(m, {}).get("proportion"),
                    "count": t.get(m, {}).get("count"),
                    "n": t.get(m, {}).get("n")}
                for m in models
            }
        elif t.get("test") == "chi-square":
            finding["values"] = {
                "cramers_v": t.get("cramers_v"),
                "counts": t.get("counts"),
            }

        findings.append(finding)

    # Sort by FDR significance (significant first), then effect size descending
    findings.sort(key=lambda f: (
        not f.get("fdr_significant", False),
        -(f.get("effect_size") or 0),
    ))

    # Assign ranks
    for i, f in enumerate(findings, 1):
        f["rank"] = i

    return findings


def group_by_theme(findings):
    """Group findings by theme."""
    by_theme = {}
    for f in findings:
        theme = f.get("theme") or "uncategorized"
        if theme not in by_theme:
            by_theme[theme] = {"significant": [], "non_significant": [], "strongest_effect": 0}

        if f.get("fdr_significant"):
            by_theme[theme]["significant"].append(f)
        else:
            by_theme[theme]["non_significant"].append(f)

        es = f.get("effect_size") or 0
        if es > by_theme[theme]["strongest_effect"]:
            by_theme[theme]["strongest_effect"] = round(es, 4)

    return by_theme


def group_by_cross_cut(findings):
    """Group findings by cross-cut."""
    by_cc = {}
    for f in findings:
        cc = f.get("cross_cut", "overall")
        if cc not in by_cc:
            by_cc[cc] = {"significant": [], "non_significant": []}

        if f.get("fdr_significant"):
            by_cc[cc]["significant"].append(f)
        else:
            by_cc[cc]["non_significant"].append(f)

    return by_cc


def main():
    parser = argparse.ArgumentParser(
        description="Generate ranked findings registry from stat tests")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to data directory")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="Path to analysis directory")
    parser.add_argument("--config", default=None,
                        help="Path to analysis_config.json")
    parser.add_argument("--output", default=None,
                        help="Output path (default: analysis-dir/findings.json)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    analysis_dir = (args.analysis_dir or data_dir.parent / "analysis").resolve()
    output_path = args.output or str(analysis_dir / "findings.json")

    # Load stat tests
    stat_tests_path = analysis_dir / "stat-tests.json"
    if not stat_tests_path.exists():
        print(f"Error: {stat_tests_path} not found. Run stat_tests.py first.",
              file=sys.stderr)
        sys.exit(1)

    with open(stat_tests_path) as f:
        stat_tests = json.load(f)

    # Load config
    config = load_config(args.config)

    # Extract flat tests
    flat_tests, models = extract_flat_tests(stat_tests)
    if not models:
        models = list(discover_model_pair(data_dir))

    print(f"Processing {len(flat_tests)} test results...")

    # Build findings
    findings = build_findings(flat_tests, models, config)

    fdr_count = sum(1 for f in findings if f.get("fdr_significant"))
    bonf_count = sum(1 for f in findings if f.get("bonferroni_significant"))
    print(f"  {len(findings)} findings total")
    print(f"  {fdr_count} FDR-significant")
    print(f"  {bonf_count} Bonferroni-significant")

    # Group by theme and cross-cut
    by_theme = group_by_theme(findings)
    by_cross_cut = group_by_cross_cut(findings)

    # Theme summary
    print(f"\nFindings by theme:")
    for theme in sorted(by_theme.keys()):
        info = by_theme[theme]
        n_sig = len(info["significant"])
        n_ns = len(info["non_significant"])
        print(f"  {theme}: {n_sig} significant, {n_ns} non-significant "
              f"(strongest effect: {info['strongest_effect']:.2f})")

    # Build output
    output = {
        "metadata": {
            "total_tests": len(findings),
            "fdr_significant": fdr_count,
            "bonferroni_significant": bonf_count,
            "correction": stat_tests.get("metadata", {}).get("correction_method", "benjamini-hochberg"),
            "alpha": config.get("correction", {}).get("alpha", 0.05),
            "models": models,
        },
        "findings": findings,
        "by_theme": by_theme,
        "by_cross_cut": by_cross_cut,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nFindings written to {output_path}")


if __name__ == "__main__":
    main()
