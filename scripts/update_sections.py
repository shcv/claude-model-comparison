#!/usr/bin/env python3
"""Update report sections from data: auto-generate expansion tables and
LLM-check prose against current analysis data.

Driven by per-section spec files in report/specs/.

Usage:
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --sections cost,edit-accuracy
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --tables-only
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --prose-only
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --dry-run
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --model opus
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

# Add scripts dir to path for table_gen import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_gen


def parse_comparison_dir(dir_path: Path) -> dict:
    """Derive model identifiers from comparison directory name.

    E.g., "opus-4.5-vs-4.6" -> model_a="opus-4-5", model_b="opus-4-6",
    display_a="4.5", display_b="4.6".
    """
    name = dir_path.name
    m = re.match(r'(\w+)-([\d.]+)-vs-([\d.]+)', name)
    if not m:
        print(f"Error: cannot parse model names from directory '{name}'", file=sys.stderr)
        print("Expected format: <family>-<version>-vs-<version>", file=sys.stderr)
        sys.exit(1)
    family = m.group(1)
    ver_a = m.group(2)
    ver_b = m.group(3)
    return {
        "model_a": f"{family}-{ver_a.replace('.', '-')}",
        "model_b": f"{family}-{ver_b.replace('.', '-')}",
        "display_a": ver_a,
        "display_b": ver_b,
        "family": family,
    }


def load_spec(spec_path: Path) -> dict:
    """Load a section spec file."""
    with open(spec_path) as f:
        return json.load(f)


def load_data_sources(spec: dict, comparison_dir: Path) -> dict:
    """Load all data sources referenced by a spec.

    Returns a flat dict keyed by source name, e.g. {"tokens": {...}, "stats": {...}}.
    """
    data = {}
    for name, rel_path in spec.get("data_sources", {}).items():
        full_path = comparison_dir / rel_path
        if not full_path.exists():
            print(f"  Warning: data source {full_path} not found", file=sys.stderr)
            data[name] = {}
            continue
        with open(full_path) as f:
            data[name] = json.load(f)
    return data


def resolve_key_metrics(metrics: list, data: dict, config: dict) -> dict:
    """Resolve a list of key_metrics paths into a summary dict."""
    result = {}
    for path in metrics:
        val = table_gen.resolve_path(data, path, config)
        expanded = path.format(**config)
        result[expanded] = val
    return result


def content_hash(content: str) -> str:
    """Short hash for caching."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Custom row generators for complex tables ──────────────────────


def generate_edit_overview(spec: dict, data: dict, config: dict) -> str:
    """Generate the edit-accuracy-overview expansion table."""
    title = spec.get("title", "Full edit overlap breakdown")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    ea = data.get("edits", {}).get(ma, {})
    eb = data.get("edits", {}).get(mb, {})

    def fmt_n(n):
        return f"{n:,}"

    def fmt_pct(n, total):
        if total == 0:
            return "0%"
        return f"{n / total * 100:.1f}%"

    total_edits_a = ea.get("total_edits", 0)
    total_edits_b = eb.get("total_edits", 0)
    total_writes_a = ea.get("total_writes", 0)
    total_writes_b = eb.get("total_writes", 0)
    overlaps_a = ea.get("total_overlaps", 0)
    overlaps_b = eb.get("total_overlaps", 0)
    by_tier_a = ea.get("by_tier", {})
    by_tier_b = eb.get("by_tier", {})
    by_class_a = ea.get("by_classification", {})
    by_class_b = eb.get("by_classification", {})

    # Table 1: overlap breakdown
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append(f'        <tr><th>Metric</th><th class="right">Opus {da}</th><th class="right">Opus {db}</th></tr>')
    lines.append("    </thead>")
    lines.append("    <tbody>")

    lines.append(f'        <tr><td class="label-cell">Total edits + writes</td>'
                 f'<td class="right mono">{fmt_n(total_edits_a)} + {fmt_n(total_writes_a)}</td>'
                 f'<td class="right mono">{fmt_n(total_edits_b)} + {fmt_n(total_writes_b)}</td></tr>')

    lines.append(f'        <tr><td class="label-cell">Overlapping edits</td>'
                 f'<td class="right mono">{fmt_n(overlaps_a)} ({fmt_pct(overlaps_a, total_edits_a)})</td>'
                 f'<td class="right mono">{fmt_n(overlaps_b)} ({fmt_pct(overlaps_b, total_edits_b)})</td></tr>')

    for tier_key, tier_label in [("exact", "Exact match"), ("containment", "Containment"), ("line_overlap", "Line overlap")]:
        ta = by_tier_a.get(tier_key, 0)
        tb = by_tier_b.get(tier_key, 0)
        lines.append(f'        <tr><td class="label-cell">&mdash; {tier_label}</td>'
                     f'<td class="right mono">{fmt_n(ta)} ({fmt_pct(ta, overlaps_a)})</td>'
                     f'<td class="right mono">{fmt_n(tb)} ({fmt_pct(tb, overlaps_b)})</td></tr>')

    lines.append("    </tbody>")
    lines.append("</table>")

    # Table 2: classification
    lines.append("<p>Classification of what drove the rework:</p>")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append(f'        <tr><th>Cause</th><th class="right">Opus {da}</th><th class="right">Opus {db}</th></tr>')
    lines.append("    </thead>")
    lines.append("    <tbody>")

    for cls_key, cls_label in [("self_correction", "Self-correction"), ("error_recovery", "Error recovery"),
                                ("user_directed", "User-directed"), ("iterative_refinement", "Iterative refinement")]:
        ca = by_class_a.get(cls_key, 0)
        cb = by_class_b.get(cls_key, 0)
        lines.append(f'        <tr><td class="label-cell">{cls_label}</td>'
                     f'<td class="right mono">{fmt_n(ca)} ({fmt_pct(ca, overlaps_a)})</td>'
                     f'<td class="right mono">{fmt_n(cb)} ({fmt_pct(cb, overlaps_b)})</td></tr>')

    lines.append("    </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def generate_iterative_refinement(spec: dict, data: dict, config: dict) -> str:
    """Generate the iterative-refinement-detail expansion."""
    title = spec.get("title", "Iterative refinement breakdown by complexity and duration")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    ea = data.get("edits", {}).get(ma, {})
    eb = data.get("edits", {}).get(mb, {})
    bins_a = ea.get("complexity_bins", {})
    bins_b = eb.get("complexity_bins", {})
    by_cx_a = bins_a.get("by_complexity", {})
    by_cx_b = bins_b.get("by_complexity", {})
    by_dur_a = bins_a.get("by_duration_tercile", {})
    by_dur_b = bins_b.get("by_duration_tercile", {})

    # By complexity
    lines.append("<h3>By Complexity</h3>")
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append('        <tr><th>Complexity</th><th>Iterative Refinement Rate</th>'
                 f'<th class="right">{config["display_a"]} (n)</th>'
                 f'<th class="right">{config["display_b"]} (n)</th></tr>')
    lines.append("    </thead>")
    lines.append("    <tbody>")

    for cx in ["trivial", "simple", "moderate", "complex+"]:
        ca = by_cx_a.get(cx, {})
        cb = by_cx_b.get(cx, {})
        rate_a = ca.get("iterative_refinement_rate", 0) * 100
        rate_b = cb.get("iterative_refinement_rate", 0) * 100
        n_a = ca.get("n", 0)
        n_b = cb.get("n", 0)
        bar = table_gen.generate_bar_pair(rate_a, rate_b, scale=50)
        cx_label = table_gen._label_case(cx)
        lines.append(f'        <tr><td class="label-cell">{cx_label}</td>')
        lines.append(f'            <td class="bar-cell">{bar}</td>')
        lines.append(f'            <td class="right mono">{n_a}</td>'
                     f'<td class="right mono">{n_b}</td></tr>')

    lines.append("    </tbody>")
    lines.append("</table>")

    # By duration
    lines.append("")
    lines.append("<h3>By Duration</h3>")
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append('        <tr><th>Duration</th><th>Iterative Refinement Rate</th>'
                 f'<th class="right">{config["display_a"]}</th>'
                 f'<th class="right">{config["display_b"]}</th></tr>')
    lines.append("    </thead>")
    lines.append("    <tbody>")

    for dur in ["short", "medium", "long"]:
        da_bin = by_dur_a.get(dur, {})
        db_bin = by_dur_b.get(dur, {})
        rate_a = da_bin.get("iterative_refinement_rate", 0) * 100
        rate_b = db_bin.get("iterative_refinement_rate", 0) * 100
        n_a = da_bin.get("n", 0)
        n_b = db_bin.get("n", 0)
        # Range labels
        range_a = da_bin.get("range", "")
        range_b = db_bin.get("range", "")
        if range_a and range_b:
            dur_label = f"{dur.title()} ({range_a} / {range_b})"
        elif range_a:
            dur_label = f"{dur.title()} ({range_a})"
        else:
            dur_label = dur.title()

        bar = table_gen.generate_bar_pair(rate_a, rate_b, scale=50)
        lines.append(f'        <tr><td class="label-cell">{dur_label}</td>')
        lines.append(f'            <td class="bar-cell">{bar}</td>')
        lines.append(f'            <td class="right mono">{n_a} tasks</td>'
                     f'<td class="right mono">{n_b} tasks</td></tr>')

    lines.append("    </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def generate_triage_top_tasks(spec: dict, data: dict, config: dict) -> str:
    """Generate the edit-triage-top-tasks expansion."""
    title = spec.get("title", "Highest-rework tasks by triage score")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    ea = data.get("edits", {}).get(ma, {})
    eb = data.get("edits", {}).get(mb, {})

    for model_data, label in [(ea, f"Opus {da}"), (eb, f"Opus {db}")]:
        triage = model_data.get("triage_top_10", [])
        lines.append(f"<h3>{label}</h3>")
        lines.append("<table>")
        lines.append("<thead>")
        lines.append("<tr>")
        lines.append('<th class="label-cell">Rank</th>')
        lines.append('<th class="right">Triage Score</th>')
        lines.append('<th class="right">User Corrections</th>')
        lines.append('<th class="right">Error Recoveries</th>')
        lines.append('<th class="right">Self-Corrections</th>')
        lines.append('<th class="right">Chain Depth</th>')
        lines.append("</tr>")
        lines.append("</thead>")
        lines.append("<tbody>")

        for item in triage:
            task_id = item.get("task_id", "")
            short_id = task_id[:8] if len(task_id) >= 8 else task_id
            score = item.get("score", 0)
            user_corr = item.get("user_corrections", 0)
            error_rec = item.get("error_recoveries", 0)
            self_corr = item.get("self_corrections", 0)
            chain = item.get("chain_depth", 0)

            lines.append("<tr>")
            lines.append(f'<td class="label-cell"><span class="mono">{short_id}</span></td>')
            lines.append(f'<td class="right">{score:.1f}</td>')
            lines.append(f'<td class="right">{user_corr}</td>')
            lines.append(f'<td class="right">{error_rec}</td>')
            lines.append(f'<td class="right">{self_corr}</td>')
            lines.append(f'<td class="right">{chain}</td>')
            lines.append("</tr>")

        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

    return "\n".join(lines)


def generate_satisfaction_stats(spec: dict, data: dict, config: dict) -> str:
    """Generate the satisfaction-stat-tests expansion.

    This is a complex multi-section expansion with several sub-tables.
    For this row_gen type, we produce the full HTML directly.
    """
    title = spec.get("title", "Full statistical test details for satisfaction metrics")
    lines = [f"<!-- title: {title} -->"]

    stats = data.get("stats", {})
    overall = stats.get("overall", {})
    metadata = stats.get("metadata", {})
    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]

    # Mann-Whitney: alignment score
    mw_tests = overall.get("mann_whitney", [])
    alignment_test = next((t for t in mw_tests if t["field"] == "alignment_score"), None)
    if alignment_test:
        lines.append("<h3>Mann-Whitney U Test: Alignment Score</h3>")
        n_a = alignment_test["n"].get(ma, 0) if isinstance(alignment_test.get("n"), dict) else metadata.get("sample_sizes", {}).get(ma, 0)
        n_b = alignment_test["n"].get(mb, 0) if isinstance(alignment_test.get("n"), dict) else metadata.get("sample_sizes", {}).get(mb, 0)
        mean_a = alignment_test.get("mean", {}).get(ma, 0)
        mean_b = alignment_test.get("mean", {}).get(mb, 0)
        median_a = alignment_test.get("median", {}).get(ma, 0)
        median_b = alignment_test.get("median", {}).get(mb, 0)
        std_a = alignment_test.get("std", {}).get(ma, 0)
        std_b = alignment_test.get("std", {}).get(mb, 0)
        better = "v-green" if mean_b > mean_a else ("v-green" if mean_a > mean_b else "")

        lines.append("<table>")
        lines.append("<thead>")
        lines.append(f'<tr><th class="label-cell">Metric</th><th class="right">Opus {da}</th><th class="right">Opus {db}</th></tr>')
        lines.append("</thead>")
        lines.append("<tbody>")
        lines.append(f'<tr><td class="label-cell">Sample size</td><td class="right mono">{n_a}</td><td class="right mono">{n_b}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Mean</td><td class="right mono">{mean_a:.3f}</td><td class="right mono {better}">{mean_b:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Median</td><td class="right mono">{median_a:.1f}</td><td class="right mono">{median_b:.1f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Std dev</td><td class="right mono">{std_a:.3f}</td><td class="right mono">{std_b:.3f}</td></tr>')
        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

        # Test statistics
        lines.append("<table>")
        lines.append("<thead>")
        lines.append(f'<tr><th class="label-cell">Test statistic</th><th class="right">Value</th></tr>')
        lines.append("</thead>")
        lines.append("<tbody>")
        lines.append(f'<tr><td class="label-cell">U statistic</td><td class="right mono">{alignment_test["u_statistic"]:.1f}</td></tr>')
        p_class = " v-green" if alignment_test.get("bonferroni_significant") else ""
        lines.append(f'<tr><td class="label-cell">p-value</td><td class="right mono{p_class}">{alignment_test["p_value"]:.6f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Cohen\'s d</td><td class="right mono">{alignment_test["cohens_d"]:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Effect size</td><td class="right">{alignment_test["effect_size_label"]}</td></tr>')
        lines.append("</tbody>")
        lines.append("</table>")

    # Proportion tests
    prop_tests = overall.get("proportions", [])
    lines.append("")
    lines.append("<h3>Proportion Tests: Task Outcomes</h3>")

    for field_name, field_label in [("complete_rate", "Complete Rate"), ("failed_rate", "Failed Rate"),
                                     ("satisfaction_rate", "Satisfaction Rate"), ("dissatisfaction_rate", "Dissatisfaction Rate")]:
        test = next((t for t in prop_tests if t["field"] == field_name), None)
        if not test:
            continue

        lines.append(f"<h4>{field_label}</h4>")

        prop_a_data = test.get(ma, {})
        prop_b_data = test.get(mb, {})

        prop_a = prop_a_data.get("proportion", 0)
        prop_b = prop_b_data.get("proportion", 0)
        count_a = prop_a_data.get("count", 0)
        total_a = prop_a_data.get("n", 0)
        count_b = prop_b_data.get("count", 0)
        total_b = prop_b_data.get("n", 0)
        ci_a = [prop_a_data.get("ci_lower", 0), prop_a_data.get("ci_upper", 0)]
        ci_b = [prop_b_data.get("ci_lower", 0), prop_b_data.get("ci_upper", 0)]

        # Determine which is "better" (higher complete, lower failed)
        if field_name in ("failed_rate", "dissatisfaction_rate"):
            better_class = "v-green" if prop_b < prop_a else ""
        else:
            better_class = "v-green" if prop_b > prop_a else ""

        lines.append("<table>")
        lines.append("<thead>")
        lines.append(f'<tr><th class="label-cell">Metric</th><th class="right">Opus {da}</th><th class="right">Opus {db}</th></tr>')
        lines.append("</thead>")
        lines.append("<tbody>")
        lines.append(f'<tr><td class="label-cell">Proportion</td><td class="right mono">{prop_a:.3f}</td>'
                     f'<td class="right mono {better_class}">{prop_b:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Count</td><td class="right mono">{count_a} / {total_a}</td>'
                     f'<td class="right mono">{count_b} / {total_b}</td></tr>')
        lines.append(f'<tr><td class="label-cell">95% CI</td><td class="right mono">[{ci_a[0]:.3f}, {ci_a[1]:.3f}]</td>'
                     f'<td class="right mono">[{ci_b[0]:.3f}, {ci_b[1]:.3f}]</td></tr>')
        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

        # Test stats
        lines.append("<table>")
        lines.append("<thead>")
        lines.append(f'<tr><th class="label-cell">Test statistic</th><th class="right">Value</th></tr>')
        lines.append("</thead>")
        lines.append("<tbody>")
        lines.append(f'<tr><td class="label-cell">z statistic</td><td class="right mono">{test["z_statistic"]:.3f}</td></tr>')
        p_class = " v-green" if test.get("bonferroni_significant") else ""
        lines.append(f'<tr><td class="label-cell">p-value</td><td class="right mono{p_class}">{test["p_value"]:.4f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Cohen\'s h</td><td class="right mono">{test["cohens_h"]:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Effect size</td><td class="right">{test["effect_size_label"]}</td></tr>')
        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

    # Chi-square: task_completion
    chi_tests = overall.get("chi_square", [])
    tc_test = next((t for t in chi_tests if t["field"] == "task_completion"), None)
    if tc_test:
        lines.append("<h3>Chi-Square Test: Task Completion Distribution</h3>")
        lines.append("")
        lines.append("<p><strong>Note:</strong> The full categorical breakdown includes "
                     f'{len(tc_test.get("categories", []))} unique completion statuses. '
                     "For clarity, simplified counts are shown below.</p>")
        lines.append("")

        # Summarize counts into major categories
        counts_a = tc_test.get("counts", {}).get(ma, {})
        counts_b = tc_test.get("counts", {}).get(mb, {})

        def bucket_counts(counts):
            complete = counts.get("complete", 0)
            partial = sum(v for k, v in counts.items() if k.startswith("partial"))
            interrupted = sum(v for k, v in counts.items() if k.startswith("interrupted"))
            failed = counts.get("failed", 0)
            other = sum(counts.values()) - complete - partial - interrupted - failed
            return {"Complete": complete, "Partial": partial, "Interrupted": interrupted,
                    "Failed": failed, "Other": other}

        buckets_a = bucket_counts(counts_a)
        buckets_b = bucket_counts(counts_b)

        lines.append("<table>")
        lines.append("<thead>")
        lines.append(f'<tr><th class="label-cell">Category</th><th class="right">Opus {da}</th><th class="right">Opus {db}</th></tr>')
        lines.append("</thead>")
        lines.append("<tbody>")
        for cat in ["Complete", "Partial", "Interrupted", "Failed", "Other"]:
            lines.append(f'<tr><td class="label-cell">{cat}</td>'
                         f'<td class="right mono">{buckets_a[cat]}</td>'
                         f'<td class="right mono">{buckets_b[cat]}</td></tr>')
        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

        lines.append("<table>")
        lines.append("<thead>")
        lines.append(f'<tr><th class="label-cell">Test statistic</th><th class="right">Value</th></tr>')
        lines.append("</thead>")
        lines.append("<tbody>")
        lines.append(f'<tr><td class="label-cell">\u03c7\u00b2 statistic</td><td class="right mono">{tc_test["chi2"]:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Degrees of freedom</td><td class="right mono">{tc_test["dof"]}</td></tr>')
        lines.append(f'<tr><td class="label-cell">p-value</td><td class="right mono">{tc_test["p_value"]:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Cram\u00e9r\'s V</td><td class="right mono">{tc_test["cramers_v"]:.3f}</td></tr>')
        lines.append(f'<tr><td class="label-cell">Effect size</td><td class="right">{tc_test["effect_size_label"]}</td></tr>')
        lines.append("</tbody>")
        lines.append("</table>")
        lines.append("")

        if tc_test.get("low_expected_warning"):
            lines.append("<p><em>Warning: Some expected cell frequencies are below 5, which may affect test validity.</em></p>")

    # Bonferroni summary
    bonf_threshold = metadata.get("bonferroni_threshold", 0.05 / 6)
    n_prop_tests = len(prop_tests)
    n_mw = 1  # just alignment_score
    n_chi = 1  # just task_completion
    total_section_tests = n_prop_tests + n_mw + n_chi
    lines.append("")
    lines.append("<h3>Bonferroni Correction</h3>")
    lines.append("")
    lines.append(f"<p>With {total_section_tests} independent tests conducted "
                 f"({n_mw} Mann-Whitney U, {n_prop_tests} proportion tests, {n_chi} chi-square), "
                 f"the Bonferroni-corrected significance threshold is \u03b1 = 0.05 / {total_section_tests} "
                 f"= {0.05 / total_section_tests:.4f}.</p>")
    lines.append("")

    # List significant results
    all_tests = []
    if alignment_test:
        all_tests.append(("Alignment score", alignment_test["p_value"], alignment_test.get("bonferroni_significant", False)))
    for t in prop_tests:
        all_tests.append((t["field"].replace("_", " ").title(), t["p_value"], t.get("bonferroni_significant", False)))
    if tc_test:
        all_tests.append(("Task completion distribution", tc_test["p_value"], tc_test.get("bonferroni_significant", False)))

    bonf_sig = [(name, p) for name, p, sig in all_tests if sig]
    marginal = [(name, p) for name, p, sig in all_tests if not sig and p < 0.05]
    non_sig = [(name, p) for name, p, sig in all_tests if not sig and p >= 0.05]

    if bonf_sig:
        lines.append(f"<p><strong>Tests surviving Bonferroni correction (p < {0.05 / total_section_tests:.4f}):</strong></p>")
        lines.append("<ul>")
        for name, p in bonf_sig:
            lines.append(f"<li><strong>{name}:</strong> p = {p:.6f} (significant)</li>")
        lines.append("</ul>")

    if marginal:
        lines.append("")
        lines.append("<p><strong>Tests significant at \u03b1 = 0.05 but not after correction:</strong></p>")
        lines.append("<ul>")
        for name, p in marginal:
            lines.append(f"<li><strong>{name}:</strong> p = {p:.4f} (marginal)</li>")
        lines.append("</ul>")

    if non_sig:
        lines.append("")
        lines.append("<p><strong>Non-significant tests:</strong></p>")
        lines.append("<ul>")
        for name, p in non_sig:
            extra = ""
            if "atisfaction" in name.lower() or "issatisfaction" in name.lower():
                extra = " (both models 0%)"
            lines.append(f"<li>{name}: p = {p:.3f}{extra}</li>")
        lines.append("</ul>")

    return "\n".join(lines)


def generate_compaction_outcomes(spec: dict, data: dict, config: dict) -> str:
    """Generate the compaction-outcomes expansion."""
    title = spec.get("title", "Pre/post compaction outcome data")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    ca = data.get("compaction", {}).get(ma, {}).get("outcome_correlation", {})
    cb = data.get("compaction", {}).get(mb, {}).get("outcome_correlation", {})

    lines.append("<table>")
    lines.append("    <thead>")
    lines.append(f'        <tr><th></th><th colspan="3" class="right">Opus {da}</th>'
                 f'<th colspan="3" class="right">Opus {db}</th></tr>')
    lines.append(f'        <tr><th>Metric</th><th class="right">Compacting &Delta;</th>'
                 f'<th class="right">Control &Delta;</th><th class="right">Net effect</th>'
                 f'<th class="right">Compacting &Delta;</th>'
                 f'<th class="right">Control &Delta;</th><th class="right">Net effect</th></tr>')
    lines.append("    </thead>")
    lines.append("    <tbody>")

    def fmt_delta(val, is_pct=False):
        if val is None:
            return "&mdash;"
        if is_pct:
            sign = "+" if val > 0 else ("&minus;" if val < 0 else "")
            return f"{sign}{abs(val * 100):.1f}pp"
        else:
            sign = "+" if val > 0 else ("&minus;" if val < 0 else "")
            return f"{sign}{abs(val):.2f}"

    metrics = [
        ("Alignment score", "avg_alignment", False),
        ("Satisfaction rate", "satisfaction_rate", True),
        ("Completion rate", "completion_rate", True),
    ]

    for label, key, is_pct in metrics:
        vals = []
        for comp_data in [ca, cb]:
            compacting = comp_data.get("compacting", {}).get("delta", {})
            control = comp_data.get("control", {}).get("delta", {})
            effect = comp_data.get("compaction_effect", {})
            vals.append(fmt_delta(compacting.get(key), is_pct))
            vals.append(fmt_delta(control.get(key), is_pct))
            vals.append(fmt_delta(effect.get(key), is_pct))

        lines.append(f'        <tr>')
        lines.append(f'            <td class="label-cell">{label}</td>')
        for v in vals:
            lines.append(f'            <td class="right mono">{v}</td>')
        lines.append(f'        </tr>')

    lines.append("    </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def generate_planning_complexity(spec: dict, data: dict, config: dict) -> str:
    """Generate the planning-complexity-detail expansion with _with_n support."""
    title = spec.get("title", "Planning alignment by complexity bin")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    pa = data.get("planning", {}).get(ma, {}).get("by_complexity", {})
    pb = data.get("planning", {}).get(mb, {}).get("by_complexity", {})

    lines.append("<table>")
    lines.append("    <thead>")
    lines.append(f'        <tr><th></th><th colspan="3" class="right">Opus {da}</th>'
                 f'<th colspan="3" class="right">Opus {db}</th></tr>')
    lines.append(f'        <tr><th>Complexity</th><th class="right">Planned</th>'
                 f'<th class="right">Unplanned</th><th class="right">&Delta;</th>'
                 f'<th class="right">Planned</th>'
                 f'<th class="right">Unplanned</th><th class="right">&Delta;</th></tr>')
    lines.append("    </thead>")
    lines.append("    <tbody>")

    for cx in ["trivial", "simple", "moderate", "complex+"]:
        row_a = pa.get(cx, {})
        row_b = pb.get(cx, {})

        def fmt_align_n(val, n):
            if val is None:
                return f"&mdash; (n={n})"
            return f"{val:.2f} (n={n})"

        def fmt_delta_val(val):
            if val is None:
                return "&mdash;"
            sign = "+" if val > 0 else ("&minus;" if val < 0 else "")
            return f"{sign}{abs(val):.2f}"

        planned_a = fmt_align_n(row_a.get("alignment_planned"), row_a.get("n_scored_planned", 0))
        unplanned_a = fmt_align_n(row_a.get("alignment_unplanned"), row_a.get("n_scored_unplanned", 0))
        delta_a = fmt_delta_val(row_a.get("alignment_delta"))

        planned_b = fmt_align_n(row_b.get("alignment_planned"), row_b.get("n_scored_planned", 0))
        unplanned_b = fmt_align_n(row_b.get("alignment_unplanned"), row_b.get("n_scored_unplanned", 0))
        delta_b = fmt_delta_val(row_b.get("alignment_delta"))

        cx_label = table_gen._label_case(cx)
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{cx_label}</td>')
        lines.append(f'            <td class="right mono">{planned_a}</td>'
                     f'<td class="right mono">{unplanned_a}</td>'
                     f'<td class="right mono">{delta_a}</td>')
        lines.append(f'            <td class="right mono">{planned_b}</td>'
                     f'<td class="right mono">{unplanned_b}</td>'
                     f'<td class="right mono">{delta_b}</td>')
        lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


# ── Custom generator dispatch ─────────────────────────────────────

CUSTOM_GENERATORS = {
    "edit_overview": generate_edit_overview,
    "iterative_refinement": generate_iterative_refinement,
    "triage_top_tasks": generate_triage_top_tasks,
    "satisfaction_stats": generate_satisfaction_stats,
    "compaction_outcomes": generate_compaction_outcomes,
    "planning_complexity": generate_planning_complexity,
    "stat_tests": None,  # handled by table_gen._generate_stat_test_rows
}


def generate_expansion_html(table_name: str, table_spec: dict, data: dict, config: dict) -> str:
    """Generate expansion HTML, dispatching to custom generators or table_gen."""
    row_gen = table_spec.get("row_gen")
    if row_gen and row_gen in CUSTOM_GENERATORS and CUSTOM_GENERATORS[row_gen]:
        return CUSTOM_GENERATORS[row_gen](table_spec, data, config)
    else:
        return table_gen.generate_table(table_spec, data, config)


# ── Prose extraction from existing expansions ─────────────────────

def extract_expansion_prose(content: str) -> dict:
    """Extract prose (preamble/postscript) from an existing expansion file.

    Returns dict with 'preamble' and 'postscript' strings.
    Prose is everything that isn't a <table>, <h3>, or <!-- title: --> line.
    """
    # Strip title comment
    if content.startswith("<!-- title:"):
        first_nl = content.index("\n") if "\n" in content else len(content)
        content = content[first_nl + 1:]

    # Find first <table or <h3 tag
    first_table = re.search(r'<(?:table|h3)\b', content)
    last_table_end = None
    for m in re.finditer(r'</(?:table|h3)>', content):
        last_table_end = m.end()

    preamble = ""
    postscript = ""

    if first_table:
        preamble = content[:first_table.start()].strip()
    if last_table_end and last_table_end < len(content):
        postscript = content[last_table_end:].strip()

    return {"preamble": preamble, "postscript": postscript}


# ── LLM Prose Check ──────────────────────────────────────────────

SECTION_PROSE_PROMPT = """You are a fact-checker for a technical report comparing two AI models.
Verify numerical claims in the HTML below against the provided data.

RULES:
1. ALWAYS correct wrong numbers, percentages, ratios, and comparisons.
2. NEVER rewrite prose that is factually correct. Preserve voice and structure.
3. If a relationship reversed (e.g., "A is higher" but data shows B is higher),
   update the direction AND number. Add: <!-- DIRECTION CHANGE: was "X", now "Y" -->
4. Update stat-card values (.value, .detail), table cells, bar-fill widths, and
   delta values to match data. Recalculate derived values (ratios, deltas, pp).
5. Preserve all HTML structure, CSS classes, and entities exactly.

CRITICAL OUTPUT FORMAT: Your entire response must be raw HTML only.
- Start your response with the first HTML tag (e.g., <h2>).
- Do NOT include any explanation, commentary, markdown, or preamble.
- Do NOT wrap output in code fences.
- If data is missing for some values, keep the original HTML value unchanged.

GUIDANCE: {guidance}

DATA:
{data_json}

SECTION HTML:
{section_html}"""

EXPANSION_PROSE_PROMPT = """You are a fact-checker for an expandable detail block in a technical report.
The block contains a data table (already updated) and interpretive prose.
Verify the prose against the data table and the provided metrics.

RULES:
1. ALWAYS correct wrong numbers and claims to match the table and data.
2. NEVER rewrite prose that is factually correct. Preserve voice and structure.
3. If a relationship reversed, flag with <!-- DIRECTION CHANGE: ... -->.
4. The table HTML is authoritative — do not modify it. Only update surrounding prose.
5. Output the complete expansion content (prose + table). Start with <!-- title: ... --> if there was one. No markdown fences.

GUIDANCE: {guidance}

DATA:
{data_json}

COMPUTED TABLE:
{table_html}

CURRENT EXPANSION PROSE (preamble before table):
{preamble}

CURRENT EXPANSION PROSE (postscript after table):
{postscript}"""


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (```html, ```xml, ``` etc.)
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1:]
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()
            stripped = stripped[:-3].rstrip()
    return stripped


def call_claude_sdk(prompt: str, model: str = "sonnet") -> str | None:
    """Call Claude via the Claude Code SDK (stdin pipe)."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  Warning: Claude SDK returned exit code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}", file=sys.stderr)
            return None
        return strip_code_fences(result.stdout.strip())
    except FileNotFoundError:
        print("  Error: 'claude' CLI not found. Install Claude Code SDK.", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("  Error: Claude SDK timed out after 300s", file=sys.stderr)
        return None


def load_prose_cache(cache_dir: Path) -> dict:
    """Load the prose check cache."""
    cache_file = cache_dir / "cache.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_prose_cache(cache_dir: Path, cache: dict):
    """Save the prose check cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "cache.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
        f.write("\n")


# ── Section extraction/replacement ────────────────────────────────

def extract_section(html: str, section_id: str) -> str | None:
    """Extract content of <section id="...">...</section>."""
    pattern = re.compile(
        rf'(<section\s+id="{re.escape(section_id)}">)(.*?)(</section>)',
        re.DOTALL
    )
    m = pattern.search(html)
    if m:
        return m.group(2)
    return None


def replace_section(html: str, section_id: str, new_content: str) -> str:
    """Replace section content in HTML."""
    pattern = re.compile(
        rf'(<section\s+id="{re.escape(section_id)}">)(.*?)(</section>)',
        re.DOTALL
    )
    return pattern.sub(rf'\g<1>{new_content}\g<3>', html)


# ── Main orchestrator ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Update report sections from data")
    parser.add_argument("--dir", type=Path, required=True,
                        help="Comparison directory (e.g., comparisons/opus-4.5-vs-4.6)")
    parser.add_argument("--sections", type=str, default=None,
                        help="Comma-separated list of section IDs to process")
    parser.add_argument("--tables-only", action="store_true",
                        help="Only regenerate expansion tables, skip prose checking")
    parser.add_argument("--prose-only", action="store_true",
                        help="Only check prose, skip table regeneration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing files")
    parser.add_argument("--model", type=str, default="opus",
                        help="Claude model for prose checking (default: opus)")
    parser.add_argument("--halt-on-reversal", action="store_true",
                        help="Exit with error if a direction change is detected")
    args = parser.parse_args()

    comparison_dir = args.dir.resolve()
    report_dir = comparison_dir / "report"
    specs_dir = report_dir / "specs"
    expansions_dir = report_dir / "expansions"
    cache_dir = report_dir / ".prose-cache"

    if not specs_dir.exists():
        print(f"Error: specs directory not found at {specs_dir}", file=sys.stderr)
        sys.exit(1)

    config = parse_comparison_dir(comparison_dir)
    print(f"Comparison: {config['model_a']} vs {config['model_b']}")
    print(f"Display: {config['display_a']} vs {config['display_b']}")

    # Load specs
    spec_files = sorted(specs_dir.glob("*.json"))
    if args.sections:
        selected = {s.strip() for s in args.sections.split(",")}
        spec_files = [f for f in spec_files if f.stem in selected]

    if not spec_files:
        print("No matching spec files found.")
        sys.exit(0)

    print(f"Specs: {', '.join(f.stem for f in spec_files)}")

    # Load template
    template_path = report_dir / "report.html"
    if not template_path.exists():
        print(f"Error: template not found at {template_path}", file=sys.stderr)
        sys.exit(1)
    template_html = template_path.read_text()
    template_modified = False

    # Load prose cache
    prose_cache = load_prose_cache(cache_dir)

    tables_generated = 0
    tables_skipped = 0
    prose_checked = 0
    prose_cached = 0
    prose_updated = 0
    direction_changes = []

    for spec_file in spec_files:
        spec = load_spec(spec_file)
        section_id = spec.get("section_id", spec_file.stem)
        print(f"\n--- Section: {section_id} ---")

        # Load data
        data = load_data_sources(spec, comparison_dir)

        # ── Table generation ──────────────────────────────────
        if not args.prose_only:
            for table_name, table_spec in spec.get("tables", {}).items():
                print(f"  Table: {table_name}")

                # Generate table HTML
                table_html = generate_expansion_html(table_name, table_spec, data, config)

                # Custom generators produce complete content; only extract
                # existing prose for spec-driven tables (no row_gen).
                is_custom = table_spec.get("row_gen") in CUSTOM_GENERATORS
                expansion_path = expansions_dir / f"{table_name}.html"
                existing_prose = {"preamble": "", "postscript": ""}
                if not is_custom and expansion_path.exists():
                    existing_content = expansion_path.read_text()
                    existing_prose = extract_expansion_prose(existing_content)

                # Extract title from generated table
                title_line = ""
                table_body = table_html
                if table_html.startswith("<!-- title:"):
                    first_nl = table_html.index("\n")
                    title_line = table_html[:first_nl]
                    table_body = table_html[first_nl + 1:]

                # Resolve title placeholders
                total_tests = data.get("stats", {}).get("metadata", {}).get("total_tests", "")
                title_line = title_line.replace("{total_tests}", str(total_tests))

                # ── Expansion prose check (if spec has prose guidance) ──
                exp_prose_spec = table_spec.get("prose", {})
                if exp_prose_spec and not args.tables_only and (existing_prose["preamble"] or existing_prose["postscript"]):
                    # Build data summary for prose check
                    key_metrics = spec.get("prose", {}).get("key_metrics", [])
                    metrics_summary = resolve_key_metrics(key_metrics, data, config)

                    cache_key = content_hash(
                        existing_prose["preamble"] + existing_prose["postscript"] +
                        table_body + json.dumps(metrics_summary, sort_keys=True)
                    )

                    if cache_key in prose_cache:
                        print(f"    Expansion prose: cached (skipping)")
                        checked_preamble = prose_cache[cache_key].get("preamble", existing_prose["preamble"])
                        checked_postscript = prose_cache[cache_key].get("postscript", existing_prose["postscript"])
                        prose_cached += 1
                    else:
                        prompt = EXPANSION_PROSE_PROMPT.format(
                            guidance=exp_prose_spec.get("guidance", ""),
                            data_json=json.dumps(metrics_summary, indent=2),
                            table_html=table_body,
                            preamble=existing_prose["preamble"] or "(no preamble)",
                            postscript=existing_prose["postscript"] or "(no postscript)",
                        )

                        print(f"    Expansion prose: checking with LLM...")
                        result = call_claude_sdk(prompt, args.model)
                        prose_checked += 1

                        if result:
                            # Parse result to extract prose around tables
                            checked_prose = extract_expansion_prose(result)
                            checked_preamble = checked_prose.get("preamble", existing_prose["preamble"])
                            checked_postscript = checked_prose.get("postscript", existing_prose["postscript"])

                            # Check for direction changes
                            if "DIRECTION CHANGE" in result:
                                changes = re.findall(r'<!-- DIRECTION CHANGE:(.+?)-->', result)
                                for change in changes:
                                    direction_changes.append(f"  {table_name}: {change.strip()}")
                                    print(f"    !! DIRECTION CHANGE: {change.strip()}")

                            prose_cache[cache_key] = {
                                "preamble": checked_preamble,
                                "postscript": checked_postscript,
                            }
                            prose_updated += 1
                        else:
                            checked_preamble = existing_prose["preamble"]
                            checked_postscript = existing_prose["postscript"]
                else:
                    checked_preamble = existing_prose["preamble"]
                    checked_postscript = existing_prose["postscript"]

                # Assemble final expansion
                parts = []
                if title_line:
                    parts.append(title_line)
                if checked_preamble:
                    parts.append(checked_preamble)
                parts.append(table_body)
                if checked_postscript:
                    parts.append(checked_postscript)
                final_html = "\n".join(parts) + "\n"

                if args.dry_run:
                    print(f"    Would write: {expansion_path}")
                    tables_generated += 1
                else:
                    expansions_dir.mkdir(parents=True, exist_ok=True)
                    expansion_path.write_text(final_html)
                    print(f"    Wrote: {expansion_path.name}")
                    tables_generated += 1

        # ── Section prose check ───────────────────────────────
        if not args.tables_only and spec.get("prose"):
            prose_spec = spec["prose"]
            key_metrics = prose_spec.get("key_metrics", [])
            metrics_summary = resolve_key_metrics(key_metrics, data, config)

            section_html = extract_section(template_html, section_id)
            if section_html is None:
                print(f"  Section '{section_id}' not found in template, skipping prose check")
                continue

            cache_key = content_hash(
                section_html + json.dumps(metrics_summary, sort_keys=True)
            )

            if cache_key in prose_cache:
                print(f"  Section prose: cached (skipping)")
                prose_cached += 1
                continue

            prompt = SECTION_PROSE_PROMPT.format(
                guidance=prose_spec.get("guidance", ""),
                data_json=json.dumps(metrics_summary, indent=2),
                section_html=section_html,
            )

            print(f"  Section prose: checking with LLM...")
            result = call_claude_sdk(prompt, args.model)
            prose_checked += 1

            if result:
                # Guard: reject LLM output that doesn't look like valid section HTML.
                # Must start with an HTML tag — reject if it leads with explanatory
                # prose, markdown, or questions. Fallback: try to extract HTML portion.
                stripped = result.strip()
                looks_like_html = (
                    stripped.startswith("<") or
                    stripped.startswith("<!--")
                )
                if not looks_like_html:
                    # Fallback: try to find where HTML starts in mixed output
                    html_start = re.search(r'\n\s*(<(?:h[1-6]|!--))', result)
                    if html_start:
                        result = result[html_start.start(1):]
                        looks_like_html = True
                        print(f"  Section '{section_id}': stripped non-HTML preamble ({html_start.start(1)} chars)")
                if not looks_like_html:
                    print(f"  Section '{section_id}': LLM returned non-HTML, skipping")
                    print(f"    Preview: {result[:120]}...")
                else:
                    # Check for direction changes
                    if "DIRECTION CHANGE" in result:
                        changes = re.findall(r'<!-- DIRECTION CHANGE:(.+?)-->', result)
                        for change in changes:
                            direction_changes.append(f"  {section_id}: {change.strip()}")
                            print(f"  !! DIRECTION CHANGE: {change.strip()}")

                    # Check if content actually changed
                    if result.strip() != section_html.strip():
                        if args.dry_run:
                            print(f"  Would update section '{section_id}' in template")
                        else:
                            template_html = replace_section(template_html, section_id, result)
                            template_modified = True
                            print(f"  Updated section '{section_id}'")
                        prose_updated += 1
                    else:
                        print(f"  Section '{section_id}': no changes needed")

                    prose_cache[cache_key] = {"unchanged": True}
                    # Also cache the post-update content so re-runs are no-ops
                    if result.strip() != section_html.strip():
                        new_cache_key = content_hash(
                            result.strip() + json.dumps(metrics_summary, sort_keys=True)
                        )
                        prose_cache[new_cache_key] = {"unchanged": True}

    # Write template if modified
    if template_modified and not args.dry_run:
        template_path.write_text(template_html)
        print(f"\nWrote updated template: {template_path}")

    # Save prose cache
    if not args.dry_run:
        save_prose_cache(cache_dir, prose_cache)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Tables generated: {tables_generated}")
    if not args.tables_only:
        print(f"  Prose checks: {prose_checked} (cached: {prose_cached}, updated: {prose_updated})")
    if direction_changes:
        print(f"  Direction changes detected: {len(direction_changes)}")
        for dc in direction_changes:
            print(f"    {dc}")
    print(f"{'='*50}")

    if args.halt_on_reversal and direction_changes:
        print("Halting due to direction changes (--halt-on-reversal)", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
