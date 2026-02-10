#!/usr/bin/env python3
"""Update report sections from data: auto-generate expansion tables and
LLM-check prose against current analysis data.

Driven by per-section spec files in report/specs/.

Usage:
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --sections cost,edit-accuracy
    python scripts/update_sections.py --dir comparisons/opus-4.5-vs-4.6 --tables-only
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


def _fmt_tokens(n: int) -> str:
    """Format large token counts with B/M/K suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n:,.0f}"
    return str(n)


def generate_dataset_composition(spec: dict, data: dict, config: dict) -> str:
    """Generate the dataset-composition expansion (per-model breakdown table)."""
    title = spec.get("title", "Per-model composition")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    ov = data.get("overview", {})
    a, b, c = ov.get(ma, {}), ov.get(mb, {}), ov.get("combined", {})

    lines.append(
        "<p>The dataset reflects organic usage patterns, not a controlled experiment. "
        f"Opus {da} accumulated sessions over two months of daily use; "
        f"Opus {db} entered evaluation in early February 2026.</p>"
    )
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append("        <tr>")
    lines.append('            <th>Metric</th>')
    lines.append(f'            <th class="right">Opus {da}</th>')
    lines.append(f'            <th class="right">Opus {db}</th>')
    lines.append(f'            <th class="right">Combined</th>')
    lines.append("        </tr>")
    lines.append("    </thead>")
    lines.append("    <tbody>")

    def _fmt_date(d):
        if not d:
            return "&mdash;"
        from datetime import datetime
        dt = datetime.strptime(d, "%Y-%m-%d")
        return dt.strftime("%b %-d")

    dr_a = a.get("date_range", {})
    dr_b = b.get("date_range", {})
    dr_c = c.get("date_range", {})

    rows = [
        ("Sessions", f'{a.get("sessions", 0):,}', f'{b.get("sessions", 0):,}',
         f'{c.get("sessions", 0):,}'),
        ("Tasks", f'{a.get("tasks", 0):,}', f'{b.get("tasks", 0):,}',
         f'{c.get("tasks", 0):,}'),
        ("Tasks / session", f'{a.get("tasks_per_session", 0)}',
         f'{b.get("tasks_per_session", 0)}',
         f'{c["tasks"] / c["sessions"]:.1f}' if c.get("sessions") else "&mdash;"),
        ("Projects", f'{a.get("projects", 0):,}', f'{b.get("projects", 0):,}',
         f'{c.get("projects", 0):,}'),
        ("Date range",
         f'{_fmt_date(dr_a.get("start"))} &ndash; {_fmt_date(dr_a.get("end"))}',
         f'{_fmt_date(dr_b.get("start"))} &ndash; {_fmt_date(dr_b.get("end"))}',
         f'{_fmt_date(dr_c.get("start"))} &ndash; {_fmt_date(dr_c.get("end"))}'),
        ("User messages", f'{a.get("total_user_messages", 0):,}',
         f'{b.get("total_user_messages", 0):,}',
         f'{c.get("total_user_messages", 0):,}'),
        ("Tool calls", f'{a.get("total_tool_calls", 0):,}',
         f'{b.get("total_tool_calls", 0):,}',
         f'{c.get("total_tool_calls", 0):,}'),
    ]

    for label, va, vb, vc in rows:
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{label}</td>')
        lines.append(f'            <td class="right mono">{va}</td>')
        lines.append(f'            <td class="right mono">{vb}</td>')
        lines.append(f'            <td class="right mono">{vc}</td>')
        lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    ratio = a.get("sessions", 1) / max(b.get("sessions", 1), 1)
    shared = len(set(a.get("project_names", [])) & set(b.get("project_names", [])))
    lines.append("")
    lines.append(
        f"<p>The {ratio:.0f}:1 session ratio means per-task averages for Opus {da} "
        f"are more robust, while Opus {db} estimates carry wider confidence intervals. "
        f"Opus {db} sessions are concentrated across {b.get('projects', 0)} projects "
        f"(all of which also have Opus {da} sessions), providing natural overlap "
        "for matched-pair comparisons where they apply.</p>"
    )

    return "\n".join(lines)


def generate_dataset_task_mix(spec: dict, data: dict, config: dict) -> str:
    """Generate the dataset-task-mix expansion (type + complexity distributions)."""
    title = spec.get("title", "Task type and complexity distribution")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    ov = data.get("overview", {})
    a, b = ov.get(ma, {}), ov.get(mb, {})

    n_a, n_b = a.get("tasks", 1), b.get("tasks", 1)
    types_a = a.get("task_types", {})
    types_b = b.get("task_types", {})

    # Ordered by 4.5 count descending, unknown last
    type_order = [t for t in sorted(types_a, key=lambda t: types_a[t], reverse=True)
                  if t != "unknown"]
    # Add types only in model B
    for t in sorted(types_b, key=lambda t: types_b[t], reverse=True):
        if t != "unknown" and t not in type_order:
            type_order.append(t)
    type_order.append("unknown")

    lines.append("<h3>By task type</h3>")
    lines.append(
        "<p>Tasks are classified by primary type using heuristic pattern matching on "
        'prompts, tool usage, and file operations. "Unknown" tasks lacked clear '
        "classification signals.</p>"
    )
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append("        <tr>")
    lines.append('            <th>Type</th>')
    lines.append(f'            <th class="right">{da} count</th>')
    lines.append(f'            <th class="right">{da} %</th>')
    lines.append(f'            <th class="right">{db} count</th>')
    lines.append(f'            <th class="right">{db} %</th>')
    lines.append("        </tr>")
    lines.append("    </thead>")
    lines.append("    <tbody>")

    for t in type_order:
        ca = types_a.get(t, 0)
        cb = types_b.get(t, 0)
        pct_a = f"{ca / n_a * 100:.1f}%" if ca else "&mdash;"
        pct_b = f"{cb / n_b * 100:.1f}%" if cb else "&mdash;"
        label = t.title() if t != "unknown" else "Unknown"
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{label}</td>')
        lines.append(f'            <td class="right mono">{ca:,}</td>')
        lines.append(f'            <td class="right mono">{pct_a}</td>')
        lines.append(f'            <td class="right mono">{cb:,}</td>')
        lines.append(f'            <td class="right mono">{pct_b}</td>')
        lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    # Complexity distribution
    cx_order = ["trivial", "simple", "moderate", "complex", "major"]
    cx_a = a.get("complexity_distribution", {})
    cx_b = b.get("complexity_distribution", {})

    lines.append("")
    lines.append("<h3>By complexity</h3>")
    lines.append(
        "<p>Complexity is inferred from tool count, files touched, and lines changed. "
        "Over half of all tasks are trivial (single-turn interactions), while major "
        "tasks (&gt;50 tool calls or &gt;500 lines) represent ~1% of volume but a "
        "significant share of cost.</p>"
    )
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append("        <tr>")
    lines.append('            <th>Complexity</th>')
    lines.append(f'            <th class="right">{da} count</th>')
    lines.append(f'            <th class="right">{da} %</th>')
    lines.append(f'            <th class="right">{db} count</th>')
    lines.append(f'            <th class="right">{db} %</th>')
    lines.append('            <th>Distribution</th>')
    lines.append("        </tr>")
    lines.append("    </thead>")
    lines.append("    <tbody>")

    for cx in cx_order:
        ca = cx_a.get(cx, 0)
        cb = cx_b.get(cx, 0)
        pct_a = ca / n_a * 100
        pct_b = cb / n_b * 100
        bar = table_gen.generate_bar_pair(pct_a, pct_b, scale=100)
        label = cx.title()
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{label}</td>')
        lines.append(f'            <td class="right mono">{ca:,}</td>')
        lines.append(f'            <td class="right mono">{pct_a:.1f}%</td>')
        lines.append(f'            <td class="right mono">{cb:,}</td>')
        lines.append(f'            <td class="right mono">{pct_b:.1f}%</td>')
        lines.append(f'            <td class="bar-cell">{bar}</td>')
        lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    # Summary note
    mod_plus_a = sum(cx_a.get(c, 0) for c in ["moderate", "complex", "major"])
    mod_plus_b = sum(cx_b.get(c, 0) for c in ["moderate", "complex", "major"])
    pct_mod_a = mod_plus_a / n_a * 100
    pct_mod_b = mod_plus_b / n_b * 100
    lines.append("")
    lines.append(
        "<p>The task type distributions are broadly similar across models, "
        "suggesting the user's work patterns remained consistent. "
        "The complexity mix is also comparable, though Opus "
        f"{db} has a slightly higher share of moderate-and-above tasks "
        f"({pct_mod_b:.1f}% vs {pct_mod_a:.1f}%), likely reflecting the evaluation "
        "period's focus on substantive work rather than quick queries.</p>"
    )

    return "\n".join(lines)


def generate_dataset_volume(spec: dict, data: dict, config: dict) -> str:
    """Generate the dataset-volume expansion (token volumes and code output)."""
    title = spec.get("title", "Token volumes and code output")
    lines = [f"<!-- title: {title} -->"]

    ma, mb = config["model_a"], config["model_b"]
    da, db = config["display_a"], config["display_b"]
    ov = data.get("overview", {})
    a, b, c = ov.get(ma, {}), ov.get(mb, {}), ov.get("combined", {})

    lines.append(
        '<p>Raw token volumes across the full dataset. These are absolute totals, '
        'not per-task averages (see <a href="#cost">&sect;2</a> for normalized '
        'comparisons).</p>'
    )
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append("        <tr>")
    lines.append('            <th>Metric</th>')
    lines.append(f'            <th class="right">Opus {da}</th>')
    lines.append(f'            <th class="right">Opus {db}</th>')
    lines.append(f'            <th class="right">Combined</th>')
    lines.append("        </tr>")
    lines.append("    </thead>")
    lines.append("    <tbody>")

    token_rows = [
        ("Output tokens", "total_output_tokens"),
        ("Input tokens (fresh)", "total_input_tokens"),
        ("Cache read tokens", "cache_read_tokens"),
        ("Cache write tokens", "cache_write_tokens"),
    ]
    for label, key in token_rows:
        va = _fmt_tokens(a.get(key, 0))
        vb = _fmt_tokens(b.get(key, 0))
        vc = _fmt_tokens(c.get(key, 0))
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{label}</td>')
        lines.append(f'            <td class="right mono">{va}</td>')
        lines.append(f'            <td class="right mono">{vb}</td>')
        lines.append(f'            <td class="right mono">{vc}</td>')
        lines.append("        </tr>")

    # Cost row
    lines.append("        <tr>")
    lines.append(f'            <td class="label-cell">Total API cost</td>')
    lines.append(f'            <td class="right mono">${a.get("total_cost_usd", 0):,.2f}</td>')
    lines.append(f'            <td class="right mono">${b.get("total_cost_usd", 0):,.2f}</td>')
    lines.append(f'            <td class="right mono">${c.get("total_cost_usd", 0):,.2f}</td>')
    lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    # Output composition
    lines.append("")
    lines.append("<h3>Output composition</h3>")
    lines.append(
        "<p>Model output splits into <em>thinking</em> (extended thinking / "
        "chain-of-thought, not billed as output) and <em>text</em> (visible response, "
        "code, tool calls). Estimated from character counts with a 3:1 chars-to-tokens "
        "ratio for thinking.</p>"
    )
    lines.append("")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append("        <tr>")
    lines.append('            <th>Metric</th>')
    lines.append(f'            <th class="right">Opus {da}</th>')
    lines.append(f'            <th class="right">Opus {db}</th>')
    lines.append("        </tr>")
    lines.append("    </thead>")
    lines.append("    <tbody>")

    comp_rows = [
        ("Est. thinking tokens", "estimated_thinking_tokens", True),
        ("Est. text tokens", "estimated_text_tokens", True),
        ("Thinking ratio (tasks using thinking)", "thinking_ratio", False),
        ("Avg requests / task", "avg_requests_per_task", False),
    ]
    for label, key, is_count in comp_rows:
        va_raw = a.get(key, 0)
        vb_raw = b.get(key, 0)
        if is_count:
            va, vb = f"{va_raw:,}", f"{vb_raw:,}"
        elif key == "thinking_ratio":
            va, vb = f"{va_raw * 100:.1f}%", f"{vb_raw * 100:.1f}%"
        else:
            va, vb = f"{va_raw}", f"{vb_raw}"
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{label}</td>')
        lines.append(f'            <td class="right mono">{va}</td>')
        lines.append(f'            <td class="right mono">{vb}</td>')
        lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    # Code output
    lines.append("")
    lines.append("<h3>Code output</h3>")
    lines.append("<table>")
    lines.append("    <thead>")
    lines.append("        <tr>")
    lines.append('            <th>Metric</th>')
    lines.append(f'            <th class="right">Opus {da}</th>')
    lines.append(f'            <th class="right">Opus {db}</th>')
    lines.append(f'            <th class="right">Combined</th>')
    lines.append("        </tr>")
    lines.append("    </thead>")
    lines.append("    <tbody>")

    code_rows = [
        ("Files touched", "total_files_touched"),
        ("Lines added", "total_lines_added"),
        ("Lines removed", "total_lines_removed"),
    ]
    for label, key in code_rows:
        lines.append("        <tr>")
        lines.append(f'            <td class="label-cell">{label}</td>')
        lines.append(f'            <td class="right mono">{a.get(key, 0):,}</td>')
        lines.append(f'            <td class="right mono">{b.get(key, 0):,}</td>')
        lines.append(f'            <td class="right mono">{c.get(key, 0):,}</td>')
        lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    # Closing note about cache dominance
    total_all = (c.get("total_input_tokens", 0) + c.get("total_output_tokens", 0)
                 + c.get("cache_read_tokens", 0) + c.get("cache_write_tokens", 0))
    cache_pct = c.get("cache_read_tokens", 0) / total_all * 100 if total_all else 0
    lines.append("")
    lines.append(
        f"<p>Cache reads dominate the token budget: {cache_pct:.0f}% of all tokens "
        "processed were served from cache rather than freshly encoded. This reflects "
        "Claude Code's prompt architecture, where the system prompt and conversation "
        "history are re-sent with each API call but largely hit the prompt cache.</p>"
    )

    return "\n".join(lines)


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
        bar = table_gen.generate_bar_pair(rate_a, rate_b, scale=100)
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

        bar = table_gen.generate_bar_pair(rate_a, rate_b, scale=100)
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
    "dataset_composition": generate_dataset_composition,
    "dataset_task_mix": generate_dataset_task_mix,
    "dataset_volume": generate_dataset_volume,
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

    # Strip GENERATED-TABLE markers before extracting prose
    content = content.replace("<!-- GENERATED-TABLE -->\n", "")
    content = content.replace("<!-- /GENERATED-TABLE -->\n", "")
    content = content.replace("<!-- GENERATED-TABLE -->", "")
    content = content.replace("<!-- /GENERATED-TABLE -->", "")

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

DOCUMENT_PROSE_PROMPT = """You are a fact-checker for a technical report comparing two AI models.
The document below has been auto-updated with fresh data tables, but the
surrounding prose may contain stale numbers. Review ALL prose and identify
corrections needed.

RULES:
1. Only report numbers, percentages, ratios, comparisons, or bar-fill widths
   that are WRONG when checked against the provided DATA.
2. Do NOT report content that is factually correct.
3. Do NOT touch content between <!-- GENERATED-TABLE --> and
   <!-- /GENERATED-TABLE --> markers.
4. The "old" field must be an exact substring of the document — include enough
   surrounding text to make the match unique (aim for 40-120 chars).
5. Preserve all HTML tags, CSS classes, and entities in "new".
6. If a direction reversed (e.g., "A is higher" but data shows B is higher),
   include "DIRECTION CHANGE" at the start of the reason.

OUTPUT FORMAT: Return ONLY a JSON array. No commentary, no markdown fences.
Each element:
  {{"old": "exact substring to find", "new": "replacement", "reason": "brief explanation"}}

If everything is factually correct, return: []

SECTION NOTES:
{guidance}

DATA:
{data_json}

DOCUMENT:
{document}"""


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
            timeout=600,
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
        print("  Error: Claude SDK timed out after 600s", file=sys.stderr)
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


# ── Annotated template build/decompose ────────────────────────────

def build_annotated_template(template_html: str, expansions_dir: Path,
                              overrides: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Replace <!-- expand: name --> with BEGIN/END-EXPANSION sentinels.

    Inlines expansion file content (or override content) between markers so
    the LLM sees the complete document in a single pass.

    Args:
        template_html: The report template with <!-- expand: name --> markers.
        expansions_dir: Directory containing expansion HTML files.
        overrides: Optional dict of expansion name -> content to use instead
                   of reading from disk (used for dry-run with freshly
                   generated tables).

    Returns:
        (annotated_html, expansion_names) tuple.
    """
    expansion_names = []
    overrides = overrides or {}

    def replace_marker(m):
        name = m.group(1).strip()
        expansion_names.append(name)
        if name in overrides:
            content = overrides[name].rstrip("\n")
        else:
            exp_path = expansions_dir / f"{name}.html"
            if exp_path.exists():
                content = exp_path.read_text().rstrip("\n")
            else:
                content = f"<!-- expansion file not found: {name} -->"
        return f"<!-- BEGIN-EXPANSION: {name} -->\n{content}\n<!-- END-EXPANSION: {name} -->"

    annotated = re.sub(r'<!-- expand: (\S+) -->', replace_marker, template_html)
    return annotated, expansion_names


def decompose_document(doc: str, expansion_names: list[str]) -> tuple[str, dict[str, str]]:
    """Split annotated document back into template and expansion files.

    Extracts content between BEGIN/END-EXPANSION sentinel pairs, restoring
    <!-- expand: name --> markers in the template.

    Returns:
        (template_html, expansions) where expansions maps name -> content.
    """
    expansions = {}

    def extract_and_replace(m):
        name = m.group(1).strip()
        content = m.group(2)
        # Trim exactly one leading/trailing newline from the sentinel wrapper
        if content.startswith("\n"):
            content = content[1:]
        if content.endswith("\n"):
            content = content[:-1]
        expansions[name] = content
        return f"<!-- expand: {name} -->"

    template = re.sub(
        r'<!-- BEGIN-EXPANSION: (\S+) -->\n?(.*?)\n?<!-- END-EXPANSION: \1 -->',
        extract_and_replace,
        doc,
        flags=re.DOTALL,
    )

    for name in expansion_names:
        if name not in expansions:
            print(f"  Warning: expansion '{name}' not found in LLM output", file=sys.stderr)

    return template, expansions


def parse_corrections(raw: str) -> list[dict] | None:
    """Parse LLM output as a JSON array of corrections.

    Handles code fences and leading/trailing junk around the JSON.
    Returns None if parsing fails.
    """
    text = strip_code_fences(raw).strip()
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Try to extract JSON array from surrounding text
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def apply_corrections(doc: str, corrections: list[dict]) -> tuple[str, int, list[str]]:
    """Apply a list of corrections to the annotated document.

    Each correction is {"old": "...", "new": "...", "reason": "..."}.

    Returns:
        (modified_doc, n_applied, direction_changes)
    """
    n_applied = 0
    direction_changes = []
    skipped = []

    for i, corr in enumerate(corrections):
        old = corr.get("old", "")
        new = corr.get("new", "")
        reason = corr.get("reason", "")

        if not old or not new:
            print(f"    Correction {i}: empty old/new, skipping")
            continue

        if old == new:
            continue

        # Check the correction isn't inside a GENERATED-TABLE block
        pos = doc.find(old)
        if pos == -1:
            skipped.append(f"Correction {i}: old text not found: {old[:80]}...")
            continue

        # Find the nearest GENERATED-TABLE markers
        gen_start = doc.rfind("<!-- GENERATED-TABLE -->", 0, pos)
        gen_end = doc.find("<!-- /GENERATED-TABLE -->", pos)
        if gen_start != -1 and gen_end != -1:
            # old text is between markers — check if the end marker comes
            # before any intervening close marker
            close_before = doc.find("<!-- /GENERATED-TABLE -->", gen_start)
            if close_before >= pos:
                skipped.append(f"Correction {i}: inside GENERATED-TABLE block, skipping: {old[:60]}...")
                continue

        # Check uniqueness
        count = doc.count(old)
        if count > 1:
            skipped.append(f"Correction {i}: old text matches {count} locations, skipping: {old[:60]}...")
            continue

        doc = doc.replace(old, new, 1)
        n_applied += 1
        print(f"    Applied: {reason}")

        if "DIRECTION CHANGE" in reason.upper():
            direction_changes.append(reason)

    if skipped:
        for msg in skipped:
            print(f"    Skipped: {msg}")

    return doc, n_applied, direction_changes


def collect_all_metrics(specs_loaded: list[tuple], config: dict) -> dict:
    """Merge key_metrics from all loaded specs into a single dict.

    Args:
        specs_loaded: List of (spec_file, spec, data) tuples.
        config: Model config from parse_comparison_dir.
    """
    all_metrics = {}
    for _spec_file, spec, data in specs_loaded:
        key_metrics = spec.get("prose", {}).get("key_metrics", [])
        metrics = resolve_key_metrics(key_metrics, data, config)
        all_metrics.update(metrics)
    return all_metrics


def collect_guidance(specs_loaded: list[tuple]) -> str:
    """Build per-section guidance notes from all loaded specs."""
    lines = []
    for _spec_file, spec, _data in specs_loaded:
        section_id = spec.get("section_id", "")
        section_guidance = spec.get("prose", {}).get("guidance", "")
        if section_guidance:
            lines.append(f"- {section_id}: {section_guidance}")
        for table_name, table_spec in spec.get("tables", {}).items():
            table_guidance = table_spec.get("prose", {}).get("guidance", "")
            if table_guidance:
                lines.append(f"- {table_name}: {table_guidance}")
    return "\n".join(lines)


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

    tables_generated = 0
    direction_changes = []
    specs_loaded = []  # (spec_file, spec, data) for phase 2
    generated_expansions = {}  # name -> final_html for dry-run support

    # ── Phase 1: Table generation ─────────────────────────────

    for spec_file in spec_files:
        spec = load_spec(spec_file)
        section_id = spec.get("section_id", spec_file.stem)
        print(f"\n--- Section: {section_id} ---")

        data = load_data_sources(spec, comparison_dir)
        specs_loaded.append((spec_file, spec, data))

        for table_name, table_spec in spec.get("tables", {}).items():
            print(f"  Table: {table_name}")

            table_html = generate_expansion_html(table_name, table_spec, data, config)

            is_custom = table_spec.get("row_gen") in CUSTOM_GENERATORS

            # Extract title line
            title_line = ""
            table_body = table_html
            if table_html.startswith("<!-- title:"):
                first_nl = table_html.index("\n")
                title_line = table_html[:first_nl]
                table_body = table_html[first_nl + 1:]

            # Resolve title placeholders
            total_tests = data.get("stats", {}).get("metadata", {}).get("total_tests", "")
            title_line = title_line.replace("{total_tests}", str(total_tests))

            # Assemble expansion with GENERATED-TABLE markers
            expansion_path = expansions_dir / f"{table_name}.html"
            parts = []
            if title_line:
                parts.append(title_line)

            if is_custom:
                # Custom generators: wrap entire body in markers
                parts.append("<!-- GENERATED-TABLE -->")
                parts.append(table_body)
                parts.append("<!-- /GENERATED-TABLE -->")
            else:
                # Spec-driven: preserve existing prose, wrap just the table
                existing_prose = {"preamble": "", "postscript": ""}
                if expansion_path.exists():
                    existing_content = expansion_path.read_text()
                    existing_prose = extract_expansion_prose(existing_content)

                if existing_prose["preamble"]:
                    parts.append(existing_prose["preamble"])
                parts.append("<!-- GENERATED-TABLE -->")
                parts.append(table_body)
                parts.append("<!-- /GENERATED-TABLE -->")
                if existing_prose["postscript"]:
                    parts.append(existing_prose["postscript"])

            final_html = "\n".join(parts) + "\n"
            generated_expansions[table_name] = final_html

            if args.dry_run:
                print(f"    Would write: {expansion_path}")
            else:
                expansions_dir.mkdir(parents=True, exist_ok=True)
                expansion_path.write_text(final_html)
                print(f"    Wrote: {expansion_path.name}")
            tables_generated += 1

    # ── Phase 2: Whole-document prose check ───────────────────

    prose_checked = 0
    prose_cached = 0
    prose_updated = 0

    if not args.tables_only:
        # Build annotated template — use overrides so dry-run sees fresh tables
        annotated, expansion_names = build_annotated_template(
            template_html, expansions_dir, overrides=generated_expansions)

        # Collect all metrics and guidance from processed specs
        all_metrics = collect_all_metrics(specs_loaded, config)
        guidance = collect_guidance(specs_loaded)

        # Single cache key (include sections filter for correctness)
        sections_key = args.sections or "all"
        cache_input = sections_key + "\n" + annotated + json.dumps(all_metrics, sort_keys=True)
        cache_key = content_hash(cache_input)

        prose_cache = load_prose_cache(cache_dir)

        if cache_key in prose_cache:
            print(f"\nDocument prose: cached (skipping)")
            prose_cached = 1
        else:
            prompt = DOCUMENT_PROSE_PROMPT.format(
                guidance=guidance,
                data_json=json.dumps(all_metrics, indent=2),
                document=annotated,
            )

            print(f"\nDocument prose: checking with LLM...")
            result = call_claude_sdk(prompt, args.model)
            prose_checked = 1

            if result:
                corrections = parse_corrections(result)
                if corrections is None:
                    print(f"  Warning: could not parse LLM output as JSON", file=sys.stderr)
                    print(f"  Preview: {result[:200]}...", file=sys.stderr)
                elif len(corrections) == 0:
                    print(f"  No corrections needed (all prose is factually correct)")
                    prose_cache[cache_key] = {"unchanged": True}
                    if not args.dry_run:
                        save_prose_cache(cache_dir, prose_cache)
                else:
                    print(f"  {len(corrections)} correction(s) found:")

                    if args.dry_run:
                        # Show what would change without applying
                        for i, corr in enumerate(corrections):
                            reason = corr.get("reason", "")
                            old_preview = corr.get("old", "")[:80]
                            print(f"    [{i}] {reason}: {old_preview}...")
                            if "DIRECTION CHANGE" in reason.upper():
                                direction_changes.append(reason)
                        prose_updated = len(corrections)
                    else:
                        # Apply corrections to annotated template
                        modified, n_applied, dir_changes = apply_corrections(
                            annotated, corrections)
                        direction_changes.extend(dir_changes)

                        if n_applied > 0:
                            # Decompose back into template + expansions
                            new_template, new_expansions = decompose_document(
                                modified, expansion_names)

                            # Write template
                            if new_template.strip() != template_html.strip():
                                template_path.write_text(new_template)
                                print(f"  Updated template: {template_path}")
                                prose_updated += 1

                            # Write modified expansions
                            for name, content in new_expansions.items():
                                exp_path = expansions_dir / f"{name}.html"
                                new_content = content + "\n"
                                if exp_path.exists() and exp_path.read_text() == new_content:
                                    continue
                                exp_path.write_text(new_content)
                                print(f"  Updated expansion: {name}")
                                prose_updated += 1
                        else:
                            print(f"  No corrections could be applied")

                        # Cache — both pre and post states
                        prose_cache[cache_key] = {"unchanged": True}
                        if n_applied > 0:
                            new_annotated, _ = build_annotated_template(
                                new_template, expansions_dir)
                            new_cache_input = sections_key + "\n" + new_annotated + json.dumps(all_metrics, sort_keys=True)
                            new_cache_key = content_hash(new_cache_input)
                            if new_cache_key != cache_key:
                                prose_cache[new_cache_key] = {"unchanged": True}
                        save_prose_cache(cache_dir, prose_cache)
            else:
                print("  LLM call failed, prose unchanged")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Tables generated: {tables_generated}")
    if not args.tables_only:
        print(f"  Prose: {prose_checked} LLM call(s) (cached: {prose_cached}, updated: {prose_updated})")
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
