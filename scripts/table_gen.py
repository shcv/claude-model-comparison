#!/usr/bin/env python3
"""Pure table generation from spec + data.

No LLM dependency. Produces HTML expansion fragments from table specs
and resolved data. Used by update_sections.py.
"""

import re
from typing import Any


# Canonical complexity ordering used across the report
COMPLEXITY_ORDER = ["trivial", "simple", "moderate", "complex", "major"]
COMPLEXITY_PLUS_ORDER = ["trivial", "simple", "moderate", "complex+"]


def resolve_path(data: dict, path: str, config: dict) -> Any:
    """Resolve a dotted path against nested data, substituting model placeholders.

    Paths like "tokens.{model_a}.overall.avg_cost_usd" are first expanded
    using config's model_a/model_b values, then walked against the data dict.

    Args:
        data: Merged data dict (keyed by data_source name)
        path: Dotted path with optional {model_a}/{model_b} placeholders
        config: Must contain 'model_a' and 'model_b' keys (e.g. "opus-4-5")

    Returns:
        The resolved value, or None if path doesn't resolve.
    """
    expanded = path.format(**config)
    parts = expanded.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None
        elif isinstance(current, list):
            # Support [key=value] filter syntax for arrays of dicts
            m = re.match(r'\[(\w+)=([^\]]+)\]', part)
            if m:
                filter_key, filter_val = m.group(1), m.group(2)
                found = None
                for item in current:
                    if isinstance(item, dict) and str(item.get(filter_key)) == filter_val:
                        found = item
                        break
                if found is None:
                    return None
                current = found
            else:
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
        else:
            return None
    return current


def format_value(value: Any, fmt: str) -> str:
    """Apply a Python format string to a value.

    Handles None/null gracefully. The format string should use Python's
    str.format() syntax, e.g. "{:,.0f}", "{:.1%}", "{:.2f}".

    Special formats:
        "dash_if_none" - returns em-dash for None values
        "pct" - multiplies by 100 and appends %
    """
    if value is None:
        return "&mdash;"
    if fmt == "dash_if_none":
        return "&mdash;" if value is None else str(value)
    if fmt == "pct":
        return f"{value * 100:.0f}%"
    if fmt == "pct1":
        return f"{value * 100:.1f}%"
    if fmt == "pp":
        sign = "+" if value > 0 else ("&minus;" if value < 0 else "")
        return f"{sign}{abs(value * 100):.1f}pp"
    if fmt == "pp_raw":
        # Value already in percentage points (not a fraction)
        sign = "+" if value > 0 else ("&minus;" if value < 0 else "")
        return f"{sign}{abs(value):.1f}pp"
    if fmt == "delta":
        if value is None:
            return "&mdash;"
        sign = "+" if value > 0 else ("&minus;" if value < 0 else "")
        return f"{sign}{abs(value):.2f}"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def generate_bar_pair(val_a: float, val_b: float, scale: float = 100.0) -> str:
    """Generate bar-pair HTML matching the existing CSS conventions.

    Produces the bar-cell structure used in iterative-refinement-detail.html.

    Args:
        val_a: Value for model A (percentage, 0-100 scale)
        val_b: Value for model B (percentage, 0-100 scale)
        scale: Maximum value for scaling bar widths (default 100)
    """
    width_a = min(val_a / scale * 100, 100) if scale else 0
    width_b = min(val_b / scale * 100, 100) if scale else 0
    return (
        f'<div class="bar-pair">\n'
        f'                <div class="bar-row"><span class="bar-tag">A</span>'
        f'<div class="bar-track"><div class="bar-fill a" style="width:{width_a:.1f}%">'
        f'</div></div><span class="bar-val">{val_a:.1f}%</span></div>\n'
        f'                <div class="bar-row"><span class="bar-tag">B</span>'
        f'<div class="bar-track"><div class="bar-fill b" style="width:{width_b:.1f}%">'
        f'</div></div><span class="bar-val">{val_b:.1f}%</span></div>\n'
        f'            </div>'
    )


def generate_bar_pair_cell(val_a: float, val_b: float, scale: float = 100.0) -> str:
    """Generate a bar-pair table cell wrapping generate_bar_pair()."""
    bar = generate_bar_pair(val_a, val_b, scale)
    return f'<td class="bar-cell">{bar}</td>'


def generate_delta_cell(val_a: float, val_b: float, mode: str = "pct_change") -> str:
    """Generate a delta value with appropriate CSS class.

    Modes:
        pct_change: (b-a)/a * 100, formatted as +X.X%
        absolute: b-a, formatted with sign
        ratio: b/a, formatted as X.Xx
    """
    if val_a is None or val_b is None:
        return "&mdash;"

    if mode == "pct_change":
        if val_a == 0:
            return "&mdash;"
        delta = (val_b - val_a) / val_a * 100
        sign = "+" if delta > 0 else ("&minus;" if delta < 0 else "")
        css = "v-green" if delta > 0 else ("v-orange" if delta < 0 else "")
        return f'<span class="{css}">{sign}{abs(delta):.1f}%</span>'
    elif mode == "absolute":
        delta = val_b - val_a
        sign = "+" if delta > 0 else ("&minus;" if delta < 0 else "")
        css = "v-green" if delta > 0 else ("v-orange" if delta < 0 else "")
        return f'<span class="{css}">{sign}{abs(delta):.2f}</span>'
    elif mode == "ratio":
        if val_a == 0:
            return "&mdash;"
        ratio = val_b / val_a
        return f"{ratio:.2f}x"
    return "&mdash;"


def generate_stat_card(label: str, value: str, delta: str | None = None,
                       note: str | None = None) -> str:
    """Generate HTML for a stat card.

    Args:
        label: Card label text (uppercase header)
        value: Main value to display
        delta: Optional delta annotation (e.g. "+15%")
        note: Optional detail text below the value
    """
    lines = ['<div class="stat-card">']
    lines.append(f'    <div class="label">{label}</div>')
    lines.append(f'    <div class="value v-dark">{value}</div>')
    if delta:
        lines.append(f'    <div class="detail">{delta}</div>')
    if note:
        lines.append(f'    <div class="detail">{note}</div>')
    lines.append('</div>')
    return "\n".join(lines)


def generate_stat_note(test_name: str, p_value: float, significant: bool,
                       threshold: float) -> str:
    """Generate a significance note HTML snippet.

    Args:
        test_name: Name of the statistical test
        p_value: The p-value
        significant: Whether it passed the significance threshold
        threshold: The threshold used (e.g. Bonferroni-corrected alpha)
    """
    if significant:
        css = "v-green"
        label = "significant"
    else:
        css = ""
        label = "not significant"
    return (f'<span class="{css}">{test_name}: p = {p_value:.4f} '
            f'({label}, &alpha; = {threshold:.4f})</span>')


def _label_case(key: str) -> str:
    """Capitalize a snake_case or lowercase key for display."""
    return key.replace("_", " ").replace("+", "+").title()


def _resolve_row_data(spec: dict, row_key: str, data: dict, config: dict) -> dict:
    """Resolve all column values for a single row."""
    row = {}
    for col in spec["columns"]:
        header = col["header"].format(**config)
        col_type = col.get("type")
        key = col.get("key")
        path = col.get("path")
        numerator = col.get("numerator")
        denominator = col.get("denominator")

        if col_type == "bar_pair":
            path_a = col["path_a"].replace("{row_key}", row_key)
            path_b = col["path_b"].replace("{row_key}", row_key)
            val_a = resolve_path(data, path_a, config) or 0
            val_b = resolve_path(data, path_b, config) or 0
            scale = col.get("scale", 100)
            row[header] = generate_bar_pair(val_a, val_b, scale)
        elif col_type == "delta":
            path_a = col["path_a"].replace("{row_key}", row_key)
            path_b = col["path_b"].replace("{row_key}", row_key)
            val_a = resolve_path(data, path_a, config)
            val_b = resolve_path(data, path_b, config)
            mode = col.get("mode", "pct_change")
            row[header] = generate_delta_cell(val_a, val_b, mode)
        elif col_type == "composite":
            value_path = col["value_path"].replace("{row_key}", row_key)
            count_path = col["count_path"].replace("{row_key}", row_key)
            val = resolve_path(data, value_path, config)
            count = resolve_path(data, count_path, config)
            fmt = col.get("format", "{:.2f}")
            val_str = format_value(val, fmt)
            count_str = f"{count:,}" if isinstance(count, (int, float)) else str(count or 0)
            row[header] = f"{val_str} (n={count_str})"
        elif key == "_label":
            row[header] = _label_case(row_key)
        elif key == "_key":
            row[header] = row_key
        elif numerator and denominator:
            # Computed ratio column
            num_path = numerator.replace("{row_key}", row_key)
            den_path = denominator.replace("{row_key}", row_key)
            num_val = resolve_path(data, num_path, config)
            den_val = resolve_path(data, den_path, config)
            if num_val is not None and den_val is not None and den_val != 0:
                val = num_val / den_val
            else:
                val = None
            fmt = col.get("format", "{:.2f}")
            row[header] = format_value(val, fmt)
        elif path:
            # Path may contain {row_key} for the current row
            resolved_path = path.replace("{row_key}", row_key)
            val = resolve_path(data, resolved_path, config)
            fmt = col.get("format", "{}")
            row[header] = format_value(val, fmt)
        elif "value" in col:
            # Static value from spec
            row[header] = col["value"]
        else:
            row[header] = ""
    return row


def generate_table(spec: dict, data: dict, config: dict) -> str:
    """Produce expansion HTML from a table spec + data.

    Args:
        spec: Table specification dict with keys:
            - title: str - expansion title
            - data_source: str - key into data dict
            - row_source: str - dotted path to iterable within data source
            - row_order: list[str] - ordered row keys
            - columns: list of column defs
            - multi_header: optional list of header group defs
        data: Merged data dict (keyed by data_source name)
        config: Dict with model_a, model_b, display_a, display_b

    Returns:
        HTML string with <!-- title: ... --> header per build_report convention.
    """
    title = spec.get("title", "Details")
    lines = [f"<!-- title: {title} -->"]

    # Build rows
    row_source = spec.get("row_source")
    row_order = spec.get("row_order", [])
    columns = spec["columns"]

    # Dynamic row discovery: auto-discover keys from data
    if spec.get("row_source_keys") and row_source and not row_order:
        source_data = resolve_path(data, row_source, config)
        if isinstance(source_data, dict):
            row_order = list(source_data.keys())

    # Multi-header support (e.g., model-grouped columns)
    multi_header = spec.get("multi_header")

    # Custom row generation mode
    row_gen = spec.get("row_gen")

    # Start table
    lines.append("<table>")

    # Thead
    lines.append("    <thead>")
    if multi_header:
        lines.append("        <tr>")
        for mh in multi_header:
            colspan = mh.get("colspan", 1)
            css = f' class="{mh["class"]}"' if "class" in mh else ""
            if colspan > 1:
                lines.append(f'            <th{css} colspan="{colspan}">{mh["header"]}</th>')
            else:
                lines.append(f'            <th{css}>{mh["header"]}</th>')
        lines.append("        </tr>")

    lines.append("        <tr>")
    for col in columns:
        css_class = col.get("class", "")
        css = f' class="{css_class}"' if css_class else ""
        header = col["header"].format(**config)
        lines.append(f"            <th{css}>{header}</th>")
    lines.append("        </tr>")
    lines.append("    </thead>")

    # Tbody
    lines.append("    <tbody>")

    if row_gen == "stat_tests":
        _generate_stat_test_rows(lines, spec, data, config)
    else:
        for row_key in row_order:
            row = _resolve_row_data(spec, row_key, data, config)
            lines.append("        <tr>")
            for col in columns:
                css_class = col.get("cell_class", "")
                css = f' class="{css_class}"' if css_class else ""
                val = row.get(col["header"].format(**config), "")
                lines.append(f"            <td{css}>{val}</td>")
            lines.append("        </tr>")

    lines.append("    </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def _p_value_class(p: float, bonferroni_sig: bool) -> str:
    """Return CSS class for p-value significance highlighting."""
    if bonferroni_sig:
        return "v-green"
    return ""


def _bonferroni_mark(sig: bool) -> str:
    """Return checkmark for Bonferroni-significant results."""
    if sig:
        return "✓"
    return ""


def _generate_stat_test_rows(lines: list, spec: dict, data: dict, config: dict):
    """Generate rows for stat-test-full-results table.

    Handles the special rowspan grouping by test category.
    """
    test_groups = spec.get("test_groups", [])
    stats_data = data.get(spec.get("data_source", "stats"), {})
    overall = stats_data.get("overall", {})

    for group in test_groups:
        test_type = group["test_type"]
        data_key = group["data_key"]
        fields = group.get("fields")

        test_results = overall.get(data_key, [])
        if fields:
            test_results = [t for t in test_results if t["field"] in fields]

        for i, result in enumerate(test_results):
            lines.append("        <tr>")
            if i == 0:
                lines.append(
                    f'            <td class="label-cell" rowspan="{len(test_results)}">'
                    f'{group["label"]}</td>'
                )

            field = result["field"]
            p_val = result["p_value"]
            bonf = result.get("bonferroni_significant", False)
            p_class = _p_value_class(p_val, bonf)
            p_css = f' class="right mono{" " + p_class if p_class else ""}"'

            # Effect size formatting depends on test type
            if test_type == "mann_whitney":
                effect = f"d = {result['cohens_d']:.4f}"
            elif test_type == "proportions":
                effect = f"h = {result['cohens_h']:.4f}"
            elif test_type == "chi_square":
                effect = f"V = {result['cramers_v']:.4f}"
            else:
                effect = ""

            # Result text
            result_text = _build_result_text(result, test_type, config)

            bonf_mark = _bonferroni_mark(bonf)
            bonf_css = ' class="right v-green"' if bonf else ' class="right"'

            lines.append(f'            <td class="label-cell">{field}</td>')
            lines.append(f'            <td{p_css}>{p_val:.6f}</td>')
            lines.append(f'            <td class="right mono">{effect}</td>')
            lines.append(f'            <td{bonf_css}>{bonf_mark}</td>')
            lines.append(f"            <td>{result_text}</td>")
            lines.append("        </tr>")


def _build_result_text(result: dict, test_type: str, config: dict) -> str:
    """Build the descriptive result text for a stat test row."""
    p = result["p_value"]
    bonf = result.get("bonferroni_significant", False)
    field = result["field"]

    if test_type == "mann_whitney":
        d = result["cohens_d"]
        if not result.get("significant_p05", False):
            return "No significant difference"
        direction = "higher" if d < 0 else "lower"
        model_label = f"Opus {config.get('display_b', '4.6')}"
        if bonf:
            return f"{model_label} {direction} (Bonferroni significant)"
        return f"{model_label} {direction} (p < 0.05)"

    elif test_type == "proportions":
        h = result["cohens_h"]
        if not result.get("significant_p05", False):
            return "No significant difference"
        direction = "higher" if h < 0 else "lower"
        model_label = f"Opus {config.get('display_b', '4.6')}"
        if bonf:
            return f"{model_label} {direction} (Bonferroni significant)"
        return f"{model_label} {direction} (p < 0.05)"

    elif test_type == "chi_square":
        v = result["cramers_v"]
        low_warn = result.get("low_expected_warning", False)
        warn_html = " <em>(low cell counts)</em>" if low_warn else ""
        if not result.get("significant_p05", False):
            return f"No significant difference (V = {v:.4f}){warn_html}"
        return f"Distribution differs (p < 0.05, V = {v:.4f}){warn_html}"

    return ""


def generate_expansion(table_spec: dict, data: dict, config: dict,
                       existing_prose: dict | None = None) -> str:
    """Generate a complete expansion fragment: table + prose.

    Args:
        table_spec: Table specification from the section spec
        data: Merged data dict
        config: Model config
        existing_prose: Dict with optional 'preamble' and 'postscript' text

    Returns:
        Complete HTML fragment with title comment and content.
    """
    table_html = generate_table(table_spec, data, config)

    # Extract title line and table body
    title_line = ""
    body = table_html
    if table_html.startswith("<!-- title:"):
        first_nl = table_html.index("\n")
        title_line = table_html[:first_nl]
        body = table_html[first_nl + 1:]

    parts = [title_line] if title_line else []

    if existing_prose and existing_prose.get("preamble"):
        parts.append(existing_prose["preamble"])

    parts.append(body)

    if existing_prose and existing_prose.get("postscript"):
        parts.append(existing_prose["postscript"])

    return "\n".join(parts) + "\n"
