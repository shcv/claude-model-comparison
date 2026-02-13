#!/usr/bin/env python3
"""Build report.html from template + terms + expansions.

Reads the source template (report/report.html), processes it:
1. Wraps first occurrences of glossary terms with tooltip markup
2. Inserts expansion fragments at <!-- expand: name --> markers
3. Writes the assembled output to dist/public/report.html

Expansion fragments are cached in report/expansions/. A manifest
tracks section hashes for invalidation.

Usage:
    python scripts/build_report.py --dir comparisons/opus-4.5-vs-4.6
    python scripts/build_report.py --dir comparisons/opus-4.5-vs-4.6 --check-stale
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def load_terms(report_dir):
    """Load term definitions from terms.json."""
    terms_path = report_dir / "terms.json"
    if not terms_path.exists():
        return {}
    with open(terms_path) as f:
        return json.load(f)


def load_manifest(report_dir):
    """Load the expansion manifest (section hash -> expansion mapping)."""
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(report_dir, manifest):
    """Write the expansion manifest."""
    manifest_path = report_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def hash_section(content):
    """Compute a short hash of section content for invalidation."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_sections(html):
    """Extract named sections from the HTML for hashing.

    Returns dict of section_id -> section content (between <section> tags).
    """
    sections = {}
    pattern = re.compile(
        r'<section\s+id="([^"]+)">(.*?)</section>',
        re.DOTALL
    )
    for m in pattern.finditer(html):
        sections[m.group(1)] = m.group(2)
    return sections


def apply_terms(html, terms):
    """Wrap first occurrence of each term with tooltip markup.

    Only processes text inside <body>, skips content inside HTML tags,
    existing tooltips, <code>, <script>, <style>, and <h1>-<h6>.
    Terms are matched case-insensitively but preserve original case.
    """
    if not terms:
        return html

    # Split at body to only process body content
    body_match = re.search(r'(<body[^>]*>)(.*)(</body>)', html, re.DOTALL)
    if not body_match:
        return html

    pre_body = html[:body_match.start(2)]
    body = body_match.group(2)
    post_body = html[body_match.end(2):]

    # Sort terms by length (longest first) to avoid partial matches
    sorted_terms = sorted(terms.keys(), key=len, reverse=True)

    for term in sorted_terms:
        definition = terms[term]
        # Escape for use in HTML attribute
        escaped_def = (definition
                       .replace("&", "&amp;")
                       .replace('"', "&quot;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))

        # Build pattern that matches the term in text content only.
        # Use word boundaries and case-insensitive matching.
        # Skip matches inside HTML tags, <code>, <style>, <script>, headings,
        # or existing data-def spans.
        term_pattern = re.compile(
            r'(?<![<\w])(' + re.escape(term) + r')(?![^<]*>)(?!.*?</span>)',
            re.IGNORECASE
        )

        # Find all matches, but only replace the first one that's in
        # actual text content (not inside a tag or special element)
        parts = []
        last_end = 0
        replaced = False
        skip_tags = re.compile(
            r'<(code|style|script|h[1-6]|span[^>]*data-def)[^>]*>.*?</\1>',
            re.DOTALL | re.IGNORECASE
        )
        # Build a set of character ranges to skip
        skip_ranges = []
        for sm in skip_tags.finditer(body):
            skip_ranges.append((sm.start(), sm.end()))
        # Also skip inside HTML tags
        for sm in re.finditer(r'<[^>]+>', body):
            skip_ranges.append((sm.start(), sm.end()))

        for m in term_pattern.finditer(body):
            if replaced:
                break
            # Check if this match falls inside a skip range
            in_skip = False
            for (s, e) in skip_ranges:
                if s <= m.start() < e:
                    in_skip = True
                    break
            if in_skip:
                continue

            # Replace this occurrence
            parts.append(body[last_end:m.start()])
            original_text = m.group(1)
            parts.append(
                f'<span class="term" data-def="{escaped_def}">'
                f'{original_text}</span>'
            )
            last_end = m.end()
            replaced = True

        if replaced:
            parts.append(body[last_end:])
            body = "".join(parts)

    return pre_body + body + post_body


def insert_expansions(html, report_dir):
    """Replace <!-- expand: name --> markers with expansion content.

    Expansion fragments are loaded from report/expansions/{name}.html.
    Each is wrapped in a <details class="expansion"> element.
    """
    expansions_dir = report_dir / "expansions"

    def replace_marker(m):
        name = m.group(1).strip()
        frag_path = expansions_dir / f"{name}.html"
        if not frag_path.exists():
            return f'<!-- expand: {name} (missing) -->'
        content = frag_path.read_text()
        # Extract title from first line if it starts with <!-- title: ... -->
        title = "Show details"
        title_match = re.match(r'^<!--\s*title:\s*(.+?)\s*-->\n?', content)
        if title_match:
            title = title_match.group(1)
            content = content[title_match.end():]
        # Wrap <table> elements in a scroll container so they don't
        # spill out, while keeping the expansion body overflow-visible
        # for tooltips.
        content = re.sub(
            r'(<table\b.*?</table>)',
            r'<div class="table-scroll">\1</div>',
            content,
            flags=re.DOTALL,
        )
        return (
            f'<details class="expansion">\n'
            f'    <summary>{title}</summary>\n'
            f'    <div class="expansion-body">\n'
            f'        {content}\n'
            f'    </div>\n'
            f'</details>'
        )

    pattern = re.compile(r'<!--\s*expand:\s*(\S+)\s*-->')
    return pattern.sub(replace_marker, html)


def check_stale(html, report_dir, manifest):
    """Check which expansions are stale (section content has changed).

    Returns list of (expansion_name, section_id, reason).
    """
    sections = extract_sections(html)
    stale = []

    for name, entry in manifest.items():
        section_id = entry.get("section")
        old_hash = entry.get("hash")
        if section_id and section_id in sections:
            current_hash = hash_section(sections[section_id])
            if current_hash != old_hash:
                stale.append((name, section_id, "section content changed"))
        elif section_id:
            stale.append((name, section_id, "section not found"))

    return stale


def inject_css(html):
    """Add tooltip and expansion CSS if not already present."""
    marker = "/* -- Terms & Expansions -- */"
    if marker in html:
        return html

    css = f"""
        {marker}
        .term {{
            background: linear-gradient(to bottom, transparent 60%, #6a9bcc18 60%);
            border-bottom: 1.5px dotted var(--blue);
            cursor: help;
            position: relative;
            padding: 0 1px;
        }}
        .term:hover::after,
        .term:focus::after {{
            content: attr(data-def);
            position: absolute;
            bottom: calc(100% + 6px);
            left: 50%;
            transform: translateX(-50%);
            background: var(--dark);
            color: var(--light);
            font-family: 'Poppins', Arial, sans-serif;
            font-size: 0.72rem;
            font-style: normal;
            font-weight: 400;
            line-height: 1.45;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            width: max-content;
            max-width: 320px;
            z-index: 100;
            pointer-events: none;
            white-space: normal;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .term:hover::before,
        .term:focus::before {{
            content: '';
            position: absolute;
            bottom: calc(100% + 2px);
            left: 50%;
            transform: translateX(-50%);
            border: 5px solid transparent;
            border-top-color: var(--dark);
            z-index: 100;
            pointer-events: none;
        }}

        details.expansion {{
            margin: 1rem 0;
            border: 1px solid var(--light-gray);
            border-radius: 6px;
        }}
        details.expansion summary {{
            font-family: 'Poppins', Arial, sans-serif;
            font-size: 0.75rem;
            font-weight: 500;
            letter-spacing: 0.03em;
            color: var(--blue);
            padding: 0.5rem 1rem;
            cursor: pointer;
            user-select: none;
            list-style: none;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        details.expansion summary::-webkit-details-marker {{
            display: none;
        }}
        details.expansion summary::before {{
            content: '\\25B6';
            font-size: 0.6rem;
            transition: transform 0.15s ease;
        }}
        details.expansion[open] summary::before {{
            transform: rotate(90deg);
        }}
        details.expansion summary:hover {{
            background: var(--blue-light);
            border-radius: 6px 6px 0 0;
        }}
        details.expansion .expansion-body {{
            padding: 0.75rem 1rem 1rem;
            border-top: 1px solid var(--light-gray);
            font-size: 0.88rem;
        }}
        details.expansion .expansion-body .table-scroll {{
            overflow-x: auto;
        }}
        details.expansion .expansion-body table {{
            font-size: 0.82rem;
        }}
"""

    # Insert before closing </style>
    return html.replace("</style>", css + "\n        </style>", 1)


def build(report_dir, output_path, check_only=False):
    """Run the full build pipeline."""
    template_path = report_dir / "report.html"
    if not template_path.exists():
        print(f"Error: template not found at {template_path}", file=sys.stderr)
        return False

    html = template_path.read_text()
    manifest = load_manifest(report_dir)
    terms = load_terms(report_dir)

    if check_only:
        stale = check_stale(html, report_dir, manifest)
        if stale:
            print("Stale expansions:")
            for name, section_id, reason in stale:
                print(f"  {name} (section: {section_id}): {reason}")
        else:
            print("All expansions are up to date.")
        return len(stale) == 0

    # 1. Update manifest with section hashes from the source template
    #    (before any transforms, so --check-stale compares against the same basis)
    source_sections = extract_sections(html)
    for name in manifest:
        section_id = manifest[name].get("section")
        if section_id and section_id in source_sections:
            manifest[name]["hash"] = hash_section(source_sections[section_id])
    save_manifest(report_dir, manifest)

    # 2. Strip variable sentinels (keep resolved values, remove markers)
    html = re.sub(r'<!-- var: .+? -->(.*?)<!-- /var -->', r'\1', html)

    # 3. Inject CSS for terms and expansions
    html = inject_css(html)

    # 4. Insert expansion fragments
    html = insert_expansions(html, report_dir)

    # 5. Wrap terms with tooltip markup
    html = apply_terms(html, terms)

    # 6. Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    n_terms = len(terms)
    n_expansions = len(list((report_dir / "expansions").glob("*.html")))
    print(f"Built {output_path}")
    print(f"  Terms: {n_terms} definitions")
    print(f"  Expansions: {n_expansions} fragments")
    return True


def check_stale_vars(html):
    """Scan built report for unresolved {{...}} template variables.

    Returns list of unresolved variable names.
    """
    pattern = re.compile(r'\{\{(.+?)\}\}')
    return pattern.findall(html)


def main():
    parser = argparse.ArgumentParser(description="Build report from template")
    parser.add_argument("--dir", type=Path, required=True,
                        help="Comparison directory (e.g., comparisons/opus-4.5-vs-4.6)")
    parser.add_argument("--check-stale", action="store_true",
                        help="Only check for stale expansions, don't build")
    parser.add_argument("--check-stale-vars", action="store_true",
                        help="After building, check for unresolved template variables")
    args = parser.parse_args()

    comparison_dir = args.dir.resolve()
    report_dir = comparison_dir / "report"
    output_path = comparison_dir / "dist" / "public" / "report.html"

    success = build(report_dir, output_path, check_only=args.check_stale)
    if not success:
        sys.exit(1)

    if args.check_stale_vars and output_path.exists():
        built_html = output_path.read_text()
        unresolved = check_stale_vars(built_html)
        if unresolved:
            print(f"\nUnresolved template variables ({len(unresolved)}):")
            for var in sorted(set(unresolved)):
                print(f"  {{{{{var}}}}}")
            sys.exit(1)
        else:
            print("All template variables resolved.")


if __name__ == "__main__":
    main()
