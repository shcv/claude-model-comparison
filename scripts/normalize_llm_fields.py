#!/usr/bin/env python3
"""
Normalize LLM Free-Text Fields

Reclassifies free-text LLM analysis fields into consistent categories:
- execution_quality → excellent/good/adequate/poor/failed
- work_category → investigation/directed_impl/creative_impl/verification/correction
- user_sentiment → satisfied/neutral/dissatisfied/ambiguous

Uses regex bucketing first, falls back to LLM reclassification for ambiguous cases.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models


# Model name mapping (fallback display names)
MODEL_NAMES = {
    'opus-4-5': 'Opus 4.5',
    'opus-4-6': 'Opus 4.6',
}

# --- Normalization rules ---

EXECUTION_QUALITY_PATTERNS = {
    'excellent': [
        r'\bexcellent\b', r'\bexceptional\b', r'\boutstanding\b',
        r'\bperfect(ly)?\b', r'\bflawless\b', r'\bsuperb\b',
        r'\bvery well\b', r'\bhigh quality\b', r'\bexpertly\b',
    ],
    'good': [
        r'\bgood\b', r'\bwell[\s-]executed\b', r'\bsolid\b',
        r'\bcompetent\b', r'\beffective(ly)?\b', r'\bstrong\b',
        r'\bsuccessful(ly)?\b', r'\bcapable\b', r'\bproficient\b',
        r'\bsmooth(ly)?\b', r'\befficient(ly)?\b',
    ],
    'adequate': [
        r'\badequate\b', r'\bacceptable\b', r'\breasonable\b',
        r'\bsatisfactor(y|ily)\b', r'\bokay\b', r'\bdecent\b',
        r'\bmoderate(ly)?\b', r'\bfair(ly)?\b', r'\bsufficient\b',
        r'\bbasic(ally)?\b', r'\bstandard\b',
    ],
    'poor': [
        r'\bpoor(ly)?\b', r'\binadequate\b', r'\bsubpar\b',
        r'\bweak\b', r'\bstrugg', r'\binefficient\b',
        r'\bproblematic\b', r'\bsuboptimal\b', r'\bmisalign',
        r'\bmissed\b', r'\bincorrect\b',
    ],
    'failed': [
        r'\bfail(ed|ure)?\b', r'\bunsuccessful\b', r'\bdid not complete\b',
        r'\bcould not\b', r'\bunable to\b', r'\bbroken\b',
    ],
}

WORK_CATEGORY_PATTERNS = {
    'investigation': [
        r'\binvestigat', r'\bexplor', r'\bresearch', r'\banalysi[sz]',
        r'\breview', r'\bunderstand', r'\bexamin', r'\bdiagnos',
        r'\bdebug', r'\blook(ed|ing)?\s+(at|into)\b', r'\bfind(ing)?\b',
        r'\bread(ing)?\s+(and|the|through)\b', r'\bsearch',
    ],
    'directed_impl': [
        r'\bimplement(ed|ing)?\b.*\b(requested|asked|specified)\b',
        r'\bfollow(ed|ing)?\s+(the\s+)?instruction',
        r'\bdirect(ed|ly)\s+implement',
        r'\bsimple\s+(change|edit|fix|update|modification)',
        r'\bstraightforward\b.*\b(implement|change|update)',
        r'\bminor\s+(change|edit|fix|update|tweak)',
    ],
    'creative_impl': [
        r'\bcreativ', r'\bdesign(ed|ing)?\b', r'\barchitect',
        r'\bgreenfield\b', r'\bfrom\s+scratch\b', r'\bnew\s+(feature|system|module)',
        r'\bbuild(ing)?\s+(a|an|the|new)\b', r'\bcomplex\s+implement',
        r'\bsubstantial\b', r'\blarge[\s-]scale\b',
    ],
    'verification': [
        r'\bverif(y|ied|ication)\b', r'\btest(ed|ing)?\b',
        r'\bvalidat', r'\bcheck(ed|ing)?\b', r'\bconfirm',
        r'\baudit', r'\binspect',
    ],
    'correction': [
        r'\bcorrect(ed|ion|ing)?\b', r'\bfix(ed|ing)?\b',
        r'\brepair', r'\bresolv(e|ed|ing)\b', r'\bpatch',
        r'\bbug\s*fix', r'\baddress(ed|ing)?\b.*\b(issue|error|bug)',
        r'\brefactor',
    ],
}

SENTIMENT_PATTERNS = {
    'satisfied': [
        r'\bsatisf(ied|action)\b', r'\bpositive\b', r'\bhappy\b',
        r'\bpleased\b', r'\bapprov', r'\bappreciat',
        r'\bthanks?\b', r'\bgrateful\b', r'\bimpressed\b',
        r'\bcontent\b', r'\bdelighted\b',
    ],
    'neutral': [
        r'\bneutral\b', r'\bunclear\b', r'\bambiguous\b',
        r'\bno\s+(clear\s+)?sentiment\b', r'\bsession\s+end',
        r'\bno\s+explicit\b', r'\bno\s+direct\b',
        r'\bmixed\b', r'\bindeterminate\b',
    ],
    'dissatisfied': [
        r'\bdissatisf', r'\bnegative\b', r'\bfrustrat',
        r'\bunhappy\b', r'\bdispleas', r'\bdisappoint',
        r'\bcorrect(ion|ive)\b', r'\bwrong\b',
    ],
    'ambiguous': [
        r'\biterativ', r'\bcollaborativ', r'\brefinement\b',
        r'\bcontinuing\b', r'\bfollow[\s-]up\b', r'\bongoing\b',
    ],
}


def regex_classify(text: str, patterns: dict[str, list[str]]) -> Optional[str]:
    """Classify text using regex patterns. Returns category or None if ambiguous."""
    text_lower = text.lower()
    matches = {}

    for category, category_patterns in patterns.items():
        score = 0
        for pattern in category_patterns:
            if re.search(pattern, text_lower):
                score += 1
        if score > 0:
            matches[category] = score

    if not matches:
        return None

    # Return top match if it's clearly dominant
    sorted_matches = sorted(matches.items(), key=lambda x: -x[1])
    if len(sorted_matches) == 1:
        return sorted_matches[0][0]

    # If top match has at least 2x the score of second, use it
    if sorted_matches[0][1] >= 2 * sorted_matches[1][1]:
        return sorted_matches[0][0]

    # Ambiguous — first match wins if scores are equal
    return sorted_matches[0][0]


def llm_classify(text: str, field_name: str, categories: list[str]) -> Optional[str]:
    """Use LLM to classify ambiguous text."""
    prompt = f"""Classify the following text into exactly one of these categories: {', '.join(categories)}

Text: "{text}"

Field: {field_name}

Return ONLY the category name, nothing else."""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "haiku", "--output-format", "text"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            response = result.stdout.strip().lower()
            # Find the closest matching category
            for cat in categories:
                if cat in response:
                    return cat
        return None
    except Exception:
        return None


def normalize_analysis(analysis: dict, use_llm: bool = False) -> dict:
    """Normalize an individual task analysis entry."""
    # Skip if already normalized
    if analysis.get('normalized_execution_quality'):
        return analysis

    result = dict(analysis)

    # --- execution_quality ---
    eq = analysis.get('execution_quality', '')
    normalized_eq = regex_classify(eq, EXECUTION_QUALITY_PATTERNS)
    if normalized_eq is None and use_llm:
        normalized_eq = llm_classify(eq, 'execution_quality',
                                     ['excellent', 'good', 'adequate', 'poor', 'failed'])
    result['normalized_execution_quality'] = normalized_eq or 'adequate'

    # --- work_category ---
    wc = analysis.get('work_category', '')
    normalized_wc = regex_classify(wc, WORK_CATEGORY_PATTERNS)
    if normalized_wc is None and use_llm:
        normalized_wc = llm_classify(wc, 'work_category',
                                     ['investigation', 'directed_impl', 'creative_impl',
                                      'verification', 'correction'])
    result['normalized_work_category'] = normalized_wc or 'directed_impl'

    # --- user_sentiment ---
    us = analysis.get('user_sentiment', '')
    normalized_us = regex_classify(us, SENTIMENT_PATTERNS)
    if normalized_us is None and use_llm:
        normalized_us = llm_classify(us, 'user_sentiment',
                                     ['satisfied', 'neutral', 'dissatisfied', 'ambiguous'])
    result['normalized_user_sentiment'] = normalized_us or 'neutral'

    return result


def print_stats(analyses: list[dict], model: str, field: str):
    """Print distribution of a normalized field."""
    dist = {}
    source = {'regex': 0, 'llm': 0, 'default': 0}

    for a in analyses:
        val = a.get(f'normalized_{field}', 'unknown')
        dist[val] = dist.get(val, 0) + 1

    total = len(analyses)
    print(f"\n  {field}:")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        bar = '#' * int(pct / 2)
        print(f"    {cat:<20} {count:3d} ({pct:5.1f}%) {bar}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Normalize LLM analysis fields')
    parser.add_argument('--analysis-dir', type=Path, default=Path('analysis'))
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for ambiguous cases (requires claude CLI)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats without modifying files')
    args = parser.parse_args()

    print("=" * 60)
    print("LLM FIELD NORMALIZATION")
    print("=" * 60)

    for model in discover_models(args.analysis_dir, prefix="llm-analysis"):
        input_file = args.analysis_dir / f'llm-analysis-{model}.json'
        if not input_file.exists():
            print(f"\nSkipping {model}: {input_file} not found")
            continue

        with open(input_file) as f:
            analyses = json.load(f)

        print(f"\n{MODEL_NAMES.get(model, model)} ({len(analyses)} tasks)")
        print("-" * 40)

        # Track normalization statistics
        regex_count = 0
        llm_count = 0
        default_count = 0

        normalized = []
        for a in analyses:
            # Check if already normalized
            if a.get('normalized_execution_quality'):
                normalized.append(a)
                continue

            result = normalize_analysis(a, use_llm=args.use_llm)
            normalized.append(result)

        # Print distributions
        for field in ['execution_quality', 'work_category', 'user_sentiment']:
            print_stats(normalized, model, field)

        # Count how many had regex matches vs defaults
        for a in normalized:
            for field in ['execution_quality', 'work_category', 'user_sentiment']:
                original = a.get(field, '')
                normalized_val = a.get(f'normalized_{field}', '')
                if regex_classify(original, {
                    'execution_quality': EXECUTION_QUALITY_PATTERNS,
                    'work_category': WORK_CATEGORY_PATTERNS,
                    'user_sentiment': SENTIMENT_PATTERNS,
                }.get(field, {})):
                    regex_count += 1
                else:
                    default_count += 1

        total_fields = len(normalized) * 3
        print(f"\n  Regex-matched: {regex_count}/{total_fields} ({100*regex_count/total_fields:.1f}%)")
        print(f"  Default: {default_count}/{total_fields} ({100*default_count/total_fields:.1f}%)")

        if not args.dry_run:
            with open(input_file, 'w') as f:
                json.dump(normalized, f, indent=2)
            print(f"\n  Updated {input_file}")
        else:
            print(f"\n  [DRY RUN] Would update {input_file}")


if __name__ == '__main__':
    main()
