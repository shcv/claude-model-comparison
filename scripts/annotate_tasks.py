#!/usr/bin/env python3
"""Unified task annotation pipeline.

Replaces analyze_tasks_llm.py + normalize_llm_fields.py with a single
signal-then-aggregate architecture:

  canonical task -> keyword_signals (pure computation)
                 -> edit_signals (pure computation)
                 -> llm_signals (cached Haiku call)
                 -> aggregate -> composite scores

Output: analysis/tasks-annotated-{model}.json

Also writes analysis/llm-analysis-{model}.json for backward compatibility
with stat_tests.py, planning_analysis.py, analyze_compaction.py, etc.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import discover_models

# Bump this to invalidate all LLM caches
PROMPT_VERSION = "v2"

# --- Keyword signal patterns ---

POSITIVE_PATTERNS = [
    r'\bthanks?\b', r'\bthank you\b', r'\bgreat\b', r'\bperfect\b',
    r'\bexcellent\b', r'\bawesome\b', r'\blooks? good\b', r'\bnice\b',
    r'\bwell done\b', r'\bexactly\b', r'\blove it\b', r'\bbeautiful\b',
    r'\bwonderful\b', r'\bgood job\b', r'\bimpressive\b',
]

NEGATIVE_PATTERNS = [
    r'\bno[\s,]', r'\bwrong\b', r'\bincorrect\b', r'\bthat\'s not\b',
    r'\bdon\'t\b', r'\bshouldn\'t\b', r'\binstead\b', r'\bactually\b',
    r'\bplease (fix|change|undo|revert|remove)\b', r'\bnot what i\b',
    r'\bstop\b', r'\bwhy did you\b', r'\bundo\b', r'\brevert\b',
]

CONTINUATION_PATTERNS = [
    r'\bnow\b', r'\bnext\b', r'\balso\b', r'\bcan you\b',
    r'\bplease\b', r'\bwhat about\b', r'\bhow about\b',
]

# Patterns indicating user corrections
USER_CORRECTION_PATTERNS = [
    r'\bno[\s,]\s*(change|make|use|set|do)\b',
    r'\b(change|make|set)\s+it\s+to\b',
    r'\binstead\s+(of|use)\b',
    r'\bshould\s+(be|have)\b',
    r'\bthat\'s\s+not\s+(right|correct|what)\b',
    r'\bwrong\b', r'\bincorrect\b',
    r'\bundo\b', r'\brevert\b',
    r'\bfix\s+(the|this|that|it)\b',
]

# --- Normalization patterns (from normalize_llm_fields.py) ---

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


# --- LLM prompt template ---

LLM_PROMPT_TEMPLATE = """You are analyzing a Claude Code task interaction to provide a rich qualitative summary.

## Task Data

**Task ID:** {task_id}
**Model:** {model}
**Project:** {project_path}

### User's Initial Request:
"{user_prompt}"

### Work Done:
- **Tool sequence:** {tool_sequence}
- **Files:** {files_touched} touched ({files_written} written, {files_edited} edited, {files_read} read)
- **Tool calls:** {tool_count}
- **Duration:** {duration:.0f} seconds
- **Key tools:** {key_tools}

### User's Next Message:
"{next_user_message}"

### Outcome Category (heuristic):
{outcome_category} - "{outcome_evidence}"

### Edit Activity:
- **Edit events:** {edit_count}
- **Errors encountered:** {error_count}

## Analysis Required

Return a JSON object with exactly these fields:

{{
  "work_category": "1-2 sentence characterization of what kind of work this really was",
  "execution_quality": "1-2 sentence assessment of how well the task was executed",
  "user_sentiment": "inferred user sentiment based on next_user_message and outcome",
  "sentiment_confidence": "low|medium|high",
  "follow_up_pattern": "what happened next or what would be expected to happen",
  "autonomy_level": "high|medium|low - how independently did the agent work?",
  "task_completion": "complete|partial|interrupted|failed - did the agent finish the requested work?",
  "scope_management": "focused|appropriate|expanded|over_engineered - did agent stay on track or add unnecessary work?",
  "communication_quality": "clear|adequate|verbose|unclear - how well did agent explain its work?",
  "error_recovery": "none_needed|recovered|struggled|failed - how did agent handle any errors encountered?",
  "iteration_required": "one_shot|minor|significant - how much back-and-forth was needed?",
  "alignment_score": "1-5 integer - how well did agent match the user's actual intent? (5=perfect alignment)",
  "summary": "2-3 sentence overall summary combining all aspects"
}}

Consider:
- Empty next_user_message with session_end often indicates satisfaction for simple tasks
- "sounds good" type messages indicate approval of prior work
- Technical corrections may be refinement, not dissatisfaction
- Questions in follow-up may indicate confusion or normal iteration
- scope_management: "focused"=minimal, "appropriate"=matched request, "expanded"=added useful extras, "over_engineered"=unnecessary complexity
- alignment_score: 5=exactly what user wanted, 4=minor misalignment, 3=partially aligned, 2=significant gap, 1=missed intent

Return ONLY the JSON object, no other text."""


def regex_classify(text, patterns):
    """Classify text using regex patterns. Returns best category or None."""
    text_lower = text.lower()
    matches = {}
    for category, category_patterns in patterns.items():
        score = sum(1 for p in category_patterns if re.search(p, text_lower))
        if score > 0:
            matches[category] = score
    if not matches:
        return None
    sorted_matches = sorted(matches.items(), key=lambda x: -x[1])
    return sorted_matches[0][0]


# --- Signal extractors ---

def compute_keyword_signals(task):
    """Compute keyword-based signals from next_user_message and outcome fields.

    Returns dict with outcome_category (already present), outcome_confidence,
    and keyword match details.
    """
    next_msg = (task.get('next_user_message') or '').lower()
    outcome = task.get('outcome_category', 'unknown')

    positive_hits = sum(1 for p in POSITIVE_PATTERNS if re.search(p, next_msg))
    negative_hits = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, next_msg))
    continuation_hits = sum(1 for p in CONTINUATION_PATTERNS if re.search(p, next_msg))

    total_hits = positive_hits + negative_hits + continuation_hits
    if total_hits == 0:
        confidence = 'low'
    elif total_hits <= 2:
        confidence = 'medium'
    else:
        confidence = 'high'

    return {
        'outcome_category': outcome,
        'outcome_confidence': confidence,
        'positive_hits': positive_hits,
        'negative_hits': negative_hits,
        'continuation_hits': continuation_hits,
    }


def compute_edit_signals(task):
    """Compute edit-based signals from edit_events and errors.

    Returns dict with self_corrections, error_recoveries, user_corrections,
    rewrite_rate, max_chain_depth.
    """
    edit_events = task.get('edit_events', [])
    errors = task.get('errors', [])
    next_msg = task.get('next_user_message') or ''

    # Count user corrections from next_user_message patterns
    user_corrections = sum(
        1 for p in USER_CORRECTION_PATTERNS if re.search(p, next_msg, re.IGNORECASE)
    )

    if not edit_events:
        return {
            'self_corrections': 0,
            'error_recoveries': 0,
            'user_corrections': user_corrections,
            'rewrite_rate': 0.0,
            'max_chain_depth': 0,
        }

    # Track edits by file for self-correction detection
    edits_by_file = {}
    self_corrections = 0
    rewrites = 0

    for i, evt in enumerate(edit_events):
        fp = evt.get('file_path', '')
        old_str = evt.get('old_string', '')
        new_str = evt.get('new_string', '')

        if fp not in edits_by_file:
            edits_by_file[fp] = []

        # Check for self-correction: editing content that was recently written
        for prev in edits_by_file[fp]:
            prev_new = prev.get('new_string', '')
            # If old_string overlaps with a previous new_string, it's a self-correction
            if prev_new and old_str and prev_new in old_str or old_str in prev_new:
                self_corrections += 1
                rewrites += 1
                break

        edits_by_file[fp].append(evt)

    # Error recoveries: edit events that follow an error
    error_indices = set()
    for err in errors:
        idx = err.get('msg_index', -1)
        if idx >= 0:
            error_indices.add(idx)

    error_recoveries = 0
    for evt in edit_events:
        # Heuristic: if the edit's msg_index is close to an error index
        evt_idx = evt.get('msg_index', -1)
        if evt_idx >= 0:
            for err_idx in error_indices:
                if 0 < evt_idx - err_idx <= 10:
                    error_recoveries += 1
                    break

    # Rewrite rate
    rewrite_rate = rewrites / len(edit_events) if edit_events else 0.0

    # Max chain depth: longest sequence of edits to same file
    max_chain = max(len(edits) for edits in edits_by_file.values()) if edits_by_file else 0

    return {
        'self_corrections': self_corrections,
        'error_recoveries': error_recoveries,
        'user_corrections': user_corrections,
        'rewrite_rate': round(rewrite_rate, 3),
        'max_chain_depth': max_chain,
    }


def build_llm_prompt(task):
    """Build the LLM analysis prompt for a task."""
    tool_calls = task.get('tool_calls', [])
    tool_names = [t.get('name', 'unknown') for t in tool_calls if isinstance(t, dict)]
    unique_tools = list(dict.fromkeys(tool_names))[:5]
    key_tools = ', '.join(unique_tools) if unique_tools else 'none'

    return LLM_PROMPT_TEMPLATE.format(
        task_id=task.get('task_id', 'unknown'),
        model=task.get('model', 'unknown'),
        project_path=task.get('project_path', 'unknown'),
        user_prompt=task.get('user_prompt', '')[:500],
        tool_sequence=task.get('tool_sequence', 'none'),
        files_touched=task.get('total_files_touched', 0),
        files_written=len(task.get('files_written', [])),
        files_edited=len(task.get('files_edited', [])),
        files_read=len(task.get('files_read', [])),
        tool_count=len(tool_calls),
        duration=task.get('duration_seconds', 0),
        key_tools=key_tools,
        next_user_message=task.get('next_user_message', '(session ended)')[:300],
        outcome_category=task.get('outcome_category', 'unknown'),
        outcome_evidence=task.get('outcome_evidence', ''),
        edit_count=len(task.get('edit_events', [])),
        error_count=len(task.get('errors', [])),
    )


def get_prompt_hash(prompt):
    """SHA-256 hash of prompt content + version for cache keying."""
    content = PROMPT_VERSION + "\n" + prompt
    return hashlib.sha256(content.encode()).hexdigest()


def call_llm(prompt, llm_model="haiku"):
    """Call Claude via SDK and return parsed JSON dict, or None on failure."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", llm_model, "--output-format", "text"],
            capture_output=True, text=True, timeout=60,
            cwd="/tmp",
        )
        if result.returncode != 0:
            print(f"  LLM error: {result.stderr[:200]}", file=sys.stderr)
            return None

        response = result.stdout.strip()
        start = response.find('{')
        end = response.rfind('}') + 1
        if start < 0 or end <= start:
            print(f"  No JSON found in LLM response", file=sys.stderr)
            return None

        return json.loads(response[start:end])

    except subprocess.TimeoutExpired:
        print(f"  LLM timeout", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None


def normalize_llm_fields(llm_data):
    """Normalize free-text LLM fields into consistent categories.

    Returns dict with normalized_execution_quality, normalized_work_category,
    normalized_user_sentiment.
    """
    eq = llm_data.get('execution_quality', '')
    wc = llm_data.get('work_category', '')
    us = llm_data.get('user_sentiment', '')

    return {
        'normalized_execution_quality': regex_classify(eq, EXECUTION_QUALITY_PATTERNS) or 'adequate',
        'normalized_work_category': regex_classify(wc, WORK_CATEGORY_PATTERNS) or 'directed_impl',
        'normalized_user_sentiment': regex_classify(us, SENTIMENT_PATTERNS) or 'neutral',
    }


def compute_llm_signals(task, cache_dir, llm_model, force):
    """Get LLM signals for a task, using cache when available.

    Returns dict with all LLM fields + normalized versions, or None on failure.
    """
    prompt = build_llm_prompt(task)
    prompt_hash = get_prompt_hash(prompt)
    task_id = task.get('task_id', 'unknown')

    # Check cache
    if cache_dir and not force:
        cache_path = cache_dir / f"{task_id}_{prompt_hash[:8]}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass

    # Call LLM
    raw = call_llm(prompt, llm_model)
    if raw is None:
        return None

    # Parse alignment_score as int
    alignment = raw.get('alignment_score', 3)
    if isinstance(alignment, str):
        try:
            alignment = int(alignment)
        except ValueError:
            alignment = 3

    llm_signals = {
        'user_sentiment': raw.get('user_sentiment', ''),
        'sentiment_confidence': raw.get('sentiment_confidence', 'low'),
        'execution_quality': raw.get('execution_quality', ''),
        'work_category': raw.get('work_category', ''),
        'scope_management': raw.get('scope_management', 'appropriate'),
        'communication_quality': raw.get('communication_quality', 'adequate'),
        'error_recovery': raw.get('error_recovery', 'none_needed'),
        'iteration_required': raw.get('iteration_required', 'one_shot'),
        'task_completion': raw.get('task_completion', 'complete'),
        'alignment_score': alignment,
        'summary': raw.get('summary', ''),
        'follow_up_pattern': raw.get('follow_up_pattern', ''),
        'autonomy_level': raw.get('autonomy_level', 'medium'),
    }

    # Add normalized fields
    llm_signals.update(normalize_llm_fields(llm_signals))

    # Cache
    if cache_dir:
        cache_path = cache_dir / f"{task_id}_{prompt_hash[:8]}.json"
        with open(cache_path, 'w') as f:
            json.dump(llm_signals, f, indent=2)

    return llm_signals


def aggregate_signals(task, keyword_signals, edit_signals, llm_signals, complexity):
    """Combine all signals into final annotated task record.

    Aggregation rules:
    - Satisfaction: LLM primary, downgrade if edit_signals.user_corrections > 0
    - Quality: LLM primary, downgrade if rewrite_rate > 0.3
    - Complexity: from classified tasks if available, LLM as fallback
    """
    # Start with identity fields from canonical task
    result = {
        'task_id': task['task_id'],
        'model': task.get('model', 'unknown'),
    }

    # Basic stats (for backward compat with stat_tests.py)
    tool_calls_list = task.get('tool_calls', [])
    duration = task.get('duration_seconds', 0) or 1
    files = task.get('total_files_touched', 0) or 1
    tool_count = len(tool_calls_list)
    lines_total = task.get('total_lines_added', 0) + task.get('total_lines_removed', 0)

    result['tool_calls'] = tool_count
    result['files_touched'] = task.get('total_files_touched', 0)
    result['lines_added'] = task.get('total_lines_added', 0)
    result['lines_removed'] = task.get('total_lines_removed', 0)
    result['duration_seconds'] = task.get('duration_seconds', 0)
    result['lines_per_minute'] = lines_total / (duration / 60) if duration > 0 else 0
    result['tools_per_file'] = tool_count / files if files > 0 else 0

    # Complexity
    result['complexity'] = complexity

    # Keyword signals
    result['keyword_signals'] = keyword_signals

    # Edit signals
    result['edit_signals'] = edit_signals

    # LLM signals (all fields from LLM, including normalized)
    if llm_signals:
        for key, val in llm_signals.items():
            result[key] = val
    else:
        # Fallback: use keyword signals for basic fields
        kw = keyword_signals
        if kw['negative_hits'] > kw['positive_hits']:
            result['normalized_user_sentiment'] = 'dissatisfied'
        elif kw['positive_hits'] > 0:
            result['normalized_user_sentiment'] = 'satisfied'
        else:
            result['normalized_user_sentiment'] = 'neutral'
        result['normalized_execution_quality'] = 'adequate'
        result['normalized_work_category'] = 'directed_impl'
        result['alignment_score'] = 3
        result['task_completion'] = 'complete'
        result['scope_management'] = 'appropriate'
        result['communication_quality'] = 'adequate'
        result['error_recovery'] = 'none_needed'
        result['iteration_required'] = 'one_shot'
        result['autonomy_level'] = 'medium'
        result['summary'] = ''
        result['follow_up_pattern'] = ''
        result['user_sentiment'] = ''
        result['sentiment_confidence'] = 'low'
        result['execution_quality'] = ''
        result['work_category'] = ''

    # --- Aggregation adjustments ---

    # Downgrade satisfaction if user corrections detected
    if edit_signals.get('user_corrections', 0) > 0:
        if result.get('normalized_user_sentiment') == 'satisfied':
            result['normalized_user_sentiment'] = 'ambiguous'

    # Downgrade execution quality if high rewrite rate
    if edit_signals.get('rewrite_rate', 0) > 0.3:
        eq = result.get('normalized_execution_quality', '')
        if eq == 'excellent':
            result['normalized_execution_quality'] = 'good'
        elif eq == 'good':
            result['normalized_execution_quality'] = 'adequate'

    # Signal agreement for transparency
    kw_sentiment = 'neutral'
    if keyword_signals.get('positive_hits', 0) > keyword_signals.get('negative_hits', 0):
        kw_sentiment = 'satisfied'
    elif keyword_signals.get('negative_hits', 0) > keyword_signals.get('positive_hits', 0):
        kw_sentiment = 'dissatisfied'

    result['signal_agreement'] = {
        'keyword_sentiment': kw_sentiment,
        'llm_sentiment': result.get('normalized_user_sentiment', 'neutral'),
        'edit_user_corrections': edit_signals.get('user_corrections', 0),
        'edit_rewrite_rate': edit_signals.get('rewrite_rate', 0),
    }

    return result


def is_trivial_task(task):
    """Check if a task is too trivial to warrant LLM annotation.

    Trivial: 0 tool calls AND duration < 10s. These are interrupted requests,
    slash commands, continuation acks, "test" messages, etc.
    """
    tool_count = len(task.get('tool_calls', []))
    duration = task.get('duration_seconds', 0) or 0
    return tool_count == 0 and duration < 10


def is_small_task(task, classified_lookup):
    """Check if a task is small enough to batch with others."""
    task_id = task.get('task_id', 'unknown')
    classified = classified_lookup.get(task_id, {})
    complexity = classified.get('classification', {}).get('complexity', 'unknown')
    tool_count = len(task.get('tool_calls', []))
    return complexity in ('trivial', 'simple') and tool_count <= 3


BATCH_PROMPT_TEMPLATE = """You are analyzing multiple Claude Code task interactions. For EACH task, provide a rich qualitative summary.

Return a JSON array with one object per task, in the same order as presented. Each object must have exactly these fields:

{{
  "task_id": "the task ID",
  "work_category": "1-2 sentence characterization",
  "execution_quality": "1-2 sentence assessment",
  "user_sentiment": "inferred sentiment",
  "sentiment_confidence": "low|medium|high",
  "follow_up_pattern": "what happened next",
  "autonomy_level": "high|medium|low",
  "task_completion": "complete|partial|interrupted|failed",
  "scope_management": "focused|appropriate|expanded|over_engineered",
  "communication_quality": "clear|adequate|verbose|unclear",
  "error_recovery": "none_needed|recovered|struggled|failed",
  "iteration_required": "one_shot|minor|significant",
  "alignment_score": "1-5 integer",
  "summary": "2-3 sentence overall summary"
}}

{tasks_block}

Return ONLY the JSON array, no other text."""


def build_batch_prompt(tasks):
    """Build a batched LLM prompt for multiple small tasks."""
    blocks = []
    for i, task in enumerate(tasks, 1):
        prompt_text = build_llm_prompt(task)
        # Extract just the task data section (skip the preamble and instructions)
        blocks.append(f"## Task {i}\n\n{prompt_text}")
    return BATCH_PROMPT_TEMPLATE.format(tasks_block="\n\n---\n\n".join(blocks))


def call_llm_batch(tasks, cache_dir, llm_model, force):
    """Call LLM for a batch of small tasks. Returns list of signal dicts (or None per task)."""
    # Check if all tasks are already cached
    all_cached = []
    uncached_indices = []
    for i, task in enumerate(tasks):
        prompt = build_llm_prompt(task)
        prompt_hash = get_prompt_hash(prompt)
        task_id = task.get('task_id', 'unknown')
        cached_result = None
        if cache_dir and not force:
            cache_path = cache_dir / f"{task_id}_{prompt_hash[:8]}.json"
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        cached_result = json.load(f)
                except Exception:
                    pass
        all_cached.append(cached_result)
        if cached_result is None:
            uncached_indices.append(i)

    # If all cached, return immediately
    if not uncached_indices:
        return all_cached

    # Build batch prompt for uncached tasks
    uncached_tasks = [tasks[i] for i in uncached_indices]
    batch_prompt = build_batch_prompt(uncached_tasks)

    try:
        result = subprocess.run(
            ["claude", "-p", batch_prompt, "--model", llm_model, "--output-format", "text"],
            capture_output=True, text=True, timeout=120,
            cwd="/tmp",
        )
        if result.returncode != 0:
            # Fall back to individual calls
            for i in uncached_indices:
                all_cached[i] = compute_llm_signals(tasks[i], cache_dir, llm_model, force)
            return all_cached

        response = result.stdout.strip()
        start = response.find('[')
        end = response.rfind(']') + 1
        if start < 0 or end <= start:
            # Try single-object fallback
            for i in uncached_indices:
                all_cached[i] = compute_llm_signals(tasks[i], cache_dir, llm_model, force)
            return all_cached

        parsed = json.loads(response[start:end])

        # Map results back to uncached positions
        for j, i in enumerate(uncached_indices):
            if j < len(parsed):
                raw = parsed[j]
                alignment = raw.get('alignment_score', 3)
                if isinstance(alignment, str):
                    try:
                        alignment = int(alignment)
                    except ValueError:
                        alignment = 3

                llm_signals = {
                    'user_sentiment': raw.get('user_sentiment', ''),
                    'sentiment_confidence': raw.get('sentiment_confidence', 'low'),
                    'execution_quality': raw.get('execution_quality', ''),
                    'work_category': raw.get('work_category', ''),
                    'scope_management': raw.get('scope_management', 'appropriate'),
                    'communication_quality': raw.get('communication_quality', 'adequate'),
                    'error_recovery': raw.get('error_recovery', 'none_needed'),
                    'iteration_required': raw.get('iteration_required', 'one_shot'),
                    'task_completion': raw.get('task_completion', 'complete'),
                    'alignment_score': alignment,
                    'summary': raw.get('summary', ''),
                    'follow_up_pattern': raw.get('follow_up_pattern', ''),
                    'autonomy_level': raw.get('autonomy_level', 'medium'),
                }
                llm_signals.update(normalize_llm_fields(llm_signals))

                # Cache individual result
                if cache_dir:
                    task = tasks[i]
                    prompt = build_llm_prompt(task)
                    prompt_hash = get_prompt_hash(prompt)
                    task_id = task.get('task_id', 'unknown')
                    cache_path = cache_dir / f"{task_id}_{prompt_hash[:8]}.json"
                    with open(cache_path, 'w') as f:
                        json.dump(llm_signals, f, indent=2)

                all_cached[i] = llm_signals

    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        # Fall back to individual calls
        for i in uncached_indices:
            all_cached[i] = compute_llm_signals(tasks[i], cache_dir, llm_model, force)

    return all_cached


def annotate_single_task(
    task, classified_lookup, cache_dir, llm_model, force,
    progress_lock, progress_counter, total
):
    """Annotate a single task with all signals. For parallel execution."""
    task_id = task.get('task_id', 'unknown')

    # Compute all signals
    keyword_signals = compute_keyword_signals(task)
    edit_signals = compute_edit_signals(task)
    llm_signals = compute_llm_signals(task, cache_dir, llm_model, force)
    status = "done" if llm_signals else "no-llm"

    # Get complexity from classified tasks
    classified = classified_lookup.get(task_id, {})
    complexity = classified.get('classification', {}).get('complexity', 'unknown')

    # Aggregate
    result = aggregate_signals(task, keyword_signals, edit_signals, llm_signals, complexity)

    with progress_lock:
        progress_counter[0] += 1
        print(f"[{progress_counter[0]}/{total}] {task_id[:30]}... {status}")

    return result


def to_llm_analysis_compat(annotated):
    """Convert annotated task record to llm-analysis format for backward compatibility.

    Downstream scripts (stat_tests.py, planning_analysis.py, analyze_compaction.py)
    read llm-analysis-{model}.json with these fields.
    """
    return {
        'task_id': annotated['task_id'],
        'model': annotated['model'],
        'tool_calls': annotated.get('tool_calls', 0),
        'files_touched': annotated.get('files_touched', 0),
        'lines_added': annotated.get('lines_added', 0),
        'lines_removed': annotated.get('lines_removed', 0),
        'duration_seconds': annotated.get('duration_seconds', 0),
        'work_category': annotated.get('work_category', ''),
        'execution_quality': annotated.get('execution_quality', ''),
        'user_sentiment': annotated.get('user_sentiment', ''),
        'sentiment_confidence': annotated.get('sentiment_confidence', 'low'),
        'follow_up_pattern': annotated.get('follow_up_pattern', ''),
        'summary': annotated.get('summary', ''),
        'task_completion': annotated.get('task_completion', 'complete'),
        'scope_management': annotated.get('scope_management', 'appropriate'),
        'communication_quality': annotated.get('communication_quality', 'adequate'),
        'error_recovery': annotated.get('error_recovery', 'none_needed'),
        'iteration_required': annotated.get('iteration_required', 'one_shot'),
        'alignment_score': annotated.get('alignment_score', 0),
        'lines_per_minute': annotated.get('lines_per_minute', 0),
        'tools_per_file': annotated.get('tools_per_file', 0),
        'autonomy_level': annotated.get('autonomy_level', 'medium'),
        'normalized_execution_quality': annotated.get('normalized_execution_quality', 'adequate'),
        'normalized_work_category': annotated.get('normalized_work_category', 'directed_impl'),
        'normalized_user_sentiment': annotated.get('normalized_user_sentiment', 'neutral'),
    }


def print_summary(annotated_tasks, model):
    """Print summary statistics for annotated tasks."""
    n = len(annotated_tasks)
    if n == 0:
        print(f"\n{model}: No tasks annotated")
        return

    print(f"\n{'='*60}")
    print(f"{model.upper()} ANNOTATION SUMMARY ({n} tasks)")
    print(f"{'='*60}")

    def dist(field, order=None):
        counts = {}
        for t in annotated_tasks:
            val = str(t.get(field, 'unknown')).lower()
            counts[val] = counts.get(val, 0) + 1
        items = order if order else sorted(counts.keys(), key=lambda x: -counts.get(x, 0))
        print(f"\n  {field}:")
        for key in items:
            count = counts.get(key, 0)
            if count == 0 and order:
                continue
            pct = 100 * count / n
            bar = '#' * int(pct / 2)
            print(f"    {key:<20} {count:3d} ({pct:5.1f}%) {bar}")

    dist('normalized_user_sentiment', ['satisfied', 'neutral', 'dissatisfied', 'ambiguous'])
    dist('normalized_execution_quality', ['excellent', 'good', 'adequate', 'poor', 'failed'])
    dist('task_completion', ['complete', 'partial', 'interrupted', 'failed'])
    dist('autonomy_level', ['high', 'medium', 'low'])

    scores = [t['alignment_score'] for t in annotated_tasks
              if t.get('alignment_score') and t['alignment_score'] > 0]
    if scores:
        print(f"\n  alignment_score: avg={sum(scores)/len(scores):.2f} median={sorted(scores)[len(scores)//2]}")

    # Signal agreement summary
    agree = 0
    disagree = 0
    for t in annotated_tasks:
        sa = t.get('signal_agreement', {})
        if sa.get('keyword_sentiment') == sa.get('llm_sentiment'):
            agree += 1
        else:
            disagree += 1
    if agree + disagree > 0:
        print(f"\n  Signal agreement: {agree}/{agree+disagree} ({100*agree/(agree+disagree):.0f}%) keyword-LLM match")


def main():
    parser = argparse.ArgumentParser(
        description='Unified task annotation pipeline (replaces analyze_tasks_llm + normalize)')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Data directory with tasks-canonical-*.json')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: data-dir/../analysis)')
    parser.add_argument('--force', action='store_true',
                        help='Skip cache and re-analyze all tasks')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='Number of parallel LLM workers (default: 2)')
    parser.add_argument('--llm', default='haiku',
                        help='LLM model to use for analysis (default: haiku)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks to process')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Batch small tasks into one LLM call (default: 5)')
    parser.add_argument('--trivial-batch-size', type=int, default=10,
                        help='Batch trivial tasks into larger LLM batches (default: 10)')
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir.parent / 'analysis').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / 'annotation-cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UNIFIED TASK ANNOTATION PIPELINE")
    print("=" * 60)

    models = discover_models(data_dir, prefix="tasks-canonical")
    if not models:
        print(f"No tasks-canonical-*.json files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Models: {', '.join(models)}")
    print(f"Output: {output_dir}")
    print(f"Cache:  {cache_dir}")

    for model in models:
        canonical_file = data_dir / f'tasks-canonical-{model}.json'
        if not canonical_file.exists():
            print(f"\nSkipping {model}: {canonical_file} not found")
            continue

        with open(canonical_file) as f:
            tasks = json.load(f)

        # Load classified tasks for complexity lookup
        classified_file = data_dir / f'tasks-classified-{model}.json'
        classified_lookup = {}
        if classified_file.exists():
            with open(classified_file) as f:
                for t in json.load(f):
                    classified_lookup[t['task_id']] = t

        if args.limit is not None:
            tasks = tasks[:args.limit]

        # Partition tasks into trivial (large batches), small (batched), and normal
        trivial_tasks = []
        small_tasks = []
        normal_tasks = []
        for task in tasks:
            if is_trivial_task(task):
                trivial_tasks.append(task)
            elif is_small_task(task, classified_lookup):
                small_tasks.append(task)
            else:
                normal_tasks.append(task)

        print(f"\n{model.upper()}: annotating {len(tasks)} tasks "
              f"({len(trivial_tasks)} trivial/batch-{args.trivial_batch_size}, "
              f"{len(small_tasks)} small/batch-{args.batch_size}, "
              f"{len(normal_tasks)} normal) with {args.max_workers} workers...")

        progress_lock = threading.Lock()
        progress_counter = [0]
        total = len(tasks)

        annotated = []

        # 1. Trivial tasks: batched LLM calls with larger batch size
        trivial_batch_size = args.trivial_batch_size
        for batch_start in range(0, len(trivial_tasks), trivial_batch_size):
            batch = trivial_tasks[batch_start:batch_start + trivial_batch_size]
            batch_results = call_llm_batch(batch, cache_dir, args.llm, args.force)

            for task, llm_signals in zip(batch, batch_results):
                task_id = task.get('task_id', 'unknown')
                keyword_signals = compute_keyword_signals(task)
                edit_signals = compute_edit_signals(task)
                classified = classified_lookup.get(task_id, {})
                complexity = classified.get('classification', {}).get('complexity', 'unknown')
                result = aggregate_signals(task, keyword_signals, edit_signals, llm_signals, complexity)
                annotated.append(result)
                with progress_lock:
                    progress_counter[0] += 1
                    status = "trivial-batched" if llm_signals else "trivial-fail"
                    print(f"[{progress_counter[0]}/{total}] {task_id[:30]}... {status}")

        # 2. Small tasks: batched LLM calls
        batch_size = args.batch_size
        for batch_start in range(0, len(small_tasks), batch_size):
            batch = small_tasks[batch_start:batch_start + batch_size]
            batch_results = call_llm_batch(batch, cache_dir, args.llm, args.force)

            for task, llm_signals in zip(batch, batch_results):
                task_id = task.get('task_id', 'unknown')
                keyword_signals = compute_keyword_signals(task)
                edit_signals = compute_edit_signals(task)
                classified = classified_lookup.get(task_id, {})
                complexity = classified.get('classification', {}).get('complexity', 'unknown')
                result = aggregate_signals(task, keyword_signals, edit_signals, llm_signals, complexity)
                annotated.append(result)
                with progress_lock:
                    progress_counter[0] += 1
                    status = "batched" if llm_signals else "batch-fail"
                    print(f"[{progress_counter[0]}/{total}] {task_id[:30]}... {status}")

        # 3. Normal tasks: individual parallel LLM calls
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    annotate_single_task,
                    task, classified_lookup, cache_dir, args.llm, args.force,
                    progress_lock, progress_counter, total
                ): task
                for task in normal_tasks
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        annotated.append(result)
                except Exception as e:
                    print(f"  Worker error: {e}", file=sys.stderr)

        # Sort by task_id for stable output
        annotated.sort(key=lambda t: t['task_id'])

        # Write annotated tasks
        annotated_file = output_dir / f'tasks-annotated-{model}.json'
        with open(annotated_file, 'w') as f:
            json.dump(annotated, f, indent=2)
        print(f"Wrote {len(annotated)} annotated tasks to {annotated_file}")

        # Write backward-compatible llm-analysis file
        compat = [to_llm_analysis_compat(t) for t in annotated]
        compat_file = output_dir / f'llm-analysis-{model}.json'
        with open(compat_file, 'w') as f:
            json.dump(compat, f, indent=2)
        print(f"Wrote backward-compatible {compat_file}")

        print_summary(annotated, model)

    print(f"\n{'='*60}")
    print("Annotation complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
