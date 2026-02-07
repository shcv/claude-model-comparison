# Report Generation Prompt

You are generating a complete HTML report comparing two Claude Code models.

## Models

- **Opus 4.5** (claude-opus-4-5-20251101) — Orange color, bar tag "A"
- **Opus 4.6** (claude-opus-4-6) — Blue color, bar tag "B"

## Output Format

Output a single, complete, standalone HTML document. Include the full CSS from the style reference (copy it exactly). The HTML should render beautifully with no external dependencies except Google Fonts.

## Component Reference

You have a style reference file showing all available HTML/CSS components. Use these patterns:

- **Stat cards** (`.stat-row > .stat-card`) — For headline numbers in the executive summary
- **Dual-bar comparison tables** — The primary visualization. Each row shows two horizontal bars (A and B) side by side. Set `bar-fill` width as a percentage of the maximum value in that table.
- **Single-bar tables** — For showing independent distributions
- **Plain data tables** — For numeric grids without bars
- **Callouts** (`.callout`, `.callout.blue`, `.callout.green`) — For key insights
- **Pipeline diagram** (`.pipeline`) — For the methodology section
- **Profile cards** (`.profile-grid`) — For side-by-side model summaries
- **Delta column** — For tables that show which model wins

CSS classes for bar fills: `.a` (orange/Opus 4.5), `.b` (blue/Opus 4.6), `.green`, `.red`.
Value color classes: `.v-orange`, `.v-blue`, `.v-green`, `.v-dark`.

## Report Structure

Write these sections (you may add or adjust based on what the data supports):

### Header
- Eyebrow: "Claude Code EAP Report"
- Title: Something descriptive about the comparison
- Subtitle: Mention it's a single-user comparative study, include total task count
- Meta: "Samuel H. Christie V · February 2026 · Claude Code Early Access Program"

### 1. Executive Summary
- 3-4 stat cards with the most striking findings (delta format — see Formatting Requirements)
- A callout with the key takeaway
- Brief prose overview: focus on what the models do differently and what matters for choosing between them
- Do NOT lead with methodology as a "finding" — methodology lessons belong in section 11
- **Table of Contents** with key findings: a `<nav class="toc">` listing sections 2-11 with one-line finding highlights and anchor links

### Section ordering principle
Sections are ordered by **effect strength**: larger, more robust findings first, weaker or corrective findings later. Statistical significance notes are inline in each section (`.stat-note` divs), not in a standalone section.

### 2. Token Economy & Cost
- Per-task cost is nearly identical (~$2.40) despite very different strategies
- Opus 4.6 produces 2.7x more output tokens but is 9-45% cheaper per task at every complexity level
- Thinking calibration: 4.6 skips thinking on 71% of trivial tasks, thinks on 100% of major tasks; 4.5 over-thinks easy problems (66% trivial thinking rate)
- Output verbosity by task type: refactoring is 6.3x more verbose for 4.6, continuation is 1.0x
- Cost savings come from more efficient prompt caching and fewer API round-trips
- Data from extract_tokens.py and analysis/token-analysis.json

### 3. Complexity-Controlled Comparison
- Task distribution by complexity level
- Resource usage by complexity (duration, tools, files, lines)
- Callout about the exploration-satisfaction tradeoff
- **Inline stat note**: Tool calls (p=0.000002) and tools/file (p=0.000003) are Bonferroni-significant

### 4. Behavioral Patterns
- Subagent usage comparison (tasks using subagents, autonomous %)
- Planning mode adoption
- Subagent type distribution (Explore, general-purpose, Bash, Plan)
- Callout about the "invisible subagent" insight
- **Inline stat note**: Autonomy (p=0.007) and one-shot rate (p=0.004) NOT Bonferroni-significant

### 5. Session Dynamics
- Warm-up effects: Do models improve within a session? (from session-analysis.json)
- Effort distribution: research vs implementation tool ratios per model
- Session length effects: both degrade, 4.6 more steeply BUT 4.6 still scores higher even in long sessions (2.96 vs 2.85) — DO NOT recommend 4.5 for long sessions
- **Inline stat note**: Duration (p=0.015) NOT Bonferroni-significant

### 6. Quality & Satisfaction
- Dual-bar table: sentiment distribution (satisfied, neutral, dissatisfied percentages)
- Dual-bar table: completion distribution (complete, partial, interrupted, failed)
- Note about dissatisfaction false-positive correction
- **Inline stat note**: Alignment (p=0.000328) IS Bonferroni-significant; satisfaction/dissatisfaction NOT significant

### 7. Parallelization & Directive Compliance
- How each model responds to "run in parallel" directives
- Background task usage
- Callout about the parallelism paradox
- Small sample (n=6-8), no statistical testing

### 8. Refactoring: A False Signal
- Automated sentiment flagged 9.1% vs 5.9% dissatisfaction — all 4 cases were false positives from "fix" in task requests
- With 23-71 tasks per model, one misclassification shifts rates by 1.4-4.3pp
- Git retention ground-truth: LLM agent cross-referenced session timestamps with commit history
  - Strabo: 4.5 had 50% commit rate (3/6 sessions), 4.6 had 0% but ~1,923 lines uncommitted WIP
  - Peach: 4.5 had 80% commit rate (8/10 sessions), no 4.6 refactoring sessions
  - Grocery List (head-to-head): 4.5 had 69% commit rate (9/13), 4.6 had 100% (5/5)
  - No refactoring commits were reverted in any repo
  - Temporal bias: 4.5 had 2+ months, 4.6 had 4 days
- Callout: retention depends on project difficulty, not model choice
- DO NOT recommend routing refactoring to Opus 4.5

### 9. Planning: Correlation vs Causation
- Opus 4.6 uses planning 10x more frequently
- But planning only improves alignment by +0.04
- The planning-quality correlation may reflect common cause (model thoroughness), not direct effect
- Callout with the nuanced takeaway

### 10. Model Profiles
- Profile cards for each model
- Summary of when to use each model
- Include routing recommendations (which tasks to send to which model)
- Refactoring should be "Either" — no measurable quality difference
- Long sessions should be "Either" — 4.6 degrades faster but still scores higher

### 11. Methodology
- Pipeline diagram showing the analysis steps
- Brief methodology description
- **Statistical testing** summary: 105 tests, Bonferroni correction, 3 survive
- **Development Process** subsection: describe approaches that were tried and abandoned
  - Automated dissatisfaction detection had 73-93% false positive rate — corrected via LLM-agent audit
  - LLM quality judgement was unreliable and dropped entirely
  - Refactoring routing recommendation was reversed after audit + git cross-referencing
- **Limitations** (single user, observational not experimental, complexity confound, sample asymmetry)
- **LLM-in-the-loop** limitation: all classification and auditing done by LLM agents with human review — NOT manual/hand auditing

### Footer
- "Claude Code Early Access Program · Single-user behavioral analysis · February 2026"

## Formatting Requirements

### Stat cards must show deltas
Executive summary stat cards should show the **difference between models** as the primary big number, not absolute values. Use the format: `+X.Xpp Satisfaction Gap`, `+X.Xpp Subagent Gap`, etc. The detail line beneath can show the individual model values for context.

Example: Instead of "Opus 4.6 Satisfied: 43.3%" with small comparison text, show "Satisfaction Gap: +8.9pp" as the primary number with "43.3% vs 34.4%" as the detail.

### Use stacked bar-pair format for ALL comparison tables
Every table comparing the two models must use the `bar-pair` format (two bars stacked within the same table row — A on top, B below). Do NOT use split `bar-single` format where each model gets its own separate row. This applies to ALL comparison tables including:
- Sentiment distribution
- Completion distribution
- Subagent type distribution
- Complexity distribution
- Resource usage

Each row should have: metric label | stacked A+B bars | A value | B value | delta.

### No self-referential language
Present all findings as standalone conclusions. Do NOT reference "earlier versions of this analysis," "previous findings," "corrections," or imply that the report has been revised. Every finding should read as if this is the definitive first presentation of the data.

Bad: "An earlier version of this analysis flagged..."
Good: "Automated sentiment detection flagged..."

Bad: "Correction: both models are suitable"
Good: "Bottom line: both models are suitable"

## Data Processing Guidelines

- Calculate percentages from raw counts (e.g., satisfied_count / total_tasks * 100)
- For bar widths, scale relative to the maximum value in each comparison (so the largest bar is near 100%)
- Round percentages to 1 decimal place
- Use `pp` (percentage points) for differences between percentages
- The dissatisfaction numbers in the aggregate stats may include false positives from system-generated signals; note this caveat

## Writing Style

- Analytical but accessible. Write for a technical audience interested in AI model behavior.
- Lead with findings, not data descriptions. "Opus 4.6 explores 3x more before implementation" not "The data shows that..."
- Use callouts for counter-intuitive or especially important findings
- Keep paragraphs to 2-3 sentences max
- Don't include a table of contents
