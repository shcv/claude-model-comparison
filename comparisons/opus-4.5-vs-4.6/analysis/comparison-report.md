# Opus 4.6 vs Opus 4.5 Model Comparison Report

**Generated:** 2025-02-04
**Data Period:** Last 7 days
**Methodology:** Session analysis with LLM-verified classification

## Executive Summary

After correcting for false positives in outcome detection (73-93% of "dissatisfaction" signals were system-generated content, not user complaints), **Opus 4.6 outperforms Opus 4.5 on most metrics**:

- **Lower dissatisfaction**: 1.3% vs 3.7%
- **Higher efficiency**: 23.2 vs 39.3 tools per 100 lines of code (41% fewer tools)
- **Larger task scope**: 5.3 files & 378 lines avg vs 3.4 files & 322 lines
- **Better on complex tasks**: 0% vs 4.2% dissatisfaction on complex tasks

## Data Quality Improvements

### Original vs Corrected Dissatisfaction Rates

| Model | Original | Corrected | False Positive Rate |
|-------|----------|-----------|---------------------|
| Opus 4.5 | 7.1%  | 3.7%      | 73% |
| Opus 4.6 | 8.7%  | 1.3%      | 93% |

### Sources of False Positives

1. **Session continuation messages** - "This session is being continued..." containing summary text with words like "fix"
2. **Task notifications** - `<task-notification>` from subagents reporting "Fix X completed"
3. **Plan templates** - "Implement the following plan: Fix..." where "fix" is the task name
4. **Questions** - "will this fix it?" misread as complaints

### Classification Improvements

| Category | Opus 4.5 Before | Opus 4.5 After | Opus 4.6 Before | Opus 4.6 After |
|----------|-----------------|----------------|-----------------|----------------|
| Unknown  | 32.5%           | 29.3%          | 28.3%           | 23.0%          |
| Continuation | 8.8%        | 11.8%          | 15.7%           | 21.6%          |

## Detailed Comparison

### By Task Type (excluding continuations)

| Type          | O45 # | O46 # | O45-Tools | O46-Tools | O45-Files | O46-Files | O45-T/File | O46-T/File | O45-Dis% | O46-Dis% |
|---------------|--------|---------|---------|---------|---------|---------|----------|----------|--------|--------|
| bugfix        | 25     | 11      | 12.3    | 26.5    | 3.2     | 4.5     | 10.9     | 6.3      | 12.0   | 0.0    |
| feature       | 47     | 21      | 14.5    | 14.5    | 3.7     | 3.5     | 6.8      | 5.1      | 2.1    | 0.0    |
| greenfield    | 36     | 19      | 8.9     | 30.9    | 3.4     | 7.4     | 7.9      | 5.6      | 2.8    | 0.0    |
| investigation | 119    | 48      | 10.0    | 9.7     | 4.0     | 5.8     | 5.0      | 4.1      | 0.8    | 2.1    |
| port          | 11     | 8       | 7.8     | 37.8    | 2.2     | 7.8     | 4.3      | 10.7     | 9.1    | 0.0    |
| refactor      | 34     | 22      | 16.9    | 39.0    | 6.0     | 12.7    | 5.8      | 5.9      | 5.9    | 9.1    |
| sysadmin      | 34     | 25      | 6.4     | 7.0     | 1.6     | 2.5     | 10.4     | 8.7      | 5.9    | 4.0    |

**Key observations:**
- Opus 4.6 has 0% dissatisfaction on bugfix, feature, greenfield, and port tasks
- Opus 4.6 touches 2x more files on greenfield (7.4 vs 3.4) and refactor (12.7 vs 6.0)
- Refactor is the only category where Opus 4.6 has higher dissatisfaction (9.1% vs 5.9%)

### By Complexity

| Complexity | O45 # | O46 # | O45-Tools | O46-Tools | O45-T/100L | O46-T/100L | O45-Dis% | O46-Dis% |
|------------|--------|---------|---------|---------|----------|----------|--------|--------|
| trivial    | 229    | 73      | 0.9     | 0.9     | 35.2     | 26.4     | 3.9    | 2.7    |
| simple     | 78     | 46      | 5.6     | 5.8     | 34.9     | 28.0     | 5.1    | 4.3    |
| moderate   | 122    | 60      | 13.2    | 12.6    | 41.7     | 32.5     | 1.6    | 0.0    |
| complex    | 48     | 43      | 34.7    | 42.4    | 19.1     | 11.7     | 4.2    | 0.0    |
| major      | 8      | 10      | 78.6    | 87.6    | 157.5    | 9.0      | 0.0    | 0.0    |

**Key observations:**
- Opus 4.6 is more efficient (lower T/100L) at EVERY complexity level
- The gap widens dramatically at major complexity: 9.0 vs 157.5 tools per 100 lines
- Opus 4.6 has 0% dissatisfaction on moderate, complex, and major tasks

### Efficiency Summary (tasks with file changes)

| Metric | Opus 4.5 | Opus 4.6 | Difference |
|--------|----------|----------|------------|
| Tasks with changes | 242 | 155 | |
| Avg tools/task | 16.9 | 23.4 | +6.6 |
| Avg files/task | 3.4 | 5.3 | +1.9 |
| Avg lines/task | 322 | 378 | +56 |
| **Tools per file** | 6.63 | 6.14 | **-0.50** |
| **Tools per 100 lines** | 39.32 | 23.23 | **-16.09** |
| **Dissatisfaction %** | 3.7 | 1.3 | **-2.4** |

## Model Profiles

### Opus 4.5 (claude-opus-4-5-20251101)

**Strengths:**
- More investigation tasks completed (119 vs 48)
- Lower tool count on individual tasks (appears more "focused")
- Slightly better on refactoring tasks (5.9% vs 9.1% dissatisfaction)

**Weaknesses:**
- Higher dissatisfaction rate overall (3.7% vs 1.3%)
- Less efficient per unit of work (39.3 vs 23.2 tools per 100 lines)
- Struggles more on complex tasks (4.2% vs 0% dissatisfaction)
- Major tasks show extreme inefficiency (157.5 tools per 100 lines)

**Best for:**
- Quick investigations and research
- Smaller, focused changes
- Refactoring tasks

### Opus 4.6 (claude-opus-4-6)

**Strengths:**
- Lower dissatisfaction rate (1.3% vs 3.7%)
- Much more efficient per line of code (41% fewer tools)
- Handles larger scope tasks well (5.3 vs 3.4 files avg)
- Excellent on complex/major tasks (0% dissatisfaction)
- Dramatic efficiency on major tasks (9.0 vs 157.5 tools per 100 lines)

**Weaknesses:**
- Higher dissatisfaction on refactoring (9.1% vs 5.9%)
- Uses more tools overall (but does more work)
- Slightly worse on investigation tasks (2.1% vs 0.8% dissatisfaction)

**Best for:**
- Large implementation tasks
- Greenfield development
- Complex multi-file changes
- Port/migration work

## Matched Pairs Analysis

From 10 matched task pairs (similar type, scale, complexity):

- **Opus 4.5 wins:** 4
- **Opus 4.6 wins:** 5
- **Ties:** 1
- **Average efficiency difference:** -0.49 tools per 100 lines (Opus 4.6 more efficient)

Notable patterns:
- When outcomes differ, the model achieving satisfaction wins regardless of efficiency
- Opus 4.6 tends to complete faster on trivial tasks
- Opus 4.5 showed a potential loop behavior on one major task (100 tools vs Opus 4.6's 6)

## Caveats and Limitations

1. **Sample size**: 550 Opus 4.5 tasks, 296 Opus 4.6 tasks over 7 days
2. **User population**: Same user, but potentially different project contexts
3. **Task distribution**: Opus 4.6 used more on larger implementation tasks
4. **Outcome detection**: While improved, heuristic-based outcome detection has limits
5. **No ground truth**: Success based on user signals, not verified correctness

## Recommendations

1. **Prefer Opus 4.6 for implementation work** - especially greenfield, features, and complex tasks
2. **Use Opus 4.5 for quick investigations** - slightly better on research/exploration
3. **Consider Opus 4.5 for refactoring** - lower dissatisfaction rate on restructuring work
4. **Opus 4.6 for large tasks** - dramatically more efficient at scale

## Next Steps

1. **Replication study**: Run identical prompts through both models for direct comparison
2. **Refactor deep-dive**: Investigate why Opus 4.6 has higher dissatisfaction on refactoring
3. **Efficiency analysis**: Understand what makes Opus 4.6 more efficient (parallelization? better planning?)
