# Model Comparison Prompt

You are comparing the performance of two Claude models based on aggregated task data.

## Models Being Compared

- **Opus 4.5** (claude-opus-4-5-20251101): The standard flagship model
- **Opus 4.6** (claude-opus-4-6): Successor model

## Aggregated Statistics

### Opus 4.5 Tasks
- Total tasks: {{opus-4-5_task_count}}
- Success rate: {{opus-4-5_success_rate}}%
- Avg tool calls per task: {{opus-4-5_avg_tools}}
- Avg duration: {{opus-4-5_avg_duration}}s
- Completion signals: {{opus-4-5_completion_signals}}

### Opus 4.6 Tasks
- Total tasks: {{opus-4-6_task_count}}
- Success rate: {{opus-4-6_success_rate}}%
- Avg tool calls per task: {{opus-4-6_avg_tools}}
- Avg duration: {{opus-4-6_avg_duration}}s
- Completion signals: {{opus-4-6_completion_signals}}

## Analysis Instructions

Compare the models across these dimensions:

### 1. Task Completion Patterns
- Which model has higher apparent success rates?
- Are there differences in how tasks end?
- Any patterns in abandonment or dissatisfaction?

### 2. Efficiency Comparison
- Which model uses fewer tool calls on average?
- Are there differences in task duration?
- Does one model take more direct paths?

### 3. User Satisfaction Signals
- Compare "explicit_done" vs "user_dissatisfied" ratios
- Which model gets more positive completion signals?

### 4. Strengths and Weaknesses
- What does each model do well?
- Where does each model struggle?
- Are there task types where one model excels?

### 5. Overall Assessment
- Which model performs better overall?
- Is the difference significant or marginal?
- What caveats apply to this comparison?

## Output Format

Provide a structured comparison report:

```markdown
# Model Comparison: Opus 4.5 vs Opus 4.6

## Executive Summary
[2-3 sentence overview]

## Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Detailed Comparison

### Task Completion
[Analysis]

### Efficiency
[Analysis]

### User Satisfaction
[Analysis]

## Model Profiles

### Opus 4.5 Strengths/Weaknesses
- Strengths: ...
- Weaknesses: ...

### Opus 4.6 Strengths/Weaknesses
- Strengths: ...
- Weaknesses: ...

## Recommendations
[When to prefer each model]

## Caveats
[Limitations of this analysis]
```

## Sample Task Data

### Opus 4.5 Sample Tasks
{{opus-4-5_sample_tasks}}

### Opus 4.6 Sample Tasks
{{opus-4-6_sample_tasks}}
