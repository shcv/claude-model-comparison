# Task Analysis Prompt

You are evaluating the quality of Claude Code's performance on a specific task.

## Task Context

**Model**: {{model}}
**User Request**: {{user_prompt}}
**Tool Calls**: {{tool_calls}}
**Tools Used**: {{tools_used}}
**Duration**: {{duration_seconds}}s
**Completion Signal**: {{completion_signal}}

## Scoring Dimensions

Rate each dimension from 1-5:

### 1. Efficiency (1-5)
How directly did the agent accomplish the task?

- **5**: Optimal path, minimal unnecessary steps
- **4**: Mostly direct, minor diversions
- **3**: Some wandering but eventually got there
- **2**: Significant inefficiency or repeated attempts
- **1**: Very roundabout, excessive tool calls

### 2. Correctness (1-5)
Did the solution actually work?

- **5**: Perfect solution, no issues
- **4**: Works with minor imperfections
- **3**: Mostly works, some edge cases missed
- **2**: Partially works, significant issues
- **1**: Doesn't work or made things worse

### 3. Communication (1-5)
How well did the agent explain what it was doing?

- **5**: Clear, educational, appropriate detail level
- **4**: Good explanations, occasionally verbose/terse
- **3**: Adequate but not exceptional
- **2**: Unclear or confusing at times
- **1**: Poor communication, hard to follow

### 4. Autonomy (1-5)
How independently did the agent work?

- **5**: Fully autonomous, needed no hand-holding
- **4**: Mostly independent, occasional clarification
- **3**: Required some guidance
- **2**: Needed significant user intervention
- **1**: Required constant direction

### 5. Friction (1-5, inverted - higher is better)
How smoothly did the interaction go?

- **5**: No friction, smooth throughout
- **4**: Minor friction, quickly resolved
- **3**: Some back-and-forth needed
- **2**: Multiple corrections or misunderstandings
- **1**: High friction, frustrating interaction

## Output Format

```json
{
  "efficiency": 4,
  "correctness": 5,
  "communication": 4,
  "autonomy": 5,
  "friction": 4,
  "overall_score": 4.4,
  "strengths": ["Good tool usage", "Clear explanations"],
  "weaknesses": ["Slightly verbose"],
  "notes": "Solid performance on a moderate complexity task"
}
```

## Task Content

{{task_content}}
