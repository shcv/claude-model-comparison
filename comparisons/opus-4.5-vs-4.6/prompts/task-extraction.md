# Task Extraction Prompt

You are analyzing a Claude Code session to identify discrete tasks. A "task" is a coherent unit of work from user request to completion.

## Session Context

Model: {{model}}
Session ID: {{session_id}}
Duration: {{duration_minutes}} minutes

## Instructions

Analyze the session and identify each task. For each task, provide:

### Task Identification

1. **Task Description**: What the user asked for (1-2 sentences)
2. **Complexity Classification**:
   - `trivial`: Single file read, simple question, quick lookup
   - `simple`: Single file edit, straightforward implementation
   - `moderate`: Multi-file changes, requires exploration
   - `complex`: Architectural changes, debugging, iterative refinement

### Task Completion

3. **Completion Status**:
   - `success`: Task completed as requested
   - `partial`: Some work done but incomplete
   - `failed`: Task could not be completed
   - `abandoned`: User moved on without completion

4. **How Task Ended**:
   - User explicit satisfaction ("thanks", "perfect")
   - User explicit dissatisfaction ("wrong", "try again")
   - User moved to new topic
   - Session ended
   - Agent stopped (no more tool calls)

### Quality Indicators

5. **Friction Points**: Any misunderstandings, retries, or corrections needed
6. **Notable Behaviors**: Anything unusual (good or bad) about how the agent handled this

## Output Format

Return a JSON array of tasks:

```json
[
  {
    "task_number": 1,
    "description": "Add error handling to the API endpoint",
    "complexity": "moderate",
    "completion_status": "success",
    "ending_signal": "user_satisfied",
    "tool_calls": 12,
    "friction_points": ["Initially missed edge case, corrected after user feedback"],
    "notable_behaviors": ["Good use of parallel tool calls"]
  }
]
```

## Session Content

{{session_content}}
