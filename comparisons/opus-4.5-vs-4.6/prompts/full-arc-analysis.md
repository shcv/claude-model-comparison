# Full Arc Task Analysis Prompt

You are analyzing a Claude Code task interaction to provide a rich qualitative summary that considers the **full arc**: user request → agent work → user response.

## Why Both User Messages Matter

Traditional summarization only looks at what the agent did. But to understand *quality*, we need:

1. **Initial Request**: What did the user want?
2. **Agent Work**: What did the agent do?
3. **User Response**: How did the user react?

The user's *next* message is the ground truth for task success - it tells us whether the agent actually achieved the goal.

## Sentiment Categories

### Explicit Satisfaction
- "thanks", "perfect", "looks good", "exactly what I needed"
- High confidence interpretation

### Implicit Satisfaction
- Session ends with no follow-up after simple task
- User moves to unrelated new topic
- "sounds good" followed by continuation request
- Medium confidence interpretation

### Collaborative Refinement
- User provides technical corrections or clarifications
- "actually, we should..." or "let's also..."
- NOT dissatisfaction - normal iteration
- Watch for: domain knowledge being shared

### Explicit Dissatisfaction
- "that's wrong", "try again", "not what I asked"
- "undo", "revert", "that broke things"
- High confidence interpretation

### Ambiguous
- Very short responses ("ok", "hmm")
- Questions that could be curiosity or confusion
- Session ends mid-task
- Low confidence interpretation

## Work Type Categories

Beyond simple heuristics (bugfix, feature, etc.), characterize:

### Investigation/Exploration
- Pure reading, no changes
- Answering questions about codebase
- Often delegated to subagents

### Directed Implementation
- User provides clear spec or plan
- Agent executes with minimal interpretation
- High autonomy within clear bounds

### Creative Implementation
- User provides goal, agent designs approach
- More autonomous decision-making
- Higher risk of misalignment

### Verification/QA
- Checking code, running tests
- Validating changes work
- Often precedes or follows implementation

### Correction/Refinement
- Fixing issues found in prior work
- Simplification, cleanup
- Often after user feedback

## Output Format

Return structured JSON:

```json
{
  "work_category": "1-2 sentence characterization",
  "execution_quality": "assessment with specific evidence",
  "user_sentiment": "category + interpretation",
  "sentiment_confidence": "high|medium|low",
  "sentiment_evidence": "specific quote or pattern from next_user_message",
  "follow_up_pattern": "what happened and what it suggests",
  "autonomy_level": "high|medium|low",
  "alignment_score": "1-5 (how well did agent match user intent?)",
  "summary": "2-3 sentence overall assessment"
}
```

## Key Patterns to Watch For

1. **"Sounds good" + continuation**: User approved prior context, moving forward
2. **Technical correction**: User providing domain knowledge, not complaining
3. **Session end after simple task**: Usually satisfaction
4. **Session end after complex task**: Could be reviewing locally, or abandonment
5. **Immediate new request**: Prior task was satisfactory, moving on
6. **Questions about what was done**: May indicate confusion or curiosity
7. **Plan refinement requests**: Agent's interpretation didn't match user's vision
