# Claude Model Comparison

Comparative behavioral analysis of Claude model pairs using Claude Code session logs.

## Project Structure

```
scripts/                         # Reusable analysis pipeline
replication/                     # Controlled replication tasks
comparisons/<model-a>-vs-<model-b>/
    data/                        # Session metadata, classified tasks, tokens (private)
    analysis/                    # Statistical results (mix of private/public)
    prompts/                     # Report generation prompts
    dist/public/report.html      # The generated report
```

## Running a Comparison

Most scripts accept `--data-dir` and `--analysis-dir` arguments pointing to the comparison directory:

```sh
python scripts/collect_sessions.py --data-dir comparisons/opus-4.5-vs-4.6/data
python scripts/extract_tasks.py --data-dir comparisons/opus-4.5-vs-4.6/data
python scripts/classify_tasks.py --data-dir comparisons/opus-4.5-vs-4.6/data
python scripts/analyze_behavior.py --data-dir comparisons/opus-4.5-vs-4.6/data
python scripts/extract_tokens.py --dir comparisons/opus-4.5-vs-4.6
python scripts/stat_tests.py --data-dir comparisons/opus-4.5-vs-4.6/data --analysis-dir comparisons/opus-4.5-vs-4.6/analysis
```

## Adding a New Comparison

1. Create `comparisons/<model-a>-vs-<model-b>/` with `data/`, `analysis/`, `prompts/`, `dist/public/` subdirectories
2. Collect sessions for each model into `data/sessions-<model>.json`
3. Run the pipeline scripts above
4. Customize `prompts/report-generation.md` for the specific findings
5. Generate the report

## Privacy

The `data/` directory contains session metadata referencing local file paths and user prompts. These are gitignored. Aggregated analysis files (stat-tests.json, token-analysis.json, etc.) contain only statistical summaries and are safe to commit.

## Dependencies

- Python 3.11+
- scipy (for stat_tests.py)
- Claude Code SDK (for LLM classification steps)
