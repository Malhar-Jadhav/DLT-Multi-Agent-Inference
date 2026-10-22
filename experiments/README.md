# experiments

Scripts that run a system under a configuration against a served model and record the results.

| File | What it does |
|---|---|
| `run_debate.py` | Runs the GSM8K debate with C debates in flight, writes `config.json`, `calls.jsonl`, `tasks.jsonl`, `transcripts.jsonl` and `summary.json` to `logs/runs/<run-name>/`. Supports `--resume`. |

Run from the repository root with `python -m experiments.run_debate`. See the "Running the
debate" section of the main README for examples.
