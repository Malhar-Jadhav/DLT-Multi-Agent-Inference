# tests

Unit tests run with pytest. None of them need a GPU, a network connection, or a running server.

| File | What it covers |
|---|---|
| `test_models.py` | Model registry, vLLM launch command, and endpoint parsing. |
| `test_data.py` | Seeded sampling, JSONL helpers, GSM8K answer parsing, bug injection, program runner, and build order. |
| `test_debate.py` | Debate prompts (identical to the released code), round structure, voting, scoring, and the L1 and L2 switches. |
| `test_tracer.py` | Per-task latency breakdown, critical path, files written, and the run summary. |
| `test_runner.py` | Closed-loop concurrency limit, error handling, seeds, and resume bookkeeping. |
| `fake_llm.py` | Not a test: a fake model with the `LLMClient` interface that the three files above use. |

Run them from the repository root with `python -m pytest`.
