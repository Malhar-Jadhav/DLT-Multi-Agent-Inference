# tests

Unit tests run with pytest. None of them need a GPU, a network connection, or a running server.

| File | What it covers |
|---|---|
| `test_models.py` | Model registry, vLLM launch command, and endpoint parsing. |
| `test_data.py` | Seeded sampling, JSONL helpers, GSM8K answer parsing, bug injection, program runner, and build order. |

Run them from the repository root with `python -m pytest`.
