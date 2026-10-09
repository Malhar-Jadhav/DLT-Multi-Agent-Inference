# data/common

Helpers shared by every dataset script.

| File | What it does |
|---|---|
| `io_utils.py` | Reads and writes JSONL, picks seeded random subsets, fingerprints id lists, and writes `manifest.json`. |
| `hf_utils.py` | Dumps a Hugging Face dataset to local JSONL files, one per split. |
