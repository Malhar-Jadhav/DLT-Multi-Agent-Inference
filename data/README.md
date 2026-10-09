# data

Dataset download, preprocessing, and loading for all experiments.

| Path | What it holds |
|---|---|
| `dataset_config.py` | Registry of datasets, their dependencies, and the raw and processed folder paths. Register new datasets here. |
| `prepare_all.py` | Builds every dataset (or a chosen subset) in dependency order. |
| `loader.py` | `load_split(name, split)`, the function experiment code uses to read processed data. |
| `common/` | Shared helpers for JSONL files, seeded sampling, manifests, and Hub downloads. |
| `gsm8k/` | GSM8K for the debate system. |
| `humanevalplus/` | HumanEval+ for the group chat system. |
| `bugged_solutions/` | Verified buggy solutions for the critic catch rate. |
| `chatdev_srdd/` | SRDD requirements for the ChatDev stretch goal. |
| `raw/` | Downloaded source files. Git-ignored. |
| `processed/` | `eval.jsonl`, `dev.jsonl`, and `manifest.json` per dataset. Git-ignored. |

Each dataset folder has a `download.py` and a `preprocess.py`. Build everything with
`python -m data.prepare_all`, or one dataset with `python -m data.prepare_all --datasets gsm8k`.
Selection is seeded (default 0), so everyone gets the same splits.
