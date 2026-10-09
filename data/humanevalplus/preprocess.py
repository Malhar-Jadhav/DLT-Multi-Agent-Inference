"""Turn raw HumanEval+ into the eval and dev splits used by the group chat experiments.

Splits:
  * eval: all 164 problems, as preregistered.
  * dev: 30 problems sampled by seed, used to freeze the model-tiering assignments.
    NOTE: the preregistration evaluates all 164 problems, so the dev subset overlaps eval.
    The team should decide whether to exclude dev ids from reported results.

Each processed row has: id (the original task_id), prompt (function signature and docstring),
entry_point, canonical_solution, and test (a check(candidate) function with the extended
HumanEval+ tests). build_program assembles a runnable script for scoring a candidate solution.

Usage:
    python -m data.humanevalplus.preprocess --seed 0
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data.common.io_utils import read_jsonl, seeded_sample, write_jsonl, write_manifest
from data.dataset_config import DEFAULT_SEED, processed_dir, raw_dir

DEV_SIZE = 30


def to_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one raw HumanEval+ record into a processed row."""
    return {
        "id": raw["task_id"],
        "prompt": raw["prompt"],
        "entry_point": raw["entry_point"],
        "canonical_solution": raw["canonical_solution"],
        "test": raw["test"],
    }


def build_program(row: Dict[str, Any], completion: str, include_prompt: bool = True) -> str:
    """Assemble a runnable script that checks a completion against the HumanEval+ tests.

    If include_prompt is True the completion is treated as a function body and appended to the
    prompt. If it is False the completion is treated as a full function definition.
    """
    code = row["prompt"] + completion if include_prompt else completion
    return f"{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"


def preprocess(raw_path: Path, out_dir: Path, seed: int = DEFAULT_SEED) -> None:
    """Build eval.jsonl, dev.jsonl and manifest.json from the raw HumanEval+ file."""
    rows = [to_row(r) for r in read_jsonl(raw_path / "test.jsonl")]
    splits: Dict[str, List[Dict[str, Any]]] = {
        "eval": rows,
        "dev": seeded_sample(rows, DEV_SIZE, seed),
    }
    for name, split_rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", split_rows)
    write_manifest(out_dir, "humanevalplus", seed, splits, notes="dev is a subset of eval, see module docstring")
    print(f"HumanEval+ processed files written to {out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Preprocess HumanEval+ into eval and dev splits.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    preprocess(raw_dir("humanevalplus"), processed_dir("humanevalplus"), args.seed)


if __name__ == "__main__":
    main()
