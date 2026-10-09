"""Turn raw GSM8K files into the eval and dev splits used by the debate experiments.

Splits (selection is fixed by the seed, preregistered as 0):
  * eval: 200 problems sampled from the GSM8K test split.
  * dev: 30 problems sampled from the GSM8K train split, used to freeze the model-tiering
    assignments. Using train keeps the dev set disjoint from the eval set.

Each processed row has: id, question, solution (full reference reasoning), and answer (the
final number as a clean string). The helpers parse_gold_answer and extract_final_number are
also what the debate harness should use to score model output by exact match.

Usage:
    python -m data.gsm8k.preprocess --seed 0
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.common.io_utils import read_jsonl, seeded_sample, write_jsonl, write_manifest
from data.dataset_config import DEFAULT_SEED, processed_dir, raw_dir

EVAL_SIZE = 200
DEV_SIZE = 30

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(text: str) -> str:
    """Strip commas, dollar signs and trailing zeros so that "1,200.0" and "1200" compare equal."""
    cleaned = text.replace(",", "").replace("$", "").strip()
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def parse_gold_answer(answer_field: str) -> str:
    """Extract the final answer from a raw GSM8K answer string, which ends with "#### <number>"."""
    if "####" not in answer_field:
        raise ValueError("GSM8K answer has no '####' marker")
    return normalize_number(answer_field.split("####")[-1])


def extract_final_number(model_output: str) -> Optional[str]:
    """Return the last number in a model's output as a normalized string, or None if there is none.

    This is the exact-match rule used for scoring: the final number in the response is the answer.
    """
    matches = _NUMBER.findall(model_output)
    return normalize_number(matches[-1]) if matches else None


def to_row(index: int, split: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one raw GSM8K record into a processed row with a stable id."""
    return {
        "id": f"gsm8k-{split}-{index}",
        "question": raw["question"].strip(),
        "solution": raw["answer"].split("####")[0].strip(),
        "answer": parse_gold_answer(raw["answer"]),
    }


def preprocess(raw_path: Path, out_dir: Path, seed: int = DEFAULT_SEED) -> None:
    """Build eval.jsonl, dev.jsonl and manifest.json from the raw GSM8K files."""
    test_rows = [to_row(i, "test", r) for i, r in enumerate(read_jsonl(raw_path / "test.jsonl"))]
    train_rows = [to_row(i, "train", r) for i, r in enumerate(read_jsonl(raw_path / "train.jsonl"))]

    splits: Dict[str, List[Dict[str, Any]]] = {
        "eval": seeded_sample(test_rows, EVAL_SIZE, seed),
        "dev": seeded_sample(train_rows, DEV_SIZE, seed),
    }
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    write_manifest(out_dir, "gsm8k", seed, splits, notes="eval from test split, dev from train split")
    print(f"GSM8K processed files written to {out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Preprocess GSM8K into eval and dev splits.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    preprocess(raw_dir("gsm8k"), processed_dir("gsm8k"), args.seed)


if __name__ == "__main__":
    main()
