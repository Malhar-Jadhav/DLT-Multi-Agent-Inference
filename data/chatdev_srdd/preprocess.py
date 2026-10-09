"""Turn the raw SRDD table into eval and dev splits for the ChatDev stretch goal.

Splits (selection is fixed by the seed):
  * eval: 50 requirements sampled from the 1200 rows.
  * dev: 30 further requirements sampled from the rows not used in eval.

Each processed row has: id, name, description (the software requirement given to ChatDev),
and category.

Usage:
    python -m data.chatdev_srdd.preprocess --seed 0
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from data.common.io_utils import seeded_sample, write_jsonl, write_manifest
from data.dataset_config import DEFAULT_SEED, processed_dir, raw_dir

EVAL_SIZE = 50
DEV_SIZE = 30


def read_srdd(csv_path: Path) -> List[Dict[str, Any]]:
    """Read the SRDD CSV into processed rows with stable ids based on the row position."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [
            {
                "id": f"srdd-{i}",
                "name": r["Name"].strip(),
                "description": r["Description"].strip(),
                "category": r["Category"].strip(),
            }
            for i, r in enumerate(csv.DictReader(f))
        ]


def preprocess(raw_path: Path, out_dir: Path, seed: int = DEFAULT_SEED) -> None:
    """Build eval.jsonl, dev.jsonl and manifest.json from the raw SRDD CSV."""
    rows = read_srdd(raw_path / "srdd.csv")
    eval_rows = seeded_sample(rows, EVAL_SIZE, seed)
    eval_ids = {r["id"] for r in eval_rows}
    remaining = [r for r in rows if r["id"] not in eval_ids]
    splits = {"eval": eval_rows, "dev": seeded_sample(remaining, DEV_SIZE, seed)}
    for name, split_rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", split_rows)
    write_manifest(out_dir, "chatdev_srdd", seed, splits, notes="dev is disjoint from eval")
    print(f"SRDD processed files written to {out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Preprocess SRDD into eval and dev splits.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    preprocess(raw_dir("chatdev_srdd"), processed_dir("chatdev_srdd"), args.seed)


if __name__ == "__main__":
    main()
