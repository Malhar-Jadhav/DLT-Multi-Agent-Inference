"""Load processed dataset splits from experiment code.

This is the only function experiment code needs from the data layer:

    from data.loader import load_split
    problems = load_split("gsm8k", "eval")

Run data/prepare_all.py first to create the processed files.
"""

from typing import Any, Dict, List

from data.common.io_utils import read_jsonl
from data.dataset_config import get_dataset, processed_dir


def load_split(name: str, split: str = "eval") -> List[Dict[str, Any]]:
    """Return the rows of a processed split ("eval" or "dev") as a list of dicts."""
    get_dataset(name)
    path = processed_dir(name) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m data.prepare_all --datasets {name}")
    return read_jsonl(path)
