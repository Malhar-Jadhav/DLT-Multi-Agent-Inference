"""Download HumanEval+ (evalplus/humanevalplus on the Hub) into data/raw/humanevalplus/test.jsonl.

Usage:
    python -m data.humanevalplus.download
"""

from pathlib import Path

from data.common.hf_utils import dump_hf_dataset
from data.dataset_config import raw_dir


def download(out_dir: Path) -> None:
    """Fetch the 164 HumanEval+ problems from the Hugging Face Hub and write them as JSONL."""
    counts = dump_hf_dataset("evalplus/humanevalplus", out_dir)
    print(f"HumanEval+ raw files written to {out_dir}: {counts}")


if __name__ == "__main__":
    download(raw_dir("humanevalplus"))
