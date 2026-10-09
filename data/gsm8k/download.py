"""Download GSM8K (openai/gsm8k, "main" config) into data/raw/gsm8k/ as train.jsonl and test.jsonl.

Usage:
    python -m data.gsm8k.download
"""

from pathlib import Path

from data.common.hf_utils import dump_hf_dataset
from data.dataset_config import raw_dir


def download(out_dir: Path) -> None:
    """Fetch every GSM8K split from the Hugging Face Hub and write it as JSONL under out_dir."""
    counts = dump_hf_dataset("openai/gsm8k", out_dir, config="main")
    print(f"GSM8K raw files written to {out_dir}: {counts}")


if __name__ == "__main__":
    download(raw_dir("gsm8k"))
