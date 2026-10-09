"""Helper for dumping a Hugging Face dataset to local JSONL files.

Raw data is stored as plain JSONL (one file per split) so that preprocessing scripts do not
depend on the datasets library and can be rerun offline once the download is done.
"""

from pathlib import Path
from typing import Dict, Optional

from data.common.io_utils import write_jsonl


def dump_hf_dataset(repo_id: str, out_dir: Path, config: Optional[str] = None) -> Dict[str, int]:
    """Download every split of a Hub dataset and write each as <split>.jsonl in out_dir.

    Returns a dict of split name to row count so callers can print a summary.
    """
    from datasets import load_dataset

    dataset = load_dataset(repo_id, config) if config else load_dataset(repo_id)
    counts = {}
    for split_name, split in dataset.items():
        counts[split_name] = write_jsonl(out_dir / f"{split_name}.jsonl", (dict(row) for row in split))
    return counts
