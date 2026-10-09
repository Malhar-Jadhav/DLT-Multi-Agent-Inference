"""Placeholder download step for the bugged solutions dataset.

This dataset is derived from HumanEval+ rather than downloaded, so there is nothing to fetch.
The function exists so that data/prepare_all.py can treat every dataset the same way.
"""

from pathlib import Path


def download(out_dir: Path) -> None:
    """Do nothing except explain that the data is derived from humanevalplus."""
    print("bugged_solutions is derived from humanevalplus, nothing to download.")


if __name__ == "__main__":
    download(Path("."))
