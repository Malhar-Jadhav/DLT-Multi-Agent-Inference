"""Download and preprocess every dataset (or a chosen subset) in one command.

Datasets are built in dependency order, so bugged_solutions runs after humanevalplus.
The ChatDev SRDD stretch dataset is skipped unless --include-stretch is given.

Usage:
    python -m data.prepare_all                          # core datasets
    python -m data.prepare_all --include-stretch        # also ChatDev SRDD
    python -m data.prepare_all --datasets gsm8k         # one dataset
    python -m data.prepare_all --skip-download          # reuse files already in data/raw
"""

import argparse
import importlib
from typing import List, Optional

from data.dataset_config import DEFAULT_SEED, DATASETS, list_datasets, processed_dir, raw_dir


def resolve_order(names: List[str]) -> List[str]:
    """Expand a list of dataset names with their dependencies, in the order they must be built."""
    ordered: List[str] = []

    def visit(name: str) -> None:
        """Add a dataset to the order after first adding everything it requires."""
        if name in ordered:
            return
        for dep in DATASETS[name].requires:
            visit(dep)
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


def build(name: str, seed: int, skip_download: bool) -> None:
    """Run the download and preprocess steps of one dataset package."""
    print(f"== {name} ==")
    if not skip_download:
        importlib.import_module(f"data.{name}.download").download(raw_dir(name))
    importlib.import_module(f"data.{name}.preprocess").preprocess(raw_dir(name), processed_dir(name), seed)


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and build the requested datasets."""
    parser = argparse.ArgumentParser(description="Download and preprocess project datasets.")
    parser.add_argument("--datasets", nargs="*", help="Dataset names (default: all core datasets).")
    parser.add_argument("--include-stretch", action="store_true", help="Also build the ChatDev SRDD dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Selection seed (preregistered value is 0).")
    parser.add_argument("--skip-download", action="store_true", help="Preprocess files already in data/raw.")
    args = parser.parse_args(argv)

    names = args.datasets or [d.name for d in list_datasets(include_stretch=args.include_stretch)]
    for name in resolve_order(names):
        build(name, args.seed, args.skip_download)


if __name__ == "__main__":
    main()
