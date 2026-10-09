"""Registry of the datasets used in this project and where their files live.

Each dataset has its own folder under data/ with a download.py and a preprocess.py that
follow the same two-function interface:

    download(raw_dir: Path) -> None
    preprocess(raw_dir: Path, out_dir: Path, seed: int) -> None

Raw files go to data/raw/<name>/ and processed splits go to data/processed/<name>/. Both
folders are git-ignored; rebuild them with data/prepare_all.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DATA_ROOT = Path(__file__).resolve().parent
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

DEFAULT_SEED = 0


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one dataset.

    name: folder name and registry key.
    system: which multi-agent system uses it.
    description: one line summary of the processed splits.
    requires: names of datasets that must be built first (empty for most).
    stretch: True for the optional ChatDev stretch goal.
    """

    name: str
    system: str
    description: str
    requires: tuple = ()
    stretch: bool = False


DATASETS: Dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        "gsm8k", "debate",
        "200 seeded test problems (eval) plus 30 train problems (dev) for the multi-agent debate.",
    ),
    "humanevalplus": DatasetSpec(
        "humanevalplus", "groupchat",
        "All 164 HumanEval+ problems (eval) plus a seeded 30 problem dev subset for the AutoGen group chat.",
    ),
    "bugged_solutions": DatasetSpec(
        "bugged_solutions", "groupchat",
        "50 canonical HumanEval+ solutions with an injected bug, used to measure the critic catch rate.",
        requires=("humanevalplus",),
    ),
    "chatdev_srdd": DatasetSpec(
        "chatdev_srdd", "chatdev",
        "50 seeded SRDD software requirements (eval) plus 30 more (dev) for the ChatDev stretch goal.",
        stretch=True,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    """Return the DatasetSpec for a name, with a helpful error for unknown names."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(DATASETS))}")
    return DATASETS[name]


def raw_dir(name: str) -> Path:
    """Folder holding the raw downloaded files for a dataset."""
    return RAW_ROOT / name


def processed_dir(name: str) -> Path:
    """Folder holding the processed splits and manifest for a dataset."""
    return PROCESSED_ROOT / name


def list_datasets(include_stretch: bool = True) -> List[DatasetSpec]:
    """List registered datasets, optionally leaving out the stretch goal."""
    return [d for d in DATASETS.values() if include_stretch or not d.stretch]
