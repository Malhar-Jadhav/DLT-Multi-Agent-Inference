"""Small I/O and sampling helpers shared by every dataset script.

Keeps JSONL reading and writing, seeded subset selection, and manifest writing in one place
so all datasets are sampled and recorded the same way.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write dict rows to a JSONL file (creating parent folders) and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seeded_sample(items: Sequence[Any], k: int, seed: int) -> List[Any]:
    """Pick k items without replacement using a private RNG, so the same seed gives the same subset.

    The original order of the items is preserved in the result, which keeps files diff-friendly.
    """
    if k > len(items):
        raise ValueError(f"Cannot sample {k} items from a pool of {len(items)}")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in chosen]


def ids_fingerprint(ids: Iterable[str]) -> str:
    """Return a short hash of a list of ids, used to confirm two machines built the same split."""
    digest = hashlib.sha256("\n".join(str(i) for i in ids).encode("utf-8")).hexdigest()
    return digest[:16]


def write_manifest(out_dir: Path, name: str, seed: int, splits: Dict[str, List[Dict[str, Any]]], notes: str = "") -> None:
    """Write manifest.json next to the processed splits.

    The manifest records the dataset name, selection seed, size of each split, and an id
    fingerprint per split. Compare fingerprints to verify that everyone has identical data.
    """
    manifest = {
        "dataset": name,
        "seed": seed,
        "splits": {
            split: {"count": len(rows), "ids_fingerprint": ids_fingerprint(r["id"] for r in rows)}
            for split, rows in splits.items()
        },
        "notes": notes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
