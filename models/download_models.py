"""Download model weights from the Hugging Face Hub into the local cache.

Run this once on a node with network access (on ICE, through slurm/download_models.sbatch)
so that the vLLM servers can start without downloading anything.

Usage:
    python -m models.download_models                      # default large + small pair
    python -m models.download_models --models qwen3-14b phi-4-mini
    python -m models.download_models --all                # every registered model
"""

import argparse
from typing import List, Optional

from models.model_config import ModelSpec, MODELS, default_pair, get_model


def select_models(keys: Optional[List[str]], download_all: bool) -> List[ModelSpec]:
    """Resolve command line choices into a list of ModelSpec objects."""
    if download_all:
        return list(MODELS.values())
    if keys:
        return [get_model(k) for k in keys]
    return list(default_pair())


def download(spec: ModelSpec, cache_dir: Optional[str] = None) -> str:
    """Download one model snapshot and return the local directory it was stored in."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {spec.key} ({spec.hf_id})")
    return snapshot_download(repo_id=spec.hf_id, cache_dir=cache_dir)


def main() -> None:
    """Parse arguments and download every requested model."""
    parser = argparse.ArgumentParser(description="Download project models from the Hugging Face Hub.")
    parser.add_argument("--models", nargs="*", help="Registry keys to download (default: large + small pair).")
    parser.add_argument("--all", action="store_true", help="Download every registered model.")
    parser.add_argument("--cache-dir", default=None, help="Override the Hugging Face cache directory.")
    args = parser.parse_args()

    for spec in select_models(args.models, args.all):
        path = download(spec, args.cache_dir)
        print(f"  stored at {path}")


if __name__ == "__main__":
    main()
