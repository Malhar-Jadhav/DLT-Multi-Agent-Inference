"""Shared experiment runner.

Usage (from the repo root):

    python -m harness.runner \
        --config configs/ablation_debate.yaml \
        --run-dir results/runs/debate_C1_seed0 \
        --config-name C1 \
        --seed 0

The runner:
  1. Loads the YAML config and selects the requested LeverConfig by name.
  2. Builds LLMClient(s) for the config.
  3. Loads the dataset split.
  4. Runs every task through the appropriate harness (debate or groupchat).
  5. Writes a manifest.json and traces.jsonl to run_dir.
  6. Runs the equivalence check (if a baseline trace exists in the same results tree).

Run the baseline (C0) first so later configs can be checked against it.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from data.loader import load_split
from harness.levers import LeverConfig, build_clients
from harness.tracer import Tracer

log = logging.getLogger(__name__)


def _load_config_yaml(path: Path, config_name: str) -> tuple[Dict[str, Any], LeverConfig]:
    """Return the top-level YAML dict and the LeverConfig for the named config."""
    with path.open() as f:
        doc = yaml.safe_load(f)
    matches = [c for c in doc["configs"] if c["name"] == config_name]
    if not matches:
        raise ValueError(f"Config '{config_name}' not found in {path}. "
                         f"Available: {[c['name'] for c in doc['configs']]}")
    return doc, LeverConfig.from_dict(matches[0])


def _run_id(system: str, config_name: str, seed: int, concurrency: int) -> str:
    return f"{system}_{config_name}_seed{seed}_c{concurrency}"


async def run_experiment(
    yaml_path: Path,
    config_name: str,
    seed: int,
    run_dir: Path,
    concurrency: int = 1,
    dry_run: bool = False,
) -> Path:
    """Execute one (system, config, seed, concurrency) experiment and return run_dir."""
    doc, lever_cfg = _load_config_yaml(yaml_path, config_name)
    system = doc["system"]
    dataset = doc["dataset"]
    split = doc.get("split", "eval")
    large_model = doc.get("large_model", "qwen3-14b")
    small_model = doc.get("small_model", "qwen3-1.7b")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest before running so it's there even if the run crashes.
    manifest = {
        "run_id": _run_id(system, config_name, seed, concurrency),
        "system": system,
        "dataset": dataset,
        "split": split,
        "config": config_name,
        "levers": lever_cfg.to_dict(),
        "seed": seed,
        "concurrency": concurrency,
        "large_model": large_model,
        "small_model": small_model if lever_cfg.l4_tier else None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if dry_run:
        log.info("Dry run: manifest written to %s, exiting.", run_dir)
        return run_dir

    tasks = load_split(dataset, split)
    large, small = build_clients(lever_cfg, large_model, small_model)

    with Tracer(run_dir) as tracer:
        if system == "debate":
            from harness.debate import run_debate_tasks
            await run_debate_tasks(
                tasks=tasks, lever_cfg=lever_cfg,
                large_client=large, small_client=small,
                tracer=tracer, seed=seed, concurrency=concurrency,
            )
        elif system == "groupchat":
            from harness.groupchat import run_groupchat_tasks
            await run_groupchat_tasks(
                tasks=tasks, lever_cfg=lever_cfg,
                large_client=large, small_client=small,
                tracer=tracer, seed=seed, concurrency=concurrency,
            )
        else:
            raise ValueError(f"Unknown system '{system}'. Expected 'debate' or 'groupchat'.")

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Run complete: %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Run one ablation experiment.")
    p.add_argument("--config", required=True, type=Path,
                   help="Path to ablation YAML (e.g. configs/ablation_debate.yaml)")
    p.add_argument("--config-name", required=True,
                   help="Config name to run, e.g. C0 or C5")
    p.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Directory to write traces.jsonl and manifest.json")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of tasks to run in parallel (for the load sweep)")
    p.add_argument("--dry-run", action="store_true",
                   help="Write manifest only, do not call models")
    args = p.parse_args()

    asyncio.run(run_experiment(
        yaml_path=args.config,
        config_name=args.config_name,
        seed=args.seed,
        run_dir=args.run_dir,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
