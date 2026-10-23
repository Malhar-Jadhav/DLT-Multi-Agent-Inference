"""Load sweep: rerun C0, C1, C5 at concurrency 1/5/10/20/32 to test H2.

H2: L1's add-one gain at concurrency 32 is less than half its gain at concurrency 1.

Usage:
    python -m loadgen.sweep \
        --config configs/load_sweep.yaml \
        --seed 0 \
        --results-dir results/load_sweep
"""

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from harness.runner import run_experiment

log = logging.getLogger(__name__)


async def run_sweep(yaml_path: Path, seed: int, results_dir: Path) -> None:
    with yaml_path.open() as f:
        doc = yaml.safe_load(f)

    concurrency_levels = doc.get("concurrency_levels", [1, 5, 10, 20, 32])
    configs = [c["name"] for c in doc["configs"]]

    tasks = []
    for cfg_name in configs:
        for conc in concurrency_levels:
            run_dir = results_dir / f"{cfg_name}_seed{seed}_c{conc}"
            tasks.append((cfg_name, conc, run_dir))

    # Run sequentially to avoid saturating the GPU with sweep + experiment concurrency.
    for cfg_name, conc, run_dir in tasks:
        log.info("Sweep: config=%s seed=%d concurrency=%d → %s", cfg_name, seed, conc, run_dir)
        await run_experiment(
            yaml_path=yaml_path,
            config_name=cfg_name,
            seed=seed,
            run_dir=run_dir,
            concurrency=conc,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Load sweep for H2.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--results-dir", type=Path, default=Path("results/load_sweep"))
    args = p.parse_args()
    asyncio.run(run_sweep(args.config, args.seed, args.results_dir))


if __name__ == "__main__":
    main()
