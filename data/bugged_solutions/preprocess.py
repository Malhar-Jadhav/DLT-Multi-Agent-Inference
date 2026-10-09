"""Build the 50 bugged solutions used to measure the critic catch rate.

For HumanEval+ problems taken in a seeded random order, the script injects one bug (operator
or off-by-one, see mutations.py) into the canonical solution and keeps the mutant only if it
really is wrong: the canonical solution must pass the HumanEval+ tests and the mutant must
fail them (a timeout does not count as a failure). Candidate sites are tried in a seeded
order until one produces a valid failing mutant. Generated code is executed in a separate
Python subprocess with a timeout.

Each processed row has: id, task_id, prompt, entry_point, buggy_solution (full program
source including the signature), mutation_kind, and mutation_description. Only an "eval"
split of 50 rows is written.

Requires data/processed/humanevalplus to exist (data/prepare_all.py handles the order).

Usage:
    python -m data.bugged_solutions.preprocess --seed 0 --count 50
"""

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.bugged_solutions.mutations import count_sites, mutate_source
from data.common.io_utils import read_jsonl, write_jsonl, write_manifest
from data.dataset_config import DEFAULT_SEED, processed_dir
from data.humanevalplus.preprocess import build_program

TARGET_COUNT = 50
TIMEOUT_SECONDS = 20


def run_program(source: str, timeout: int = TIMEOUT_SECONDS) -> str:
    """Execute a program in an isolated subprocess and return "pass", "fail" or "timeout"."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-E", str(script)], capture_output=True, timeout=timeout, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return "timeout"
    return "pass" if proc.returncode == 0 else "fail"


def make_bugged_row(problem: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Try to build one verified bugged solution for a problem, returning None if impossible."""
    canonical = problem["prompt"] + problem["canonical_solution"]
    if run_program(build_program(problem, canonical, include_prompt=False)) != "pass":
        return None
    sites = list(range(count_sites(canonical)))
    rng.shuffle(sites)
    for site_index in sites:
        try:
            mutant, kind, description = mutate_source(canonical, site_index)
        except (SyntaxError, IndexError, ValueError):
            continue
        if run_program(build_program(problem, mutant, include_prompt=False)) == "fail":
            return {
                "id": f"bug-{problem['id']}",
                "task_id": problem["id"],
                "prompt": problem["prompt"],
                "entry_point": problem["entry_point"],
                "buggy_solution": mutant,
                "mutation_kind": kind,
                "mutation_description": description,
            }
    return None


def preprocess(raw_path: Path, out_dir: Path, seed: int = DEFAULT_SEED, count: int = TARGET_COUNT) -> None:
    """Build eval.jsonl and manifest.json with `count` verified bugged solutions.

    raw_path is unused because the input is the processed HumanEval+ split; the argument is kept
    so that every dataset shares the same preprocess signature.
    """
    problems = read_jsonl(processed_dir("humanevalplus") / "eval.jsonl")
    rng = random.Random(seed)
    order = list(range(len(problems)))
    rng.shuffle(order)

    rows: List[Dict[str, Any]] = []
    for i in order:
        row = make_bugged_row(problems[i], rng)
        if row is not None:
            rows.append(row)
        if len(rows) == count:
            break
    if len(rows) < count:
        raise RuntimeError(
            f"Only built {len(rows)} verified bugged solutions, needed {count}. "
            "If this is 0, check that numpy is installed, since the HumanEval+ tests import it."
        )

    write_jsonl(out_dir / "eval.jsonl", rows)
    write_manifest(out_dir, "bugged_solutions", seed, {"eval": rows}, notes="each mutant verified to fail HumanEval+ tests")
    print(f"Bugged solutions written to {out_dir}: eval={len(rows)}")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Build verified bugged HumanEval+ solutions.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    args = parser.parse_args()
    preprocess(Path("."), processed_dir("bugged_solutions"), args.seed, args.count)


if __name__ == "__main__":
    main()
