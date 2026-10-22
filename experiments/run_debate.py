"""Run the multi-agent debate on GSM8K against a running vLLM server and record everything.

Each run keeps exactly C debates in flight (a closed loop: when one finishes, the next
problem starts), writes the per-call and per-task traces, the full transcripts, the exact
configuration, and a summary with the preregistered metric (median task completion time).

A run folder is logs/runs/<run-name>/ (git-ignored, full traces stay out of git) and holds:
    config.json        arguments, levers, sampling, git commit, versions
    calls.jsonl        one row per model call (tracing.tracer.CallRecord)
    tasks.jsonl        one row per debate (tracing.tracer.TaskRecord)
    transcripts.jsonl  every agent's messages, answers, vote and gold answer
    summary.json       aggregate numbers

If a job is cut off, rerun the same command with --resume: problems already finished are
skipped and the summary is rebuilt from all rows. To keep a result, copy summary.json and
config.json into results/debate/ (see results/README.md).

The server must already be running (see models/serve_vllm.py). --prefix-caching only records
which mode the server was started in; the summary's cache hit fraction shows whether that
was true (it must be 0 with caching off).

Usage:
    python -m experiments.run_debate --model qwen3-14b --prefix-caching off --limit 20
    python -m experiments.run_debate --model qwen3-14b --prefix-caching on --parallel --concurrency 4
"""

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from models.model_config import DEFAULT_LARGE
from systems.debate import AsyncLLM, DebateResult, Levers, run_debate
from tracing.tracer import Tracer, load_jsonl, summarize

REPO_ROOT = Path(__file__).resolve().parent.parent


async def run_closed_loop(
    client: AsyncLLM,
    problems: List[Dict[str, Any]],
    levers: Levers,
    concurrency: int,
    tracer: Tracer,
    base_seed: int = 0,
    max_tokens: Optional[int] = None,
    indices: Optional[List[int]] = None,
    on_result: Optional[Callable[[DebateResult], None]] = None,
) -> List[DebateResult]:
    """Run debates with at most `concurrency` in flight and return the finished results.

    A debate that raises is recorded in the tracer as a task with an error, and the loop goes on.
    indices gives each problem's task index for the per-request seeds (default 0, 1, 2, ...);
    a resumed run passes the original positions so that seeds do not change.
    """
    queue: "asyncio.Queue[tuple]" = asyncio.Queue()
    for position, problem in enumerate(problems):
        queue.put_nowait((position if indices is None else indices[position], problem))
    results: List[DebateResult] = []

    async def worker() -> None:
        """Take problems from the queue until it is empty."""
        while True:
            try:
                index, problem = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await run_debate(
                    client, problem, levers, tracer, task_index=index, base_seed=base_seed, max_tokens=max_tokens,
                )
            except Exception as exc:  # keep the run going; the error is stored in tasks.jsonl
                tracer.end_task(problem["id"], error=f"{type(exc).__name__}: {exc}")
                continue
            results.append(result)
            if on_result is not None:
                on_result(result)

    await asyncio.gather(*(worker() for _ in range(max(concurrency, 1))))
    return results


def git_commit() -> str:
    """Current commit hash of the repository, or "unknown" if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def package_version(name: str) -> str:
    """Installed version of a package, or "not installed"."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not installed"


def latest_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the last row for each task_id, so a resumed run replaces an earlier errored attempt."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        by_id[row["task_id"]] = row
    return list(by_id.values())


def transcript_row(result: DebateResult) -> Dict[str, Any]:
    """The JSON row stored in transcripts.jsonl for one debate."""
    return {
        "task_id": result.task_id, "answers": result.answers, "final": result.final,
        "gold": result.gold, "correct": result.correct, "contexts": result.contexts,
    }


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON line to a file."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Command line arguments."""
    p = argparse.ArgumentParser(description="Run the GSM8K debate against a served model.")
    p.add_argument("--model", default=DEFAULT_LARGE, help="Registry key of the served model.")
    p.add_argument("--endpoint", default=None, help="Server URL, default http://localhost:<port>/v1.")
    p.add_argument("--prefix-caching", choices=["on", "off"], required=True,
                   help="How the server was started (recorded, and checked against the cache hit fraction).")
    p.add_argument("--parallel", action="store_true", help="L1: call the agents of a round concurrently.")
    p.add_argument("--coalesce", action="store_true", help="L2: send the identical round 1 prompts as one n=3 request.")
    p.add_argument("--concurrency", type=int, default=1, help="C: debates kept in flight.")
    p.add_argument("--split", default="eval", help="GSM8K split to run.")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N problems.")
    p.add_argument("--warmup", type=int, default=0, help="Dev problems run first and discarded (server warm-up).")
    p.add_argument("--seed", type=int, default=0, help="Base seed for per-request seeds (preregistered: 0, 1, 2).")
    p.add_argument("--max-tokens", type=int, default=None, help="Override the output token cap.")
    p.add_argument("--run-name", default=None, help="Folder name under --out-dir.")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "logs" / "runs"), help="Parent folder for run folders.")
    p.add_argument("--resume", action="store_true", help="Continue an existing run folder, skipping finished problems.")
    return p


async def main_async(args: argparse.Namespace) -> int:
    """Set up the run folder, run the debates, write the summary. Returns the process exit code."""
    from data.loader import load_split
    from models.client import LLMClient

    levers = Levers(parallel=args.parallel, coalesce=args.coalesce)
    run_name = args.run_name or (
        f"debate-{args.model}-{levers.label}-cache{args.prefix_caching}-c{args.concurrency}-seed{args.seed}"
    )
    run_dir = Path(args.out_dir) / run_name
    if run_dir.exists() and not args.resume:
        print(f"{run_dir} already exists. Use --resume to continue it or pick another --run-name.")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient(args.model, base_url=args.endpoint)
    if not client.is_ready():
        print(f"No server answering at {client.base_url}. Start one with models.serve_vllm.")
        return 2

    problems = load_split("gsm8k", args.split)
    if args.limit is not None:
        problems = problems[: args.limit]
    done = {t["task_id"] for t in load_jsonl(run_dir / "tasks.jsonl") if not t.get("error")}
    todo_indices = [i for i, p in enumerate(problems) if p["id"] not in done]
    todo = [problems[i] for i in todo_indices]

    if not (run_dir / "config.json").exists():
        config = {
            "args": vars(args), "run_name": run_name, "levers": levers.label, "model": args.model,
            "endpoint": client.base_url, "sampling": asdict(client.sampling), "git_commit": git_commit(),
            "python": platform.python_version(), "openai": package_version("openai"),
            "vllm_client_host_has_vllm": package_version("vllm"),
            "started_utc": datetime.now(timezone.utc).isoformat(), "num_problems": len(problems),
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    if args.warmup > 0:
        warm = load_split("gsm8k", "dev")[: args.warmup]
        print(f"Warm-up: {len(warm)} dev problems (discarded)")
        await run_closed_loop(
            client, warm, levers, args.concurrency, Tracer(), args.seed, args.max_tokens,
            indices=[10_000 + i for i in range(len(warm))],
        )

    print(f"Running {len(todo)} of {len(problems)} problems, levers={levers.label}, "
          f"prefix caching={args.prefix_caching}, C={args.concurrency}")
    tracer = Tracer(out_dir=run_dir)
    transcripts = run_dir / "transcripts.jsonl"
    await run_closed_loop(
        client, todo, levers, args.concurrency, tracer, args.seed, args.max_tokens,
        indices=todo_indices, on_result=lambda r: append_jsonl(transcripts, transcript_row(r)),
    )

    summary = summarize(latest_rows(load_jsonl(run_dir / "tasks.jsonl")))
    summary["run_name"] = run_name
    if args.prefix_caching == "off" and summary.get("cache_hit_fraction", 0) > 0:
        summary["warning"] = "prefix caching was declared off but the server reported cached tokens"
    if args.prefix_caching == "on" and summary.get("cache_hit_fraction", 1) == 0:
        summary["warning"] = "prefix caching was declared on but no cached tokens were reported"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    """Command line entry point."""
    args = build_parser().parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
