"""Aggregate trace JSONL files into a summary CSV.

Usage:
    python -m analysis.aggregate --runs-dir results/runs --out results/summary.csv

Reads every traces.jsonl + manifest.json under runs-dir and computes:
  p50, p95, std, speedup S (vs. C0 of same system/seed/concurrency),
  bootstrap CI lower bound, cache_hit_rate, token totals.

Writes results/summary.csv with one row per (system, config, seed, concurrency).
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas required: pip install pandas")


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_lower(values: List[float], stat=None, n_boot: int = 2000, ci: float = 0.95) -> float:
    """Return the lower bound of the bootstrap CI for the median."""
    if not values:
        return float("nan")
    rng = random.Random(42)
    if stat is None:
        stat = _median
    boots = [stat(rng.choices(values, k=len(values))) for _ in range(n_boot)]
    boots.sort()
    lower_idx = int((1 - ci) / 2 * n_boot)
    return boots[lower_idx]


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _percentile(xs: List[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    traces_path = run_dir / "traces.jsonl"
    if not manifest_path.exists() or not traces_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    records = [json.loads(l) for l in traces_path.read_text().splitlines() if l.strip()]
    return {"manifest": manifest, "records": records}


def per_task_latency(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Sum call latencies per task_id to get task completion time."""
    totals: Dict[str, float] = {}
    for r in records:
        tid = r["task_id"]
        totals[tid] = totals.get(tid, 0.0) + r["latency"]
    return totals


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(runs_dir: Path) -> "pd.DataFrame":
    rows = []
    # First pass: collect all run data
    all_runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        data = load_run(run_dir)
        if data is None:
            continue
        all_runs.append(data)

    # Second pass: build summary rows
    # Build baseline lookup: (system, seed, concurrency) → {task_id: latency}
    baselines: Dict[tuple, Dict[str, float]] = {}
    for data in all_runs:
        m = data["manifest"]
        if m["config"] == "C0":
            key = (m["system"], m["seed"], m["concurrency"])
            baselines[key] = per_task_latency(data["records"])

    for data in all_runs:
        m = data["manifest"]
        records = data["records"]
        if not records:
            continue

        task_latencies = list(per_task_latency(records).values())
        if not task_latencies:
            continue

        p50 = _median(task_latencies)
        p95 = _percentile(task_latencies, 95)
        std = _std(task_latencies)

        bkey = (m["system"], m["seed"], m["concurrency"])
        base_lats = baselines.get(bkey, {})
        if base_lats:
            # Compute speedup per task, then take median
            base_vals = [base_lats[tid] for tid in per_task_latency(records) if tid in base_lats]
            opt_vals = [v for tid, v in per_task_latency(records).items() if tid in base_lats]
            speedups = [b / o for b, o in zip(base_vals, opt_vals) if o > 0]
            S = _median(speedups) if speedups else float("nan")
            ci_lower = bootstrap_lower(speedups) if speedups else float("nan")
        else:
            S = 1.0 if m["config"] == "C0" else float("nan")
            ci_lower = float("nan")

        # Token and cache stats
        prompt_tokens = sum(r["prompt_tokens"] for r in records)
        cached_tokens = sum(r["cached_tokens"] for r in records)
        output_tokens = sum(r["output_tokens"] for r in records)
        cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0

        rows.append({
            "system": m["system"],
            "config": m["config"],
            "seed": m["seed"],
            "concurrency": m["concurrency"],
            "p50_s": round(p50, 3),
            "p95_s": round(p95, 3),
            "std_s": round(std, 3),
            "speedup_S": round(S, 3),
            "ci_lower": round(ci_lower, 3),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "n_tasks": len(task_latencies),
            "run_dir": str(runs_dir),
        })

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    p.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    args = p.parse_args()
    df = aggregate(args.runs_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
