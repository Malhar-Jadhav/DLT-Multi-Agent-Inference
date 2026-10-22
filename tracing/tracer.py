"""Call tracer: records every model call and a per-task latency breakdown.

The headline metric of the project is the median task completion time, measured from the
submission of a task's first model call to the recording of its final answer. This module
produces exactly that number for every task, plus the pieces needed to explain it:

  * one CallRecord per model call: submit, first-token and completion times, token counts
    (prompt, cached, output), the agent and debate round it belongs to;
  * one TaskRecord per task: completion time, the part of it spent with at least one call in
    flight (busy) and the rest (idle: barriers and orchestration), the critical path (the
    sum over rounds of the slowest call in the round), and the speedup bound it implies.

All times are seconds since the tracer was created, read from time.perf_counter(), the same
clock models.client uses. Records are appended to calls.jsonl and tasks.jsonl as they are
produced, so a job that is cut off by the cluster time limit keeps everything finished so far.

Usage:
    tracer = Tracer(out_dir=Path("results/run1"))
    tracer.begin_task("gsm8k-test-17")
    tracer.record_call("gsm8k-test-17", call_result, agent=0)
    tracer.end_task("gsm8k-test-17", correct=True)
    print(summarize(load_jsonl(Path("results/run1/tasks.jsonl"))))
"""

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models.client import CallResult


@dataclass
class CallRecord:
    """One model call. Times are seconds since the tracer started; first_token is None if unknown.

    agent is the agent index, or -1 for a call shared by all agents (a coalesced request).
    n is the number of samples requested; output_tokens counts all of them.
    """

    task_id: str
    agent: int
    round: int
    role: str
    model: str
    n: int
    submit: float
    first_token: Optional[float]
    completion: float
    latency: float
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass
class TaskRecord:
    """One task (one debate). Times in seconds.

    completion_time: end minus start, the preregistered metric.
    busy_time: wall time with at least one call in flight. idle_time: completion minus busy.
    critical_path: sum over rounds of the longest call in the round, the best time that
        parallel dispatch could reach with these call durations.
    speedup_bound: completion_time divided by critical_path.
    prefill_call_seconds, decode_call_seconds: summed over calls, from submit to first token
        (queueing plus prefill) and from first token to completion. Calls without a first
        token (n > 1) are counted in unsplit_call_seconds instead.
    """

    task_id: str
    start: float
    end: float
    completion_time: float
    num_calls: int
    busy_time: float
    idle_time: float
    critical_path: float
    speedup_bound: float
    prefill_call_seconds: float
    decode_call_seconds: float
    unsplit_call_seconds: float
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    correct: Optional[bool] = None
    error: str = ""


def union_length(intervals: Iterable[Tuple[float, float]]) -> float:
    """Total length covered by a set of (start, end) intervals, counting overlaps once."""
    total = 0.0
    current_start: Optional[float] = None
    current_end = 0.0
    for start, end in sorted(intervals):
        if current_start is None or start > current_end:
            if current_start is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def critical_path(calls: List[CallRecord]) -> float:
    """Sum over rounds of the longest call latency in that round. Calls with no round form one group."""
    longest: Dict[int, float] = {}
    for c in calls:
        longest[c.round] = max(longest.get(c.round, 0.0), c.latency)
    return sum(longest.values())


class Tracer:
    """Collects CallRecords and TaskRecords for one run and optionally appends them to disk."""

    def __init__(self, out_dir: Optional[Path] = None) -> None:
        """Start the clock. If out_dir is given, calls.jsonl and tasks.jsonl are appended there."""
        self.t0 = time.perf_counter()
        self.calls: List[CallRecord] = []
        self.tasks: List[TaskRecord] = []
        self._starts: Dict[str, float] = {}
        self._task_calls: Dict[str, List[CallRecord]] = {}
        self._calls_path: Optional[Path] = None
        self._tasks_path: Optional[Path] = None
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._calls_path = out_dir / "calls.jsonl"
            self._tasks_path = out_dir / "tasks.jsonl"

    @staticmethod
    def now() -> float:
        """Current reading of the clock all timings use."""
        return time.perf_counter()

    def begin_task(self, task_id: str, start: Optional[float] = None) -> float:
        """Mark the start of a task and return the raw clock reading used as its start."""
        raw = self.now() if start is None else start
        self._starts[task_id] = raw
        self._task_calls[task_id] = []
        return raw

    def record_call(self, task_id: str, result: CallResult, agent: int = -1, n: int = 1) -> CallRecord:
        """Store one finished call under a task and return its record."""
        first = None if result.first_token_time is None else result.first_token_time - self.t0
        record = CallRecord(
            task_id=task_id, agent=agent, round=result.round, role=result.role, model=result.model,
            n=max(n, len(result.texts)),
            submit=result.submit_time - self.t0, first_token=first,
            completion=result.completion_time - self.t0, latency=result.latency,
            prompt_tokens=result.prompt_tokens, cached_tokens=result.cached_tokens,
            output_tokens=result.output_tokens,
        )
        self.calls.append(record)
        self._task_calls.setdefault(task_id, []).append(record)
        self._append(self._calls_path, asdict(record))
        return record

    def end_task(
        self, task_id: str, correct: Optional[bool] = None, error: str = "", end: Optional[float] = None,
    ) -> TaskRecord:
        """Close a task, compute its latency breakdown, and return its record.

        end is a raw clock reading, defaulting to now; tests pass explicit values.
        """
        raw_end = self.now() if end is None else end
        raw_start = self._starts.pop(task_id)
        calls = self._task_calls.pop(task_id, [])
        start, finish = raw_start - self.t0, raw_end - self.t0
        completion = finish - start
        busy = union_length((c.submit, c.completion) for c in calls)
        path = critical_path(calls)
        prefill = sum(c.first_token - c.submit for c in calls if c.first_token is not None)
        decode = sum(c.completion - c.first_token for c in calls if c.first_token is not None)
        unsplit = sum(c.latency for c in calls if c.first_token is None)
        record = TaskRecord(
            task_id=task_id, start=start, end=finish, completion_time=completion, num_calls=len(calls),
            busy_time=busy, idle_time=max(completion - busy, 0.0), critical_path=path,
            speedup_bound=(completion / path) if path > 0 else 0.0,
            prefill_call_seconds=prefill, decode_call_seconds=decode, unsplit_call_seconds=unsplit,
            prompt_tokens=sum(c.prompt_tokens for c in calls),
            cached_tokens=sum(c.cached_tokens for c in calls),
            output_tokens=sum(c.output_tokens for c in calls),
            correct=correct, error=error,
        )
        self.tasks.append(record)
        self._append(self._tasks_path, asdict(record))
        return record

    @staticmethod
    def _append(path: Optional[Path], row: Dict[str, Any]) -> None:
        """Append one JSON line to a file, if a path was configured."""
        if path is None:
            return
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, returning an empty list if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def percentile(sorted_values: List[float], q: float) -> float:
    """Nearest-rank percentile (q between 0 and 1) of an already sorted, non-empty list."""
    rank = max(math.ceil(q * len(sorted_values)), 1)
    return sorted_values[min(rank, len(sorted_values)) - 1]


def summarize(task_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate task rows (as read from tasks.jsonl) into the numbers reported for a run.

    Tasks with an error are counted but left out of every latency, token and accuracy figure.
    """
    ok = [t for t in task_rows if not t.get("error")]
    summary: Dict[str, Any] = {"num_tasks": len(task_rows), "num_errors": len(task_rows) - len(ok)}
    if not ok:
        return summary
    times = sorted(t["completion_time"] for t in ok)
    scored = [t["correct"] for t in ok if t.get("correct") is not None]
    prompt = sum(t["prompt_tokens"] for t in ok)
    summary.update({
        "median_completion_time": statistics.median(times),
        "mean_completion_time": statistics.fmean(times),
        "p95_completion_time": percentile(times, 0.95),
        "accuracy": (sum(1 for c in scored if c) / len(scored)) if scored else None,
        "prompt_tokens_per_task": prompt / len(ok),
        "cached_tokens_per_task": sum(t["cached_tokens"] for t in ok) / len(ok),
        "cache_hit_fraction": (sum(t["cached_tokens"] for t in ok) / prompt) if prompt else 0.0,
        "output_tokens_per_task": sum(t["output_tokens"] for t in ok) / len(ok),
        "calls_per_task": sum(t["num_calls"] for t in ok) / len(ok),
        "median_idle_fraction": statistics.median(
            t["idle_time"] / t["completion_time"] for t in ok if t["completion_time"] > 0
        ),
        "median_speedup_bound": statistics.median(t["speedup_bound"] for t in ok),
    })
    return summary
