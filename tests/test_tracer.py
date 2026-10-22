"""Tests for the call tracer: task breakdown, critical path, files on disk, and the run summary."""

import json
import tempfile
from pathlib import Path

from models.client import CallResult
from tracing.tracer import Tracer, critical_path, load_jsonl, percentile, summarize, union_length


def make_call(t0, submit, first, done, round_, role="agent0", prompt=100, cached=0, out=50):
    """A CallResult whose times are given relative to the tracer's clock origin t0."""
    return CallResult(
        texts=["x"], submit_time=t0 + submit, completion_time=t0 + done, prompt_tokens=prompt,
        cached_tokens=cached, output_tokens=out, model="fake",
        first_token_time=None if first is None else t0 + first, role=role, round=round_,
    )


def test_union_length_counts_overlap_once():
    """Overlapping intervals are merged; gaps are not counted."""
    assert union_length([(0, 2), (1, 3), (5, 6)]) == 4.0
    assert union_length([]) == 0.0


def test_task_breakdown_and_critical_path():
    """Two rounds of three parallel calls with a gap between them."""
    tracer = Tracer()
    t0 = tracer.t0
    start = tracer.begin_task("t", start=t0)
    plan = [
        (0, 0.2, 1.0, 0), (0, 0.5, 2.0, 0), (0, 0.3, 1.5, 0),          # round 0: slowest ends at 2.0
        (2.5, 2.7, 3.5, 1), (2.5, 3.0, 4.0, 1), (2.5, 2.6, 3.0, 1),    # round 1: starts 0.5 after the barrier
    ]
    for i, (sub, first, done, rnd) in enumerate(plan):
        tracer.record_call("t", make_call(t0, sub, first, done, rnd, role=f"agent{i % 3}"), agent=i % 3)
    rec = tracer.end_task("t", correct=True, end=t0 + 4.0)
    assert abs(rec.completion_time - 4.0) < 1e-9
    assert abs(rec.busy_time - 3.5) < 1e-9 and abs(rec.idle_time - 0.5) < 1e-9
    assert abs(rec.critical_path - 3.5) < 1e-9          # 2.0 + 1.5
    assert abs(rec.speedup_bound - 4.0 / 3.5) < 1e-9
    assert rec.num_calls == 6 and rec.correct is True
    assert abs(rec.prefill_call_seconds + rec.decode_call_seconds - sum(d - s for s, _, d, _ in plan)) < 1e-9
    assert rec.unsplit_call_seconds == 0.0
    assert start == t0


def test_call_without_first_token_is_unsplit():
    """A call with no first-token time (n > 1) goes to unsplit_call_seconds."""
    tracer = Tracer()
    t0 = tracer.t0
    tracer.begin_task("t", start=t0)
    tracer.record_call("t", make_call(t0, 0, None, 2.0, 0), agent=-1)
    rec = tracer.end_task("t", end=t0 + 2.0)
    assert rec.unsplit_call_seconds == 2.0 and rec.prefill_call_seconds == 0.0


def test_critical_path_groups_by_round():
    """The critical path adds the longest call of each round."""
    tracer = Tracer()
    t0 = tracer.t0
    tracer.begin_task("t", start=t0)
    records = [
        tracer.record_call("t", make_call(t0, 0, 0.1, 1.0, 0), agent=0),
        tracer.record_call("t", make_call(t0, 0, 0.1, 3.0, 0), agent=1),
        tracer.record_call("t", make_call(t0, 3.0, 3.1, 4.0, 1), agent=0),
    ]
    assert abs(critical_path(records) - 4.0) < 1e-9


def test_files_are_written_as_the_run_goes():
    """calls.jsonl and tasks.jsonl hold one row per call and per task."""
    with tempfile.TemporaryDirectory() as tmp:
        tracer = Tracer(out_dir=Path(tmp))
        t0 = tracer.t0
        tracer.begin_task("t", start=t0)
        tracer.record_call("t", make_call(t0, 0, 0.1, 1.0, 0, cached=40), agent=0)
        tracer.end_task("t", correct=False, end=t0 + 1.0)
        calls, tasks = load_jsonl(Path(tmp) / "calls.jsonl"), load_jsonl(Path(tmp) / "tasks.jsonl")
        assert len(calls) == 1 and calls[0]["cached_tokens"] == 40 and calls[0]["task_id"] == "t"
        assert len(tasks) == 1 and tasks[0]["correct"] is False
        json.dumps(tasks[0])


def test_load_jsonl_missing_file_is_empty():
    """A missing file reads as no rows."""
    assert load_jsonl(Path("/nonexistent/none.jsonl")) == []


def row(task_id, completion, correct=True, error="", prompt=100, cached=0):
    """A minimal tasks.jsonl row for summarize."""
    return {
        "task_id": task_id, "completion_time": completion, "correct": correct, "error": error,
        "prompt_tokens": prompt, "cached_tokens": cached, "output_tokens": 10, "num_calls": 6,
        "idle_time": completion * 0.1, "speedup_bound": 2.0,
    }


def test_summarize_numbers_and_errors():
    """Median, p95, accuracy and cache hit fraction come from successful tasks only."""
    rows = [row(f"t{i}", float(i + 1), correct=(i % 2 == 0), cached=25) for i in range(10)]
    rows.append(row("bad", 99.0, error="Timeout"))
    s = summarize(rows)
    assert s["num_tasks"] == 11 and s["num_errors"] == 1
    assert s["median_completion_time"] == 5.5 and s["p95_completion_time"] == 10.0
    assert s["accuracy"] == 0.5 and s["cache_hit_fraction"] == 0.25
    assert abs(s["median_idle_fraction"] - 0.1) < 1e-9 and s["median_speedup_bound"] == 2.0


def test_summarize_empty_and_percentile():
    """Nothing to summarize gives only counts; the percentile uses the nearest rank."""
    assert summarize([]) == {"num_tasks": 0, "num_errors": 0}
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0], 0.95) == 1.0
