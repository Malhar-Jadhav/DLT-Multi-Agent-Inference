"""Tests for harness/tracer.py — no GPU or network required."""

import json
import tempfile
from pathlib import Path

from harness.tracer import CallRecord, Tracer


def _make_record(**overrides) -> CallRecord:
    defaults = dict(
        task_id="t1", system="debate", config="C0",
        levers={"l1": False, "l2": False, "l3": False, "l4": False},
        seed=0, concurrency=1,
        role="solver", round=1,
        submit_time=1000.0, first_token_time=1000.5, completion_time=1002.0,
        prompt_tokens=100, cached_tokens=0, output_tokens=50,
        model="qwen3-14b",
    )
    defaults.update(overrides)
    return CallRecord(**defaults)


def test_call_record_latency():
    rec = _make_record(submit_time=10.0, completion_time=12.5)
    assert abs(rec.latency - 2.5) < 1e-9


def test_call_record_ttft():
    rec = _make_record(submit_time=10.0, first_token_time=10.3)
    assert abs(rec.ttft - 0.3) < 1e-9


def test_call_record_ttft_none():
    rec = _make_record(first_token_time=None)
    assert rec.ttft is None


def test_tracer_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        with Tracer(run_dir) as tracer:
            rec1 = _make_record(task_id="a", role="solver")
            rec2 = _make_record(task_id="b", role="critic_check")
            tracer.write(rec1)
            tracer.write(rec2)

        lines = (run_dir / "traces.jsonl").read_text().splitlines()
        assert len(lines) == 2
        d1 = json.loads(lines[0])
        assert d1["task_id"] == "a"
        assert d1["role"] == "solver"
        assert "latency" in d1


def test_tracer_appends_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        with Tracer(run_dir) as t1:
            t1.write(_make_record(task_id="x"))
        with Tracer(run_dir) as t2:
            t2.write(_make_record(task_id="y"))

        lines = (run_dir / "traces.jsonl").read_text().splitlines()
        assert len(lines) == 2
