"""Tests for the closed-loop runner: concurrency limit, error handling, seeds and resume bookkeeping."""

import asyncio

from experiments.run_debate import latest_rows, run_closed_loop
from systems.debate import Levers
from tests.fake_llm import FakeLLM
from tracing.tracer import Tracer, summarize


def problems(n, bad=()):
    """n fake GSM8K problems; those whose index is in `bad` make FakeLLM fail."""
    return [
        {"id": f"p{i}", "question": ("BOOM " if i in bad else "") + f"Question {i}?", "answer": "18"}
        for i in range(n)
    ]


def run_loop(fake, probs, levers, concurrency, indices=None):
    """Run the closed loop on the fake model and return (results, tracer)."""
    tracer = Tracer()
    results = asyncio.run(run_closed_loop(fake, probs, levers, concurrency, tracer, indices=indices))
    return results, tracer


def test_concurrency_limits_debates_in_flight():
    """With sequential agents, C debates in flight means at most C calls in flight."""
    fake = FakeLLM(delay=0.01)
    results, tracer = run_loop(fake, problems(6), Levers(), concurrency=2)
    assert len(results) == 6 and len(tracer.tasks) == 6
    assert fake.max_in_flight == 2


def test_parallel_dispatch_multiplies_calls_in_flight():
    """With L1 on, each of the C debates can have 3 calls in flight."""
    fake = FakeLLM(delay=0.01)
    run_loop(fake, problems(6), Levers(parallel=True), concurrency=2)
    assert fake.max_in_flight == 6


def test_a_failing_debate_is_recorded_and_the_rest_continue():
    """An exception becomes a task with an error; other problems still finish and are summarized."""
    fake = FakeLLM(delay=0.005, fail_marker="BOOM")
    results, tracer = run_loop(fake, problems(5, bad=(2,)), Levers(), concurrency=2)
    assert len(results) == 4
    errored = [t for t in tracer.tasks if t.error]
    assert len(errored) == 1 and errored[0].task_id == "p2" and "fake failure" in errored[0].error
    summary = summarize([vars(t) for t in tracer.tasks])
    assert summary["num_errors"] == 1 and summary["accuracy"] == 1.0


def test_indices_set_the_seeds():
    """Passing original positions (as a resumed run does) reproduces the same seeds."""
    first, second = FakeLLM(delay=0.001), FakeLLM(delay=0.001)
    run_loop(first, problems(1), Levers(), 1, indices=[7])
    run_loop(second, problems(1), Levers(), 1, indices=[7])
    assert [c["seed"] for c in first.calls] == [c["seed"] for c in second.calls]
    other = FakeLLM(delay=0.001)
    run_loop(other, problems(1), Levers(), 1, indices=[8])
    assert [c["seed"] for c in other.calls] != [c["seed"] for c in first.calls]


def test_latest_rows_keeps_the_last_attempt():
    """A resumed run's successful row replaces the earlier errored row for the same task."""
    rows = [{"task_id": "a", "error": "x"}, {"task_id": "b", "error": ""}, {"task_id": "a", "error": ""}]
    assert latest_rows(rows) == [{"task_id": "a", "error": ""}, {"task_id": "b", "error": ""}]
