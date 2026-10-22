"""Tests for the debate harness: prompts, round structure, voting, scoring and the L1 and L2 switches.

They use FakeLLM, so no GPU, server, or network is needed.
"""

import asyncio

from systems.debate import (
    Levers, answers_match, build_others_message, build_round1_message, call_seed, most_frequent,
    parse_answer, run_debate,
)
from tests.fake_llm import FakeLLM
from tracing.tracer import Tracer

PROBLEM = {"id": "p0", "question": "What is 2+2?", "answer": "18"}


def run(levers=Levers(), values=None, delay=0.01, problem=PROBLEM):
    """Run one debate on the fake model and return (result, fake, tracer)."""
    fake, tracer = FakeLLM(values=values, delay=delay), Tracer()
    result = asyncio.run(run_debate(fake, problem, levers, tracer, task_index=0))
    return result, fake, tracer


def test_round1_prompt_matches_released_code():
    """The first message is the released question template, character for character."""
    expected = (
        "Can you solve the following math problem? What is 2+2? Explain your reasoning. Your final answer "
        "should be a single numerical number, in the form \\boxed{answer}, at the end of your response. "
    )
    assert build_round1_message("What is 2+2?") == {"role": "user", "content": expected}


def test_others_message_matches_released_code():
    """The round 2 message lists other agents' answers in the released format."""
    others = [[{"content": "q"}, {"content": "X"}], [{"content": "q"}, {"content": "Y"}]]
    expected = (
        "These are the solutions to the problem from other agents: "
        "\n\n One agent solution: ```X```"
        "\n\n One agent solution: ```Y```"
        "\n\n Using the solutions from other agents as additional information, can you provide your answer "
        "to the math problem? \n The original math problem is Q. Your final answer should be a single "
        "numerical number, in the form \\boxed{answer}, at the end of your response."
    )
    assert build_others_message(others, "Q", 1) == {"role": "user", "content": expected}


def test_baseline_makes_six_sequential_calls():
    """3 agents x 2 rounds is 6 calls, in agent order within each round, one at a time."""
    result, fake, tracer = run()
    assert [(c["role"], c["round"]) for c in fake.calls] == [
        ("agent0", 0), ("agent1", 0), ("agent2", 0), ("agent0", 1), ("agent1", 1), ("agent2", 1),
    ]
    assert fake.max_in_flight == 1
    assert len(tracer.calls) == 6 and len(tracer.tasks) == 1
    assert tracer.tasks[0].num_calls == 6


def test_round1_prompts_identical_and_round2_prompts_see_only_others():
    """Round 1 prompts are equal; in round 2 each agent sees the other agents' answers but not its own."""
    result, fake, _ = run()
    round1 = [c["messages"] for c in fake.calls if c["round"] == 0]
    assert round1[0] == round1[1] == round1[2]
    for call in (c for c in fake.calls if c["round"] == 1):
        agent = int(call["role"][-1])
        assert [m["role"] for m in call["messages"]] == ["user", "assistant", "user"]
        shown = call["messages"][2]["content"]
        assert f"[agent{agent}|0]" not in shown
        for other in (a for a in range(3) if a != agent):
            assert f"[agent{other}|0]" in shown
        assert call["messages"][1]["content"].startswith(f"[agent{agent}|0]")


def test_majority_vote_and_scoring():
    """The vote picks the most common parsed answer and is scored against the gold number."""
    result, _, _ = run(values=["18", "18", "5"])
    assert result.answers == ["18", "18", "5"] and result.final == "18" and result.correct is True
    result, _, _ = run(values=["5", "7", "9"])
    assert result.final == "5" and result.correct is False


def test_unparseable_answers_are_wrong():
    """A debate where nobody gives a number scores wrong (the released code scored it right)."""
    result, _, _ = run(values=[None, None, None])
    assert result.final is None and result.correct is False


def test_parallel_matches_sequential_and_is_faster():
    """L1 changes timing only: same transcripts, and about 3x less time for equal call durations."""
    seq, _, seq_tracer = run(Levers(), delay=0.05)
    par, fake, par_tracer = run(Levers(parallel=True), delay=0.05)
    assert seq.contexts == par.contexts and seq.answers == par.answers
    assert fake.max_in_flight == 3
    assert par_tracer.tasks[0].completion_time < 0.7 * seq_tracer.tasks[0].completion_time


def test_coalescing_sends_round1_as_one_request():
    """L2 sends the identical round 1 prompts as a single n=3 request, then round 2 as three calls."""
    result, fake, tracer = run(Levers(coalesce=True))
    assert len(fake.calls) == 4
    assert fake.calls[0]["n"] == 3 and fake.calls[0]["round"] == 0
    assert all(c["n"] == 1 for c in fake.calls[1:])
    assert result.contexts[1][1]["content"].startswith("[agent1|0]")
    assert tracer.tasks[0].num_calls == 4


def test_levers_label():
    """The label names the active levers."""
    assert Levers().label == "baseline"
    assert Levers(parallel=True).label == "L1"
    assert Levers(parallel=True, coalesce=True).label == "L1+L2"


def test_parse_answer():
    """Boxed numbers win over other numbers; cleanup handles $, commas, trailing dots and minus signs."""
    assert parse_answer("so \\boxed{18}") == "18"
    assert parse_answer("\\boxed{$1,200.50}") == "1200.5"
    assert parse_answer("\\boxed{18.}") == "18"
    assert parse_answer("\\boxed{-5}") == "-5"
    assert parse_answer("first {3} then \\boxed{7}") == "7"
    assert parse_answer("The total is 42 apples.") == "42"
    assert parse_answer("answer 9 \\boxed{}") == "9"
    assert parse_answer("no digits here") is None


def test_most_frequent_and_answers_match():
    """Ties go to the first answer, unparseable answers are ignored, and 18.0 equals 18."""
    assert most_frequent(["5", "7", "9"]) == "5"
    assert most_frequent([None, "7", "7"]) == "7"
    assert most_frequent([None, None, None]) is None
    assert answers_match("18.0", "18") and not answers_match(None, "18") and not answers_match("17", "18")


def test_call_seed_is_deterministic_and_distinct():
    """Seeds repeat for the same call and differ across tasks, agents and rounds."""
    assert call_seed(0, 3, 1, 0) == call_seed(0, 3, 1, 0)
    seeds = {call_seed(b, t, a, r) for b in (0, 1) for t in range(5) for a in range(3) for r in range(2)}
    assert len(seeds) == 2 * 5 * 3 * 2
