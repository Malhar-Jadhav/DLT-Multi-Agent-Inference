"""Tests for eval/gsm8k.py and eval/catch_rate.py — no GPU or network required."""

import pytest

from eval.gsm8k import exact_match, score_results, equivalence_check as gsm_equiv
from eval.catch_rate import is_caught, score_catch_rate, equivalence_check as catch_equiv


# ---------------------------------------------------------------------------
# GSM8K exact match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold,expected", [
    ("42", "42", True),
    ("1,200", "1200", True),
    ("42.00", "42", True),
    ("42", "43", False),
    (None, "42", False),
    ("42", None, False),
    ("-5", "-5", True),
])
def test_exact_match(pred, gold, expected):
    assert exact_match(pred, gold) == expected


def test_score_results():
    results = [
        {"predicted": "42", "gold": "42"},
        {"predicted": "10", "gold": "11"},
        {"predicted": "5", "gold": "5"},
    ]
    acc, correct, total = score_results(results)
    assert total == 3
    assert correct == 2
    assert abs(acc - 2 / 3) < 1e-9


def test_gsm_equivalence_check():
    baseline = [{"task_id": "a", "predicted": "42"}, {"task_id": "b", "predicted": "7"}]
    optimized = [{"task_id": "a", "predicted": "42"}, {"task_id": "b", "predicted": "8"}]
    mismatches = gsm_equiv(baseline, optimized)
    assert mismatches == ["b"]


# ---------------------------------------------------------------------------
# Catch rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict,expected", [
    ("LGTM", False),
    ("lgtm", False),
    ("LGTM looks fine", False),
    ("ISSUES: off-by-one in loop", True),
    ("BUG: missing return", True),
    ("INCORRECT: result is wrong", True),
    ("", False),   # empty = miss (conservative)
])
def test_is_caught(verdict, expected):
    assert is_caught(verdict) == expected


def test_score_catch_rate():
    results = [
        {"task_id": "1", "critic_verdict": "ISSUES: wrong operator"},
        {"task_id": "2", "critic_verdict": "LGTM"},
        {"task_id": "3", "critic_verdict": "ISSUES: index error"},
        {"task_id": "4", "critic_verdict": "LGTM"},
    ]
    rate, caught, total = score_catch_rate(results)
    assert total == 4
    assert caught == 2
    assert abs(rate - 0.5) < 1e-9


def test_catch_equivalence_check():
    baseline = [
        {"task_id": "x", "critic_verdict": "ISSUES: bug"},
        {"task_id": "y", "critic_verdict": "LGTM"},
    ]
    optimized = [
        {"task_id": "x", "critic_verdict": "LGTM"},   # changed: missed the bug
        {"task_id": "y", "critic_verdict": "LGTM"},   # same
    ]
    changed = catch_equiv(baseline, optimized)
    assert changed == ["x"]
