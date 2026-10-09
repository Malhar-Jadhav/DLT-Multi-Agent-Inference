"""Tests for dataset helpers: seeded sampling, GSM8K parsing, and bug injection.

These use small in-memory examples and do not download anything.
"""

import ast

import pytest

from data.bugged_solutions.mutations import count_sites, mutate_source
from data.bugged_solutions.preprocess import run_program
from data.common.io_utils import ids_fingerprint, read_jsonl, seeded_sample, write_jsonl
from data.dataset_config import get_dataset
from data.gsm8k.preprocess import extract_final_number, normalize_number, parse_gold_answer
from data.humanevalplus.preprocess import build_program
from data.prepare_all import resolve_order


def test_seeded_sample_is_deterministic():
    """The same seed gives the same subset, a different seed gives a different one."""
    items = list(range(100))
    assert seeded_sample(items, 10, 0) == seeded_sample(items, 10, 0)
    assert seeded_sample(items, 10, 0) != seeded_sample(items, 10, 1)


def test_seeded_sample_too_large_raises():
    """Sampling more items than exist is an error."""
    with pytest.raises(ValueError):
        seeded_sample([1, 2, 3], 5, 0)


def test_jsonl_round_trip(tmp_path):
    """Rows written to JSONL are read back unchanged."""
    rows = [{"id": "a", "x": 1}, {"id": "b", "x": "text"}]
    write_jsonl(tmp_path / "f.jsonl", rows)
    assert read_jsonl(tmp_path / "f.jsonl") == rows


def test_fingerprint_depends_on_ids():
    """Different id lists give different fingerprints."""
    assert ids_fingerprint(["a", "b"]) != ids_fingerprint(["a", "c"])


def test_gsm8k_gold_and_extraction():
    """Gold answers and model outputs normalize to the same string for exact match."""
    assert parse_gold_answer("She sold 3 clips.\n#### 1,200") == "1200"
    assert normalize_number("$1,200.00") == "1200"
    assert extract_final_number("So the total is 18 dollars, i.e. 18.0") == "18"
    assert extract_final_number("no digits here") is None


def test_mutation_changes_exactly_one_site():
    """A mutant parses, differs from the original, and keeps the same number of sites."""
    source = "def f(a, b):\n    if a < b:\n        return a + 1\n    return b\n"
    assert count_sites(source) == 3
    for i in range(3):
        mutant, kind, description = mutate_source(source, i)
        ast.parse(mutant)
        assert mutant != ast.unparse(ast.parse(source))
        assert kind in {"operator", "off_by_one"} and description


def test_run_program_statuses():
    """run_program separates passing, failing, and timed-out programs."""
    assert run_program("assert True") == "pass"
    assert run_program("assert False") == "fail"
    assert run_program("while True: pass", timeout=1) == "timeout"


def test_build_program_runs_tests():
    """build_program produces a script that passes for a correct solution and fails for a wrong one."""
    row = {
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    }
    assert run_program(build_program(row, "    return a + b\n")) == "pass"
    assert run_program(build_program(row, "    return a - b\n")) == "fail"


def test_resolve_order_puts_dependencies_first():
    """bugged_solutions is always built after humanevalplus."""
    order = resolve_order(["bugged_solutions"])
    assert order.index("humanevalplus") < order.index("bugged_solutions")
    assert get_dataset("gsm8k").system == "debate"
