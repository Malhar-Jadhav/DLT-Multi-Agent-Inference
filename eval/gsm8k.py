"""GSM8K evaluation: exact-match on the final number, format-normalized.

Usage:
    from eval.gsm8k import score_results, equivalence_check

    results = [{"task_id": "0", "predicted": "42", "gold": "42", ...}, ...]
    acc = score_results(results)                  # float 0-1
    mismatches = equivalence_check(baseline, optimized)  # list of task_ids that differ
"""

import re
from typing import Any, Dict, List, Optional, Tuple


def _normalize(s: Optional[str]) -> Optional[str]:
    """Strip commas, trailing zeros after decimal, and whitespace."""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    # Normalize e.g. "42.00" → "42"
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except (ValueError, OverflowError):
        return s


def exact_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    return _normalize(predicted) == _normalize(gold) and predicted is not None


def score_results(results: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    """Return (accuracy, correct_count, total_count)."""
    correct = sum(1 for r in results if exact_match(r.get("predicted"), r.get("gold")))
    total = len(results)
    return (correct / total if total else 0.0), correct, total


def equivalence_check(
    baseline: List[Dict[str, Any]],
    optimized: List[Dict[str, Any]],
) -> List[str]:
    """Return task_ids where the optimized answer differs from the baseline answer.

    A mismatch means a supposedly fake wait was actually a real dependency.
    Mismatches are logged and counted; they do NOT mean the optimized answer is
    wrong (only that it disagrees with baseline).
    """
    base_map = {r["task_id"]: _normalize(r.get("predicted")) for r in baseline}
    mismatches = []
    for r in optimized:
        tid = r["task_id"]
        if base_map.get(tid) != _normalize(r.get("predicted")):
            mismatches.append(tid)
    return mismatches
