"""Critic catch-rate evaluation on bugged solutions.

Tests whether the critic agent flags deliberately injected bugs.
A "catch" is when the critic's verdict contains "ISSUES" (case-insensitive).
A "miss" is a "LGTM" verdict on a bugged solution.

Usage:
    from eval.catch_rate import score_catch_rate

    results = [{"task_id": "hep_42", "critic_verdict": "ISSUES: off-by-one", ...}, ...]
    rate, caught, total = score_catch_rate(results)
"""

from typing import Any, Dict, List, Tuple


def is_caught(critic_verdict: str) -> bool:
    """Return True if the critic flagged an issue (did NOT say LGTM)."""
    v = critic_verdict.strip().upper()
    if v.startswith("LGTM"):
        return False
    if "ISSUES" in v or "BUG" in v or "ERROR" in v or "INCORRECT" in v:
        return True
    # If neither LGTM nor a clear issue signal, treat as a miss (conservative).
    return False


def score_catch_rate(results: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    """Return (catch_rate, caught_count, total_count).

    Each result dict must have a "critic_verdict" key.
    All inputs are assumed to be bugged solutions (positive class).
    """
    caught = sum(1 for r in results if is_caught(r.get("critic_verdict", "")))
    total = len(results)
    return (caught / total if total else 0.0), caught, total


def equivalence_check(
    baseline: List[Dict[str, Any]],
    optimized: List[Dict[str, Any]],
) -> List[str]:
    """Return task_ids where the critic verdict changed between baseline and optimized.

    A changed verdict (LGTM → ISSUES or vice versa) is flagged for inspection.
    """
    base_map = {r["task_id"]: is_caught(r.get("critic_verdict", "")) for r in baseline}
    changed = []
    for r in optimized:
        tid = r["task_id"]
        if base_map.get(tid) != is_caught(r.get("critic_verdict", "")):
            changed.append(tid)
    return changed
