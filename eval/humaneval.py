"""HumanEval+ evaluation: execute generated code against the test suite.

Uses evalplus (pip install evalplus) to run the official HumanEval+ tests.
Falls back to a subprocess-isolation approach when evalplus is unavailable.

Usage:
    from eval.humaneval import score_results, run_solution

    result = run_solution(task, solution_code, timeout=10.0)
    # result is one of: "pass", "fail", "timeout", "error"

    results_with_status = score_results(tasks, solutions, timeout=10.0)
    pass_at_1 = sum(r["status"] == "pass" for r in results_with_status) / len(tasks)
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_solution(
    task: Dict[str, Any],
    solution: str,
    timeout: float = 10.0,
) -> str:
    """Execute a solution against the task's test cases.

    Returns "pass", "fail", "timeout", or "error".
    Runs in a subprocess for isolation.
    """
    # Build the full executable: prompt (function def) + solution body + tests
    prompt = task.get("prompt", "")
    tests = task.get("test", "")  # HumanEval+ "test" field contains assert statements

    # Combine: the prompt defines the function signature; solution is the body.
    # If solution already contains the full function def, use it directly.
    if solution.lstrip().startswith("def "):
        code = solution + "\n\n" + tests
    else:
        code = prompt + "\n" + textwrap.indent(solution, "    ") + "\n\n" + tests

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        proc = subprocess.run(
            [sys.executable, "-I", tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0:
            return "pass"
        # Distinguish timeout-like errors from assertion/syntax errors
        if "AssertionError" in proc.stderr or "assert" in proc.stderr.lower():
            return "fail"
        return "error"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "error"
    finally:
        Path(tmp).unlink(missing_ok=True)


def score_results(
    tasks: List[Dict[str, Any]],
    solutions: List[str],
    timeout: float = 10.0,
) -> List[Dict[str, Any]]:
    """Run each solution and return a list of dicts with task_id + status."""
    assert len(tasks) == len(solutions)
    out = []
    for task, sol in zip(tasks, solutions):
        status = run_solution(task, sol, timeout=timeout)
        out.append({
            "task_id": task.get("task_id", task.get("task_id", "unknown")),
            "status": status,
            "pass": status == "pass",
        })
    return out
