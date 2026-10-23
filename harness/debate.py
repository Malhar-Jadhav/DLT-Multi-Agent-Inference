"""Multi-agent debate harness for GSM8K.

Protocol
--------
- 3 solver agents, 2 rounds, majority vote on the final number.
- Round 1 (fan-out): all 3 solvers see only the problem.
- Round 2 (fan-out): all 3 solvers see the problem + all 3 round-1 answers.
- Vote: majority vote on the extracted final number.

L1 (parallel dispatch): rounds 1 and 2 each fire all 3 calls via asyncio.gather.
L2 (request coalescing): round 1 sends a single n=3 request (identical prompt).
    Round 2 cannot coalesce because each solver sees a different history.
L3 (prefix caching): the system prompt is identical for all calls; the server caches it.
L4 (model tiering): the vote_aggregator role uses the small model.

Equivalence check
-----------------
Each task result includes the majority-vote answer so the runner can compare
it against the baseline (C0) answer after the fact.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional

from models.client import LLMClient
from harness.levers import LeverConfig, coalesce_n, dispatch
from harness.tracer import Tracer, TracedClient

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner. "
    "Solve the problem step by step and end your answer with: "
    "'The answer is <number>.' where <number> is just the final numeric value."
)

ROUND2_INSTRUCTION = (
    "Here are answers from your fellow solvers:\n{others}\n\n"
    "Review them and provide your own final answer. "
    "End with: 'The answer is <number>.'"
)


def _extract_number(text: str) -> Optional[str]:
    """Extract the final number from 'The answer is X.' pattern."""
    m = re.search(r"[Tt]he answer is\s*([0-9,.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last standalone number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Return the most common non-None answer, or None if all are None."""
    counts: Dict[str, int] = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


async def run_one_debate(
    task: Dict[str, Any],
    lever_cfg: LeverConfig,
    large_client: LLMClient,
    small_client: Optional[LLMClient],
    tracer: Tracer,
    seed: int,
    config_name: str,
    concurrency: int,
) -> Dict[str, Any]:
    """Run the debate protocol on one GSM8K problem and return a result dict."""
    task_id = str(task.get("task_id", task.get("id", "unknown")))
    problem = task["question"]
    gold = _extract_number(str(task.get("answer", "")))

    def make_client(role: str) -> TracedClient:
        base = lever_cfg.client_for_role(large_client, small_client, role)
        return TracedClient(
            base, tracer,
            task_id=task_id, system="debate", config=config_name,
            levers=lever_cfg.to_dict(), seed=seed, concurrency=concurrency,
        )

    user_msg = [{"role": "user", "content": problem}]
    sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_r1 = sys_msg + user_msg

    # ---- Round 1 ----
    if lever_cfg.l2_coalesce:
        # All 3 solvers have identical prompts in round 1 → coalesce into n=3
        coalesce_client = make_client("solver")
        r1_results = await coalesce_n(coalesce_client, messages_r1, n=3, seed=seed,
                                      role="solver", round=1)
        r1_texts = [r.text for r in r1_results]
    else:
        async def solve_r1(i: int):
            c = make_client("solver")
            return await c.agenerate(messages_r1, seed=seed, role="solver", round=1)

        coros_r1 = [solve_r1(i) for i in range(3)]
        r1_results_raw = await dispatch(lever_cfg, *coros_r1)
        r1_texts = [r.text for r in r1_results_raw]

    # ---- Round 2 ----
    async def solve_r2(i: int):
        others = "\n".join(
            f"Solver {j+1}: {t}" for j, t in enumerate(r1_texts) if j != i
        )
        msgs = sys_msg + user_msg + [
            {"role": "assistant", "content": r1_texts[i]},
            {"role": "user", "content": ROUND2_INSTRUCTION.format(others=others)},
        ]
        c = make_client("solver")
        return await c.agenerate(msgs, seed=seed, role="solver", round=2)

    # Round 2 prompts differ per solver (each sees the other two's answers), so no coalescing.
    coros_r2 = [solve_r2(i) for i in range(3)]
    r2_results = await dispatch(lever_cfg, *coros_r2)
    r2_texts = [r.text for r in r2_results]

    # ---- Vote ----
    answers = [_extract_number(t) for t in r2_texts]
    predicted = _majority_vote(answers)

    return {
        "task_id": task_id,
        "predicted": predicted,
        "gold": gold,
        "correct": predicted == gold if predicted and gold else False,
        "r1_texts": r1_texts,
        "r2_texts": r2_texts,
    }


async def run_debate_tasks(
    tasks: List[Dict[str, Any]],
    lever_cfg: LeverConfig,
    large_client: LLMClient,
    small_client: Optional[LLMClient],
    tracer: Tracer,
    seed: int,
    concurrency: int = 1,
    config_name: str = "unknown",
) -> List[Dict[str, Any]]:
    """Run all tasks, with up to `concurrency` running at the same time."""
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(task):
        async with semaphore:
            return await run_one_debate(
                task, lever_cfg, large_client, small_client,
                tracer, seed, config_name, concurrency,
            )

    return list(await asyncio.gather(*[guarded(t) for t in tasks]))
