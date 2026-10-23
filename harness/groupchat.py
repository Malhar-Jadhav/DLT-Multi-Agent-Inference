"""AutoGen-style SelectorGroupChat harness for HumanEval+.

Three agents take serialized turns: Planner → Coder → Critic.
The growing transcript is passed to each agent so each call sees all prior turns.

L1 (parallel dispatch): Planner and Critic sub-checks can run in parallel where
    their inputs don't depend on each other. In the basic 3-turn pipeline the
    turns are genuinely sequential, so L1 has limited effect here (as the
    preregistered hypothesis predicts).
L3 (prefix caching): The system prompt is long and identical on every call of
    the same role. With a growing transcript, the cached prefix saves reprocessing
    the role instructions on each turn.
L4 (model tiering): The critic_check role uses the small model.
"""

import asyncio
from typing import Any, Dict, List, Optional

from models.client import LLMClient
from harness.levers import LeverConfig, dispatch
from harness.tracer import Tracer, TracedClient

PLANNER_SYS = (
    "You are a software architect. Given a Python coding problem, write a clear "
    "step-by-step plan to solve it. Be concise. Do not write code yet."
)

CODER_SYS = (
    "You are an expert Python programmer. Given a plan and a coding problem, "
    "write a complete Python solution. Return ONLY the function body inside a "
    "```python ... ``` block. No explanation outside the block."
)

CRITIC_SYS = (
    "You are a code reviewer. Examine the proposed solution for bugs, edge cases, "
    "and correctness. If it is correct, reply: 'LGTM'. "
    "If there are issues, reply: 'ISSUES: <brief description>'."
)


def _extract_code(text: str) -> str:
    """Pull the first ```python ... ``` block from the coder's output."""
    import re
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


async def run_one_groupchat(
    task: Dict[str, Any],
    lever_cfg: LeverConfig,
    large_client: LLMClient,
    small_client: Optional[LLMClient],
    tracer: Tracer,
    seed: int,
    config_name: str,
    concurrency: int,
) -> Dict[str, Any]:
    task_id = str(task.get("task_id", task.get("task_id", "unknown")))
    prompt = task["prompt"]  # HumanEval+ uses "prompt" for the function signature + docstring

    def make_client(role: str) -> TracedClient:
        base = lever_cfg.client_for_role(large_client, small_client, role)
        return TracedClient(
            base, tracer,
            task_id=task_id, system="groupchat", config=config_name,
            levers=lever_cfg.to_dict(), seed=seed, concurrency=concurrency,
        )

    # Turn 1 — Planner
    planner_msgs = [
        {"role": "system", "content": PLANNER_SYS},
        {"role": "user", "content": f"Problem:\n{prompt}"},
    ]
    planner_result = await make_client("planner").agenerate(
        planner_msgs, seed=seed, role="planner", round=1
    )
    plan = planner_result.text

    # Turn 2 — Coder (sees problem + plan)
    coder_msgs = [
        {"role": "system", "content": CODER_SYS},
        {"role": "user", "content": f"Problem:\n{prompt}\n\nPlan:\n{plan}"},
    ]
    coder_result = await make_client("coder").agenerate(
        coder_msgs, seed=seed, role="coder", round=2
    )
    solution = _extract_code(coder_result.text)

    # Turn 3 — Critic (sees problem + plan + solution)
    critic_msgs = [
        {"role": "system", "content": CRITIC_SYS},
        {"role": "user", "content": (
            f"Problem:\n{prompt}\n\nPlan:\n{plan}\n\nSolution:\n```python\n{solution}\n```"
        )},
    ]
    critic_result = await make_client("critic_check").agenerate(
        critic_msgs, seed=seed, role="critic_check", round=3
    )
    critic_verdict = critic_result.text.strip()

    return {
        "task_id": task_id,
        "solution": solution,
        "plan": plan,
        "critic_verdict": critic_verdict,
        "lgtm": critic_verdict.startswith("LGTM"),
        # eval/humaneval.py will execute the solution against tests separately
    }


async def run_groupchat_tasks(
    tasks: List[Dict[str, Any]],
    lever_cfg: LeverConfig,
    large_client: LLMClient,
    small_client: Optional[LLMClient],
    tracer: Tracer,
    seed: int,
    concurrency: int = 1,
    config_name: str = "unknown",
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(task):
        async with semaphore:
            return await run_one_groupchat(
                task, lever_cfg, large_client, small_client,
                tracer, seed, config_name, concurrency,
            )

    return list(await asyncio.gather(*[guarded(t) for t in tasks]))
