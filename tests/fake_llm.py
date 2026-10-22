"""A stand-in for LLMClient used by the tests, so no GPU, server or network is needed.

FakeLLM has the same agenerate signature as models.client.LLMClient and returns CallResult
objects with real timings (it sleeps for `delay` seconds per call). Each reply contains a
marker with the agent role and round, then a boxed number taken from `values`, so tests can
check which agent saw which text. A value of None produces a reply with no number at all.

It also keeps every request it received (messages, n, seed, role, round) and the largest
number of requests that were in flight at the same time.
"""

import asyncio
import copy
import time
from typing import Any, Dict, List, Optional

from models.client import CallResult


class FakeLLM:
    """Deterministic fake model. values[i] is the number agent i answers with."""

    def __init__(self, values: Optional[List[Optional[str]]] = None, delay: float = 0.01, fail_marker: str = "") -> None:
        """values: one answer per agent. delay: seconds per call. fail_marker: raise if a prompt contains it."""
        self.values = values if values is not None else ["18", "18", "18"]
        self.delay = delay
        self.fail_marker = fail_marker
        self.calls: List[Dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def _reply(self, agent: int, role: str, rnd: int) -> str:
        """Reply text for one agent in one round."""
        value = self.values[agent]
        if value is None:
            return "I do not know."
        return f"[{role}|{rnd}] some reasoning \\boxed{{{value}}}"

    async def agenerate(
        self, messages: List[Dict[str, str]], n: int = 1, seed: Optional[int] = None,
        max_tokens: Optional[int] = None, temperature: Optional[float] = None,
        top_p: Optional[float] = None, role: str = "", round: int = -1,
    ) -> CallResult:
        """Pretend to run one chat request."""
        self.calls.append({
            "messages": copy.deepcopy(messages), "n": n, "seed": seed, "role": role, "round": round,
        })
        if self.fail_marker and self.fail_marker in messages[0]["content"]:
            raise RuntimeError("fake failure")
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        submit = time.perf_counter()
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        done = time.perf_counter()
        if n > 1:
            texts = [self._reply(i, f"agent{i}", round) for i in range(n)]
            first = None
        else:
            agent = int(role.replace("agent", "")) if role.startswith("agent") and role[5:].isdigit() else 0
            texts = [self._reply(agent, role, round)]
            first = submit + self.delay / 2
        return CallResult(
            texts=texts, submit_time=submit, completion_time=done,
            prompt_tokens=sum(len(m["content"].split()) for m in messages), cached_tokens=0,
            output_tokens=sum(len(t.split()) for t in texts), model="fake",
            first_token_time=first, role=role, round=round,
        )
