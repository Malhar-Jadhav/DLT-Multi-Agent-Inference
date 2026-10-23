"""The four inference levers as independent boolean switches.

Each lever is a single flag in a LeverConfig.  LeverConfig.from_dict() loads one
from a YAML config dict.  build_clients() resolves the config into two LLMClient
instances (one large, one small) and a coalesce helper, ready to use.

The four levers
---------------
L1 - parallel_dispatch
    True  → callers use asyncio.gather to fire independent calls concurrently.
    False → callers await each call in sequence (the baseline).
    Implemented at call-site; this flag is read by harnesses (debate.py, groupchat.py).

L2 - request_coalescing
    True  → identical-prompt calls are sent as a single n= request.
    False → each call is a separate request.
    Implemented via coalesce_n() helper below.

L3 - prefix_caching
    True  → use the vLLM server that was started WITH --enable-prefix-caching.
             The port difference is encoded in model_config.py via CachedModelSpec
             or by the serve script writing a port file.  Here we simply use the
             model keys as registered; the caller passes use_cache=True so that
             build_clients() selects the right base_url.
    False → use the server started with --no-prefix-caching (the baseline server).
    NOTE: In practice both servers run on the same port; caching is toggled at
          server start-up via the sbatch --cache flag.  This flag controls which
          base_url is selected; see CACHE_PORT_OFFSET below.

L4 - model_tiering
    True  → low-stakes roles (guard, critic-check, majority-vote) use the small model.
    False → every call uses the large model.
    Implemented via LeverConfig.client_for_role().
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.client import CallResult, LLMClient, Messages
from models.model_config import DEFAULT_LARGE, DEFAULT_SMALL, SamplingConfig, DEFAULT_SAMPLING

# Roles considered "low-stakes" when model tiering is active.
LOW_STAKES_ROLES = frozenset({"guard", "critic_check", "vote_aggregator"})

# When prefix-caching is on, we expect the server was launched with that flag.
# The port is the same; the difference is recorded in the run manifest.
# If you run both servers simultaneously on different ports, override base_url here.
CACHE_PORT_OFFSET = 0  # set to e.g. 100 if you run a separate cached server on port+100


@dataclass(frozen=True)
class LeverConfig:
    """All four lever settings for one experiment configuration."""

    name: str               # e.g. "C0", "C1", "C5"
    l1_parallel: bool = False
    l2_coalesce: bool = False
    l3_cache: bool = False
    l4_tier: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeverConfig":
        return cls(
            name=d["name"],
            l1_parallel=bool(d.get("l1_parallel", False)),
            l2_coalesce=bool(d.get("l2_coalesce", False)),
            l3_cache=bool(d.get("l3_cache", False)),
            l4_tier=bool(d.get("l4_tier", False)),
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "l1": self.l1_parallel,
            "l2": self.l2_coalesce,
            "l3": self.l3_cache,
            "l4": self.l4_tier,
        }

    def client_for_role(
        self,
        large_client: LLMClient,
        small_client: Optional[LLMClient],
        role: str,
    ) -> LLMClient:
        """Return the appropriate client based on the role and tiering flag."""
        if self.l4_tier and small_client is not None and role in LOW_STAKES_ROLES:
            return small_client
        return large_client


def build_clients(
    cfg: LeverConfig,
    large_model: str = DEFAULT_LARGE,
    small_model: str = DEFAULT_SMALL,
    sampling: SamplingConfig = DEFAULT_SAMPLING,
) -> tuple[LLMClient, Optional[LLMClient]]:
    """Create the large and (optionally) small LLMClient for this config.

    Returns (large_client, small_client).  small_client is None when L4 is off.
    """
    large = LLMClient(large_model, sampling=sampling)
    small = LLMClient(small_model, sampling=sampling) if cfg.l4_tier else None
    return large, small


# ---------------------------------------------------------------------------
# L2 coalescing helper
# ---------------------------------------------------------------------------

async def coalesce_n(
    client: LLMClient,
    messages: Messages,
    n: int,
    seed: int,
    role: str = "",
    round: int = -1,
) -> List[CallResult]:
    """Send one n-sample request and return n CallResult objects (one per completion).

    When L2 is off, the caller should just fire n individual agenerate() calls.
    When L2 is on, call this instead: it sends a single request with n= and
    unpacks the texts into separate CallResult-like objects for uniform handling.

    Note: n>1 requests are not streamed, so first_token_time will be None.
    """
    result = await client.agenerate(messages, n=n, seed=seed, role=role, round=round)
    # Split into per-completion results so the rest of the harness sees one result per agent.
    results: List[CallResult] = []
    for i, text in enumerate(result.texts):
        from models.client import CallResult as CR
        per = CR(
            texts=[text],
            submit_time=result.submit_time,
            completion_time=result.completion_time,
            prompt_tokens=result.prompt_tokens if i == 0 else 0,
            cached_tokens=result.cached_tokens if i == 0 else 0,
            output_tokens=result.output_tokens // len(result.texts),
            model=result.model,
            first_token_time=None,
            role=role,
            round=round,
        )
        results.append(per)
    return results


# ---------------------------------------------------------------------------
# L1 parallel dispatch helpers
# ---------------------------------------------------------------------------

async def parallel_gather(*coros) -> tuple:
    """Thin alias for asyncio.gather; makes L1 intent explicit at call sites."""
    return await asyncio.gather(*coros)


async def sequential_gather(*coros) -> tuple:
    """Run coroutines one after another; the baseline (L1 off) equivalent."""
    results = []
    for coro in coros:
        results.append(await coro)
    return tuple(results)


async def dispatch(cfg: LeverConfig, *coros) -> tuple:
    """Run coroutines in parallel (L1 on) or sequentially (L1 off)."""
    if cfg.l1_parallel:
        return await parallel_gather(*coros)
    return await sequential_gather(*coros)
