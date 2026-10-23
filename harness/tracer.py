"""Per-call tracer: wraps LLMClient to emit CallRecord rows to a JSONL trace file.

Every model call made through a TracedClient is logged as a JSON line with:
  task_id, system, config, levers, seed, concurrency, role, round,
  and all fields from CallResult (timings + token counts).

Usage:
    tracer = Tracer(run_dir=Path("results/runs/run_001"))
    client = TracedClient(LLMClient("qwen3-14b"), tracer, task_id="gsm8k_42",
                          system="debate", config="C1", levers={"l1": True},
                          seed=0, concurrency=1)
    result = await client.agenerate(messages, role="solver", round=1)
    # A JSON line was appended to results/runs/run_001/traces.jsonl

After all tasks finish, call tracer.close().  The tracer is thread- and coroutine-safe:
multiple TracedClients writing to the same Tracer are fine.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.client import CallResult, LLMClient, Messages


@dataclass
class CallRecord:
    """One logged row: a CallResult annotated with experiment context."""

    # experiment context
    task_id: str
    system: str        # "debate" | "groupchat"
    config: str        # "C0" … "C9"
    levers: Dict[str, bool]
    seed: int
    concurrency: int

    # call metadata (from the caller)
    role: str
    round: int

    # timing
    submit_time: float
    first_token_time: Optional[float]
    completion_time: float

    # token counts
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int

    # model used
    model: str

    @property
    def latency(self) -> float:
        return self.completion_time - self.submit_time

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.submit_time

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency"] = self.latency
        d["ttft"] = self.ttft
        return d


class Tracer:
    """Writes CallRecord rows to a JSONL file under run_dir.

    Thread-safe: multiple coroutines can call write() concurrently.
    """

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "traces.jsonl"
        self._lock = threading.Lock()
        self._fh = self._path.open("a", encoding="utf-8")

    def write(self, record: CallRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class TracedClient:
    """Wraps an LLMClient to log every call through a Tracer.

    Callers use this exactly like LLMClient; the extra context (task_id,
    system, config, levers, seed, concurrency) is set at construction time
    so it doesn't clutter the call sites.
    """

    def __init__(
        self,
        client: LLMClient,
        tracer: Tracer,
        task_id: str,
        system: str,
        config: str,
        levers: Dict[str, bool],
        seed: int,
        concurrency: int = 1,
    ) -> None:
        self._client = client
        self._tracer = tracer
        self._ctx = dict(
            task_id=task_id, system=system, config=config,
            levers=levers, seed=seed, concurrency=concurrency,
        )

    def _record(self, result: CallResult) -> None:
        rec = CallRecord(
            **self._ctx,
            role=result.role,
            round=result.round,
            submit_time=result.submit_time,
            first_token_time=result.first_token_time,
            completion_time=result.completion_time,
            prompt_tokens=result.prompt_tokens,
            cached_tokens=result.cached_tokens,
            output_tokens=result.output_tokens,
            model=result.model,
        )
        self._tracer.write(rec)

    def generate(self, messages: Messages, **kwargs) -> CallResult:
        result = self._client.generate(messages, **kwargs)
        self._record(result)
        return result

    async def agenerate(self, messages: Messages, **kwargs) -> CallResult:
        result = await self._client.agenerate(messages, **kwargs)
        self._record(result)
        return result

    # Expose the underlying spec so harnesses can read model metadata if needed.
    @property
    def spec(self):
        return self._client.spec
