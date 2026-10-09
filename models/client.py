"""Client for talking to a vLLM server through its OpenAI-compatible API.

All experiment code should call models through LLMClient instead of using the openai package
directly. It applies the project sampling defaults, turns Qwen3 thinking mode off, supports
n-sample requests (used by the coalescing lever), and returns a CallResult that already holds
the timing and token fields the tracer needs: submit time, first-token time, completion time,
prompt tokens, cached prompt tokens, and output tokens.

First-token time is measured only for single-sample requests (n == 1), because those are
streamed. Requests with n > 1 are not streamed and report first_token_time as None.

Usage:
    client = LLMClient("qwen3-14b")
    result = client.generate([{"role": "user", "content": "What is 2 + 2?"}], seed=0)
    result = await client.agenerate(messages, n=3, seed=0)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.model_config import DEFAULT_SAMPLING, ModelSpec, SamplingConfig, get_model

Messages = List[Dict[str, str]]


@dataclass
class CallResult:
    """Output and measurements for one model call.

    texts: the generated completions, one per requested sample.
    submit_time, completion_time: time.perf_counter() readings in seconds.
    first_token_time: perf_counter reading of the first streamed token, or None if not streamed.
    prompt_tokens, cached_tokens, output_tokens: token counts reported by the server.
    model: registry key of the model that served the call.
    role, round: optional labels set by the caller for tracing (agent role, debate round).
    """

    texts: List[str]
    submit_time: float
    completion_time: float
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    model: str
    first_token_time: Optional[float] = None
    role: str = ""
    round: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency(self) -> float:
        """Total seconds from submit to completion."""
        return self.completion_time - self.submit_time

    @property
    def text(self) -> str:
        """First completion, for the common single-sample case."""
        return self.texts[0]


class LLMClient:
    """Thin wrapper around the OpenAI client bound to one served model."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        sampling: SamplingConfig = DEFAULT_SAMPLING,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
    ) -> None:
        """Create sync and async clients for a model. base_url defaults to localhost on the model's port."""
        from openai import AsyncOpenAI, OpenAI

        self.spec: ModelSpec = get_model(model)
        self.base_url = base_url or f"http://localhost:{self.spec.port}/v1"
        self.sampling = sampling
        self._sync = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)
        self._async = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)

    def _request_kwargs(
        self, messages: Messages, n: int, seed: Optional[int],
        max_tokens: Optional[int], temperature: Optional[float], top_p: Optional[float],
    ) -> Dict[str, Any]:
        """Assemble the keyword arguments of a chat completion request."""
        kwargs: Dict[str, Any] = dict(
            model=self.spec.key,
            messages=messages,
            n=n,
            temperature=self.sampling.temperature if temperature is None else temperature,
            top_p=self.sampling.top_p if top_p is None else top_p,
            max_tokens=max_tokens or self.sampling.max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        if self.spec.thinking_toggle:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if n == 1:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    @staticmethod
    def _cached_tokens(usage: Any) -> int:
        """Read cached prompt tokens from a usage object, returning 0 when the server omits them."""
        details = getattr(usage, "prompt_tokens_details", None)
        return int(getattr(details, "cached_tokens", 0) or 0) if details else 0

    def _build_result(self, texts, usage, submit, first, done, role, round_) -> CallResult:
        """Package texts, usage, and timestamps into a CallResult."""
        return CallResult(
            texts=texts, submit_time=submit, completion_time=done,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            cached_tokens=self._cached_tokens(usage),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=self.spec.key, first_token_time=first, role=role, round=round_,
        )

    def generate(
        self, messages: Messages, n: int = 1, seed: Optional[int] = None,
        max_tokens: Optional[int] = None, temperature: Optional[float] = None,
        top_p: Optional[float] = None, role: str = "", round: int = -1,
    ) -> CallResult:
        """Send one blocking chat request and return its CallResult."""
        kwargs = self._request_kwargs(messages, n, seed, max_tokens, temperature, top_p)
        submit = time.perf_counter()
        if n > 1:
            resp = self._sync.chat.completions.create(**kwargs)
            done = time.perf_counter()
            texts = [c.message.content or "" for c in sorted(resp.choices, key=lambda c: c.index)]
            return self._build_result(texts, resp.usage, submit, None, done, role, round)
        parts: List[str] = []
        first: Optional[float] = None
        usage = None
        for chunk in self._sync.chat.completions.create(**kwargs):
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter()
                parts.append(chunk.choices[0].delta.content)
        return self._build_result(["".join(parts)], usage, submit, first, time.perf_counter(), role, round)

    async def agenerate(
        self, messages: Messages, n: int = 1, seed: Optional[int] = None,
        max_tokens: Optional[int] = None, temperature: Optional[float] = None,
        top_p: Optional[float] = None, role: str = "", round: int = -1,
    ) -> CallResult:
        """Async version of generate, used by the parallel dispatch lever (asyncio.gather)."""
        kwargs = self._request_kwargs(messages, n, seed, max_tokens, temperature, top_p)
        submit = time.perf_counter()
        if n > 1:
            resp = await self._async.chat.completions.create(**kwargs)
            done = time.perf_counter()
            texts = [c.message.content or "" for c in sorted(resp.choices, key=lambda c: c.index)]
            return self._build_result(texts, resp.usage, submit, None, done, role, round)
        parts: List[str] = []
        first: Optional[float] = None
        usage = None
        stream = await self._async.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter()
                parts.append(chunk.choices[0].delta.content)
        return self._build_result(["".join(parts)], usage, submit, first, time.perf_counter(), role, round)

    def is_ready(self) -> bool:
        """Return True if the server answers a model listing request."""
        try:
            self._sync.models.list()
            return True
        except Exception:
            return False
