"""Multi-agent debate on GSM8K, following the released code of Du et al. (ICML 2024).

Three agents answer the same question independently, then in round 2 each agent reads the
other two agents' round 1 answers and answers again. The final answer is the majority vote
over the agents' last answers. Prompts and message layout are copied from the released
script (llm_multiagent_debate, gsm/gen_gsm.py): the same question template, the same
"One agent solution" template, and the same context layout [user, assistant, user, assistant].

Two levers from the project plan are switches here; the baseline has both off:
  * L1 parallel dispatch: the agents of a round are called concurrently with asyncio.gather
    instead of one after another.
  * L2 request coalescing: in round 1 the three prompts are identical, so they are sent as
    one request asking for n = 3 samples. Round 2 prompts differ per agent, so they are not
    coalesced.
Prefix caching (L3) is a server flag and model tiering (L4) is not implemented yet.

Deviations from the released scoring (eval_gsm.py), kept on purpose:
  * An answer that cannot be parsed counts as wrong. The released code counts it as correct.
  * Unparseable answers do not take part in the vote.
  * A leading minus sign is kept.

Usage (see experiments/run_debate.py for the command line):
    result = await run_debate(client, problem, Levers(parallel=True), tracer, task_index=0)
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from data.gsm8k.preprocess import extract_final_number, normalize_number
from models.client import CallResult
from tracing.tracer import Tracer

NUM_AGENTS = 3
NUM_ROUNDS = 2

# Templates copied from gsm/gen_gsm.py of the released code. The doubled braces are format escapes.
ROUND1_PROMPT = (
    "Can you solve the following math problem? {} Explain your reasoning. Your final answer "
    "should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. "
)
OTHERS_PREFIX = "These are the solutions to the problem from other agents: "
OTHERS_ONE = "\n\n One agent solution: ```{}```"
OTHERS_SUFFIX = (
    "\n\n Using the solutions from other agents as additional information, can you provide your "
    "answer to the math problem? \n The original math problem is {}. Your final answer should be a "
    "single numerical number, in the form \\boxed{{answer}}, at the end of your response."
)

Messages = List[Dict[str, str]]


class AsyncLLM(Protocol):
    """What the harness needs from a client: models.client.LLMClient satisfies it."""

    async def agenerate(
        self, messages: Messages, n: int = 1, seed: Optional[int] = None,
        max_tokens: Optional[int] = None, temperature: Optional[float] = None,
        top_p: Optional[float] = None, role: str = "", round: int = -1,
    ) -> CallResult:
        """Send one chat request and return its CallResult."""
        ...


@dataclass(frozen=True)
class Levers:
    """Which client-side levers are on. The baseline is Levers() with everything off."""

    parallel: bool = False  # L1
    coalesce: bool = False  # L2

    @property
    def label(self) -> str:
        """Short name used in run folders, for example "baseline" or "L1+L2"."""
        parts = [name for name, on in (("L1", self.parallel), ("L2", self.coalesce)) if on]
        return "+".join(parts) if parts else "baseline"


@dataclass
class DebateResult:
    """Outcome of one debate: per-agent answers, the vote, correctness, and the full transcripts."""

    task_id: str
    answers: List[Optional[str]]
    final: Optional[str]
    gold: Optional[str]
    correct: Optional[bool]
    contexts: List[Messages] = field(default_factory=list)


def build_round1_message(question: str) -> Dict[str, str]:
    """The first user message, identical for every agent."""
    return {"role": "user", "content": ROUND1_PROMPT.format(question)}


def build_others_message(other_contexts: List[Messages], question: str, index: int) -> Dict[str, str]:
    """The follow-up user message showing the other agents' answers, as in the released construct_message.

    index is the position of the answer to show inside each other agent's context: 2 * round - 1,
    which is the assistant message of the previous round.
    """
    text = OTHERS_PREFIX
    for context in other_contexts:
        text += OTHERS_ONE.format(context[index]["content"])
    text += OTHERS_SUFFIX.format(question)
    return {"role": "user", "content": text}


def call_seed(base_seed: int, task_index: int, agent: int, rnd: int) -> int:
    """Deterministic per-request seed, so a rerun of the same problem uses the same seeds."""
    return (base_seed * 1_000_003 + task_index) * 101 + agent * 7 + rnd


def _clean_number(text: str) -> Optional[str]:
    """Normalize the text of a boxed answer, returning None if no digits are left."""
    cleaned = normalize_number(text).strip().rstrip(".")
    return cleaned if re.search(r"\d", cleaned) else None


def parse_answer(response: str) -> Optional[str]:
    """Final numeric answer of one response: the last \\boxed{...} number, else the last number, else None.

    The boxed pattern is the released one (a braced group of digits, dots, commas and dollar
    signs), extended to allow a leading minus.
    """
    for match in reversed(re.findall(r"\{\s*(-?[0-9.,$]*)\s*\}", response)):
        value = _clean_number(match)
        if value is not None:
            return value
    fallback = extract_final_number(response)
    return _clean_number(fallback) if fallback is not None else None


def most_frequent(answers: List[Optional[str]]) -> Optional[str]:
    """Majority vote over parsed answers; ties go to the answer given first. None if nothing parsed."""
    valid = [a for a in answers if a is not None]
    best: Optional[str] = None
    best_count = 0
    for a in valid:
        count = valid.count(a)
        if count > best_count:
            best, best_count = a, count
    return best


def answers_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    """True if both numbers are present and equal as text or as floats. A missing prediction is wrong."""
    if predicted is None or gold is None:
        return False
    if predicted == gold:
        return True
    try:
        return float(predicted) == float(gold)
    except ValueError:
        return False


async def run_debate(
    client: AsyncLLM,
    problem: Dict[str, Any],
    levers: Levers,
    tracer: Tracer,
    task_index: int = 0,
    base_seed: int = 0,
    num_agents: int = NUM_AGENTS,
    num_rounds: int = NUM_ROUNDS,
    max_tokens: Optional[int] = None,
) -> DebateResult:
    """Run one debate on a GSM8K problem and trace every call.

    problem needs "id", "question" and "answer" (the gold number as a string).
    task_index and base_seed only feed the per-request seeds.
    """
    task_id, question = problem["id"], problem["question"]
    contexts: List[Messages] = [[build_round1_message(question)] for _ in range(num_agents)]
    tracer.begin_task(task_id)

    async def call(agent: int, rnd: int) -> str:
        """One agent's call in one round; records the trace and returns the reply text."""
        result = await client.agenerate(
            list(contexts[agent]), seed=call_seed(base_seed, task_index, agent, rnd),
            max_tokens=max_tokens, role=f"agent{agent}", round=rnd,
        )
        tracer.record_call(task_id, result, agent=agent)
        return result.text

    for rnd in range(num_rounds):
        if rnd > 0:
            for i in range(num_agents):
                others = contexts[:i] + contexts[i + 1:]
                contexts[i].append(build_others_message(others, question, 2 * rnd - 1))
        if rnd == 0 and levers.coalesce:
            result = await client.agenerate(
                list(contexts[0]), n=num_agents, seed=call_seed(base_seed, task_index, 0, 0),
                max_tokens=max_tokens, role="agents", round=0,
            )
            tracer.record_call(task_id, result, agent=-1, n=num_agents)
            replies = list(result.texts)
        elif levers.parallel:
            replies = list(await asyncio.gather(*(call(i, rnd) for i in range(num_agents))))
        else:
            replies = [await call(i, rnd) for i in range(num_agents)]
        for i, reply in enumerate(replies):
            contexts[i].append({"role": "assistant", "content": reply})

    answers = [parse_answer(context[-1]["content"]) for context in contexts]
    final = most_frequent(answers)
    gold = problem.get("answer")
    correct = answers_match(final, gold) if gold is not None else None
    tracer.end_task(task_id, correct=correct)
    return DebateResult(task_id, answers, final, gold, correct, contexts)
