# systems

The multi-agent systems under test. Each harness takes a client (`models.client.LLMClient` or any
object with the same `agenerate`), a problem, a lever configuration and a tracer.

| File | What it does |
|---|---|
| `debate.py` | Multi-agent debate on GSM8K: 3 agents, 2 rounds, majority vote, with the prompts of the released code of Du et al. Levers L1 (parallel dispatch) and L2 (coalescing round 1) are switches on `Levers`. |

Scoring differs from the released script in three deliberate ways, all documented at the top of
`debate.py` and in `docs/project_plan.md`: an unparseable answer counts as wrong, it does not vote,
and a leading minus sign is kept.
