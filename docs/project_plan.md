# Project plan and preregistration summary

This page condenses the proposal and the preregistration so the decisions that shape the code
are easy to find. The full proposal is the source of truth. Any later revision should be
logged here with its date and reason.

## Research question

For published multi-agent systems served locally by an open-weight model on one H100 GPU, how
much of the end-to-end speedup from four standard levers is attributable to each lever, and
does the dominant lever depend on the system's communication shape? The headline metric is the
median task completion time at unchanged answer quality.

## Systems and datasets

| System | Shape | Dataset | Eval size | Folder |
|---|---|---|---|---|
| Multi-agent debate (3 agents, 2 rounds, majority vote) | Parallel fan-out and fan-in | GSM8K test | 200 problems, seeded | data/gsm8k |
| AutoGen SelectorGroupChat (planner, coder, critic) | Serialized turns over a growing transcript | HumanEval+ | all 164 problems | data/humanevalplus |
| Critic catch rate study | Guardrail for the group chat | Bugged HumanEval+ solutions | 50 mutants | data/bugged_solutions |
| ChatDev (stretch goal) | Long repeated role prompts | SRDD requirements | 50 tasks | data/chatdev_srdd |

## The four levers

| Id | Lever | Where it acts |
|---|---|---|
| L1 | Parallel dispatch: independent calls are issued concurrently with asyncio | Client layer |
| L2 | Request coalescing: identical prompts are merged into one request with n samples | Client layer |
| L3 | Prefix caching: vLLM automatic prefix caching on or off | Server flag |
| L4 | Model tiering: low-stakes calls go to the small model | Client layer |

## Models and serving

* Large model: Qwen3-14B (fallback Mistral-Nemo-12B-Instruct).
* Small model: Qwen3-1.7B (fallback Phi-4-mini-instruct).
* Both stay resident on the GPU as two vLLM instances with fixed memory fractions in every
  configuration, bf16, thinking mode off, temperature 0.7, top-p 0.8.
* The baseline runs calls sequentially, with prefix caching off and every call on the large model.

## Decision rule (written before any numbers are seen)

The hypothesis is supported only if all three hold:

1. Speedup S is at least 1.5 for both debate and group chat, and the lower bound of the 95%
   bootstrap interval over tasks is above 1.2.
2. Quality is held: accuracy drops by no more than 2.0 percentage points and critic catch rate
   by no more than 5 points. A lever that breaks this is inadmissible and S is recomputed
   without it.
3. The lever with the largest leave-one-out loss is L1 or L2 for debate, and L3 or L4 for the
   group chat.

Secondary (H2): supported if the add-one gain of L1 at 32 concurrent tasks is less than half
its gain at 1 task, for debate.

## Open items that affect the code

* **vLLM version.** Not yet pinned. Pin it in requirements-serving.txt before the first
  benchmark run.
* **Dev split overlap on HumanEval+.** The preregistration evaluates all 164 problems, so the
  30 problem dev subset used to freeze tiering assignments overlaps with eval. Decide whether
  to report results with or without the dev ids.
* **GPU memory fractions.** The defaults in models/model_config.py (0.60 and 0.20) are starting
  points and must be checked on the real node, then frozen for every configuration.
* **ICE partition and GPU names.** The sbatch files assume `--gres=gpu:H100:1`. Adjust to the
  queue actually available.

## Timeline

| Week | Work |
|---|---|
| 6 (Oct 1) | Proposal, ICE and vLLM setup, call tracer |
| 7 | Debate harness, baseline profile and critical path |
| 8 | AutoGen harness and EvalPlus sandbox, baseline profile |
| 9 | Levers L1 to L4 as switches, correctness checks |
| 10 (Oct 29) | Milestone with first debate numbers |
| 11 | Ten-configuration ablation for both systems |
| 12 | Load sweep, catch-rate study, ChatDev stretch |
| 13 | Analysis, figures, error analysis |
| 14 | Thanksgiving, rerun buffer |
| 15 (Dec 3) | Report, video, reproducible repository |
