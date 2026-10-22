# DLT Multi-Agent Inference

Multi-agent LLM systems, such as a group of debaters or a planner, coder and critic, tend to
give better answers than a single model, but they make many more model calls to get there.
Most research code pays for those calls the slow way: one after another, with the same prompts
re-read on every turn and one large model handling every request.

This project measures where the time actually goes when these systems run on a local open-weight
model, and how much each of four standard speedups helps: parallel dispatch, request
coalescing, prefix caching, and model tiering. We apply them to unmodified multi-agent
systems on a single H100, check that answer quality does not drop, and report which speedup
matters for which kind of workflow, and at what load it stops helping. We are not building a
new serving engine.

This is a course project for CSE 8803 (Fall 2026). The full proposal summary and the
preregistered decision rule are in [docs/project_plan.md](docs/project_plan.md).

## Status

Version 1 is the shared foundation: the model layer and the dataset layer. Version 2 adds the
call tracer, the debate harness with levers L1 and L2 as switches, and a closed-loop runner.
Version 2 was tested only with a fake model (no GPU), so its first real run is still to do.

| Area | Status |
|---|---|
| Model registry, download, vLLM serving, client, comparison script | Done |
| Dataset download and preprocessing (GSM8K, HumanEval+, bugged solutions, SRDD) | Done |
| Call tracer (per-call and per-task records, critical path) | Done, tested with a fake model |
| Debate harness (prompts copied from the released code) | Done, tested with a fake model |
| Levers L1 (parallel dispatch) and L2 (coalescing) for debate | Done, tested with a fake model |
| Closed-loop runner with concurrency C, resume, run config capture | Done, tested with a fake model |
| First run against a real vLLM server | Not started |
| Lever L3 (prefix caching) | Server flag exists; needs the first real run |
| Lever L4 (model tiering) | Not started |
| AutoGen group chat harness | Not started |
| Analysis scripts (ablation table, load sweep) | Not started |

## Repository layout

```
DLT-Multi-Agent-Inference/
|-- README.md                  This file
|-- requirements.txt           Client-side Python dependencies
|-- requirements-serving.txt   vLLM, installed on the GPU node only
|-- docs/
|   `-- project_plan.md        Research question, levers, decision rule, open items, timeline
|-- models/                    Everything about the language models
|   |-- model_config.py        Registry of models, ports, memory fractions, sampling defaults
|   |-- download_models.py     Download weights from the Hugging Face Hub
|   |-- serve_vllm.py          Start a vLLM OpenAI-compatible server for one model
|   |-- client.py              LLMClient: sync and async calls with timing and token counts
|   |-- run_model.py           Send one prompt to a served model (smoke test)
|   `-- compare_models.py      Run the same prompts through several models
|-- data/                      Everything about the datasets
|   |-- dataset_config.py      Registry of datasets and their folder paths
|   |-- prepare_all.py         Download and preprocess everything in one command
|   |-- loader.py              load_split(): the function experiment code calls
|   |-- common/                JSONL helpers, seeded sampling, manifests, Hub download helper
|   |-- gsm8k/                 download.py, preprocess.py (also answer parsing for scoring)
|   |-- humanevalplus/         download.py, preprocess.py (also build_program for scoring)
|   |-- bugged_solutions/      download.py, mutations.py, preprocess.py
|   |-- chatdev_srdd/          download.py, preprocess.py (stretch goal)
|   |-- raw/                   Downloaded files (git-ignored)
|   `-- processed/             eval.jsonl, dev.jsonl, manifest.json per dataset (git-ignored)
|-- tracing/                   Call tracer: per-call and per-task timing records
|   `-- tracer.py
|-- systems/                   The multi-agent systems under test
|   `-- debate.py              Debate harness with the L1 and L2 switches
|-- experiments/               Entry points that run a system and record results
|   `-- run_debate.py          Closed-loop GSM8K debate run, writes traces and a summary
|-- slurm/                     Batch scripts for the ICE cluster
|   |-- download_models.sbatch
|   |-- prepare_data.sbatch
|   |-- serve_models.sbatch    Serves the large and small model together
|   `-- run_debate.sbatch      Serves both models, runs the debate, stops the servers
|-- tests/                     Unit tests (pytest), no GPU or network needed
|-- results/                   Committed experiment outputs, tables, and figures
`-- logs/                      Cluster and run logs (git-ignored)
```

Each folder has its own short README that lists its files and what they do.

## Setup

```bash
git clone https://github.com/arihanth26/DLT-Multi-Agent-Inference.git
cd DLT-Multi-Agent-Inference
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # should pass without a GPU
```

On the GPU node, also install the serving dependencies in the environment that runs vLLM:

```bash
pip install -r requirements-serving.txt
```

All commands below are run from the repository root with `python -m`, so imports resolve
without extra path settings.

## Models

The registry in [models/model_config.py](models/model_config.py) is the single place that
defines which models exist. Each model has a short key that is used on every command line.

| Key | Hugging Face id | Role | Default port |
|---|---|---|---|
| `qwen3-14b` | Qwen/Qwen3-14B | large (default) | 8000 |
| `mistral-nemo-12b` | mistralai/Mistral-Nemo-Instruct-2407 | large (fallback) | 8000 |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | small (default) | 8001 |
| `phi-4-mini` | microsoft/Phi-4-mini-instruct | small (fallback) | 8001 |

To add a model, add one line to the `MODELS` dictionary. Every script picks it up.

**1. Download the weights** (once, on a node with network access):

```bash
python -m models.download_models                     # default large and small pair
python -m models.download_models --all               # every registered model
```

**2. Start a server.** The prefix caching choice is explicit because it is one of the levers.
The baseline must use `--no-prefix-caching`.

```bash
python -m models.serve_vllm --model qwen3-14b --no-prefix-caching
python -m models.serve_vllm --model qwen3-1.7b --no-prefix-caching    # in a second process
```

On ICE, `sbatch slurm/serve_models.sbatch off` starts both servers on one GPU, and
`sbatch slurm/serve_models.sbatch on` does the same with prefix caching enabled.

**3. Call the model from your own code** through `LLMClient`, so that every experiment gets the
same sampling settings and the same timing fields:

```python
from models.client import LLMClient

client = LLMClient("qwen3-14b")                     # http://localhost:8000/v1 by default
result = client.generate([{"role": "user", "content": "What is 17 * 24?"}], seed=0)
print(result.text, result.latency, result.prompt_tokens, result.cached_tokens)

# Async version, used for parallel dispatch, and n samples in one request for coalescing
results = await asyncio.gather(client.agenerate(msgs, seed=0), client.agenerate(msgs, seed=1))
batch = await client.agenerate(msgs, n=3, seed=0)   # batch.texts has three completions
```

A `CallResult` carries the generated text(s), submit, first-token and completion times, prompt,
cached and output token counts, and optional `role` and `round` labels for the tracer.
First-token time is only recorded for single-sample requests.

**4. Try things out.**

```bash
python -m models.run_model --model qwen3-14b --prompt "Explain prefix caching in one sentence."
python -m models.compare_models --prompts prompts.jsonl --models qwen3-14b qwen3-1.7b
```

`compare_models` reads a JSONL file with a `prompt` field per line and writes one row per
(prompt, model) with the answer, latency and token counts to `logs/compare.jsonl`. This is the
easiest way to compare models side by side. If a model runs on another node, pass
`--endpoint qwen3-14b=http://<node>:8000/v1`.

Sampling defaults follow the proposal: temperature 0.7, top-p 0.8, Qwen3 thinking mode off.

## Datasets

| Name | Used by | Source | Processed splits |
|---|---|---|---|
| `gsm8k` | Debate | openai/gsm8k on the Hub | eval: 200 problems from test, dev: 30 from train |
| `humanevalplus` | Group chat | evalplus/humanevalplus on the Hub | eval: all 164, dev: 30 sampled |
| `bugged_solutions` | Critic catch rate | Derived from HumanEval+ | eval: 50 verified buggy solutions |
| `chatdev_srdd` | ChatDev (stretch) | SRDD table in the ChatDev repository | eval: 50, dev: 30 |

Build everything with one command:

```bash
python -m data.prepare_all                       # core datasets
python -m data.prepare_all --include-stretch     # also ChatDev SRDD
python -m data.prepare_all --datasets gsm8k      # a single dataset
```

On ICE, use `sbatch slurm/prepare_data.sbatch`. Raw files land in `data/raw/<name>/` and
processed splits in `data/processed/<name>/`. Both folders are git-ignored because they are
rebuilt from the scripts.

Selection is fixed by a seed (default 0, the preregistered value), so everyone gets identical
splits. Each dataset writes a `manifest.json` with the split sizes and an id fingerprint. If two
people get different fingerprints, their data differs.

Load a split in experiment code with:

```python
from data.loader import load_split

problems = load_split("gsm8k", "eval")      # list of dicts
dev = load_split("humanevalplus", "dev")
```

Row fields:

| Dataset | Fields |
|---|---|
| `gsm8k` | `id`, `question`, `solution`, `answer` (final number as a clean string) |
| `humanevalplus` | `id`, `prompt`, `entry_point`, `canonical_solution`, `test` |
| `bugged_solutions` | `id`, `task_id`, `prompt`, `entry_point`, `buggy_solution`, `mutation_kind`, `mutation_description` |
| `chatdev_srdd` | `id`, `name`, `description`, `category` |

Scoring helpers live next to the data they belong to:

* `data.gsm8k.preprocess.extract_final_number` pulls the last number from a model response,
  and `normalize_number` makes `1,200.00` and `1200` compare equal. Exact match compares these.
* `data.humanevalplus.preprocess.build_program` assembles a runnable script that checks a
  candidate solution against the HumanEval+ tests. HumanEval+ tests import numpy.
* `data.bugged_solutions.preprocess.run_program` runs a program in a separate Python process
  with a timeout and returns `pass`, `fail` or `timeout`.

The bugged solutions are made by injecting one bug into a canonical solution, either an
operator swap (`+` and `-`, `<` and `<=`, `>` and `>=`, `==` and `!=`) or adding one to an
integer literal. A mutant is kept only if the canonical solution passes the tests and the
mutant fails them, so every bug in the set is real.

## Running the debate

The baseline is sequential calls, no coalescing, prefix caching off, every call on the large
model. Each lever is a switch on top of it. Start the servers first (step 2 above), then:

```bash
# Baseline, 20 problems, 3 warm-up debates that are discarded
python -m experiments.run_debate --prefix-caching off --limit 20 --warmup 3

# L1 parallel dispatch, 4 debates in flight, server started with prefix caching on
python -m experiments.run_debate --prefix-caching on --parallel --concurrency 4
```

On ICE, `sbatch slurm/run_debate.sbatch off --limit 20 --warmup 3` does both steps in one job.
If a job is cut off, run the same command again with `--resume`.

`--prefix-caching` records how the server was started. It does not change the server, so it
must match the `serve_vllm` flag. The summary warns if the cache hit fraction disagrees.

Each run writes a folder `logs/runs/<run-name>/` (git-ignored):

| File | Content |
|---|---|
| `config.json` | Arguments, levers, sampling settings, git commit, package versions |
| `calls.jsonl` | One row per model call: timings, prompt, cached and output tokens, agent, round |
| `tasks.jsonl` | One row per debate: completion time, idle time, critical path, tokens, correct |
| `transcripts.jsonl` | Every agent's messages, parsed answers, the vote and the gold answer |
| `summary.json` | Median, mean and p95 completion time, accuracy, cache hit fraction, idle fraction |

To keep a result, copy `summary.json` and `config.json` into `results/debate/` (see
[results/README.md](results/README.md)). Per-call and per-task fields are defined in
[tracing/tracer.py](tracing/tracer.py); scoring and prompts are in
[systems/debate.py](systems/debate.py).

## Where to find things

| I want to... | Look at |
|---|---|
| Change which models we use, or their ports and memory share | `models/model_config.py` |
| Start or stop a model server | `models/serve_vllm.py`, `slurm/serve_models.sbatch` |
| Call a model from experiment code | `models/client.py` |
| Compare models on the same prompts | `models/compare_models.py` |
| Run the debate under a configuration | `experiments/run_debate.py` |
| Change the debate prompts, voting or scoring | `systems/debate.py` |
| Change what is recorded per call or per task | `tracing/tracer.py` |
| Change how a dataset is sampled or cleaned | `data/<dataset>/preprocess.py` |
| Add a new dataset | Copy a folder in `data/`, then register it in `data/dataset_config.py` |
| Score a GSM8K answer | `data/gsm8k/preprocess.py` |
| Score a HumanEval+ solution | `data/humanevalplus/preprocess.py` and `data/bugged_solutions/preprocess.py` |
| Save or find experiment outputs, tables, and figures | `results/` |
| See the decision rule and open items | `docs/project_plan.md` |

## Conventions

* Every file starts with a header comment saying what it is for and how to run it, and every
  function has a docstring saying what it does.
* All commands run from the repository root as `python -m package.module`.
* Datasets, model weights, and logs are never committed. Small result files go in `results/`.
* Experiments should use a fixed seed and record it. The preregistered seeds are 0, 1 and 2.
