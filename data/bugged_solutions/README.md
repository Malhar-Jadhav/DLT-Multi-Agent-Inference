# data/bugged_solutions

Fifty HumanEval+ solutions with one injected bug each, used to measure how often the critic
agent catches a wrong solution.

| File | What it does |
|---|---|
| `mutations.py` | Finds mutation sites in Python source and applies one change: an operator swap or an integer literal plus one. |
| `preprocess.py` | Builds the 50 mutants from the processed HumanEval+ data and keeps only those that really fail the tests. Also has `run_program`, which runs code in a subprocess with a timeout. |
| `download.py` | Placeholder, since the data is derived from HumanEval+. |

This dataset needs `data/processed/humanevalplus` first. `python -m data.prepare_all` handles
the order. Processed fields: `id`, `task_id`, `prompt`, `entry_point`, `buggy_solution`,
`mutation_kind`, `mutation_description`.
