# data/humanevalplus

HumanEval+ coding problems used by the AutoGen group chat system.

| File | What it does |
|---|---|
| `download.py` | Downloads `evalplus/humanevalplus` to `data/raw/humanevalplus/`. |
| `preprocess.py` | Builds `eval` (all 164 problems) and `dev` (30 sampled), and provides `build_program` for checking a solution against the tests. |

Processed fields: `id`, `prompt`, `entry_point`, `canonical_solution`, `test`. The tests import
numpy. The dev split overlaps eval, see `docs/project_plan.md`.
