# data/gsm8k

Grade-school math problems used by the multi-agent debate system.

| File | What it does |
|---|---|
| `download.py` | Downloads `openai/gsm8k` (main config) to `data/raw/gsm8k/`. |
| `preprocess.py` | Builds `eval` (200 test problems, seeded) and `dev` (30 train problems), and holds the answer parsing used for exact-match scoring. |

Processed fields: `id`, `question`, `solution`, `answer`. To score a model response, compare
`extract_final_number(response)` with `answer`.
