# data/chatdev_srdd

Software requirements from the ChatDev SRDD table, used by the ChatDev stretch goal.

| File | What it does |
|---|---|
| `download.py` | Downloads the 1200 row SRDD CSV from the ChatDev repository (chatdev1.0 branch) to `data/raw/chatdev_srdd/srdd.csv`. |
| `preprocess.py` | Builds `eval` (50 requirements, seeded) and `dev` (30 more, disjoint from eval). |

Processed fields: `id`, `name`, `description`, `category`. Build it with
`python -m data.prepare_all --include-stretch`.
