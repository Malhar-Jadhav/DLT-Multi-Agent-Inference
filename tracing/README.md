# tracing

Records what happens during a run so that the speedup can be explained, not just measured.

| File | What it does |
|---|---|
| `tracer.py` | `Tracer` stores one `CallRecord` per model call and one `TaskRecord` per task, appending them to `calls.jsonl` and `tasks.jsonl` as they happen. `summarize` turns task rows into the numbers of a run. |

All times are seconds from `time.perf_counter()`, the same clock `models/client.py` uses.
The preregistered metric is `TaskRecord.completion_time`, from the first call's submission to
the final answer being recorded. `idle_time` is the part of it with no call in flight, and
`critical_path` is the sum over rounds of the slowest call, which bounds what parallel dispatch can reach.
