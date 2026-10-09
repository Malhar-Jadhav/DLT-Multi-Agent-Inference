# results

Committed outputs of experiments: summary tables, small JSON or CSV files, and figures used in
the report. Large raw outputs, such as full call traces, stay out of git and go to cluster
storage instead (see `logs/`).

Suggested layout, created as each part of the project produces output:

| Folder | Contents |
|---|---|
| `debate/` | Debate runs: per-configuration timing and accuracy summaries. |
| `groupchat/` | Group chat runs: timing, pass@1, and critic catch rate summaries. |
| `load_sweep/` | Results for 1, 4, 16, and 32 concurrent tasks. |
| `figures/` | Plots and tables used in the final report. |

Naming: put the system, configuration, and seed in the file name, for example
`debate_L1_L2_seed0.json`, so that runs are easy to find and compare. Record the seed and the
vLLM version with every result.
