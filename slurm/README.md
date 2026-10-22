# slurm

Batch scripts for the ICE cluster. Submit from the repository root so that relative paths work.

| File | What it does | Example |
|---|---|---|
| `download_models.sbatch` | Downloads model weights into the Hugging Face cache. | `sbatch slurm/download_models.sbatch` |
| `prepare_data.sbatch` | Downloads and preprocesses the datasets. | `sbatch slurm/prepare_data.sbatch` |
| `serve_models.sbatch` | Serves the large and small model together on one H100. The argument is `on` or `off` for prefix caching. | `sbatch slurm/serve_models.sbatch off` |
| `run_debate.sbatch` | Serves both models, waits until they answer, runs the GSM8K debate, stops the servers. First argument `on` or `off`, the rest goes to `experiments.run_debate`. | `sbatch slurm/run_debate.sbatch off --limit 20 --warmup 3` |

Activate your Python environment where the scripts say so before submitting. Job logs go to
`logs/`. The GPU line in `serve_models.sbatch` may need adjusting to the ICE queue.
