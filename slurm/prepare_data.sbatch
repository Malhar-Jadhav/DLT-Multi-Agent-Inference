#!/bin/bash
# Download and preprocess all datasets on a CPU node.
# Submit with: sbatch slurm/prepare_data.sbatch   (add --include-stretch for ChatDev SRDD)
#SBATCH --job-name=dlt-prepare-data
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepare_data_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# source your python environment here, for example: source .venv/bin/activate

python -m data.prepare_all "$@"
