#!/bin/bash
# Download the project models into the Hugging Face cache on ICE.
# Models are cached under $HF_HOME, so point it at a project or scratch directory with space.
# Submit with: sbatch slurm/download_models.sbatch   (add --all as an argument for every model)
#SBATCH --job-name=dlt-download-models
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_models_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# source your python environment here, for example: source .venv/bin/activate

python -m models.download_models "$@"
