#!/bin/bash
#SBATCH --job-name=dlt-experiment
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-h100   # adjust to the ICE queue name on your account

# ---------------------------------------------------------------------------
# Usage:
#   sbatch slurm/run_experiment.sbatch \
#          configs/ablation_debate.yaml C1 0 results/runs/debate_C1_seed0
#
# Positional args:  CONFIG  CONFIG_NAME  SEED  RUN_DIR
# ---------------------------------------------------------------------------

set -euo pipefail

CONFIG=${1:?usage: sbatch run_experiment.sbatch CONFIG CONFIG_NAME SEED RUN_DIR}
CONFIG_NAME=${2:?}
SEED=${3:?}
RUN_DIR=${4:?}

echo "=== DLT experiment ==="
echo "CONFIG:      $CONFIG"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "SEED:        $SEED"
echo "RUN_DIR:     $RUN_DIR"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "NODE:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Activate the project virtual environment (uv-managed)
source .venv/bin/activate

# Ensure the vLLM servers are running.
# The serve script is expected to already be running in a separate job or
# persistent allocation. Check both ports are reachable before proceeding.
python -c "
from models.client import LLMClient
import sys
for m in ['qwen3-14b', 'qwen3-1.7b']:
    c = LLMClient(m)
    if not c.is_ready():
        print(f'ERROR: {m} server not ready', flush=True)
        sys.exit(1)
    print(f'OK: {m} ready')
"

# Run the experiment
python -m harness.runner \
    --config "$CONFIG" \
    --config-name "$CONFIG_NAME" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR"

echo "=== Done: $RUN_DIR ==="
