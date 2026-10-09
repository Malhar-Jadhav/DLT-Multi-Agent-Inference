#!/bin/bash
# Serve the large and small model side by side on one H100 and keep them up for the whole job.
# The first argument sets prefix caching: "on" (default) or "off" (use "off" for the baseline).
# Both servers use fixed memory fractions from models/model_config.py in every configuration.
# Submit with: sbatch slurm/serve_models.sbatch off
#
# The partition and GPU type depend on the ICE queue; adjust the --gres line if needed.
#SBATCH --job-name=dlt-serve
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/serve_models_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# source your vLLM environment here, for example: source .venv-serving/bin/activate

LARGE="${LARGE_MODEL:-qwen3-14b}"
SMALL="${SMALL_MODEL:-qwen3-1.7b}"
if [ "${1:-on}" = "off" ]; then CACHE_FLAG="--no-prefix-caching"; else CACHE_FLAG="--prefix-caching"; fi

python -m models.serve_vllm --model "$LARGE" $CACHE_FLAG &
LARGE_PID=$!
# Stagger the start so the two servers do not measure free GPU memory at the same moment.
sleep 120
python -m models.serve_vllm --model "$SMALL" $CACHE_FLAG &
SMALL_PID=$!

echo "Serving $LARGE (pid $LARGE_PID) and $SMALL (pid $SMALL_PID) on $(hostname) with $CACHE_FLAG"
wait $LARGE_PID $SMALL_PID
