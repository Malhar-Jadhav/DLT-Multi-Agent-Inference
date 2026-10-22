#!/bin/bash
# Serve the large and small model, run the GSM8K debate against the large one, then stop the servers.
# The first argument is the prefix caching mode of the servers: "on" or "off" (baseline: off).
# Everything after it is passed to experiments.run_debate, for example the levers and the load.
# Submit with: sbatch slurm/run_debate.sbatch off --limit 20 --warmup 3
#              sbatch slurm/run_debate.sbatch on --parallel --concurrency 4
# If the job hits the time limit, submit the same line again with --resume added.
#
# The partition and GPU type depend on the ICE queue; adjust the --gres line if needed.
#SBATCH --job-name=dlt-debate
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/debate_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# source your environment here: the vLLM environment for the servers, the client one for the run.
# If they are different environments, start the servers inside a subshell that activates the first.

MODE="${1:?first argument must be on or off}"
shift
case "$MODE" in
  on)  CACHE_FLAG="--prefix-caching" ;;
  off) CACHE_FLAG="--no-prefix-caching" ;;
  *)   echo "first argument must be on or off, got: $MODE" >&2; exit 2 ;;
esac

LARGE="${LARGE_MODEL:-qwen3-14b}"
SMALL="${SMALL_MODEL:-qwen3-1.7b}"

python -m models.serve_vllm --model "$LARGE" $CACHE_FLAG &
LARGE_PID=$!
trap 'kill $LARGE_PID ${SMALL_PID:-} 2>/dev/null || true' EXIT
# Stagger the start so the two servers do not measure free GPU memory at the same moment.
sleep 120
python -m models.serve_vllm --model "$SMALL" $CACHE_FLAG &
SMALL_PID=$!

# Wait up to 30 minutes for both servers to answer, so loading time is not part of any measurement.
wait_ready() {
  for _ in $(seq 1 180); do
    if curl -sf "http://localhost:$1/v1/models" > /dev/null; then return 0; fi
    sleep 10
  done
  echo "server on port $1 did not become ready" >&2
  return 1
}
wait_ready 8000
wait_ready 8001

python -m experiments.run_debate --model "$LARGE" --prefix-caching "$MODE" "$@"
