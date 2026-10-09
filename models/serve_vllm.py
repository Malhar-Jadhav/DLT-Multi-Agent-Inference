"""Launch a vLLM OpenAI-compatible server for one registered model.

The proposal runs two vLLM instances on one H100 (large and small model) with fixed memory
fractions in every configuration. This script starts one instance; start it twice, once per
model, on different ports (see slurm/serve_models.sbatch). Prefix caching is the L3 lever, so
it is an explicit on/off flag here, and the baseline must use --no-prefix-caching.

Usage:
    python -m models.serve_vllm --model qwen3-14b --no-prefix-caching
    python -m models.serve_vllm --model qwen3-1.7b --prefix-caching --port 8001
    python -m models.serve_vllm --model qwen3-14b --no-prefix-caching --dry-run
"""

import argparse
import subprocess
import sys
from typing import List, Optional

from models.model_config import ModelSpec, get_model


def build_command(
    spec: ModelSpec,
    prefix_caching: bool,
    port: Optional[int] = None,
    gpu_memory_fraction: Optional[float] = None,
    max_model_len: Optional[int] = None,
    seed: int = 0,
) -> List[str]:
    """Build the vLLM command line for a model, applying registry defaults for unset values."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec.hf_id,
        "--served-model-name", spec.key,
        "--port", str(port if port is not None else spec.port),
        "--gpu-memory-utilization", str(gpu_memory_fraction or spec.gpu_memory_fraction),
        "--max-model-len", str(max_model_len or spec.max_model_len),
        "--dtype", "bfloat16",
        "--seed", str(seed),
        # Lets the client read how many prompt tokens were served from the prefix cache.
        "--enable-prompt-tokens-details",
    ]
    cmd.append("--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching")
    return cmd


def main() -> None:
    """Parse arguments, then print or run the server command."""
    parser = argparse.ArgumentParser(description="Start a vLLM server for a registered model.")
    parser.add_argument("--model", required=True, help="Registry key, for example qwen3-14b.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prefix-caching", action="store_true", help="Enable automatic prefix caching (L3 on).")
    group.add_argument("--no-prefix-caching", action="store_true", help="Disable prefix caching (baseline).")
    parser.add_argument("--port", type=int, default=None, help="Override the default port.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=None, help="Override the memory fraction.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override the context window.")
    parser.add_argument("--seed", type=int, default=0, help="Server-side random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    cmd = build_command(
        get_model(args.model), args.prefix_caching, args.port,
        args.gpu_memory_fraction, args.max_model_len, args.seed,
    )
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
