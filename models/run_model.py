"""Send a single prompt to a served model and print the answer with its timing and token counts.

Use this as a smoke test after starting a server, or to try a prompt by hand.

Usage:
    python -m models.run_model --model qwen3-14b --prompt "Write a haiku about GPUs."
    python -m models.run_model --model qwen3-1.7b --system "Answer briefly." --prompt "Hi" --n 3
"""

import argparse

from models.client import LLMClient


def main() -> None:
    """Parse arguments, run one request, and print the result."""
    parser = argparse.ArgumentParser(description="Query a running vLLM server.")
    parser.add_argument("--model", required=True, help="Registry key of the served model.")
    parser.add_argument("--prompt", required=True, help="User message.")
    parser.add_argument("--system", default=None, help="Optional system message.")
    parser.add_argument("--base-url", default=None, help="Server URL, default localhost on the model's port.")
    parser.add_argument("--n", type=int, default=1, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    client = LLMClient(args.model, base_url=args.base_url)
    result = client.generate(messages, n=args.n, seed=args.seed, max_tokens=args.max_tokens)

    for i, text in enumerate(result.texts):
        print(f"--- sample {i} ---\n{text}")
    ttft = "n/a" if result.first_token_time is None else f"{result.first_token_time - result.submit_time:.3f}s"
    print(
        f"\nlatency {result.latency:.3f}s | first token {ttft} | prompt {result.prompt_tokens} tokens "
        f"({result.cached_tokens} cached) | output {result.output_tokens} tokens"
    )


if __name__ == "__main__":
    main()
