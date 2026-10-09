"""Run the same prompts through several served models and save the outputs side by side.

This is the entry point for quick experiments between models: it reads a JSONL file of
prompts, sends each prompt to every listed model, and writes one JSONL row per (prompt, model)
with the answer, latency, and token counts. Each model must already be served. By default a
model is reached at the port in the registry, and --endpoint overrides that.

Input JSONL rows need a "prompt" field and may have an "id" field.

Usage:
    python -m models.compare_models --prompts prompts.jsonl --models qwen3-14b qwen3-1.7b
    python -m models.compare_models --prompts prompts.jsonl --models qwen3-14b \
        --endpoint qwen3-14b=http://node042:8000/v1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from models.client import LLMClient


def parse_endpoints(pairs: List[str]) -> Dict[str, str]:
    """Turn ["key=url", ...] command line values into a dict of model key to base URL."""
    endpoints = {}
    for pair in pairs:
        key, _, url = pair.partition("=")
        if not url:
            raise ValueError(f"Endpoint '{pair}' must look like key=url")
        endpoints[key] = url
    return endpoints


def read_prompts(path: Path) -> List[dict]:
    """Read prompt rows from a JSONL file, filling in a numeric id when one is missing."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, row in enumerate(rows):
        row.setdefault("id", str(i))
    return rows


def main() -> None:
    """Parse arguments, query every model for every prompt, and write the comparison file."""
    parser = argparse.ArgumentParser(description="Compare several served models on the same prompts.")
    parser.add_argument("--prompts", required=True, help="JSONL file with a prompt field per row.")
    parser.add_argument("--models", nargs="+", required=True, help="Registry keys to compare.")
    parser.add_argument("--endpoint", nargs="*", default=[], help="Optional key=url overrides.")
    parser.add_argument("--output", default="logs/compare.jsonl", help="Where to write results.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    endpoints = parse_endpoints(args.endpoint)
    clients = {m: LLMClient(m, base_url=endpoints.get(m)) for m in args.models}
    prompts = read_prompts(Path(args.prompts))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for row in prompts:
            for model, client in clients.items():
                res = client.generate([{"role": "user", "content": row["prompt"]}], seed=args.seed)
                out.write(json.dumps({
                    "id": row["id"], "model": model, "answer": res.text, "latency_s": round(res.latency, 4),
                    "prompt_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens,
                    "output_tokens": res.output_tokens,
                }) + "\n")
            print(f"done {row['id']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
