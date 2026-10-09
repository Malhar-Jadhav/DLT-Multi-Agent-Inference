# models

Everything needed to download, serve, and call the language models used in the project.

| File | What it does |
|---|---|
| `model_config.py` | Registry of models (key, Hugging Face id, role, port, memory fraction) and the default sampling settings. Add or change a model here. |
| `download_models.py` | Downloads model weights from the Hugging Face Hub into the local cache. |
| `serve_vllm.py` | Starts a vLLM OpenAI-compatible server for one model, with prefix caching set explicitly on or off. |
| `client.py` | `LLMClient` for sync and async calls. Returns a `CallResult` with the text, timing, and token counts. |
| `run_model.py` | Sends one prompt to a served model. Use it as a smoke test. |
| `compare_models.py` | Runs the same prompts through several served models and writes the results to a JSONL file. |

Typical order: `download_models`, then `serve_vllm`, then call the model through `client.py`
from experiment code. Run everything from the repository root, for example
`python -m models.run_model --model qwen3-14b --prompt "Hello"`.
