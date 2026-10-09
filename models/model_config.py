"""Central registry of the language models used in this project.

Every other script (download, serve, client, comparison) reads model details from here, so
adding or swapping a model is a one-place change. The proposal fixes two roles:

  * large: the model that handles every call in the baseline (Qwen3-14B, with
    Mistral-Nemo-12B-Instruct as the fallback).
  * small: the cheaper model used by the model-tiering lever (Qwen3-1.7B, with
    Phi-4-mini-instruct as the fallback).

Sampling defaults follow the proposal: temperature 0.7, top-p 0.8, thinking mode off.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one model and how it is served.

    key: short name used on every command line and as the served model name.
    hf_id: Hugging Face repository id used for download and by vLLM.
    role: "large" or "small", matching the roles in the proposal.
    port: default port for this model's vLLM server, so both can run on one node.
    gpu_memory_fraction: default fraction of GPU memory given to this server.
    max_model_len: context window requested from vLLM.
    thinking_toggle: True if the chat template accepts enable_thinking (Qwen3 family).
    """

    key: str
    hf_id: str
    role: str
    port: int
    gpu_memory_fraction: float
    max_model_len: int = 8192
    thinking_toggle: bool = False


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling settings shared by every call unless a caller overrides them."""

    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 1024


MODELS: Dict[str, ModelSpec] = {
    "qwen3-14b": ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", "large", 8000, 0.60, thinking_toggle=True),
    "mistral-nemo-12b": ModelSpec(
        "mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407", "large", 8000, 0.60
    ),
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "small", 8001, 0.20, thinking_toggle=True),
    "phi-4-mini": ModelSpec("phi-4-mini", "microsoft/Phi-4-mini-instruct", "small", 8001, 0.20),
}

DEFAULT_LARGE = "qwen3-14b"
DEFAULT_SMALL = "qwen3-1.7b"
DEFAULT_SAMPLING = SamplingConfig()


def get_model(key: str) -> ModelSpec:
    """Return the ModelSpec for a registry key, with a helpful error for unknown keys."""
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Available: {', '.join(sorted(MODELS))}")
    return MODELS[key]


def list_models(role: str = "") -> List[ModelSpec]:
    """List registered models, optionally filtered to the "large" or "small" role."""
    specs = list(MODELS.values())
    return [s for s in specs if not role or s.role == role]


def default_pair() -> Tuple[ModelSpec, ModelSpec]:
    """Return the (large, small) pair used by default in every configuration."""
    return MODELS[DEFAULT_LARGE], MODELS[DEFAULT_SMALL]


def with_overrides(spec: ModelSpec, **changes) -> ModelSpec:
    """Return a copy of a ModelSpec with some fields changed, for example a different port."""
    return replace(spec, **changes)
