"""Tests for the model registry, the vLLM launch command, and endpoint parsing.

None of these tests need a GPU, a network connection, or a running server.
"""

import pytest

from models.compare_models import parse_endpoints
from models.model_config import default_pair, get_model, list_models
from models.serve_vllm import build_command


def test_default_pair_roles():
    """The default pair is one large and one small model on different ports."""
    large, small = default_pair()
    assert large.role == "large" and small.role == "small"
    assert large.port != small.port


def test_unknown_model_raises():
    """Asking for an unregistered model gives a KeyError listing the options."""
    with pytest.raises(KeyError):
        get_model("not-a-model")


def test_list_models_filters_by_role():
    """Role filtering returns only models with that role."""
    assert all(m.role == "small" for m in list_models("small"))
    assert len(list_models()) >= 4


def test_prefix_caching_flag_is_explicit():
    """The launch command always states the prefix caching choice, on or off."""
    spec = get_model("qwen3-14b")
    assert "--enable-prefix-caching" in build_command(spec, prefix_caching=True)
    assert "--no-enable-prefix-caching" in build_command(spec, prefix_caching=False)


def test_build_command_uses_registry_defaults_and_overrides():
    """Registry values are used by default and command line overrides win."""
    spec = get_model("qwen3-1.7b")
    cmd = build_command(spec, prefix_caching=False)
    assert cmd[cmd.index("--port") + 1] == str(spec.port)
    cmd = build_command(spec, prefix_caching=False, port=9000)
    assert cmd[cmd.index("--port") + 1] == "9000"


def test_parse_endpoints():
    """key=url pairs become a dict and malformed pairs are rejected."""
    assert parse_endpoints(["a=http://x:1/v1"]) == {"a": "http://x:1/v1"}
    with pytest.raises(ValueError):
        parse_endpoints(["missing-url"])
