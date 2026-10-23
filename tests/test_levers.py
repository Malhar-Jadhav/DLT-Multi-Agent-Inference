"""Tests for harness/levers.py — no GPU or network required."""

import asyncio
import pytest

from harness.levers import LeverConfig, dispatch, sequential_gather, parallel_gather


def test_lever_config_from_dict_baseline():
    cfg = LeverConfig.from_dict({"name": "C0"})
    assert cfg.name == "C0"
    assert not cfg.l1_parallel
    assert not cfg.l2_coalesce
    assert not cfg.l3_cache
    assert not cfg.l4_tier


def test_lever_config_from_dict_all_on():
    cfg = LeverConfig.from_dict({
        "name": "C5",
        "l1_parallel": True, "l2_coalesce": True, "l3_cache": True, "l4_tier": True,
    })
    assert cfg.l1_parallel and cfg.l2_coalesce and cfg.l3_cache and cfg.l4_tier


def test_lever_config_to_dict():
    cfg = LeverConfig(name="C1", l1_parallel=True)
    d = cfg.to_dict()
    assert d == {"l1": True, "l2": False, "l3": False, "l4": False}


def test_client_for_role_tiering_off(monkeypatch):
    """When L4 is off, all roles get the large client."""
    cfg = LeverConfig(name="C0")
    large = object()
    small = object()
    assert cfg.client_for_role(large, small, "critic_check") is large
    assert cfg.client_for_role(large, small, "solver") is large


def test_client_for_role_tiering_on(monkeypatch):
    """When L4 is on, low-stakes roles get the small client."""
    cfg = LeverConfig(name="C4", l4_tier=True)
    large = object()
    small = object()
    assert cfg.client_for_role(large, small, "critic_check") is small
    assert cfg.client_for_role(large, small, "solver") is large


def test_dispatch_sequential():
    """dispatch with L1 off runs coroutines in order."""
    order = []

    async def coro(i):
        order.append(i)
        return i

    cfg = LeverConfig(name="C0", l1_parallel=False)
    result = asyncio.run(dispatch(cfg, coro(0), coro(1), coro(2)))
    assert list(result) == [0, 1, 2]
    assert order == [0, 1, 2]


def test_dispatch_parallel():
    """dispatch with L1 on returns all results (order may differ)."""
    async def coro(i):
        return i * 10

    cfg = LeverConfig(name="C1", l1_parallel=True)
    result = asyncio.run(dispatch(cfg, coro(1), coro(2), coro(3)))
    assert sorted(result) == [10, 20, 30]
