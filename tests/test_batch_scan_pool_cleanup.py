# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for pooled model slot cleanup without provider calls."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from contrib.batch_scan.api_pool import ApiKey, ApiKeyPool, PooledChatModel


def _model() -> PooledChatModel:
    return PooledChatModel(
        ApiKeyPool([ApiKey(key="test", model="test", base_url=None, max_concurrent=1)])
    )


@pytest.mark.parametrize("async_call", [False, True])
async def test_model_construction_failure_releases_slot(monkeypatch, async_call: bool) -> None:
    model = _model()
    monkeypatch.setattr(model, "_build_llm", Mock(side_effect=ValueError("invalid client")))
    with pytest.raises(ValueError, match="invalid client"):
        if async_call:
            await model.ainvoke("test")
        else:
            model.invoke("test")
    assert model._pool.active_requests == 0
    key = model._pool.acquire(timeout=0.05)
    model._pool.release(key)


async def test_cancelling_inflight_call_releases_slot(monkeypatch) -> None:
    model = _model()
    started = asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(model, "_build_llm", Mock(return_value=Mock(ainvoke=blocked)))
    task = asyncio.create_task(model.ainvoke("test"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model._pool.active_requests == 0


async def test_cancelling_waiter_does_not_reserve_a_slot_later(monkeypatch) -> None:
    model = _model()
    held = model._pool.acquire()
    tried = asyncio.Event()
    original_try = model._pool.try_acquire

    def try_acquire():
        tried.set()
        return original_try()

    monkeypatch.setattr(model._pool, "try_acquire", try_acquire)
    # The old thread-based acquire can outlive cancellation. Bound it so the
    # regression test cleans up its executor even when run against old code.
    original_acquire = model._pool.acquire
    monkeypatch.setattr(model._pool, "acquire", lambda: original_acquire(timeout=0.1))
    task = asyncio.create_task(model.ainvoke("test"))
    await asyncio.wait_for(tried.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    model._pool.release(held)
    await asyncio.sleep(0.15)
    assert model._pool.active_requests == 0


async def test_async_waiter_runs_after_capacity_is_released(monkeypatch) -> None:
    model = _model()
    held = model._pool.acquire()
    invoke = AsyncMock(return_value="result")
    monkeypatch.setattr(model, "_build_llm", Mock(return_value=Mock(ainvoke=invoke)))
    task = asyncio.create_task(model.ainvoke("test"))
    await asyncio.sleep(0)
    assert not task.done()
    model._pool.release(held)
    assert await asyncio.wait_for(task, timeout=1) == "result"
    assert model._pool.active_requests == 0


@pytest.mark.parametrize("async_call", [False, True])
async def test_rate_limit_retry_releases_each_slot_once(monkeypatch, async_call: bool) -> None:
    pool = ApiKeyPool(
        [ApiKey(key=key, model="test", base_url=None, max_concurrent=1) for key in ("one", "two")]
    )
    model = PooledChatModel(pool, max_retries=1)
    release = Mock(wraps=pool.release)
    monkeypatch.setattr(pool, "release", release)
    invoke = AsyncMock if async_call else Mock
    clients = [Mock(), Mock()]
    for client, outcome in zip(clients, [RuntimeError("429 rate limit"), "result"], strict=True):
        setattr(client, "ainvoke" if async_call else "invoke", invoke(side_effect=[outcome]))
    monkeypatch.setattr(model, "_build_llm", Mock(side_effect=clients))
    result = await model.ainvoke("test") if async_call else model.invoke("test")
    assert result == "result"
    assert [call.kwargs["success"] for call in release.call_args_list] == [False, True]
    assert pool.active_requests == 0
    assert pool.rate_limits_hit == 1
    assert pool.retry_successes == 1
