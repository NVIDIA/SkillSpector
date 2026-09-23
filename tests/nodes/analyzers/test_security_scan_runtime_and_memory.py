# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cooperative cancellation and cache-bound tests for the security scan.

Both properties regressed when the scan stopped walking character by character:
seeking jumps over the periodic checkpoints a walk would hit, and memoizing a
derived view retains far more than the entry count suggests because
normalization can expand its input several-fold. These tests pin both.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from skillspector.artifacts import (
    _DERIVED_VIEW_CACHE_BUDGET_CHARS,
    _NORMALIZED_VIEW_CACHE,
    _RUNTIME_CHECKPOINT_STRIDE,
    _SizeBoundedViewCache,
    _token_bridging_gap_spans,
    clear_security_text_caches,
    normalized_security_view,
)


class _AbortError(Exception):
    pass


# Wide enough that the interleaving is reached on the first attempt, short
# enough that both race tests finish in well under a tenth of a second.
_CACHE_RACE_WINDOW_SECONDS = 0.02


@pytest.fixture(autouse=True)
def _clear():
    clear_security_text_caches()
    yield
    clear_security_text_caches()


@pytest.mark.parametrize("length", [0, 4095, 4096, 10_000, 32_768, 100_000])
def test_checkpoint_cadence_matches_a_character_walk(length: int) -> None:
    """Seeking must fire the checks a character-by-character walk would fire."""
    fired = 0

    def check() -> None:
        nonlocal fired
        fired += 1

    list(_token_bridging_gap_spans("a" * length, check_runtime=check))
    assert fired == length // _RUNTIME_CHECKPOINT_STRIDE


def test_cancellation_is_observed_across_sparse_gaps() -> None:
    """In-word gaps that yield no spans must still reach the runtime check."""
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise _AbortError

    with pytest.raises(_AbortError):
        list(_token_bridging_gap_spans("aᅟa " * 8192, check_runtime=check))


def test_cancellation_is_observed_on_text_with_no_candidates() -> None:
    """The whole-string skip must not swallow cancellation either."""
    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise _AbortError

    with pytest.raises(_AbortError):
        list(_token_bridging_gap_spans("a" * 100_000, check_runtime=check))


def test_derived_view_cache_is_bounded_by_size_not_entry_count() -> None:
    """NFKC expands U+FDFA to 18 characters, so entry count bounds nothing."""
    chunk = "ﷺ" * 4000
    for index in range(40):
        normalized_security_view(f"{index} {chunk}")
    assert _NORMALIZED_VIEW_CACHE.stored_chars <= _DERIVED_VIEW_CACHE_BUDGET_CHARS


def test_a_single_oversized_view_is_not_retained() -> None:
    oversized = "ﷺ" * (_DERIVED_VIEW_CACHE_BUDGET_CHARS // 10)
    normalized_security_view(oversized)
    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0


def test_clearing_releases_everything() -> None:
    normalized_security_view("ﷺ" * 1000)
    assert _NORMALIZED_VIEW_CACHE.stored_chars > 0
    clear_security_text_caches()
    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0


def test_scan_teardown_clears_the_caches() -> None:
    """cleanup_result is the hook that releases scan-local state."""
    from skillspector.cleanup import cleanup_result

    normalized_security_view("ﷺ" * 1000)
    assert _NORMALIZED_VIEW_CACHE.stored_chars > 0
    cleanup_result({})
    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0


def test_eviction_does_not_change_results() -> None:
    """A view evicted and rebuilt must be identical to the first one."""
    text = "ﷺ" * 500
    first = normalized_security_view(text)
    for index in range(60):
        normalized_security_view(f"{index} " + "ﷺ" * 4000)
    rebuilt = normalized_security_view(text)
    assert rebuilt.text == first.text
    assert list(rebuilt.source_offsets or []) == list(first.source_offsets or [])


class _SlowLookupEntries(OrderedDict):
    """Widen the window between a lookup and the ``move_to_end`` that follows it."""

    def get(self, key, default=None):  # type: ignore[no-untyped-def]
        value = super().get(key, default)
        time.sleep(_CACHE_RACE_WINDOW_SECONDS)
        return value


class _SlowSizes(dict):
    """Widen the window between recording an entry and recording its size."""

    def __setitem__(self, key, value):  # type: ignore[no-untyped-def]
        time.sleep(_CACHE_RACE_WINDOW_SECONDS)
        super().__setitem__(key, value)


def _cache_accounting_holds(cache: _SizeBoundedViewCache) -> bool:
    """The reported total must match the views actually retained."""
    return cache.stored_chars == sum(len(view.text) for view in cache._entries.values())


def test_a_clear_cannot_land_inside_a_cache_lookup() -> None:
    """LangGraph runs the analyzers as threads, so teardown races every lookup.

    ``get`` looks a key up and then marks it most-recent. Unsynchronized, a
    clear arriving between those two steps leaves ``move_to_end`` with a key
    that is gone, and the ``KeyError`` escapes into whichever analyzer was
    reading. The slow lookup here only widens that window; it does not create
    it.
    """
    cache = _SizeBoundedViewCache(_DERIVED_VIEW_CACHE_BUDGET_CHARS)
    view = normalized_security_view("a view every analyzer asks for")
    cache._entries = _SlowLookupEntries()
    errors: list[BaseException] = []
    stop = threading.Event()
    start = threading.Barrier(2)

    def read_repeatedly() -> None:
        try:
            start.wait()
            for _ in range(8):
                cache.get("key")
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)
        finally:
            stop.set()

    def store_and_clear_repeatedly() -> None:
        try:
            start.wait()
            while not stop.is_set():
                cache.store("key", view, len(view.text))
                cache.clear()
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in (pool.submit(read_repeatedly), pool.submit(store_and_clear_repeatedly)):
            future.result()

    assert errors == []


def test_a_clear_cannot_corrupt_the_stored_character_total() -> None:
    """A clear landing inside a store must not leave the budget over-counted.

    Unsynchronized, the clear empties the cache after the entry is recorded but
    before its size is, so the store adds to a total the clear already zeroed:
    the cache then reports characters it is not holding, and keeps doing so for
    the life of the process.
    """
    cache = _SizeBoundedViewCache(_DERIVED_VIEW_CACHE_BUDGET_CHARS)
    view = normalized_security_view("a view stored while teardown runs")
    cache._sizes = _SlowSizes()
    start = threading.Barrier(2)

    def store() -> None:
        start.wait()
        cache.store("key", view, len(view.text))

    def clear_midway() -> None:
        start.wait()
        time.sleep(_CACHE_RACE_WINDOW_SECONDS / 2)
        cache.clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in (pool.submit(store), pool.submit(clear_midway)):
            future.result()

    assert _cache_accounting_holds(cache)


def test_cleanup_releases_the_caches_without_a_result() -> None:
    """A scan that produced no result still has caches to release."""
    from skillspector.cleanup import cleanup_result

    normalized_security_view("ﷺ" * 1000)
    assert _NORMALIZED_VIEW_CACHE.stored_chars > 0
    cleanup_result(None)
    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0


async def test_an_mcp_scan_that_raises_still_releases_the_caches(monkeypatch) -> None:
    """Teardown that only runs on a result never runs for a failed scan.

    A long-lived MCP server would keep the derived views, and the scanned text
    itself as a predicate-cache key, until the next scan that happens to end
    successfully.
    """
    from skillspector import mcp_server

    async def explode(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("graph failed")

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    monkeypatch.setattr(mcp_server, "graph", SimpleNamespace(ainvoke=explode))
    normalized_security_view("ﷺ" * 1000)
    assert _NORMALIZED_VIEW_CACHE.stored_chars > 0

    with pytest.raises(RuntimeError):
        await mcp_server.run_scan("fixture", use_llm=False, output_format="json")

    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0


async def test_a_cancelled_mcp_scan_still_releases_the_caches(monkeypatch) -> None:
    """Cancellation is the other way a scan ends without a result."""
    from skillspector import mcp_server

    async def cancel(*args: object, **kwargs: object) -> dict[str, object]:
        raise asyncio.CancelledError

    monkeypatch.setattr(mcp_server, "is_llm_available", lambda: (False, "no llm"))
    monkeypatch.setattr(mcp_server, "graph", SimpleNamespace(ainvoke=cancel))
    normalized_security_view("ﷺ" * 1000)
    assert _NORMALIZED_VIEW_CACHE.stored_chars > 0

    with pytest.raises(asyncio.CancelledError):
        await mcp_server.run_scan("fixture", use_llm=False, output_format="json")

    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0


def test_a_cli_scan_that_raises_still_releases_the_caches(monkeypatch, tmp_path) -> None:
    """The CLI reports the failure and exits 2; the caches go with it."""
    from typer.testing import CliRunner

    from skillspector import cli

    def explode(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("graph failed")

    monkeypatch.setattr(cli, "graph", SimpleNamespace(invoke=explode, stream=explode))
    normalized_security_view("ﷺ" * 1000)
    assert _NORMALIZED_VIEW_CACHE.stored_chars > 0

    result = CliRunner().invoke(cli.app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])

    assert result.exit_code == 2
    assert _NORMALIZED_VIEW_CACHE.stored_chars == 0
