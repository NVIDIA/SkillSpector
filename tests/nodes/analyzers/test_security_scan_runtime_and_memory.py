# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cooperative cancellation and cache-bound tests for the security scan.

Both properties regressed when the scan stopped walking character by character:
seeking jumps over the periodic checkpoints a walk would hit, and memoizing a
derived view retains far more than the entry count suggests because
normalization can expand its input several-fold. These tests pin both.
"""

from __future__ import annotations

import pytest

from skillspector.artifacts import (
    _DERIVED_VIEW_CACHE_BUDGET_CHARS,
    _NORMALIZED_VIEW_CACHE,
    _RUNTIME_CHECKPOINT_STRIDE,
    _token_bridging_gap_spans,
    clear_security_text_caches,
    normalized_security_view,
)


class _AbortError(Exception):
    pass


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
