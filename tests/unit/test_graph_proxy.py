# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy package-level graph export."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Generator

import pytest

import skillspector
from skillspector.graph_proxy import LazyGraph
from skillspector.graph_proxy import graph as lazy_graph


@pytest.fixture
def _graph_import_state(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Snapshot and restore all shared graph import state.

    Closes rng1995/yashrajp22 reviews on #436: graph import-order tests must
    exercise the real factory/import paths without leaking ``sys.modules``
    entries, the package ``graph`` attribute, or ``LazyGraph`` compiled
    state into other tests.
    """
    monkeypatch.delitem(sys.modules, "skillspector.graph", raising=False)
    monkeypatch.delitem(sys.modules, "skillspector.mcp_server", raising=False)
    monkeypatch.setattr(skillspector, "graph", lazy_graph, raising=False)
    monkeypatch.setattr(lazy_graph, "_compiled", None, raising=False)
    # A leaked instance attribute would shadow __getattr__ delegation;
    # drop it so the proxy is pristine, restoring the exact prior value
    # (or its absence) after the test. monkeypatch.delattr cannot cover
    # this: LazyGraph.__getattr__ delegation makes hasattr() true even
    # when the instance dict holds no such attribute, so the instance
    # dict is snapshotted and restored explicitly.
    had_invoke = "invoke" in lazy_graph.__dict__
    prior_invoke = lazy_graph.__dict__.pop("invoke", None)
    try:
        yield
    finally:
        if had_invoke:
            lazy_graph.__dict__["invoke"] = prior_invoke


def _assert_graph_export_invokable() -> None:
    exported = skillspector.graph
    for name in ("invoke", "ainvoke", "stream"):
        assert callable(getattr(exported, name)), (
            f"skillspector.graph lost {name} after import-order change"
        )


def test_create_graph_first_preserves_lazy_export(
    _graph_import_state: None,
) -> None:
    """rng1995 #436: create_graph() as first access keeps the lazy export."""
    skillspector.create_graph()
    assert isinstance(skillspector.graph, LazyGraph)
    _assert_graph_export_invokable()


def test_mcp_import_first_preserves_lazy_export(_graph_import_state: None) -> None:
    """yashrajp22 #436: importing MCP first keeps the lazy export."""
    importlib.import_module("skillspector.mcp_server")
    assert isinstance(skillspector.graph, LazyGraph)
    _assert_graph_export_invokable()


def test_direct_submodule_import_keeps_graph_invokable(
    _graph_import_state: None,
) -> None:
    """yashrajp22 #436: direct submodule import keeps skillspector.graph invokable."""
    importlib.import_module("skillspector.graph")
    _assert_graph_export_invokable()


def test_graph_import_state_restores_preexisting_invoke_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rng1995 #436: isolation must not delete a pre-existing ``invoke`` attribute."""
    sentinel = object()
    lazy_graph.__dict__["invoke"] = sentinel
    try:
        fixture_gen = _graph_import_state.__wrapped__(monkeypatch)
        next(fixture_gen)
        try:
            assert "invoke" not in lazy_graph.__dict__
        finally:
            with pytest.raises(StopIteration):
                next(fixture_gen)
        assert lazy_graph.__dict__["invoke"] is sentinel
    finally:
        lazy_graph.__dict__.pop("invoke", None)
