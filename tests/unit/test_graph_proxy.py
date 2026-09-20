# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy package-level graph export."""

from __future__ import annotations

import importlib
import sys

import pytest

import skillspector
from skillspector.graph_proxy import LazyGraph
from skillspector.graph_proxy import graph as lazy_graph


@pytest.fixture
def _graph_import_state(monkeypatch: pytest.MonkeyPatch) -> None:
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
    # drop it so the proxy is pristine (nothing to restore: the pristine
    # singleton carries no such attribute).
    lazy_graph.__dict__.pop("invoke", None)


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
