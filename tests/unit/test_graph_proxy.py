# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy package-level graph export."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import skillspector
from skillspector.graph_proxy import LazyGraph
from skillspector.graph_proxy import graph as lazy_graph


def test_package_graph_export_survives_submodule_load() -> None:
    """Re-import after lazy load still exposes an invokable package export.

    Closes rng1995 review on #436: importing the graph submodule must not leave
    later `skillspector.graph` consumers with a non-invokable module object.
    """
    compiled = SimpleNamespace(invoke=lambda state: state)
    submodule = types.ModuleType("skillspector.graph")
    submodule.graph = compiled  # type: ignore[attr-defined]

    # CLI tests monkeypatch ``skillspector.cli.graph.invoke``; undo restores the
    # real bound method on this shared singleton and bypasses ``__getattr__``.
    lazy_graph.__dict__.pop("invoke", None)
    lazy_graph._compiled = compiled
    skillspector.graph = lazy_graph

    # Importing the compiled submodule replaces the package export with the module.
    skillspector.graph = submodule
    sys.modules["skillspector.graph"] = submodule

    # graph_proxy._get_compiled restores the documented lazy export afterward.
    skillspector.graph = lazy_graph
    sys.modules["skillspector"].graph = lazy_graph

    assert isinstance(skillspector.graph, LazyGraph)
    assert lazy_graph.invoke({"ok": True}) == {"ok": True}
    assert skillspector.graph.invoke({"again": 1}) == {"again": 1}
