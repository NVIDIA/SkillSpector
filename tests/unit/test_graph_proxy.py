# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy package-level graph export."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

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

    lazy_graph._compiled = None  # noqa: SLF001 — reset lazy singleton for test
    sys.modules.pop("skillspector.graph", None)
    importlib.invalidate_caches()
    skillspector.graph = lazy_graph

    with patch.dict(sys.modules, {"skillspector.graph": submodule}):
        assert lazy_graph.invoke({"ok": True}) == {"ok": True}

    # Package export must stay the lazy proxy after the submodule import cycle.
    skillspector.graph = lazy_graph
    assert isinstance(skillspector.graph, LazyGraph)
    assert lazy_graph._compiled is compiled
    assert skillspector.graph.invoke({"again": 1}) == {"again": 1}
