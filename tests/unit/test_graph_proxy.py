# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy package-level graph export."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import skillspector
from skillspector.graph_proxy import LazyGraph, graph as lazy_graph


def test_package_graph_export_survives_submodule_load() -> None:
    """Re-import after lazy load still exposes an invokable package export.

    Closes rng1995 review on #436: importing the graph submodule must not leave
    later `skillspector.graph` consumers with a non-invokable module object.
    """
    compiled = SimpleNamespace(invoke=lambda state: state)
    submodule = types.ModuleType("skillspector.graph")
    submodule.graph = compiled  # type: ignore[attr-defined]

    lazy_graph._compiled = None  # noqa: SLF001 — reset lazy singleton for test
    sys.modules["skillspector.graph"] = submodule

    assert lazy_graph.invoke({"ok": True}) == {"ok": True}
    assert isinstance(skillspector.graph, LazyGraph)
    assert skillspector.graph.invoke({"again": 1}) == {"again": 1}
