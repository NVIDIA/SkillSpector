# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LangGraph workflow wiring Skillspector's analyzer nodes."""

# Roadmap: analyzer auto-discovery and stage-as-category ordering (meta_analyzer last); respect
# requires_api_key / is_available() so analyzers are skipped or warned when unavailable.
# Roadmap: optional `skillspector serve` HTTP API (POST /scan, GET /results/{id}, GET /health).

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from skillspector.nodes.analyzers import ANALYZER_NODE_IDS, ANALYZER_NODES
from skillspector.nodes.build_context import build_context
from skillspector.nodes.meta_analyzer import meta_analyzer
from skillspector.nodes.report import report
from skillspector.nodes.resolve_input import resolve_input
from skillspector.state import SkillspectorState


def create_graph():
    """Create and compile Skillspector workflow graph."""
    workflow = StateGraph(SkillspectorState)

    workflow.add_node("resolve_input", resolve_input)
    workflow.add_node("build_context", build_context)
    workflow.add_node("meta_analyzer", meta_analyzer)
    workflow.add_node("report", report)

    for analyzer_id in ANALYZER_NODE_IDS:
        workflow.add_node(analyzer_id, ANALYZER_NODES[analyzer_id])

    workflow.add_edge(START, "resolve_input")
    workflow.add_edge("resolve_input", "build_context")
    for analyzer_id in ANALYZER_NODE_IDS:
        workflow.add_edge("build_context", analyzer_id)
        workflow.add_edge(analyzer_id, "meta_analyzer")
    workflow.add_edge("meta_analyzer", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


graph = create_graph()
