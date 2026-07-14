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

"""Structured skill role summary analyzer (SSR-*)."""

from __future__ import annotations

from skillspector.state import AnalyzerNodeResponse, SkillspectorState

ANALYZER_ID = "structured_skill_roles"


def _string_list(value: object) -> list[str]:
    """Return a compact list of string values for summary payload fields."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _build_summary(context: dict[str, object]) -> dict[str, object]:
    """Build a single SSR-1 structured summary from validated context."""
    protocol = str(context.get("protocol", "AISOP/AISP"))
    layout_kind = str(context.get("layout_kind", "structured"))
    bundle_path = str(context.get("bundle_path", ""))
    declared_tools = sorted(_string_list(context.get("declared_tools")))
    workflow_nodes = _string_list(context.get("workflow_nodes"))
    constraints = _string_list(context.get("constraint_anchors"))
    resources = _string_list(context.get("resource_anchors"))

    return {
        "id": "SSR-1",
        "message": f"Structured {layout_kind} bundle detected ({protocol})",
        "file": bundle_path,
        "protocol": protocol,
        "layout_kind": layout_kind,
        "declared_tools": declared_tools,
        "workflow_nodes": workflow_nodes,
        "constraints": constraints,
        "resources": resources,
        "tags": ["AISOP", "AISP", "structured-skill"],
    }


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Emit one SSR-1 structured summary when structured context is present."""
    context = state.get("structured_skill_context")
    if not isinstance(context, dict):
        return {"findings": []}

    return {"findings": [], "structured_summaries": [_build_summary(context)]}
