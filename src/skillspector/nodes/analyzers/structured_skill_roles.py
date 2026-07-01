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

from skillspector.models import Finding
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

ANALYZER_ID = "structured_skill_roles"


def _build_finding(context: dict[str, object]) -> Finding:
    """Build a single SSR-1 finding from validated structured-skill context."""
    protocol = str(context.get("protocol", "AISOP/AISP"))
    layout_kind = str(context.get("layout_kind", "structured"))
    bundle_path = str(context.get("bundle_path", ""))
    tools = context.get("declared_tools") or []
    tools_text = ", ".join(sorted(str(t) for t in tools)) if isinstance(tools, list) else ""
    if not tools_text:
        tools_text = "(no declared tools)"

    workflow_nodes = context.get("workflow_nodes") or []
    workflow_text = ", ".join(str(n) for n in workflow_nodes) if workflow_nodes else "(none)"

    constraints = context.get("constraint_anchors") or []
    constraints_text = ", ".join(str(c) for c in constraints) if constraints else "(none)"

    resources = context.get("resource_anchors") or []
    resources_text = ", ".join(str(r) for r in resources) if resources else "(none)"

    return Finding(
        rule_id="SSR-1",
        message=f"Structured {layout_kind} bundle detected ({protocol})",
        severity="LOW",
        confidence=1.0,
        file=bundle_path,
        tags=["AISOP", "AISP", "structured-skill"],
        context=(
            "Detected structured AISOP/AISP workflow for scan context. "
            f"declared_tools=[{tools_text}], workflow_nodes=[{workflow_text}], "
            f"constraints=[{constraints_text}], resources=[{resources_text}]"
        ),
        matched_text=(
            f"layout_kind={layout_kind}, protocol={protocol}, "
            f"bundle={bundle_path}, declared_tools={tools_text}"
        ),
        explanation=(
            "This scan target appears to define a structured AISOP/AISP workflow. "
            "The detector found a valid two-message AISOP/AISP contract and summarized "
            "workflow roles, declared tool set, constraints, and resource anchors."
        ),
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Emit one LOW SSR-1 summary finding when structured context is present."""
    context = state.get("structured_skill_context")
    if not isinstance(context, dict):
        return {"findings": []}

    return {"findings": [_build_finding(context)]}
