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

"""State schema for the Skillspector LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from skillspector.models import Finding


class SkillspectorState(TypedDict, total=False):
    """Graph state shared by all nodes."""

    # Input: resolve_input node consumes input_path or skill_path, sets skill_path
    input_path: str | None
    skill_path: str | None
    # Set by resolve_input when a temp dir was created (git/url/zip/file); caller should clean up
    temp_dir_for_cleanup: str | None
    zip_bytes: bytes | None
    mode: str

    # build_context node populates these
    components: list[str]
    file_cache: dict[str, str]
    ast_cache: dict[str, str]
    manifest: dict[str, object]
    previous_manifest: dict[str, object] | None

    # Accumulated findings (reducer: analyzer nodes append to this list)
    findings: Annotated[list[Finding], operator.add]
    filtered_findings: list[Finding]

    # Baseline / false-positive suppression. `baseline` is a loaded
    # skillspector.suppression.Baseline (set by CLI/API); the report node drops
    # matching findings before scoring. `show_suppressed` keeps them in the
    # report (marked) for review; `suppressed_findings` is the report output.
    baseline: object | None
    show_suppressed: bool
    suppressed_findings: list[object]

    # Model IDs per LLM-using node: e.g. {"default": "...", "meta_analyzer": "..."}
    model_config: dict[str, str]

    # Component metadata for reporting and risk scoring (from build_context)
    component_metadata: list[dict[str, object]]
    has_executable_scripts: bool

    # Output: report node writes formatted string here
    output_format: str
    report_body: str

    # LLM: when False, LLM-based nodes (meta_analyzer, mcp_tool_poisoning's TP4,
    # and the semantic_* analyzers) return immediately without calling the LLM.
    # Each such node checks use_llm itself; there is no graph-level routing.
    use_llm: bool

    # Risk: report node sets these from risk_score
    risk_severity: str
    risk_recommendation: str

    sarif_report: dict[str, object]
    risk_score: int

    # Additional YARA rules directory (user-specified via --yara-rules-dir)
    yara_rules_dir: str | None


class AnalyzerNodeResponse(TypedDict):
    """Strict analyzer update payload for graph state."""

    findings: list[Finding]


class MetaAnalyzerResponse(TypedDict):
    """Strict meta-analyzer update payload for graph state."""

    filtered_findings: list[Finding]
