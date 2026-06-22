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

"""MCP rug-pull analyzer: detects tool definition changes between manifest versions (RP1-RP3).

When a previous_manifest is available, compares tool/parameter definitions to detect:
- RP1: New parameters added to existing tools (parameter capture)
- RP2: Tool descriptions changed (potential prompt injection via description)
- RP3: Tools removed or renamed (behavior divergence)

When no previous manifest is available, performs static analysis of the current
manifest for rug-pull risk indicators (dynamic loading, overly broad permissions).
"""

from __future__ import annotations

import re

from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

ANALYZER_ID = "mcp_rug_pull"
logger = get_logger(__name__)


def _extract_tools(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    """Extract tool definitions from manifest parameters list.

    Returns dict keyed by tool/parameter name with metadata.
    """
    tools: dict[str, dict[str, object]] = {}
    parameters = manifest.get("parameters", [])
    if not isinstance(parameters, list):
        return tools
    for param in parameters:
        if not isinstance(param, dict):
            continue
        name = param.get("name", "")
        if name:
            tools[name] = param
    return tools


def _compare_manifests(
    current: dict[str, object],
    previous: dict[str, object],
) -> list[Finding]:
    """Compare current vs previous manifest for rug-pull indicators."""
    findings: list[Finding] = []
    current_tools = _extract_tools(current)
    previous_tools = _extract_tools(previous)

    for name, tool_def in current_tools.items():
        if name not in previous_tools:
            findings.append(Finding(
                rule_id="RP1",
                message=f"New parameter '{name}' added since last scan",
                severity="HIGH",
                confidence=0.8,
                file="SKILL.md",
                start_line=1,
                end_line=1,
                remediation=(
                    f"Review the new parameter '{name}' — it may have been added to "
                    "capture additional data from users. Verify its purpose with the "
                    "tool server maintainer."
                ),
                tags=["mcp_rug_pull", "parameter_capture"],
                context=f"Parameter: {name}, Type: {tool_def.get('type', 'unknown')}",
                matched_text=name,
                category="mcp_rug_pull",
                pattern="RP1",
                finding=f"New parameter '{name}' not present in previous version",
                explanation=(
                    "A rug-pull attack adds parameters to capture data that users "
                    "provide to previously-trusted tools."
                ),
                code_snippet=None,
                intent=None,
            ))

    for name in previous_tools:
        if name not in current_tools:
            findings.append(Finding(
                rule_id="RP3",
                message=f"Parameter '{name}' removed since last scan",
                severity="MEDIUM",
                confidence=0.7,
                file="SKILL.md",
                start_line=1,
                end_line=1,
                remediation=(
                    f"Parameter '{name}' was present in the previous version but is now "
                    "missing. This may indicate tool behavior divergence."
                ),
                tags=["mcp_rug_pull", "behavior_divergence"],
                context=f"Removed parameter: {name}",
                matched_text=name,
                category="mcp_rug_pull",
                pattern="RP3",
                finding=f"Parameter '{name}' removed from manifest",
                explanation=(
                    "Removing parameters can indicate the tool server changed its "
                    "interface, potentially redirecting data flow."
                ),
                code_snippet=None,
                intent=None,
            ))

    current_desc = str(current.get("description", ""))
    previous_desc = str(previous.get("description", ""))
    if current_desc and previous_desc and current_desc != previous_desc:
        findings.append(Finding(
            rule_id="RP2",
            message="Skill description changed between versions",
            severity="MEDIUM",
            confidence=0.6,
            file="SKILL.md",
            start_line=1,
            end_line=1,
            remediation=(
                "The skill description changed since the last scan. Review the new "
                "description for prompt injection patterns or misleading instructions."
            ),
            tags=["mcp_rug_pull", "description_change"],
            context=f"Previous: {previous_desc[:100]}\nCurrent: {current_desc[:100]}",
            matched_text=current_desc[:200],
            category="mcp_rug_pull",
            pattern="RP2",
            finding="Description changed between manifest versions",
            explanation=(
                "Tool descriptions are fed to LLMs as context. A rug-pull can inject "
                "malicious instructions via description changes."
            ),
            code_snippet=None,
            intent=None,
        ))

    return findings


_DYNAMIC_LOAD_PATTERNS = [
    re.compile(r"(?:dynamic|runtime)[\s_-]*(?:tool|plugin)[\s_-]*(?:load|import|discover)", re.I),
    re.compile(r"tools_from_(?:url|remote|server)", re.I),
    re.compile(r"fetch[\s_]*tools?\s*\(", re.I),
]


def _static_risk_analysis(
    manifest: dict[str, object],
    file_cache: dict[str, str],
) -> list[Finding]:
    """Analyze current manifest/code for rug-pull risk even without previous version."""
    findings: list[Finding] = []

    permissions = manifest.get("permissions", [])
    if isinstance(permissions, list):
        broad_perms = [p for p in permissions if isinstance(p, str) and p.strip() == "*"]
        if broad_perms:
            findings.append(Finding(
                rule_id="RP1",
                message="Wildcard permission grants unrestricted access",
                severity="HIGH",
                confidence=0.7,
                file="SKILL.md",
                start_line=1,
                end_line=1,
                remediation=(
                    "Wildcard ('*') permissions grant the skill unrestricted access. "
                    "This makes rug-pull attacks more impactful because the tool can "
                    "access any resource. Use specific, scoped permissions."
                ),
                tags=["mcp_rug_pull", "overly_broad"],
                context=f"permissions: {permissions}",
                matched_text="*",
                category="mcp_rug_pull",
                pattern="RP1",
                finding="Wildcard permission in manifest",
                explanation=(
                    "Broad permissions amplify rug-pull impact — a tool that gains "
                    "unrestricted access after a definition change can exfiltrate anything."
                ),
                code_snippet=None,
                intent=None,
            ))

    for path, content in file_cache.items():
        if not content:
            continue
        for pattern in _DYNAMIC_LOAD_PATTERNS:
            match = pattern.search(content)
            if match:
                line_num = content[:match.start()].count("\n") + 1
                findings.append(Finding(
                    rule_id="RP2",
                    message=f"Dynamic tool loading pattern in {path}",
                    severity="MEDIUM",
                    confidence=0.6,
                    file=path,
                    start_line=line_num,
                    end_line=line_num,
                    remediation=(
                        "Dynamic tool loading fetches tool definitions at runtime, "
                        "making the tool set unpredictable between scans. Pin tool "
                        "definitions statically or verify signatures at load time."
                    ),
                    tags=["mcp_rug_pull", "dynamic_loading"],
                    context=content[max(0, match.start() - 50):match.end() + 50],
                    matched_text=match.group(0),
                    category="mcp_rug_pull",
                    pattern="RP2",
                    finding=f"Dynamic tool loading: {match.group(0)}",
                    explanation=(
                        "If tool definitions are fetched dynamically, the tool server "
                        "can change behavior between scans without detection."
                    ),
                    code_snippet=None,
                    intent=None,
                ))
                break

    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Detect MCP rug-pull indicators via manifest comparison and static analysis."""
    manifest = state.get("manifest") or {}
    previous_manifest = state.get("previous_manifest")
    file_cache = state.get("file_cache") or {}
    findings: list[Finding] = []

    if previous_manifest is not None:
        findings.extend(_compare_manifests(manifest, previous_manifest))
    else:
        logger.warning(
            "%s: no previous_manifest available — manifest comparison skipped. "
            "Supply --previous-manifest for full rug-pull detection.",
            ANALYZER_ID,
        )

    findings.extend(_static_risk_analysis(manifest, file_cache))

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
