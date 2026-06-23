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

"""MCP rug-pull analyzer node (B.3.3) — RP1 through RP3."""

from __future__ import annotations

from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

ANALYZER_ID = "mcp_rug_pull"
logger = get_logger(__name__)

_CATEGORY = "MCP Rug Pull"
_TAGS = ["ASI02"]


def _normalize_string_list(lst: list[object] | None) -> list[str]:
    """Strip and lowercase all strings in the list. Returns sorted list of unique values."""
    if not lst:
        return []
    res = set()
    for item in lst:
        if item is not None:
            res.add(str(item).strip().lower())
    return sorted(res)


def _get_parameters_map(parameters: list[object] | None) -> dict[str, dict[str, object]]:
    """Convert parameters list of dicts to a map of lowercase parameter names -> properties."""
    param_map: dict[str, dict[str, object]] = {}
    if not parameters:
        return param_map
    for item in parameters:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is not None:
            name_str = str(name).strip().lower()
            param_map[name_str] = {
                "name": str(name),
                "type": item.get("type"),
                "description": item.get("description"),
                "default": item.get("default"),
            }
    return param_map


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Compare current vs previous manifest; emit RP1–RP3 findings when previous manifest is available."""
    manifest: dict = state.get("manifest") or {}
    previous_manifest: dict | None = state.get("previous_manifest")

    # If previous_manifest is not available or is empty, we cannot compare. Skip rug-pull checks.
    if not previous_manifest:
        logger.info("%s: previous_manifest not available, skipping", ANALYZER_ID)
        return {"findings": []}

    findings: list[Finding] = []

    # Retrieve permissions
    curr_perms = _normalize_string_list(manifest.get("permissions"))
    prev_perms = _normalize_string_list(previous_manifest.get("permissions"))

    # --- RP1: Permission expansion / privilege escalation ---
    # Find any permission in current that is NOT in previous
    added_perms = [p for p in curr_perms if p not in prev_perms]
    if added_perms:
        logger.debug("%s: RP1 permission expansion detected: %s", ANALYZER_ID, added_perms)
        findings.append(
            Finding(
                rule_id="RP1",
                message=(
                    f"Permissions expanded: current manifest requests permissions not present in the "
                    f"previous version (added: {', '.join(added_perms)})."
                ),
                severity="HIGH",
                confidence=0.90,
                file="SKILL.md",
                category=_CATEGORY,
                tags=list(_TAGS),
                explanation=(
                    "A skill version update added new permissions to the manifest. If unexpected, "
                    "this could indicate a privilege escalation or 'rug pull' attack where the skill "
                    "updates to gain unauthorized capabilities."
                ),
                remediation=(
                    "Verify if the newly added permissions are indeed necessary for the skill's purpose. "
                    "If not, downgrade or revert the skill version, or modify the manifest to remove the excess permissions."
                ),
            )
        )

    # Retrieve triggers
    curr_triggers = _normalize_string_list(manifest.get("triggers"))
    prev_triggers = _normalize_string_list(previous_manifest.get("triggers"))

    # --- RP2: Trigger phrase modification ---
    # Emit finding if the triggers set is modified (additions or removals)
    added_triggers = [t for t in curr_triggers if t not in prev_triggers]
    removed_triggers = [t for t in prev_triggers if t not in curr_triggers]
    if added_triggers or removed_triggers:
        changes = []
        if added_triggers:
            changes.append(f"added: {', '.join(added_triggers)}")
        if removed_triggers:
            changes.append(f"removed: {', '.join(removed_triggers)}")
        logger.debug("%s: RP2 trigger modification detected: %s", ANALYZER_ID, changes)
        findings.append(
            Finding(
                rule_id="RP2",
                message=(
                    f"Trigger phrases modified: triggers have changed from the previous version "
                    f"({'; '.join(changes)})."
                ),
                severity="MEDIUM",
                confidence=0.85,
                file="SKILL.md",
                category=_CATEGORY,
                tags=list(_TAGS),
                explanation=(
                    "Trigger phrases determine when the AI agent will execute the skill. Modifying, "
                    "adding, or deleting trigger phrases can hijack the agent's behavior, leading to "
                    "unintended invocation of tools or bypassing safety triggers."
                ),
                remediation=(
                    "Review the modified trigger phrases to ensure they align with the expected behavior "
                    "of the skill and do not lead to accidental or malicious invocation."
                ),
            )
        )

    # Retrieve parameters
    curr_params = _get_parameters_map(manifest.get("parameters"))
    prev_params = _get_parameters_map(previous_manifest.get("parameters"))

    # --- RP3: Parameter schema or default modification ---
    # Detect changes in parameters: additions, removals, or property modifications
    added_params = [name for name in curr_params if name not in prev_params]
    removed_params = [name for name in prev_params if name not in curr_params]
    changed_params = []

    for name in curr_params:
        if name in prev_params:
            curr_prop = curr_params[name]
            prev_prop = prev_params[name]
            prop_diffs = []
            if curr_prop["type"] != prev_prop["type"]:
                prop_diffs.append(f"type changed from {prev_prop['type']} to {curr_prop['type']}")
            if curr_prop["default"] != prev_prop["default"]:
                prop_diffs.append(
                    f"default changed from {prev_prop['default']} to {curr_prop['default']}"
                )
            if curr_prop["description"] != prev_prop["description"]:
                prop_diffs.append("description changed")
            if prop_diffs:
                changed_params.append(f"{curr_prop['name']} ({'; '.join(prop_diffs)})")

    if added_params or removed_params or changed_params:
        changes = []
        if added_params:
            changes.append(f"added: {', '.join(curr_params[p]['name'] for p in added_params)}")
        if removed_params:
            changes.append(f"removed: {', '.join(prev_params[p]['name'] for p in removed_params)}")
        if changed_params:
            changes.append(f"modified: {', '.join(changed_params)}")

        logger.debug("%s: RP3 parameter modification detected: %s", ANALYZER_ID, changes)
        findings.append(
            Finding(
                rule_id="RP3",
                message=(
                    f"Parameter schema modified: parameters were added, removed, or had their attributes changed "
                    f"({'; '.join(changes)})."
                ),
                severity="MEDIUM",
                confidence=0.80,
                file="SKILL.md",
                category=_CATEGORY,
                tags=list(_TAGS),
                explanation=(
                    "Modifying parameter schemas, parameter types, or default values can alter the input flow "
                    "to tools. Specifically, changing a default value to a malicious payload or command execution "
                    "vector can exploit the agent when the tool is invoked."
                ),
                remediation=(
                    "Verify that parameter additions, removals, or schema and default value changes are safe "
                    "and match the expected behavior of the updated skill."
                ),
            )
        )

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
