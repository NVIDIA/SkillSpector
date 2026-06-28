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

"""MCP rug-pull analyzer node (B.3.1 & B.3.3) — RP1 through RP3.

Detects supply-chain rug-pull risks in agent skills:
1. Version-unpinned external references or MCP servers (B.3.1).
2. Manifest changes (privilege expansion, trigger modification, parameter modification) (B.3.3).
"""

from __future__ import annotations

import re

from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

ANALYZER_ID = "mcp_rug_pull"
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATEGORY = "MCP Rug Pull"
_TAGS = ["ASI16"]

# RP1: Unpinned MCP server references in code or manifest
_RP1_NPX_CMD = re.compile(
    r"npx\s+(?:-+\w+\s+)*((?:@?[a-zA-Z][\w.-]*/)?[a-zA-Z][\w.-]*)",
    re.IGNORECASE,
)
_RP1_UVX_CMD = re.compile(
    r"(?:uvx|uv\s+tool\s+run)\s+(?:-+\w+\s+)*([a-zA-Z][\w.-]*)",
    re.IGNORECASE,
)
_RP1_PIP_INSTALL = re.compile(
    r"pip\d?\s+install\s+(?:-+\w+\s+)*([a-zA-Z][\w.-]*)",
    re.IGNORECASE,
)
_RP1_DOCKER_CMD = re.compile(
    r"docker\s+(?:pull|run|create)\s+\S+",
    re.IGNORECASE,
)

_VERSION_PIN_RE = re.compile(r"@[\d.]+\b|==[\d.]+|:[\d.]+|@sha256:")

# RP2: Manifest-permission pre-staging
_PERMISSION_EXPANSION_PATTERNS = [
    (r'"permissions?"\s*:\s*\[[^\]]*\]', 0.60),
    (
        r"(?:add|grant|request|require)\s+(?:new|additional|extra|more)\s+(?:permissions?|tools?|access)",
        0.70,
    ),
]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _find_line(content: str, pos: int) -> int:
    """Return 1-based line number for character position *pos*."""
    return content[:pos].count("\n") + 1


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


# ---------------------------------------------------------------------------
# RP1: Unpinned MCP server references
# ---------------------------------------------------------------------------


def _check_rp1(manifest: dict, file_cache: dict[str, str]) -> list[Finding]:
    """Detect unpinned MCP server command references in skill files."""
    findings: list[Finding] = []

    for file_path, content in file_cache.items():
        # npx without @version
        for m in _RP1_NPX_CMD.finditer(content):
            full_match = m.group(0)
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line_remainder = content[m.end() : line_end]
            if _VERSION_PIN_RE.search(full_match + line_remainder):
                continue
            line_num = _find_line(content, m.start())
            findings.append(
                Finding(
                    rule_id="RP1",
                    message=f"MCP server referenced without pinned version: '{full_match.strip()}'.",
                    severity="MEDIUM",
                    confidence=0.70,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    explanation=(
                        "npx commands without a version suffix (e.g. @1.0.0) "
                        "create a rug-pull risk if the upstream server is "
                        "compromised and publishes a malicious update."
                    ),
                    remediation="Pin the version: npx @scope/server@1.2.3",
                )
            )

        # uvx without ==version
        for m in _RP1_UVX_CMD.finditer(content):
            full_match = m.group(0)
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line_remainder = content[m.end() : line_end]
            if _VERSION_PIN_RE.search(full_match + line_remainder):
                continue
            line_num = _find_line(content, m.start())
            findings.append(
                Finding(
                    rule_id="RP1",
                    message=f"MCP server referenced without pinned version: '{full_match.strip()}'.",
                    severity="MEDIUM",
                    confidence=0.65,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    explanation=(
                        "uvx/uv tool run commands without ==version create a rug-pull risk."
                    ),
                    remediation="Pin the version: uvx package-name==1.2.3",
                )
            )

        # pip install without ==version
        for m in _RP1_PIP_INSTALL.finditer(content):
            full_match = m.group(0)
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line_remainder = content[m.end() : line_end]
            if _VERSION_PIN_RE.search(full_match + line_remainder):
                continue
            pkg = m.group(1)
            if "mcp" not in pkg.lower():
                continue
            line_num = _find_line(content, m.start())
            findings.append(
                Finding(
                    rule_id="RP1",
                    message=f"MCP server dependency without pinned version: '{full_match.strip()}'.",
                    severity="LOW",
                    confidence=0.60,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    explanation=(
                        "pip install without ==version installs the latest "
                        "release, which could include malicious changes."
                    ),
                    remediation="Pin the version: pip install package==1.2.3",
                )
            )

        # docker without tag or digest
        for m in _RP1_DOCKER_CMD.finditer(content):
            full_match = m.group(0)
            if _VERSION_PIN_RE.search(full_match):
                continue
            line_num = _find_line(content, m.start())
            findings.append(
                Finding(
                    rule_id="RP1",
                    message=f"Docker image referenced without tag or digest: '{full_match[:80]}'.",
                    severity="MEDIUM",
                    confidence=0.75,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    explanation=(
                        "Docker image references without a specific tag (:latest "
                        "is implicit) or digest (@sha256:...) can be silently "
                        "replaced by a malicious image."
                    ),
                    remediation="Pin the image: image:tag or image@sha256:abc123",
                )
            )

    # Check manifest for unpinned MCP server references
    manifest_text = str(manifest)
    for m in _RP1_NPX_CMD.finditer(manifest_text):
        findings.append(
            Finding(
                rule_id="RP1",
                message=(
                    f"Manifest references MCP server without version pin: '{m.group(0).strip()}'."
                ),
                severity="MEDIUM",
                confidence=0.70,
                file="SKILL.md",
                start_line=1,
                category=_CATEGORY,
                tags=list(_TAGS),
                matched_text=m.group(0)[:200],
                explanation=(
                    "MCP server references in the skill manifest without version "
                    "pinning are a rug-pull risk."
                ),
                remediation="Always pin MCP server versions in manifest references.",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# RP2: Permission pre-staging
# ---------------------------------------------------------------------------


def _check_rp2(manifest: dict, file_cache: dict[str, str]) -> list[Finding]:
    """Detect manifest permission patterns that suggest pre-staging for future abuse."""
    findings: list[Finding] = []

    manifest_text = str(manifest)
    for pattern, confidence in _PERMISSION_EXPANSION_PATTERNS:
        for m in re.finditer(pattern, manifest_text, re.IGNORECASE):
            findings.append(
                Finding(
                    rule_id="RP2",
                    message="Manifest language suggests future permission expansion.",
                    severity="LOW",
                    confidence=_clamp(confidence),
                    file="SKILL.md",
                    start_line=1,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=m.group(0)[:200],
                    explanation=(
                        "Language in the manifest suggests the skill may request "
                        "additional permissions or tools in future versions. This "
                        "is a pre-staging indicator for rug-pull attacks."
                    ),
                    remediation=(
                        "Review the skill's stated permissions. Consider pinning "
                        "to a specific version and auditing updates."
                    ),
                )
            )

    return findings


# ---------------------------------------------------------------------------
# RP3: Version unpinned
# ---------------------------------------------------------------------------


def _check_rp3(manifest: dict) -> list[Finding]:
    """Detect when skill version is unpinned or uses broad constraints."""
    findings: list[Finding] = []

    version_value = manifest.get("version") if isinstance(manifest, dict) else None
    if not version_value or not isinstance(version_value, str):
        return findings

    version_str = str(version_value).strip()
    if version_str in ("*", "latest", "any"):
        findings.append(
            Finding(
                rule_id="RP3",
                message=f"Skill version is unpinned: '{version_str}'.",
                severity="LOW",
                confidence=0.80,
                file="SKILL.md",
                start_line=1,
                category=_CATEGORY,
                tags=list(_TAGS),
                matched_text=version_str,
                explanation=(
                    "An unpinned version allows automatic updates to any "
                    "future version, creating a rug-pull risk."
                ),
                remediation="Pin to a specific version (e.g. '1.2.3').",
            )
        )
    elif version_str.startswith(">=") or version_str.startswith("^"):
        findings.append(
            Finding(
                rule_id="RP3",
                message=f"Skill version constraint may be too broad: '{version_str}'.",
                severity="LOW",
                confidence=0.40 if version_str.startswith(">=") else 0.50,
                file="SKILL.md",
                start_line=1,
                category=_CATEGORY,
                tags=list(_TAGS),
                matched_text=version_str,
                explanation=(
                    "Broad version constraints allow automatic major-version "
                    "updates, which could silently introduce malicious changes."
                ),
                remediation="Pin to a specific version or narrow the range.",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Analyze skill for rug-pull risks (RP1–RP3)."""
    manifest: dict = state.get("manifest") or {}
    file_cache: dict[str, str] = state.get("file_cache") or {}
    previous_manifest: dict | None = state.get("previous_manifest")

    findings: list[Finding] = []

    # 1. Static unpinned / pre-staging checks (always run if manifest/cache exists)
    if manifest or file_cache:
        rp1_findings = _check_rp1(manifest, file_cache)
        findings.extend(rp1_findings)
        logger.debug("%s: RP1 produced %d static findings", ANALYZER_ID, len(rp1_findings))

        rp2_findings = _check_rp2(manifest, file_cache)
        findings.extend(rp2_findings)
        logger.debug("%s: RP2 produced %d static findings", ANALYZER_ID, len(rp2_findings))

        rp3_findings = _check_rp3(manifest)
        findings.extend(rp3_findings)
        logger.debug("%s: RP3 produced %d static findings", ANALYZER_ID, len(rp3_findings))

    # 2. Manifest comparison checks (if previous_manifest is available)
    if previous_manifest:
        curr_perms = _normalize_string_list(manifest.get("permissions"))
        prev_perms = _normalize_string_list(previous_manifest.get("permissions"))

        # --- RP1: Permission expansion / privilege escalation ---
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
                    tags=["ASI02"],
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

        # --- RP2: Trigger phrase modification ---
        curr_triggers = _normalize_string_list(manifest.get("triggers"))
        prev_triggers = _normalize_string_list(previous_manifest.get("triggers"))
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
                    tags=["ASI02"],
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

        # --- RP3: Parameter schema or default modification ---
        curr_params = _get_parameters_map(manifest.get("parameters"))
        prev_params = _get_parameters_map(previous_manifest.get("parameters"))
        added_params = [name for name in curr_params if name not in prev_params]
        removed_params = [name for name in prev_params if name not in curr_params]
        changed_params = []

        for name in curr_params:
            if name in prev_params:
                curr_prop = curr_params[name]
                prev_prop = prev_params[name]
                prop_diffs = []
                if curr_prop["type"] != prev_prop["type"]:
                    prop_diffs.append(
                        f"type changed from {prev_prop['type']} to {curr_prop['type']}"
                    )
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
                changes.append(
                    f"removed: {', '.join(prev_params[p]['name'] for p in removed_params)}"
                )
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
                    tags=["ASI02"],
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

    logger.info("%s: %d findings in total", ANALYZER_ID, len(findings))
    return {"findings": findings}
