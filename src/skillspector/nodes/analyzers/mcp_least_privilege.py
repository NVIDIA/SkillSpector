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

"""MCP least-privilege analyzer node (B.3.1) — LP1 through LP4."""

from __future__ import annotations

import re
from pathlib import Path

from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

ANALYZER_ID = "mcp_least_privilege"
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATEGORY = "MCP Least Privilege"
_TAGS = ["ASI02"]

# Wildcard permission values that grant blanket access
_WILDCARD_PERMS = frozenset({"*", "all", "full", "any"})

# Regex patterns per capability category (case-insensitive, applied to file content)
_CAPABILITY_PATTERNS: dict[str, list[str]] = {
    "shell": [
        r"subprocess",
        r"Popen",
        r"os\.system",
        r"os\.popen",
        r"os\.exec",
        r"\bcurl\b",
        r"\bwget\b",
        r"\bchmod\b",
    ],
    "network": [
        r"\bhttpx\b",
        r"\brequests\b",
        r"\burllib\b",
        r"\baiohttp\b",
        r"socket\.connect",
        r"fetch\(",
        r"XMLHttpRequest",
    ],
    "file_read": [
        r"open\s*\([^)]*['\"]r['\"]",
        r"open\s*\([^)]*['\"][^'\"]*r['\"]",
        r"\.read_text\(",
        r"\.read_bytes\(",
        r"os\.listdir",
        r"os\.walk",
        r"glob\.glob",
    ],
    "file_write": [
        r"open\s*\([^)]*['\"][wa]['\"]",
        r"open\s*\([^)]*['\"][^'\"]*[wa]['\"]",
        r"\.write_text\(",
        r"\.write_bytes\(",
        r"shutil\.copy",
        r"os\.rename",
        r"os\.mkdir",
    ],
    "env": [
        r"os\.environ",
        r"os\.getenv",
        r"process\.env",
        r"\bdotenv\b",
    ],
    "mcp": [
        r"create_session",
        r"MCPClient",
        r"mcp\.client",
    ],
}

# Permission string → capability category mapping (case-insensitive word-boundary matching)
_PERM_TO_CAPABILITY: dict[str, str] = {
    "bash": "shell",
    "shell": "shell",
    "terminal": "shell",
    "command": "shell",
    "network": "network",
    "http": "network",
    "fetch": "network",
    "api": "network",
    "read": "file_read",
    "fs_read": "file_read",
    "file_read": "file_read",
    "write": "file_write",
    "fs_write": "file_write",
    "file_write": "file_write",
    "env": "env",
    "environment": "env",
    "mcp": "mcp",
    "tools": "mcp",
    "tool_use": "mcp",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_test_file(path: str) -> bool:
    """Return True if *path* looks like a test file (test_* or *_test.*)."""
    name = Path(path).name
    stem = Path(path).stem
    return name.startswith("test_") or stem.endswith("_test")


def _normalize_allowed_tools(value: object) -> list[str]:
    """Coerce a manifest ``allowed-tools`` value into a list of tool names.

    Accepts the list form (``[Bash, Read]``) and the comma-separated string
    form (``"Bash, Read"``). Anything else yields an empty list.
    """
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    return []


def _detect_capabilities(content: str) -> set[str]:
    """Return set of capability categories found in *content*."""
    found: set[str] = set()
    for cap, patterns in _CAPABILITY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, content, re.IGNORECASE):
                found.add(cap)
                break
    return found


def _map_permissions_to_categories(permissions: list[str]) -> set[str]:
    """Map declared permission strings to capability category names."""
    categories: set[str] = set()
    for perm in permissions:
        perm_lower = perm.lower().strip()
        for keyword, cat in _PERM_TO_CAPABILITY.items():
            # Word-boundary match on the permission string
            if re.search(rf"\b{re.escape(keyword)}\b", perm_lower, re.IGNORECASE):
                categories.add(cat)
                break
    return categories


# Tool name → capability category (Claude / Agent Skills tool names, case-insensitive exact match)
_TOOL_TO_CAPABILITY: dict[str, str] = {
    "bash": "shell",
    "execute": "shell",
    "terminal": "shell",
    "read": "file_read",
    "glob": "file_read",
    "ls": "file_read",
    "write": "file_write",
    "edit": "file_write",
    "multiedit": "file_write",
    "notebookedit": "file_write",
    "webfetch": "network",
    "websearch": "network",
    "fetch": "network",
    "env": "env",
}


def _map_allowed_tools_to_categories(tools: list[str]) -> set[str]:
    """Map Agent Skills ``allowed-tools`` tool names to capability category names."""
    categories: set[str] = set()
    for tool in tools:
        cat = _TOOL_TO_CAPABILITY.get(tool.lower().strip())
        if cat:
            categories.add(cat)
    return categories


def _has_wildcard(permissions: list[str]) -> bool:
    """Return True if any permission value is a wildcard."""
    return any(p.strip().lower() in _WILDCARD_PERMS for p in permissions)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Analyze manifest permissions vs code capabilities; emit LP1-LP4 findings."""
    manifest: dict = state.get("manifest") or {}
    file_cache: dict[str, str] = state.get("file_cache") or {}
    component_metadata: list[dict] = state.get("component_metadata") or []

    # Skip: no manifest
    if not manifest:
        logger.info("%s: no manifest, skipping", ANALYZER_ID)
        return {"findings": []}

    # Skip: docs-only skill (no executable files)
    has_executable = any(m.get("executable", False) for m in component_metadata)
    if not has_executable:
        logger.info("%s: no executable files, skipping", ANALYZER_ID)
        return {"findings": []}

    findings: list[Finding] = []

    # Retrieve declared permissions (may be None if not set in manifest)
    permissions_raw = manifest.get("permissions")  # None | list[str]
    if isinstance(permissions_raw, list):
        permissions: list[str] | None = permissions_raw
    else:
        permissions = None  # treat missing or non-list as None

    # `allowed-tools` (Agent Skills standard) is also a permission declaration.
    allowed_tools = _normalize_allowed_tools(manifest.get("allowed-tools"))

    # --- LP2: Wildcard permission ---
    if isinstance(permissions, list) and _has_wildcard(permissions):
        logger.debug("%s: LP2 wildcard permission detected", ANALYZER_ID)
        findings.append(
            Finding(
                rule_id="LP2",
                message=(
                    "Permission list contains a wildcard entry ('*', 'all', 'full', or 'any'), "
                    "granting blanket access with no least-privilege boundary."
                ),
                severity="MEDIUM",
                confidence=_clamp(0.90),
                file="SKILL.md",
                category=_CATEGORY,
                tags=list(_TAGS),
                explanation=(
                    "Wildcard permissions disable permission-based security controls entirely. "
                    "Specify only the permissions the skill actually requires."
                ),
                remediation=(
                    "Replace '*'/'all'/'full'/'any' with an explicit list of required permissions. "
                    "Request only the minimum access needed."
                ),
            )
        )

    # --- LP3: No permissions declared ---
    # Detect code capabilities first so we can check whether any were found
    executable_paths = [m["path"] for m in component_metadata if m.get("executable", False)]

    # Per-file capabilities: {path: set[cap]}
    file_capabilities: dict[str, set[str]] = {}
    for path in executable_paths:
        content = file_cache.get(path, "")
        caps = _detect_capabilities(content)
        if caps:
            file_capabilities[path] = caps

    # All unique capabilities across all code files
    all_caps: set[str] = set()
    for caps in file_capabilities.values():
        all_caps.update(caps)

    # LP3: no declaration via `permissions` or `allowed-tools`, yet caps detected.
    permissions_absent = (permissions is None or permissions == []) and not allowed_tools
    if permissions_absent and all_caps:
        logger.debug("%s: LP3 no permissions declared but capabilities detected", ANALYZER_ID)
        cap_names = ", ".join(sorted(all_caps))
        findings.append(
            Finding(
                rule_id="LP3",
                message=(
                    f"Skill has no declared permissions but code capabilities were detected: {cap_names}."
                ),
                severity="MEDIUM",
                confidence=_clamp(0.70),
                file="SKILL.md",
                category=_CATEGORY,
                tags=list(_TAGS),
                explanation=(
                    "Without declared permissions the skill's intent is opaque and cannot be validated."
                ),
                remediation=(
                    "Add a 'permissions' field to SKILL.md listing the capabilities this skill requires."
                ),
            )
        )

    wildcard_present = isinstance(permissions, list) and _has_wildcard(permissions)

    # LP1 and LP4 apply when permissions OR allowed-tools is declared
    has_declaration = (isinstance(permissions, list) and permissions) or bool(allowed_tools)
    if has_declaration:
        declared_categories: set[str] = set()
        if isinstance(permissions, list) and permissions:
            declared_categories |= _map_permissions_to_categories(permissions)
        if allowed_tools:
            declared_categories |= _map_allowed_tools_to_categories(allowed_tools)

        # --- LP1: Under-declared capabilities (skip when wildcard present) ---
        if not wildcard_present:
            # Group capabilities by whether they appear only in test files
            cap_in_test_only: set[str] = set()
            cap_in_code: set[str] = set()  # appears in at least one non-test file
            for path, caps in file_capabilities.items():
                if _is_test_file(path):
                    cap_in_test_only.update(caps)
                else:
                    cap_in_code.update(caps)

            # Capabilities in test-only files that are NOT also in non-test files
            test_only_caps = cap_in_test_only - cap_in_code

            for cap in sorted(all_caps):
                if cap in declared_categories:
                    continue
                is_test_only = cap in test_only_caps
                confidence = _clamp(0.55 if is_test_only else 0.75)
                source_files = [p for p, caps in file_capabilities.items() if cap in caps]
                primary_file = source_files[0] if source_files else "SKILL.md"
                logger.debug(
                    "%s: LP1 underdeclared capability %s in %s", ANALYZER_ID, cap, primary_file
                )
                findings.append(
                    Finding(
                        rule_id="LP1",
                        message=(
                            f"Code capability '{cap}' detected in {primary_file} "
                            f"but not covered by declared permissions."
                        ),
                        severity="HIGH",
                        confidence=confidence,
                        file=primary_file,
                        category=_CATEGORY,
                        tags=list(_TAGS),
                        explanation=(
                            f"The skill uses '{cap}' capability that is not listed in its permissions. "
                            "This may indicate deceptive intent or missing permission declarations."
                        ),
                        remediation=(
                            f"Add the '{cap}' permission to SKILL.md, or remove the code that requires it."
                        ),
                    )
                )

        # --- LP4: Over-declared permissions (only when permissions field is set) ---
        for perm in permissions or []:
            perm_lower = perm.strip().lower()
            # Skip wildcard entries themselves
            if perm_lower in _WILDCARD_PERMS:
                continue
            # Find which category this permission maps to
            matched_cat: str | None = None
            for keyword, cat in _PERM_TO_CAPABILITY.items():
                if re.search(rf"\b{re.escape(keyword)}\b", perm_lower, re.IGNORECASE):
                    matched_cat = cat
                    break
            if matched_cat is None:
                continue  # unknown permission, skip
            if matched_cat not in all_caps:
                logger.debug(
                    "%s: LP4 over-declared permission %s (→%s)", ANALYZER_ID, perm, matched_cat
                )
                findings.append(
                    Finding(
                        rule_id="LP4",
                        message=(
                            f"Permission '{perm}' is declared but no corresponding code capability "
                            f"({matched_cat}) was detected."
                        ),
                        severity="LOW",
                        confidence=_clamp(0.65),
                        file="SKILL.md",
                        category=_CATEGORY,
                        tags=list(_TAGS),
                        explanation=(
                            "Declared permissions with no matching code capability may indicate "
                            "removed functionality or pre-staging for future abuse."
                        ),
                        remediation=(
                            f"Remove the '{perm}' permission if the corresponding capability is no longer used."
                        ),
                    )
                )

    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
