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
import time
from dataclasses import dataclass, field
from pathlib import Path

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_event,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import Finding
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    transitive_remaining_seconds,
)

from .static_runner import MAX_FINDINGS_PER_ANALYZER, MAX_FINDINGS_PER_ARTIFACT

ANALYZER_ID = "mcp_least_privilege"
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATEGORY = "MCP Least Privilege"
_TAGS = ["ASI02"]
_CAPABILITY_WINDOW_CHARS = 64 * 1024
_CAPABILITY_WINDOW_OVERLAP_CHARS = 4096
_MAX_CAPABILITY_MATCH_CHARS = 4096
_MAX_FILE_MODE_CHARS = 64
_MAX_EVIDENCE_CHARS = 512
_MAX_DECLARATION_VALUES = MAX_FINDINGS_PER_ANALYZER * 2

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
        rf"open\s*\([^)]{{0,{_MAX_CAPABILITY_MATCH_CHARS}}}['\"]r['\"]",
        rf"open\s*\([^)]{{0,{_MAX_CAPABILITY_MATCH_CHARS}}}"
        rf"['\"][^'\"]{{0,{_MAX_FILE_MODE_CHARS}}}r['\"]",
        r"\.read_text\(",
        r"\.read_bytes\(",
        r"os\.listdir",
        r"os\.walk",
        r"glob\.glob",
    ],
    "file_write": [
        rf"open\s*\([^)]{{0,{_MAX_CAPABILITY_MATCH_CHARS}}}['\"][wa]['\"]",
        rf"open\s*\([^)]{{0,{_MAX_CAPABILITY_MATCH_CHARS}}}"
        rf"['\"][^'\"]{{0,{_MAX_FILE_MODE_CHARS}}}[wa]['\"]",
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
_COMPILED_CAPABILITY_PATTERNS = {
    capability: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for capability, patterns in _CAPABILITY_PATTERNS.items()
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
_COMPILED_PERMISSION_PATTERNS = tuple(
    (
        capability,
        re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE),
    )
    for keyword, capability in _PERM_TO_CAPABILITY.items()
)


class _LeastPrivilegeResourceLimitError(RuntimeError):
    """Stop attacker-controlled work while retaining a bounded prefix."""

    def __init__(
        self,
        reason: LedgerReason,
        metrics: dict[str, int | float],
        *,
        path: str | None = None,
    ) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics
        self.path = path


@dataclass
class _LeastPrivilegeBudget:
    """Enforce the shared runtime and per-scope construction ceilings."""

    state: SkillspectorState
    started_at: float = field(default_factory=time.monotonic)
    initial_allowance: float | None = None
    findings: list[Finding] = field(default_factory=list)
    artifact_findings: dict[str, int] = field(default_factory=dict)
    completed_paths: set[str] = field(default_factory=set)
    current_path: str = "SKILL.md"

    def check_runtime(self, path: str | None = None) -> None:
        if path is not None:
            self.current_path = path
        remaining = transitive_remaining_seconds(self.state)
        if remaining is None:
            return
        if self.initial_allowance is None:
            self.initial_allowance = max(0.0, remaining)
        if remaining <= 0:
            raise _LeastPrivilegeResourceLimitError(
                LedgerReason.RUNTIME_LIMIT,
                {
                    "observed_seconds": max(0.0, time.monotonic() - self.started_at),
                    "limit_seconds": self.initial_allowance,
                },
            )

    def emit(self, finding: Finding) -> None:
        """Append one finding after checking runtime and both finding limits."""
        self.check_runtime()
        artifact_observed = self.artifact_findings.get(finding.file, 0) + 1
        analyzer_observed = len(self.findings) + 1
        if artifact_observed > MAX_FINDINGS_PER_ARTIFACT:
            raise _LeastPrivilegeResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": artifact_observed,
                    "limit_findings": MAX_FINDINGS_PER_ARTIFACT,
                },
                path=finding.file,
            )
        if analyzer_observed > MAX_FINDINGS_PER_ANALYZER:
            raise _LeastPrivilegeResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": analyzer_observed,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
                path=finding.file,
            )
        self.findings.append(finding)
        self.artifact_findings[finding.file] = artifact_observed


def _bounded_evidence(value: object) -> str:
    """Return a fixed-size display projection of one untrusted declaration."""
    text = str(value)
    if len(text) <= _MAX_EVIDENCE_CHARS:
        return text
    return f"{text[: _MAX_EVIDENCE_CHARS - 1]}…"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_test_file(path: str) -> bool:
    """Return True if *path* looks like a test file (test_* or *_test.*)."""
    name = Path(path).name
    stem = Path(path).stem
    return name.startswith("test_") or stem.endswith("_test")


def _normalize_allowed_tools(
    value: object,
    budget: _LeastPrivilegeBudget | None = None,
) -> list[str]:
    """Coerce a manifest ``allowed-tools`` value into a list of tool names.

    Accepts the list form (``[Bash, Read]``) and the comma-separated string
    form (``"Bash, Read"``). Anything else yields an empty list.
    """
    tools: list[str] = []
    if isinstance(value, list):
        candidates = iter(value)
    elif isinstance(value, str):
        # A bounded split prevents a comma-dense declaration from creating an
        # arbitrarily large temporary list before the analyzer can stop it.
        candidates = iter(value.split(",", _MAX_DECLARATION_VALUES))
    else:
        return tools

    for index, tool in enumerate(candidates, start=1):
        if budget is not None:
            budget.check_runtime("SKILL.md")
        if index > _MAX_DECLARATION_VALUES:
            raise _LeastPrivilegeResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_records": index,
                    "limit_records": _MAX_DECLARATION_VALUES,
                },
            )
        normalized = str(tool).strip()
        if normalized:
            tools.append(normalized)
    return tools


def _detect_capabilities(
    content: str,
    budget: _LeastPrivilegeBudget | None = None,
    path: str = "SKILL.md",
) -> set[str]:
    """Return capabilities using bounded windows and cooperative checks."""
    found: set[str] = set()
    step = _CAPABILITY_WINDOW_CHARS - _CAPABILITY_WINDOW_OVERLAP_CHARS
    for start in range(0, max(1, len(content)), step):
        if budget is not None:
            budget.check_runtime(path)
        window = content[start : start + _CAPABILITY_WINDOW_CHARS]
        for capability, patterns in _COMPILED_CAPABILITY_PATTERNS.items():
            if capability in found:
                continue
            for pattern in patterns:
                if budget is not None:
                    budget.check_runtime(path)
                if pattern.search(window) is not None:
                    found.add(capability)
                    break
        if start + _CAPABILITY_WINDOW_CHARS >= len(content):
            break
    return found


def _permission_category(
    permission: str,
    budget: _LeastPrivilegeBudget | None = None,
) -> str | None:
    """Map one value without retaining a potentially large lowercase copy."""
    step = _CAPABILITY_WINDOW_CHARS - _CAPABILITY_WINDOW_OVERLAP_CHARS
    # Keep the historical keyword precedence while bounding each regex input.
    for capability, pattern in _COMPILED_PERMISSION_PATTERNS:
        for start in range(0, max(1, len(permission)), step):
            if budget is not None:
                budget.check_runtime("SKILL.md")
            window = permission[start : start + _CAPABILITY_WINDOW_CHARS]
            if pattern.search(window) is not None:
                return capability
            if start + _CAPABILITY_WINDOW_CHARS >= len(permission):
                break
    return None


def _map_permissions_to_categories(
    permissions: list[str],
    budget: _LeastPrivilegeBudget | None = None,
) -> set[str]:
    """Map declared permission strings to capability category names."""
    categories: set[str] = set()
    for index, permission in enumerate(permissions, start=1):
        if budget is not None:
            budget.check_runtime("SKILL.md")
        if index > _MAX_DECLARATION_VALUES:
            raise _LeastPrivilegeResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_records": index,
                    "limit_records": _MAX_DECLARATION_VALUES,
                },
            )
        category = _permission_category(str(permission), budget)
        if category is not None:
            categories.add(category)
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


def _map_allowed_tools_to_categories(
    tools: list[str],
    budget: _LeastPrivilegeBudget | None = None,
) -> set[str]:
    """Map Agent Skills ``allowed-tools`` tool names to capability category names."""
    categories: set[str] = set()
    for tool in tools:
        if budget is not None:
            budget.check_runtime("SKILL.md")
        if len(tool) > 64:
            continue
        cat = _TOOL_TO_CAPABILITY.get(tool.lower().strip())
        if cat:
            categories.add(cat)
    return categories


def _has_wildcard(
    permissions: list[str],
    budget: _LeastPrivilegeBudget | None = None,
) -> bool:
    """Return True if any permission value is a wildcard."""
    for index, permission in enumerate(permissions, start=1):
        if budget is not None:
            budget.check_runtime("SKILL.md")
        if index > _MAX_DECLARATION_VALUES:
            raise _LeastPrivilegeResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_records": index,
                    "limit_records": _MAX_DECLARATION_VALUES,
                },
            )
        # Values longer than the longest wildcard cannot be exact matches, so
        # avoid copying/case-folding attacker-controlled large strings.
        value = str(permission)
        if len(value) <= 16 and value.strip().lower() in _WILDCARD_PERMS:
            return True
    return False


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _partial_limit_event(
    path: str,
    limit: _LeastPrivilegeResourceLimitError,
    emitted_finding_ids: list[str] | None = None,
) -> InspectionLedgerEvent:
    """Account one current or omitted scope using canonical bounded metrics."""
    return ledger_event(
        analyzer_id=ANALYZER_ID,
        outcome=LedgerOutcome.PARTIAL,
        phase="static",
        path=path,
        reason=limit.reason,
        emitted_finding_ids=emitted_finding_ids or (),
        observed_findings=(
            int(limit.metrics["observed_findings"])
            if "observed_findings" in limit.metrics
            else None
        ),
        limit_findings=(
            int(limit.metrics["limit_findings"]) if "limit_findings" in limit.metrics else None
        ),
        observed_records=(
            int(limit.metrics["observed_records"]) if "observed_records" in limit.metrics else None
        ),
        limit_records=(
            int(limit.metrics["limit_records"]) if "limit_records" in limit.metrics else None
        ),
        observed_seconds=(
            float(limit.metrics["observed_seconds"])
            if "observed_seconds" in limit.metrics
            else None
        ),
        limit_seconds=(
            float(limit.metrics["limit_seconds"]) if "limit_seconds" in limit.metrics else None
        ),
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Analyze manifest permissions vs code capabilities within shared bounds."""
    manifest: dict = state.get("manifest") or {}
    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    component_metadata: list[dict] = state.get("component_metadata") or []

    # Skip: no manifest
    if not manifest:
        logger.info("%s: no manifest, skipping", ANALYZER_ID)
        return {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id=ANALYZER_ID,
                    status="not_applicable",
                    reason=LedgerReason.MANIFEST_ABSENT,
                )
            ],
        }

    # Skip: docs-only skill (no executable files)
    has_executable = any(m.get("executable", False) for m in component_metadata)
    if not has_executable:
        logger.info("%s: no executable files, skipping", ANALYZER_ID)
        return {
            "findings": [],
            "inspection_ledger": [],
            "analyzer_status_events": [
                analyzer_status_event(
                    analyzer_id=ANALYZER_ID,
                    status="not_applicable",
                    reason=LedgerReason.NO_APPLICABLE_FILES,
                )
            ],
        }

    # Retrieve declared permissions (may be None if not set in manifest).
    permissions_raw = manifest.get("permissions")  # None | list[str]
    if isinstance(permissions_raw, list):
        permissions: list[str] | None = permissions_raw
    else:
        permissions = None  # treat missing or non-list as None

    executable_paths = list(
        dict.fromkeys(
            str(metadata["path"])
            for metadata in component_metadata
            if metadata.get("executable", False)
        )
    )
    planned_paths = list(dict.fromkeys([*executable_paths, "SKILL.md"]))
    budget = _LeastPrivilegeBudget(state)
    file_capabilities: dict[str, set[str]] = {}
    all_caps: set[str] = set()
    resource_limit: _LeastPrivilegeResourceLimitError | None = None
    partial_paths: set[str] = set()
    aggregate_partial_paths: set[str] = set()

    try:
        # Scan each executable before making whole-skill comparisons. A timeout
        # therefore never turns a partially observed capability set into LP4.
        for path in executable_paths:
            budget.check_runtime(path)
            capabilities = _detect_capabilities(file_cache.get(path, ""), budget, path)
            if capabilities:
                file_capabilities[path] = capabilities
                all_caps.update(capabilities)
            budget.completed_paths.add(path)

        # Whole-skill declaration work can affect the verdict for every source
        # artifact. Narrow this set to SKILL.md once all LP1 work completes.
        aggregate_partial_paths = {*executable_paths, "SKILL.md"}
        budget.check_runtime("SKILL.md")
        allowed_tools = _normalize_allowed_tools(manifest.get("allowed-tools"), budget)
        wildcard_present = isinstance(permissions, list) and _has_wildcard(permissions, budget)

        if wildcard_present:
            logger.debug("%s: LP2 wildcard permission detected", ANALYZER_ID)
            budget.emit(
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

        permissions_absent = (permissions is None or permissions == []) and not allowed_tools
        if permissions_absent and all_caps:
            logger.debug("%s: LP3 no permissions declared but capabilities detected", ANALYZER_ID)
            cap_names = ", ".join(sorted(all_caps))
            budget.emit(
                Finding(
                    rule_id="LP3",
                    message=(
                        "Skill declares no tool scope ('permissions' or 'allowed-tools') "
                        f"but code capabilities were detected: {cap_names}."
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
                        "Declare the skill's tool scope: for Claude Code / Agent Skills "
                        "SKILL.md, list the tools the skill may invoke in the "
                        "'allowed-tools' frontmatter field; for MCP server manifests, "
                        "add a 'permissions' list naming the required capabilities."
                    ),
                )
            )

        has_declaration = (isinstance(permissions, list) and permissions) or bool(allowed_tools)
        if has_declaration:
            declared_categories: set[str] = set()
            if isinstance(permissions, list) and permissions:
                declared_categories |= _map_permissions_to_categories(permissions, budget)
            if allowed_tools:
                declared_categories |= _map_allowed_tools_to_categories(allowed_tools, budget)

            if not wildcard_present:
                cap_in_test_only: set[str] = set()
                cap_in_code: set[str] = set()
                for path, capabilities in file_capabilities.items():
                    budget.check_runtime("SKILL.md")
                    if _is_test_file(path):
                        cap_in_test_only.update(capabilities)
                    else:
                        cap_in_code.update(capabilities)
                test_only_caps = cap_in_test_only - cap_in_code

                for capability in sorted(all_caps):
                    budget.check_runtime("SKILL.md")
                    if capability in declared_categories:
                        continue
                    primary_file = "SKILL.md"
                    for path, capabilities in file_capabilities.items():
                        budget.check_runtime("SKILL.md")
                        if capability in capabilities:
                            primary_file = path
                            break
                    confidence = _clamp(0.55 if capability in test_only_caps else 0.75)
                    remediation = (
                        f"Add a tool that covers the '{capability}' capability to the "
                        "'allowed-tools' frontmatter field in SKILL.md, or remove "
                        "the code that requires it."
                        if allowed_tools
                        else f"Add the '{capability}' capability to the MCP server manifest's "
                        "'permissions' list, or remove the code that requires it."
                    )
                    budget.emit(
                        Finding(
                            rule_id="LP1",
                            message=(
                                f"Code capability '{capability}' detected in {primary_file} "
                                "but not covered by declared permissions."
                            ),
                            severity="HIGH",
                            confidence=confidence,
                            file=primary_file,
                            category=_CATEGORY,
                            tags=list(_TAGS),
                            explanation=(
                                f"The skill uses '{capability}' capability that is not listed in "
                                "its permissions. This may indicate deceptive intent or missing "
                                "permission declarations."
                            ),
                            remediation=remediation,
                        )
                    )

            aggregate_partial_paths = {"SKILL.md"}
            for index, permission in enumerate(permissions or [], start=1):
                budget.check_runtime("SKILL.md")
                if index > _MAX_DECLARATION_VALUES:
                    raise _LeastPrivilegeResourceLimitError(
                        LedgerReason.OUTPUT_LIMIT,
                        {
                            "observed_records": index,
                            "limit_records": _MAX_DECLARATION_VALUES,
                        },
                        path="SKILL.md",
                    )
                permission_text = str(permission)
                if (
                    len(permission_text) <= 16
                    and permission_text.strip().lower() in _WILDCARD_PERMS
                ):
                    continue
                matched_category = _permission_category(permission_text, budget)
                if matched_category is None or matched_category in all_caps:
                    continue
                display_permission = _bounded_evidence(permission_text)
                logger.debug(
                    "%s: LP4 over-declared permission maps to %s",
                    ANALYZER_ID,
                    matched_category,
                )
                budget.emit(
                    Finding(
                        rule_id="LP4",
                        message=(
                            f"Permission '{display_permission}' is declared but no corresponding "
                            f"code capability ({matched_category}) was detected."
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
                            f"Remove the '{display_permission}' permission if the corresponding "
                            "capability is no longer used."
                        ),
                    )
                )

        budget.check_runtime("SKILL.md")
        budget.completed_paths.add("SKILL.md")
    except _LeastPrivilegeResourceLimitError as exc:
        resource_limit = exc
        partial_paths.add(budget.current_path)
        partial_paths.update(aggregate_partial_paths)
        if exc.path is not None:
            partial_paths.add(exc.path)

    findings_by_path: dict[str, list[str]] = {}
    for finding in budget.findings:
        findings_by_path.setdefault(finding.file, []).append(finding.finding_id)

    events = []
    for path in planned_paths:
        emitted_ids = findings_by_path.get(path, [])
        if resource_limit is not None and (
            path in partial_paths or path not in budget.completed_paths
        ):
            events.append(_partial_limit_event(path, resource_limit, emitted_ids))
        else:
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.COMPLETED,
                    phase="static",
                    path=path,
                    emitted_finding_ids=emitted_ids,
                )
            )

    logger.info("%s: %d findings", ANALYZER_ID, len(budget.findings))
    return {
        "findings": budget.findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(ANALYZER_ID, events)],
    }
