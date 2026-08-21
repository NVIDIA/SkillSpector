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
import time
from dataclasses import dataclass, field

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_event,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import Finding, compute_match_fingerprint
from skillspector.state import (
    AnalyzerNodeResponse,
    SkillspectorState,
    transitive_remaining_seconds,
)

from .static_runner import MAX_FINDINGS_PER_ANALYZER, MAX_FINDINGS_PER_ARTIFACT

ANALYZER_ID = "mcp_rug_pull"
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATEGORY = "MCP Rug Pull"
_TAGS = ["ASI16"]


class _RugPullResourceLimitError(RuntimeError):
    """Internal fail-closed signal for construction-time resource ceilings."""

    def __init__(self, reason: LedgerReason, metrics: dict[str, int | float]) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.metrics = metrics


@dataclass
class _RugPullBudget:
    """Retain a bounded prefix of evidence while enforcing shared runtime."""

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
            raise _RugPullResourceLimitError(
                LedgerReason.RUNTIME_LIMIT,
                {
                    "observed_seconds": max(0.0, time.monotonic() - self.started_at),
                    "limit_seconds": self.initial_allowance,
                },
            )

    def emit(self, finding: Finding) -> None:
        self.check_runtime(finding.file)
        artifact_observed = self.artifact_findings.get(finding.file, 0) + 1
        analyzer_observed = len(self.findings) + 1
        if artifact_observed > MAX_FINDINGS_PER_ARTIFACT:
            raise _RugPullResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": artifact_observed,
                    "limit_findings": MAX_FINDINGS_PER_ARTIFACT,
                },
            )
        if analyzer_observed > MAX_FINDINGS_PER_ANALYZER:
            raise _RugPullResourceLimitError(
                LedgerReason.OUTPUT_LIMIT,
                {
                    "observed_findings": analyzer_observed,
                    "limit_findings": MAX_FINDINGS_PER_ANALYZER,
                },
            )
        self.findings.append(finding)
        self.artifact_findings[finding.file] = artifact_observed

    def analyzer_exhausted(self) -> bool:
        return len(self.findings) >= MAX_FINDINGS_PER_ANALYZER


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
    return content.count("\n", 0, pos) + 1


def _normalize_string_list(
    lst: list[object] | None,
    budget: _RugPullBudget | None = None,
) -> list[str]:
    """Strip and lowercase all strings in the list. Returns sorted list of unique values."""
    if not lst:
        return []
    res = set()
    for item in lst:
        if budget is not None:
            budget.check_runtime("SKILL.md")
        if item is not None:
            res.add(str(item).strip().lower())
    return sorted(res)


def _get_parameters_map(
    parameters: list[object] | None,
    budget: _RugPullBudget | None = None,
) -> dict[str, dict[str, object]]:
    """Convert parameters list of dicts to a map of lowercase parameter names -> properties."""
    param_map: dict[str, dict[str, object]] = {}
    if not parameters:
        return param_map
    for item in parameters:
        if budget is not None:
            budget.check_runtime("SKILL.md")
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


def _check_rp1(
    manifest: dict,
    file_cache: dict[str, str],
    budget: _RugPullBudget,
) -> None:
    """Detect unpinned MCP server command references in skill files."""
    for file_path, content in file_cache.items():
        budget.check_runtime(file_path)
        # npx without @version
        for m in _RP1_NPX_CMD.finditer(content):
            budget.check_runtime(file_path)
            full_match = m.group(0)
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line_remainder = content[m.end() : min(line_end, m.end() + 256)]
            if _VERSION_PIN_RE.search(full_match) or _VERSION_PIN_RE.search(line_remainder):
                continue
            line_num = _find_line(content, m.start())
            budget.emit(
                Finding(
                    rule_id="RP1",
                    message=(
                        "MCP server referenced without pinned version: "
                        f"'{full_match.strip()[:200]}'."
                    ),
                    severity="MEDIUM",
                    confidence=0.70,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    match_fingerprint=compute_match_fingerprint("RP1", full_match),
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
            budget.check_runtime(file_path)
            full_match = m.group(0)
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line_remainder = content[m.end() : min(line_end, m.end() + 256)]
            if _VERSION_PIN_RE.search(full_match) or _VERSION_PIN_RE.search(line_remainder):
                continue
            line_num = _find_line(content, m.start())
            budget.emit(
                Finding(
                    rule_id="RP1",
                    message=(
                        "MCP server referenced without pinned version: "
                        f"'{full_match.strip()[:200]}'."
                    ),
                    severity="MEDIUM",
                    confidence=0.65,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    match_fingerprint=compute_match_fingerprint("RP1", full_match),
                    explanation=(
                        "uvx/uv tool run commands without ==version create a rug-pull risk."
                    ),
                    remediation="Pin the version: uvx package-name==1.2.3",
                )
            )

        # pip install without ==version
        for m in _RP1_PIP_INSTALL.finditer(content):
            budget.check_runtime(file_path)
            full_match = m.group(0)
            line_end = content.find("\n", m.end())
            if line_end == -1:
                line_end = len(content)
            line_remainder = content[m.end() : min(line_end, m.end() + 256)]
            if _VERSION_PIN_RE.search(full_match) or _VERSION_PIN_RE.search(line_remainder):
                continue
            pkg = m.group(1)
            if "mcp" not in pkg.lower():
                continue
            line_num = _find_line(content, m.start())
            budget.emit(
                Finding(
                    rule_id="RP1",
                    message=(
                        "MCP server dependency without pinned version: "
                        f"'{full_match.strip()[:200]}'."
                    ),
                    severity="LOW",
                    confidence=0.60,
                    file=file_path,
                    start_line=line_num,
                    category=_CATEGORY,
                    tags=list(_TAGS),
                    matched_text=full_match[:200],
                    match_fingerprint=compute_match_fingerprint("RP1", full_match),
                    explanation=(
                        "pip install without ==version installs the latest "
                        "release, which could include malicious changes."
                    ),
                    remediation="Pin the version: pip install package==1.2.3",
                )
            )

        # docker without tag or digest
        for m in _RP1_DOCKER_CMD.finditer(content):
            budget.check_runtime(file_path)
            full_match = m.group(0)
            if _VERSION_PIN_RE.search(full_match):
                continue
            line_num = _find_line(content, m.start())
            budget.emit(
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
                    match_fingerprint=compute_match_fingerprint("RP1", full_match),
                    explanation=(
                        "Docker image references without a specific tag (:latest "
                        "is implicit) or digest (@sha256:...) can be silently "
                        "replaced by a malicious image."
                    ),
                    remediation="Pin the image: image:tag or image@sha256:abc123",
                )
            )

        if file_path != "SKILL.md":
            budget.completed_paths.add(file_path)

    if not manifest:
        return

    # Check manifest for unpinned MCP server references.
    budget.check_runtime("SKILL.md")
    manifest_text = str(manifest)
    for m in _RP1_NPX_CMD.finditer(manifest_text):
        budget.check_runtime("SKILL.md")
        budget.emit(
            Finding(
                rule_id="RP1",
                message=(
                    "Manifest references MCP server without version pin: "
                    f"'{m.group(0).strip()[:200]}'."
                ),
                severity="MEDIUM",
                confidence=0.70,
                file="SKILL.md",
                start_line=1,
                category=_CATEGORY,
                tags=list(_TAGS),
                matched_text=m.group(0)[:200],
                match_fingerprint=compute_match_fingerprint("RP1", m.group(0)),
                explanation=(
                    "MCP server references in the skill manifest without version "
                    "pinning are a rug-pull risk."
                ),
                remediation="Always pin MCP server versions in manifest references.",
            )
        )


# ---------------------------------------------------------------------------
# RP2: Permission pre-staging
# ---------------------------------------------------------------------------


def _check_rp2(manifest: dict, budget: _RugPullBudget) -> None:
    """Detect manifest permission patterns that suggest pre-staging for future abuse."""
    budget.check_runtime("SKILL.md")
    manifest_text = str(manifest)
    for pattern, confidence in _PERMISSION_EXPANSION_PATTERNS:
        for m in re.finditer(pattern, manifest_text, re.IGNORECASE):
            budget.check_runtime("SKILL.md")
            budget.emit(
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
                    match_fingerprint=compute_match_fingerprint("RP2", m.group(0)),
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


# ---------------------------------------------------------------------------
# RP3: Version unpinned
# ---------------------------------------------------------------------------


def _check_rp3(manifest: dict, budget: _RugPullBudget) -> None:
    """Detect when skill version is unpinned or uses broad constraints."""
    budget.check_runtime("SKILL.md")
    version_value = manifest.get("version") if isinstance(manifest, dict) else None
    if not version_value or not isinstance(version_value, str):
        return

    version_str = str(version_value).strip()
    if version_str in ("*", "latest", "any"):
        budget.emit(
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
        budget.emit(
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


# ---------------------------------------------------------------------------
# Manifest comparison and terminal accounting
# ---------------------------------------------------------------------------


def _bounded_display(values: list[str], *, max_items: int = 32, max_chars: int = 1024) -> str:
    """Render attacker-controlled change lists under a deterministic output cap."""
    rendered = ", ".join(values[:max_items])
    if len(values) > max_items:
        rendered = f"{rendered}, ... ({len(values) - max_items} more)"
    return rendered[:max_chars]


def _check_manifest_changes(
    manifest: dict,
    previous_manifest: dict,
    budget: _RugPullBudget,
) -> None:
    """Emit bounded RP1-RP3 findings for changes from a previous manifest."""
    budget.check_runtime("SKILL.md")
    curr_perms = _normalize_string_list(manifest.get("permissions"), budget)
    prev_perms = _normalize_string_list(previous_manifest.get("permissions"), budget)
    prev_perm_set = set(prev_perms)
    added_perms = [permission for permission in curr_perms if permission not in prev_perm_set]
    if added_perms:
        budget.emit(
            Finding(
                rule_id="RP1",
                message=(
                    "Permissions expanded: current manifest requests permissions not present "
                    f"in the previous version (added: {_bounded_display(added_perms)})."
                ),
                severity="HIGH",
                confidence=0.90,
                file="SKILL.md",
                category=_CATEGORY,
                tags=["ASI02"],
                explanation=(
                    "A skill version update added new permissions to the manifest. If unexpected, "
                    "this could indicate a privilege escalation or rug-pull attack."
                ),
                remediation="Verify each added permission and remove any that are unnecessary.",
            )
        )

    curr_triggers = _normalize_string_list(manifest.get("triggers"), budget)
    prev_triggers = _normalize_string_list(previous_manifest.get("triggers"), budget)
    prev_trigger_set = set(prev_triggers)
    curr_trigger_set = set(curr_triggers)
    added_triggers = [trigger for trigger in curr_triggers if trigger not in prev_trigger_set]
    removed_triggers = [trigger for trigger in prev_triggers if trigger not in curr_trigger_set]
    if added_triggers or removed_triggers:
        changes: list[str] = []
        if added_triggers:
            changes.append(f"added: {_bounded_display(added_triggers)}")
        if removed_triggers:
            changes.append(f"removed: {_bounded_display(removed_triggers)}")
        budget.emit(
            Finding(
                rule_id="RP2",
                message=f"Trigger phrases modified ({'; '.join(changes)[:2048]}).",
                severity="MEDIUM",
                confidence=0.85,
                file="SKILL.md",
                category=_CATEGORY,
                tags=["ASI02"],
                explanation=(
                    "Changing triggers can cause unintended invocation or bypass expected safety "
                    "boundaries."
                ),
                remediation="Verify that every trigger remains aligned with the declared behavior.",
            )
        )

    curr_params = _get_parameters_map(manifest.get("parameters"), budget)
    prev_params = _get_parameters_map(previous_manifest.get("parameters"), budget)
    added_params = [name for name in curr_params if name not in prev_params]
    removed_params = [name for name in prev_params if name not in curr_params]
    changed_params: list[str] = []
    for name, curr_prop in curr_params.items():
        budget.check_runtime("SKILL.md")
        prev_prop = prev_params.get(name)
        if prev_prop is None:
            continue
        prop_diffs: list[str] = []
        if curr_prop["type"] != prev_prop["type"]:
            prop_diffs.append(
                f"type changed from {str(prev_prop['type'])[:128]} "
                f"to {str(curr_prop['type'])[:128]}"
            )
        if curr_prop["default"] != prev_prop["default"]:
            prop_diffs.append(
                f"default changed from {str(prev_prop['default'])[:128]} "
                f"to {str(curr_prop['default'])[:128]}"
            )
        if curr_prop["description"] != prev_prop["description"]:
            prop_diffs.append("description changed")
        if prop_diffs:
            changed_params.append(f"{str(curr_prop['name'])[:128]} ({'; '.join(prop_diffs)})")

    if added_params or removed_params or changed_params:
        changes = []
        if added_params:
            changes.append(
                "added: "
                + _bounded_display([str(curr_params[name]["name"]) for name in added_params])
            )
        if removed_params:
            changes.append(
                "removed: "
                + _bounded_display([str(prev_params[name]["name"]) for name in removed_params])
            )
        if changed_params:
            changes.append("modified: " + _bounded_display(changed_params))
        budget.emit(
            Finding(
                rule_id="RP3",
                message=f"Parameter schema modified ({'; '.join(changes)[:3072]}).",
                severity="MEDIUM",
                confidence=0.80,
                file="SKILL.md",
                category=_CATEGORY,
                tags=["ASI02"],
                explanation=(
                    "Parameter additions, removals, or changed defaults can alter tool input flow "
                    "and behavior."
                ),
                remediation="Verify that every parameter change is safe and expected.",
            )
        )


def _partial_limit_event(
    path: str,
    limit: _RugPullResourceLimitError,
    emitted_finding_ids: list[str],
) -> InspectionLedgerEvent:
    return ledger_event(
        analyzer_id=ANALYZER_ID,
        outcome=LedgerOutcome.PARTIAL,
        phase="static",
        path=path,
        reason=limit.reason,
        emitted_finding_ids=emitted_finding_ids,
        observed_findings=(
            int(limit.metrics["observed_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        limit_findings=(
            int(limit.metrics["limit_findings"])
            if limit.reason is LedgerReason.OUTPUT_LIMIT
            else None
        ),
        observed_seconds=(
            float(limit.metrics["observed_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
        limit_seconds=(
            float(limit.metrics["limit_seconds"])
            if limit.reason is LedgerReason.RUNTIME_LIMIT
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Analyze skill for rug-pull risks (RP1-RP3) within explicit bounds."""
    manifest: dict = state.get("manifest") or {}
    file_cache: dict[str, str] = state.get("local_file_cache") or state.get("file_cache") or {}
    previous_manifest: dict | None = state.get("previous_manifest")

    if not manifest and not file_cache:
        logger.info("%s: no manifest or files, skipping", ANALYZER_ID)
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

    budget = _RugPullBudget(state)
    resource_limit: _RugPullResourceLimitError | None = None
    try:
        _check_rp1(manifest, file_cache, budget)
        if manifest:
            _check_rp2(manifest, budget)
            _check_rp3(manifest, budget)
        if manifest and previous_manifest:
            _check_manifest_changes(manifest, previous_manifest, budget)
        budget.completed_paths.update(file_cache)
        if manifest:
            budget.completed_paths.add("SKILL.md")
    except _RugPullResourceLimitError as exc:
        resource_limit = exc

    findings = budget.findings
    findings_by_path: dict[str, list[str]] = {}
    for finding in findings:
        findings_by_path.setdefault(finding.file, []).append(finding.finding_id)

    planned_paths = list(file_cache)
    if manifest and "SKILL.md" not in planned_paths:
        planned_paths.append("SKILL.md")
    events = []
    for path in planned_paths:
        emitted_ids = findings_by_path.get(path, [])
        if resource_limit is None or path in budget.completed_paths:
            events.append(
                ledger_event(
                    analyzer_id=ANALYZER_ID,
                    outcome=LedgerOutcome.COMPLETED,
                    phase="static",
                    path=path,
                    emitted_finding_ids=emitted_ids,
                )
            )
        else:
            events.append(_partial_limit_event(path, resource_limit, emitted_ids))

    logger.info("%s: %d findings in total", ANALYZER_ID, len(findings))
    return {
        "findings": findings,
        "inspection_ledger": events,
        "analyzer_status_events": [analyzer_status_for_events(ANALYZER_ID, events)],
    }
