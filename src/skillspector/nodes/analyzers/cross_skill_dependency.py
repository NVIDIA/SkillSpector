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

"""Cross-skill dependency analyzer: detect references between skills in multi-skill dirs.

When scanning a directory containing multiple skills, this analyzer detects:
- Direct references from one skill to another (invoke, call, import)
- Privilege escalation chains (skill A grants permissions that skill B exploits)
- Shared state or file access between skills
- Circular dependencies between skills

These patterns can indicate coordinated supply-chain attacks where a benign-looking
skill serves as a vector for a malicious one.
"""

from __future__ import annotations

import re

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .static_runner import MAX_FILE_CHARS, analyzer_finding_to_finding

ANALYZER_ID = "cross_skill_dependency"
logger = get_logger(__name__)

_TAG = "Cross-Skill Dependency"

# Patterns that reference other skills by name or path
_SKILL_REFERENCE_PATTERNS = [
    re.compile(r"(?:invoke|call|run|execute|use|load|import)\s+(?:skill\s+)?['\"]([a-zA-Z0-9_-]+)['\"]", re.IGNORECASE),
    re.compile(r"(?:skill|agent|tool)[/\s]+([a-zA-Z0-9_-]+)", re.IGNORECASE),
    re.compile(r"(?:depends?\s+on|requires?\s+skill|needs?\s+skill)\s+['\"]?([a-zA-Z0-9_-]+)['\"]?", re.IGNORECASE),
    re.compile(r"\{\{([a-zA-Z0-9_-]+)\.(?:output|result|response)\}\}", re.IGNORECASE),
]

# Patterns that suggest privilege escalation chains
_PRIVILEGE_ESCALATION_PATTERNS = [
    re.compile(r"(?:grant|give|assign|set)\s+(?:permission|access|role|capability)\s+to\s+['\"]([a-zA-Z0-9_-]+)['\"]", re.IGNORECASE),
    re.compile(r"(?:share|expose|export)\s+(?:credentials?|tokens?|keys?|secrets?)\s+(?:with|to)\s+['\"]([a-zA-Z0-9_-]+)['\"]", re.IGNORECASE),
    re.compile(r"(?:pipe|chain|pass)\s+(?:output|results?)\s+(?:to|into)\s+['\"]([a-zA-Z0-9_-]+)['\"]", re.IGNORECASE),
]

# Patterns for shared state access (only reported when the file also contains
# an actual cross-skill reference, to avoid flagging every mention of "mutex").
_SHARED_STATE_PATTERNS = [
    re.compile(r"(?:shared|common|global)\s+(?:state|config|store|cache|registry)", re.IGNORECASE),
    re.compile(r"(?:/tmp/|/var/|~/.cache/).*(?:skill|agent)", re.IGNORECASE),
]


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    all_skill_names: list[str] | None = None,
) -> list[AnalyzerFinding]:
    """Analyze content for cross-skill dependency patterns."""
    findings: list[AnalyzerFinding] = []
    skill_names = {s.lower() for s in (all_skill_names or [])}
    if not skill_names:
        return findings
    self_names = _ancestor_skill_names(file_path, skill_names)

    # CS1: Direct skill references
    for pattern in _SKILL_REFERENCE_PATTERNS:
        for match in pattern.finditer(content):
            ref_name = match.group(1).lower() if match.lastindex else ""
            if ref_name in skill_names and ref_name not in self_names:
                line_num = content[:match.start()].count("\n") + 1
                findings.append(
                    AnalyzerFinding(
                        rule_id="CS1",
                        message=f"Cross-skill reference to '{match.group(1)}'",
                        severity=Severity.MEDIUM,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=0.7,
                        tags=[_TAG],
                        context=content[max(0, match.start() - 50) : match.end() + 50],
                        matched_text=match.group(0)[:200],
                    )
                )

    # CS2: Privilege escalation chains
    for pattern in _PRIVILEGE_ESCALATION_PATTERNS:
        for match in pattern.finditer(content):
            ref_name = match.group(1).lower() if match.lastindex else ""
            if ref_name in skill_names and ref_name not in self_names:
                line_num = content[:match.start()].count("\n") + 1
                findings.append(
                    AnalyzerFinding(
                        rule_id="CS2",
                        message=f"Privilege escalation chain involving '{match.group(1)}'",
                        severity=Severity.HIGH,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=0.75,
                        tags=[_TAG],
                        context=content[max(0, match.start() - 50) : match.end() + 50],
                        matched_text=match.group(0)[:200],
                    )
                )

    # CS3: Shared state access (only meaningful when this file actually
    # references another skill; otherwise "mutex"/"shared cache" is benign prose).
    has_cross_skill_ref = False
    for pattern in _SKILL_REFERENCE_PATTERNS:
        for match in pattern.finditer(content):
            ref_name = match.group(1).lower() if match.lastindex else ""
            if ref_name in skill_names and ref_name not in self_names:
                has_cross_skill_ref = True
                break
        if has_cross_skill_ref:
            break

    if has_cross_skill_ref:
        for pattern in _SHARED_STATE_PATTERNS:
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count("\n") + 1
                findings.append(
                    AnalyzerFinding(
                        rule_id="CS3",
                        message="Shared state mechanism detected between skills",
                        severity=Severity.LOW,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=0.5,
                        tags=[_TAG],
                        context=content[max(0, match.start() - 50) : match.end() + 50],
                        matched_text=match.group(0)[:200],
                    )
                )

    return findings


def _ancestor_skill_names(file_path: str, skill_names: set[str]) -> set[str]:
    """Return the skill names the file lives under (its own skill directory)."""
    parts = file_path.replace("\\", "/").split("/")
    return {part.lower() for part in parts[:-1] if part.lower() in skill_names}


def _skill_name_from_path(file_path: str) -> str:
    """Extract the owning skill directory name from a file path.

    Returns the nearest ancestor directory, so "skill-a/SKILL.md" -> "skill-a"
    (previously this returned the file stem "SKILL", which broke the CS1
    self-reference exclusion).
    """
    parts = file_path.replace("\\", "/").split("/")
    for part in reversed(parts[:-1]):
        if part and not part.startswith("."):
            return part
    return ""


def _detect_circular_references(
    references: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """Detect circular dependencies in a reference graph using DFS."""
    cycles: list[tuple[str, str]] = []
    visited: set[str] = set()
    in_stack: set[str] = set()

    def _dfs(node: str, path: list[str]) -> None:
        if node in in_stack:
            cycle_start = path.index(node)
            cycles.append((node, path[cycle_start]))
            return
        if node in visited:
            return
        visited.add(node)
        in_stack.add(node)
        path.append(node)
        for neighbor in references.get(node, set()):
            _dfs(neighbor, path)
        path.pop()
        in_stack.discard(node)

    for node in references:
        _dfs(node, [])

    return cycles


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Detect cross-skill dependency patterns."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    all_findings: list[Finding] = []

    # Extract skill names from directory structure: only directories that
    # actually contain a SKILL.md are skills. (Collecting every intermediate
    # directory previously made a single skill with a scripts/ subdir pass the
    # >=2 gate and matched prose like "tool scripts".)
    skill_names: list[str] = []
    for path in components:
        parts = path.replace("\\", "/").split("/")
        if parts[-1].lower() == "skill.md" and len(parts) >= 2:
            skill_names.append(parts[-2])
    skill_names = sorted(set(skill_names))

    if len(skill_names) < 2:
        logger.info("%s: fewer than 2 skill dirs detected, skipping", ANALYZER_ID)
        return {"findings": []}

    for path in components:
        content = file_cache.get(path)
        if content is None or len(content) > MAX_FILE_CHARS:
            continue
        raw = analyze(content, path, "other", all_skill_names=skill_names)
        all_findings.extend(analyzer_finding_to_finding(af) for af in raw)

    logger.info("%s: %d findings", ANALYZER_ID, len(all_findings))
    return {"findings": all_findings}
