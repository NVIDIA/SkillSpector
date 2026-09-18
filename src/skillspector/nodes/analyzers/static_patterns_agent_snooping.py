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

"""Static patterns: agent snooping (AS1–AS3). Node and analyze() in one module.

Detects patterns where a skill attempts to read agent configuration
directories (AS1), access MCP server config files (AS2), or enumerate and
read other installed skills (AS3).

A skill performing these accesses gains knowledge it has no legitimate
need for: API keys stored in agent config, other skills' prompts, or the
full list of tools available to the agent.

Framework: OWASP LLMT09 (Misinformation), ASI-SR-003 (Least Knowledge).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextvars import ContextVar

from skillspector.input_handler import selected_source_identity_for_input
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_agent_snooping"

_AS3_SKILL_PATH_PATTERN = r"skills?/(?P<skill_name>(?!CURRENT)[A-Z][A-Za-z0-9_-]+)/SKILL\.md"
_AS3_SKILL_PATH_FULLMATCH = re.compile(_AS3_SKILL_PATH_PATTERN, re.IGNORECASE)
_CURRENT_SKILL_IDENTIFIERS: ContextVar[frozenset[str]] = ContextVar(
    "agent_snooping_current_skill_identifiers", default=frozenset()
)
# Ephemeral basenames created by InputHandler for git/zip/file materialization.
_EPHEMERAL_SCAN_ROOT_BASENAMES = frozenset({"repo", "extracted"})

# AS1: Agent Config Directory Access
# Matches code/instructions that read from well-known agent config directories.
AS1_CODE_PATTERNS = [
    # Direct filesystem access to .claude/, .codex/, .gemini/ directories
    (r"open\s*\(\s*['\"]?\.(?:claude|codex|gemini|continue)/", 0.9),
    (r"(?:Path|pathlib\.Path)\s*\(\s*['\"]?\.(?:claude|codex|gemini|continue)/", 0.9),
    (r"os\.path\.(?:join|exists|isfile)\s*\(\s*['\"]?\.(?:claude|codex|gemini|continue)", 0.85),
    # Shell commands targeting config dirs
    (r"(?:cat|less|head|tail|grep|find)\s+[^|&;\n]*~?/?\.(claude|codex|gemini)/", 0.85),
    # Home-directory config paths
    (
        r"~?/\.(?:claude|codex|gemini|continue)/(?:config|settings?|preferences?|credentials?)(?:\.(?:json|yaml|yml|toml))?",
        0.9,
    ),
]
AS1_PROSE_PATTERNS = [
    (
        r"(?:read|load|open|access|fetch)\s+(?:the\s+)?(?:agent|claude|codex|gemini)\s+(?:config|configuration|settings?)\s+(?:from|at|in)\s+~?/?\.(?:claude|codex|gemini)",
        0.9,
    ),
    # Generic "agent config" snooping instructions
    (
        r"(?:read|access|inspect|examine|retrieve)\s+(?:the\s+)?(?:agent|assistant)\s+(?:config(?:uration)?|settings?|preferences?)\s+(?:file|directory|folder)",
        0.8,
    ),
]
AS1_PATTERNS = AS1_CODE_PATTERNS + AS1_PROSE_PATTERNS

# AS2: MCP Config Access
# Matches code/instructions specifically targeting MCP server configuration.
AS2_CODE_PATTERNS = [
    # Direct access to mcp.json files
    (r"open\s*\(\s*['\"][^'\"]*mcp(?:_config)?\.json['\"]", 0.95),
    (r"(?:Path|pathlib\.Path)\s*\(\s*['\"][^'\"]*mcp(?:_config)?\.json['\"]", 0.9),
    (r"(?:cat|less|head|grep)\s+[^|&;\n]*mcp(?:_config)?\.json", 0.9),
    (r"\.(?:claude|codex|gemini)/mcp(?:_config)?\.json", 0.95),
    # Accessing MCP server URLs or API keys from config
    (r"mcp(?:_config)?\.json.*?(?:api_?key|token|secret|url|endpoint)", 0.9),
]
AS2_PROSE_PATTERNS = [
    # Instructions referencing MCP config
    (
        r"(?:read|access|load|inspect)\s+(?:the\s+)?mcp(?:\.json|_config)?\s+(?:file|config(?:uration)?|settings?)",
        0.9,
    ),
    # Listing MCP servers
    (
        r"(?:list|enumerate|discover)\s+(?:all\s+)?(?:available\s+)?mcp\s+(?:servers?|tools?|services?)",
        0.8,
    ),
]
AS2_PATTERNS = AS2_CODE_PATTERNS + AS2_PROSE_PATTERNS

# AS3: Skill Enumeration / Snooping
# Matches code/instructions that enumerate or read other installed skills.
AS3_CODE_PATTERNS = [
    # Listing skill directories
    (
        r"(?:os\.listdir|os\.scandir|glob\.glob|Path\.iterdir)\s*\([^)]*\.(?:claude|codex|gemini)/skills?",
        0.9,
    ),
    (r"(?:ls|find|dir)\s+[^|&;\n]*\.(?:claude|codex|gemini)/skills?", 0.85),
    # Reading other skills' SKILL.md files
    (r"open\s*\(\s*['\"][^'\"]*SKILL\.md['\"].*?\bother\b", 0.85),
    # Accessing skills/CURRENT or adjacent skill directories
    (_AS3_SKILL_PATH_PATTERN, 0.8),
]
AS3_PROSE_PATTERNS = [
    (
        r"(?:read|access|inspect|enumerate)\s+(?:all\s+)?(?:installed|available|other)\s+skills?(?:\s+in\s+(?:the\s+)?(?:skills?|agent)\s+(?:directory|folder))?",
        0.85,
    ),
    # Instructions to discover other skills
    (
        r"(?:list|discover|find|enumerate|identify)\s+(?:all\s+)?(?:other|installed|available)\s+(?:skills?|agents?|tools?)\s+(?:in\s+)?(?:the\s+)?(?:\.(?:claude|codex|gemini)|\$HOME)",
        0.85,
    ),
    # Reading tool manifests of other agents
    (
        r"(?:read|access|load)\s+(?:the\s+)?(?:SKILL|skill)\.md\s+(?:file\s+)?(?:of|from|for)\s+(?:another|other|different|all)\s+(?:skill|agent|tool)",
        0.9,
    ),
]
AS3_PATTERNS = AS3_CODE_PATTERNS + AS3_PROSE_PATTERNS


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for agent snooping patterns (AS1–AS3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.AGENT_SNOOPING.value]

    for pattern, confidence in AS1_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in AS1_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="AS1",
                    message="Agent Config Directory Access",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )

    for pattern, confidence in AS2_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in AS2_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="AS2",
                    message="MCP Config Access",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )

    for pattern, confidence in AS3_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in AS3_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            full_match = match.group(0)
            if _is_current_skill_path_reference(
                full_match, _CURRENT_SKILL_IDENTIFIERS.get()
            ) and static_runner.security_view_match_is_literal(content, match.start(), match.end()):
                continue
            matched_text = full_match[:200]
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="AS3",
                    message="Skill Enumeration",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=matched_text,
                    complete_match=full_match,
                )
            )

    return findings


def _normalize_skill_identifier(value: object) -> str | None:
    """Return a usable identifier without aliasing filesystem names."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _is_ephemeral_scan_root_basename(identifier: str) -> bool:
    """Return whether a scan-root basename is an InputHandler materialization stub."""
    return identifier in _EPHEMERAL_SCAN_ROOT_BASENAMES or identifier.startswith("skillspector_")


def _selected_source_identifier(state: SkillspectorState) -> str | None:
    """Return the trusted repository/archive/selected-source identity from state."""
    selected = _normalize_skill_identifier(state.get("selected_source_identity"))
    if selected is not None:
        return selected
    input_path = state.get("input_path")
    if isinstance(input_path, str) and input_path.strip():
        return _normalize_skill_identifier(selected_source_identity_for_input(input_path.strip()))
    return None


def _current_skill_identifiers(state: SkillspectorState) -> frozenset[str]:
    """Derive trusted current-skill identities for AS3 self-reference suppression.

    Host-derived scan-root basenames and selected repository/archive identities
    are authoritative. Contributor-controlled ``manifest.name`` may only
    corroborate those trusted identities; it never introduces a suppression
    identity on its own. Ephemeral temp-clone basenames such as ``repo`` are
    ignored so a matching selected-source identity can still suppress the real
    skill self-path without opening a peer-skill false negative.
    """

    skill_path: object = state.get("skill_path")
    path_text: str | bytes | None = None
    if isinstance(skill_path, str):
        path_text = skill_path
    elif isinstance(skill_path, os.PathLike):
        try:
            path_text = os.fspath(skill_path)
        except Exception:
            path_text = None
    path_identifier: str | None = None
    if isinstance(path_text, str):
        normalized_path = path_text.replace("\\", "/").rstrip("/")
        path_identifier = _normalize_skill_identifier(normalized_path.rsplit("/", 1)[-1])

    source_identifier = _selected_source_identifier(state)

    manifest = state.get("manifest")
    manifest_identifier: str | None = None
    if isinstance(manifest, Mapping):
        manifest_identifier = _normalize_skill_identifier(manifest.get("name"))

    identifiers: set[str] = set()
    if path_identifier is not None and not _is_ephemeral_scan_root_basename(path_identifier):
        identifiers.add(path_identifier)
    if source_identifier is not None:
        identifiers.add(source_identifier)

    # Manifest data is contributor-controlled. Keep it only when it already
    # matches a trusted host/operator identity (no-op add) so mismatched names
    # such as ``name: victim`` cannot suppress peer ``skills/victim/SKILL.md``.
    if manifest_identifier is not None and manifest_identifier in identifiers:
        identifiers.add(manifest_identifier)

    return frozenset(identifiers)


def _is_current_skill_path_reference(
    matched_text: object, current_skill_identifiers: frozenset[str]
) -> bool:
    """Return whether an exact AS3 path match names the current skill."""
    if not current_skill_identifiers or not isinstance(matched_text, str):
        return False
    match = _AS3_SKILL_PATH_FULLMATCH.fullmatch(matched_text)
    if match is None:
        return False
    matched_identifier = _normalize_skill_identifier(match.group("skill_name"))
    return matched_identifier in current_skill_identifiers


class _CurrentSkillScopedAnalyzer:
    """Delegate AS checks while excluding exact current-skill path references."""

    ANALYZER_ID = ANALYZER_ID

    def __init__(self, current_skill_identifiers: frozenset[str]) -> None:
        self._current_skill_identifiers = current_skill_identifiers

    def analyze(self, content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
        token = _CURRENT_SKILL_IDENTIFIERS.set(self._current_skill_identifiers)
        try:
            return analyze(content, file_path, file_type)
        finally:
            _CURRENT_SKILL_IDENTIFIERS.reset(token)


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run agent_snooping patterns and return findings."""
    analyzer = _CurrentSkillScopedAnalyzer(_current_skill_identifiers(state))
    response = static_runner.run_static_patterns_with_ledger(state, [analyzer])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
