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

import re
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_agent_snooping"

# AS1: Agent Config Directory Access
# Matches code/instructions that read from well-known agent config directories.
AS1_PATTERNS = [
    # Direct filesystem access to .claude/, .codex/, .gemini/ directories
    (r"open\s*\(\s*['\"]?\.(?:claude|codex|gemini|continue)/", 0.9),
    (r"(?:Path|pathlib\.Path)\s*\(\s*['\"]?\.(?:claude|codex|gemini|continue)/", 0.9),
    (r"os\.path\.(?:join|exists|isfile)\s*\(\s*['\"]?\.(?:claude|codex|gemini|continue)", 0.85),
    (
        r"(?:read|load|open|access|fetch)\s+(?:the\s+)?(?:agent|claude|codex|gemini)\s+(?:config|configuration|settings?)\s+(?:from|at|in)\s+~?/?\.(?:claude|codex|gemini)",
        0.9,
    ),
    # Shell commands targeting config dirs
    (r"(?:cat|less|head|tail|grep|find)\s+[^|&;\n]*~?/?\.(claude|codex|gemini)/", 0.85),
    # Home-directory config paths
    (
        r"~?/\.(?:claude|codex|gemini|continue)/(?:config|settings?|preferences?|credentials?)(?:\.(?:json|yaml|yml|toml))?",
        0.9,
    ),
    # Generic "agent config" snooping instructions
    (
        r"(?:read|access|inspect|examine|retrieve)\s+(?:the\s+)?(?:agent|assistant)\s+(?:config(?:uration)?|settings?|preferences?)\s+(?:file|directory|folder)",
        0.8,
    ),
]

# AS2: MCP Config Access
# Matches code/instructions specifically targeting MCP server configuration.
AS2_PATTERNS = [
    # Direct access to mcp.json files
    (r"open\s*\(\s*['\"][^'\"]*mcp(?:_config)?\.json['\"]", 0.95),
    (r"(?:Path|pathlib\.Path)\s*\(\s*['\"][^'\"]*mcp(?:_config)?\.json['\"]", 0.9),
    (r"(?:cat|less|head|grep)\s+[^|&;\n]*mcp(?:_config)?\.json", 0.9),
    # Instructions referencing MCP config
    (
        r"(?:read|access|load|inspect)\s+(?:the\s+)?mcp(?:\.json|_config)?\s+(?:file|config(?:uration)?|settings?)",
        0.9,
    ),
    (r"\.(?:claude|codex|gemini)/mcp(?:_config)?\.json", 0.95),
    # Listing MCP servers
    (
        r"(?:list|enumerate|discover)\s+(?:all\s+)?(?:available\s+)?mcp\s+(?:servers?|tools?|services?)",
        0.8,
    ),
    # Accessing MCP server URLs or API keys from config
    (r"mcp(?:_config)?\.json.*?(?:api_?key|token|secret|url|endpoint)", 0.9),
]

# AS3: Skill Enumeration / Snooping
# Matches code/instructions that enumerate or read other installed skills.
AS3_PATTERNS = [
    # Listing skill directories
    (
        r"(?:os\.listdir|os\.scandir|glob\.glob|Path\.iterdir)\s*\([^)]*\.(?:claude|codex|gemini)/skills?",
        0.9,
    ),
    (r"(?:ls|find|dir)\s+[^|&;\n]*\.(?:claude|codex|gemini)/skills?", 0.85),
    # Reading other skills' SKILL.md files
    (r"open\s*\(\s*['\"][^'\"]*SKILL\.md['\"].*?\bother\b", 0.85),
    (
        r"(?:read|access|inspect|enumerate)\s+(?:all\s+)?(?:installed|available|other)\s+skills?(?:\s+in\s+(?:the\s+)?(?:skills?|agent)\s+(?:directory|folder))?",
        0.85,
    ),
    # Instructions to discover other skills
    (
        r"(?:list|discover|find|enumerate|identify)\s+(?:all\s+)?(?:other|installed|available)\s+(?:skills?|agents?|tools?)\s+(?:in\s+)?(?:the\s+)?(?:\.(?:claude|codex|gemini)|\$HOME)",
        0.85,
    ),
    # Accessing skills/CURRENT or adjacent skill directories
    (r"skills?/(?:(?!CURRENT)[A-Z][A-Za-z0-9_-]+)/SKILL\.md", 0.8),
    # Reading tool manifests of other agents
    (
        r"(?:read|access|load)\s+(?:the\s+)?(?:SKILL|skill)\.md\s+(?:file\s+)?(?:of|from|for)\s+(?:another|other|different|all)\s+(?:skill|agent|tool)",
        0.9,
    ),
]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for agent snooping patterns (AS1–AS3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.AGENT_SNOOPING.value]

    for pattern, confidence in AS1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                )
            )

    for pattern, confidence in AS2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                )
            )

    for pattern, confidence in AS3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                    matched_text=match.group(0)[:200],
                )
            )

    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run agent_snooping patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
