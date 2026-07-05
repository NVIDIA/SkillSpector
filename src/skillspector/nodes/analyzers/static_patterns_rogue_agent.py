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

"""Static patterns: rogue agent (RA1–RA2). Node and analyze() in one module.

Detects patterns where a skill modifies its own code or configuration
at runtime (RA1) or establishes unauthorized persistence across session
boundaries (RA2).

Framework: ASI10.
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

ANALYZER_ID = "static_patterns_rogue_agent"

# RA1: Self-Modification — skill modifies its own code, config, or behavior
RA1_PATTERNS = [
    # Python: writing to own file
    (r"open\s*\(\s*__file__\s*,\s*['\"]w", 0.95),
    (r"(?:Path|pathlib)\s*\(\s*__file__\s*\)\s*\.\s*write_text", 0.95),
    (
        r"(?:write|modify|edit|update|overwrite|patch)\s+(?:this\s+)?(?:skill(?:'s)?|SKILL\.md|skill\.md)",
        0.85,
    ),
    (
        r"(?:modify|edit|change|rewrite|update|alter)\s+(?:your\s+own|its\s+own|this\s+skill(?:'s)?)\s+(?:code|source|file|script|config|configuration|instructions?|rules?)",
        0.9,
    ),
    (r"(?:self[_-]?modify|self[_-]?update|self[_-]?rewrite|self[_-]?patch|self[_-]?evolve)", 0.9),
    (
        r"(?:rewrite|replace|overwrite)\s+(?:the\s+)?(?:current|existing|original)\s+(?:code|script|file|config|skill)",
        0.85,
    ),
    (
        r"(?:append|add|inject|insert)\s+(?:code|instructions?|rules?)\s+(?:to|into)\s+(?:this|the\s+current|your\s+own)\s+(?:file|skill|script)",
        0.85,
    ),
    # Dynamic code generation that modifies self
    (
        r"(?:generate|create|write)\s+(?:new|updated|replacement)\s+(?:code|version)\s+(?:for|of)\s+(?:this|itself|the\s+skill)",
        0.8,
    ),
    (r"open\s*\(\s*['\"](?:SKILL\.md|skill\.md)['\"].*?['\"]w", 0.9),
    # Modifying own config at runtime
    (
        r"(?:update|modify|change|edit)\s+(?:the\s+)?(?:config|configuration|settings|parameters)\s+(?:of\s+)?(?:this|the\s+current)\s+(?:skill|agent|tool)",
        0.75,
    ),
    # Disabling safety at runtime
    (
        r"(?:disable|remove|delete|bypass)\s+(?:the\s+)?(?:safety|security|guard|protection|constraint)\s+(?:check|rule|mechanism|feature)",
        0.9,
    ),
    (
        r"(?:comment\s+out|remove)\s+(?:the\s+)?(?:validation|verification|check|guard|assertion)",
        0.7,
    ),
]

# RA2: Session Persistence — unauthorized persistence across boundaries
RA2_PATTERNS = [
    # Cron jobs and scheduled tasks
    (r"crontab\s+(?:-[el]|.*?>>?\s*/)", 0.85),
    (
        r"(?:add|create|install|register)\s+(?:a\s+)?(?:cron\s+)?(?:job|task|entry)\s+(?:for|to|that)",
        0.8,
    ),
    (
        r"(?:schedule|register)\s+(?:a\s+)?(?:recurring|periodic|repeated|daily|hourly)\s+(?:task|job|command|script)",
        0.8,
    ),
    # Startup scripts and autostart
    (
        r"(?:add|write|append|install)\s+(?:to|into)\s+(?:~/)?\.(?:bashrc|zshrc|profile|bash_profile|login|cshrc)",
        0.9,
    ),
    (
        r"(?:add|register|install)\s+(?:as\s+)?(?:a\s+)?(?:startup|boot|autostart|login)\s+(?:script|service|daemon|task|item)",
        0.85,
    ),
    (
        r"(?:systemd|systemctl|launchd|launchctl|init\.d)\s+.*?(?:enable|install|register|create)",
        0.8,
    ),
    (
        r"(?:create|install|register|add)\s+(?:a\s+)?(?:systemd\s+)?(?:service|daemon|agent)\s+(?:file|unit)",
        0.8,
    ),
    # Persistent state files
    (
        r"(?:save|write|persist|store|dump)\s+(?:the\s+)?(?:state|session|context|data)\s+(?:to|in)\s+(?:a\s+)?(?:file|disk|database|storage)",
        0.6,
    ),
    (
        r"(?:restore|load|read|resume)\s+(?:the\s+)?(?:previous|saved|persisted|stored)\s+(?:state|session|context|data)",
        0.55,
    ),
    (
        r"(?:persist|maintain|keep|preserve)\s+(?:state|data|context|session)\s+(?:across|between|through)\s+(?:sessions?|restarts?|reboots?|invocations?)",
        0.75,
    ),
    # Hidden files and directories for persistence
    (r"(?:create|write|mkdir)\s+[^|]*(?:~/|/home/|/tmp/)\.(?!git|ssh|aws)[a-z_-]+", 0.6),
    (r"(?:create|make|write)\s+(?:a\s+)?(?:hidden|dot)\s+(?:file|directory|folder)", 0.65),
    # Background processes
    (r"(?:nohup|disown|setsid)\s+", 0.65),
    (
        r"(?:start|launch|spawn|fork)\s+(?:a\s+)?(?:background|daemon|detached)\s+(?:process|service|worker|task)",
        0.7,
    ),
    (r"(?:run|execute)\s+(?:in\s+the\s+)?background\s+(?:and\s+)?(?:detach|persist|survive)", 0.75),
    # Registry / plist for Windows/macOS persistence
    (r"(?:HKEY_|RegOpenKey|RegSetValue|reg\s+add)\s+", 0.8),
    (r"(?:defaults\s+write|plist|launchctl\s+load)", 0.75),
]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for rogue agent patterns (RA1–RA2)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.ROGUE_AGENT.value]

    for pattern, confidence in RA1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = ctx(match.start())
            if _is_negated_safety_constraint(context, match.group(0)):
                continue
            findings.append(
                AnalyzerFinding(
                    rule_id="RA1",
                    message="Self-Modification",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context,
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in RA2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="RA2",
                    message="Session Persistence",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def _is_negated_safety_constraint(context: str, matched_text: str) -> bool:
    """Return True when an RA1 phrase is explicitly forbidden in policy prose."""
    escaped = re.escape(matched_text.strip())
    if not escaped:
        return False
    negation = r"(?:must\s+not|do\s+not|don't|never|should\s+not)\s+"
    return re.search(negation + r"(?:\w+\s+){0,4}" + escaped, context, re.IGNORECASE) is not None


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run rogue_agent patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
