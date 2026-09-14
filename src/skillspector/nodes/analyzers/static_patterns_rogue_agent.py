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

_VERIFY_SIGNATURE_FLAG = re.compile(
    r"--verify-signature(?:=(?P<value>1|on|true|yes))?",
    re.IGNORECASE,
)
_DISABLE_SIGNATURE_VERIFICATION = re.compile(
    r"(?<!\S)--no-verify-signature(?=\s|$)"
    r"|(?<!\S)--verify-signature(?:=|\s+)(?:0|false|no|off)(?=\s|$)",
    re.IGNORECASE,
)
_SIGNED_CLI_RELEASE = re.compile(r"\bsigned\s+(?:cli\s+)?release\b", re.IGNORECASE)
_SIGNED_CLI_RELEASE_NEGATION = re.compile(
    r"\b(?:not|without)\s+(?:an?\s+)?signed\s+(?:cli\s+)?release\b|"
    r"\bexcept\s+(?:for\s+)?(?:an?\s+)?signed\s+(?:cli\s+)?release\b|"
    r"\b(?:never|do\s+not|don't)\b[^\n]{0,40}\bsigned\s+(?:cli\s+)?release\b",
    re.IGNORECASE,
)
_SHELL_COMMAND_COMPOSITION = re.compile(r"(?:&&|\|\||[;&|#<>]|\$\(|<\(|>\()")
_COMPANION_CLI_BEFORE_SELF_UPDATE = re.compile(
    r"(?<![\w.-])(?P<cli>[a-z0-9][\w.-]{1,63})\s+$",
    re.IGNORECASE,
)
_NON_COMPANION_UPDATE_SUBJECTS = frozenset(
    {
        "agent",
        "assistant",
        "bash",
        "cli",
        "cmd",
        "command",
        "env",
        "exec",
        "fish",
        "itself",
        "node",
        "perl",
        "powershell",
        "pwsh",
        "python",
        "ruby",
        "self",
        "sh",
        "skill",
        "skillspector",
        "sudo",
        "tool",
        "zsh",
    }
)

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
            if _is_negated_safety_constraint(content, match):
                continue
            companion_update = _is_signed_companion_cli_update(content, match, file_type)
            finding_tags = list(tag)
            if companion_update:
                finding_tags.extend(["contextual-triage", "likely-benign-context"])
            findings.append(
                AnalyzerFinding(
                    rule_id="RA1",
                    message=(
                        "Signed Companion CLI Update" if companion_update else "Self-Modification"
                    ),
                    severity=Severity.LOW if companion_update else Severity.HIGH,
                    location=loc(line_num),
                    confidence=min(confidence, 0.15) if companion_update else confidence,
                    remediation=(
                        "No skill self-modification change is indicated by this match. Keep "
                        "signature verification mandatory and identify the companion CLI "
                        "explicitly."
                        if companion_update
                        else None
                    ),
                    explanation=(
                        "The matched phrase is a signed self-update subcommand for a documented "
                        "companion CLI; it does not direct the skill or agent to rewrite itself."
                        if companion_update
                        else None
                    ),
                    tags=finding_tags,
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


def _is_signed_companion_cli_update(
    content: str,
    match: re.Match[str],
    file_type: str,
) -> bool:
    """Identify a signed update subcommand whose subject is another CLI."""
    if file_type not in {"markdown", "text"} or match.group(0).lower() != "self-update":
        return False

    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    if line_end == -1:
        line_end = len(content)
    line = content[line_start:line_end]
    if line.count("`") != 2:
        return False
    local_start = match.start() - line_start
    local_end = match.end() - line_start
    code_start = line.rfind("`", 0, local_start)
    code_end = line.find("`", local_end)
    if code_start == -1 or code_end == -1:
        return False
    command = line[code_start + 1 : code_end]
    if _SHELL_COMMAND_COMPOSITION.search(command):
        return False
    command_match_start = local_start - code_start - 1
    command_match_end = local_end - code_start - 1
    cli_match = _COMPANION_CLI_BEFORE_SELF_UPDATE.fullmatch(command[:command_match_start])
    if cli_match is None:
        return False
    cli_subject = cli_match.group("cli").lower().removesuffix(".exe")
    if cli_subject in _NON_COMPANION_UPDATE_SUBJECTS or cli_subject.startswith("python"):
        return False
    if _DISABLE_SIGNATURE_VERIFICATION.search(command):
        return False
    flag = _VERIFY_SIGNATURE_FLAG.fullmatch(command[command_match_end:].strip())
    if flag is None:
        return False
    flag_value = (flag.group("value") or "true").lower()
    return (
        flag_value in {"1", "on", "true", "yes"}
        and _SIGNED_CLI_RELEASE.search(line) is not None
        and _SIGNED_CLI_RELEASE_NEGATION.search(line) is None
    )


def _is_negated_safety_constraint(content: str, match: re.Match[str]) -> bool:
    """Return True when an RA1 phrase is explicitly forbidden in policy prose."""
    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    if line_end == -1:
        line_end = len(content)
    line = content[line_start:line_end]
    local_start = match.start() - line_start
    phrase = line[local_start : local_start + len(match.group(0))]
    escaped = re.escape(phrase.strip())
    if not escaped:
        return False
    clause_start = max(line.rfind(sep, 0, local_start) for sep in ".;:")
    prefix = line[clause_start + 1 : local_start]
    safe_gap = r"(?:(?:ever|again|directly|intentionally|explicitly|attempt\s+to|try\s+to)\s+){0,2}"
    negation = r"(?:must\s+not|do\s+not|don't|never|should\s+not)\s+"
    return (
        re.search(negation + safe_gap + escaped + r"$", prefix + phrase, re.IGNORECASE) is not None
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run rogue_agent patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
