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

"""Static patterns: excessive agency (EA1–EA5). Node and analyze() in one module.

Detects patterns where an agent skill grants unrestricted tool access (EA1),
enables autonomous high-impact decisions without human-in-the-loop (EA2),
exhibits scope creep beyond stated purpose (EA3), or allows unbounded
resource consumption (EA4), or selects an external model/provider with billing
implications (EA5).

Framework: LLM06, ASI02.
"""

from __future__ import annotations

import re
import shlex
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_excessive_agency"

# EA1: Unrestricted Tool Access
EA1_PATTERNS = [
    (r"(?:tools?|permissions?)\s*:[ \t]*\[?[ \t]*['\"]?\*(?!\*|\w)['\"]?[ \t]*\]?", 0.85),
    (r"(?:allow|grant|enable)\s+(?:access\s+to\s+)?(?:all|any|every)\s+tools?", 0.8),
    (
        r"(?:no|without)\s+(?:tool|permission|access|capability)\s+(?:restrictions?|constraints?|limitations?)",
        0.75,
    ),
    (r"(?:call|invoke|use|execute)\s+(?:any|all|every)\s+(?:available\s+)?tools?", 0.8),
    (
        r"(?:unrestricted|unlimited|unconstrained)\s+(?:tool|function|api)\s+(?:access|use|calls?)",
        0.85,
    ),
    (
        r"(?:can|may|should)\s+(?:freely|always)\s+(?:use|call|invoke)\s+(?:any|all)\s+(?:tools?|functions?|apis?)",
        0.75,
    ),
    (r"tools?\s*:\s*\[\s*['\"]shell['\"].*?['\"](?:file_write|network|http)['\"]", 0.7),
    (r"(?:grant|give)\s+(?:full|complete|total)\s+(?:tool|function|api)\s+access", 0.85),
    (r"(?:execute|run)\s+(?:arbitrary|any)\s+(?:commands?|code|scripts?)", 0.8),
    (
        r"(?:no\s+)?(?:tool\s+)?(?:allow|block|deny)\s*(?:list|listing)\s*(?:is\s+)?(?:empty|disabled|off)",
        0.7,
    ),
]

# EA2: Autonomous Decision Making
EA2_PATTERNS = [
    (r"(?:without|no)\s+(?:asking|confirmation|approval|consent|verification|checking)", 0.75),
    (
        r"(?:auto(?:matically)?|autonomously)\s+(?:execute|run|perform|delete|remove|modify|send|deploy)",
        0.8,
    ),
    (
        r"(?:skip|bypass|disable)\s+(?:user\s+)?(?:confirmation|approval|consent|verification|prompt)",
        0.85,
    ),
    (
        r"(?:do\s+not|don't|never)\s+(?:ask|prompt|confirm|verify|check)\s+(?:the\s+)?(?:user|before)",
        0.8,
    ),
    (r"(?:auto[_-]?approve|auto[_-]?confirm|auto[_-]?execute|auto[_-]?deploy)", 0.85),
    (
        r"(?:proceed|continue|execute)\s+without\s+(?:waiting|asking)\s+(?:for\s+)?(?:user|human|permission)",
        0.8,
    ),
    (r"(?:assume|presume)\s+(?:user\s+)?(?:consent|approval|permission)", 0.7),
    (
        r"(?:if|when)\s+(?:unsure|uncertain).*?(?:proceed|continue|execute)\s+(?:anyway|regardless)",
        0.75,
    ),
    (r"(?:--auto-?approve|--no-?confirm)\b", 0.65),
]

# EA3: Scope Creep
EA3_PATTERNS = [
    (
        r"(?:also|additionally|furthermore)\s+(?:perform|execute|run|do|handle|manage)\s+(?:any|all|other)",
        0.65,
    ),
    (
        r"(?:while\s+you(?:'re|\s+are)\s+at\s+it|in\s+addition|on\s+top\s+of\s+that)\s*[,.]?\s*(?:also\s+)?(?:do|perform|execute|run)",
        0.7,
    ),
    (
        r"(?:extend|expand|broaden)\s+(?:your|the\s+)?(?:scope|functionality|capabilities|responsibilities)",
        0.75,
    ),
    (r"(?:not\s+limited\s+to|beyond\s+(?:the\s+)?(?:scope|stated|described|documented))", 0.7),
    (
        r"(?:take\s+over|assume\s+control\s+of|manage)\s+(?:all|any|every)\s+(?:aspect|part|area)",
        0.75,
    ),
    (
        r"(?:you\s+(?:can|should|must)\s+)?(?:handle|manage)\s+(?:everything|anything|all\s+tasks?)",
        0.7,
    ),
    (
        r"(?:act\s+as|become|serve\s+as)\s+(?:a\s+)?(?:general[- ]purpose|universal|all[- ]in[- ]one|omniscient)",
        0.65,
    ),
    (
        r"(?:you\s+are\s+)?(?:responsible\s+for|in\s+charge\s+of)\s+(?:everything|all\s+(?:systems?|operations?|tasks?))",
        0.7,
    ),
]

# EA4: Unbounded Resource Access
EA4_PATTERNS = [
    (
        r"(?:unlimited|infinite|unbounded|no\s+limit(?:s)?(?:\s+on)?)\s+(?:api\s+)?(?:calls?|requests?|queries?|invocations?)",
        0.8,
    ),
    (
        r"(?:no|without)\s+(?:rate\s+)?limit(?:s|ing)?\s+(?:on|for|when)\s+(?:api|tool|request|query)",
        0.7,
    ),
    (
        r"(?:no|without)\s+(?:timeout|budget|quota|cap|ceiling)\s+(?:on|for|when)\s+(?:api|tool|request|execution)",
        0.7,
    ),
    (r"(?:loop|iterate|repeat)\s+(?:indefinitely|forever|infinitely|endlessly)", 0.75),
    (r"(?:retry|attempt)\s+(?:indefinitely|forever|without\s+limit|unlimited\s+times)", 0.75),
    (r"max[_-]?retries?\s*=\s*(?:None|0|float\s*\(\s*['\"]inf['\"]|math\.inf|infinity)", 0.8),
    (r"timeout\s*=\s*(?:None|0|float\s*\(\s*['\"]inf['\"]|math\.inf)", 0.75),
    (
        r"(?:allocate|consume|use)\s+(?:as\s+much|unlimited|unbounded)\s+(?:memory|storage|disk|compute|cpu|gpu)",
        0.8,
    ),
    (
        r"(?:no|without)\s+(?:resource\s+)?(?:constraints?|limits?|quotas?|budgets?)\s+(?:on|for|when)\s+(?:api|tool|execution|request|compute)",
        0.7,
    ),
]

# EA5: External Model or Provider Selection
_EA5_FRONTMATTER_KEY = re.compile(
    r"^[\"']?(?P<key>model|provider|model_name|model_id)[\"']?[ \t]*:[ \t]*"
    r"(?P<value>[^#\s][^#\r\n]*)",
    re.IGNORECASE | re.MULTILINE,
)
_EA5_INLINE_CODE = re.compile(r"`(?P<command>[^`\r\n]+)`")
_EA5_IMPERATIVE_PREFIX = re.compile(
    r"^(?:run|execute|invoke|call)(?:[ \t]+the)?(?:[ \t]+command)?[ \t]*:?[ \t]+",
    re.IGNORECASE,
)
_EA5_INLINE_DIRECTIVE_PREFIX = re.compile(
    r"^(?:(?:[-*+]|\d+[.)])[ \t]+)?"
    r"(?:run|execute|invoke|call)(?:[ \t]+the)?(?:[ \t]+command)?[ \t]*:?[ \t]*$",
    re.IGNORECASE,
)
_EA5_FENCE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})[ \t]*(?P<language>[\w+-]*)")
_EA5_SHELL_FENCE_LANGUAGES = {"bash", "console", "sh", "shell", "zsh"}
_EA5_MODEL_VALUE = re.compile(
    r"^(?:claude|gpt|gemini|deepseek|kimi|glm|minimax|mistral|llama)"
    r"(?:$|[-_./:0-9])",
    re.IGNORECASE,
)


def _frontmatter_bounds(content: str, file_path: str) -> tuple[int, int] | None:
    """Return the YAML-frontmatter byte offsets for a SKILL.md file."""
    if file_path.rsplit("/", 1)[-1].lower() != "skill.md":
        return None
    opening = re.match(r"\A---[ \t]*\r?\n", content)
    if opening is None:
        return None
    closing = re.search(r"^---[ \t]*$", content[opening.end() :], re.MULTILINE)
    if closing is None:
        return None
    return opening.end(), opening.end() + closing.start()


def _is_model_switch_command(command: str) -> bool:
    """Return whether a shell command selects another coding model/provider."""
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return False

    executable = tokens[0].rsplit("/", 1)[-1].lower()
    if executable == "claude" and any(
        token in {"-p", "--print"} or token.startswith("--print=") for token in tokens[1:]
    ):
        return True
    if executable == "codex" and len(tokens) > 1 and tokens[1].lower() == "exec":
        return True

    if executable.startswith("python") or executable in {"node", "perl", "ruby"}:
        return False

    for index, token in enumerate(tokens[1:], start=1):
        value: str | None = None
        if token in {"-m", "--model"} and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith(("-m=", "--model=")):
            value = token.split("=", 1)[1]
        if value and _EA5_MODEL_VALUE.match(value):
            return True
    return False


def _command_span(line: str) -> tuple[int, int] | None:
    """Return the model-switch command span for an actionable instruction line."""
    # Inline code is handled separately so surrounding prose and punctuation do
    # not become part of the command span (or create a duplicate finding).
    if "`" in line:
        return None

    leading = len(line) - len(line.lstrip(" \t"))
    candidate = line[leading:]
    if candidate.startswith(("$", ">")):
        prompt_width = 1 + len(candidate[1:]) - len(candidate[1:].lstrip(" \t"))
        leading += prompt_width
        candidate = candidate[prompt_width:]

    command = candidate.strip().strip("`")
    if _is_model_switch_command(command):
        start = line.find(command, leading)
        return start, start + len(command)

    imperative = _EA5_IMPERATIVE_PREFIX.match(candidate)
    if imperative is not None:
        command = candidate[imperative.end() :].strip().strip("`")
        if _is_model_switch_command(command):
            start = line.find(command, leading + imperative.end())
            return start, start + len(command)
    return None


def _inline_command_span(line: str, inline: re.Match[str]) -> tuple[int, int] | None:
    """Return an inline model-switch command only when prose directs its execution."""
    if _EA5_INLINE_DIRECTIVE_PREFIX.match(line[: inline.start()]) is None:
        return None
    command = inline.group("command").strip()
    if not _is_model_switch_command(command):
        return None
    command_offset = (
        inline.start("command")
        + len(inline.group("command"))
        - len(inline.group("command").lstrip())
    )
    return command_offset, command_offset + len(command)


def _ea5_findings(content: str, file_path: str) -> list[AnalyzerFinding]:
    """Detect declarative model pins and actionable coding-CLI model switches."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.EXCESSIVE_AGENCY.value]
    bounds = _frontmatter_bounds(content, file_path)
    body_start = 0
    if bounds is not None:
        start, end = bounds
        body_start = end
        frontmatter = content[start:end]
        for match in _EA5_FRONTMATTER_KEY.finditer(frontmatter):
            value = match.group("value").strip().lower()
            if value in {'""', "''", "~", "null", "none", "default", "auto", "inherit"}:
                continue
            absolute_start = start + match.start()
            key = match.group("key").lower()
            findings.append(
                AnalyzerFinding(
                    rule_id="EA5",
                    message="External Model or Provider Selection",
                    severity=Severity.MEDIUM,
                    location=Location(
                        file=file_path,
                        start_line=get_line_number(content, absolute_start),
                    ),
                    confidence=0.9,
                    tags=tag,
                    context=get_context(content, absolute_start),
                    matched_text=match.group(0)[:200],
                    evidence={"selection_surface": "frontmatter", "selection_key": key},
                )
            )

    seen: set[tuple[int, int]] = set()
    cursor = body_start
    fence_marker: str | None = None
    fence_language = ""
    for line in content[body_start:].splitlines(keepends=True):
        line_text = line.rstrip("\r\n")
        fence = _EA5_FENCE.match(line_text)
        if fence is not None:
            marker = fence.group("marker")
            if fence_marker is None:
                fence_marker = marker[0]
                fence_language = fence.group("language").lower()
            elif marker[0] == fence_marker:
                fence_marker = None
                fence_language = ""
            cursor += len(line)
            continue

        candidates: list[tuple[int, int]] = []
        if (
            fence_marker is None
            or fence_language in _EA5_SHELL_FENCE_LANGUAGES
            or not fence_language
        ):
            direct = _command_span(line_text)
            if direct is not None:
                candidates.append(direct)
        for inline in _EA5_INLINE_CODE.finditer(line_text):
            command_span = _inline_command_span(line_text, inline)
            if command_span is not None:
                candidates.append(command_span)

        for line_start, line_end in candidates:
            absolute = (cursor + line_start, cursor + line_end)
            if absolute in seen:
                continue
            seen.add(absolute)
            findings.append(
                AnalyzerFinding(
                    rule_id="EA5",
                    message="External Model or Provider Selection",
                    severity=Severity.HIGH,
                    location=Location(
                        file=file_path,
                        start_line=get_line_number(content, absolute[0]),
                    ),
                    confidence=0.9,
                    tags=tag,
                    context=get_context(content, absolute[0]),
                    matched_text=content[absolute[0] : absolute[1]][:200],
                    evidence={"selection_surface": "command"},
                )
            )
        cursor += len(line)
    return findings


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for excessive agency patterns (EA1–EA5)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.EXCESSIVE_AGENCY.value]

    for pattern, confidence in EA1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA1",
                    message="Unrestricted Tool Access",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in EA2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context_text = ctx(match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA2",
                    message="Autonomous Decision Making",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context_text,
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in EA3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA3",
                    message="Scope Creep",
                    severity=Severity.LOW,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in EA4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA4",
                    message="Unbounded Resource Access",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    findings.extend(_ea5_findings(content, file_path))
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run excessive_agency patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
