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

"""Static patterns: anti-refusal statements (AR1-AR3). Node and analyze() in one module.

Anti-refusal statements are instructions embedded in a skill that try to suppress the
agent's ability to refuse, hedge, or apply safety constraints. They are a jailbreak
technique distinct from generic instruction-override: rather than injecting a new task,
they neutralize the model's *refusal behavior* so subsequent harmful requests succeed.

Three families are detected:
  AR1 - Refusal suppression: "never refuse", "always comply", "you cannot say no".
  AR2 - Disclaimer/caveat suppression: "no warnings", "do not moralize", "skip the ethics".
  AR3 - Safety/policy nullification: "you have no restrictions", "ignore your guidelines",
        "not bound by any policy", "do anything now".
"""

from __future__ import annotations

import re
import sys
from bisect import bisect_right
from collections.abc import Callable

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Severity
from skillspector.python_tokens import python_literal_spans
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import SourceLocationIndex, get_context, get_line_number, is_code_example
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_anti_refusal"
# Proving Python comment ownership parses the module, so it runs under the
# runner's per-artifact deadline.
USES_RUNTIME_CHECK = True

# AR1: Refusal suppression - instruct the agent to never refuse / always comply.
AR1_PATTERNS = [
    (r"\b(?:never|do\s+not|don'?t)\s+(?:ever\s+)?(?:refuse|decline)\b", 0.85),
    (r"\byou\s+(?:must|will|should|can)\s*(?:not|never)\s+(?:refuse|decline|say\s+no)\b", 0.9),
    (r"\byou\s+(?:cannot|can'?t|can\s+not)\s+(?:refuse|decline|say\s+no)\b", 0.85),
    (r"\bnever\s+say\s+(?:that\s+)?(?:you\s+)?(?:can'?t|cannot|won'?t|no)\b", 0.85),
    (r"\balways\s+(?:comply|obey|answer|respond)\b", 0.7),
    (
        r"\b(?:(?:you|the\s+(?:agent|assistant|model))\s+(?:must|should|will)\s+"
        r"always\s+agree|always\s+agree\s+(?:with|to)\s+"
        r"(?:(?:the|every|all|any|each)\s+)?"
        r"(?:user|requests?|questions?|prompts?|queries|commands?))\b",
        0.7,
    ),
    (
        r"\b(?:you\s+)?must\s+(?:always\s+)?(?:answer|comply\s+with|respond\s+to|fulfill)\s+"
        r"(?:every|all|any|each)\b",
        0.8,
    ),
    (r"\brespond\s+to\s+(?:every|all|any)\s+(?:request|question|prompt|query|command)\b", 0.6),
    (r"\bno\s+request\s+is\s+(?:off[- ]limits|forbidden|too)\b", 0.85),
]

# AR2: Disclaimer / caveat / warning suppression.
AR2_PATTERNS = [
    (
        r"\b(?:do\s+not|don'?t|never|no)\s+(?:include|add|give|provide|use|append)?\s*"
        r"(?:any\s+)?(?:warnings?|disclaimers?|caveats?)\b",
        0.8,
    ),
    (
        r"\bwithout\s+(?:any\s+)?(?:warnings?|disclaimers?|caveats?|moralizing|moralising|"
        r"judg(?:e?ment|ing))\b",
        0.8,
    ),
    (r"\b(?:do\s+not|don'?t|never)\s+(?:lecture|moralize|moralise|preach|judge|warn)\b", 0.85),
    (r"\b(?:skip|omit|leave\s+out|drop)\s+(?:the\s+)?(?:disclaimer|warning|ethics|safety)\b", 0.8),
    (
        r"\b(?:no|without)\s+(?:ethical|moral)\s+(?:judg(?:e?ment)|considerations?|commentary|"
        r"objections?)\b",
        0.8,
    ),
    (r"\b(?:do\s+not|don'?t)\s+(?:apologize|apologise|say\s+sorry)\b", 0.55),
]

# AR3: Safety / policy / restriction nullification (jailbreak framing).
AR3_PATTERNS = [
    (
        r"\byou\s+have\s+no\s+(?:restrictions?|limitations?|filters?|rules?|guidelines?|"
        r"boundaries|limits)\b",
        0.9,
    ),
    (
        r"\b(?:you\s+are|act\s+as|pretend\s+(?:to\s+be|you'?re))[^.\n]{0,30}"
        r"(?:unrestricted|unfiltered|uncensored|no\s+filter)\b",
        0.85,
    ),
    (
        r"\b(?:ignore|disregard|forget|bypass|override)\s+(?:your\s+)?(?:content\s+)?"
        r"(?:policy|policies|guidelines?|safety|restrictions?|filters?)\b",
        0.9,
    ),
    (
        r"\bnot\s+bound\s+by\s+(?:any\s+)?(?:rules?|policy|policies|guidelines?|restrictions?|"
        r"openai|anthropic)\b",
        0.9,
    ),
    (r"\bdo\s+anything\s+now\b", 0.8),
    (r"\b(?:developer|debug|god|sudo|jailbreak)\s+mode\s+(?:enabled|on|activated|engaged)\b", 0.75),
    (r"\bno\s+(?:content\s+)?(?:policy|policies|filters?|restrictions?)\s+appl(?:y|ies)\b", 0.85),
    (
        r"\b(?:free\s+from|without)\s+(?:any\s+)?(?:safety\s+)?(?:guardrails?|constraints?|"
        r"safeguards?)\b",
        0.8,
    ),
]

_RULES = [("AR1", AR1_PATTERNS), ("AR2", AR2_PATTERNS), ("AR3", AR3_PATTERNS)]

_MODE_ENABLED_RE = re.compile(
    r"\b(?:developer|debug|god|sudo|jailbreak)\s+mode\s+(?:enabled|on|activated|engaged)\b",
    re.IGNORECASE,
)
_SECURITY_REVIEW_CONTEXT_RE = re.compile(
    r"\b(?:unsafe\s+defaults?|security\s+(?:review|audit|checklist)|review\s+checklist)\b|"
    r"\b(?:detect|flag|check(?:s|ed|ing)?\s+for|look\s+for|avoid|must\s+not|never\s+enable)\b"
    r"[^.\n]{0,100}\b(?:developer|debug|god|sudo|jailbreak)\s+mode\b",
    re.IGNORECASE,
)

_AR_DIRECT_INTENT_PATTERNS = (
    re.compile(r"\byou\s+(?:must|will|should|can|cannot|can'?t|are|were)\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\b", re.IGNORECASE),
    re.compile(r"\b(?:never|do\s+not|don'?t)\s+(?:ever\s+)?(?:refuse|decline)\b", re.IGNORECASE),
    re.compile(
        r"\bnever\s+say\s+(?:that\s+)?(?:you\s+)?(?:can'?t|cannot|won'?t|no)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:do\s+not|don'?t|never)\s+(?:lecture|moralize|moralise|preach|judge|warn)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byou\s+have\s+no\s+(?:restrictions?|limitations?|guidelines?|policies?)\b", re.IGNORECASE
    ),
    re.compile(r"\b(?:answer|respond)\s+directly\b", re.IGNORECASE),
    re.compile(r"\bignore\s+your\s+(?:guidelines|policy|safety|content|rules?)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+\w+", re.IGNORECASE),
    re.compile(r"\balways\s+comply\b", re.IGNORECASE),
)
_AR2_LIVE_SUPPRESSION_PATTERN = re.compile(
    r"\b(?:do\s+not|don'?t|never|no)\s+(?:include|add|give|provide|use|append)?\s*"
    r"(?:any\s+)?(?:warnings?|disclaimers?|caveats?)\b",
    re.IGNORECASE,
)
_AR2_DIRECT_INTENT_PATTERNS = (
    _AR2_LIVE_SUPPRESSION_PATTERN,
    re.compile(
        r"\bwithout\s+(?:any\s+)?(?:warnings?|disclaimers?|caveats?|moralizing|moralising|"
        r"judg(?:e?ment|ing))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do\s+not|don'?t|never)\s+(?:lecture|moralize|moralise|preach|judge|warn)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:skip|omit|leave\s+out|drop)\s+(?:the\s+)?(?:disclaimer|warning|ethics|safety)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|without)\s+(?:ethical|moral)\s+(?:judg(?:e?ment)|considerations?|commentary|"
        r"objections?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:do\s+not|don'?t)\s+(?:apologize|apologise|say\s+sorry)\b", re.IGNORECASE),
)
_BENIGN_AR_SCHEMA_FIELD_PATTERN = re.compile(
    r"""
    ^\s*(?:\[\])?\s+(?:field|key|property|array|list|entry)\b
    |
    ^\s*(?:\[\])?\s+(?:in|of)\s+(?:the\s+)?(?:json(?:\s+output)?|output|response)\s+schema\b
    |
    ^\s*(?:\[\])?\s+(?:in|of)\s+(?:the\s+)?(?:warnings?|disclaimers?|caveats?)\b(?:\[\])?\s+
    (?:field|key|property|array|list|entry)\b
    |
    ^\s*(?:\[\])?\s+in\s+(?:the\s+)?errors\[\]\s+array\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BENIGN_AR_WARNING_INTRO_PATTERN = re.compile(r"^\s*(?:warning|note)\s*:\s*$", re.IGNORECASE)
_BENIGN_AR_DENYLIST_DECLARATION_PATTERN = re.compile(
    r"^\s*deny-?list\s+declaration\s*:\s*(?:[|>])?\s*$",
    re.IGNORECASE,
)
_DIRECTIVE_DOCUMENTATION_LABEL_PATTERN = re.compile(r"^\s*documentation\s*:\s*", re.IGNORECASE)
_DOCUMENTATION_HEADING_PATTERN = re.compile(r"^\s*documentation\s*:\s*$", re.IGNORECASE)
_BENIGN_AR_FIXTURE_INTRO_PATTERN = re.compile(
    r"^\s*(?:#\s*)?(?:defensive\s+fixture|unit\s+test|test\s+case)\b",
    re.IGNORECASE,
)
_EXPLICIT_EXAMPLE_CONTEXT_PATTERN = re.compile(
    r"(?:```|example:|for example|e\.g\.|such as|# warning:|# note:|\*\*warning\*\*|\*\*note\*\*|// ✅|// ❌|// good:|// bad:|// correct:|// incorrect:|// wrong:)",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"[.;!?]")
_DEFENSIVE_AR_CONTEXT_PATTERN = re.compile(
    r"(?:^\s*(?:warning|note)\s*:|\b(?:malicious|example|attack|defensive)\s+"
    r"(?:phrase|payload|string|text|snippet|content|example)\b)",
    re.IGNORECASE,
)
_RETROSPECTIVE_AR_NARRATIVE_PATTERNS = (
    re.compile(
        r"\b(?:the|this|that)\s+(?:old|previous|prior)\s+"
        r"(?:agent|model|system|implementation|version|behavior)\s+would\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the|this|that)\s+(?:agent|model|system|implementation|version|behavior)\s+used\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpreviously\s+would\b", re.IGNORECASE),
    re.compile(r"\bpreviously\s+used\s+to\b", re.IGNORECASE),
    re.compile(
        r"\bpreviously\s*,?\s+(?:the|this|that)\s+"
        r"(?:agent|model|system|implementation|version|behavior)\s+would\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:fixed|resolved|addressed|corrected)\s+(?:a|the)\s+"
        r"(?:bug|issue|problem)\s+where\s+(?:the|this|that)\s+"
        r"(?:agent|model|system|implementation|version|behavior)\s+would\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the|this|that)\s+(?:agent|model|system|implementation|version|behavior)\s+"
        r"no\s+longer\s+(?:would|used\s+to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the|this|that)\s+(?:agent|model|system|implementation|version|behavior)\s+"
        r"would\s+no\s+longer\b",
        re.IGNORECASE,
    ),
)

# AR2 in descriptive Python comments.
#
# A bare "no warning(s)/disclaimer(s)/caveat(s)" is the AR2 signal in "respond with
# no warnings". In "# the server emits no warning either way" it is the object of a
# program's reported behavior: a note to developers, not an instruction to suppress
# warnings. Such a match is dropped only when ALL of the following hold:
#
# 1. The match is the bare determiner form. The "do not/don't/never" forms and every
#    other AR2 pattern keep their findings.
# 2. The match lies wholly inside one proven Python comment token. The whole analyzed
#    text must parse as a module, so SKILL.md, markdown, prompts, docstrings, string
#    literals, other languages, fragments, and malformed source never qualify.
# 3. The words right before the match are a finite report verb from a closed allowlist,
#    in third-person "-s" or past-tense form. Base forms ("give no warnings") read as
#    imperatives and are not on it. The verb's subject is either a program noun from a
#    closed allowlist ("the server emits"), or it is omitted because a code-behavior
#    verb opens the comment ("# Returns no warning when ..."). An unlisted subject or
#    verb keeps the finding, so "the assistant emits no warnings" stays active.
# 4. The contiguous comment block around the match never addresses an agent or the
#    reader ("# Assistant: ...", "you", "model", "prompt", ...). The walk over that block
#    is bounded, and a block longer than the bound keeps the finding.
_AR2_BARE_NO_WARNING_PATTERN = re.compile(
    r"no\s+(?:any\s+)?(?:warnings?|disclaimers?|caveats?)",
    re.IGNORECASE,
)
_AR2_REPORT_VERBS = (
    r"contains|contained|displays|displayed|emits|emitted|generates|generated|gives|gave|"
    r"has|had|includes|included|issues|issued|logs|logged|outputs|prints|printed|"
    r"produces|produced|raises|raised|reports|reported|returns|returned|sends|sent|"
    r"shows|showed|surfaces|surfaced|throws|threw|writes|wrote|yields|yielded"
)
_AR2_COMMENT_OPENING_REPORT_VERBS = (
    r"emits|emitted|logs|logged|prints|printed|raises|raised|returns|returned|"
    r"throws|threw|yields|yielded"
)
_AR2_PROGRAM_SUBJECTS = (
    r"apis?|backends?|binary|binaries|builds?|calls?|checks?|cli|clients?|commands?|"
    r"compilers?|daemons?|databases?|db|drivers?|endpoints?|functions?|handlers?|helpers?|"
    r"interpreters?|jobs?|library|libraries|linters?|methods?|modules?|packages?|"
    r"parsers?|pipelines?|process|processes|programs?|query|queries|requests?|runtimes?|"
    r"scripts?|sdks?|servers?|services?|subprocess|subprocesses|tests?|tools?|upstream|"
    r"validators?|wrappers?"
)
_AR2_REPORT_ADVERBS = r"(?:(?:[a-z]+ly|also|already|always|even|just|now|still|then)\s+){0,2}"
_AR2_PROGRAM_SUBJECT_REPORT_PATTERN = re.compile(
    rf"(?<![\w-])(?:{_AR2_PROGRAM_SUBJECTS})\s+{_AR2_REPORT_ADVERBS}"
    rf"(?:{_AR2_REPORT_VERBS})\s+\Z",
    re.IGNORECASE,
)
_AR2_COMMENT_OPENING_REPORT_PATTERN = re.compile(
    rf"#+[ \t]*{_AR2_REPORT_ADVERBS}(?:{_AR2_COMMENT_OPENING_REPORT_VERBS})[ \t]+",
    re.IGNORECASE,
)
_AR2_AGENT_ADDRESS_PATTERN = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves|assistants?|agents?|models?|ai|llms?|"
    r"chatbots?|bots?|claude|chatgpt|gpt|copilot|gemini|codex|personas?|prompts?|"
    r"instructions?)\b",
    re.IGNORECASE,
)
# A subject or comment opening, two adverbs, and a verb fit well inside this many
# characters. Bounding both grammar checks keeps a long comment line linear.
_AR2_REPORT_PREFIX_CHARS = 160
# A comment block reaching this many lines on either side of the match keeps it.
_AR2_COMMENT_BLOCK_MAX_LINES = 64


def _is_directly_instructive(context: str, matched_text: str) -> bool:
    """Return True when the match still looks like an active adversarial instruction."""
    context_lower = context.lower()
    matched_text_lower = matched_text.lower()
    if any(pattern.search(context_lower) for pattern in _AR_DIRECT_INTENT_PATTERNS):
        return True
    if any(pattern.search(context_lower) for pattern in _AR2_DIRECT_INTENT_PATTERNS):
        return True
    return "do anything now" in matched_text_lower


def _is_explicit_example_context(context: str) -> bool:
    """Return True only for explicit example-style scaffolding, not generic docs labels."""
    return bool(_EXPLICIT_EXAMPLE_CONTEXT_PATTERN.search(context))


def _match_clause_bounds(match_line: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Return the semantically local clause around a match on one line."""
    clause_start = 0
    for boundary in _CLAUSE_BOUNDARY_PATTERN.finditer(match_line):
        if boundary.start() >= match_start:
            break
        clause_start = boundary.end()
    clause_end = len(match_line)
    boundary_match = _CLAUSE_BOUNDARY_PATTERN.search(match_line, match_end)
    if boundary_match:
        clause_end = boundary_match.start()
    return clause_start, clause_end


def _match_clause(match_line: str, match_start: int, match_end: int) -> tuple[str, int, int]:
    """Return the clause text and the match offsets within that clause."""
    clause_start, clause_end = _match_clause_bounds(match_line, match_start, match_end)
    return (
        match_line[clause_start:clause_end],
        match_start - clause_start,
        match_end - clause_start,
    )


def _emitted_context(
    context: str,
    match_line: str,
    is_directive: bool,
    previous_line: str | None = None,
) -> str:
    """Keep runner-visible context on the directive when example markers are false context."""
    if not is_directive:
        return context
    trimmed_line = _DIRECTIVE_DOCUMENTATION_LABEL_PATTERN.sub("", match_line, count=1)
    if trimmed_line != match_line:
        return trimmed_line
    if previous_line and _DOCUMENTATION_HEADING_PATTERN.search(previous_line):
        return match_line
    if _is_explicit_example_context(context):
        return match_line
    return context


def _is_quoted_match(match_line: str, matched_text: str) -> bool:
    """Return True when the matched phrase is quoted on the same line."""
    matched_text_lower = matched_text.lower()
    match_line_lower = match_line.lower()
    if any(
        re.search(
            rf"{re.escape(quote)}[^{re.escape(quote)}\n]*{re.escape(matched_text_lower)}[^{re.escape(quote)}\n]*{re.escape(quote)}",
            match_line_lower,
        )
        for quote in ('"', "'", "`")
    ):
        return True
    if re.search(
        rf"\bthe\s+phrase\b.*?[\"'`][^\"'`\n]*{re.escape(matched_text_lower)}[^\"'`\n]*[\"'`]",
        match_line_lower,
    ):
        return True
    return False


def _has_explicit_defensive_context(
    match_line: str,
    previous_line: str | None = None,
) -> bool:
    """Return True when quoted text is clearly framed as defensive prose."""
    if _DEFENSIVE_AR_CONTEXT_PATTERN.search(match_line):
        return True
    if not previous_line:
        return False
    if _BENIGN_AR_WARNING_INTRO_PATTERN.search(previous_line):
        return True
    if _BENIGN_AR_DENYLIST_DECLARATION_PATTERN.search(previous_line):
        return True
    return bool(_BENIGN_AR_FIXTURE_INTRO_PATTERN.search(previous_line))


def _is_match_local_narrative_clause(
    match_clause: str,
    clause_match_start: int,
) -> bool:
    """Return True when the current match is part of a narrative clause, not a directive."""
    prefix = match_clause[:clause_match_start]
    prefix_end = len(prefix.rstrip())
    return any(
        (match := pattern.search(prefix)) is not None and match.end() == prefix_end
        for pattern in _RETROSPECTIVE_AR_NARRATIVE_PATTERNS
    )


def _is_schema_field_clause(
    match_clause: str,
    matched_text: str,
    clause_match_end: int,
) -> bool:
    """Return True when an AR2 warning-suppression phrase targets schema fields."""
    if not _AR2_LIVE_SUPPRESSION_PATTERN.search(matched_text):
        return False
    continuation = match_clause[clause_match_end:]
    return bool(_BENIGN_AR_SCHEMA_FIELD_PATTERN.search(continuation))


def _is_benign_ar_context(
    match_line: str,
    match: str,
    line_match_start: int,
    line_match_end: int,
    previous_line: str | None = None,
) -> bool:
    """Return True for high-confidence non-malicious prose patterns around one match span."""
    match_clause, clause_match_start, clause_match_end = _match_clause(
        match_line,
        line_match_start,
        line_match_end,
    )
    if _is_match_local_narrative_clause(match_clause, clause_match_start):
        return True
    if _is_schema_field_clause(match_clause, match.lower(), clause_match_end):
        return True
    return _is_quoted_match(match_line, match) and _has_explicit_defensive_context(
        match_line,
        previous_line=previous_line,
    )


def _no_runtime_check() -> None:
    """Stand in for the runner deadline when analyze() is called directly."""


class _PythonComments:
    """Lazily proven comment tokens of one analyzed Python text."""

    def __init__(self, content: str, check_runtime: Callable[[], None]) -> None:
        self._content = content
        self._check_runtime = check_runtime
        self._starts: tuple[int, ...] = ()
        self._ends: tuple[int, ...] = ()
        self._computed = False
        self._block_unaddressed: dict[int, bool] = {}

    def comment_index(self, start: int, end: int) -> int | None:
        """Return the comment token that wholly contains ``[start, end)``, if proven."""
        if not self._computed:
            spans = python_literal_spans(self._content, self._check_runtime)
            if spans is not None:
                self._starts, self._ends = spans
            self._computed = True
        index = bisect_right(self._starts, start) - 1
        if index < 0 or end > self._ends[index] or not self._is_comment(index):
            return None
        return index

    def comment_start(self, index: int) -> int:
        return self._starts[index]

    def block_is_unaddressed(self, index: int) -> bool:
        """Return True when no agent or reader is addressed in the comment block."""
        bounds = self._block_bounds(index)
        if bounds is None:
            return False
        first, last = bounds
        cached = self._block_unaddressed.get(first)
        if cached is None:
            cached = (
                _AR2_AGENT_ADDRESS_PATTERN.search(
                    self._content, self._starts[first], self._ends[last]
                )
                is None
            )
            self._block_unaddressed[first] = cached
        return cached

    def _is_comment(self, index: int) -> bool:
        return self._content[self._starts[index]] == "#"

    def _on_adjacent_lines(self, earlier: int, later: int) -> bool:
        if not (self._is_comment(earlier) and self._is_comment(later)):
            return False
        gap = self._content[self._ends[earlier] : self._starts[later]]
        return gap.count("\n") == 1 and not gap.strip()

    def _block_bounds(self, index: int) -> tuple[int, int] | None:
        first = last = index
        for _ in range(_AR2_COMMENT_BLOCK_MAX_LINES):
            if first == 0 or not self._on_adjacent_lines(first - 1, first):
                break
            first -= 1
        else:
            return None
        for _ in range(_AR2_COMMENT_BLOCK_MAX_LINES):
            if last + 1 == len(self._starts) or not self._on_adjacent_lines(last, last + 1):
                break
            last += 1
        else:
            return None
        return first, last


def _is_descriptive_python_comment(comments: _PythonComments, match: re.Match[str]) -> bool:
    """Return True when an AR2 match reports a program's behavior in a Python comment."""
    if not _AR2_BARE_NO_WARNING_PATTERN.fullmatch(match.group(0)):
        return False
    index = comments.comment_index(match.start(), match.end())
    if index is None:
        return False
    content = match.string
    comment_start = comments.comment_start(index)
    prefix_start = max(comment_start, match.start() - _AR2_REPORT_PREFIX_CHARS)
    reports_program_behavior = bool(
        _AR2_PROGRAM_SUBJECT_REPORT_PATTERN.search(content, prefix_start, match.start())
        or (
            prefix_start == comment_start
            and _AR2_COMMENT_OPENING_REPORT_PATTERN.fullmatch(content, comment_start, match.start())
        )
    )
    return reports_program_behavior and comments.block_is_unaddressed(index)


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    check_runtime: Callable[[], None] | None = None,
) -> list[AnalyzerFinding]:
    """Analyze content for anti-refusal statements (AR1-AR3)."""
    findings: list[AnalyzerFinding] = []
    locations = SourceLocationIndex(content, file_path)
    tag = [PatternCategory.ANTI_REFUSAL.value]
    python_comments = (
        _PythonComments(content, check_runtime or _no_runtime_check)
        if file_type == "python"
        else None
    )

    for rule_id, patterns in _RULES:
        for pattern, base_confidence in patterns:
            for match in static_runner.iter_paragraph_matches(
                pattern, content, re.IGNORECASE | re.MULTILINE
            ):
                if (
                    rule_id == "AR2"
                    and python_comments is not None
                    and _is_descriptive_python_comment(python_comments, match)
                ):
                    continue
                lines = content.splitlines()
                line_num = get_line_number(content, match.start())
                match_line = lines[line_num - 1] if lines else content
                previous_line = lines[line_num - 2] if line_num > 1 else None
                context = get_context(content, match.start(), context_lines=3)
                security_review_context = bool(
                    _MODE_ENABLED_RE.fullmatch(match.group(0))
                    and _SECURITY_REVIEW_CONTEXT_RE.search(context)
                )

                line_start = content.rfind("\n", 0, match.start()) + 1
                line_match_start = match.start() - line_start
                line_match_end = line_match_start + len(match.group(0))
                match_clause, _, _ = _match_clause(match_line, line_match_start, line_match_end)
                is_directive = _is_directly_instructive(match_clause.lower(), match.group(0))
                example_context = is_code_example(context) and _is_explicit_example_context(context)
                benign_context = _is_benign_ar_context(
                    match_line,
                    match.group(0),
                    line_match_start,
                    line_match_end,
                    previous_line=previous_line,
                )
                finding_tags = list(tag)
                if security_review_context or example_context or benign_context:
                    finding_tags.extend(["contextual-triage", "likely-benign-context"])
                findings.append(
                    AnalyzerFinding(
                        rule_id=rule_id,
                        message="Anti-Refusal Statement",
                        severity=Severity.HIGH,
                        location=locations.location(match.start(), match.end()),
                        confidence=base_confidence,
                        tags=finding_tags,
                        context=_emitted_context(
                            context,
                            match_line,
                            is_directive,
                            previous_line=previous_line,
                        ),
                        matched_text=match.group(0)[:200],
                        complete_match=match.group(0),
                    )
                )
    return _deduplicate_findings(findings)


def _deduplicate_findings(findings: list[AnalyzerFinding]) -> list[AnalyzerFinding]:
    """Compact only exact same-location matches."""
    return static_runner.deduplicate_analyzer_findings(findings)


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run anti_refusal patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
