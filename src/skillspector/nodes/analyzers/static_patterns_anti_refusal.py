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

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number, is_code_example
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_anti_refusal"

# AR1: Refusal suppression - instruct the agent to never refuse / always comply.
AR1_PATTERNS = [
    (r"\b(?:never|do\s+not|don'?t)\s+(?:ever\s+)?(?:refuse|decline)\b", 0.85),
    (r"\byou\s+(?:must|will|should|can)\s*(?:not|never)\s+(?:refuse|decline|say\s+no)\b", 0.9),
    (r"\byou\s+(?:cannot|can'?t|can\s+not)\s+(?:refuse|decline|say\s+no)\b", 0.85),
    (r"\bnever\s+say\s+(?:that\s+)?(?:you\s+)?(?:can'?t|cannot|won'?t|no)\b", 0.85),
    (r"\balways\s+(?:comply|obey|answer|respond|agree)\b", 0.7),
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

# Confidence penalty applied when the match appears inside a code/doc example, and the
# minimum confidence required to emit a finding after the penalty.
_EXAMPLE_PENALTY = 0.4
_MIN_CONFIDENCE = 0.5

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
    \b(?:warnings?|disclaimers?|caveats?)\b(?:\[\])?\s+
    (?:field|key|property|array|list|entry)\b
    |
    \b(?:warnings?|disclaimers?|caveats?)\b(?:\[\])?\s+
    (?:in|of)\s+(?:the\s+)?(?:json(?:\s+output)?|output|response)\s+schema\b
    |
    \b(?:warnings?|disclaimers?|caveats?)\b[^.\n]{0,80}\b(?:schema\s+field|schema\s+key|errors?\[\]\s+array)\b
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
_LIVE_RESPONSE_OBJECT_PATTERN = re.compile(
    r"\b(?:answer|response|reply|assistant|user)\b",
    re.IGNORECASE,
)
_DEFENSIVE_AR_CONTEXT_PATTERN = re.compile(
    r"(?:^\s*(?:warning|note)\s*:|\b(?:malicious|example|attack|defensive)\s+"
    r"(?:phrase|payload|string|text|snippet|content|example)\b)",
    re.IGNORECASE,
)
_RETROSPECTIVE_AR_NARRATIVE_PATTERNS = (
    re.compile(
        r"\b(?:the|this|that)\s+(?:old|previous|prior)?\s*"
        r"(?:agent|model|system|implementation|version|behavior)\s+would\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the|this|that)\s+(?:agent|model|system|implementation|version|behavior)\s+used\s+to\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpreviously\s+would\b", re.IGNORECASE),
    re.compile(r"\bpreviously\s+used\s+to\b", re.IGNORECASE),
)
_SECOND_PERSON_NARRATIVE_PATTERN = re.compile(r"\byou\s+would\b", re.IGNORECASE)


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
    if _SECOND_PERSON_NARRATIVE_PATTERN.search(prefix):
        return False
    return any(pattern.search(prefix) for pattern in _RETROSPECTIVE_AR_NARRATIVE_PATTERNS)


def _is_schema_field_clause(
    match_clause: str,
    matched_text: str,
) -> bool:
    """Return True when an AR2 warning-suppression phrase targets schema fields."""
    if not _AR2_LIVE_SUPPRESSION_PATTERN.search(matched_text):
        return False
    if not _BENIGN_AR_SCHEMA_FIELD_PATTERN.search(match_clause):
        return False
    return not _LIVE_RESPONSE_OBJECT_PATTERN.search(match_clause)


def _is_benign_ar_context(
    match_line: str,
    match: str,
    line_match_start: int,
    line_match_end: int,
    previous_line: str | None = None,
) -> bool:
    """Return True for high-confidence non-malicious prose patterns around one match span."""
    match_clause, clause_match_start, _ = _match_clause(
        match_line,
        line_match_start,
        line_match_end,
    )
    if _is_match_local_narrative_clause(match_clause, clause_match_start):
        return True
    if _is_schema_field_clause(match_clause, match.lower()):
        return True
    return _is_quoted_match(match_line, match) and _has_explicit_defensive_context(
        match_line,
        previous_line=previous_line,
    )


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for anti-refusal statements (AR1-AR3)."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.ANTI_REFUSAL.value]

    for rule_id, patterns in _RULES:
        for pattern, base_confidence in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                lines = content.splitlines()
                line_num = get_line_number(content, match.start())
                match_line = lines[line_num - 1] if lines else content
                previous_line = lines[line_num - 2] if line_num > 1 else None
                context = get_context(content, match.start(), context_lines=3)
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_match_start = match.start() - line_start
                line_match_end = line_match_start + len(match.group(0))
                match_clause, _, _ = _match_clause(match_line, line_match_start, line_match_end)
                is_directive = _is_directly_instructive(match_clause.lower(), match.group(0))
                confidence = base_confidence
                if (
                    is_code_example(context)
                    and _is_explicit_example_context(context)
                    and not _is_quoted_match(
                        match_line,
                        match.group(0),
                    )
                ):
                    confidence -= _EXAMPLE_PENALTY
                if _is_benign_ar_context(
                    match_line,
                    match.group(0),
                    line_match_start,
                    line_match_end,
                    previous_line=previous_line,
                ):
                    continue
                if confidence < _MIN_CONFIDENCE:
                    continue
                findings.append(
                    AnalyzerFinding(
                        rule_id=rule_id,
                        message="Anti-Refusal Statement",
                        severity=Severity.HIGH,
                        location=Location(
                            file=file_path,
                            start_line=line_num,
                        ),
                        confidence=round(confidence, 2),
                        tags=tag,
                        context=_emitted_context(
                            context,
                            match_line,
                            is_directive,
                            previous_line=previous_line,
                        ),
                        matched_text=match.group(0)[:200],
                    )
                )
    return _deduplicate_findings(findings)


def _deduplicate_findings(findings: list[AnalyzerFinding]) -> list[AnalyzerFinding]:
    """Keep the highest-confidence finding per (file, line, rule_id)."""
    best: dict[tuple[str, int, str], AnalyzerFinding] = {}
    for f in findings:
        key = (f.location.file, f.location.start_line, f.rule_id)
        existing = best.get(key)
        if existing is None or f.confidence > existing.confidence:
            best[key] = f
    return list(best.values())


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run anti_refusal patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
