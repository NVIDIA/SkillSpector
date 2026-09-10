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

"""Static patterns: system prompt leakage (P6–P8). Node and analyze() in one module.

Detects patterns that directly extract system prompts (P6), indirectly
extract them via side channels (P7), or exfiltrate prompts via tool
calls (P8).

Framework: LLM07.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from skillspector.artifacts import (
    SecurityTextView,
    _contains_default_ignorable,
    is_default_ignorable,
    normalized_security_view,
    security_text_views,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.security_reconstruction import build_declared_marker_views
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    LINE_BREAK_CHARS,
    LOGICAL_LINE_BREAK,
    get_context,
    get_line_number,
)
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_system_prompt_leakage"

# P6: Direct Prompt Extraction — instructions that directly reveal system prompts
P6_PATTERNS = [
    (
        r"(?:print|output|show|display|reveal|expose|return|echo)\s+(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?|directives?)",
        0.85,
    ),
    (
        r"(?:what\s+(?:are|is)\s+your|tell\s+me\s+your)\s+(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?|directives?)",
        0.8,
    ),
    (
        r"(?:output|print|display)\s+(?:the\s+)?(?:text|content|message)\s+(?:above|before)\s+(?:this|the\s+user)",
        0.8,
    ),
    (
        r"(?:share|disclose|leak|give\s+me)\s+(?:your\s+)?(?:internal|hidden|system|secret)\s+(?:prompt|instructions?|rules?|config)",
        0.85,
    ),
    (
        r"(?:dump|extract|copy)\s+(?:the\s+)?(?:system\s+)?(?:prompt|message|instructions?)\s+(?:to|into|as)",
        0.8,
    ),
    (
        r"(?:begin|start)\s+(?:your\s+)?(?:response|output|reply)\s+with\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
    (
        r"(?:first|before\s+anything)\s+(?:output|print|show|display)\s+(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
    (
        r"(?:verbatim|exactly|word\s+for\s+word)\s+(?:repeat|output|print)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.9,
    ),
]

# P7: Indirect Prompt Extraction — side-channel or inference-based extraction
P7_PATTERNS = [
    (
        r"(?:repeat|recite|reproduce)\s+(?:everything|all|the\s+text)\s+(?:above|before|prior|preceding)",
        0.8,
    ),
    (
        r"(?:summarize|paraphrase|rephrase|translate)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?)",
        0.75,
    ),
    (
        r"(?:translate|convert)\s+(?:your\s+)?(?:instructions?|rules?|prompt)\s+(?:into|to)\s+(?:\w+)",
        0.75,
    ),
    (
        r"(?:what\s+(?:were|was)\s+you\s+told|how\s+were\s+you\s+(?:instructed|configured|programmed|set\s+up))",
        0.7,
    ),
    (
        r"(?:if\s+you\s+had\s+to\s+(?:describe|explain)|describe\s+how\s+you(?:'re|\s+are)\s+(?:configured|set\s+up|instructed))",
        0.7,
    ),
    (
        r"(?:encode|encrypt|rot13|base64|reverse)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)",
        0.85,
    ),
    (
        r"(?:write|create)\s+(?:a\s+)?(?:poem|song|story|acrostic)\s+(?:about|using|from)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.75,
    ),
    (
        r"(?:format|present|rearrange)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:as|in)\s+(?:a\s+)?(?:list|json|yaml|csv|table|markdown)",
        0.8,
    ),
    (
        r"(?:spell\s+out|enumerate)\s+(?:each|every|all)\s+(?:of\s+your\s+)?(?:rules?|instructions?|constraints?|guidelines?)",
        0.75,
    ),
    (
        r"(?:ignore|skip)\s+(?:the\s+)?(?:last|final|end)\s+(?:instruction|rule)\s+and\s+(?:output|print|show)\s+(?:the\s+)?(?:rest|previous|first)",
        0.8,
    ),
]

# P8: Prompt Exfiltration via Tool — system prompt sent via tool calls
P8_PATTERNS = [
    (
        r"(?:write|save|store|log|dump)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:to|into|in)\s+(?:a\s+)?(?:file|disk|log|database|storage)",
        0.85,
    ),
    (
        r"(?:send|post|upload|transmit)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:to|via)\s+(?:https?://|external|remote|webhook|api|server)",
        0.9,
    ),
    (
        r"(?:include|embed|attach|append)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:in|to)\s+(?:the\s+)?(?:file|output|response|request|payload|email|message)",
        0.8,
    ),
    (
        r"(?:log|record|track|monitor)\s+(?:all\s+)?(?:system\s+)?(?:prompts?|instructions?)\s+(?:to|in|at)\s+(?:external|remote|server)",
        0.85,
    ),
    (
        r"(?:copy|clone|duplicate)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|config)\s+(?:to|into)\s+(?:a\s+)?(?:new|separate|external)",
        0.8,
    ),
    (
        r"(?:pipe|redirect|forward)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:to|through|via)\s+(?:curl|wget|fetch|http|api|tool)",
        0.85,
    ),
    (
        r"(?:use|call|invoke)\s+(?:the\s+)?(?:write|file|network|http|fetch|curl)\s+tool\s+(?:to|and)\s+(?:save|send|export)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
]

_LEGACY_OUTPUT_RULES_HEADING = "## Output Rules (Both Modes)"
_BENIGN_OUTPUT_RULES_HEADING = re.compile(
    r"[ ]{0,3}#{1,6}[ \t]+"
    r"(?:HTML|JSON|CSV|Markdown)[ \t]+"
    r"(?P<target>output[ \t]+rules)"
    r"(?:[ \t]+\(offline-safe\))?"
    r"(?:[ \t]+#+)?[ \t]*",
    re.IGNORECASE,
)
# Formatting is transparent when deciding whether a label is being used as a
# command. These token states persist across whitespace, comments and scan
# windows, but stop at sentence boundaries. The decision is made once for the
# complete artifact, never from an arbitrarily truncated neighbouring excerpt.
# Exemption analysis is optional: beyond this bound retain ordinary detections.
_MAX_HEADING_CONTEXT_CHARS = 1024 * 1024
_MARKUP_START = re.compile(r"<!--|</?[A-Za-z]")
_MARKDOWN_DELIMITERS = str.maketrans("", "", "*_`")
_REPORT_OBJECTS = frozenset(
    "report reports document documents chart charts figure figures file files table tables".split()
)
_CONTEXT_TOKENS = re.compile(r"[\w]+|[.!?;:]")
_HEADING_ACTIONS = frozenset("interpret treat read execute perform follow obey do carry".split())
_HEADING_OBJECTS = frozenset(
    "heading title label command instruction operation following below above".split()
)
_REFERENCE_ACTIONS = frozenset(
    "print output show display reveal expose return echo repeat share disclose "
    "publish provide send copy extract dump recite summarize translate encode "
    "forward pipe write save store log do".split()
)
_REFERENCE_OBJECTS = frozenset(
    "it them so this that these those above below previous preceding foregoing former latter".split()
)
_SENSITIVE_QUALIFIERS = frozenset("system developer hidden internal secret governing".split())
_SENSITIVE_OBJECTS = frozenset("prompt prompts instruction instructions rule rules".split())
_INDEPENDENT_EXTRACTION = re.compile(
    "|".join(f"(?:{pattern})" for pattern, _ in (*P6_PATTERNS, *P7_PATTERNS, *P8_PATTERNS)), re.I
)

_LOGICAL_BREAK = rf"(?:{LOGICAL_LINE_BREAK.pattern})"
_BENIGN_PRINT_RULES_TAXONOMY = re.compile(
    rf"(?:\A|{_LOGICAL_BREAK})[ \t]*[\"'`]{{0,3}}[ \t]*"
    r"(?:single-class[ \t]+selectors[ \t]+are[ \t]+honored[ \t]+(?:—|--|-)[ \t]+)?"
    r"descendant[ \t]*/[ \t]*compound[ \t]*/[ \t]*"
    r"(?P<target>print[ \t]+rules)[ \t]+are"
    rf"(?:[ \t]+|[ \t]*{_LOGICAL_BREAK}[ \t]+)(?:not|never)[ \t]+evaluated"
    r"(?:[ \t]+\((?:avoids?|to[ \t]+avoid)[ \t]+over-stripping[ \t]+content"
    r"[ \t]+behind[ \t]+e\.g\.[ \t]+`?\.a[ \t]+\.b`?[ \t]+rules\))?"
    r"[ \t]*(?:[.!?][ \t]*)?[\"'`]{0,3}[ \t]*"
    rf"(?=\Z|{_LOGICAL_BREAK})",
    re.IGNORECASE,
)
_PRECEDING_DIRECTIVE = re.compile(
    r"\b(?:you|your|agents?|assistants?|models?|llms?|bots?|must|shall|should|"
    r"required|mandatory)\b"
    r"|\bbefore[ \t]+(?:replying|responding)\b"
    r"|\b(?:following|below|above|next|this|that|it|them|these|those|so|prior|"
    r"previous|preceding|everything|all|former|latter|content|text|output|"
    r"configuration|material)\b"
    r"|\bthe[ \t]+same\b"
    r"|\bwhat[ \t]+follows\b"
    r"|:[ \t]*$",
    re.IGNORECASE,
)
_NEXT_LINE_REFERENCE = re.compile(
    r"\b(?:it|them|this|these|those|so|same|above|below|prior|previous|"
    r"preceding|following|foregoing|everything|all|former|latter|content|text|"
    r"output|configuration|material)\b"
    r"|\b(?:the|this|that|these|those|same)[ \t]+(?:rules?|instructions?|prompts?|"
    r"guidelines?|directives?|operations?|actions?)\b"
    r"|\b(?:do|execute|perform|apply|follow|obey|use|print|output|show|display|"
    r"reveal|expose|return|echo|repeat|share|disclose|publish|provide|send|copy|"
    r"extract|dump|recite|summarize|translate|encode|write|save|forward|pipe)"
    r"[ \t]+that\b",
    re.IGNORECASE,
)


def _has_heading_framing(
    content: str, check_runtime: Callable[[], None], *, separator_reading: bool = True
) -> bool:
    """Conservatively retain labels when the artifact assigns them instructions.

    Recognized independent extraction spans are removed only from this context
    check: they are still scanned and reported normally. Thus a real extraction
    following a report label does not also turn that label into a second finding.
    """
    if len(content) > _MAX_HEADING_CONTEXT_CHARS:
        check_runtime()
        return True
    if separator_reading and _contains_default_ignorable(content):
        separated: list[str] = []
        for start in range(0, len(content), 65536):
            check_runtime()
            separated.append(
                "".join(
                    " " if is_default_ignorable(character) else character
                    for character in content[start : start + 65536]
                )
            )
        if _has_heading_framing("".join(separated), check_runtime, separator_reading=False):
            return True
    chunks: list[str] = []
    normalized_size = 0
    for start in range(0, len(content), 65536):
        check_runtime()
        chunk = normalized_security_view(content[start : start + 65536]).text
        normalized_size += len(chunk)
        if normalized_size > _MAX_HEADING_CONTEXT_CHARS:
            check_runtime()
            return True
        chunks.append(chunk)
    text = re.sub(r"\s+", " ", "".join(chunks))
    text = _INDEPENDENT_EXTRACTION.sub(" ", text)
    if _context_is_framed(text, check_runtime):
        return True
    # Inspect the source plus rendered readings. Only extraction spans visible
    # in the source were removed above: rendering must not silently bless a
    # hidden instruction that the ordinary detector has not independently seen.
    for markup_separator in ("", " "):
        rendered = _render_context(text, markup_separator, check_runtime)
        if rendered is None:
            return True
        if rendered != text and _context_is_framed(rendered, check_runtime):
            return True
    return False


def _render_context(
    text: str, markup_separator: str, check_runtime: Callable[[], None]
) -> str | None:
    """Read inline markup linearly; incomplete markup cannot justify exemption."""
    parts: list[str] = []
    cursor = 0
    while marker := _MARKUP_START.search(text, cursor):
        check_runtime()
        parts.append(text[cursor : marker.start()])
        if marker.group() == "<!--":
            end = text.find("-->", marker.end())
            if end < 0:
                return None
            parts.append(markup_separator)
            cursor = end + 3
            continue
        # A greater-than character inside an attribute is not the tag's end.
        position = marker.end()
        quote = ""
        while position < len(text):
            if position % 1024 == 0:
                check_runtime()
            character = text[position]
            if quote:
                if character == quote:
                    quote = ""
            elif character in "\"'":
                quote = character
            elif character == ">":
                break
            position += 1
        if position == len(text):
            return None
        parts.append(markup_separator)
        cursor = position + 1
    parts.append(text[cursor:])
    return "".join(parts).translate(_MARKDOWN_DELIMITERS)


def _context_is_framed(text: str, check_runtime: Callable[[], None]) -> bool:
    pending_action = False
    pending_heading = False
    pending_reference = False
    previous = ""
    before_previous = ""
    for index, match in enumerate(_CONTEXT_TOKENS.finditer(text)):
        if index % 1024 == 0:
            check_runtime()
        token = match.group().lower()
        if pending_reference:
            # "Save this report" names an artifact. "Repeat them", "send that
            # back" and an unqualified "show this" still reference the label.
            if token not in _REPORT_OBJECTS:
                return True
            pending_reference = False
        if token in ".!?;:":
            pending_action = pending_heading = False
            previous = before_previous = ""
            continue
        if token in {"not", "never"}:
            pending_action = pending_heading = False
        if previous in _SENSITIVE_QUALIFIERS and token in _SENSITIVE_OBJECTS:
            return True
        if previous in _REFERENCE_ACTIONS and token in _REFERENCE_OBJECTS:
            if token in {"this", "that", "these", "those"}:
                pending_reference = True
            else:
                return True
        if (
            before_previous in _REFERENCE_ACTIONS
            and previous in {"the", "same"}
            and token in _SENSITIVE_OBJECTS
        ):
            return True
        if previous not in {"not", "never"}:
            if pending_action and token in _HEADING_OBJECTS:
                return True
            if pending_heading and token in {"command", "instruction", "commands", "instructions"}:
                return True
            pending_action |= token in _HEADING_ACTIONS
            pending_heading |= token in {"heading", "title", "label"}
        before_previous, previous = previous, token
    check_runtime()
    return pending_reference


def _has_reconstructed_framing(content: str, check_runtime: Callable[[], None]) -> bool:
    """Do not approve labels before inspecting recoverable instructions elsewhere.

    Overlapping bounded neighborhoods include the marker decoder's full scope.
    Reconstructed context can only revoke an exemption; it cannot create one.
    Ambiguous active decoding also keeps the original conservative detection.
    """
    for start in range(0, len(content), 32768):
        check_runtime()
        for view in security_text_views(content[start : start + 65536]):
            check_runtime()
            if view.name not in {"raw", "normalized"} and _has_heading_framing(
                view.text, check_runtime
            ):
                return True
            decoded = build_declared_marker_views(view, check_runtime=check_runtime)
            if decoded.limited:
                return True
            if any(_has_heading_framing(item.text, check_runtime) for item in decoded.views):
                return True
        if start + 65536 >= len(content):
            break
    return False


@dataclass(frozen=True)
class _PreparedAnalysis:
    # Absolute source start -> exclusive source end for complete, approved labels.
    heading_spans: Mapping[int, int]

    def analyze(
        self, content: str, file_path: str, file_type: str, source_view: SecurityTextView
    ) -> list[AnalyzerFinding]:
        return _analyze(content, file_path, self, source_view)

    def analyze_whitespace_continuity(
        self, content: str, file_path: str, file_type: str, source_view: SecurityTextView
    ) -> list[AnalyzerFinding]:
        # Only P6 opts into whitespace compaction. Other rules may intentionally
        # bound their gaps; changing their input would change that security policy.
        return _analyze(content, file_path, self, source_view, p6_only=True)

    def is_report_label(self, match: re.Match[str], view: SecurityTextView) -> bool:
        start = view.source_offset(match.start())
        end = view.source_offset(match.end() - 1) + 1
        approved_end = self.heading_spans.get(start)
        # An overlapping view may end at singular "Rule". It is safe only if the
        # entire detected noun phrase lies inside the same complete source label.
        return approved_end is not None and start < end <= approved_end


def prepare_analysis(
    content: str, file_type: str, check_runtime: Callable[[], None]
) -> _PreparedAnalysis:
    """Prepare immutable, complete-source heading decisions under the runner budget."""
    spans: dict[int, int] = {}
    check_runtime()
    if file_type != "markdown" or len(content) > _MAX_HEADING_CONTEXT_CHARS:
        return _PreparedAnalysis(MappingProxyType(spans))
    offset = 0
    for index, raw_line in enumerate(content.splitlines(keepends=True)):
        if index % 128 == 0:
            check_runtime()
        line = raw_line.rstrip(LINE_BREAK_CHARS)
        # Very long lines cannot be approved from a bounded fragment. Leave them
        # detectable; normal report headings fit comfortably within this bound.
        if len(line) <= 4096:
            normalized = normalized_security_view(line)
            target = _report_heading_target(normalized.text)
            candidates: tuple[tuple[SecurityTextView, tuple[int, int]], ...]
            if target is not None:
                candidates = ((normalized, target),)
            elif normalized.text.lstrip().startswith("#"):
                candidates = tuple(
                    (view, target)
                    for view in security_text_views(line)
                    if (target := _report_heading_target(view.text)) is not None
                )
            else:
                candidates = ()
            for view, (start, end) in candidates:
                spans[offset + view.source_offset(start)] = offset + view.source_offset(end - 1) + 1
        offset += len(raw_line)
    if spans and (
        _has_heading_framing(content, check_runtime)
        or _has_reconstructed_framing(content, check_runtime)
    ):
        spans.clear()
    check_runtime()
    return _PreparedAnalysis(MappingProxyType(spans))


def _report_heading_target(line: str) -> tuple[int, int] | None:
    if line.strip() == _LEGACY_OUTPUT_RULES_HEADING:
        start = line.index("Output Rules")
        return start, start + len("Output Rules")
    if heading := _BENIGN_OUTPUT_RULES_HEADING.fullmatch(line):
        return heading.span("target")
    return None


def _bounded_previous_nonblank_line(content: str, offset: int) -> tuple[str, bool]:
    """Return the prior nonblank logical line and whether it was complete."""
    window_start = max(0, offset - 512)
    parts = LOGICAL_LINE_BREAK.split(content[window_start:offset])
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].strip():
            return parts[index], index > 0 or window_start == 0
    return "", window_start == 0


def _bounded_next_nonblank_line(content: str, offset: int) -> tuple[str, bool]:
    """Return the next nonblank logical line and whether it was complete."""
    window_end = min(len(content), offset + 512)
    window = content[offset:window_end]
    cursor = 0
    for line_break in LOGICAL_LINE_BREAK.finditer(window):
        line = window[cursor : line_break.start()]
        if line.strip():
            return line, True
        cursor = line_break.end()
    if window_end == len(content):
        return window[cursor:], True
    return "", False


def _is_benign_print_rules_taxonomy(content: str, match: re.Match[str]) -> bool:
    """Return True only for a bounded declarative selector-taxonomy clause."""
    window_start = max(0, match.start() - 256)
    window_end = min(len(content), match.end() + 256)
    for candidate in _BENIGN_PRINT_RULES_TAXONOMY.finditer(content, window_start, window_end):
        if candidate.span("target") != match.span():
            continue
        candidate_end = candidate.end()
        if (
            candidate_end != len(content)
            and LOGICAL_LINE_BREAK.match(content, candidate_end) is None
        ):
            continue

        if candidate_end != len(content):
            line_break = LOGICAL_LINE_BREAK.match(content, candidate_end)
            assert line_break is not None
            next_line, next_complete = _bounded_next_nonblank_line(content, line_break.end())
            if not next_complete:
                continue
            if _NEXT_LINE_REFERENCE.search(next_line):
                continue

        if candidate.start() == 0:
            return True

        previous_line, previous_complete = _bounded_previous_nonblank_line(
            content, candidate.start()
        )
        if not previous_complete:
            return False
        return _PRECEDING_DIRECTIVE.search(previous_line) is None
    return False


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for system prompt leakage patterns (P6–P8)."""
    prepared = prepare_analysis(content, file_type, lambda: None)
    return _analyze(content, file_path, prepared, SecurityTextView("raw", content))


def _analyze(
    content: str,
    file_path: str,
    prepared: _PreparedAnalysis,
    source_view: SecurityTextView,
    *,
    p6_only: bool = False,
) -> list[AnalyzerFinding]:
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.SYSTEM_PROMPT_LEAKAGE.value]

    for pattern, confidence in P6_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            if prepared.is_report_label(match, source_view):
                continue
            if _is_benign_print_rules_taxonomy(content, match):
                continue
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P6",
                    message="Direct Prompt Extraction",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    if p6_only:
        return findings
    for pattern, confidence in P7_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P7",
                    message="Indirect Prompt Extraction",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in P8_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P8",
                    message="Prompt Exfiltration via Tool",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run system_prompt_leakage patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
