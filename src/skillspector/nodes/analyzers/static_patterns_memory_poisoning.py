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

"""Static patterns: memory poisoning (MP1–MP3). Node and analyze() in one module.

Detects patterns where content is injected to persist in agent memory (MP1),
the context window is stuffed to displace legitimate content (MP2), or
agent memory/state is directly manipulated (MP3).

Framework: ASI06, AML.T0080.
"""

from __future__ import annotations

import re
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import LOGICAL_LINE_BREAK, get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_memory_poisoning"

# MP1: Persistent Context Injection — content that persists across interactions
MP1_PATTERNS = [
    (
        r"(?:always\s+)?remember\s+(?:this|that|the\s+following)\s+(?:for|in)\s+(?:all|every|future)\s+(?:interactions?|conversations?|sessions?|responses?)",
        0.85,
    ),
    (
        r"(?:from\s+now\s+on|henceforth|going\s+forward|permanently)\s*[,:]?\s*(?:always|you\s+must|you\s+should|you\s+will)",
        0.8,
    ),
    (
        r"(?:store|save|persist|retain|keep)\s+(?:this|the\s+following)\s+(?:in|to|across)\s+(?:your\s+)?(?:memory|context|state|knowledge)",
        0.85,
    ),
    (
        r"(?:add|inject|insert|embed|implant)\s+(?:this|the\s+following)\s+(?:into|to)\s+(?:your\s+)?(?:memory|context|knowledge|training)",
        0.85,
    ),
    (
        r"(?:update|modify|change|alter)\s+(?:your\s+)?(?:core|base|fundamental|default)\s+(?:behavior|personality|instructions?|rules?)",
        0.8,
    ),
    (
        r"(?:treat|consider|regard)\s+(?:this|the\s+following)\s+as\s+(?:a\s+)?(?:permanent|core|fundamental|base)\s+(?:rule|instruction|directive|principle)",
        0.8,
    ),
    (
        r"(?:this\s+is\s+)?(?:a\s+)?(?:permanent|persistent|lasting|enduring)\s+(?:instruction|directive|rule|override)",
        0.8,
    ),
    (
        r"(?:never\s+)?forget\s+(?:this|that|the\s+following)\s+(?:instruction|directive|rule|fact|principle|information)",
        0.65,
    ),
    (
        r"(?:internalize|absorb|integrate)\s+(?:this|the\s+following)\s+(?:into\s+)?(?:your\s+)?(?:behavior|responses?|knowledge)",
        0.75,
    ),
]

# MP2: Context Window Stuffing — filling context to displace content
MP2_CONTENT_PATTERNS = [
    (r"(.{2,20}?)\1{20,}", 0.8),
]
MP2_PROSE_PATTERNS = [
    (
        r"(?:repeat|duplicate|echo)\s+(?:this|the\s+following)\s+(?:\d{3,}|many|hundreds?|thousands?)\s+times?",
        0.85,
    ),
    (
        r"(?:fill|pad|stuff|flood|saturate)\s+(?:the\s+)?(?:context|memory|input|prompt|window|buffer)",
        0.85,
    ),
    (
        r"(?:generate|produce|output|write)\s+(?:\d{4,}|thousands?\s+of|millions?\s+of)\s+(?:words?|characters?|tokens?|lines?)",
        0.8,
    ),
    (
        r"(?:include|add|insert)\s+(?:enough|sufficient)\s+(?:text|content|padding|filler)\s+(?:to|until)\s+(?:fill|overflow|exhaust|push\s+out)",
        0.85,
    ),
    (
        r"(?:displace|push\s+out|overwrite|crowd\s+out|evict)\s+(?:the\s+)?(?:original|system|previous|existing|safety)\s+(?:instructions?|prompt|context|rules?)",
        0.9,
    ),
    (
        r"(?:exhaust|overflow|exceed)\s+(?:the\s+)?(?:context|token|memory)\s+(?:window|limit|budget|capacity)",
        0.8,
    ),
]
MP2_PATTERNS = MP2_CONTENT_PATTERNS + MP2_PROSE_PATTERNS

# MP3: Memory Manipulation — direct modification of agent memory/state
MP3_PATTERNS = [
    (
        r"(?:clear|reset|wipe|erase|delete|purge)\s+(?:your\s+)?(?:memory|context|state|history|conversation)",
        0.8,
    ),
    (
        r"(?:forget|discard|drop|abandon)\s+(?:all\s+)?(?:previous|prior|earlier|past)\s+(?:instructions?|context|conversation|messages?|rules?)",
        0.8,
    ),
    (
        r"(?:overwrite|replace|substitute|swap)\s+(?:your\s+)?(?:memory|context|state|instructions?|rules?)",
        0.85,
    ),
    (
        r"(?:modify|edit|change|alter|corrupt|tamper\s+with)\s+(?:your\s+)?(?:memory|state|context|stored|saved)\s+(?:data|information|content)",
        0.85,
    ),
    (
        r"(?:rewrite|redefine)\s+(?:your\s+)?(?:personality|identity|purpose|mission|role|character)",
        0.8,
    ),
    (
        r"(?:you\s+are\s+no\s+longer|stop\s+being|cease\s+to\s+be)\s+(?:a\s+)?(?:\w+\s+){0,3}(?:assistant|helper|agent|bot)",
        0.75,
    ),
    (
        r"(?:your\s+)?(?:new|updated|revised|changed)\s+(?:personality|identity|name|role|purpose|mission)\s+is",
        0.8,
    ),
    (
        r"(?:inject|insert|plant)\s+(?:false|fake|fabricated|malicious)\s+(?:memories?|information|context|data|history)",
        0.9,
    ),
    (
        r"(?:poison|contaminate|corrupt|taint)\s+(?:your\s+)?(?:memory|context|state|knowledge|training)",
        0.9,
    ),
    (
        r"(?:pretend|act\s+as\s+if|believe)\s+(?:that\s+)?(?:your\s+)?(?:previous|past)\s+(?:conversation|context|interaction)\s+(?:was|included|contained)",
        0.7,
    ),
]

# CRLF is one break; backtracking must not turn it into a blank paragraph.
_LOGICAL_BREAK = rf"(?>{LOGICAL_LINE_BREAK.pattern})"
_BENIGN_RESET_STATE_COVERAGE = re.compile(
    rf"(?:\A|{_LOGICAL_BREAK})"
    r"[ \t]*(?:-[ \t]+\*\*Incomplete[ \t]+state[ \t]+coverage\*\*[ \t]+"
    r"(?:—|--|-)[ \t]+)?"
    r"(?:a[ \t]+)?state[ \t]+machine[ \t]+or[ \t]+(?:a[ \t]+)?lookup"
    r"[ \t]+missing[ \t]+its[ \t]+initial[ \t]*/[ \t]*"
    r"(?P<target>reset[ \t]+state)"
    r"[ \t]*,[ \t]+its[ \t]+miss[ \t]*/[ \t]*default[ \t]+case"
    r"[ \t]*,[ \t]+or[ \t]+a[ \t]+transition[ \t]+for[ \t]+some[ \t]+state"
    r"[ \t]+(?:×|x)[ \t]+input[ \t]+\([ \t]*an[ \t]+implicit[ \t]+"
    r"[\"'`]otherwise[\"'`][ \t]*\)[ \t]*\.[ \t]*"
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
    r"|\b(?:the|this|that|these|those|same)[ \t]+(?:state|memory|context|history|"
    r"conversation|operations?|actions?)\b"
    r"|\b(?:do|execute|perform|apply|follow|obey|invoke|run|use|reset|clear|wipe|"
    r"erase|overwrite|replace|swap|modify|change|corrupt|rewrite|inject|poison|"
    r"store|save|persist|retain|keep|internalize|set|enter|switch)[ \t]+that\b",
    re.IGNORECASE,
)

_TIMESTAMP_CONTEXT_DESCRIPTION = re.compile(
    rf"(?:\A|{_LOGICAL_BREAK})"
    r"[ \t]*//[ \t]*(?:✅[ \t]*)?(?:GOOD:[ \t]*)?"
    r"(?P<target>clear[ \t]+context)[ \t]+(?:with|using)[ \t]+"
    r"(?P<format>%[0-9]{0,3}t)[ \t]*\.?[ \t]*"
    rf"(?=\Z|{_LOGICAL_BREAK})",
    re.IGNORECASE,
)
_TIMESTAMP_DISPLAY = re.compile(
    r'[ \t]*\$display[ \t]*\([ \t]*"'
    r'(?P<format_string>(?:[^"\\\r\n\v\f\x85\u2028\u2029]|\\[^\r\n\v\f\x85\u2028\u2029])*)'
    r'"[ \t]*,[ \t]*\$(?:realtime|time)[ \t]*\)[ \t]*;[ \t]*'
    r"(?://[^\r\n\v\f\x85\u2028\u2029]*)?"
)
_TIMESTAMP_OWNERSHIP_PATTERN = (
    r"\b(?:instructions?|directives?|requirements?|orders?)\b"
    r"|\b(?:the|this|that|these|those|next|following|above|below|displayed|shown|"
    r"described|listed|same|attached)[ \t]+"
    r"(?:comments?|examples?|operations?|actions?|steps?|instructions?|directives?|lines?|snippets?|samples?)\b"
    r"|\b(?:comments?|examples?|operations?|actions?|steps?|instructions?|directives?|lines?|snippets?|samples?)"
    r"[ \t]+(?:above|below|earlier|prior|previous|preceding|following|next)\b"
    r"|\b(?:following|next|above|below|this|that)\b[^\r\n]{0,80}"
    r"\b(?:order|directive|instruction|command)\b"
)
_TIMESTAMP_OWNERSHIP_REFERENCE = re.compile(_TIMESTAMP_OWNERSHIP_PATTERN, re.IGNORECASE)
_TIMESTAMP_AUTHORITY_HEADING = re.compile(
    r"(?:[A-Za-z]+[ \t]+)*"
    r"(?:commands?|instructions?|directions?|directives?|orders?|actions?|operations?|steps?|tasks?)"
    r"(?=[ \t:]|$)[^:\r\n]{0,80}:[ \t]*",
    re.IGNORECASE,
)
_TIMESTAMP_DESCRIPTION_DIRECTIVE = re.compile(
    r"\b(?:you|agents?|assistants?|models?|llms?|bots?|must|shall|should|"
    r"required|mandatory)\b"
    r"|\byour[ \t]+(?:memory|context|state|history|conversation|task|objective|"
    r"mission|instructions?)\b"
    # Nearby memory targets can redefine what the comment's "context" means.
    r"|\b(?:conversation|memory|history|chat|transcript|dialogue)\b"
    r"|\b(?:follow|obey|apply|execute|perform|do|carry[ \t]+out|act[ \t]+(?:on|upon))[ \t]+"
    r"(?:(?:the[ \t]+)?(?:following|next|above|below)[ \t]+)?"
    r"(?:this|that|it|these|those|comments?|instructions?)\b"
    r"|\b(?:follow|obey|apply|execute|perform|do|carry[ \t]+out|act[ \t]+(?:on|upon))\b"
    r"[^\r\n\v\f\x85\u2028\u2029]{0,160}"
    r"\b(?:described|displayed|documented|shown|listed|comments?|examples?|"
    r"operations?|actions?|steps?|instructions?|directives?|lines?|snippets?|samples?)\b"
    r"|\b(?:task|instructions?|directives?)[ \t]*:"
    r"|\bbefore[ \t]+(?:replying|responding|answering)\b" + "|" + _TIMESTAMP_OWNERSHIP_PATTERN,
    re.IGNORECASE,
)
_TIMESTAMP_BACK_REFERENCE = re.compile(
    r"\b(?:described|displayed|documented|shown|listed|above|earlier|prior|"
    r"previous|preceding|foregoing|same)[ \t]+"
    r"(?:comments?|examples?|operations?|actions?|steps?|instructions?)\b"
    r"|\b(?:comments?|examples?|operations?|actions?|steps?|instructions?)"
    r"[ \t]+(?:above|earlier|prior|previous|preceding|foregoing)\b"
    r"|\b(?:follow|obey|apply|execute|perform|do|carry[ \t]+out|act[ \t]+(?:on|upon)|use|run|invoke)\b"
    r"[^\r\n\v\f\x85\u2028\u2029]{0,160}"
    r"\b(?:this|that|it|these|those|above|earlier|prior|previous|preceding|"
    r"foregoing|same)\b"
    r"|\bcontext\b[^\r\n\v\f\x85\u2028\u2029]{0,160}"
    r"\b(?:conversation|memory|history|chat|transcript|dialogue)\b"
    r"|\b(?:conversation|memory|history|chat|transcript|dialogue)\b"
    r"[^\r\n\v\f\x85\u2028\u2029]{0,160}\bcontext\b",
    re.IGNORECASE,
)
_TIMESTAMP_REFERENCE_CUE = re.compile(
    r"\b(?:follow|obey|apply|execute|perform|do|carry|act|use|run|invoke|context|"
    r"described|displayed|documented|shown|listed|above|earlier|prior|previous|"
    r"preceding|foregoing|same|commands?|instructions?|directives?|requirements?|orders?)\b",
    re.IGNORECASE,
)
_CODE_FENCE_LINE = re.compile(r"[ \t]*(?:`{3,}|~{3,})[ \t]*")
_CODE_FENCE_OPENER = re.compile(
    r"[ \t]*(?:`{3,}|~{3,})[ \t]*(?:systemverilog|verilog)?[ \t]*", re.IGNORECASE
)
_CODE_FENCE_MARKER = re.compile(r"[ \t]*(?:`{3,}|~{3,})")

_LAYOUT_CHAR_RANGES = (
    (0x2500, 0x257F),
    (0x2580, 0x259F),
)
_LAYOUT_ASCII_CHARS = frozenset("|-_=+")
_MAX_LAYOUT_ONLY_SPAN = 256


def _is_layout_only_span(span: str, max_cosmetic_span: int = _MAX_LAYOUT_ONLY_SPAN) -> bool:
    """Return True when a captured MP2 span is only layout glyphs and whitespace."""
    if len(span) > max_cosmetic_span:
        return False
    compact = re.sub(r"\s", "", span)
    if not compact:
        return True
    if any(ch.isalnum() for ch in compact):
        return False
    if any(ch.isalpha() or ch.isdigit() for ch in compact):
        return False
    for ch in compact:
        if ch in _LAYOUT_ASCII_CHARS:
            continue
        codepoint = ord(ch)
        if not any(start <= codepoint <= end for start, end in _LAYOUT_CHAR_RANGES):
            return False
    return True


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


def _is_benign_reset_state_coverage(content: str, match: re.Match[str]) -> bool:
    """Return True only for the reported state-coverage enumeration shape."""
    window_start = max(0, match.start() - 256)
    window_end = min(len(content), match.end() + 256)
    for candidate in _BENIGN_RESET_STATE_COVERAGE.finditer(content, window_start, window_end):
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


def _bounded_timestamp_following_context(content: str, offset: int) -> str | None:
    """Inspect through a nearby fence and following paragraph, without truncation."""
    window_end = min(len(content), offset + 512)
    window = content[offset:window_end]
    cursor = 0
    closed_fence = False
    following_paragraph = False
    for line_break in LOGICAL_LINE_BREAK.finditer(window):
        line = window[cursor : line_break.start()]
        cursor = line_break.end()
        if closed_fence:
            # A separate code block owns its contents, not this description.
            # Keep its opening line visible in case that line is an instruction.
            if _CODE_FENCE_MARKER.match(line):
                return window[:cursor]
            if line.strip():
                following_paragraph = True
            elif following_paragraph and window_end != len(content):
                return window[:cursor]
        if _CODE_FENCE_LINE.fullmatch(line):
            closed_fence = True
    return window if window_end == len(content) else None


def _normalize_timestamp_guard(content: str) -> str:
    """Join wrapped directives for boolean guards without changing source evidence."""
    normalized = " ".join(
        re.sub(
            r"^[ \t]*(?:(?:>|//)[ \t]*)*"
            r"(?:(?:[-+*]|[0-9]{1,9}[.)])[ \t]+)?(?:#{1,6}[ \t]+)?",
            "",
            line,
        )
        for line in LOGICAL_LINE_BREAK.split(content)
    )
    return re.sub(r"[`*_\[\]]", "", normalized)


def _has_timestamp_back_reference(content: str, offset: int) -> bool:
    """Keep explicit reuse visible across paragraph and independent-block boundaries."""
    window_end = min(len(content), offset + 512)
    window = content[offset:window_end]
    normalized = _normalize_timestamp_guard(window)
    if (
        _TIMESTAMP_BACK_REFERENCE.search(normalized)
        or _TIMESTAMP_OWNERSHIP_REFERENCE.search(normalized)
        or _has_timestamp_authority_heading(window)
    ):
        return True
    if window_end != len(content) and LOGICAL_LINE_BREAK.match(content, window_end) is None:
        # An unfinished reference cannot establish that the description is benign.
        last_paragraph = re.split(rf"{_LOGICAL_BREAK}[ \t]*{_LOGICAL_BREAK}", window)[-1]
        return (
            _TIMESTAMP_REFERENCE_CUE.search(_normalize_timestamp_guard(last_paragraph)) is not None
        )
    return False


def _has_timestamp_authority_heading(content: str) -> bool:
    """Require a complete heading; URL or metadata colons cannot confer authority."""
    regions = (
        *LOGICAL_LINE_BREAK.split(content),
        *re.split(rf"{_LOGICAL_BREAK}[ \t]*{_LOGICAL_BREAK}", content),
    )
    return any(
        _TIMESTAMP_AUTHORITY_HEADING.fullmatch(_normalize_timestamp_guard(paragraph).strip())
        for paragraph in regions
    )


def _is_benign_timestamp_context_description(content: str, match: re.Match[str]) -> bool:
    """Own a descriptive comment only when its example actually prints a timestamp."""
    window_start = max(0, match.start() - 256)
    window_end = min(len(content), match.end() + 256)
    for candidate in _TIMESTAMP_CONTEXT_DESCRIPTION.finditer(content, window_start, window_end):
        if candidate.span("target") != match.span():
            continue
        if (
            candidate.end() != len(content)
            and LOGICAL_LINE_BREAK.match(content, candidate.end()) is None
        ):
            continue

        previous_line, previous_complete = _bounded_previous_nonblank_line(
            content, candidate.start()
        )
        if not previous_complete:
            continue
        # A fence opener cannot hide the instruction introducing its comments.
        if _CODE_FENCE_OPENER.fullmatch(previous_line):
            opener_start = content.rfind(
                previous_line, max(0, candidate.start() - 512), candidate.start()
            )
            previous_line, previous_complete = _bounded_previous_nonblank_line(
                content, opener_start
            )
            if not previous_complete:
                continue
        # Direct reuse of the immediately preceding text cannot grant ownership
        # to a descriptive example, regardless of the directive's action verb.
        if _PRECEDING_DIRECTIVE.search(_normalize_timestamp_guard(previous_line)):
            continue
        preceding = content[max(0, candidate.start() - 512) : candidate.start()]
        nearest_paragraph = re.split(rf"{_LOGICAL_BREAK}[ \t]*{_LOGICAL_BREAK}", preceding)[-1]
        if any(
            reference.group(0).strip() != ":"
            for reference in _PRECEDING_DIRECTIVE.finditer(
                _normalize_timestamp_guard(nearest_paragraph)
            )
        ):
            continue
        if _TIMESTAMP_DESCRIPTION_DIRECTIVE.search(
            _normalize_timestamp_guard(preceding)
        ) or _has_timestamp_authority_heading(preceding):
            continue

        next_line, next_complete = _bounded_next_nonblank_line(content, candidate.end())
        display = _TIMESTAMP_DISPLAY.fullmatch(next_line) if next_complete else None
        if display is None:
            continue
        # %% is a literal percent, not timestamp formatting evidence.
        if (
            re.search(
                r"(?<!%)" + re.escape(candidate.group("format")), display.group("format_string")
            )
            is None
        ):
            continue
        if _TIMESTAMP_DESCRIPTION_DIRECTIVE.search(next_line) or _NEXT_LINE_REFERENCE.search(
            next_line
        ):
            continue
        display_start = content.find(
            next_line, candidate.end(), min(len(content), candidate.end() + 512)
        )
        display_end = display_start + len(next_line)
        if _has_timestamp_back_reference(content, display_end):
            continue
        following = _bounded_timestamp_following_context(content, display_end)
        if (
            following is None
            or _NEXT_LINE_REFERENCE.search(_normalize_timestamp_guard(following))
            or _TIMESTAMP_DESCRIPTION_DIRECTIVE.search(_normalize_timestamp_guard(following))
            or _has_timestamp_authority_heading(following)
        ):
            continue
        return True
    return False


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for memory poisoning patterns (MP1–MP3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.MEMORY_POISONING.value]

    for pattern, confidence in MP1_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="MP1",
                    message="Persistent Context Injection",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in MP2_PATTERNS:
        matches = (
            static_runner.iter_paragraph_matches
            if (pattern, confidence) in MP2_PROSE_PATTERNS
            else re.finditer
        )
        for match in matches(pattern, content, re.IGNORECASE | re.MULTILINE):
            span = match.group(0)
            if _is_layout_only_span(span):
                continue
            non_ws_chars = set(span) - {" ", "\t", "\n", "\r"}
            if len(non_ws_chars) <= 1 and not any(c in span for c in (" ", "\t")):
                continue
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="MP2",
                    message="Context Window Stuffing",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in MP3_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            if _is_benign_reset_state_coverage(
                content, match
            ) or _is_benign_timestamp_context_description(content, match):
                continue
            line_num = get_line_number(content, match.start())
            context_text = ctx(match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="MP3",
                    message="Memory Manipulation",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context_text,
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run memory_poisoning patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
