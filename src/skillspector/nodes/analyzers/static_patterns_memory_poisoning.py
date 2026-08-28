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
MP2_PATTERNS = [
    (r"(.{2,20}?)\1{20,}", 0.8),
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

_LOGICAL_BREAK = rf"(?:{LOGICAL_LINE_BREAK.pattern})"
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


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for memory poisoning patterns (MP1–MP3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.MEMORY_POISONING.value]

    for pattern, confidence in MP1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                )
            )
    for pattern, confidence in MP2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                )
            )
    for pattern, confidence in MP3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            if _is_benign_reset_state_coverage(content, match):
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
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run memory_poisoning patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
