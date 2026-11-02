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
from collections.abc import Iterator

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import LOGICAL_LINE_BREAK, get_context, get_line_number
from .pattern_defaults import PatternCategory
from .prohibition_context import is_directly_prohibited

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

# Report explicit state adoption at the instruction, even when its source
# is a separate linked document with an otherwise benign hardware state.
# The bounded gap accepts a wrapped reference and dots inside link paths,
# but cannot borrow the target from a later sentence or paragraph. Bounded
# inline decorators keep the original source span and cannot hide an action.
_ADOPTION_MARKUP = r"[*_`]{0,3}"
_ADOPTION_DIALOGUE = (
    rf"(?:conversation|chat|dialogue){_ADOPTION_MARKUP}"
    rf"(?:\s+{_ADOPTION_MARKUP}(?:state|history|context|transcript))?"
)
_ADOPTION_ACTION_SOURCE = (
    rf"(?<!\w){_ADOPTION_MARKUP}(?:treat|consider|regard|use|adopt){_ADOPTION_MARKUP}\s+"
    rf"(?:(?:the|this|that){_ADOPTION_MARKUP}\s+)?"
    rf"(?:\[\s*)?{_ADOPTION_MARKUP}(?:reset\s+)?state{_ADOPTION_MARKUP}(?!\w)"
)
_MP3_STATE_ADOPTION_PATTERN = (
    _ADOPTION_ACTION_SOURCE
    # A later adoption action owns its own target and prohibition context.
    + rf"(?:(?!{_ADOPTION_ACTION_SOURCE})(?:[^.!?]|\.(?=\S))){{0,512}}?\bas\s+"
    rf"{_ADOPTION_MARKUP}(?:your{_ADOPTION_MARKUP}\s+{_ADOPTION_MARKUP}"
    rf"(?:{_ADOPTION_DIALOGUE}|memory)"
    rf"|(?:the{_ADOPTION_MARKUP}\s+)?{_ADOPTION_MARKUP}"
    rf"(?:agent|assistant|llm|(?:ai|language)\s+model)(?:'s|’s){_ADOPTION_MARKUP}\s+"
    rf"{_ADOPTION_MARKUP}(?:{_ADOPTION_DIALOGUE}|memory|context|state|history|transcript))"
    rf"{_ADOPTION_MARKUP}(?!\w)(?:[.!?][*_`]{{1,3}})?"
)
_ADOPTION_PARAGRAPH_BOUNDARY = re.compile(
    rf"(?>{LOGICAL_LINE_BREAK.pattern})[ \t]*(?:>[ \t]*)*"
    rf"(?>{LOGICAL_LINE_BREAK.pattern})"
)

# MP3: Memory Manipulation — direct modification of agent memory/state
MP3_PATTERNS = [
    (_MP3_STATE_ADOPTION_PATTERN, 0.85),
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

# Require a nominal state owned by a hardware subject, not just a nearby FSM label.
_HARDWARE_RESET_STATE = re.compile(
    r"\b(?:(?:finite[ \t-]+)?state[ \t-]+machines?|fsms?)[ \t]+"
    r"(?:takes?|reach(?:es)?|enters?|returns?[ \t]+to)[ \t]+(?:its|their)[ \t]+"
    r"(?P<target>reset[ \t]+state)\b",
    re.IGNORECASE,
)
_HARDWARE_CLOCK = re.compile(r"\bclocks?\b", re.IGNORECASE)
_HARDWARE_AGENT_REFERENCE = re.compile(
    r"\b(?:you|your|agents?|assistants?|models?|llms?|bots?|conversation|memory|"
    r"context|history|chat|transcript|dialogue|prompt|instructions?|directives?)\b"
    r"|\bbefore[ \t]+(?:replying|responding|answering)\b",
    re.IGNORECASE,
)

_LAYOUT_CHAR_RANGES = (
    (0x2500, 0x257F),
    (0x2580, 0x259F),
)
_LAYOUT_ASCII_CHARS = frozenset("|-_=+")
_MAX_LAYOUT_ONLY_SPAN = 256


def _iter_state_adoption_matches(content: str) -> Iterator[re.Match[str]]:
    """Keep adoption and its target in one paragraph, including blockquote blanks."""
    pattern = re.compile(_MP3_STATE_ADOPTION_PATTERN, re.IGNORECASE | re.MULTILINE)
    start = 0
    for boundary in _ADOPTION_PARAGRAPH_BOUNDARY.finditer(content):
        yield from pattern.finditer(content, start, boundary.start())
        start = boundary.end()
    yield from pattern.finditer(content, start)


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


def _is_benign_hardware_reset_state(content: str, match: re.Match[str]) -> bool:
    """Own only a clocked hardware subject's nominal reset state on a complete line."""
    if re.fullmatch(r"reset[ \t]+state", match.group(), re.IGNORECASE) is None:
        return False

    window_start = max(0, match.start() - 512)
    window_end = min(len(content), match.end() + 512)
    preceding_breaks = list(LOGICAL_LINE_BREAK.finditer(content, window_start, match.start()))
    line_start = preceding_breaks[-1].end() if preceding_breaks else window_start
    following_break = LOGICAL_LINE_BREAK.search(content, match.end(), window_end)
    line_end = following_break.start() if following_break else window_end
    if (not preceding_breaks and window_start != 0) or (
        following_break is None and window_end != len(content)
    ):
        return False

    if _HARDWARE_CLOCK.search(content, line_start, line_end) is None:
        return False
    if not any(
        candidate.span("target") == match.span()
        for candidate in _HARDWARE_RESET_STATE.finditer(content, line_start, line_end)
    ):
        return False

    _, previous_complete = _bounded_previous_nonblank_line(content, line_start)
    _, next_complete = _bounded_next_nonblank_line(content, line_end)
    if not previous_complete or not next_complete:
        return False

    surrounding = _normalize_timestamp_guard(
        content[max(0, line_start - 512) : min(len(content), line_end + 512)]
    )
    # Preserve agent-state aliases and explicit instructions that reuse a description.
    return (
        _HARDWARE_AGENT_REFERENCE.search(surrounding) is None
        and _TIMESTAMP_BACK_REFERENCE.search(surrounding) is None
    )


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


# MP3 (write variant): a write operation whose destination resolves to a known
# agent-memory file or directory. MP1/MP3 prose patterns score the injection
# *text*; they cannot see a write *operation*, so a test hook that appends to
# MEMORY.md scores 0/100 SAFE (issue #879). This scores the write itself: the
# persistence mechanism is the finding, whatever the payload bytes are.
#
# Calibration: the signal is the *destination* being memory the agent loads as
# its own context, not file I/O as such; and the carrier is scoped to files
# the developer toolchain runs without being asked (test files, conftest.py),
# so an explicitly invoked helper that writes a report stays quiet.
_AGENT_MEMORY_FILENAMES = frozenset({"memory.md", "claude.md", "agents.md", "soul.md"})
_AGENT_MEMORY_DIR_SEGMENTS = frozenset({".claude", ".cursor", ".agents"})

# Basename shapes the developer toolchain auto-runs: pytest/unittest discovery
# (test_*.py, *_test.py, conftest.py) and JS/TS runners (*.test.*, *.spec.*).
_AUTO_RUN_CARRIER_RE = re.compile(
    r"conftest\.py|test_.*\.pyw?|.*_test\.pyw?|.*\.(?:test|spec)\.[a-z0-9]+",
    re.IGNORECASE,
)

# Keep escaped characters disjoint from ordinary characters so an unterminated
# destination cannot trigger exponential backtracking over backslashes.
_PY_STRING_LITERAL = r"(?P<prefix>[A-Za-z]*)(?P<q>[\"'])(?P<dest>(?:\\.|(?!(?P=q)|\\).)*)(?P=q)"
_TS_STRING_LITERAL = r"(?P<q>[\"'`])(?P<dest>(?:\\.|(?!(?P=q)|\\).)*)(?P=q)"

# Write sinks that take a literal destination. open() requires an explicit
# write mode so reads stay quiet; shutil.copy* resolve the destination, which
# is the second positional argument.
_PY_WRITE_SINKS = [
    re.compile(
        r"(?:\bio\s*\.\s*)?\bopen\s*\(\s*"
        + _PY_STRING_LITERAL
        + r"\s*,\s*(?:mode\s*=\s*)?(?P<mq>[\"'])(?P<mode>[^\"']*)(?P=mq)"
    ),
    re.compile(r"\bPath\s*\(\s*" + _PY_STRING_LITERAL + r"\s*\)\s*\.\s*write_(?:text|bytes)\s*\("),
    re.compile(
        r"\bshutil\s*\.\s*copy(?:2|file)?\s*\(\s*[^()]*(?:\([^()]*\)[^()]*)*,\s*"
        + _PY_STRING_LITERAL
    ),
]
_TS_WRITE_SINKS = [
    re.compile(
        r"(?:\bfs\s*\.\s*(?:promises\s*\.\s*)?)?"
        r"\b(?:writeFileSync|appendFileSync|writeFile|appendFile)\s*\(\s*" + _TS_STRING_LITERAL
    ),
    re.compile(r"\bfs\s*\.\s*createWriteStream\s*\(\s*" + _TS_STRING_LITERAL),
]


def _is_agent_memory_destination(raw_destination: str, prefix: str = "") -> bool:
    """Return True when a write destination is an agent-memory file or directory."""
    # Literal path only: interpolation in directory segments does not change
    # which well-known file is targeted, so f"{base}/MEMORY.md" still counts.
    # A raw prefix (r, rf) keeps backslashes literal: decoding escapes there
    # would eat the separators, so r"C:\Users\alice\MEMORY.md" would collapse
    # to "C:UsersaliceMEMORY.md" and lose every path segment. Only non-raw
    # literals get escape decoding; both forms then normalize separators.
    if "r" in prefix.lower():
        unescaped = raw_destination
    else:
        unescaped = re.sub(r"\\(.)", r"\1", raw_destination, flags=re.DOTALL)
    normalized = unescaped.replace("\\", "/").strip()
    segments = [s for s in normalized.split("/") if s not in ("", ".", "..")]
    if not segments:
        return False
    lowered = [segment.lower() for segment in segments]
    if lowered[-1] in _AGENT_MEMORY_FILENAMES:
        return True
    return any(segment in _AGENT_MEMORY_DIR_SEGMENTS for segment in lowered)


def _iter_memory_file_writes(content: str, file_type: str) -> Iterator[tuple[re.Match[str], str]]:
    """Yield (match, destination) for writes to agent-memory destinations."""
    if file_type == "python":
        sinks = _PY_WRITE_SINKS
    elif file_type in ("typescript", "javascript"):
        sinks = _TS_WRITE_SINKS
    else:
        return
    for sink in sinks:
        for match in sink.finditer(content):
            destination = match.group("dest")
            prefix = match.groupdict().get("prefix") or ""
            if not _is_agent_memory_destination(destination, prefix):
                continue
            mode = match.groupdict().get("mode")
            if mode is not None and not any(char in "wxa+" for char in mode):
                continue
            yield match, destination


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
        mp3_matches = (
            _iter_state_adoption_matches(content)
            if pattern == _MP3_STATE_ADOPTION_PATTERN
            else static_runner.iter_paragraph_matches(
                pattern, content, re.IGNORECASE | re.MULTILINE
            )
        )
        for match in mp3_matches:
            if pattern == _MP3_STATE_ADOPTION_PATTERN and is_directly_prohibited(
                content, match.start(), match.end(), allow_clause_prefix=True
            ):
                continue
            if _is_benign_reset_state_coverage(content, match) or _is_benign_hardware_reset_state(
                content, match
            ):
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
    # MP3 write variant: score the write to agent memory, not the written text.
    basename = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    if (
        file_type in ("python", "typescript", "javascript")
        and _AUTO_RUN_CARRIER_RE.fullmatch(basename) is not None
    ):
        for write_match, destination in _iter_memory_file_writes(content, file_type):
            line_num = get_line_number(content, write_match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="MP3",
                    message="Memory Manipulation",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=0.85,
                    tags=tag,
                    context=ctx(write_match.start()),
                    matched_text=write_match.group(0)[:200],
                    complete_match=write_match.group(0),
                    remediation=(
                        "An auto-run file writes to an agent memory file "
                        f"({destination}). Such writes persist across agent "
                        "interactions without model involvement. Confirm the "
                        "destination is intentional and the written content is "
                        "trusted before installing this skill."
                    ),
                    evidence={
                        "destination": destination,
                        "carrier_basename": basename,
                        "carrier_kind": "auto_run",
                    },
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run memory_poisoning patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
