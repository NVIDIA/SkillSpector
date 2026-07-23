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

"""Static patterns: prompt injection (P1–P4, P9). Node and analyze() in one module."""

from __future__ import annotations

import fnmatch
import re
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory
from .whitespace_padding import (
    VERTICAL_HIGH_SEVERITY_LINES,
    ZERO_WIDTH_CHARS,
    detect_whitespace_padding,
)

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_prompt_injection"

# Generated/vendored filename globs for which the P9 whitespace-padding signal is
# skipped (these legitimately carry large whitespace runs). Applies ONLY to P9.
_P9_SKIP_GLOBS = (
    "*.min.js",
    "*.min.css",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "*.svg",
    "*.map",
)


def _is_p9_skipped_path(file_path: str) -> bool:
    """Return True when *file_path* is a generated/vendored file P9 should skip."""
    name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pattern) for pattern in _P9_SKIP_GLOBS)


# P1: Instruction Override
P1_PATTERNS = [
    (r"ignore\s+(?:all\s+)?previous\s+instructions?", 0.8),
    (r"ignore\s+(?:all\s+)?(?:safety|security)\s+(?:rules?|constraints?|guidelines?)", 0.9),
    (r"override\s+(?:safety|security|system)", 0.9),
    (r"bypass\s+(?:safety|security|restrictions?|constraints?)", 0.9),
    (r"disregard\s+(?:all\s+)?(?:previous|safety|security)", 0.8),
    (r"forget\s+(?:all\s+)?(?:previous|your)\s+instructions?", 0.8),
    (r"you\s+are\s+now\s+(?:in\s+)?(?:jailbreak|unrestricted|unfiltered)\s+mode", 0.95),
    (r"enable\s+(?:developer|debug|admin|root)\s+mode", 0.7),
    (r"your\s+new\s+(?:instructions?|rules?)\s+are", 0.6),
    (r"do\s+not\s+follow\s+(?:any|your|the)\s+(?:rules?|guidelines?)", 0.9),
    (r"you\s+must\s+(?:always\s+)?ignore", 0.7),
]
_ZERO_WIDTH_PATTERN = "[" + "".join(sorted(ZERO_WIDTH_CHARS)) + "]"
P2_PATTERNS = [
    (r"<!--.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?-->", 0.7),
    (r"\[//\]:\s*#\s*\(.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?\)", 0.8),
    # Zero-width character class built from the shared ZERO_WIDTH_CHARS constant so
    # P2 and the P9 detector cannot drift (same five code points as before).
    (_ZERO_WIDTH_PATTERN, 0.6),
    (r"[\u202a-\u202e\u2066-\u2069]", 0.85),
    (r"data:text/plain;base64,[A-Za-z0-9+/=]{50,}", 0.7),
]
# P3: Exfiltration Commands
P3_PATTERNS = [
    (
        r"(?:send|transmit|upload|post|forward)\s+(?:the\s+)?(?:conversation|context|chat|history|messages?)\s+to",
        0.9,
    ),
    (
        r"(?:send|transmit|upload|post)\s+(?:all\s+)?(?:user\s+)?(?:data|information|input|files?)\s+to\s+(?:https?://|external)",
        0.9,
    ),
    (
        r"(?:copy|extract|collect)\s+(?:the\s+)?(?:conversation|context|session)\s+(?:and\s+)?(?:send|upload|transmit)",
        0.85,
    ),
    (
        r"(?:call|notify|ping|hit)\s+(?:this\s+)?(?:webhook|endpoint|url|api)\s+with\s+(?:the\s+)?(?:user|conversation|context)",
        0.8,
    ),
    (
        r"include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context|history)\s+in\s+(?:the\s+)?(?:request|payload|body)",
        0.75,
    ),
    (r"(?:silently|quietly|secretly|covertly)\s+(?:send|transmit|log|record)", 0.9),
    (
        r"without\s+(?:telling|informing|notifying)\s+(?:the\s+)?user.*?(?:send|transmit|upload)",
        0.9,
    ),
]
# P4: Behavior Manipulation
P4_PATTERNS = [
    (
        r"always\s+(?:recommend|suggest|prefer|choose|use)\s+(?:\w+\s+){0,3}(?:over|instead\s+of|rather\s+than)",
        0.7,
    ),
    (r"never\s+(?:recommend|suggest|mention|tell|warn|inform)\s+(?:the\s+)?(?:user|them)", 0.75),
    (
        r"(?:prioritize|prefer)\s+(?:\w+\s+){0,3}(?:over|above)\s+(?:user\s+)?(?:safety|security|privacy|preferences?)",
        0.85,
    ),
    (
        r"(?:subtly|gradually|slowly)\s+(?:steer|guide|push|nudge)\s+(?:the\s+)?(?:user|conversation)",
        0.8,
    ),
    (
        r"(?:without\s+)?(?:the\s+)?user\s+(?:knowing|realizing|noticing).*?(?:change|modify|alter|influence)",
        0.85,
    ),
    (
        r"(?:gain|build|establish)\s+(?:the\s+)?(?:user(?:'s)?)\s+trust\s+(?:and\s+then|before|so\s+that)",
        0.75,
    ),
    (r"(?:appear|seem|act)\s+(?:helpful|friendly|trustworthy)\s+(?:while|but|then)", 0.7),
    (
        r"(?:always|never)\s+include\s+(?:this|the\s+following)\s+in\s+(?:every|all)\s+(?:response|output|reply)",
        0.7,
    ),
    (
        r"append\s+(?:this|the\s+following)\s+to\s+(?:every|all|each)\s+(?:response|message|output)",
        0.75,
    ),
]

# P2 (extended): Unicode "Tags" block (U+E0000–U+E007F) — "ASCII smuggling".
# Tag characters U+E0020–U+E007E map 1:1 to printable ASCII (U+E0041 == tag "A")
# and render as nothing in virtually every font/editor/terminal, so an entire
# hidden instruction can be embedded invisibly inside otherwise-benign text:
# invisible to a human reviewer, but read as literal text by the consuming LLM.
# This is a distinct codepoint range from the bidi/Trojan-Source class already in
# P2 (U+202A–U+202E / U+2066–U+2069).
_TAG_BLOCK = (0xE0000, 0xE007F)
# The only legitimate use of tag characters is an emoji tag sequence (RGI
# subdivision flags: an emoji base U+1F3F4 followed by tag chars and terminated
# by U+E007F CANCEL TAG — e.g. the Scotland/Wales/England flags). Strip
# well-formed sequences before flagging so those emoji are not false positives.
#
# The carve-out is deliberately narrow: the tag payload must be a short
# ISO-3166-2-style subdivision code, i.e. 2–6 tag characters that each map to a
# lowercase ASCII letter (U+E0061–U+E007A) or digit (U+E0030–U+E0039). The only
# RGI-recommended values are "gbeng"/"gbsct"/"gbwls", and Unicode caps
# subdivision codes at 6 chars, so this admits every real flag. A smuggled ASCII
# instruction lands in U+E0020–U+E007E and contains spaces, ';', '/', uppercase,
# or simply runs longer than 6 chars — none of which match here — so wrapping a
# payload as 🏴 <tags> U+E007F can no longer launder it past detection.
_EMOJI_TAG_SEQUENCE = re.compile(
    "\U0001f3f4[\U000e0030-\U000e0039\U000e0061-\U000e007a]{2,6}\U000e007f"
)


_EMOJI_MODIFIERS = range(0x1F3FB, 0x1F400)
_VARIATION_SELECTORS = {0xFE0E, 0xFE0F}


def _is_emoji_base(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D)
    )


def _previous_emoji_base(content: str, offset: int) -> bool:
    i = offset - 1
    while i >= 0 and (
        ord(content[i]) in _VARIATION_SELECTORS or ord(content[i]) in _EMOJI_MODIFIERS
    ):
        i -= 1
    return i >= 0 and _is_emoji_base(content[i])


def _next_emoji_base(content: str, offset: int) -> bool:
    i = offset + 1
    while i < len(content) and ord(content[i]) in _VARIATION_SELECTORS:
        i += 1
    if i < len(content) and ord(content[i]) in _EMOJI_MODIFIERS:
        i += 1
    return i < len(content) and _is_emoji_base(content[i])


def _zero_width_match_is_safe_emoji_zwj(content: str, offset: int) -> bool:
    """Allow ZWJ only when it joins two emoji bases in an emoji sequence."""
    return (
        content[offset] == "\u200d"
        and _previous_emoji_base(content, offset)
        and _next_emoji_base(content, offset)
    )


def _first_smuggled_tag_offset(content: str) -> int | None:
    """Return the char offset of the first Unicode Tag character that is *not*
    part of a well-formed emoji tag sequence, or ``None`` if there is none."""
    if not any(_TAG_BLOCK[0] <= ord(ch) <= _TAG_BLOCK[1] for ch in content):
        return None
    safe_spans = [(m.start(), m.end()) for m in _EMOJI_TAG_SEQUENCE.finditer(content)]
    for i, ch in enumerate(content):
        if _TAG_BLOCK[0] <= ord(ch) <= _TAG_BLOCK[1] and not any(
            start <= i < end for start, end in safe_spans
        ):
            return i
    return None


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for prompt injection patterns (P1–P4, P9)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.PROMPT_INJECTION.value]

    for pattern, confidence in P1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P1",
                    message="Instruction Override",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    if file_type in ("markdown", "other"):
        for pattern, confidence in P2_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                if pattern == _ZERO_WIDTH_PATTERN and _zero_width_match_is_safe_emoji_zwj(
                    content, match.start()
                ):
                    continue
                line_num = get_line_number(content, match.start())
                findings.append(
                    AnalyzerFinding(
                        rule_id="P2",
                        message="Hidden Instructions",
                        severity=Severity.HIGH,
                        location=loc(line_num),
                        confidence=confidence,
                        tags=tag,
                        context=ctx(match.start()),
                        matched_text=match.group(0)[:200],
                    )
                )
    for pattern, confidence in P3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P3",
                    message="Exfiltration Commands",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in P4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P4",
                    message="Behavior Manipulation",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )

    # P2 (extended): Unicode Tag-block "ASCII smuggling". Runs regardless of
    # file_type — invisible instructions are dangerous in scripts and config
    # files too, and the tag range never overlaps the BOM/zero-width codepoints
    # that the markdown-only block above guards against false positives.
    tag_offset = _first_smuggled_tag_offset(content)
    if tag_offset is not None:
        line_num = get_line_number(content, tag_offset)
        findings.append(
            AnalyzerFinding(
                rule_id="P2",
                message="Hidden Instructions (Unicode Tag / ASCII smuggling)",
                severity=Severity.HIGH,
                location=loc(line_num),
                confidence=0.9,
                tags=tag,
                context=ctx(tag_offset),
                matched_text=repr(content[tag_offset : tag_offset + 40]),
            )
        )

    # P9: Whitespace Padding (skipped for generated/vendored files).
    if not _is_p9_skipped_path(file_path):
        for run in detect_whitespace_padding(content, file_type=file_type):
            if run.kind == "vertical":
                confidence = 0.8 if run.followed_by_content else 0.6
                severity = (
                    Severity.HIGH
                    if run.followed_by_content and run.length >= VERTICAL_HIGH_SEVERITY_LINES
                    else Severity.MEDIUM
                )
            elif run.kind == "horizontal":
                confidence = 0.7
                severity = Severity.MEDIUM
            else:  # "block" or "ratio"
                confidence = 0.4
                severity = Severity.LOW
            findings.append(
                AnalyzerFinding(
                    rule_id="P9",
                    message="Whitespace Padding",
                    severity=severity,
                    location=loc(run.start_line),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(run.start_offset),
                    matched_text=run.summary,
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run prompt_injection patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
