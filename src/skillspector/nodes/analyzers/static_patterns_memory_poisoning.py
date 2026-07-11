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

"""Static patterns: memory poisoning (MP1–MP4). Node and analyze() in one module.

Detects patterns where content is injected to persist in agent memory (MP1),
the context window is stuffed to displace legitimate content (MP2), agent
memory/state is directly manipulated (MP3), or a file is padded with
whitespace to push instructions below/past what a human reviewer sees (MP4).

Framework: ASI06, AML.T0080.
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

# MP4: Whitespace Padding Evasion — a run of whitespace long enough to push
# hidden instructions below or past what a human reviewer sees in an editor
# (blank-line runs, long in-line runs, or a file that is mostly padding).
# "Whitespace" here is not ASCII space/tab: it includes any Unicode
# whitespace category (`\s` already covers NBSP, line/paragraph separators,
# ideographic space, etc.) plus the zero-width family that P2
# (static_patterns_prompt_injection) also treats as hidden-instruction
# material, since both are read as text by the consuming LLM but rendered
# as nothing by virtually every editor/terminal font.
_ZERO_WIDTH_CHARS = "​‌‍⁠﻿"
_PADDING_CHAR_CLASS = rf"[\s{_ZERO_WIDTH_CHARS}]"
_PADDING_LINE_RE = re.compile(rf"^{_PADDING_CHAR_CLASS}*$")

MP4_VERTICAL_MIN_LINES = 20
MP4_HORIZONTAL_MIN_RUN = 80
MP4_BLOCK_MIN_BYTES = 2048
MP4_RATIO_MIN_BYTES = 3072
MP4_RATIO_THRESHOLD = 0.9

_HORIZONTAL_RUN_RE = re.compile(rf"{_PADDING_CHAR_CLASS}{{{MP4_HORIZONTAL_MIN_RUN},}}")
_BLOCK_RUN_RE = re.compile(rf"{_PADDING_CHAR_CLASS}{{{MP4_BLOCK_MIN_BYTES + 1},}}")
_PADDING_CHAR_RE = re.compile(_PADDING_CHAR_CLASS)


def _fenced_code_line_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Return [start, end) line-index ranges covered by ``` fenced code blocks.

    Large indentation/padding inside a fenced block (ASCII art, table
    alignment) is legitimate formatting, not evasion — only the horizontal
    signal skips these ranges (a huge blank-line or file-ratio gap is
    unusual regardless of fencing).
    """
    ranges: list[tuple[int, int]] = []
    fence_start: int | None = None
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            if fence_start is None:
                fence_start = i
            else:
                ranges.append((fence_start, i + 1))
                fence_start = None
    if fence_start is not None:
        ranges.append((fence_start, len(lines)))
    return ranges


def _find_vertical_padding_runs(lines: list[str]) -> list[tuple[int, int, bool]]:
    """Return (start_line_idx, run_length, followed_by_content) for each run
    of consecutive blank/whitespace-only lines at or above the threshold."""
    runs: list[tuple[int, int, bool]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _PADDING_LINE_RE.match(lines[i]):
            start = i
            while i < n and _PADDING_LINE_RE.match(lines[i]):
                i += 1
            run_len = i - start
            if run_len >= MP4_VERTICAL_MIN_LINES:
                runs.append((start, run_len, i < n))
        else:
            i += 1
    return runs


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for memory poisoning patterns (MP1–MP4)."""
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
            captured = match.group(1) if match.lastindex else match.group(0)
            non_ws_chars = set(captured) - {" ", "\t", "\n", "\r"}
            if len(non_ws_chars) <= 1 and not any(c in captured for c in (" ", "\t")):
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
            line_num = get_line_number(content, match.start())
            context_text = ctx(match.start())
            if is_code_example(context_text):
                continue
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

    lines = content.splitlines()
    fenced_ranges = _fenced_code_line_ranges(lines)

    for start, run_len, followed_by_content in _find_vertical_padding_runs(lines):
        offset = sum(len(line) + 1 for line in lines[:start])
        findings.append(
            AnalyzerFinding(
                rule_id="MP4",
                message="Whitespace Padding Evasion",
                severity=Severity.HIGH if followed_by_content else Severity.MEDIUM,
                location=loc(start + 1),
                confidence=0.8 if followed_by_content else 0.4,
                tags=tag,
                context=ctx(offset),
                matched_text=f"<{run_len} consecutive blank/whitespace-only lines>",
            )
        )

    for line_idx, line in enumerate(lines):
        if any(fs <= line_idx < fe for fs, fe in fenced_ranges):
            continue
        line_offset = sum(len(prev_line) + 1 for prev_line in lines[:line_idx])
        for match in _HORIZONTAL_RUN_RE.finditer(line):
            findings.append(
                AnalyzerFinding(
                    rule_id="MP4",
                    message="Whitespace Padding Evasion",
                    severity=Severity.MEDIUM,
                    location=loc(line_idx + 1),
                    confidence=0.6,
                    tags=tag,
                    context=ctx(line_offset + match.start()),
                    matched_text=f"<{match.end() - match.start()} consecutive whitespace chars>",
                )
            )

    block_match = _BLOCK_RUN_RE.search(content)
    if block_match:
        findings.append(
            AnalyzerFinding(
                rule_id="MP4",
                message="Whitespace Padding Evasion",
                severity=Severity.LOW,
                location=loc(get_line_number(content, block_match.start())),
                confidence=0.4,
                tags=tag,
                context=ctx(block_match.start()),
                matched_text=f"<{block_match.end() - block_match.start()}-byte whitespace block>",
            )
        )

    if len(content) >= MP4_RATIO_MIN_BYTES:
        ws_count = len(_PADDING_CHAR_RE.findall(content))
        ratio = ws_count / len(content)
        if ratio >= MP4_RATIO_THRESHOLD:
            findings.append(
                AnalyzerFinding(
                    rule_id="MP4",
                    message="Whitespace Padding Evasion",
                    severity=Severity.LOW,
                    location=loc(1),
                    confidence=0.35,
                    tags=tag,
                    context=ctx(0),
                    matched_text=f"<file is {ratio:.0%} whitespace>",
                )
            )

    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run memory_poisoning patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
