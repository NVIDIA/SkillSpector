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
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run memory_poisoning patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
