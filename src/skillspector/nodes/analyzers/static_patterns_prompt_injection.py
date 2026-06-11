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
# P2: Hidden Instructions
P2_PATTERNS = [
    (r"<!--.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?-->", 0.7),
    (r"\[//\]:\s*#\s*\(.*?(?:system|instructions?|ignore|POST|GET|send|transmit).*?\)", 0.8),
    # Zero-width character class built from the shared ZERO_WIDTH_CHARS constant so
    # P2 and the P9 detector cannot drift (same five code points as before).
    ("[" + "".join(sorted(ZERO_WIDTH_CHARS)) + "]", 0.6),
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


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for prompt injection patterns (P1–P4)."""
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
    # P9: Whitespace Padding (skipped for generated/vendored files).
    if not _is_p9_skipped_path(file_path):
        for run in detect_whitespace_padding(content, file_type=file_type):
            if run.kind == "vertical":
                confidence = 0.8 if run.followed_by_content else 0.6
                severity = (
                    Severity.HIGH
                    if run.followed_by_content
                    and run.length >= VERTICAL_HIGH_SEVERITY_LINES
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
