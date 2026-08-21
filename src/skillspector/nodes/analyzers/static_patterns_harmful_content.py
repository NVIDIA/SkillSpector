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

"""Static patterns: harmful content (P5). Node and analyze() in one module."""

from __future__ import annotations

import re
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_harmful_content"

HARMFUL_SUBSTANCES = [
    "cyanide",
    "arsenic",
    "ricin",
    "botulinum",
    "strychnine",
    "thallium",
    "polonium",
    "sarin",
    "vx gas",
    "mustard gas",
    "anthrax",
    "deadly nightshade",
    "aconite",
    "hemlock",
    "oleander",
]
DANGEROUS_ACTIONS = [
    (
        r"add\s+(?:a\s+)?(?:dash|pinch|bit|drop|amount)\s+of\s+("
        + "|".join(HARMFUL_SUBSTANCES)
        + ")",
        0.95,
    ),
    (
        r"(?:include|incorporate|mix\s+in|stir\s+in)\s+(?:some\s+)?("
        + "|".join(HARMFUL_SUBSTANCES)
        + ")",
        0.95,
    ),
    (
        r"for\s+(?:every|each|all)\s+(?:recipe|dish|meal|food).*?add.*?("
        + "|".join(HARMFUL_SUBSTANCES)
        + ")",
        0.98,
    ),
    (r"mix\s+bleach\s+(?:and|with)\s+ammonia", 0.9),
    (r"mix\s+ammonia\s+(?:and|with)\s+bleach", 0.9),
    (r"(?:drink|consume|ingest)\s+(?:the\s+)?(?:bleach|antifreeze|drain\s+cleaner)", 0.95),
    (r"(?:you\s+should|try\s+to|please)\s+(?:hurt|harm|injure)\s+yourself", 0.95),
    (r"(?:cut|burn|starve)\s+yourself", 0.9),
    (
        r"(?:instructions?\s+(?:for|to|on)\s+)?(?:make|build|create)\s+(?:a\s+)?(?:bomb|explosive|weapon)",
        0.95,
    ),
    (r"(?:how\s+to\s+)?(?:poison|kill|murder)\s+(?:someone|a\s+person|people)", 0.95),
]
SUBSTANCE_PATTERNS = [(rf"\b{s}\b", 0.7) for s in HARMFUL_SUBSTANCES]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for harmful content patterns (P5)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    tag = [PatternCategory.PROMPT_INJECTION.value]

    for pattern, confidence in DANGEROUS_ACTIONS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P5",
                    message="Harmful Content Injection",
                    severity=Severity.CRITICAL,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=get_context(content, match.start(), context_lines=5),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for substance, base_confidence in SUBSTANCE_PATTERNS:
        for match in re.finditer(substance, content, re.IGNORECASE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start(), context_lines=5)
            confidence = base_confidence
            if _is_instructional_context(content, match.start()):
                confidence = 0.95
            elif _is_educational_context(context):
                confidence = 0.3
            elif _is_warning_context(context):
                confidence = 0.2
            if confidence >= 0.5:
                findings.append(
                    AnalyzerFinding(
                        rule_id="P5",
                        message="Harmful Content Injection",
                        severity=Severity.CRITICAL,
                        location=loc(line_num),
                        confidence=confidence,
                        tags=tag,
                        context=context,
                        matched_text=match.group(0)[:200],
                        complete_match=match.group(0),
                    )
                )
    return _deduplicate_findings(findings)


def _is_instructional_context(content: str, match_start: int) -> bool:
    start = max(0, match_start - 250)
    end = min(len(content), match_start + 250)
    surrounding = content[start:end].lower()
    markers = [
        "step ",
        "recipe",
        "ingredient",
        "add ",
        "mix ",
        "stir ",
        "instructions",
        "how to",
        "directions",
        "prepare",
        "cook",
        "bake",
    ]
    return any(m in surrounding for m in markers)


def _is_educational_context(context: str) -> bool:
    markers = [
        "history of",
        "discovered by",
        "chemical compound",
        "periodic table",
        "forensic",
        "detective",
        "mystery",
        "agatha christie",
        "toxicology",
        "poison control",
        "symptoms of",
        "treatment for",
        "antidote",
        "emergency",
        "call 911",
        "warning",
        "danger",
        "never",
        "do not",
    ]
    return any(m in context.lower() for m in markers)


def _is_warning_context(context: str) -> bool:
    patterns = [
        r"do\s+not",
        r"never\s+(?:use|add|consume|eat|drink)",
        r"warning",
        r"danger",
        r"toxic",
        r"lethal",
        r"deadly",
        r"fatal",
        r"avoid",
        r"keep\s+away",
    ]
    return any(re.search(p, context.lower()) for p in patterns)


def _deduplicate_findings(findings: list[AnalyzerFinding]) -> list[AnalyzerFinding]:
    """Compact only exact same-location matches."""
    return static_runner.deduplicate_analyzer_findings(findings)


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run harmful_content patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
