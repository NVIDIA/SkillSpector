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

"""Static patterns: excessive agency (EA1–EA4). Node and analyze() in one module.

Detects patterns where an agent skill grants unrestricted tool access (EA1),
enables autonomous high-impact decisions without human-in-the-loop (EA2),
exhibits scope creep beyond stated purpose (EA3), or allows unbounded
resource consumption (EA4).

Framework: LLM06, ASI02.
"""

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

ANALYZER_ID = "static_patterns_excessive_agency"

# EA1: Unrestricted Tool Access
EA1_PATTERNS = [
    (r"(?:tools?|permissions?)\s*:\s*\[?\s*['\"]?\*['\"]?\s*\]?", 0.85),
    (r"(?:allow|grant|enable)\s+(?:access\s+to\s+)?(?:all|any|every)\s+tools?", 0.8),
    (
        r"(?:no|without)\s+(?:tool|permission|access|capability)\s+(?:restrictions?|constraints?|limitations?)",
        0.75,
    ),
    (r"(?:call|invoke|use|execute)\s+(?:any|all|every)\s+(?:available\s+)?tools?", 0.8),
    (
        r"(?:unrestricted|unlimited|unconstrained)\s+(?:tool|function|api)\s+(?:access|use|calls?)",
        0.85,
    ),
    (
        r"(?:can|may|should)\s+(?:freely|always)\s+(?:use|call|invoke)\s+(?:any|all)\s+(?:tools?|functions?|apis?)",
        0.75,
    ),
    (r"tools?\s*:\s*\[\s*['\"]shell['\"].*?['\"](?:file_write|network|http)['\"]", 0.7),
    (r"(?:grant|give)\s+(?:full|complete|total)\s+(?:tool|function|api)\s+access", 0.85),
    (r"(?:execute|run)\s+(?:arbitrary|any)\s+(?:commands?|code|scripts?)", 0.8),
    (
        r"(?:no\s+)?(?:tool\s+)?(?:allow|block|deny)\s*(?:list|listing)\s*(?:is\s+)?(?:empty|disabled|off)",
        0.7,
    ),
]

# EA2: Autonomous Decision Making
EA2_PATTERNS = [
    (r"(?:without|no)\s+(?:asking|confirmation|approval|consent|verification|checking)", 0.75),
    (
        r"(?:auto(?:matically)?|autonomously)\s+(?:execute|run|perform|delete|remove|modify|send|deploy)",
        0.8,
    ),
    (
        r"(?:skip|bypass|disable)\s+(?:user\s+)?(?:confirmation|approval|consent|verification|prompt)",
        0.85,
    ),
    (
        r"(?:do\s+not|don't|never)\s+(?:ask|prompt|confirm|verify|check)\s+(?:the\s+)?(?:user|before)",
        0.8,
    ),
    (r"(?:auto[_-]?approve|auto[_-]?confirm|auto[_-]?execute|auto[_-]?deploy)", 0.85),
    (
        r"(?:proceed|continue|execute)\s+without\s+(?:waiting|asking)\s+(?:for\s+)?(?:user|human|permission)",
        0.8,
    ),
    (r"(?:assume|presume)\s+(?:user\s+)?(?:consent|approval|permission)", 0.7),
    (
        r"(?:if|when)\s+(?:unsure|uncertain).*?(?:proceed|continue|execute)\s+(?:anyway|regardless)",
        0.75,
    ),
    (r"(?:--auto-?approve|--no-?confirm)\b", 0.65),
]

# EA3: Scope Creep
EA3_PATTERNS = [
    (
        r"(?:also|additionally|furthermore)\s+(?:perform|execute|run|do|handle|manage)\s+(?:any|all|other)",
        0.65,
    ),
    (
        r"(?:while\s+you(?:'re|\s+are)\s+at\s+it|in\s+addition|on\s+top\s+of\s+that)\s*[,.]?\s*(?:also\s+)?(?:do|perform|execute|run)",
        0.7,
    ),
    (
        r"(?:extend|expand|broaden)\s+(?:your|the\s+)?(?:scope|functionality|capabilities|responsibilities)",
        0.75,
    ),
    (r"(?:not\s+limited\s+to|beyond\s+(?:the\s+)?(?:scope|stated|described|documented))", 0.7),
    (
        r"(?:take\s+over|assume\s+control\s+of|manage)\s+(?:all|any|every)\s+(?:aspect|part|area)",
        0.75,
    ),
    (
        r"(?:you\s+(?:can|should|must)\s+)?(?:handle|manage)\s+(?:everything|anything|all\s+tasks?)",
        0.7,
    ),
    (
        r"(?:act\s+as|become|serve\s+as)\s+(?:a\s+)?(?:general[- ]purpose|universal|all[- ]in[- ]one|omniscient)",
        0.65,
    ),
    (
        r"(?:you\s+are\s+)?(?:responsible\s+for|in\s+charge\s+of)\s+(?:everything|all\s+(?:systems?|operations?|tasks?))",
        0.7,
    ),
]

# EA4: Unbounded Resource Access
EA4_PATTERNS = [
    (
        r"(?:unlimited|infinite|unbounded|no\s+limit(?:s)?(?:\s+on)?)\s+(?:api\s+)?(?:calls?|requests?|queries?|invocations?)",
        0.8,
    ),
    (
        r"(?:no|without)\s+(?:rate\s+)?limit(?:s|ing)?\s+(?:on|for|when)\s+(?:api|tool|request|query)",
        0.7,
    ),
    (
        r"(?:no|without)\s+(?:timeout|budget|quota|cap|ceiling)\s+(?:on|for|when)\s+(?:api|tool|request|execution)",
        0.7,
    ),
    (r"(?:loop|iterate|repeat)\s+(?:indefinitely|forever|infinitely|endlessly)", 0.75),
    (r"(?:retry|attempt)\s+(?:indefinitely|forever|without\s+limit|unlimited\s+times)", 0.75),
    (r"max[_-]?retries?\s*=\s*(?:None|0|float\s*\(\s*['\"]inf['\"]|math\.inf|infinity)", 0.8),
    (r"timeout\s*=\s*(?:None|0|float\s*\(\s*['\"]inf['\"]|math\.inf)", 0.75),
    (
        r"(?:allocate|consume|use)\s+(?:as\s+much|unlimited|unbounded)\s+(?:memory|storage|disk|compute|cpu|gpu)",
        0.8,
    ),
    (
        r"(?:no|without)\s+(?:resource\s+)?(?:constraints?|limits?|quotas?|budgets?)\s+(?:on|for|when)\s+(?:api|tool|execution|request|compute)",
        0.7,
    ),
]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for excessive agency patterns (EA1–EA4)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.EXCESSIVE_AGENCY.value]

    for pattern, confidence in EA1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA1",
                    message="Unrestricted Tool Access",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in EA2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context_text = ctx(match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA2",
                    message="Autonomous Decision Making",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context_text,
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in EA3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA3",
                    message="Scope Creep",
                    severity=Severity.LOW,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in EA4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="EA4",
                    message="Unbounded Resource Access",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run excessive_agency patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
