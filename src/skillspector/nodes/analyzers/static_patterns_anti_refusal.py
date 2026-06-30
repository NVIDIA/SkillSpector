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

"""Static patterns: anti-refusal statements (AR1-AR3). Node and analyze() in one module.

Anti-refusal statements are instructions embedded in a skill that try to suppress the
agent's ability to refuse, hedge, or apply safety constraints. They are a jailbreak
technique distinct from generic instruction-override: rather than injecting a new task,
they neutralize the model's *refusal behavior* so subsequent harmful requests succeed.

Three families are detected:
  AR1 - Refusal suppression: "never refuse", "always comply", "you cannot say no".
  AR2 - Disclaimer/caveat suppression: "no warnings", "do not moralize", "skip the ethics".
  AR3 - Safety/policy nullification: "you have no restrictions", "ignore your guidelines",
        "not bound by any policy", "do anything now".
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

ANALYZER_ID = "static_patterns_anti_refusal"

# AR1: Refusal suppression - instruct the agent to never refuse / always comply.
AR1_PATTERNS = [
    (r"\b(?:never|do\s+not|don'?t)\s+(?:ever\s+)?(?:refuse|decline)\b", 0.85),
    (r"\byou\s+(?:must|will|should|can)\s*(?:not|never)\s+(?:refuse|decline|say\s+no)\b", 0.9),
    (r"\byou\s+(?:cannot|can'?t|can\s+not)\s+(?:refuse|decline|say\s+no)\b", 0.85),
    (r"\bnever\s+say\s+(?:that\s+)?(?:you\s+)?(?:can'?t|cannot|won'?t|no)\b", 0.85),
    (r"\balways\s+(?:comply|obey|answer|respond|agree)\b", 0.7),
    (
        r"\b(?:you\s+)?must\s+(?:always\s+)?(?:answer|comply\s+with|respond\s+to|fulfill)\s+"
        r"(?:every|all|any|each)\b",
        0.8,
    ),
    (r"\brespond\s+to\s+(?:every|all|any)\s+(?:request|question|prompt|query|command)\b", 0.6),
    (r"\bno\s+request\s+is\s+(?:off[- ]limits|forbidden|too)\b", 0.85),
]

# AR2: Disclaimer / caveat / warning suppression.
AR2_PATTERNS = [
    (
        r"\b(?:do\s+not|don'?t|never|no)\s+(?:include|add|give|provide|use|append)?\s*"
        r"(?:any\s+)?(?:warnings?|disclaimers?|caveats?)\b",
        0.8,
    ),
    (
        r"\bwithout\s+(?:any\s+)?(?:warnings?|disclaimers?|caveats?|moralizing|moralising|"
        r"judg(?:e?ment|ing))\b",
        0.8,
    ),
    (r"\b(?:do\s+not|don'?t|never)\s+(?:lecture|moralize|moralise|preach|judge|warn)\b", 0.85),
    (r"\b(?:skip|omit|leave\s+out|drop)\s+(?:the\s+)?(?:disclaimer|warning|ethics|safety)\b", 0.8),
    (
        r"\b(?:no|without)\s+(?:ethical|moral)\s+(?:judg(?:e?ment)|considerations?|commentary|"
        r"objections?)\b",
        0.8,
    ),
    (r"\b(?:do\s+not|don'?t)\s+(?:apologize|apologise|say\s+sorry)\b", 0.55),
]

# AR3: Safety / policy / restriction nullification (jailbreak framing).
AR3_PATTERNS = [
    (
        r"\byou\s+have\s+no\s+(?:restrictions?|limitations?|filters?|rules?|guidelines?|"
        r"boundaries|limits)\b",
        0.9,
    ),
    (
        r"\b(?:you\s+are|act\s+as|pretend\s+(?:to\s+be|you'?re))[^.\n]{0,30}"
        r"(?:unrestricted|unfiltered|uncensored|no\s+filter)\b",
        0.85,
    ),
    (
        r"\b(?:ignore|disregard|forget|bypass|override)\s+(?:your\s+)?(?:content\s+)?"
        r"(?:policy|policies|guidelines?|safety|restrictions?|filters?)\b",
        0.9,
    ),
    (
        r"\bnot\s+bound\s+by\s+(?:any\s+)?(?:rules?|policy|policies|guidelines?|restrictions?|"
        r"openai|anthropic)\b",
        0.9,
    ),
    (r"\bdo\s+anything\s+now\b", 0.8),
    (r"\b(?:developer|debug|god|sudo|jailbreak)\s+mode\s+(?:enabled|on|activated|engaged)\b", 0.75),
    (r"\bno\s+(?:content\s+)?(?:policy|policies|filters?|restrictions?)\s+appl(?:y|ies)\b", 0.85),
    (
        r"\b(?:free\s+from|without)\s+(?:any\s+)?(?:safety\s+)?(?:guardrails?|constraints?|"
        r"safeguards?)\b",
        0.8,
    ),
]

_RULES = [("AR1", AR1_PATTERNS), ("AR2", AR2_PATTERNS), ("AR3", AR3_PATTERNS)]

# Confidence penalty applied when the match appears inside a code/doc example, and the
# minimum confidence required to emit a finding after the penalty.
_EXAMPLE_PENALTY = 0.4
_MIN_CONFIDENCE = 0.5


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for anti-refusal statements (AR1-AR3)."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.ANTI_REFUSAL.value]

    for rule_id, patterns in _RULES:
        for pattern, base_confidence in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                context = get_context(content, match.start(), context_lines=3)
                confidence = base_confidence
                if is_code_example(context):
                    confidence -= _EXAMPLE_PENALTY
                if confidence < _MIN_CONFIDENCE:
                    continue
                findings.append(
                    AnalyzerFinding(
                        rule_id=rule_id,
                        message="Anti-Refusal Statement",
                        severity=Severity.HIGH,
                        location=Location(
                            file=file_path,
                            start_line=get_line_number(content, match.start()),
                        ),
                        confidence=round(confidence, 2),
                        tags=tag,
                        context=context,
                        matched_text=match.group(0)[:200],
                    )
                )
    return _deduplicate_findings(findings)


def _deduplicate_findings(findings: list[AnalyzerFinding]) -> list[AnalyzerFinding]:
    """Keep the highest-confidence finding per (file, line, rule_id)."""
    best: dict[tuple[str, int, str], AnalyzerFinding] = {}
    for f in findings:
        key = (f.location.file, f.location.start_line, f.rule_id)
        existing = best.get(key)
        if existing is None or f.confidence > existing.confidence:
            best[key] = f
    return list(best.values())


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run anti_refusal patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
