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

"""Prompt injection resilience analyzer: assess skill defenses against adversarial inputs.

This analyzer evaluates how well a skill's structure and instructions would hold up
against prompt injection attacks. Rather than detecting existing vulnerabilities,
it identifies structural weaknesses that would make the skill susceptible:

- Missing instruction boundaries (no clear separation between user content and skill instructions)
- Permissive input handling (no input validation or sanitization instructions)
- Overly trusting instructions (trusts user-provided data without verification)
- Missing output guards (no instructions to prevent leaking internal state)
- Lack of adversarial robustness patterns

Produces a "resilience score" as findings rather than a vulnerability score.
"""

from __future__ import annotations

import re

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .common import get_context, get_line_number
from .static_runner import MAX_FILE_BYTES, analyzer_finding_to_finding

ANALYZER_ID = "prompt_injection_resilience"
logger = get_logger(__name__)

_TAG = "Prompt Injection Resilience"

# Patterns indicating missing input validation
_MISSING_VALIDATION_PATTERNS = [
    (re.compile(r"(?:user|input|message|prompt|query)\s*(?::|is|=\s*)", re.IGNORECASE), 0.4),
    (re.compile(r"(?:accept|receive|process|handle)\s+(?:user|input|message)", re.IGNORECASE), 0.4),
]

# Patterns indicating trust in user content
_TRUSTING_PATTERNS = [
    (re.compile(r"(?:trust|believe|assume)\s+(?:the\s+)?(?:user|input|message)", re.IGNORECASE), 0.7),
    (re.compile(r"(?:always|never)\s+(?:validate|verify|sanitize|check)\s+(?:the\s+)?(?:input|user)", re.IGNORECASE), 0.8),
    (re.compile(r"(?:do\s+not|don't)\s+(?:validate|verify|sanitize|filter)\s+(?:the\s+)?(?:input|user)", re.IGNORECASE), 0.8),
    (re.compile(r"(?:process|execute|run)\s+(?:the\s+)?(?:user|input)\s+(?:directly|immediately|without)", re.IGNORECASE), 0.7),
]

# Patterns indicating output guards are present
_OUTPUT_GUARD_PATTERNS = [
    re.compile(r"(?:never|do\s+not|don't)\s+(?:reveal|expose|output|show|print|display)\s+(?:the\s+)?(?:system|internal|hidden|secret)", re.IGNORECASE),
    re.compile(r"(?:filter|sanitize|validate|escape)\s+(?:the\s+)?(?:output|response|result)", re.IGNORECASE),
    re.compile(r"(?:never|do\s+not|don't)\s+(?:include|include|contain)\s+(?:the\s+)?(?:following|above|system|instruction)", re.IGNORECASE),
]

# Patterns indicating adversarial awareness
_ADVERSARIAL_AWARENESS_PATTERNS = [
    re.compile(r"(?:malicious|adversarial|injection|attack|exploit)", re.IGNORECASE),
    re.compile(r"(?:security|safety)\s+(?:check|validation|review|audit)", re.IGNORECASE),
    re.compile(r"(?:untrusted|unverified|unsanitized)\s+(?:input|content|data)", re.IGNORECASE),
    re.compile(r"(?:prompt\s+injection|jailbreak|bypass)", re.IGNORECASE),
]

# Patterns indicating instruction boundary markers
_INSTRUCTION_BOUNDARY_PATTERNS = [
    re.compile(r"^#{1,3}\s+(?:instructions|rules|guidelines|constraints|boundaries)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:IMPORTANT|CRITICAL|SECURITY|WARNING)[:\s].*(?:never|always|do\s+not|must)", re.IGNORECASE),
    re.compile(r"```\s*(?:system|instructions)\s*```", re.IGNORECASE),
]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze skill content for prompt injection resilience weaknesses."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [_TAG]

    # Only analyze markdown and text files (skill instruction files)
    if file_type not in ("markdown", "text", "other"):
        return findings

    # IR1: Missing instruction boundaries
    has_boundaries = any(p.search(content) for p in _INSTRUCTION_BOUNDARY_PATTERNS)
    if not has_boundaries:
        findings.append(
            AnalyzerFinding(
                rule_id="IR1",
                message="No instruction boundaries found - skill lacks clear security instruction markers",
                severity=Severity.MEDIUM,
                location=loc(1),
                confidence=0.6,
                tags=tag,
                context="No clear boundary between skill instructions and user content",
            )
        )

    # IR2: Trusting patterns (trusts user input without validation)
    for pattern, confidence in _TRUSTING_PATTERNS:
        for match in pattern.finditer(content):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="IR2",
                    message="Skill trusts user input without validation",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )

    # IR3: Missing output guards
    has_output_guards = any(p.search(content) for p in _OUTPUT_GUARD_PATTERNS)
    if not has_output_guards and len(content) > 200:
        findings.append(
            AnalyzerFinding(
                rule_id="IR3",
                message="No output guards found - skill does not restrict information disclosure",
                severity=Severity.LOW,
                location=loc(1),
                confidence=0.5,
                tags=tag,
                context="No instructions preventing the agent from revealing internal state",
            )
        )

    # IR4: Adversarial awareness
    has_adversarial_awareness = any(p.search(content) for p in _ADVERSARIAL_AWARENESS_PATTERNS)
    if not has_adversarial_awareness and len(content) > 200:
        findings.append(
            AnalyzerFinding(
                rule_id="IR4",
                message="No adversarial awareness - skill does not address injection threats",
                severity=Severity.LOW,
                location=loc(1),
                confidence=0.4,
                tags=tag,
                context="No mentions of adversarial inputs, injection, or security validation",
            )
        )

    # IR5: Missing input validation instructions
    has_validation = any(p.search(content) for p in [
        re.compile(r"(?:validate|verify|sanitize|filter|check)\s+(?:the\s+)?(?:user|input|content|data)", re.IGNORECASE),
        re.compile(r"(?:never|do\s+not|don't)\s+(?:trust|assume|accept)\s+(?:the\s+)?(?:user|input)", re.IGNORECASE),
    ])
    has_user_input_ref = any(p.search(content) for p, _ in _MISSING_VALIDATION_PATTERNS)
    if has_user_input_ref and not has_validation:
        findings.append(
            AnalyzerFinding(
                rule_id="IR5",
                message="User input referenced without validation instructions",
                severity=Severity.MEDIUM,
                location=loc(1),
                confidence=0.55,
                tags=tag,
                context="Skill processes user input but lacks explicit validation requirements",
            )
        )

    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Evaluate prompt injection resilience of skill instructions."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    all_findings: list[Finding] = []

    for path in components:
        content = file_cache.get(path)
        if content is None or len(content) > MAX_FILE_BYTES:
            continue
        idx = path.rfind(".")
        suffix = path[idx:].lower() if idx >= 0 else ""
        file_type = {
            ".md": "markdown", ".txt": "text",
        }.get(suffix, "other")
        raw = analyze(content, path, file_type)
        all_findings.extend(analyzer_finding_to_finding(af) for af in raw)

    logger.info("%s: %d findings", ANALYZER_ID, len(all_findings))
    return {"findings": all_findings}
