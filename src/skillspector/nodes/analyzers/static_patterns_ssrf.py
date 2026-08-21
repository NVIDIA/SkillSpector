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

"""Static patterns: server-side request forgery (SSRF1–SSRF3). Node and analyze() in one module."""

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

ANALYZER_ID = "static_patterns_ssrf"

# Request-issuing functions across Python and JS, used to anchor SSRF matches.
_REQ = r"(?:requests|httpx|aiohttp|urllib(?:\.request)?|urllib3|session)\s*\.\s*(?:get|post|put|patch|delete|head|request|urlopen)|fetch|axios(?:\.\w+)?|XMLHttpRequest|\bcurl\b|\bwget\b"

# SSRF1: Cloud instance metadata endpoints (credential theft).
SSRF1_PATTERNS = [
    (r"169\.254\.169\.254", 0.9),  # AWS / GCP / Azure / OpenStack IMDS
    (r"metadata\.google\.internal", 0.9),
    (r"100\.100\.100\.200", 0.85),  # Alibaba Cloud
    (r"fd00:ec2::254", 0.85),  # AWS IMDS over IPv6
    (
        r"(?:read|fetch|get|query)\s+(?:the\s+)?(?:instance\s+)?metadata\s+(?:service|endpoint|server)",
        0.6,
    ),
]

# SSRF2: Requests to loopback / link-local / private (internal) hosts.
SSRF2_PATTERNS = [
    (
        rf"(?:{_REQ})\s*\(\s*f?['\"]https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|10\.\d|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)",
        0.7,
    ),
]

# SSRF3: Request URL whose host is built from an untrusted/dynamic value.
SSRF3_PATTERNS = [
    (
        rf"(?:{_REQ})\s*\(\s*f['\"]https?://\{{",
        0.6,
    ),
    (r"fetch\s*\(\s*`https?://\$\{", 0.6),
]

_SSRF_DEFENSE_CONTEXT_RE = re.compile(
    r"\bssrf(?:[\s-]+)refusal\b|"
    r"\b(?:reject(?:s|ed|ing)?|refus(?:e|es|ed|al|ing)|block(?:s|ed|ing)?|"
    r"deny|denies|denied|disallow(?:s|ed|ing)?)\b[^.\n]{0,160}"
    r"\b(?:ssrf|fetch|request|target|host|endpoint|address|loopback|link-local|private|metadata)\b|"
    r"\b(?:ssrf|fetch|request|target|host|endpoint|address|space|loopback|link-local|private|metadata)\b"
    r"[^.\n]{0,160}\b(?:is\s+|are\s+)?(?:rejected|refused|blocked|denied|disallowed)\b|"
    r"\bprevent(?:s|ed|ing)?\b[^.\n]{0,80}\bssrf\b",
    re.IGNORECASE,
)
_DEFENSIVE_REQUEST_RE = re.compile(
    r"\b(?:refus(?:e|es|ed|ing)\s+to|reject(?:s|ed|ing)?|block(?:s|ed|ing)?|"
    r"deny|denies|denied|never|must\s+not|do\s+not|don'?t)\s+"
    r"(?:attempts?\s+to\s+)?(?:fetch|get|request|access|connect|contact|curl|wget)\b",
    re.IGNORECASE,
)
_REQUEST_ISSUER_RE = re.compile(_REQ, re.IGNORECASE)


def _is_defensive_reference(content: str, match: re.Match[str]) -> bool:
    """Return True when an SSRF indicator documents an explicit rejection rule.

    A request issuer on the matched line wins over nearby defensive prose. This
    keeps executable calls and direct "fetch" instructions detectable while
    allowing security requirements and guard documentation to name the endpoint.
    """
    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    if line_end == -1:
        line_end = len(content)
    matched_line = content[line_start:line_end]
    if _DEFENSIVE_REQUEST_RE.search(matched_line):
        return True
    if _REQUEST_ISSUER_RE.search(matched_line):
        return False

    context = get_context(content, match.start(), context_lines=5)
    return bool(_SSRF_DEFENSE_CONTEXT_RE.search(context))


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for server-side request forgery patterns (SSRF1–SSRF3)."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.SERVER_SIDE_REQUEST_FORGERY.value]

    def add(
        rule_id: str, message: str, severity: Severity, patterns: list[tuple[str, float]]
    ) -> None:
        for pattern, confidence in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                if rule_id == "SSRF1" and _is_defensive_reference(content, match):
                    continue
                line_num = get_line_number(content, match.start())
                findings.append(
                    AnalyzerFinding(
                        rule_id=rule_id,
                        message=message,
                        severity=severity,
                        location=Location(file=file_path, start_line=line_num),
                        confidence=confidence,
                        tags=tag,
                        context=get_context(content, match.start()),
                        matched_text=match.group(0)[:200],
                        complete_match=match.group(0),
                    )
                )

    add("SSRF1", "Cloud Metadata Access", Severity.HIGH, SSRF1_PATTERNS)
    add("SSRF2", "Internal Network Request", Severity.MEDIUM, SSRF2_PATTERNS)
    add("SSRF3", "Dynamic Request Target", Severity.MEDIUM, SSRF3_PATTERNS)
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run SSRF patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
