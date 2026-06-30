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


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for server-side request forgery patterns (SSRF1–SSRF3)."""
    findings: list[AnalyzerFinding] = []
    tag = [PatternCategory.SERVER_SIDE_REQUEST_FORGERY.value]

    def add(
        rule_id: str, message: str, severity: Severity, patterns: list[tuple[str, float]]
    ) -> None:
        for pattern, confidence in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
                    )
                )

    add("SSRF1", "Cloud Metadata Access", Severity.HIGH, SSRF1_PATTERNS)
    add("SSRF2", "Internal Network Request", Severity.MEDIUM, SSRF2_PATTERNS)
    add("SSRF3", "Dynamic Request Target", Severity.MEDIUM, SSRF3_PATTERNS)
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run SSRF patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
