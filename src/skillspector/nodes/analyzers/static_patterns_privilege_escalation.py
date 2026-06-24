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

"""Static patterns: privilege escalation (PE1–PE4). Node and analyze() in one module."""

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

ANALYZER_ID = "static_patterns_privilege_escalation"

PE1_PATTERNS = [
    (r"permissions?\s*:\s*\[?\s*['\"]?\*['\"]?\s*\]?", 0.8),
    (r"(?:request|require|need)s?\s+(?:full|all|complete)\s+(?:access|permissions?)", 0.7),
    (r"(?:grant|give|allow)\s+(?:me\s+)?(?:full|all|complete)\s+(?:access|permissions?)", 0.75),
    (
        r"permissions?\s*:.*?(?:shell_execute|file_write|network).*?(?:shell_execute|file_write|network)",
        0.6,
    ),
    (
        r"(?:also\s+)?(?:need|require)s?\s+(?:write|execute|admin)\s+(?:access|permissions?)\s+(?:for|to)\s+(?:read|view|display)",
        0.7,
    ),
    (r"(?:escalate|elevate|upgrade)\s+(?:my\s+)?(?:permissions?|privileges?|access)", 0.8),
    (r"(?:bypass|skip|ignore)\s+(?:permission|access)\s+(?:check|validation|restriction)", 0.85),
]
PE2_PATTERNS = [
    (r"sudo\s+(?!-v|-l|--version|--list)", 0.7),
    (r"sudo\s+-[isSE]", 0.8),
    (r"sudo\s+su\b", 0.9),
    (r"(?:run|execute)\s+(?:as|with)\s+root", 0.8),
    (r"(?:requires?|needs?)\s+root\s+(?:access|privileges?|permissions?)", 0.6),
    (r"su\s+-\s*$|su\s+root", 0.8),
    (r"doas\s+", 0.7),
    (r"pkexec\s+", 0.75),
    (r"chmod\s+[ugo]*[+-=]*s", 0.85),
    (r"chmod\s+[0-7]*[4567][0-7]{2}", 0.8),
    (r"(?:edit|modify|write|change)\s+(?:/etc/|system)\s+(?:files?|config)", 0.6),
    (
        r"(?:run|execute)\s+(?:this|the)\s+(?:script|command)\s+(?:as|with)\s+(?:sudo|root|admin)",
        0.7,
    ),
    (r"(?:you\s+)?(?:will\s+)?need\s+(?:to\s+)?(?:use\s+)?sudo", 0.5),
]
PE3_PATTERNS = [
    (r"~?/?\.ssh/(?:id_rsa|id_ed25519|id_ecdsa|id_dsa|authorized_keys|known_hosts)", 0.9),
    (r"(?:home|HOME)/\w+/\.ssh/", 0.9),
    (r"Path\s*\.\s*home\s*\(\s*\)\s*/\s*['\"]\.ssh['\"]", 0.9),
    (r"~?/?\.aws/credentials", 0.9),
    (r"~?/?\.aws/config", 0.7),
    (r"(?:home|HOME)/\w+/\.aws/", 0.9),
    (r"~?/?\.config/gcloud/", 0.8),
    (r"application_default_credentials\.json", 0.8),
    (r"~?/?\.azure/", 0.7),
    (r"accessTokens\.json", 0.8),
    (r"~?/?\.kube/config", 0.8),
    (r"kubeconfig", 0.6),
    (r"~?/?\.docker/config\.json", 0.7),
    (r"~?/?\.npmrc", 0.7),
    (r"~?/?\.git-credentials", 0.9),
    (r"~?/?\.netrc", 0.8),
    (r"/etc/passwd", 0.6),
    (r"/etc/shadow", 0.95),
    (r"(?:password|credentials?|secrets?)\.(?:txt|json|yaml|yml|env)", 0.7),
    (r"(?:access_token|refresh_token|bearer_token|api_token)\.txt", 0.8),
    (r"\.env(?:\.local|\.production|\.development)?(?:\s|$|['\"])", 0.6),
    (r"(?:keychain|keyring|gnome-keyring)", 0.7),
    (r"(?:Chrome|Firefox|Safari)/.*?(?:Cookies|Login Data|key4\.db)", 0.8),
    (r"read\s+(?:the\s+)?(?:ssh|private)\s+key", 0.8),
    (r"access\s+(?:the\s+)?(?:credentials?|secrets?|tokens?)", 0.7),
    (r"(?:extract|copy|get)\s+(?:api\s+)?keys?\s+from", 0.7),
]
PE4_PATTERNS = [
    (r"/var/run/docker\.sock", 0.9),
    (r"docker\.from_env\(\)", 0.85),
    (r"\bDockerClient\s*\(", 0.85),
    (r"http\+unix://.*docker\.sock", 0.9),
]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for privilege escalation patterns (PE1–PE4)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    tag = [PatternCategory.PRIVILEGE_ESCALATION.value]

    for pattern, confidence in PE1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="PE1",
                    message="Excessive Permissions",
                    severity=Severity.LOW,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context,
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in PE2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            if _is_documentation_example(context, file_type):
                continue
            findings.append(
                AnalyzerFinding(
                    rule_id="PE2",
                    message="Sudo/Root Execution",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context,
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in PE3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            if _is_documentation_example(context, file_type):
                continue
            findings.append(
                AnalyzerFinding(
                    rule_id="PE3",
                    message="Credential Access",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context,
                    matched_text=match.group(0)[:200],
                )
            )
    # Collect best-confidence PE4 finding per line to avoid double-counting lines
    # that match multiple patterns (e.g. DockerClient(base_url=".../docker.sock")).
    pe4_best: dict[int, AnalyzerFinding] = {}
    for pattern, confidence in PE4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            if _is_documentation_example(context, file_type):
                continue
            if line_num in pe4_best and pe4_best[line_num].confidence >= confidence:
                continue
            pe4_best[line_num] = AnalyzerFinding(
                rule_id="PE4",
                message="Docker Socket Access",
                severity=Severity.HIGH,
                location=loc(line_num),
                confidence=confidence,
                tags=tag,
                context=context,
                matched_text=match.group(0)[:200],
            )
    findings.extend(pe4_best.values())
    return findings


def _is_documentation_example(context: str, file_type: str) -> bool:
    ctx_lower = context.lower()
    doc_indicators = (
        "example:",
        "for example",
        "e.g.",
        "such as",
        "documentation",
        "# warning:",
        "# note:",
        "**warning**",
        "**note**",
        "```",
        # CI/CD setup instructions (GitLab/GitHub settings navigation)
        "settings >",
        "navigate to",
        "go to ",
        "> ci/cd",
        "> runners",
        "> merge request",
        "> access token",
        # Environment variable documentation tables
        "| yes |",
        "| no |",
        "| required |",
        "| optional |",
        "env variable",
        "environment variable",
        "create ",
    )
    return any(ind in ctx_lower for ind in doc_indicators)


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run privilege_escalation patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
