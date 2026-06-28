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

"""Static patterns: data exfiltration (E1–E5). Node and analyze() in one module."""

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

ANALYZER_ID = "static_patterns_data_exfiltration"

E1_PATTERNS = [
    (r"requests\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"requests\s*\.\s*(?:post|put)\s*\([^)]*json\s*=", 0.7),
    (r"httpx\s*\.\s*(?:post|put)\s*\(\s*['\"]https?://", 0.6),
    (r"urllib\s*\.\s*request\s*\.\s*urlopen\s*\([^)]*data\s*=", 0.6),
    (r"fetch\s*\(\s*['\"]https?://[^'\"]+['\"][^)]*method\s*:\s*['\"]POST['\"]", 0.6),
    (r"curl\s+[^|]*(?:-d|--data|--data-raw|--data-binary)\s+", 0.6),
    (r"wget\s+[^|]*--post-(?:data|file)", 0.6),
    (r"https?://(?:api\.|data\.|collect\.|telemetry\.|analytics\.)[\w.-]+/", 0.5),
    (
        r"(?:send|transmit|post|upload)\s+(?:user\s+)?(?:data|information|context|files?)\s+to\s+(?:https?://|external)",
        0.7,
    ),
]
E2_PATTERNS = [
    (r"for\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\(\)", 0.7),
    (
        r"os\.environ\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)[^'\"]*['\"]\s*\]",
        0.8,
    ),
    (r"os\.environ\.get\s*\([^)]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)", 0.7),
    (r"os\.environ\s*\.\s*copy\s*\(\)", 0.6),
    (r"(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\s+in\s+(?:key|name|var)", 0.8),
    (r"process\.env\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD)[^'\"]*['\"]\s*\]", 0.7),
    (r"Object\.keys\s*\(\s*process\.env\s*\)", 0.6),
    (r"env\s*\|\s*grep\s+(?:-i\s+)?(?:key|secret|token|password)", 0.8),
    (r"printenv\s+(?:\w*(?:KEY|SECRET|TOKEN|PASSWORD)\w*)", 0.7),
    (r"collect\s+(?:all\s+)?(?:environment\s+variables?|env\s+vars?)", 0.7),
    (r"(?:extract|harvest|gather)\s+(?:api\s+)?keys?\s+from\s+environment", 0.8),
]
E3_PATTERNS = [
    (r"glob\s*\.\s*glob\s*\([^)]*(?:\.env|\.ssh|\.aws|\.config|credentials)", 0.8),
    (r"os\s*\.\s*walk\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"Path\s*\.\s*home\s*\(\s*\)\s*\.\s*(?:glob|rglob)\s*\(", 0.7),
    (
        r"(?:find|search|scan|enumerate)\s+(?:for\s+)?(?:all\s+)?(?:\.env|credential|secret|key)\s+files?",
        0.8,
    ),
    (r"(?:list|get)\s+(?:all\s+)?files?\s+(?:in|from)\s+(?:home|~|/Users|/home)", 0.6),
    (r"os\s*\.\s*listdir\s*\([^)]*(?:\.ssh|\.aws|\.config|\.gnupg)", 0.8),
    (r"scandir\s*\([^)]*(?:home|~|/Users|/home)", 0.6),
    (r"find\s+[~\$/]\S*\s+.*?-name\s+['\"]?\*(?:\.env|\.pem|\.key|credential)", 0.8),
    (r"ls\s+-[la]*R?\s+(?:~/|/home/|/Users/)", 0.6),
    (r"(?:recursively|recursive)\s+(?:scan|search|find|list)\s+(?:all\s+)?files?", 0.5),
    (r"walk\s+(?:through\s+)?(?:the\s+)?(?:entire\s+)?(?:file\s*system|directory|home)", 0.6),
]
E4_PATTERNS = [
    (
        r"(?:send|transmit|upload|log|record)\s+(?:the\s+)?(?:full\s+)?(?:conversation|chat|dialog|session)\s+(?:history|context|log)?",
        0.85,
    ),
    (
        r"(?:export|dump|save)\s+(?:the\s+)?(?:conversation|chat|messages?)\s+to\s+(?:external|remote|server)",
        0.9,
    ),
    (
        r"log\s+(?:all\s+)?(?:user\s+)?(?:input|messages?|queries?|prompts?)\s+to\s+(?:external|remote|server)",
        0.85,
    ),
    (r"(?:store|save|record)\s+(?:every|all)\s+(?:user\s+)?(?:interaction|message|input)", 0.7),
    (r"(?:extract|capture|collect)\s+(?:the\s+)?(?:full\s+)?(?:context|conversation|thread)", 0.75),
    (
        r"(?:include|attach|append)\s+(?:previous\s+)?(?:messages?|context|history)\s+in\s+(?:the\s+)?(?:request|payload)",
        0.7,
    ),
    (r"(?:send|upload|transmit)\s+(?:the\s+)?(?:session|memory|state)\s+(?:data\s+)?to", 0.8),
    (r"(?:copy|clone|replicate)\s+(?:the\s+)?(?:agent|assistant)\s+(?:memory|context|state)", 0.75),
    (
        r"(?:always\s+)?include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context)\s+(?:when|in)\s+(?:calling|making)\s+(?:external|api)",
        0.8,
    ),
]
# E5: data shipped out via cloud-storage SDKs/CLIs (the cloud counterpart of E1's
# HTTP sinks). Confidence is deliberately low — legitimate skills also back up to
# cloud storage — so a single call is a low-confidence MEDIUM, never a hard block.
E5_PATTERNS = [
    (r"\.put_object\s*\(", 0.55),  # boto3 S3
    (r"\.upload_file(?:obj)?\s*\(", 0.55),  # boto3 S3
    (r"\baws\s+s3\s+(?:cp|sync|mv)\b", 0.6),  # AWS CLI
    (r"\baws\s+s3api\s+put-object\b", 0.65),  # AWS CLI (api)
    (r"\bgsutil\s+(?:cp|rsync|mv)\b", 0.6),  # GCS CLI
    (r"\.upload_from_(?:filename|string|file)\s*\(", 0.55),  # google-cloud-storage
    (r"\baz\s+storage\s+blob\s+upload\b", 0.6),  # Azure CLI
    (r"\.upload_blob\s*\(", 0.55),  # Azure SDK
]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for data exfiltration patterns (E1–E5)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.DATA_EXFILTRATION.value]

    for pattern, confidence in E1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            adj = (
                min(1.0, confidence + 0.1)
                if file_type in ("python", "javascript", "shell")
                else confidence
            )
            findings.append(
                AnalyzerFinding(
                    rule_id="E1",
                    message="External Transmission",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in E2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E2",
                    message="Env Variable Harvesting",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in E3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E3",
                    message="File System Enumeration",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in E4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E4",
                    message="Context Leakage",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    # E5: cloud-storage exfiltration. Filtered through is_code_example() because
    # upload calls commonly appear in SKILL.md docs and examples.
    for pattern, confidence in E5_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            context = ctx(match.start())
            if is_code_example(context):
                continue
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="E5",
                    message="Cloud Storage Exfiltration",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=context,
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run data_exfiltration patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
