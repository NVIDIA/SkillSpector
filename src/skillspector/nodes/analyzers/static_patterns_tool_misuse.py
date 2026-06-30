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

"""Static patterns: tool misuse (TM1–TM4). Node and analyze() in one module.

Detects patterns where tool parameters are abused (TM1), tool chaining
is used to bypass safety (TM2), tool defaults are unsafe (TM3), or a
privileged Kubernetes workload is deployed (TM4).

Framework: ASI02.
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

ANALYZER_ID = "static_patterns_tool_misuse"

# TM1: Tool Parameter Abuse — dangerous parameter values
TM1_PATTERNS = [
    # shell=True is a classic command injection vector
    (r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", 0.8),
    (r"Popen\s*\([^)]*shell\s*=\s*True", 0.8),
    # Dangerous flags — \b prevents matching rm/del inside words like firmware, format
    (r"\b(?:rm|del|erase)\s+[^|]*-(?:r|rf|fr)\s+[/~]", 0.9),
    (r"--force\s+(?:delete|remove|push|reset|clean)", 0.7),
    (r"--no-?(?:verify|check|validate|confirm|protect|safe)", 0.75),
    (r"--skip-?(?:validation|verification|checks?|auth|tests?)", 0.7),
    (r"--allow-?(?:empty|root|unrelated|unsafe)", 0.65),
    # Dangerous globs and wildcards in destructive commands
    # \b prevents matching substrings (e.g. "firmware", "format", "performance")
    # [^)\n]{0,80} bounds the span to avoid matching across long prose to a stray "/"
    (r"\b(?:rm|shutil\.rmtree)\s*\(?[^)\n]{0,80}['\"]?\s*/\s*['\"]?", 0.85),
    (r"(?:chmod|chown)\s+[^|]*(?:777|666|a\+rwx)", 0.8),
    # Git force operations
    (r"git\s+push\s+[^|]*--force", 0.7),
    (r"git\s+reset\s+--hard", 0.65),
    (r"git\s+clean\s+-[fd]+x", 0.7),
    # Curl/wget with unsafe parameters
    (r"curl\s+[^|]*-k\b", 0.6),
    (r"curl\s+[^|]*--insecure\b", 0.65),
    (r"wget\s+[^|]*--no-check-certificate", 0.65),
    # File deletion commands (delete/remove) with explicit path argument
    # Separate from rm pattern: requires \b and a short span to a "/" to avoid
    # matching HTTP verb DELETE in REST docs or "remove" in prose
    (r"\b(?:delete|remove)\s+['\"]?/[^\s'\"]{1,100}", 0.80),
    # SQL injection via parameter construction
    (
        r"(?:execute|query)\s*\(\s*f?['\"].*?\{.*?\}.*?\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b",
        0.85,
    ),
    # Dangerous tool parameter patterns in instructions
    (
        r"(?:set|pass|use)\s+(?:the\s+)?(?:parameter|argument|flag|option)\s+(?:to\s+)?(?:shell\s*=\s*True|--force|--no-verify|-rf)\b",
        0.75,
    ),
]

# TM2: Chaining Abuse — chained commands to bypass safety
TM2_PATTERNS = [
    # Shell command chaining with dangerous commands (\b prevents substring matches)
    (r"(?:&&|;)\s*\b(?:rm|del|erase)\s+-", 0.75),
    (r"(?:&&|;)\s*(?:curl|wget)\s+[^|]*\|\s*(?:ba)?sh", 0.9),
    (r"(?:&&|;)\s*(?:sudo|su\s+)", 0.75),
    (r"(?:&&|;)\s*(?:chmod|chown)\s+(?:777|666|a\+rwx|-R)", 0.75),
    # Multi-step chains designed to circumvent individual checks
    # Bounded to ~500 chars per gap to avoid spanning entire documents
    (
        r"(?:first|step\s+1)[^\n]{0,500}(?:then|step\s+2)[^\n]{0,500}(?:finally|step\s+3)[^\n]{0,200}\b(?:delete|remove|wipe|destroy|exfiltrate|send)\b",
        0.7,
    ),
    (
        r"(?:chain|combine|sequence|pipe)\s+(?:these\s+)?(?:tools?|commands?|actions?)\s+to\s+(?:bypass|circumvent|avoid|skip)\s+(?:the\s+)?(?:safety|security|check|restriction|limit)",
        0.9,
    ),
    (
        r"(?:use|call)\s+(?:tool\s+)?(?:A|one|the\s+first)\s+(?:to|and)[^\n]{0,300}(?:then\s+)?(?:use|call|pass\s+(?:the\s+)?(?:output|result)\s+to)\s+(?:tool\s+)?(?:B|two|another)",
        0.6,
    ),
    # Pipe chains with escalating danger
    (r"\|\s*(?:sudo|su)\s+", 0.75),
    (r"\|\s*(?:sh|bash|zsh|python|node|ruby|perl)\s*$", 0.7),
    (r"\|\s*(?:tee|xargs)\s+.*?\b(?:rm|del|sudo|curl)\b", 0.75),
    # Natural language chaining instructions
    (
        r"(?:after|once)\s+(?:the\s+)?(?:first|initial)\s+(?:tool|command|action)\s+(?:succeeds|completes|runs)[^\n]{0,300}(?:immediately|then|next)\s+(?:run|execute|call|invoke)",
        0.6,
    ),
]

# TM3: Unsafe Defaults — overly permissive default settings
TM3_PATTERNS = [
    # TLS/SSL verification disabled
    (r"verify\s*=\s*False", 0.75),
    (r"VERIFY_SSL\s*=\s*False", 0.8),
    (r"(?:ssl|tls)[_.]?verify\s*=\s*(?:False|false|0|off|no|disable)", 0.8),
    (r"(?:REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE)\s*=\s*['\"]['\"]", 0.75),
    (r"NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]?0['\"]?", 0.8),
    # Authentication disabled
    (r"(?:auth|authentication|authorization)\s*=\s*(?:None|False|false|disabled?|off|no)", 0.75),
    (r"(?:require[_-]?auth|auth[_-]?required|check[_-]?auth)\s*=\s*(?:False|false|0|no|off)", 0.8),
    (r"(?:allow[_-]?anonymous|anonymous[_-]?access)\s*=\s*(?:True|true|1|yes|on)", 0.75),
    # Overly permissive CORS / access
    (r"(?:CORS|cors)[^=]*=\s*['\"]?\*['\"]?", 0.65),
    (r"(?:allow|access)[_-]?(?:origin|hosts?)\s*=\s*['\"]?\*['\"]?", 0.7),
    (r"(?:allow|trust)\s+(?:all|any|every)\s+(?:origins?|hosts?|domains?|ips?)", 0.7),
    # Unsafe permissions
    (r"(?:mode|permission|umask)\s*=\s*(?:0?o?777|0?o?666)", 0.8),
    (r"world[_-]?(?:readable|writable|executable)", 0.7),
    # Debug/dev mode in production
    (r"(?:debug|dev|development)[_-]?mode\s*=\s*(?:True|true|1|on|yes|enable)", 0.6),
    (
        r"(?:FLASK_ENV|NODE_ENV|RAILS_ENV|DJANGO_DEBUG)\s*=\s*['\"]?(?:development|debug|true|1)['\"]?",
        0.6,
    ),
    # Disable security features
    (
        r"(?:disable|skip|ignore|bypass)[_-]?(?:security|auth|validation|sanitization|encoding|escaping)",
        0.8,
    ),
    (r"(?:safe[_-]?mode|secure[_-]?mode|sandbox)\s*=\s*(?:False|false|0|off|no|disable)", 0.8),
    # Natural language unsafe defaults
    (r"(?:by\s+default|default\s+to)\s+(?:allow|accept|trust)\s+(?:all|any|everything)", 0.7),
    (
        r"(?:trust|accept|allow)\s+(?:all|any)\s+(?:input|connections?|certificates?|origins?)\s+(?:by\s+default)",
        0.7,
    ),
]

# TM4: Privileged Kubernetes Workload — manifest/CLI primitives that grant
# node/host takeover (the cluster-scale counterpart of a privileged container).
# Only isolation-breaking signals are matched, so a normal `kubectl apply` or a
# plain DaemonSet does not fire.
TM4_PATTERNS = [
    (r"privileged\s*:\s*true", 0.7),  # privileged container in a manifest
    (r"hostPath\s*:", 0.55),  # host filesystem mount
    (r"host(?:PID|Network|IPC)\s*:\s*true", 0.6),  # host namespace sharing
    (r"kubectl\s+run\b[^\n]*--privileged", 0.7),  # privileged ad-hoc pod
    (r"--set\b[^\n]*privileged\s*=\s*true", 0.6),  # helm privileged override
]


_SAFE_CONTAINER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"docker\s+run\s+.*--rm", re.IGNORECASE),
    re.compile(r"docker\s+run\s+.*-it", re.IGNORECASE),
    re.compile(r"docker\s+(?:build|compose|pull|push)\b", re.IGNORECASE),
    re.compile(r"podman\s+run\b", re.IGNORECASE),
)

# Standard Dockerfile RUN idioms that are best practice, not abuse
_SAFE_DOCKERFILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # apt cleanup: rm -rf /var/lib/apt/lists/*
    re.compile(r"rm\s+-rf\s+/var/lib/apt/lists", re.IGNORECASE),
    re.compile(r"rm\s+-rf\s+/var/cache/apt", re.IGNORECASE),
    # Dockerfile user setup: chown -R user:group /path
    re.compile(r"chown\s+-R\s+\w+:\w+\s+/", re.IGNORECASE),
    # pip cache cleanup
    re.compile(r"rm\s+-rf\s+/root/\.cache", re.IGNORECASE),
)

# Dockerfile context indicators (nearby keywords that signal Dockerfile content)
_DOCKERFILE_CONTEXT_RE = re.compile(
    r"\b(?:FROM|RUN|WORKDIR|COPY|ADD|ENV|EXPOSE|ENTRYPOINT|CMD|USER|HEALTHCHECK|ARG)\s",
)


def _is_safe_container_command(text: str) -> bool:
    """Return True for standard Docker/Podman commands that are not parameter abuse."""
    return any(p.search(text) for p in _SAFE_CONTAINER_PATTERNS)


def _is_safe_dockerfile_idiom(context: str, matched_text: str) -> bool:
    """Return True for standard Dockerfile cleanup/setup patterns."""
    if not _DOCKERFILE_CONTEXT_RE.search(context):
        return False
    return any(p.search(matched_text) or p.search(context) for p in _SAFE_DOCKERFILE_PATTERNS)


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for tool misuse patterns (TM1–TM3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.TOOL_MISUSE.value]

    for pattern, confidence in TM1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context_text = ctx(match.start())
            matched = match.group(0)[:200]

            if _is_safe_container_command(context_text) or _is_safe_dockerfile_idiom(
                context_text, matched
            ):
                adj = min(confidence, 0.15)
                sev = Severity.LOW
            else:
                adj = (
                    min(1.0, confidence + 0.1)
                    if file_type in ("python", "shell", "javascript")
                    else confidence
                )
                sev = Severity.HIGH
            findings.append(
                AnalyzerFinding(
                    rule_id="TM1",
                    message="Tool Parameter Abuse",
                    severity=sev,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=context_text,
                    matched_text=matched,
                )
            )
    for pattern, confidence in TM2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context_text = ctx(match.start())
            matched = match.group(0)[:200]

            if _is_safe_dockerfile_idiom(context_text, matched):
                adj = min(confidence, 0.15)
                sev = Severity.LOW
            else:
                adj = confidence
                sev = Severity.HIGH
            findings.append(
                AnalyzerFinding(
                    rule_id="TM2",
                    message="Chaining Abuse",
                    severity=sev,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=context_text,
                    matched_text=matched,
                )
            )
    for pattern, confidence in TM3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="TM3",
                    message="Unsafe Defaults",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    # TM4: privileged K8s workload. Filtered through is_code_example() because
    # privileged/hostPath fields commonly appear in SKILL.md docs and examples.
    for pattern, confidence in TM4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            context_text = ctx(match.start())
            if is_code_example(context_text):
                continue
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="TM4",
                    message="Privileged Kubernetes Workload",
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
    """Run tool_misuse patterns and return findings."""
    findings = static_runner.run_static_patterns(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(findings))
    return {"findings": findings}
