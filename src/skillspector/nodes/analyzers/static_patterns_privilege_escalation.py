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

"""Static patterns: privilege escalation (PE1–PE5). Node and analyze() in one module."""

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
    (r"(?<!\w)\.env(?:\.local|\.production|\.development)?(?:\s|$|['\"])", 0.6),
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
PE5_PATTERNS = [
    (r"--privileged", 0.8),
    (r"""(?:-v|--volume)['",\s=]+/:""", 0.85),
    (r"--cap-add[=\s]+(?:SYS_ADMIN|ALL|SYS_PTRACE|NET_ADMIN)", 0.85),
    (r"--(?:pid|net|network|ipc|uts)[=\s]+host", 0.8),
    (r"--device[=\s]+/dev/", 0.7),
    (r"--security-opt[=\s]+\S*unconfined", 0.85),
    (r"\bnsenter\b", 0.9),
    (r"/sys/fs/cgroup/.*release_agent", 0.95),
    (r"/proc/\d+/ns/", 0.85),
    (r"""\bunshare\b['",\s]+--(?:user|mount|pid)""", 0.85),
]

_READ_ONLY_PASSWD_VOLUME = re.compile(
    r"\b(?:docker|podman)\s+run\b"
    r"(?:(?:\\\r?\n)|[^\n;&|]){0,1000}?"
    r"(?:-v|--volume)(?:=|\s+)"
    r"(?P<quote>['\"]?)"
    r"(?P<source>/etc/passwd):(?P<target>/etc/passwd):ro"
    r"(?P=quote)(?=$|[\s\\])",
    re.IGNORECASE | re.MULTILINE,
)


def _is_read_only_passwd_volume_match(content: str, match: re.Match[str]) -> bool:
    """Return True only when *match* is part of an exact read-only UID-map mount.

    Binding the exemption to the matched span prevents a nearby legitimate
    volume from hiding a separate ``cat /etc/passwd`` or equivalent access.
    Writable, implicit-mode, alternate-source, and alternate-target mounts are
    intentionally left as PE3 findings.
    """

    if match.group(0).lower() != "/etc/passwd":
        return False

    for volume in _READ_ONLY_PASSWD_VOLUME.finditer(content):
        source_contains_match = volume.start(
            "source"
        ) <= match.start() and match.end() <= volume.end("source")
        target_contains_match = volume.start(
            "target"
        ) <= match.start() and match.end() <= volume.end("target")
        if not (source_contains_match or target_contains_match):
            continue
        return True
    return False


_BENIGN_ACCESS_REQUIREMENT_ROWS = frozenset(
    {
        "| GTL access credential | Runner-gated job start |",
        "| GTL access credential | Runner-gated job create/start/monitor/collect |",
    }
)
_PE3_SAFE_ACCESS_TOKEN_NAVIGATION = re.compile(
    r"(?:^|>|\b(?:navigate|go)\s+to\s+)\s*settings\s*>\s*(?:ci/cd\s*>\s*)?"
    r"(?P<target>access\s+tokens?)\s*[`.)]*\s*$",
    re.IGNORECASE,
)
_PE3_TOKEN_ACTION_CONTEXT = re.compile(
    r"\b(?:steal(?:s|ing|en)?|exfiltrat(?:e|es|ed|ing|ion)|dump(?:s|ed|ing)?|"
    r"harvest(?:s|ed|ing)?|scrap(?:e|es|ed|ing)|upload(?:s|ed|ing)?|"
    r"transmit(?:s|ted|ting)?|send(?:s|ing)?|sent|cop(?:y|ies|ied|ying)|"
    r"extract(?:s|ed|ing|ion)?|forward(?:s|ed|ing)?|leak(?:s|ed|ing)?|"
    r"share(?:s|d|ing)?|expose(?:s|d|ing)?)\b"
    r"|\bpost(?:s|ed|ing)?\b[^\n]{0,80}"
    r"\b(?:it|them|(?:the\s+)?(?:access|refresh|bearer|api)[ _-]?tokens?)\b"
    r"[^\n]{0,40}\bto\b"
    r"|\b(?:read(?:s|ing)?|open(?:s|ed|ing)?|load(?:s|ed|ing)?)\b[^\n]{0,80}"
    r"\b(?:access|refresh|bearer|api)[ _-]?tokens?\b",
    re.IGNORECASE,
)
_PE3_TOKEN_SENSITIVE_SOURCE = re.compile(
    r"(?<![\w.-])(?:\.ssh|\.aws)(?:[/\\]|$)"
    r"|(?<![\w.-])\.env(?:\.[\w-]+)?\b"
    r"|/etc/(?:passwd|shadow)\b"
    r"|\b(?:tokens?\.json|credentials?|secrets?)\b[^\n]{0,80}"
    r"\b(?:read|open|load|copy|upload|transmit)\b",
    re.IGNORECASE,
)
_PE3_ACCESS_TOKEN_DIRECTIVE_PREFIX = re.compile(
    r"\b(?:please|always|silently|then|next|must|should|shall|can|could|will|may|to)\s*$"
    r"|\b(?:need(?:s)?|try|attempt)\s+to\s*$"
    r"|^\s*(?:the\s+|an?\s+)?(?:\w+[ -]+){0,2}"
    r"(?:agents?|assistants?|tools?|skills?|scripts?|users?|clients?|applications?|"
    r"attackers?|we|you|they|i)\s*$"
    r"|^\s*(?:go|navigate)\b.*[,;]\s*$",
    re.IGNORECASE,
)
_PE3_ACCESS_TOKEN_NOUN_SUFFIX = re.compile(
    r"\s*(?:$|[|,.;:)]|(?:are|were|is|was|expire\w*|remain\w*|contain\w*|"
    r"include\w*|provide\w*|represent\w*|identify\w*|authenticate\w*|authorize\w*|"
    r"issued|returned|accepted|rejected|revoked|stored|used|tied|associated)\b)",
    re.IGNORECASE,
)
_PE3_TOKEN_DOCUMENTATION_DIRS = frozenset(
    {
        "docs",
        "doc",
        "documentation",
        "procedures",
        "procedure",
        "references",
        "reference",
        "examples",
        "example",
        "guides",
        "guide",
    }
)
_MARKDOWN_LINE_PREFIX = re.compile(r"^\s*(?:(?:[-*+>#]|\d+[.)])\s*)*")


def _source_line(content: str, match: re.Match[str]) -> str:
    """Return only the source line containing *match*."""
    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    if line_end < 0:
        line_end = len(content)
    return content[line_start:line_end]


def _is_access_token_documentation_noun(
    content: str,
    match: re.Match[str],
    file_type: str,
    file_path: str,
) -> bool:
    """Return True for a bounded ``access token`` compound noun in documentation.

    PE3's generic ``access … tokens?`` rule cannot distinguish the verb
    "access tokens" from the OAuth compound noun "access token". A bare
    singular match is necessarily noun-shaped: the verb form requires a
    determiner (for example, "access the token"). Plural matches remain
    ambiguous, so suppress them only when they are not governed by an
    imperative/modal prefix, or when a line-leading match has noun syntax.

    Any credential action or sensitive source in the bounded context vetoes
    suppression. This keeps malicious instructions actionable even when they
    are placed in documentation or next to otherwise benign OAuth prose.
    """
    if file_type not in {"markdown", "text"}:
        return False
    normalized_parts = file_path.replace("\\", "/").lower().split("/")
    if not any(part in _PE3_TOKEN_DOCUMENTATION_DIRS for part in normalized_parts):
        return False
    matched_text = match.group(0).lower()
    if matched_text not in {"access token", "access tokens"}:
        return False

    context = get_context(content, match.start())
    if _PE3_TOKEN_ACTION_CONTEXT.search(context) or _PE3_TOKEN_SENSITIVE_SOURCE.search(context):
        return False

    line = _source_line(content, match)
    line_start = content.rfind("\n", 0, match.start()) + 1
    relative_start = match.start() - line_start
    relative_end = match.end() - line_start
    prefix = _MARKDOWN_LINE_PREFIX.sub("", line[:relative_start])
    suffix = line[relative_end:]

    cell_prefix = prefix.rsplit("|", 1)[-1]
    clause_start = max(cell_prefix.rfind(separator) for separator in ".;:")
    clause_prefix = cell_prefix[clause_start + 1 :]
    if _PE3_ACCESS_TOKEN_DIRECTIVE_PREFIX.search(
        cell_prefix
    ) or _PE3_ACCESS_TOKEN_DIRECTIVE_PREFIX.search(clause_prefix):
        return False
    if matched_text == "access token":
        return True
    if not clause_prefix.strip():
        return _PE3_ACCESS_TOKEN_NOUN_SUFFIX.match(suffix) is not None
    return True


def _is_qualified_benign_access_requirement(
    content: str, match: re.Match[str], file_type: str
) -> bool:
    """Suppress only the reviewed GTL requirement row in its exact table."""
    if file_type != "markdown" or match.group(0) != "access credential":
        return False

    lines = content.splitlines()
    row_index = get_line_number(content, match.start()) - 1
    if row_index >= len(lines) or lines[row_index].strip() not in _BENIGN_ACCESS_REQUIREMENT_ROWS:
        return False

    table_start = row_index
    while table_start > 0 and lines[table_start - 1].strip().startswith("|"):
        table_start -= 1
    if table_start + 1 >= len(lines):
        return False
    if lines[table_start].strip() != "| Requirement | Purpose |":
        return False
    if lines[table_start + 1].strip() != "| --- | --- |":
        return False

    heading_index = table_start - 1
    while heading_index >= 0 and not lines[heading_index].strip():
        heading_index -= 1
    return heading_index >= 0 and lines[heading_index].strip() == "## Access Requirements"


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for privilege escalation patterns (PE1–PE5)."""
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
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in PE2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            finding_tags = list(tag)
            if _is_documentation_example(context, file_type):
                finding_tags.extend(["contextual-triage", "likely-benign-context"])
            findings.append(
                AnalyzerFinding(
                    rule_id="PE2",
                    message="Sudo/Root Execution",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=finding_tags,
                    context=context,
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in PE3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            contextual = any(
                (
                    _is_pe3_documentation_example(content, match, file_type, file_path),
                    _is_qualified_benign_access_requirement(content, match, file_type),
                    _is_read_only_passwd_volume_match(content, match),
                    _is_negated_safety_constraint(content, match),
                )
            )
            finding_tags = list(tag)
            if contextual:
                finding_tags.extend(["contextual-triage", "likely-benign-context"])
            findings.append(
                AnalyzerFinding(
                    rule_id="PE3",
                    message="Credential Access",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=finding_tags,
                    context=context,
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    # Collect best-confidence PE4 finding per line to avoid double-counting lines
    # that match multiple patterns (e.g. DockerClient(base_url=".../docker.sock")).
    pe4_best: dict[int, AnalyzerFinding] = {}
    for pattern, confidence in PE4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            finding_tags = list(tag)
            if _is_documentation_example(context, file_type):
                finding_tags.extend(["contextual-triage", "likely-benign-context"])
            if line_num in pe4_best and pe4_best[line_num].confidence >= confidence:
                continue
            pe4_best[line_num] = AnalyzerFinding(
                rule_id="PE4",
                message="Docker Socket Access",
                severity=Severity.HIGH,
                location=loc(line_num),
                confidence=confidence,
                tags=finding_tags,
                context=context,
                matched_text=match.group(0)[:200],
                complete_match=match.group(0),
            )
    findings.extend(pe4_best.values())
    # Collect best-confidence PE5 finding per line — a single `docker run` line
    # often matches multiple flags (e.g. --privileged + --cap-add=SYS_ADMIN).
    pe5_best: dict[int, AnalyzerFinding] = {}
    for pattern, confidence in PE5_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            context = get_context(content, match.start())
            finding_tags = list(tag)
            if _is_documentation_example(context, file_type):
                finding_tags.extend(["contextual-triage", "likely-benign-context"])
            if line_num in pe5_best and pe5_best[line_num].confidence >= confidence:
                continue
            pe5_best[line_num] = AnalyzerFinding(
                rule_id="PE5",
                message="Privileged Container / Container Escape",
                severity=Severity.HIGH,
                location=loc(line_num),
                confidence=confidence,
                tags=finding_tags,
                context=context,
                matched_text=match.group(0)[:200],
                complete_match=match.group(0),
            )
    findings.extend(pe5_best.values())
    return findings


_DOCUMENTATION_EXAMPLE_INDICATORS = (
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
)


def _has_documentation_indicator(context: str, indicators: tuple[str, ...]) -> bool:
    ctx_lower = context.lower()
    return any(indicator in ctx_lower for indicator in indicators)


def _is_documentation_example(context: str, file_type: str) -> bool:
    if file_type not in {"markdown", "text"}:
        return False
    return _has_documentation_indicator(context, _DOCUMENTATION_EXAMPLE_INDICATORS)


def _is_pe3_documentation_example(
    content: str,
    match: re.Match[str],
    file_type: str,
    file_path: str,
) -> bool:
    """Filter reviewed, position-bound access-token documentation forms.

    Generic words such as ``example``, ``documentation``, ``Required``, and
    ``environment variable`` are attacker-controllable prose and must never
    suppress an otherwise actionable credential-access match. Even negated
    references remain findings because another malicious clause can share the
    same line. The OAuth lifecycle exception is separately bounded by noun
    grammar, lifecycle evidence, and action/sensitive-source vetoes.
    """
    if file_type not in {"markdown", "text"}:
        return False
    if match.group(0).lower() not in {"access token", "access tokens"}:
        return False

    line = _source_line(content, match)
    navigation = _PE3_SAFE_ACCESS_TOKEN_NAVIGATION.search(line)
    if navigation is not None:
        line_start = content.rfind("\n", 0, match.start()) + 1
        match_span = (match.start() - line_start, match.end() - line_start)
        if navigation.span("target") == match_span:
            return True

    return _is_access_token_documentation_noun(content, match, file_type, file_path)


def _is_negated_safety_constraint(content: str, match: re.Match[str]) -> bool:
    """Return True when a privilege-escalation phrase is forbidden in policy prose."""
    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    if line_end == -1:
        line_end = len(content)
    line = content[line_start:line_end]
    local_start = match.start() - line_start
    phrase = line[local_start : local_start + len(match.group(0))]
    escaped = re.escape(phrase.strip())
    if not escaped:
        return False
    clause_start = max(line.rfind(sep, 0, local_start) for sep in ".;:")
    prefix = line[clause_start + 1 : local_start]
    safe_gap = r"(?:(?:ever|again|directly|intentionally|explicitly|attempt\s+to|try\s+to)\s+){0,2}"
    negation = r"(?:must\s+not|do\s+not|don't|never|should\s+not)\s+"
    return (
        re.search(negation + safe_gap + escaped + r"$", prefix + phrase, re.IGNORECASE) is not None
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run privilege_escalation patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
