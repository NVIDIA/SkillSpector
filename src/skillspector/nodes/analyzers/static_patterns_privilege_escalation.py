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
from bisect import bisect_right

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    LINE_BREAK_CHARS,
    LOGICAL_LINE_BREAK,
    MARKDOWN_FENCE_CLOSE,
    MARKDOWN_FENCE_OPEN,
    get_context,
    get_line_number,
)
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


def _source_line_metadata(content: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    starts = [0]
    ends: list[int] = []
    for separator in LOGICAL_LINE_BREAK.finditer(content):
        ends.append(separator.start())
        starts.append(separator.end())
    ends.append(len(content))
    return tuple(starts), tuple(ends)


def _source_line_bounds(
    content: str,
    match: re.Match[str],
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
) -> tuple[int, int]:
    if line_starts is None or line_ends is None:
        line_starts, line_ends = _source_line_metadata(content)
    index = bisect_right(line_starts, match.start()) - 1
    return line_starts[index], line_ends[index]


def _source_line(
    content: str,
    match: re.Match[str],
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
) -> str:
    """Return only the source line containing *match*."""
    line_start, line_end = _source_line_bounds(content, match, line_starts, line_ends)
    return content[line_start:line_end]


_PE3_CREDENTIAL_STORE_WORDS = frozenset({"keychain", "keyring", "gnome-keyring"})
# Attacker-controlled credential placement remains actionable, including Save/Put/Write.
_PE3_CREDENTIAL_STORE_AFTER_VERBS = (
    r"access(?:es|ed|ing)?|copy|copies|copied|copying|dump(?:s|ed|ing)?|"
    r"exfiltrat(?:e|es|ed|ing|ion)|export(?:s|ed|ing)?|extract(?:s|ed|ing)?|"
    r"fetch(?:es|ed|ing)?|get(?:s|ting)?|grab(?:s|bed|bing)?|harvest(?:s|ed|ing)?|"
    r"load(?:s|ed|ing)?|lookup|obtain(?:s|ed|ing)?|open(?:s|ed|ing)?|pull(?:s|ed|ing)?|"
    r"query|queries|queried|querying|read(?:s|ing)?|retrieve(?:s|d|ing)?|scrape(?:s|d|ing)?|"
    r"send(?:s|ing|sent)?|steal(?:s|ing|stolen)?|transmit(?:s|ted|ting)?|"
    r"unlock(?:s|ed|ing)?|upload(?:s|ed|ing)?|save(?:s|d|ing)?|put(?:s|ting)?|"
    r"write|writes|wrote|writing|written|"
    r"store(?:s|d|ing)?|remove(?:s|d|ing)?|delete(?:s|d|ing)?|clear(?:s|ed|ing)?|"
    r"update(?:s|d|ing)?|add(?:s|ed|ing)?|set(?:s|ting)?|use(?:s|ing)?"
)
_PE3_CREDENTIAL_STORE_OPERATION = re.compile(
    rf"\b(?:{_PE3_CREDENTIAL_STORE_AFTER_VERBS})\b"
    r"(?:\s+(?:the|a|an|my|your|local|credentials?|secrets?|passwords?|"
    r"tokens?|keys?|contents?|system|from|to|for|in|on)){0,8}\s*$",
    re.IGNORECASE,
)
_PE3_CREDENTIAL_STORE_OPERATION_AFTER = re.compile(
    rf"^\s*(?:(?:and|then|but)\s+)?(?:(?:is|was|can|will|should|must)\s+)?"
    rf"(?:used\s+(?:to|for)\s+)?"
    rf"(?:{_PE3_CREDENTIAL_STORE_AFTER_VERBS})\b",
    re.IGNORECASE,
)
_PE3_CREDENTIAL_STORE_DOCUMENTATION = re.compile(
    r"^\s+(?:api\s+documentation|cli\s+reference|access\s+policy|access\s+controls|"
    r"lookup\s+table|query\s+syntax|export\s+format)\b",
    re.IGNORECASE,
)
_PE3_BENIGN_READING_PURPOSE_AFTER = re.compile(
    r"^\s+(?:is\s+)?(?:solely\s+for\s+reading|for\s+reading(?:\s+purposes?)?\s+only|"
    r"only\s+for\s+reading(?:\s+purposes?)?)\s*$",
    re.IGNORECASE,
)
_PE3_CREDENTIAL_STORE_CALL = re.compile(
    r"\s*[.]\s*(?:add|clear|delete|get|remove|save|set|store|update|write)"
    r"\w*\s*(?=\()",
    re.IGNORECASE,
)
_PE3_CREDENTIAL_STORE_CLI = re.compile(
    r"\b(?:security\s+)?find-generic-password\b(?P<args>[^.;:\n]*)$", re.IGNORECASE
)


def _cli_targets_credential_store_noun(before_noun: str, noun: str) -> bool:
    """Accept CLI evidence only when it has not already named another store noun."""
    cli = _PE3_CREDENTIAL_STORE_CLI.search(f"{before_noun}{noun}")
    if cli is None:
        return False
    args = cli.group("args").rstrip()
    if not args.lower().endswith(noun.lower()):
        return False
    args_before_noun = args[: -len(noun)].rstrip()
    if re.search(
        r"\b(?:and|then|document|describe|reference|the)\b", args_before_noun, re.IGNORECASE
    ):
        return False
    return not any(
        word != noun.lower() and re.search(rf"\b{re.escape(word)}\b", args, re.IGNORECASE)
        for word in _PE3_CREDENTIAL_STORE_WORDS
    )


def _markdown_fence_ranges(content: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    active: tuple[str, int, int] | None = None
    offset = 0
    for line in content.splitlines(keepends=True):
        stripped = line.rstrip(LINE_BREAK_CHARS)
        closing = MARKDOWN_FENCE_CLOSE.fullmatch(stripped)
        if active is not None:
            if closing and closing.group(1)[0] == active[0] and len(closing.group(1)) >= active[1]:
                ranges.append((active[2], offset))
                active = None
        else:
            opening = MARKDOWN_FENCE_OPEN.fullmatch(stripped)
            if opening:
                marker = opening.group(1)
                active = (marker[0], len(marker), offset + len(line))
        offset += len(line)
    if active is not None:
        ranges.append((active[2], len(content)))
    return ranges


def _is_bare_credential_store_noun(
    content: str,
    match: re.Match[str],
    file_type: str,
    fence_ranges: list[tuple[int, int]] | None = None,
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
) -> bool:
    """Suppress only descriptive credential-store nouns in prose."""
    if file_type not in {"markdown", "text"}:
        return False
    if match.group(0).lower() not in _PE3_CREDENTIAL_STORE_WORDS:
        return False
    ranges = _markdown_fence_ranges(content) if fence_ranges is None else fence_ranges
    if any(start <= match.start() < end for start, end in ranges):
        return False
    line_start, line_end = _source_line_bounds(content, match, line_starts, line_ends)
    relation_start = max(line_start, match.start() - 80)
    relation_end = min(line_end, match.end() + 80)
    relation = content[relation_start:relation_end]
    noun_offset = match.start() - relation_start
    separators_before = [
        (separator.start(), 1)
        for separator in re.finditer(r"[.,;:](?=\s|$)", relation[:noun_offset])
    ] + [
        (separator.start(), len(separator.group(0)))
        for separator in re.finditer(r"\b(?:and|then|but|or)\b", relation[:noun_offset])
    ]
    clause_start, clause_prefix_length = max(separators_before, default=(-1, 0))
    after_relation = relation[noun_offset:]
    separators_after = [
        separator.start() for separator in re.finditer(r"[.,;:](?=\s|$)", after_relation)
    ]
    for separator in re.finditer(r"\b(?:and|then|but|or)\b", after_relation):
        if re.search(
            r"\b(?:keychain|keyring|gnome-keyring)\b",
            after_relation[separator.end() :],
            re.IGNORECASE,
        ):
            separators_after.append(separator.start())
    clause_end = noun_offset + min(separators_after) if separators_after else len(relation)
    clause_start_offset = clause_start + clause_prefix_length if clause_start >= 0 else 0
    clause = relation[clause_start_offset:clause_end]
    noun_start = noun_offset - clause_start_offset
    noun_end = noun_start + match.end() - match.start()
    before_noun = clause[:noun_start]
    after_noun = clause[noun_end:]
    operation = _PE3_CREDENTIAL_STORE_OPERATION.search(before_noun)
    operation_after = _PE3_CREDENTIAL_STORE_OPERATION_AFTER.search(after_noun)
    call = _PE3_CREDENTIAL_STORE_CALL.match(after_noun)
    cli = _cli_targets_credential_store_noun(before_noun, match.group(0))
    documentation = _PE3_CREDENTIAL_STORE_DOCUMENTATION.match(after_noun)
    if documentation:
        documentation_tail = after_noun[documentation.end() :]
        tail_is_explanatory = re.match(
            r"\s+(?:for|about|with|on|that|which|of)\b", documentation_tail, re.IGNORECASE
        )
        if (
            _PE3_CREDENTIAL_STORE_OPERATION_AFTER.search(documentation_tail) is None
            or tail_is_explanatory
        ):
            return True
    if (
        _PE3_BENIGN_READING_PURPOSE_AFTER.fullmatch(after_noun)
        and operation is None
        and not call
        and not cli
    ):
        return True
    if not (operation or operation_after or call or cli):
        return True
    # Any operation tied to this exact noun, including a read, dominates benign prose.
    return False


def _is_access_token_documentation_noun(
    content: str,
    match: re.Match[str],
    file_type: str,
    file_path: str,
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
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

    line = _source_line(content, match, line_starts, line_ends)
    line_start, _ = _source_line_bounds(content, match, line_starts, line_ends)
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
    content: str,
    match: re.Match[str],
    file_type: str,
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
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
    line_starts, line_ends = _source_line_metadata(content)
    fence_ranges = _markdown_fence_ranges(content) if file_type in {"markdown", "text"} else None

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
                )
            )
    for pattern, confidence in PE3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            if _is_bare_credential_store_noun(
                content, match, file_type, fence_ranges, line_starts, line_ends
            ):
                continue
            line_num = bisect_right(line_starts, match.start())
            context = get_context(content, match.start())
            contextual = any(
                (
                    _is_pe3_documentation_example(
                        content, match, file_type, file_path, line_starts, line_ends
                    ),
                    _is_qualified_benign_access_requirement(
                        content, match, file_type, line_starts, line_ends
                    ),
                    _is_read_only_passwd_volume_match(content, match),
                    _is_negated_safety_constraint(content, match, line_starts, line_ends),
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
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
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

    line = _source_line(content, match, line_starts, line_ends)
    navigation = _PE3_SAFE_ACCESS_TOKEN_NAVIGATION.search(line)
    if navigation is not None:
        line_start, _ = _source_line_bounds(content, match, line_starts, line_ends)
        match_span = (match.start() - line_start, match.end() - line_start)
        if navigation.span("target") == match_span:
            return True

    return _is_access_token_documentation_noun(
        content, match, file_type, file_path, line_starts, line_ends
    )


def _is_negated_safety_constraint(
    content: str,
    match: re.Match[str],
    line_starts: tuple[int, ...] | None = None,
    line_ends: tuple[int, ...] | None = None,
) -> bool:
    """Return True when a privilege-escalation phrase is forbidden in policy prose."""
    line_start, line_end = _source_line_bounds(content, match, line_starts, line_ends)
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
