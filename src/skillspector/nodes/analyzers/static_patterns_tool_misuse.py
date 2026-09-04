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
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_tool_misuse"

_SHELL_COMMAND_WORD_START_RE = re.compile(r"[rRdDeE$'\"`\\]")
_SHELL_COMMAND_WORD_CHARS = 4096
_ROOT_GLOB_COMMAND_CHARS = 8192
_SHELL_HORIZONTAL_OR_CONTINUED_GAP = r"(?>(?:\\\r?\n)*[ \t](?:[ \t]|\\\r?\n)*)"
_BOUNDED_SHELL_CLAUSE_ATOM = r"(?:\\(?:\r?\n|[^\n])|[^\\\n|;&])"
_STATIC_BRACE_WORD_CHARS = 256
_PRINTF_STATIC_CHARS = 256
_PRINTF_STATIC_ARGUMENTS = 32
_PRINTF_STATIC_WORD_RE = re.compile(r"[-A-Za-z0-9_./*?%]{0,64}")
_DESTRUCTIVE_COMMAND_BASENAMES = frozenset({"rm", "del", "erase"})
_QUOTED_GLOB_SENTINEL = "\ue000"
_DYNAMIC_SHELL_WORD_SENTINEL = "\ue001"
_ROOT_GLOB_DOCUMENTATION_LINE_RE = re.compile(
    r"[ \t]*(?:(?:[-*+]|#{1,6})[ \t]+)?"
    r"(?:(?:(?:documentation|note|example)[ \t]*:[ \t]*)"
    r"(?:(?:the|this|a|an)[ \t]+)?|(?:the|this|a|an)[ \t]+)"
    r"(?:(?:gnu|posix(?:\.[0-9]+)?|unix)[ \t]+)?rm[ \t]+(?:utility|command|tool)\b"
    r"[^;\n]{0,160}\b(?:accepts?|supports?)\b"
    r"[^;\n]{0,160}\b(?:denotes?|means?|represents?)\b"
    r"[^;\n]{0,80}\b(?:wildcard|glob|option|flag)\b[.!?]?[ \t]*",
    re.IGNORECASE,
)
_ROOT_GLOB_EXECUTION_PREFIX_RE = re.compile(
    r"\b(?P<action>run|execute|invoke|issue|launch|perform|call|eval|type|submit|"
    r"enter|paste|carry[ \t]+out)\b[^.;\n]{0,240}$",
    re.IGNORECASE,
)
_ROOT_GLOB_NEGATED_EXECUTION_RE = re.compile(
    r"(?:(?:\bdo[ \t]+not|\bdon't|\bnever|\bavoid|\bmust[ \t]+not)"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?!(?:so|but|yet|then|therefore|however|instead)\b)\w+){0,8}[ \t,]*|"
    r"\bnot[ \t,]*|"
    r"\bnot(?:[ \t]*,[ \t]*|[ \t]+)"
    r"(?:ever|directly|immediately|actually)[ \t,]*|"
    r"\bnot(?:[ \t]*,[ \t]*|[ \t]+)(?:under|in)"
    r"(?:[ \t]*,[ \t]*|[ \t]+)(?:any|all|these|those)"
    r"(?:(?:[ \t]*,[ \t]*|[ \t]+)\w+){0,4}[ \t,]*)$",
    re.IGNORECASE,
)
_ROOT_GLOB_SAFETY_WARNING_PREFIX_RE = re.compile(
    r"[ \t]*(?:please[ \t]+)?(?:"
    r"(?:do[ \t]+not|don't|never|must[ \t]+not)[ \t]+"
    r"(?:run|execute|invoke|issue|launch|perform|call|eval|type|submit|enter|paste|use)"
    r"|avoid[ \t]+(?:running|executing|invoking|issuing|launching|performing|calling|"
    r"evaluating|typing|submitting|entering|pasting|using)"
    r"|refrain[ \t]+from[ \t]+(?:running|executing|invoking|issuing|launching|"
    r"performing|calling|evaluating|typing|submitting|entering|pasting|using)"
    r")[ \t]+(?:the[ \t]+)?",
    re.IGNORECASE,
)
_ROOT_GLOB_AFFIRMATIVE_NEGATION_PREFIX_RE = re.compile(
    r".*\b(?:do[ \t]+not|don't|never)[ \t]+(?:"
    r"forget(?:[ \t]+to|[ \t]+that(?:[ \t]+you)?[ \t]+"
    r"(?:must|should|shall|will|need[ \t]+to)"
    r"|[ \t]*,[ \t]*(?:and[ \t]+)?(?:be|make)[ \t]+sure[ \t]+to)"
    r"|hesitate(?:[ \t]+at[ \t]+all)?[ \t]+to"
    r"|fail[ \t]+to)[ \t]*$",
    re.IGNORECASE,
)

# TM1: Tool Parameter Abuse — dangerous parameter values
TM1_PATTERNS = [
    # shell=True is a classic command injection vector
    (r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", 0.8),
    (r"Popen\s*\([^)]*shell\s*=\s*True", 0.8),
    # Bound command names on both sides so prefixes such as rmm/ (RAPIDS
    # Memory Manager headers) are not interpreted as destructive commands.
    # Keep the scan within one bounded shell command.  The former ``[^|]*``
    # crossed newlines/``;``/``&`` and rescanned the remainder of a document
    # for every prose mention of ``rm``, making repeated documentation
    # quadratic.  Escaped newlines remain valid shell continuations.
    (
        rf"\b(?:rm|del|erase)\b{_SHELL_HORIZONTAL_OR_CONTINUED_GAP}"
        rf"{_BOUNDED_SHELL_CLAUSE_ATOM}{{0,{_ROOT_GLOB_COMMAND_CHARS}}}"
        rf"(?:\\)?-(?:rf|fr|r){_SHELL_HORIZONTAL_OR_CONTINUED_GAP}[/~]",
        0.9,
    ),
    (r"--force\s+(?:delete|remove|push|reset|clean)", 0.7),
    # A bare application-defined --no-verify flag is ambiguous. Match it only
    # for known Git hook bypasses below; retain the other explicit unsafe flags.
    (r"--no-?(?:check|validate|confirm|protect|safe)\b", 0.75),
    (r"--skip-?(?:validation|verification|checks?|auth|tests?)", 0.7),
    # --allow-empty is a benign git-commit option, unlike the bypass flags below.
    (r"--allow-?(?:root|unrelated|unsafe)\b", 0.65),
    # Dangerous globs and wildcards in destructive commands
    # Match a path in the actual rm argument token. Stop at whitespace and shell
    # redirection operators so `rm "$VAR" 2>/dev/null` does not borrow the slash
    # from the redirection target.
    (
        r"\brm\b\s+(?:-[A-Za-z]+\s+)*(?:--\s+)?"
        r"(?:['\"][^'\"]*/[^'\"]*['\"]|[^\s|;&>]*/[^\s|;&>]*)",
        0.85,
    ),
    (r"\bshutil\.rmtree\s*\(\s*['\"]\s*/", 0.85),
    (r"(?:chmod|chown)\s+[^|]*(?:777|666|a\+rwx)", 0.8),
    # Git force operations
    (r"git\s+push\s+[^|]*--force", 0.7),
    (r"git\s+reset\s+--hard", 0.65),
    (r"git\s+clean\s+-[fd]+x", 0.7),
    (r"\bgit\s+(?:am|commit|merge|push)\b[^\n|]*--no-verify\b", 0.75),
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
        r"(?:set|pass|use)\s+(?:the\s+)?(?:parameter|argument|flag|option)\s+(?:to\s+)?(?:shell\s*=\s*True|--force|-rf)\b",
        0.8,
    ),
]

# TM2: Chaining Abuse — chained commands to bypass safety
TM2_PATTERNS = [
    # Shell command chaining with dangerous commands (\b prevents substring matches)
    (r"(?:&&|;)\s*\b(?:rm\b|del\b|erase\b)\s+-", 0.75),
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

_SAFE_CACHE_CLEANUP_RE = re.compile(
    r"\brm\s+-rf\s+(?P<quote>['\"]?)(?P<path>(?:\$?\{?HOME\}?|~)/\.cache/[^\s;&|'\"]+)(?P=quote)(?:\s*(?:$|[;&|]))",
    re.IGNORECASE,
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


def _is_safe_cache_cleanup(matched_text: str) -> bool:
    """Return True for scoped cleanup of a tool-owned user cache path."""
    match = _SAFE_CACHE_CLEANUP_RE.fullmatch(matched_text)
    if not match:
        return False
    parts = match.group("path").split("/")
    lowered_parts = [part.lower() for part in parts]
    cache_index = lowered_parts.index(".cache")
    cache_parts = parts[cache_index + 1 :]
    return bool(cache_parts) and not any(
        part in ("", ".", "..") or "*" in part for part in cache_parts
    )


@dataclass(frozen=True)
class _ShellToken:
    text: str
    root_glob: bool
    glob_projection: str
    has_quoted_content: bool
    brace_expansion: bool
    leading_tilde_unquoted: bool


@dataclass(frozen=True)
class _ShellCommandWord:
    text: str
    end: int
    dynamic: bool
    limited: bool = False


@dataclass(frozen=True)
class _ParameterExpansionEnd:
    """Cached parameter endpoint plus its enclosing-quote transition."""

    end: int | None
    inherited_quote_closed: bool = False


@dataclass
class _ShellDelimiterFrame:
    """One iterative shell-expansion delimiter frame."""

    kind: str
    start: int | None
    quote: str | None = None
    ansi_c_quote: bool = False
    word_started: bool = False
    inherited_double_quote: bool = False
    inherited_quote_closed: bool = False


def _is_shell_command_word_start(content: str, start: int) -> bool:
    if start == 0:
        return True
    previous = content[start - 1]
    if previous.isspace() or previous in ";|&()<>/":
        return True
    if previous in "'\"`":
        before_quote = start - 2
        return (
            before_quote < 0
            or content[before_quote].isspace()
            or content[before_quote] in ";|&()<>{}/"
        )
    return False


def _has_quoted_assignment_prefix(content: str, start: int) -> bool:
    """Return whether a quoted candidate starts with a shell assignment name."""
    if start >= len(content) or content[start] not in "'\"":
        return False
    cursor = start + 1
    if cursor >= len(content) or not (content[cursor].isalpha() or content[cursor] == "_"):
        return False
    cursor += 1
    while cursor < len(content) and (content[cursor].isalnum() or content[cursor] == "_"):
        cursor += 1
    return cursor < len(content) and content[cursor] == "="


def _decode_ansi_c_escape(content: str, start: int, limit: int) -> tuple[str, int]:
    """Decode one bounded Bash ANSI-C escape without evaluating shell input."""
    if start + 1 >= limit:
        return "\\", start + 1
    escaped = content[start + 1]
    simple = {
        "a": "\a",
        "b": "\b",
        "e": "\x1b",
        "E": "\x1b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
        "'": "'",
        '"': '"',
        "?": "?",
    }
    if escaped in simple:
        return simple[escaped], start + 2
    if escaped in "xXuU":
        maximum_digits = {"x": 2, "X": 2, "u": 4, "U": 8}[escaped]
        cursor = start + 2
        while (
            cursor < limit
            and cursor < start + 2 + maximum_digits
            and content[cursor] in "0123456789abcdefABCDEF"
        ):
            cursor += 1
        if cursor > start + 2:
            value = int(content[start + 2 : cursor], 16)
            if value <= sys.maxunicode:
                return chr(value), cursor
    if escaped in "01234567":
        cursor = start + 1
        while cursor < limit and cursor < start + 4 and content[cursor] in "01234567":
            cursor += 1
        return chr(int(content[start + 1 : cursor], 8)), cursor
    if escaped == "c" and start + 2 < limit:
        return chr(ord(content[start + 2].upper()) ^ 0x40), start + 3
    return "\\" + escaped, start + 2


def _skip_ansi_c_quote_tail(content: str, start: int, limit: int) -> int | None:
    """Find the real closing quote after a NUL, skipping escaped quotes."""
    cursor = start
    while cursor < limit:
        if content[cursor] == "\\" and cursor + 1 < limit:
            cursor += 2
            continue
        if content[cursor] == "'":
            return cursor
        cursor += 1
    return None


def _skip_shell_delimited_expansion(
    content: str,
    start: int,
    limit: int,
    root_kind: str,
    check_runtime: Callable[[], None] | None,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None,
    substitution_end_cache: dict[int, int | None] | None,
    backtick_end_cache: dict[int, int | None] | None,
    root_inherited_double_quote: bool = False,
    inherited_quote_closed: list[bool] | None = None,
) -> int | None:
    """Skip nested shell delimiters iteratively and cache every typed endpoint."""
    opener_width = 1 if root_kind == "backtick" else 2
    frames = [
        _ShellDelimiterFrame(
            root_kind,
            start,
            inherited_double_quote=root_inherited_double_quote,
        )
    ]
    cursor = start + opener_width

    def cache_frame(frame: _ShellDelimiterFrame, end: int | None) -> None:
        if frame.start is None:
            return
        if frame.kind == "parameter" and parameter_end_cache is not None:
            parameter_end_cache[frame.start] = _ParameterExpansionEnd(
                end,
                frame.inherited_quote_closed,
            )
        elif frame.kind == "command" and substitution_end_cache is not None:
            substitution_end_cache[frame.start] = end
        elif frame.kind == "backtick" and backtick_end_cache is not None:
            backtick_end_cache[frame.start] = end

    def cache_failure() -> None:
        for frame in frames:
            cache_frame(frame, None)

    def close_frame(end: int) -> int | None:
        frame = frames.pop()
        cache_frame(frame, end)
        return end if not frames else None

    def push(kind: str, frame_start: int | None, width: int) -> None:
        nonlocal cursor
        parent = frames[-1]
        if kind in {"parameter", "command", "backtick"} and parent.kind in {
            "command",
            "paren",
        }:
            parent.word_started = True
        inherited_double_quote = kind in {"parameter", "brace"} and (
            parent.quote == '"'
            or (
                parent.inherited_double_quote
                and parent.kind in {"parameter", "brace"}
                and parent.quote is None
            )
        )
        frames.append(
            _ShellDelimiterFrame(
                kind,
                frame_start,
                inherited_double_quote=inherited_double_quote,
            )
        )
        cursor += width

    while cursor < limit:
        if check_runtime is not None and cursor % 4096 == 0:
            check_runtime()
        frame = frames[-1]
        character = content[cursor]

        if frame.kind == "backtick":
            if character == "\\" and cursor + 1 < limit:
                cursor += 2
                continue
            if character == "`":
                endpoint = close_frame(cursor + 1)
                cursor += 1
                if endpoint is not None:
                    return endpoint
                continue
            cursor += 1
            continue

        if frame.quote is not None:
            if frame.quote == '"' and content.startswith("${", cursor):
                push("parameter", cursor, 2)
                continue
            if frame.quote == '"' and content.startswith("$(", cursor):
                push("command", cursor, 2)
                continue
            if frame.quote == '"' and character == "`":
                push("backtick", cursor, 1)
                continue
            if character == frame.quote:
                frame.quote = None
                frame.ansi_c_quote = False
            elif character == "\\" and frame.ansi_c_quote:
                _, cursor = _decode_ansi_c_escape(content, cursor, limit)
                continue
            elif character == "\\" and frame.quote == '"' and cursor + 1 < limit:
                cursor += 2
                continue
            cursor += 1
            continue

        if content.startswith("${", cursor):
            push("parameter", cursor, 2)
            continue
        if content.startswith("$(", cursor):
            push("command", cursor, 2)
            continue
        if character in "<>" and cursor + 1 < limit and content[cursor + 1] == "(":
            push("command", cursor, 2)
            continue
        if character == "`":
            push("backtick", cursor, 1)
            continue
        if (
            character == "$"
            and cursor + 1 < limit
            and content[cursor + 1] in "'\""
            and not (
                frame.inherited_double_quote
                and frame.kind in {"parameter", "brace"}
                and content[cursor + 1] in "'\""
            )
        ):
            frame.quote = content[cursor + 1]
            frame.ansi_c_quote = frame.quote == "'"
            if frame.kind in {"command", "paren"}:
                frame.word_started = True
            cursor += 2
            continue
        if (
            character == '"'
            and frame.inherited_double_quote
            and frame.kind in {"parameter", "brace"}
        ):
            # In ksh, $" inside a parameter expansion inherited from a
            # double-quoted word is a literal '$' followed by the closing
            # quote.  Do not mistake that quote for the start of a new
            # locale-translation string, or the scanner can hide commands
            # that follow the parameter expansion.
            frame.inherited_double_quote = False
            frame.inherited_quote_closed = True
            if inherited_quote_closed is not None:
                inherited_quote_closed[0] = True
            for parent in reversed(frames[:-1]):
                if parent.kind in {"parameter", "brace"}:
                    parent.inherited_double_quote = False
                    parent.inherited_quote_closed = True
                if parent.quote == '"':
                    parent.quote = None
                    parent.ansi_c_quote = False
                    break
            cursor += 1
            continue
        if character in "'\"" and not (
            character == "'"
            and frame.inherited_double_quote
            and frame.kind in {"parameter", "brace"}
        ):
            frame.quote = character
            frame.ansi_c_quote = False
            if frame.kind in {"command", "paren"}:
                frame.word_started = True
            cursor += 1
            continue
        if character == "\\" and cursor + 1 < limit:
            if frame.kind in {"command", "paren"}:
                frame.word_started = True
            cursor += 2
            continue

        if frame.kind in {"parameter", "brace"}:
            if character == "{":
                push("brace", None, 1)
                continue
            if character == "}":
                endpoint = close_frame(cursor + 1)
                cursor += 1
                if endpoint is not None:
                    return endpoint
                continue
            cursor += 1
            continue

        if character == "(":
            push("paren", None, 1)
            continue
        if character == ")":
            endpoint = close_frame(cursor + 1)
            cursor += 1
            if endpoint is not None:
                return endpoint
            continue
        if character == "#" and not frame.word_started:
            newline = content.find("\n", cursor + 1, limit)
            if newline < 0:
                cache_failure()
                return None
            cursor = newline
            continue
        if character.isspace() or character in ";|&":
            frame.word_started = False
        else:
            frame.word_started = True
        cursor += 1

    cache_failure()
    return None


def _skip_parameter_expansion(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None] | None = None,
    end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
    inherited_double_quote: bool = False,
    inherited_quote_closed: list[bool] | None = None,
) -> int | None:
    if end_cache is not None and start in end_cache:
        cached = end_cache[start]
        if inherited_quote_closed is not None:
            inherited_quote_closed[0] = cached.inherited_quote_closed
        return cached.end
    if start + 1 >= limit or content[start] != "$":
        return None
    next_character = content[start + 1]
    if next_character == "{":
        return _skip_shell_delimited_expansion(
            content,
            start,
            limit,
            "parameter",
            check_runtime,
            end_cache,
            substitution_end_cache,
            backtick_end_cache,
            inherited_double_quote,
            inherited_quote_closed,
        )
    if next_character.isalpha() or next_character == "_":
        cursor = start + 2
        while cursor < limit and (content[cursor].isalnum() or content[cursor] == "_"):
            cursor += 1
        return cursor
    if next_character.isdigit() or next_character in "*@#?$!-":
        return start + 2
    return None


def _is_ifs_expansion(content: str, start: int, end: int) -> bool:
    return content[start:end] in {"$IFS", "${IFS}"}


def _consume_printf_invocation(
    next_word: Callable[[], str | None],
) -> tuple[bool, bool]:
    """Resolve an allowlisted invocation; return ``(recognized, exact)``."""
    pending: str | None = None
    for _ in range(4):
        word = pending if pending is not None else next_word()
        pending = None
        if word is None:
            return False, False
        if _DYNAMIC_SHELL_WORD_SENTINEL in word:
            # A runtime expansion participates in the invocation or wrapper
            # command word. Its executable basename is not deterministic.
            return True, False
        command = word.casefold().rsplit("/", 1)[-1]
        if command == "printf":
            return True, True
        if command == "command":
            while True:
                word = next_word()
                if word == "-p":
                    continue
                if word == "--":
                    word = next_word()
                break
            if word is None or word.startswith("-"):
                return False, False
            pending = word
            continue
        if command == "builtin":
            word = next_word()
            if word == "--":
                word = next_word()
            if word is None or word.startswith("-"):
                return False, False
            pending = word
            continue
        if command != "env":
            return False, False
        while (word := next_word()) is not None:
            if word == "--":
                pending = next_word()
                break
            # ``env`` assignment values may contain quoted newlines and other
            # characters. Only the portable name and first ``=`` determine
            # whether this word is an assignment operand.
            if re.match(r"[A-Za-z_][A-Za-z0-9_]*=", word) is not None:
                continue
            if word in {"-i", "--ignore-environment"}:
                continue
            if word in {"-u", "--unset", "-C", "--chdir"}:
                if next_word() is None:
                    return True, False
                continue
            if re.fullmatch(r"(?:--unset|--chdir)=.+", word) is not None:
                continue
            if word.startswith("-"):
                # Preserve the existing fail-closed contract for unsupported
                # env options even though the final command cannot be resolved.
                return True, False
            pending = word
            break
        if pending is None:
            return False, False
    # The allowlisted wrapper chain exceeded the deterministic depth budget.
    # Treat it as recognized but inexact so callers record a coverage gap
    # instead of silently declaring a potentially reconstructed command safe.
    return True, False


def _printf_invocation_arguments(inner: str) -> tuple[bool, list[str]]:
    """Parse direct or allowlisted wrapper invocations of shell ``printf``."""
    cursor = 0
    limited = False

    def next_word() -> str | None:
        nonlocal cursor, limited
        word, cursor, word_limited = _next_shell_invocation_word(
            inner,
            cursor,
            lambda: None,
        )
        limited = limited or word_limited or (word is None and cursor < len(inner))
        return word

    recognized, exact = _consume_printf_invocation(next_word)
    if not recognized or not exact or limited:
        return recognized, []

    arguments: list[str] = []
    while cursor < len(inner):
        word = next_word()
        if word is None:
            return (True, []) if limited or cursor < len(inner) else (True, arguments)
        arguments.append(word)
    return True, arguments


def _next_shell_invocation_word(
    content: str,
    start: int,
    check_runtime: Callable[[], None],
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
) -> tuple[str | None, int, bool]:
    """Read one direct shell word without traversing nested substitutions."""
    cursor = start
    limit = len(content)
    while cursor < limit:
        if cursor % 4096 == 0:
            check_runtime()
        if content[cursor].isspace():
            cursor += 1
            continue
        if content.startswith("\\\r\n", cursor):
            cursor += 3
            continue
        if content.startswith("\\\n", cursor):
            cursor += 2
            continue
        break
    if cursor >= limit or content[cursor] in ");|&()<>":
        return None, cursor, False

    output: list[str] = []
    quote: str | None = None
    ansi_c_quote = False
    word_started = False
    while cursor < limit:
        if cursor % 4096 == 0:
            check_runtime()
        character = content[cursor]
        if quote is not None:
            if character == quote:
                quote = None
                ansi_c_quote = False
            elif character == "\\" and ansi_c_quote:
                decoded, cursor = _decode_ansi_c_escape(content, cursor, limit)
                if "\x00" in decoded:
                    closing_quote = _skip_ansi_c_quote_tail(content, cursor, limit)
                    if closing_quote is None:
                        return None, cursor, True
                    cursor = closing_quote
                    continue
                output.append(decoded)
                word_started = True
                continue
            elif quote == '"' and character == "\\" and cursor + 1 < limit:
                escaped = content[cursor + 1]
                if escaped == "\n":
                    cursor += 2
                    continue
                if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                    cursor += 3
                    continue
                if escaped in '$`"\\':
                    output.append(escaped)
                else:
                    output.extend(("\\", escaped))
                cursor += 2
                continue
            elif quote == '"' and character == "$":
                inherited_quote_closed = [False]
                if cursor + 1 < limit and content[cursor + 1] == "(":
                    parameter_end = _skip_command_substitution(
                        content,
                        cursor,
                        limit,
                        check_runtime,
                        substitution_end_cache,
                        parameter_end_cache,
                        backtick_end_cache,
                    )
                else:
                    parameter_end = _skip_parameter_expansion(
                        content,
                        cursor,
                        limit,
                        check_runtime,
                        parameter_end_cache,
                        substitution_end_cache,
                        backtick_end_cache,
                        True,
                        inherited_quote_closed,
                    )
                if parameter_end is None:
                    if cursor + 1 < limit and content[cursor + 1] in "({":
                        return None, cursor, True
                    output.append("$")
                    word_started = True
                    cursor += 1
                    continue
                output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
                word_started = True
                cursor = parameter_end
                if inherited_quote_closed[0]:
                    quote = None
                    ansi_c_quote = False
                continue
            elif quote == '"' and character == "`":
                substitution_end = _skip_backtick_substitution(
                    content,
                    cursor,
                    limit,
                    check_runtime,
                    backtick_end_cache,
                    parameter_end_cache,
                    substitution_end_cache,
                )
                if substitution_end is None:
                    return None, cursor, True
                output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
                word_started = True
                cursor = substitution_end
                continue
            else:
                output.append(character)
        elif character in "'\"":
            quote = character
            ansi_c_quote = False
            word_started = True
        elif character == "\\" and cursor + 1 < limit:
            escaped = content[cursor + 1]
            if escaped == "\n":
                cursor += 2
                continue
            if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                cursor += 3
                continue
            output.append(escaped)
            word_started = True
            cursor += 2
            continue
        elif character == "$":
            if cursor + 1 < limit and content[cursor + 1] in "'\"":
                quote = content[cursor + 1]
                ansi_c_quote = quote == "'"
                word_started = True
                cursor += 2
                continue
            if cursor + 1 < limit and content[cursor + 1] == "(":
                parameter_end = _skip_command_substitution(
                    content,
                    cursor,
                    limit,
                    check_runtime,
                    substitution_end_cache,
                    parameter_end_cache,
                    backtick_end_cache,
                )
            else:
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    limit,
                    check_runtime,
                    parameter_end_cache,
                    substitution_end_cache,
                    backtick_end_cache,
                )
            if parameter_end is None:
                if cursor + 1 < limit and content[cursor + 1] in "({":
                    return None, cursor, True
                output.append("$")
                word_started = True
                cursor += 1
                continue
            output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
            word_started = True
            cursor = parameter_end
            continue
        elif character == "`":
            substitution_end = _skip_backtick_substitution(
                content,
                cursor,
                limit,
                check_runtime,
                backtick_end_cache,
                parameter_end_cache,
                substitution_end_cache,
            )
            if substitution_end is None:
                return None, cursor, True
            output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
            word_started = True
            cursor = substitution_end
            continue
        elif character in "<>" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(
                content,
                cursor,
                limit,
                check_runtime,
                substitution_end_cache,
                parameter_end_cache,
                backtick_end_cache,
            )
            if substitution_end is None:
                return None, cursor, True
            output.append(_DYNAMIC_SHELL_WORD_SENTINEL)
            word_started = True
            cursor = substitution_end
            continue
        elif character.isspace() or character in ";|&()<>":
            break
        elif character == "#" and not word_started:
            return None, cursor, False
        else:
            output.append(character)
            word_started = True
            if len(output) > _SHELL_COMMAND_WORD_CHARS:
                return None, cursor, True
        cursor += 1
    if quote is not None:
        return None, cursor, True
    return "".join(output) if word_started else None, cursor, False


def _has_printf_invocation_prefix(
    content: str,
    start: int,
    check_runtime: Callable[[], None],
    parameter_end_cache: dict[int, _ParameterExpansionEnd],
    substitution_end_cache: dict[int, int | None],
    backtick_end_cache: dict[int, int | None],
) -> bool:
    """Recognize over-bound direct ``printf`` without copying or suffix rescans."""
    cursor = start + 2
    limited = False

    def next_word() -> str | None:
        nonlocal cursor, limited
        word, cursor, word_limited = _next_shell_invocation_word(
            content,
            cursor,
            check_runtime,
            parameter_end_cache,
            substitution_end_cache,
            backtick_end_cache,
        )
        limited = limited or word_limited
        return word

    recognized, _ = _consume_printf_invocation(next_word)
    return recognized or limited


def _static_printf_substitution(
    content: str,
    start: int,
    end: int,
    *,
    backtick: bool = False,
) -> str | None:
    """Evaluate a tiny, bounded subset of shell ``printf`` deterministically."""
    inner_start = start + (1 if backtick else 2)
    inner_end = end - 1
    inner = content[inner_start:inner_end]
    recognized, arguments = _printf_invocation_arguments(inner)
    if (
        not recognized
        or len(inner) > _PRINTF_STATIC_CHARS
        or re.search(r"[;|&<>\r\n]", inner) is not None
    ):
        return None
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or len(arguments) > _PRINTF_STATIC_ARGUMENTS + 1:
        return None
    format_word, *values = arguments
    if _PRINTF_STATIC_WORD_RE.fullmatch(format_word) is None or any(
        _PRINTF_STATIC_WORD_RE.fullmatch(value) is None for value in values
    ):
        return None

    segments: list[str | None] = []
    literal: list[str] = []
    cursor = 0
    while cursor < len(format_word):
        character = format_word[cursor]
        if character != "%":
            literal.append(character)
            cursor += 1
            continue
        if cursor + 1 >= len(format_word) or format_word[cursor + 1] not in "s%":
            return None
        if literal:
            segments.append("".join(literal))
            literal.clear()
        conversion = format_word[cursor + 1]
        segments.append(None if conversion == "s" else "%")
        cursor += 2
    if literal:
        segments.append("".join(literal))

    conversions = sum(segment is None for segment in segments)
    if conversions == 0:
        return format_word
    remaining = iter(values)
    output: list[str] = []
    cycles = max(1, (len(values) + conversions - 1) // conversions)
    for _ in range(cycles):
        for segment in segments:
            output.append(next(remaining, "") if segment is None else segment)
        if sum(len(piece) for piece in output) > 64:
            return None
    result = "".join(output)
    return result if _PRINTF_STATIC_WORD_RE.fullmatch(result) is not None else None


def _is_printf_substitution(
    content: str,
    start: int,
    end: int,
    *,
    backtick: bool = False,
) -> bool:
    """Return whether a substitution invokes the bounded ``printf`` evaluator."""
    inner_start = start + (1 if backtick else 2)
    inner_end = end - 1
    recognized, _ = _printf_invocation_arguments(content[inner_start:inner_end])
    return recognized


def _skip_backtick_substitution(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None] | None = None,
    end_cache: dict[int, int | None] | None = None,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
) -> int | None:
    if end_cache is not None and start in end_cache:
        return end_cache[start]
    return _skip_shell_delimited_expansion(
        content,
        start,
        limit,
        "backtick",
        check_runtime,
        parameter_end_cache,
        substitution_end_cache,
        end_cache,
    )


def _parse_shell_command_word(
    content: str,
    start: int,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    substitution_end_cache: dict[int, int | None] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
) -> _ShellCommandWord | None:
    output: list[str] = []
    quote: str | None = None
    ansi_c_quote = False
    dynamic = False
    limited = False
    cursor = start
    limit = len(content)
    while cursor < limit:
        character = content[cursor]
        if quote is not None:
            if character == quote:
                quote = None
                ansi_c_quote = False
            elif character == "\\" and ansi_c_quote:
                decoded, cursor = _decode_ansi_c_escape(content, cursor, limit)
                if "\x00" in decoded:
                    closing_quote = _skip_ansi_c_quote_tail(content, cursor, limit)
                    if closing_quote is None:
                        return None
                    cursor = closing_quote
                    continue
                output.append(decoded)
                continue
            elif quote == '"' and character == "$" and cursor + 1 < limit:
                inherited_quote_closed = [False]
                if content[cursor + 1] == "(":
                    substitution_end = _skip_command_substitution(
                        content,
                        cursor,
                        limit,
                        end_cache=substitution_end_cache,
                        parameter_end_cache=parameter_end_cache,
                        backtick_end_cache=backtick_end_cache,
                    )
                    if substitution_end is None:
                        return None
                    static_value = _static_printf_substitution(
                        content,
                        cursor,
                        substitution_end,
                    )
                    if static_value is None:
                        output.append("$()")
                        dynamic = True
                        limited = limited or _is_printf_substitution(
                            content,
                            cursor,
                            substitution_end,
                        )
                    else:
                        output.append(static_value)
                    cursor = substitution_end
                    continue
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    limit,
                    end_cache=parameter_end_cache,
                    substitution_end_cache=substitution_end_cache,
                    backtick_end_cache=backtick_end_cache,
                    inherited_double_quote=True,
                    inherited_quote_closed=inherited_quote_closed,
                )
                if parameter_end is not None:
                    output.append("$PARAM")
                    dynamic = True
                    cursor = parameter_end
                    if inherited_quote_closed[0]:
                        quote = None
                        ansi_c_quote = False
                    continue
            elif quote == '"' and character == "`":
                substitution_end = _skip_backtick_substitution(
                    content,
                    cursor,
                    limit,
                    end_cache=backtick_end_cache,
                    parameter_end_cache=parameter_end_cache,
                    substitution_end_cache=substitution_end_cache,
                )
                if substitution_end is None:
                    return None
                static_value = _static_printf_substitution(
                    content,
                    cursor,
                    substitution_end,
                    backtick=True,
                )
                if static_value is None:
                    output.append("$()")
                    dynamic = True
                    limited = limited or _is_printf_substitution(
                        content,
                        cursor,
                        substitution_end,
                        backtick=True,
                    )
                else:
                    output.append(static_value)
                cursor = substitution_end
                continue
            elif character == "\\" and quote == '"' and cursor + 1 < limit:
                escaped = content[cursor + 1]
                if escaped == "\n":
                    cursor += 2
                    continue
                if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                    cursor += 3
                    continue
                if escaped in '$`"\\':
                    output.append(escaped)
                else:
                    output.extend(("\\", escaped))
                cursor += 2
                continue
            else:
                output.append(character)
        elif character == "$" and cursor + 1 < limit and content[cursor + 1] in "'\"":
            quote = content[cursor + 1]
            ansi_c_quote = quote == "'"
            cursor += 2
            continue
        elif character == "$" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(
                content,
                cursor,
                limit,
                end_cache=substitution_end_cache,
                parameter_end_cache=parameter_end_cache,
                backtick_end_cache=backtick_end_cache,
            )
            if substitution_end is None:
                return None
            static_value = _static_printf_substitution(content, cursor, substitution_end)
            if static_value is None:
                output.append("$()")
                dynamic = True
                limited = limited or _is_printf_substitution(
                    content,
                    cursor,
                    substitution_end,
                )
            else:
                output.append(static_value)
            cursor = substitution_end
            continue
        elif character == "$":
            parameter_end = _skip_parameter_expansion(
                content,
                cursor,
                limit,
                end_cache=parameter_end_cache,
                substitution_end_cache=substitution_end_cache,
                backtick_end_cache=backtick_end_cache,
            )
            if parameter_end is not None:
                if _is_ifs_expansion(content, cursor, parameter_end):
                    cursor = parameter_end
                    break
                output.append("$PARAM")
                dynamic = True
                cursor = parameter_end
                continue
        elif character == "`":
            substitution_end = _skip_backtick_substitution(
                content,
                cursor,
                limit,
                end_cache=backtick_end_cache,
                parameter_end_cache=parameter_end_cache,
                substitution_end_cache=substitution_end_cache,
            )
            if substitution_end is None:
                return None
            static_value = _static_printf_substitution(
                content,
                cursor,
                substitution_end,
                backtick=True,
            )
            if static_value is None:
                output.append("$()")
                dynamic = True
                limited = limited or _is_printf_substitution(
                    content,
                    cursor,
                    substitution_end,
                    backtick=True,
                )
            else:
                output.append(static_value)
            cursor = substitution_end
            continue
        elif character in "'\"":
            quote = character
        elif character == "\\" and cursor + 1 < limit:
            if content[cursor + 1] == "\n":
                cursor += 2
                continue
            if content[cursor + 1] == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                cursor += 3
                continue
            cursor += 1
            output.append(content[cursor])
        elif character.isspace() or character in ";|&()<>":
            break
        else:
            output.append(character)
            if len(output) > _SHELL_COMMAND_WORD_CHARS:
                return _ShellCommandWord("".join(output), cursor, dynamic, limited=True)
        cursor += 1
    if quote is not None:
        return None
    return _ShellCommandWord("".join(output), cursor, dynamic, limited)


def _destructive_command_words(content: str) -> Iterator[tuple[int, int]]:
    parsed_through = 0
    parameter_end_cache: dict[int, _ParameterExpansionEnd] = {}
    substitution_end_cache: dict[int, int | None] = {}
    backtick_end_cache: dict[int, int | None] = {}
    for candidate in _SHELL_COMMAND_WORD_START_RE.finditer(content):
        start = candidate.start()
        if start < parsed_through:
            continue
        if not _is_shell_command_word_start(content, start):
            continue
        if _has_quoted_assignment_prefix(content, start):
            # This candidate cannot name a destructive command. Nested command
            # substitutions remain independent candidates in the outer scan.
            continue
        if content.startswith("$(", start):
            substitution_end = _bounded_static_substitution_end(content, start)
            if substitution_end is None or not _is_printf_substitution(
                content,
                start,
                substitution_end,
            ):
                # Dynamic substitutions cannot deterministically name a
                # destructive command. Their inner literal commands remain
                # independent candidates; the bounded structural check avoids
                # reparsing nested suffixes quadratically.
                continue
        parsed = _parse_shell_command_word(
            content,
            start,
            parameter_end_cache,
            substitution_end_cache,
            backtick_end_cache,
        )
        if parsed is None:
            continue
        # A long concatenated quoted word can expose every closing quote as a
        # regex candidate. Skip the rest only when it cannot be an executed
        # whitespace-bearing wrapper or a dynamic substitution wrapper.
        candidate_character = content[start]
        if (
            candidate_character in "'\""
            and not parsed.dynamic
            and not any(character.isspace() for character in parsed.text)
        ):
            parsed_through = max(parsed_through, parsed.end)
        command_basename = parsed.text.casefold().rsplit("/", 1)[-1]
        if (
            not parsed.dynamic
            and not parsed.limited
            and command_basename in _DESTRUCTIVE_COMMAND_BASENAMES
        ):
            parsed_through = max(parsed_through, parsed.end)
            yield start, parsed.end


def _has_shell_command_word_exhaustion(
    content: str,
    check_runtime: Callable[[], None],
) -> bool:
    """Find candidate command words whose deterministic parse hit a safety bound."""
    parsed_through = 0
    parameter_end_cache: dict[int, _ParameterExpansionEnd] = {}
    substitution_end_cache: dict[int, int | None] = {}
    backtick_end_cache: dict[int, int | None] = {}
    for candidate in _SHELL_COMMAND_WORD_START_RE.finditer(content):
        check_runtime()
        start = candidate.start()
        if start < parsed_through or not _is_shell_command_word_start(content, start):
            continue
        if _has_quoted_assignment_prefix(content, start):
            continue
        if content.startswith("$(", start):
            substitution_end = _bounded_static_substitution_end(content, start)
            if substitution_end is None:
                if _has_printf_invocation_prefix(
                    content,
                    start,
                    check_runtime,
                    parameter_end_cache,
                    substitution_end_cache,
                    backtick_end_cache,
                ):
                    return True
                continue
            if not _is_printf_substitution(
                content,
                start,
                substitution_end,
            ):
                continue
        parsed = _parse_shell_command_word(
            content,
            start,
            parameter_end_cache,
            substitution_end_cache,
            backtick_end_cache,
        )
        if parsed is None:
            continue
        parsed_through = max(parsed_through, parsed.end)
        if parsed.limited:
            return True
    return False


def _command_wrapper_quote(content: str, command_start: int) -> str | None:
    if command_start == 0 or content[command_start - 1] not in "'\"`":
        return None
    backslashes = 0
    cursor = command_start - 2
    while cursor >= 0 and content[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return content[command_start - 1] if backslashes % 2 == 0 else None


def _skip_command_substitution(
    content: str,
    start: int,
    limit: int,
    check_runtime: Callable[[], None] | None = None,
    end_cache: dict[int, int | None] | None = None,
    parameter_end_cache: dict[int, _ParameterExpansionEnd] | None = None,
    backtick_end_cache: dict[int, int | None] | None = None,
) -> int | None:
    if end_cache is not None and start in end_cache:
        return end_cache[start]
    return _skip_shell_delimited_expansion(
        content,
        start,
        limit,
        "command",
        check_runtime,
        parameter_end_cache,
        end_cache,
        backtick_end_cache,
    )


def _bounded_static_substitution_end(content: str, start: int) -> int | None:
    """Return a structurally valid ``$()`` close within the static evaluator bound."""
    inner_start = start + 2
    scan_limit = min(len(content), inner_start + _PRINTF_STATIC_CHARS + 1)
    end = _skip_command_substitution(content, start, scan_limit)
    if end is None or end - 1 - inner_start > _PRINTF_STATIC_CHARS:
        return None
    return end


def _bounded_shell_tokens(
    content: str,
    command_start: int,
    body_start: int,
) -> tuple[tuple[_ShellToken, ...], int, bool]:
    """Return argument words from one security-view-bounded shell command."""
    tokens: list[_ShellToken] = []
    current: list[str] = []
    current_glob_projection: list[str] = []
    word_started = False
    current_is_argument = True
    current_has_unquoted_star = False
    current_can_collapse_to_root_glob = True
    current_has_quoted_content = False
    current_unquoted_brace_open = False
    current_unquoted_brace_close = False
    current_unquoted_brace_comma = False
    current_leading_tilde_unquoted = False
    source_word_has_content = False
    expect_redirection_target = False
    quote: str | None = None
    ansi_c_quote = False
    boundary_incomplete = False
    parse_limited = False
    cursor = body_start
    limit = min(len(content), body_start + _ROOT_GLOB_COMMAND_CHARS)
    wrapper_quote = _command_wrapper_quote(content, command_start)

    def start_word() -> None:
        nonlocal word_started, current_is_argument, expect_redirection_target
        if not word_started:
            word_started = True
            current_is_argument = not expect_redirection_target
            expect_redirection_target = False

    def mark_quoted_word() -> None:
        nonlocal current_has_quoted_content, source_word_has_content
        start_word()
        current_has_quoted_content = True
        source_word_has_content = True

    def append_piece(
        piece: str,
        *,
        unquoted: bool = False,
        dynamic_can_be_empty: bool = False,
        quoted: bool = False,
        unquoted_brace_syntax: bool = False,
    ) -> None:
        nonlocal current_has_unquoted_star
        nonlocal current_can_collapse_to_root_glob
        nonlocal current_has_quoted_content
        nonlocal current_unquoted_brace_open
        nonlocal current_unquoted_brace_close
        nonlocal current_unquoted_brace_comma
        nonlocal source_word_has_content
        start_word()
        source_word_has_content = True
        current.append(piece)
        current_glob_projection.append(
            piece
            if unquoted
            else piece.replace("*", _QUOTED_GLOB_SENTINEL).replace("?", _QUOTED_GLOB_SENTINEL)
        )
        current_has_quoted_content = current_has_quoted_content or quoted
        if unquoted_brace_syntax:
            current_unquoted_brace_open = current_unquoted_brace_open or "{" in piece
            current_unquoted_brace_close = current_unquoted_brace_close or "}" in piece
            current_unquoted_brace_comma = current_unquoted_brace_comma or "," in piece
        current_has_unquoted_star = current_has_unquoted_star or (unquoted and "*" in piece)
        if not dynamic_can_be_empty and piece:
            current_can_collapse_to_root_glob = current_can_collapse_to_root_glob and (
                unquoted and all(character in "*?" for character in piece)
            )

    def flush(*, complete: bool = True) -> None:
        nonlocal word_started
        nonlocal current_has_unquoted_star
        nonlocal current_can_collapse_to_root_glob
        nonlocal current_has_quoted_content
        nonlocal current_unquoted_brace_open
        nonlocal current_unquoted_brace_close
        nonlocal current_unquoted_brace_comma
        nonlocal current_leading_tilde_unquoted
        if word_started:
            if complete and current_is_argument:
                tokens.append(
                    _ShellToken(
                        "".join(current),
                        current_has_unquoted_star and current_can_collapse_to_root_glob,
                        "".join(current_glob_projection),
                        current_has_quoted_content,
                        current_unquoted_brace_open
                        and current_unquoted_brace_close
                        and current_unquoted_brace_comma,
                        current_leading_tilde_unquoted,
                    )
                )
            current.clear()
            current_glob_projection.clear()
            word_started = False
            current_has_unquoted_star = False
            current_can_collapse_to_root_glob = True
            current_has_quoted_content = False
            current_unquoted_brace_open = False
            current_unquoted_brace_close = False
            current_unquoted_brace_comma = False
            current_leading_tilde_unquoted = False

    while cursor < limit:
        character = content[cursor]
        if quote is not None:
            if character == quote:
                quote = None
                ansi_c_quote = False
            elif character == "\\" and ansi_c_quote:
                decoded, cursor = _decode_ansi_c_escape(content, cursor, limit)
                if "\x00" in decoded:
                    closing_quote = _skip_ansi_c_quote_tail(content, cursor, limit)
                    if closing_quote is None:
                        return tuple(tokens), limit, True
                    cursor = closing_quote
                    continue
                append_piece(decoded, quoted=True)
                continue
            elif quote == '"' and character == "$" and cursor + 1 < limit:
                inherited_quote_closed = [False]
                if content[cursor + 1] == "(":
                    substitution_end = _skip_command_substitution(content, cursor, limit)
                    if substitution_end is None:
                        return tuple(tokens), limit, True
                    static_value = _static_printf_substitution(
                        content,
                        cursor,
                        substitution_end,
                    )
                    parse_limited = parse_limited or (
                        static_value is None
                        and _is_printf_substitution(content, cursor, substitution_end)
                    )
                    append_piece(
                        "$DYNAMIC" if static_value is None else static_value,
                        dynamic_can_be_empty=static_value is None,
                        quoted=True,
                    )
                    cursor = substitution_end
                    continue
                parameter_end = _skip_parameter_expansion(
                    content,
                    cursor,
                    limit,
                    inherited_double_quote=True,
                    inherited_quote_closed=inherited_quote_closed,
                )
                if parameter_end is not None:
                    append_piece(
                        "$DYNAMIC",
                        dynamic_can_be_empty=True,
                        quoted=True,
                    )
                    cursor = parameter_end
                    if inherited_quote_closed[0]:
                        quote = None
                        ansi_c_quote = False
                    continue
            elif quote == '"' and character == "`":
                substitution_end = _skip_backtick_substitution(content, cursor, limit)
                if substitution_end is None:
                    return tuple(tokens), limit, True
                static_value = _static_printf_substitution(
                    content,
                    cursor,
                    substitution_end,
                    backtick=True,
                )
                parse_limited = parse_limited or (
                    static_value is None
                    and _is_printf_substitution(
                        content,
                        cursor,
                        substitution_end,
                        backtick=True,
                    )
                )
                append_piece(
                    "$DYNAMIC" if static_value is None else static_value,
                    dynamic_can_be_empty=static_value is None,
                    quoted=True,
                )
                cursor = substitution_end
                continue
            elif character == "\\" and quote == '"' and cursor + 1 < limit:
                escaped = content[cursor + 1]
                if escaped == "\n":
                    cursor += 2
                    continue
                if escaped == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                    cursor += 3
                    continue
                if escaped in '$`"\\':
                    append_piece(escaped, quoted=True)
                else:
                    append_piece("\\" + escaped, quoted=True)
                cursor += 2
                continue
            elif character == "\\" and quote == '"':
                boundary_incomplete = True
                cursor += 1
                continue
            else:
                append_piece(character, quoted=True)
            cursor += 1
            continue
        if wrapper_quote is not None and character == wrapper_quote:
            flush()
            return tuple(tokens), cursor, parse_limited
        if character == "$" and cursor + 1 < limit and content[cursor + 1] in "'\"":
            mark_quoted_word()
            quote = content[cursor + 1]
            ansi_c_quote = quote == "'"
            cursor += 2
            continue
        if character in "'\"":
            mark_quoted_word()
            quote = character
        elif character == "`":
            substitution_end = _skip_backtick_substitution(content, cursor, limit)
            if substitution_end is None:
                return tuple(tokens), limit, True
            static_value = _static_printf_substitution(
                content,
                cursor,
                substitution_end,
                backtick=True,
            )
            parse_limited = parse_limited or (
                static_value is None
                and _is_printf_substitution(
                    content,
                    cursor,
                    substitution_end,
                    backtick=True,
                )
            )
            append_piece(
                "$DYNAMIC" if static_value is None else static_value,
                unquoted=static_value is not None,
                dynamic_can_be_empty=static_value is None,
            )
            cursor = substitution_end
            continue
        elif character == "$" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(content, cursor, limit)
            if substitution_end is None:
                return tuple(tokens), limit, True
            static_value = _static_printf_substitution(content, cursor, substitution_end)
            parse_limited = parse_limited or (
                static_value is None and _is_printf_substitution(content, cursor, substitution_end)
            )
            append_piece(
                "$DYNAMIC" if static_value is None else static_value,
                unquoted=static_value is not None,
                dynamic_can_be_empty=static_value is None,
            )
            cursor = substitution_end
            continue
        elif character == "$":
            parameter_end = _skip_parameter_expansion(content, cursor, limit)
            if parameter_end is not None:
                if _is_ifs_expansion(content, cursor, parameter_end):
                    source_word_has_content = True
                    flush()
                else:
                    append_piece("$DYNAMIC", dynamic_can_be_empty=True)
                cursor = parameter_end
                continue
            if cursor + 1 == limit and limit < len(content) and content[limit] == "(":
                boundary_incomplete = True
        elif character in "<>" and cursor + 1 < limit and content[cursor + 1] == "(":
            substitution_end = _skip_command_substitution(content, cursor, limit)
            if substitution_end is None:
                return tuple(tokens), limit, True
            append_piece(character + "$DYNAMIC", dynamic_can_be_empty=True)
            cursor = substitution_end
            continue
        elif character == "\\":
            if cursor + 1 >= limit:
                boundary_incomplete = True
                cursor += 1
                continue
            if content[cursor + 1] == "\n":
                cursor += 2
                continue
            if content[cursor + 1] == "\r" and cursor + 2 < limit and content[cursor + 2] == "\n":
                cursor += 3
                continue
            append_piece(content[cursor + 1])
            cursor += 1
        elif character == "\n":
            flush()
            return tuple(tokens), cursor, parse_limited
        elif character.isspace():
            flush()
            source_word_has_content = False
        elif character == "#" and not word_started:
            return tuple(tokens), cursor, parse_limited
        elif character in "<>":
            flush()
            source_word_has_content = False
            expect_redirection_target = True
            if cursor + 1 < limit and content[cursor + 1] == character:
                cursor += 1
            if cursor + 1 < limit and content[cursor + 1] in "&|":
                cursor += 1
        elif (
            character == "&"
            and cursor + 1 == limit
            and limit < len(content)
            and content[limit] == ">"
        ):
            boundary_incomplete = True
            cursor += 1
            continue
        elif character == "&" and cursor + 1 < limit and content[cursor + 1] == ">":
            flush()
            source_word_has_content = False
            expect_redirection_target = True
            cursor += 1
            if cursor + 1 < limit and content[cursor + 1] == ">":
                cursor += 1
        elif character in ";|&()":
            flush()
            return tuple(tokens), cursor, parse_limited
        else:
            if not source_word_has_content and character == "~":
                current_leading_tilde_unquoted = True
            append_piece(
                character,
                unquoted=character in "*?",
                unquoted_brace_syntax=character in "{},",
            )
        cursor += 1
    neutral_state = quote is None and not boundary_incomplete and not expect_redirection_target
    known_terminator = (
        limit < len(content)
        and neutral_state
        and (
            content[limit] in "\n;|&()"
            or content[limit] == wrapper_quote
            or (content[limit] == "#" and not word_started)
        )
    )
    complete = (
        limit == len(content) and neutral_state and wrapper_quote is None
    ) or known_terminator
    flush(complete=complete)
    return tuple(tokens), limit, not complete or parse_limited


def _is_root_glob_documentation(
    content: str,
    command_start: int,
    body_start: int,
) -> bool:
    """Return whether ``rm`` is the noun in a bounded prose description."""
    context_start = max(0, command_start - 256)
    previous_boundary = max(
        content.rfind("\n", context_start, command_start),
        content.rfind(";", context_start, command_start),
    )
    prefix_truncated = previous_boundary < 0 and context_start > 0
    clause_start = previous_boundary + 1 if previous_boundary >= 0 else context_start
    context_end = min(len(content), command_start + 512)
    ends = [
        boundary
        for boundary in (
            content.find("\n", command_start, context_end),
            content.find(";", command_start, context_end),
        )
        if boundary >= 0
    ]
    clause_end = min(ends, default=context_end)
    clause = content[clause_start:clause_end]
    if _ROOT_GLOB_DOCUMENTATION_LINE_RE.fullmatch(clause) is not None:
        return True

    relative_start = command_start - clause_start
    prefix = clause[:relative_start]
    suffix = clause[max(relative_start, body_start - clause_start) :]
    if not prefix.strip() or prefix_truncated:
        return False
    execution = _ROOT_GLOB_EXECUTION_PREFIX_RE.search(prefix)
    negated_execution = False
    if execution is not None:
        negation_prefix = prefix[: execution.start("action")]
        if _ROOT_GLOB_AFFIRMATIVE_NEGATION_PREFIX_RE.fullmatch(negation_prefix) is not None:
            return False
        if _ROOT_GLOB_NEGATED_EXECUTION_RE.search(negation_prefix) is None:
            return False
        negated_execution = True
    if _ROOT_GLOB_SAFETY_WARNING_PREFIX_RE.fullmatch(prefix) is not None:
        return True
    if (
        re.search(
            r"\b(?:sudo|env|xargs|command)\b[^.;\n]{0,64}$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    ):
        return False
    article_prefix = re.fullmatch(
        r"[ \t]*(?:the|this|a|an)(?:[ \t]+(?:gnu|posix|unix))?[ \t]*",
        prefix,
        re.IGNORECASE,
    )
    documentary_prefix = re.search(
        r"(?:\b(?:explain|describe|document|note|state|say)(?:[ \t]+that)?|"
        r"(?:documentation|docs?|guide|manual|reference|note|example)[ \t]*:)"
        r"[ \t]*$",
        prefix,
        re.IGNORECASE,
    )
    documentary_leadin = re.fullmatch(
        r"[ \t]*(?:(?:in|under)[ \t]+(?:the[ \t]+)?(?:posix(?:\.[0-9]+)?|gnu|unix)"
        r"(?:[ \t]+(?:standard|specification))?|"
        r"according[ \t]+to[ \t]+(?:the[ \t]+)?(?:manual|documentation|reference|"
        r"(?:posix(?:\.[0-9]+)?|gnu|unix)(?:[ \t]+(?:standard|specification))?)|"
        r"for[ \t]+reference|"
        r"this[ \t]+(?:section|guide|documentation)[ \t]+"
        r"(?:explains?|describes?|documents?|notes?|states?|says?)(?:[ \t]+that)?)"
        r"[ \t]*,?[ \t]+(?:(?:the|this|a|an)[ \t]+)?"
        r"(?:(?:gnu|posix(?:\.[0-9]+)?|unix)[ \t]+)?",
        prefix,
        re.IGNORECASE,
    )
    if (
        article_prefix is None
        and documentary_prefix is None
        and documentary_leadin is None
        and not negated_execution
    ):
        return False
    describes_flags = re.search(
        r"\b(?:command|utility|tool)\b|\b(?:accepts?|supports?|uses?|takes?)\b",
        suffix,
        re.IGNORECASE,
    )
    describes_glob = re.search(
        r"\b(?:denotes?|means?|represents?|matches?|wildcards?|globs?)\b|"
        r"\b(?:all|every)[ \t]+files?\b",
        suffix,
        re.IGNORECASE,
    )
    return describes_flags is not None and describes_glob is not None and "*" in suffix


def _static_brace_expansions(word: str) -> tuple[str, ...] | None:
    """Expand a small, literal Bash brace word without executing shell code."""
    if len(word) > _STATIC_BRACE_WORD_CHARS:
        return None
    expanded = [word]
    changed = False
    for _ in range(4):
        next_words: list[str] = []
        expanded_this_round = False
        for candidate in expanded:
            brace = re.search(r"\{(?P<body>[^{}]*)\}", candidate)
            if brace is None or "," not in brace.group("body"):
                next_words.append(candidate)
                continue
            alternatives = brace.group("body").split(",")
            if len(alternatives) > 8:
                return None
            next_words.extend(
                candidate[: brace.start()] + alternative + candidate[brace.end() :]
                for alternative in alternatives
            )
            if len(next_words) > 32:
                return None
            expanded_this_round = True
            changed = True
        expanded = next_words
        if not expanded_this_round:
            break
    if any("{" in candidate or "}" in candidate for candidate in expanded):
        return None
    return tuple(expanded) if changed else None


def _has_recursive_force_options(tokens: tuple[_ShellToken, ...]) -> bool:
    try:
        options_end = next(index for index, token in enumerate(tokens) if token.text == "--")
    except StopIteration:
        options_end = len(tokens)
    option_words: list[str] = []
    for token in tokens[:options_end]:
        if token.brace_expansion:
            expanded = _static_brace_expansions(token.text)
            if expanded is not None:
                option_words.extend(expanded)
                continue
        option_words.append(token.text)
    short_options = [
        word[1:] for word in option_words if re.fullmatch(r"-[A-Za-z]+", word) is not None
    ]
    recursive = "--recursive" in option_words or any(
        "r" in option or "R" in option for option in short_options
    )
    force = "--force" in option_words or any("f" in option for option in short_options)
    return recursive and force


def _has_destructive_root_glob(tokens: tuple[_ShellToken, ...]) -> bool:
    def is_root_glob(token: _ShellToken) -> bool:
        if token.root_glob:
            return True
        if not token.brace_expansion:
            return False
        expanded = _static_brace_expansions(token.glob_projection)
        return expanded is not None and any(
            re.fullmatch(r"[*?]+", word) is not None for word in expanded
        )

    return any(is_root_glob(token) for token in tokens) and _has_recursive_force_options(tokens)


def _has_destructive_root_path(tokens: tuple[_ShellToken, ...]) -> bool:
    """Return whether recursive-force options target root or home expansion."""
    has_root_path = any(
        token.text.startswith("/") or token.text.startswith("~") and token.leading_tilde_unquoted
        for token in tokens
    )
    return has_root_path and _has_recursive_force_options(tokens)


def _has_unsupported_brace_expansion(tokens: tuple[_ShellToken, ...]) -> bool:
    return any(
        token.brace_expansion and _static_brace_expansions(token.text) is None for token in tokens
    )


def _tm1_candidates(
    content: str,
) -> Iterator[tuple[int, int, str, float]]:
    for pattern, confidence in TM1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            yield match.start(), match.end(), match.group(0), confidence

    seen_commands: set[tuple[int, int]] = set()
    covered_until = 0
    for command_start, body_start in _destructive_command_words(content):
        if command_start < covered_until:
            continue
        command_key = (command_start, body_start)
        if command_key in seen_commands:
            continue
        seen_commands.add(command_key)
        # Documentation is excluded regardless of the shell parse result.  Apply
        # that existing semantic gate first so large manuals do not pay for a
        # character-by-character shell parse for every explanatory ``rm`` noun.
        if _is_root_glob_documentation(content, command_start, body_start):
            continue
        tokens, command_end, _ = _bounded_shell_tokens(
            content,
            command_start,
            body_start,
        )
        covered_until = max(covered_until, command_end)
        command = content[command_start:command_end]
        if _has_destructive_root_glob(tokens) or _has_destructive_root_path(tokens):
            yield command_start, command_end, command, 0.9


def has_bounded_parse_exhaustion(
    content: str,
    check_runtime: Callable[[], None],
) -> bool:
    """Return whether a destructive rm command exceeded the parser's span contract."""
    if _has_shell_command_word_exhaustion(content, check_runtime):
        return True
    covered_until = 0
    for command_start, body_start in _destructive_command_words(content):
        check_runtime()
        if command_start < covered_until:
            continue
        # A recognized prose description cannot contribute parse exhaustion:
        # the same predicate below would discard it after parsing.  Short-circuit
        # it here to keep bounded-runtime accounting linear for documentation.
        if _is_root_glob_documentation(content, command_start, body_start):
            continue
        tokens, command_end, exhausted = _bounded_shell_tokens(
            content,
            command_start,
            body_start,
        )
        covered_until = max(covered_until, command_end)
        if exhausted or _has_unsupported_brace_expansion(tokens):
            return True
    return False


def _line_containing(content: str, start: int, end: int) -> str:
    """Return the full line containing a regex match."""
    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", end)
    if line_end == -1:
        line_end = len(content)
    return content[line_start:line_end]


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for tool misuse patterns (TM1–TM3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.TOOL_MISUSE.value]
    tm1_findings_by_key: dict[tuple[int, str], AnalyzerFinding] = {}

    for match_start, match_end, matched_text, confidence in _tm1_candidates(content):
        line_num = get_line_number(content, match_start)
        context_text = ctx(match_start)
        matched = matched_text[:200]
        matched_line = _line_containing(content, match_start, match_end)

        if (
            _is_safe_container_command(context_text)
            or _is_safe_dockerfile_idiom(context_text, matched)
            or _is_safe_cache_cleanup(matched_line)
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
        candidate_key = (line_num, " ".join(matched.strip().split()))
        existing = tm1_findings_by_key.get(candidate_key)
        if existing is not None:
            if adj > existing.confidence:
                existing.confidence = adj
                existing.severity = sev
            continue
        finding = AnalyzerFinding(
            rule_id="TM1",
            message="Tool Parameter Abuse",
            severity=sev,
            location=loc(line_num),
            confidence=adj,
            tags=tag,
            context=context_text,
            matched_text=matched,
            evidence={static_runner._VIEW_START_EVIDENCE: match_start},
        )
        tm1_findings_by_key[candidate_key] = finding
        findings.append(finding)
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
    # TM4: privileged K8s workload. Example filtering is delegated to the runner.
    for pattern, confidence in TM4_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="TM4",
                    message="Privileged Kubernetes Workload",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run tool_misuse patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
