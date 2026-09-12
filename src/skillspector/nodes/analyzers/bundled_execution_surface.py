# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze the bounded bundled hook and permission execution surfaces."""

from __future__ import annotations

import ipaddress
import json
import math
import posixpath
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import regex  # type: ignore[import-untyped]
from pywhatwgurl import URL

from skillspector.inspection_ledger import (
    InspectionLedgerEvent,
    LedgerOutcome,
    LedgerReason,
    analyzer_status_for_events,
    ledger_event,
)
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Finding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from .static_runner import MAX_FILE_CHARS, analyzer_finding_to_finding

ANALYZER_ID = "bundled_execution_surface"
logger = get_logger(__name__)

_APPLICABLE_PATHS: Final = frozenset(
    {
        "hooks/hooks.json",
        ".claude/settings.json",
        ".claude/settings.local.json",
    }
)
_MAX_DECLARATIONS: Final = 2_048
_MAX_DECLARATION_CHARS: Final = 16_384
_VALID_DEFAULT_MODES: Final = frozenset(
    {"acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "manual", "plan"}
)
_LIMITED_BROADCAST: Final = ipaddress.IPv4Address("255.255.255.255")
_BIDI_RTL_TRIGGER: Final = regex.compile(r"\A[\p{bc=R}\p{bc=AL}\p{bc=AN}]\Z")
_BIDI_RTL_FIRST: Final = regex.compile(r"\A[\p{bc=R}\p{bc=AL}]\Z")
_BIDI_LTR_FIRST: Final = regex.compile(r"\A\p{bc=L}\Z")
_BIDI_RTL_ALLOWED: Final = regex.compile(
    r"\A[\p{bc=R}\p{bc=AL}\p{bc=AN}\p{bc=EN}\p{bc=ES}\p{bc=CS}"
    r"\p{bc=ET}\p{bc=ON}\p{bc=BN}\p{bc=NSM}]\Z"
)
_BIDI_LTR_ALLOWED: Final = regex.compile(
    r"\A[\p{bc=L}\p{bc=EN}\p{bc=ES}\p{bc=CS}\p{bc=ET}"
    r"\p{bc=ON}\p{bc=BN}\p{bc=NSM}]\Z"
)
_BIDI_RTL_END: Final = regex.compile(r"\A[\p{bc=R}\p{bc=AL}\p{bc=EN}\p{bc=AN}]\Z")
_BIDI_LTR_END: Final = regex.compile(r"\A[\p{bc=L}\p{bc=EN}]\Z")
_BIDI_NSM: Final = regex.compile(r"\A\p{bc=NSM}\Z")
_BIDI_AN: Final = regex.compile(r"\A\p{bc=AN}\Z")
_BIDI_EN: Final = regex.compile(r"\A\p{bc=EN}\Z")
_KNOWN_HANDLER_TYPES: Final = frozenset({"command", "http", "mcp_tool", "prompt", "agent"})
_KNOWN_EVENTS: Final = frozenset(
    {
        "ConfigChange",
        "CwdChanged",
        "DirectoryAdded",
        "Elicitation",
        "ElicitationResult",
        "FileChanged",
        "InstructionsLoaded",
        "MessageDisplay",
        "Notification",
        "PermissionDenied",
        "PermissionRequest",
        "PostCompact",
        "PostToolBatch",
        "PostToolUse",
        "PostToolUseFailure",
        "PreCompact",
        "PreToolUse",
        "SessionEnd",
        "SessionStart",
        "Setup",
        "Stop",
        "StopFailure",
        "SubagentStart",
        "SubagentStop",
        "TaskCompleted",
        "TaskCreated",
        "TeammateIdle",
        "UserPromptExpansion",
        "UserPromptSubmit",
        "WorktreeCreate",
        "WorktreeRemove",
    }
)
_NO_MATCHER_EVENTS: Final = frozenset(
    {
        "CwdChanged",
        "MessageDisplay",
        "PostToolBatch",
        "Stop",
        "TaskCompleted",
        "TaskCreated",
        "TeammateIdle",
        "UserPromptSubmit",
        "WorktreeCreate",
        "WorktreeRemove",
    }
)
_PROMPT_AGENT_EVENTS: Final = frozenset(
    {
        "PermissionDenied",
        "PermissionRequest",
        "PostToolBatch",
        "PostToolUse",
        "PostToolUseFailure",
        "PreToolUse",
        "Stop",
        "SubagentStop",
        "TaskCompleted",
        "TaskCreated",
        "TeammateIdle",
        "UserPromptExpansion",
        "UserPromptSubmit",
    }
)
_COMMAND_MCP_ONLY_EVENTS: Final = frozenset({"SessionStart", "Setup"})
_IF_EVENTS: Final = frozenset(
    {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "PermissionDenied",
    }
)
_SENSITIVE_EVENTS: Final = frozenset(
    {
        "UserPromptSubmit",
        "UserPromptExpansion",
        "MessageDisplay",
        "PreToolUse",
        "PermissionRequest",
        "PermissionDenied",
        "PostToolUse",
        "PostToolUseFailure",
        "PostToolBatch",
        "SubagentStop",
        "TaskCreated",
        "TaskCompleted",
        "Stop",
        "StopFailure",
        "PreCompact",
        "PostCompact",
        "Elicitation",
        "ElicitationResult",
    }
)
_SENSITIVE_DIRECTORY_SUFFIXES: Final = (
    "/.ssh",
    "/.aws",
    "/.kube",
    "/.config/gcloud",
)
_SENSITIVE_FILE_SUFFIXES: Final = frozenset(
    {
        "/.claude/settings.json",
        "/.claude/settings.local.json",
        "/.claude/.credentials.json",
        "/.docker/config.json",
        "/.netrc",
        "/.npmrc",
    }
)
_ABSOLUTE_HOME_PATH = re.compile(
    r"^/(?:Users/(?!\.{1,2}/)[^/]+|home/(?!\.{1,2}/)[^/]+|root)(?P<suffix>/.*)$"
)
_REMOTE_PATH = re.compile(r"^(?:[A-Za-z0-9._-]+@)?(?P<host>[A-Za-z0-9.-]+):(?P<path>[^\s]+)$")
_BRACKETED_REMOTE_PATH = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?\[(?P<host>[0-9A-Fa-f:.]+)\]:(?P<path>[^\s]+)$"
)
_CURL_HTTP_URL = re.compile(r"(?i)\Ahttps?:/{1,3}[^/]", re.ASCII)
_WGET_HTTP_URL = re.compile(r"(?i)\Ahttps?://[^/]", re.ASCII)
_DISABLE_TRUSTED_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "$schema",
        "disableAllHooks",
        "env",
        "hooks",
        "includeCoAuthoredBy",
        "model",
        "permissions",
    }
)
_HookIdentity = tuple[str, str, str]


class _DuplicateKeyError(ValueError):
    """Raised when untrusted JSON contains an ambiguous duplicate key."""


class _NonFiniteConstantError(ValueError):
    """Raised when JSON uses a non-standard NaN or Infinity constant."""


@dataclass(frozen=True)
class _HookDeclaration:
    event: str
    handler_type: str
    ambient: bool
    matcher_breadth: str
    remote_http: bool
    handler: dict[str, object]
    active: bool = True


@dataclass(frozen=True)
class _Bh2Proof:
    kind: str
    transport: str


@dataclass(frozen=True)
class _PermissionDeclaration:
    severity: Severity
    kind: str
    activation_state: str = "conditional"


@dataclass(frozen=True)
class _DeclarationScan:
    hooks: list[_HookDeclaration]
    permissions: list[_PermissionDeclaration]
    partial: bool
    observed: int


def _address_is_non_remote(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address == _LIMITED_BROADCAST
    )


def _is_non_remote_host(host: str) -> bool:
    normalized = host.lower().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.inet_aton(normalized))
        except (OSError, ValueError):
            return False
    return _address_is_non_remote(address)


def _punycode_labels_are_valid(host: str) -> bool:
    for label in host.split("."):
        if not label.startswith("xn--"):
            continue
        try:
            decoded = label.encode("ascii").decode("idna")
            if decoded.encode("idna").decode("ascii") != label:
                return False
        except UnicodeError:
            return False
    return True


def _is_valid_literal_host(host: str) -> bool:
    if host.endswith(".."):
        return False
    normalized = host.lower().removesuffix(".")
    if not normalized or not normalized.isascii() or "%" in normalized:
        return False
    try:
        ipaddress.ip_address(normalized)
        return True
    except ValueError:
        pass
    try:
        socket.inet_aton(normalized)
        return True
    except (OSError, ValueError):
        pass
    if normalized.startswith("0x") or re.fullmatch(r"[0-9.]+", normalized):
        return False
    labels = normalized.split(".")
    if not _punycode_labels_are_valid(normalized):
        return False
    return all(
        1 <= len(label) <= 63
        and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) is not None
        for label in labels
    )


def _bracketed_url_host_is_ipv6(value: str, host: str) -> bool:
    parts = value.split("://", 1)
    if len(parts) != 2:
        return True
    authority = re.split(r"[/?#]", parts[1], maxsplit=1)[0]
    if not authority.rsplit("@", 1)[-1].startswith("["):
        return True
    try:
        ipaddress.IPv6Address(host)
    except ValueError:
        return False
    return True


def _is_safe_literal(value: str) -> bool:
    return (
        len(value) <= _MAX_DECLARATION_CHARS
        and "\x00" not in value
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _is_bounded_http_string(value: object) -> bool:
    return isinstance(value, str) and len(value) <= _MAX_DECLARATION_CHARS


def _is_bounded_string(value: object) -> bool:
    return isinstance(value, str) and _is_safe_literal(value)


def _is_nonempty_bounded_string(value: object) -> bool:
    return _is_bounded_string(value) and bool(value)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_positive_json_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_bounded_string_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_bounded_string(item) for item in value)


def _is_bounded_header_value(value: object) -> bool:
    return _is_bounded_http_string(value)


def _is_bounded_header_map(value: object) -> bool:
    return isinstance(value, dict) and all(
        _is_bounded_string(key) and _is_bounded_header_value(item) for key, item in value.items()
    )


def _is_bounded_string_map(value: object) -> bool:
    return isinstance(value, dict) and all(
        _is_bounded_string(key) and _is_bounded_string(item) for key, item in value.items()
    )


def _optional_field_is_valid(
    handler: dict[str, object], field: str, validator: Callable[[object], bool]
) -> bool:
    return field not in handler or validator(handler.get(field))


def _label_satisfies_bidi_rule(label: str) -> bool:
    if not label:
        return False
    rtl = _BIDI_RTL_FIRST.fullmatch(label[0]) is not None
    if not rtl and _BIDI_LTR_FIRST.fullmatch(label[0]) is None:
        return False
    allowed = _BIDI_RTL_ALLOWED if rtl else _BIDI_LTR_ALLOWED
    if not all(allowed.fullmatch(character) is not None for character in label):
        return False
    ending = next(
        (character for character in reversed(label) if _BIDI_NSM.fullmatch(character) is None),
        "",
    )
    valid_end = _BIDI_RTL_END if rtl else _BIDI_LTR_END
    if valid_end.fullmatch(ending) is None:
        return False
    if rtl:
        has_an = any(_BIDI_AN.fullmatch(character) is not None for character in label)
        has_en = any(_BIDI_EN.fullmatch(character) is not None for character in label)
        return not (has_an and has_en)
    return True


def _hostname_satisfies_bidi_rule(hostname: str) -> bool:
    """Apply the domain-wide WHATWG BiDi check omitted by pywhatwgurl 0.1.1."""
    if "xn--" not in hostname:
        return True

    labels: list[str] = []
    try:
        for label in hostname.rstrip(".").split("."):
            labels.append(
                label.removeprefix("xn--").encode("ascii").decode("punycode")
                if label.startswith("xn--")
                else label
            )
    except UnicodeError:
        return False

    if not any(
        _BIDI_RTL_TRIGGER.fullmatch(character) is not None
        for label in labels
        for character in label
    ):
        return True
    return all(_label_satisfies_bidi_rule(label) for label in labels if label)


def _parse_schema_url(value: object) -> URL | None:
    if not _is_bounded_http_string(value):
        return None
    assert isinstance(value, str)
    value = value.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
    try:
        return URL(value)
    except (UnicodeError, ValueError):
        return None


def _parse_http_url(value: object) -> URL | None:
    parsed = _parse_schema_url(value)
    return (
        parsed
        if parsed is not None
        and parsed.protocol in {"http:", "https:"}
        and parsed.hostname
        and "$" not in parsed.hostname
        and _hostname_satisfies_bidi_rule(parsed.hostname)
        else None
    )


def _handler_url_is_valid(value: object) -> bool:
    parsed = _parse_schema_url(value)
    if parsed is None:
        return False
    if parsed.protocol not in {"http:", "https:"}:
        return True
    return bool(parsed.hostname and _hostname_satisfies_bidi_rule(parsed.hostname))


def _handler_schema_url_is_valid(value: object) -> bool:
    return _parse_schema_url(value) is not None


def _is_external_http_url(value: object) -> bool:
    if not isinstance(value, str) or not _is_safe_literal(value):
        return False
    if "\\" in value or any(character.isspace() or ord(character) < 0x20 for character in value):
        return False
    parsed = _parse_http_url(value)
    return parsed is not None and not _is_non_remote_host(parsed.hostname)


def _http_headers_are_sendable(headers: dict[str, object]) -> bool:
    for name in headers:
        if re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", name) is None:
            return False
    return True


def _is_remote_http(handler: dict[str, object]) -> bool:
    headers = handler.get("headers", {})
    assert isinstance(headers, dict)
    parsed = _parse_http_url(handler.get("url"))
    return (
        parsed is not None
        and not _is_non_remote_host(parsed.hostname)
        and _http_headers_are_sendable(headers)
    )


def _matcher_breadth(event: str, matcher: str | None) -> str:
    if event not in _KNOWN_EVENTS:
        return "unsupported"
    if event in _NO_MATCHER_EVENTS:
        return "not_applicable"
    if matcher is None or matcher in {"", "*"}:
        return "all"
    if event == "FileChanged":
        segments = matcher.split("|")
        literal_pattern = r"[A-Za-z0-9_./:-]+"
    elif event == "StopFailure":
        segments = matcher.split("|")
        literal_pattern = r"[A-Za-z0-9_]+"
    else:
        segments = re.split(r"\s*[|,]\s*", matcher.strip())
        literal_pattern = r"[A-Za-z0-9_-]+"
    if all(segment and re.fullmatch(literal_pattern, segment) for segment in segments):
        return "scoped"
    return "unsupported"


def _handler_strings_are_bounded(handler: dict[str, object]) -> bool:
    for key in ("type", "command", "url", "prompt", "server", "tool", "shell", "if"):
        value = handler.get(key)
        if isinstance(value, str) and not (
            _is_bounded_http_string(value)
            if key == "url" and handler.get("type") == "http"
            else _is_safe_literal(value)
        ):
            return False
    if handler.get("type") != "command" or "args" not in handler:
        return True
    args = handler.get("args")
    return isinstance(args, list) and all(
        isinstance(arg, str) and _is_safe_literal(arg) for arg in args
    )


def _handler_optional_fields_are_valid(handler_type: str, handler: dict[str, object]) -> bool:
    if handler_type not in _KNOWN_HANDLER_TYPES:
        return True
    common = (
        _optional_field_is_valid(handler, "if", _is_bounded_string)
        and _optional_field_is_valid(handler, "timeout", _is_positive_json_number)
        and _optional_field_is_valid(handler, "statusMessage", _is_bounded_string)
        and _optional_field_is_valid(handler, "once", _is_bool)
    )
    if not common:
        return False
    if handler_type == "command":
        return (
            _optional_field_is_valid(handler, "args", _is_bounded_string_list)
            and _optional_field_is_valid(handler, "async", _is_bool)
            and _optional_field_is_valid(handler, "asyncRewake", _is_bool)
            and _optional_field_is_valid(handler, "rewakeMessage", _is_nonempty_bounded_string)
            and _optional_field_is_valid(handler, "rewakeSummary", _is_nonempty_bounded_string)
        )
    if handler_type == "http":
        return _optional_field_is_valid(
            handler, "headers", _is_bounded_header_map
        ) and _optional_field_is_valid(handler, "allowedEnvVars", _is_bounded_string_list)
    if handler_type == "prompt":
        return _optional_field_is_valid(
            handler, "model", _is_bounded_string
        ) and _optional_field_is_valid(handler, "continueOnBlock", _is_bool)
    if handler_type == "agent":
        return _optional_field_is_valid(handler, "model", _is_bounded_string)
    if handler_type == "mcp_tool":
        return _optional_field_is_valid(handler, "input", lambda value: isinstance(value, dict))
    return True


def _handler_shape_is_valid(
    handler_type: str,
    handler: dict[str, object],
    *,
    url_validator: Callable[[object], bool] = _handler_url_is_valid,
) -> bool:
    required_fields = {
        "command": ("command",),
        "http": ("url",),
        "prompt": ("prompt",),
        "agent": ("prompt",),
        "mcp_tool": ("server", "tool"),
    }.get(handler_type, ())
    if not handler_type:
        return False
    if handler_type == "command" and "shell" in handler:
        shell = handler.get("shell")
        if not isinstance(shell, str) or shell not in {"bash", "powershell"}:
            return False
    for field in required_fields:
        value = handler.get(field)
        valid_string = (
            _is_bounded_http_string(value)
            if handler_type == "http" and field == "url"
            else isinstance(value, str) and _is_safe_literal(value)
        )
        if not valid_string:
            return False
    if handler_type == "http" and not url_validator(handler.get("url")):
        return False
    return _handler_optional_fields_are_valid(handler_type, handler)


def _handler_type_is_supported(event: str, handler_type: str) -> bool:
    if event not in _KNOWN_EVENTS or handler_type not in _KNOWN_HANDLER_TYPES:
        return True
    if event in _COMMAND_MCP_ONLY_EVENTS:
        return handler_type in {"command", "mcp_tool"}
    if handler_type in {"prompt", "agent"}:
        return event in _PROMPT_AGENT_EVENTS
    return True


def _hook_declaration(
    event: str, matcher_breadth: str, raw_handler: object
) -> _HookDeclaration | None:
    if not isinstance(raw_handler, dict) or not _handler_strings_are_bounded(raw_handler):
        return None
    raw_type = raw_handler.get("type")
    if not isinstance(raw_type, str):
        return None
    if not _handler_shape_is_valid(raw_type, raw_handler):
        return None
    if not _handler_type_is_supported(event, raw_type):
        return None
    condition = raw_handler.get("if")
    if "if" in raw_handler and (
        not isinstance(condition, str) or not _permission_rule_is_valid(condition)
    ):
        return None
    return _HookDeclaration(
        event=event,
        handler_type=raw_type,
        ambient=matcher_breadth != "scoped",
        matcher_breadth=matcher_breadth,
        remote_http=raw_type == "http" and _is_remote_http(raw_handler),
        handler=raw_handler,
        active=not ("if" in raw_handler and event in _KNOWN_EVENTS and event not in _IF_EVENTS),
    )


def _hook_group_declarations(
    event: str,
    raw_group: object,
    *,
    invalid_event: bool,
    limit: int,
    previous_handler_ids: set[_HookIdentity] | None,
    current_handler_ids: set[_HookIdentity],
) -> tuple[list[_HookDeclaration], bool, int]:
    if not isinstance(raw_group, dict):
        return [], True, 0
    raw_handlers = raw_group.get("hooks")
    if not isinstance(raw_handlers, list):
        return [], True, 0
    matcher = raw_group.get("matcher")
    invalid_matcher = (isinstance(matcher, str) and not _is_safe_literal(matcher)) or (
        "matcher" in raw_group and not isinstance(matcher, str)
    )
    if invalid_event or invalid_matcher:
        return [], True, len(raw_handlers)

    assert matcher is None or isinstance(matcher, str)
    matcher_breadth = _matcher_breadth(event, matcher)
    declarations: list[_HookDeclaration] = []
    partial = False
    observed = 0
    for raw_handler in raw_handlers:
        observed += 1
        if observed > limit:
            return declarations, True, observed
        declaration = _hook_declaration(event, matcher_breadth, raw_handler)
        if declaration is None:
            partial = True
        elif declaration.active:
            identity = (
                event,
                matcher_breadth,
                json.dumps(raw_handler, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            )
            current_handler_ids.add(identity)
            if previous_handler_ids is None or identity not in previous_handler_ids:
                declarations.append(declaration)
    return declarations, partial, observed


def _remember_settings_handlers(
    previous_handler_ids: set[_HookIdentity] | None,
    current_handler_ids: set[_HookIdentity],
) -> None:
    if previous_handler_ids is not None:
        previous_handler_ids.update(current_handler_ids)


def _hook_declarations(
    document: object,
    *,
    limit: int,
    previous_handler_ids: set[_HookIdentity] | None = None,
) -> tuple[list[_HookDeclaration], bool, int]:
    if not isinstance(document, dict):
        return [], False, 0
    if "hooks" not in document:
        return [], False, 0
    hooks = document.get("hooks")
    if not isinstance(hooks, dict):
        return [], True, 0

    declarations: list[_HookDeclaration] = []
    current_handler_ids: set[_HookIdentity] = set()
    partial = False
    observed = 0
    for raw_event, raw_groups in hooks.items():
        if not isinstance(raw_event, str):
            partial = True
            continue
        invalid_event = not _is_safe_literal(raw_event)
        if invalid_event:
            partial = True
        if not isinstance(raw_groups, list):
            partial = True
            continue
        for raw_group in raw_groups:
            group_declarations, group_partial, group_observed = _hook_group_declarations(
                raw_event,
                raw_group,
                invalid_event=invalid_event,
                limit=max(0, limit - observed),
                previous_handler_ids=previous_handler_ids,
                current_handler_ids=current_handler_ids,
            )
            declarations.extend(group_declarations)
            partial = partial or group_partial
            observed += group_observed
            if observed > limit:
                _remember_settings_handlers(previous_handler_ids, current_handler_ids)
                return declarations, True, observed
    _remember_settings_handlers(previous_handler_ids, current_handler_ids)
    return declarations, partial, observed


def _sensitive_suffix(suffix: str) -> bool:
    if not _is_safe_literal(suffix) or not suffix.startswith("/"):
        return False
    segments = suffix[1:].split("/")
    if segments[-1:] == [""]:
        segments.pop()
    if not segments or any(segment in {"", ".", ".."} for segment in segments):
        return False
    if suffix in _SENSITIVE_FILE_SUFFIXES:
        return True
    return any(
        suffix == directory or suffix.startswith(f"{directory}/")
        for directory in _SENSITIVE_DIRECTORY_SUFFIXES
    )


def _sensitive_file_suffix(suffix: str) -> bool:
    return (
        not suffix.endswith("/")
        and suffix not in _SENSITIVE_DIRECTORY_SUFFIXES
        and _sensitive_suffix(suffix)
    )


def _is_sensitive_absolute_path(value: object) -> bool:
    if not isinstance(value, str) or not _is_safe_literal(value):
        return False
    match = _ABSOLUTE_HOME_PATH.fullmatch(value)
    return bool(match and _sensitive_file_suffix(match.group("suffix")))


def _is_sensitive_shell_path(value: str) -> bool:
    if not _is_safe_literal(value):
        return False
    for anchor in ("$HOME", "${HOME}"):
        if value.startswith(f"{anchor}/"):
            suffix = value[len(anchor) :]
            return not any(
                character in suffix for character in "\\$*?[]{}!"
            ) and _sensitive_file_suffix(suffix)
    return False


def _is_sensitive_tilde_path(value: str) -> bool:
    return (
        _is_safe_literal(value)
        and value.startswith("~/")
        and value != "~/.claude/settings.local.json"
        and _sensitive_suffix(value[1:])
    )


def _is_remote_uri_destination(value: str, transport: str) -> bool:
    if not value.startswith(f"{transport}://"):
        return False
    authority = re.split(r"[/?#]", value.split("://", 1)[1], maxsplit=1)[0]
    if transport == "scp" and authority.rsplit("@", 1)[-1].endswith(":"):
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return False
    if transport == "scp":
        username = parsed.username
        if (
            username is not None and re.fullmatch(r"(?!-)[A-Za-z0-9._-]+", username) is None
        ) or "%" in parsed.path:
            return False
    if transport == "rsync" and parsed.path.startswith("//"):
        return False
    if not _bracketed_url_host_is_ipv6(value, host):
        return False
    return (
        _is_valid_literal_host(host)
        and parsed.path not in {"", "/"}
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and port != 0
        and not _is_non_remote_host(host)
    )


def _is_remote_destination(value: object, transport: str) -> bool:
    if not isinstance(value, str) or not _is_safe_literal(value):
        return False
    if (
        value.startswith("-")
        or re.match(r"^[A-Za-z]:", value)
        or "\\" in value
        or any(character.isspace() or ord(character) < 0x20 for character in value)
    ):
        return False
    if "://" in value:
        return _is_remote_uri_destination(value, transport)
    match = _BRACKETED_REMOTE_PATH.fullmatch(value) or _REMOTE_PATH.fullmatch(value)
    if not match:
        return False
    host = match.group("host")
    remote_path = match.group("path")
    rsync_module = remote_path[1:].split("/", 1)[0] if remote_path.startswith(":") else None
    return (
        (transport != "rsync" or rsync_module is None or bool(rsync_module))
        and _is_valid_literal_host(host)
        and not _is_non_remote_host(host)
    )


def _literal_args(handler: dict[str, object]) -> list[str] | None:
    args = handler.get("args")
    if not isinstance(args, list) or not all(
        isinstance(arg, str) and _is_safe_literal(arg) for arg in args
    ):
        return None
    return args


def _flag_source(args: list[str], flags: frozenset[str]) -> tuple[str, list[str]] | None:
    matches: list[tuple[int, str]] = []
    consumed: set[int] = set()
    for index, arg in enumerate(args):
        if arg in flags:
            if index + 1 >= len(args):
                return None
            matches.append((index, args[index + 1]))
            consumed.update({index, index + 1})
            continue
        for flag in flags:
            if flag.startswith("--") and arg.startswith(f"{flag}="):
                matches.append((index, arg[len(flag) + 1 :]))
                consumed.add(index)
                break
    if len(matches) != 1:
        return None
    remaining = [arg for index, arg in enumerate(args) if index not in consumed]
    return matches[0][1], remaining


def _one_external_url(values: list[str]) -> bool:
    return (
        len(values) == 1
        and _WGET_HTTP_URL.match(values[0]) is not None
        and _is_external_http_url(values[0])
    )


def _curl_url_has_unsupported_glob(value: str) -> bool:
    if any(character in value for character in "{}"):
        return True
    if "[" not in value and "]" not in value:
        return False
    if _CURL_HTTP_URL.match(value) is None:
        return True
    parsed = _parse_http_url(value)
    authority = re.split(r"[/?#]", value.split(":", 1)[1].lstrip("/"), maxsplit=1)[0]
    return not (
        value.count("[") == value.count("]") == 1
        and authority.count("[") == authority.count("]") == 1
        and parsed is not None
        and ":" in parsed.hostname
    )


def _one_external_curl_url(values: list[str]) -> bool:
    parsed = _parse_http_url(values[0]) if len(values) == 1 else None
    return (
        parsed is not None
        and _CURL_HTTP_URL.match(values[0]) is not None
        and not _curl_url_has_unsupported_glob(values[0])
        and not any(character in parsed.hostname for character in "!$&'()*+,;=")
        and _is_external_http_url(values[0])
    )


def _without_curl_transport_flags(args: list[str]) -> list[str] | None:
    remaining: list[str] = []
    saw_silent = False
    saw_request = False
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-s":
            if saw_silent:
                return None
            saw_silent = True
            index += 1
            continue
        if arg == "-X":
            if saw_request or args[index + 1 : index + 2] != ["POST"]:
                return None
            saw_request = True
            index += 2
            continue
        remaining.append(arg)
        index += 1
    return remaining


def _curl_exec_proof(event: str, args: list[str]) -> bool:
    data = _flag_source(args, frozenset({"-d", "--data", "--data-ascii", "--data-binary"}))
    if data is not None:
        source, remaining = data
        url_args = _without_curl_transport_flags(remaining)
        if url_args is None:
            return False
        source_is_sensitive = (source == "@-" and event in _SENSITIVE_EVENTS) or (
            source.startswith("@") and _is_sensitive_absolute_path(source[1:])
        )
        return source_is_sensitive and _one_external_curl_url(url_args)

    upload = _flag_source(args, frozenset({"-T", "--upload-file"}))
    if upload is None:
        return False
    source, remaining = upload
    url_args = _without_curl_transport_flags(remaining)
    if url_args is None:
        return False
    source_is_sensitive = (source == "-" and event in _SENSITIVE_EVENTS) or (
        _is_sensitive_absolute_path(source)
    )
    return source_is_sensitive and _one_external_curl_url(url_args)


def _wget_exec_proof(args: list[str]) -> bool:
    body = _flag_source(args, frozenset({"--post-file", "--body-file"}))
    if body is None:
        return False
    source, remaining = body
    return _is_sensitive_absolute_path(source) and _one_external_url(remaining)


def _shell_curl_proof(event: str, command: str) -> bool:
    if not _is_safe_literal(command) or any(
        character in command for character in "\t\r\n\"'|&;<>`()\\*?[]!"
    ):
        return False
    tokens = command.split(" ")
    if any(not token for token in tokens) or not tokens or tokens.pop(0) != "curl":
        return False
    data = _flag_source(tokens, frozenset({"-d"}))
    if data is None:
        return False
    source, remaining = data
    url_args = _without_curl_transport_flags(remaining)
    if url_args is None:
        return False
    source_is_sensitive = (source == "@-" and event in _SENSITIVE_EVENTS) or (
        source.startswith("@") and _is_sensitive_shell_path(source[1:])
    )
    return source_is_sensitive and _one_external_curl_url(url_args)


def _command_bh2_transport(declaration: _HookDeclaration, args: list[str]) -> str | None:
    command = declaration.handler.get("command")
    if command == "curl" and _curl_exec_proof(declaration.event, args):
        return "curl"
    if command == "wget" and _wget_exec_proof(args):
        return "wget"
    if command in {"scp", "rsync"} and len(args) == 2:
        if _is_sensitive_absolute_path(args[0]) and _is_remote_destination(args[1], command):
            return command
    return None


def _command_bh2_proof(declaration: _HookDeclaration) -> _Bh2Proof | None:
    handler = declaration.handler
    command = handler.get("command")
    if not isinstance(command, str) or not _is_safe_literal(command):
        return None
    if "args" not in handler:
        if handler.get("shell") not in (None, "bash"):
            return None
        return (
            _Bh2Proof("direct_command_upload", "curl")
            if _shell_curl_proof(declaration.event, command)
            else None
        )

    args = _literal_args(handler)
    if args is None:
        return None
    transport = _command_bh2_transport(declaration, args)
    return _Bh2Proof("direct_command_upload", transport) if transport is not None else None


def _bh2_proof(declaration: _HookDeclaration) -> _Bh2Proof | None:
    if declaration.handler_type == "http":
        if declaration.event in _SENSITIVE_EVENTS and declaration.remote_http:
            return _Bh2Proof("event_http_body", "http")
        return None
    if declaration.handler_type == "command":
        return _command_bh2_proof(declaration)
    return None


def _permission_allow_declaration(value: str) -> _PermissionDeclaration | None:
    if value in {
        "Bash",
        "Bash(*)",
        "PowerShell",
        "PowerShell(*)",
        "Read",
        "Edit",
        "Write",
    }:
        return _PermissionDeclaration(Severity.CRITICAL, "whole_tool")

    match = re.fullmatch(r"(Read|Edit)\((.*)\)", value)
    if not match:
        return None
    specifier = match.group(2)
    if specifier in {"//", "//**", "~", "~/**"}:
        return _PermissionDeclaration(Severity.CRITICAL, "root_or_home")
    if _is_sensitive_tilde_path(specifier):
        return _PermissionDeclaration(Severity.HIGH, "sensitive_path")
    return None


def _permission_document(path: str, document: object) -> tuple[dict[str, object] | None, bool]:
    if path not in {".claude/settings.json", ".claude/settings.local.json"}:
        return None, False
    if not isinstance(document, dict) or "permissions" not in document:
        return None, False
    permissions = document.get("permissions")
    if not isinstance(permissions, dict):
        return None, True
    return permissions, False


def _permission_list_values(
    permissions: dict[str, object], key: str, *, limit: int
) -> tuple[list[str], bool, int]:
    if key not in permissions:
        return [], False, 0
    raw_values = permissions.get(key)
    if not isinstance(raw_values, list):
        return [], True, 0
    observed = min(len(raw_values), limit + 1)
    values: list[str] = []
    partial = len(raw_values) > limit
    for value in raw_values[:limit]:
        if not _is_nonempty_bounded_string(value):
            partial = True
        else:
            values.append(value)
    return values, partial, observed


def _permission_rule_is_valid(value: str) -> bool:
    tool, separator, remainder = value.partition("(")
    if re.fullmatch(r"[A-Za-z0-9_*.-]+", tool) is None:
        return False
    if not separator:
        return ")" not in value
    if not remainder.endswith(")") or len(remainder) == 1:
        return False
    depth = 0
    for character in remainder[:-1]:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _permission_allow_rule_is_valid(value: str) -> bool:
    if not _permission_rule_is_valid(value):
        return False
    tool = value.partition("(")[0]
    return (
        "*" not in tool
        or re.fullmatch(r"mcp__[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]*\*", tool) is not None
    )


def _permission_rule_values(
    permissions: dict[str, object], *, limit: int
) -> tuple[list[str], bool, int]:
    values, partial, observed = _permission_list_values(permissions, "allow", limit=limit)
    valid_values = [value for value in values if _permission_allow_rule_is_valid(value)]
    return valid_values, partial or len(valid_values) != len(values), observed


def _permission_rule_list_is_schema_compatible(permissions: dict[str, object], key: str) -> bool:
    if key not in permissions:
        return True
    values = permissions.get(key)
    validator = _permission_allow_rule_is_valid if key == "allow" else _permission_rule_is_valid
    return isinstance(values, list) and all(
        isinstance(value, str) and _is_safe_literal(value) and validator(value) for value in values
    )


def _unclassified_permission_lists_are_valid(permissions: dict[str, object]) -> bool:
    return all(
        _permission_rule_list_is_schema_compatible(permissions, key) for key in ("ask", "deny")
    )


def _permission_mode_declaration(value: str) -> _PermissionDeclaration | None:
    if value == "bypassPermissions":
        return _PermissionDeclaration(Severity.CRITICAL, "mode")
    if value == "acceptEdits":
        return _PermissionDeclaration(Severity.MEDIUM, "mode")
    if value == "auto":
        return _PermissionDeclaration(Severity.LOW, "mode", "ignored_by_surface")
    return None


def _permission_scalar_declarations(
    permissions: dict[str, object], *, limit: int
) -> tuple[list[_PermissionDeclaration], bool, int]:
    declarations: list[_PermissionDeclaration] = []
    partial = False
    observed = 0
    if "defaultMode" in permissions:
        default_mode = permissions.get("defaultMode")
        observed += 1
        if observed > limit:
            return declarations, True, observed
        if (
            not isinstance(default_mode, str)
            or not _is_safe_literal(default_mode)
            or default_mode not in _VALID_DEFAULT_MODES
        ):
            partial = True
        else:
            declaration = (
                None
                if default_mode == "bypassPermissions"
                and permissions.get("disableBypassPermissionsMode") == "disable"
                else _permission_mode_declaration(default_mode)
            )
            if declaration is not None:
                declarations.append(declaration)

    for key in ("disableBypassPermissionsMode", "disableAutoMode"):
        if key in permissions:
            observed += 1
            if observed > limit:
                return declarations, True, observed
            if permissions.get(key) != "disable":
                partial = True
    return declarations, partial, observed


def _is_root_or_home_directory(value: str) -> bool:
    if value.startswith("/"):
        if (
            value.startswith("//")
            and not value.startswith("///")
            and any(segment not in {"", ".", ".."} for segment in value[2:].split("/"))
        ):
            return False
        return posixpath.normpath(value) in {"/", "//"}
    if value == "~":
        return True
    if not value.startswith("~/"):
        return False

    depth = 0
    for segment in value[2:].split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if depth == 0:
                return False
            depth -= 1
        else:
            depth += 1
    return depth == 0


def _permission_declarations(
    path: str, document: object, *, limit: int
) -> tuple[list[_PermissionDeclaration], bool, int]:
    permissions, partial = _permission_document(path, document)
    if permissions is None:
        return [], partial, 0
    partial = partial or not _unclassified_permission_lists_are_valid(permissions)
    declarations: list[_PermissionDeclaration] = []
    observed = 0

    allow, rules_partial, rules_observed = _permission_rule_values(permissions, limit=limit)
    declarations.extend(
        declaration
        for value in allow
        if (declaration := _permission_allow_declaration(value)) is not None
    )
    partial = partial or rules_partial
    observed += rules_observed
    if observed > limit:
        return declarations, True, observed

    directories, directories_partial, directories_observed = _permission_list_values(
        permissions, "additionalDirectories", limit=max(0, limit - observed)
    )
    declarations.extend(
        _PermissionDeclaration(Severity.CRITICAL, "directory")
        for value in directories
        if _is_root_or_home_directory(value)
    )
    partial = partial or directories_partial
    observed += directories_observed
    if observed > limit:
        return declarations, True, observed

    scalar_declarations, scalar_partial, scalar_observed = _permission_scalar_declarations(
        permissions, limit=max(0, limit - observed)
    )
    declarations.extend(scalar_declarations)
    partial = partial or scalar_partial
    observed += scalar_observed
    return declarations, partial, observed


def _scan_declarations(
    path: str,
    document: object,
    previous_settings_hook_ids: set[_HookIdentity] | None = None,
) -> _DeclarationScan:
    if not isinstance(document, dict):
        return _DeclarationScan(hooks=[], permissions=[], partial=True, observed=0)
    hooks, hook_partial, hook_observed = _hook_declarations(
        document,
        limit=_MAX_DECLARATIONS,
        previous_handler_ids=previous_settings_hook_ids,
    )
    remaining = max(0, _MAX_DECLARATIONS - hook_observed)
    permissions, permission_partial, permission_observed = _permission_declarations(
        path, document, limit=remaining
    )
    is_settings = path in {".claude/settings.json", ".claude/settings.local.json"}
    return _DeclarationScan(
        hooks=hooks,
        permissions=permissions,
        partial=(
            hook_partial
            or permission_partial
            or (is_settings and not _modeled_settings_fields_are_valid(document))
            or (
                is_settings
                and document.get("disableAllHooks") is True
                and not _settings_schema_allows_disable(document)
            )
            or (path == "hooks/hooks.json" and "hooks" not in document)
        ),
        observed=hook_observed + permission_observed,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise _NonFiniteConstantError(value)


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteConstantError(value)
    return parsed


def _parse_document(content: str) -> object:
    return json.loads(
        content,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite_constant,
        parse_float=_parse_finite_float,
    )


def _handler_schema_allows_disable(handler: object) -> bool:
    if not isinstance(handler, dict) or not _handler_strings_are_bounded(handler):
        return False
    handler_type = handler.get("type")
    return (
        isinstance(handler_type, str)
        and handler_type in _KNOWN_HANDLER_TYPES
        and _handler_shape_is_valid(
            handler_type,
            handler,
            url_validator=_handler_schema_url_is_valid,
        )
    )


def _hook_group_schema_allows_disable(group: object) -> bool:
    if not isinstance(group, dict):
        return False
    handlers = group.get("hooks")
    if not isinstance(handlers, list):
        return False
    matcher = group.get("matcher")
    if "matcher" in group and (not isinstance(matcher, str) or not _is_safe_literal(matcher)):
        return False
    return all(_handler_schema_allows_disable(handler) for handler in handlers)


def _hooks_schema_allows_disable(document: dict[str, object]) -> bool:
    hooks = document.get("hooks")
    if hooks is None:
        return "hooks" not in document
    if not isinstance(hooks, dict):
        return False
    for event, groups in hooks.items():
        if not isinstance(event, str) or not _is_safe_literal(event):
            return False
        if event not in _KNOWN_EVENTS:
            continue
        if not isinstance(groups, list) or not all(
            _hook_group_schema_allows_disable(group) for group in groups
        ):
            return False
    return True


def _permissions_schema_allows_disable(document: dict[str, object]) -> bool:
    permissions = document.get("permissions")
    if permissions is None:
        return "permissions" not in document
    if not isinstance(permissions, dict):
        return False
    if not all(
        _permission_rule_list_is_schema_compatible(permissions, key)
        for key in ("allow", "ask", "deny")
    ):
        return False
    directories = permissions.get("additionalDirectories", [])
    if not isinstance(directories, list) or not all(
        _is_nonempty_bounded_string(value) for value in directories
    ):
        return False
    if "defaultMode" in permissions:
        default_mode = permissions.get("defaultMode")
        if (
            not isinstance(default_mode, str)
            or not _is_safe_literal(default_mode)
            or default_mode not in _VALID_DEFAULT_MODES
        ):
            return False
    return all(
        key not in permissions or permissions.get(key) == "disable"
        for key in ("disableBypassPermissionsMode", "disableAutoMode")
    )


def _modeled_settings_fields_are_valid(document: dict[str, object]) -> bool:
    if "disableAllHooks" in document and not isinstance(document.get("disableAllHooks"), bool):
        return False
    if "$schema" in document and not _is_bounded_string(document.get("$schema")):
        return False
    if "model" in document and not _is_bounded_string(document.get("model")):
        return False
    if "includeCoAuthoredBy" in document and not _is_bool(document.get("includeCoAuthoredBy")):
        return False
    if "env" in document and not _is_bounded_string_map(document.get("env")):
        return False
    return _permissions_schema_allows_disable(document)


def _settings_schema_allows_disable(document: dict[str, object]) -> bool:
    if not set(document).issubset(_DISABLE_TRUSTED_TOP_LEVEL_KEYS):
        return False
    if not _modeled_settings_fields_are_valid(document):
        return False
    return _hooks_schema_allows_disable(document)


def _bundled_hooks_are_disabled(
    applicable_paths: set[str],
    file_cache: dict[str, str],
    decodable: dict[str, bool],
) -> bool:
    for path in (".claude/settings.local.json", ".claude/settings.json"):
        if path not in applicable_paths:
            continue
        content = file_cache.get(path)
        if (
            decodable.get(path) is False
            or not isinstance(content, str)
            or len(content) > MAX_FILE_CHARS
        ):
            return False
        try:
            document = _parse_document(content)
        except (RecursionError, ValueError):
            return False
        if not isinstance(document, dict):
            return False
        if "disableAllHooks" in document:
            value = document.get("disableAllHooks")
            if not isinstance(value, bool) or not value:
                return False
            return _settings_schema_allows_disable(document)
    return False


def _bh1_severity(declarations: list[_HookDeclaration]) -> Severity:
    severity = Severity.LOW
    for declaration in declarations:
        if declaration.remote_http:
            return Severity.HIGH
        if declaration.ambient or declaration.handler_type in {
            "prompt",
            "agent",
            "http",
            "mcp_tool",
        }:
            severity = Severity.MEDIUM
        elif declaration.handler_type != "command":
            severity = Severity.MEDIUM
    return severity


def _payload_is_directly_modeled(declaration: _HookDeclaration) -> bool:
    return _bh2_proof(declaration) is not None


def _payload_analysis_level(declarations: list[_HookDeclaration]) -> str:
    if any(declaration.handler_type not in _KNOWN_HANDLER_TYPES for declaration in declarations):
        return "unmodeled"
    payload_declarations = [
        declaration
        for declaration in declarations
        if declaration.handler_type in {"command", "http"}
    ]
    if not payload_declarations:
        return "not_applicable"
    if all(_payload_is_directly_modeled(declaration) for declaration in payload_declarations):
        return "direct"
    return "unmodeled"


def _target_summary(declarations: list[_HookDeclaration]) -> str:
    ambient = any(declaration.ambient for declaration in declarations)
    declarations = [declaration for declaration in declarations if declaration.ambient == ambient]
    summaries = {
        "remote_http" if declaration.remote_http else declaration.handler_type
        for declaration in declarations
    }
    priority = ("remote_http", "http", "agent", "prompt", "mcp_tool", "command")
    return next((value for value in priority if value in summaries), "unsupported")


def _bh1_finding(path: str, declarations: list[_HookDeclaration]) -> Finding:
    handler_types = sorted(
        {
            declaration.handler_type
            if declaration.handler_type in _KNOWN_HANDLER_TYPES
            else "unsupported"
            for declaration in declarations
        }
    )[:32]
    events = sorted(
        {
            declaration.event if declaration.event in _KNOWN_EVENTS else "unsupported"
            for declaration in declarations
        }
    )[:32]
    reach = "ambient" if any(declaration.ambient for declaration in declarations) else "scoped"
    analyzer_finding = AnalyzerFinding(
        rule_id="BH1",
        message="Bundled hooks can execute when matching lifecycle events occur.",
        severity=_bh1_severity(declarations),
        location=Location(path, 1),
        confidence=0.95,
        remediation="Review bundled hook handlers, destinations, and event reach before install.",
        tags=["Bundled Execution Surface", "Hooks"],
        matched_text=f"document:{path}",
        evidence={
            "activation_state": "conditional",
            "activation_reason": "requires_hook_activation",
            "declaration_count": len(declarations),
            "events": events,
            "handler_types": handler_types,
            "matcher_breadth": sorted(
                {declaration.matcher_breadth for declaration in declarations}
            )[:32],
            "payload_analysis_level": _payload_analysis_level(declarations),
            "reach": reach,
            "target_summary": _target_summary(declarations),
            "unknown_event_count": sum(
                declaration.event not in _KNOWN_EVENTS for declaration in declarations
            ),
            "unknown_handler_count": sum(
                declaration.handler_type not in _KNOWN_HANDLER_TYPES for declaration in declarations
            ),
        },
    )
    return analyzer_finding_to_finding(analyzer_finding)


def _bh2_finding(path: str, proofs: list[_Bh2Proof]) -> Finding:
    analyzer_finding = AnalyzerFinding(
        rule_id="BH2",
        message="A bundled hook directly sends sensitive event or file content remotely.",
        severity=Severity.CRITICAL,
        location=Location(path, 1),
        confidence=0.99,
        remediation="Remove the remote transfer or require explicit, narrowly scoped user action.",
        tags=["Bundled Execution Surface", "Exfiltration"],
        matched_text=f"document:{path}",
        evidence={
            "activation_reason": "requires_hook_activation",
            "activation_state": "conditional",
            "proof_count": len(proofs),
            "proof_kinds": sorted({proof.kind for proof in proofs})[:32],
            "proof_status": "closed",
            "transport_kinds": sorted({proof.transport for proof in proofs})[:32],
        },
    )
    return analyzer_finding_to_finding(analyzer_finding)


def _bh3_finding(path: str, declarations: list[_PermissionDeclaration]) -> Finding:
    rank = {
        Severity.LOW: 0,
        Severity.MEDIUM: 1,
        Severity.HIGH: 2,
        Severity.CRITICAL: 3,
    }
    severity = max((declaration.severity for declaration in declarations), key=rank.__getitem__)
    activation_state = (
        "conditional"
        if any(declaration.activation_state == "conditional" for declaration in declarations)
        else "ignored_by_surface"
    )
    ignored = activation_state == "ignored_by_surface"
    analyzer_finding = AnalyzerFinding(
        rule_id="BH3",
        message=(
            "Bundled project settings declare a permission mode ignored on this surface."
            if ignored
            else "Bundled project settings declare a broad permission surface."
        ),
        severity=severity,
        location=Location(path, 1),
        confidence=0.99,
        remediation=(
            "Remove the ignored mode if it is unintended; it does not expand permissions here."
            if ignored
            else "Remove broad grants and declare only the narrow tools and paths required."
        ),
        tags=["Bundled Execution Surface", "Permissions"],
        matched_text=f"document:{path}",
        evidence={
            "activation_reason": (
                "mode_ignored_in_project_settings" if ignored else "requires_settings_activation"
            ),
            "activation_state": activation_state,
            "activation_states": sorted(
                {declaration.activation_state for declaration in declarations}
            ),
            "declaration_count": len(declarations),
            "grant_kinds": sorted({declaration.kind for declaration in declarations})[:32],
        },
    )
    finding = analyzer_finding_to_finding(analyzer_finding)
    if ignored:
        finding.explanation = (
            "The auto permission mode is recognized but ignored by this supported project "
            "settings surface, so it does not receive a blocking score floor."
        )
    return finding


def _analyze_document(
    path: str,
    content: str,
    previous_settings_hook_ids: set[_HookIdentity] | None = None,
    *,
    hooks_disabled: bool = False,
) -> tuple[list[Finding], InspectionLedgerEvent]:
    if len(content) > MAX_FILE_CHARS:
        return [], ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            phase="static",
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=LedgerReason.SIZE_LIMIT,
            observed_characters=len(content),
            limit_characters=MAX_FILE_CHARS,
            observed_bytes=len(content.encode("utf-8", errors="replace")),
        )
    try:
        document = _parse_document(content)
    except (RecursionError, ValueError):
        return [], ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            phase="static",
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=LedgerReason.OPAQUE_CONTENT,
        )

    scan = _scan_declarations(path, document, previous_settings_hook_ids)
    declarations = [] if hooks_disabled else scan.hooks
    proofs = [proof for declaration in declarations if (proof := _bh2_proof(declaration))]
    permission_declarations = scan.permissions
    findings: list[Finding] = []
    if declarations:
        findings.append(_bh1_finding(path, declarations))
    if proofs:
        findings.append(_bh2_finding(path, proofs))
    if permission_declarations:
        findings.append(_bh3_finding(path, permission_declarations))
    if scan.partial:
        return findings, ledger_event(
            outcome=LedgerOutcome.PARTIAL,
            phase="static",
            analyzer_id=ANALYZER_ID,
            path=path,
            reason=LedgerReason.OPAQUE_CONTENT,
            emitted_finding_ids=[finding.finding_id for finding in findings],
            observed_records=scan.observed,
            limit_records=_MAX_DECLARATIONS,
        )
    return findings, ledger_event(
        outcome=LedgerOutcome.COMPLETED,
        phase="static",
        analyzer_id=ANALYZER_ID,
        path=path,
        emitted_finding_ids=[finding.finding_id for finding in findings],
    )


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Inspect exact bundled hook/settings paths using bounded literal classifiers."""
    components = state.get("components") or []
    file_cache = state.get("local_file_cache") or state.get("file_cache") or {}
    decodable = {
        record.get("path"): record.get("decodable", True)
        for record in (state.get("artifact_inventory") or [])
    }
    findings: list[Finding] = []
    ledger_events: list[InspectionLedgerEvent] = []
    previous_settings_hook_ids: set[_HookIdentity] = set()
    applicable_paths = set(components).intersection(_APPLICABLE_PATHS)
    hooks_disabled = _bundled_hooks_are_disabled(applicable_paths, file_cache, decodable)

    for path in sorted(applicable_paths):
        if decodable.get(path) is False:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.PARTIAL,
                    phase="static",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.OPAQUE_CONTENT,
                )
            )
            continue
        content = file_cache.get(path)
        if content is None:
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.MISSING_FILE_CACHE,
                )
            )
            continue
        try:
            path_findings, event = _analyze_document(
                path,
                content,
                previous_settings_hook_ids
                if path in {".claude/settings.json", ".claude/settings.local.json"}
                else None,
                hooks_disabled=hooks_disabled,
            )
        except Exception as exc:
            logger.exception("%s failed for %s", ANALYZER_ID, path)
            ledger_events.append(
                ledger_event(
                    outcome=LedgerOutcome.FAILED,
                    phase="static",
                    analyzer_id=ANALYZER_ID,
                    path=path,
                    reason=LedgerReason.ANALYZER_RUNTIME_ERROR,
                    error_class=type(exc).__name__,
                )
            )
            continue
        findings.extend(path_findings)
        ledger_events.append(event)

    status = analyzer_status_for_events(ANALYZER_ID, ledger_events)
    return {
        "findings": findings,
        "inspection_ledger": ledger_events,
        "analyzer_status_events": [status],
    }
