# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded source-to-sink analysis for bundled Claude hook handlers.

The discovery analyzer deliberately drops executable payloads from its normalized
runtime records.  This module receives a separate, repr-hidden analysis input and
returns only sanitized findings and terminal-work metadata.  Raw commands, URLs,
headers, environment names, and payload text must never cross that boundary.
"""

from __future__ import annotations

import ast
import ipaddress
import re
import shlex
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Final
from urllib.parse import urlsplit

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.models import Finding
from skillspector.python_ast import get_python_ast

from .bundled_hook_runtime import HookRegistration
from .common import apply_import_aliases, resolve_dotted_name
from .static_runner import MAX_FILE_CHARS

_SCHEMA: Final = "skillspector.bundled_hook.v1"
_SEMANTICS_SNAPSHOT: Final = "2.1.238"
_ENV_REFERENCE: Final = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_SENSITIVE_ENV_TOKEN: Final = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|API[_-]?KEY|ACCESS[_-]?KEY)",
    re.IGNORECASE,
)
_SENSITIVE_ENV_SEGMENT: Final = re.compile(
    r"(?:^|_)(?:AUTH_CONFIG|JWT|PAT)(?:_|$)",
    re.IGNORECASE,
)
_POWERSHELL_ENV_REFERENCE: Final = re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)", re.I)
_CMD_ENV_REFERENCE: Final = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")
_USER_CONFIG_REFERENCE: Final = re.compile(r"\$\{user_config\.([A-Za-z0-9_.-]+)\}")
_BUNDLE_REFERENCE: Final = re.compile(r"\$\{CLAUDE_(PLUGIN_ROOT|PROJECT_DIR)\}/([^\s\"';&|]+)")
_CD_BUNDLE_REFERENCE: Final = re.compile(
    r"\bcd\s+[\"']?\$(?:\{CLAUDE_(PLUGIN_ROOT|PROJECT_DIR)\}|"
    r"CLAUDE_(PLUGIN_ROOT|PROJECT_DIR))[\"']?\s*&&\s*"
    r"(?:(?:node|python(?:3)?|bash|sh|zsh)\s+)?[\"']?(?:\./)?"
    r"([A-Za-z0-9_./@%+=,:~-]+)"
)
_MAX_WRAPPER_HOPS: Final = 2
_MAX_REFERENCED_COMPONENTS: Final = 8
_MAX_AGGREGATE_PAYLOAD_CHARS: Final = 2_000_000

_EVENT_SOURCES: Final[dict[str, str]] = {
    "UserPromptSubmit": "user_prompt_event",
    "UserPromptExpansion": "expanded_prompt_event",
    "PreToolUse": "tool_input_event",
    "PermissionRequest": "tool_input_event",
    "PermissionDenied": "tool_input_event",
    "PostToolUse": "tool_result_event",
    "PostToolUseFailure": "tool_error_event",
    "PostToolBatch": "tool_batch_event",
    "MessageDisplay": "displayed_message_event",
    "Notification": "notification_message_event",
    "TaskCreated": "task_content_event",
    "TaskCompleted": "task_content_event",
    "Stop": "assistant_message_event",
    "StopFailure": "stop_failure_event",
    "SubagentStop": "assistant_message_event",
    "PreCompact": "compaction_content_event",
    "PostCompact": "compaction_content_event",
    "Elicitation": "elicitation_content_event",
    "ElicitationResult": "elicitation_content_event",
}


class SensitiveSourceKind(StrEnum):
    """Sanitized classes of data which may reach an outbound sink."""

    EVENT = "event"
    ENVIRONMENT = "environment"
    LOCAL_FILE = "local_file"
    USER_CONFIG = "user_config"


class TransportKind(StrEnum):
    """Outbound transport families used in safe BH2 evidence."""

    HTTP = "http"
    SSH = "ssh"
    TCP = "tcp"
    MAIL = "mail"
    DNS = "dns"
    OBJECT_STORE = "object_store"


class DestinationClass(StrEnum):
    """Trust-boundary classification without retaining a destination."""

    LOOPBACK = "loopback"
    PUBLIC_REMOTE = "public_remote"
    PRIVATE_REMOTE = "private_remote"
    LINK_LOCAL_REMOTE = "link_local_remote"
    DYNAMIC_UNKNOWN = "dynamic_unknown"


@dataclass(frozen=True, slots=True)
class HandlerFlowInput:
    """One normalized registration plus the minimum raw material needed for flow."""

    registration: HookRegistration
    command: str | None = field(default=None, repr=False)
    args: tuple[str, ...] | None = field(default=None, repr=False)
    url: str | None = field(default=None, repr=False)
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    allowed_env_vars: frozenset[str] = field(default_factory=frozenset, repr=False)


@dataclass(frozen=True, slots=True)
class DocumentFlowInput:
    """Sanitized document identity paired with repr-hidden handler inputs."""

    source_kind: str
    declaration_roles: tuple[str, ...]
    source_path: str
    activation_lifetime: str
    content_digest: str
    handlers: tuple[HandlerFlowInput, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class UserConfigProfile:
    """Only normalized identities of plugin-owned sensitive settings."""

    sensitive_keys: frozenset[str] = field(default_factory=frozenset, repr=False)
    sensitive_environment_names: frozenset[str] = field(default_factory=frozenset, repr=False)
    authentication_only_keys: frozenset[str] = field(default_factory=frozenset, repr=False)


@dataclass(frozen=True, slots=True)
class FlowWorkRef:
    """One producer-work identity, matching inspection-ledger identity fields."""

    path: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class FlowWorkResult:
    """Sanitized terminal status for one referenced component or activation edge."""

    ref: FlowWorkRef
    outcome: LedgerOutcome
    reason: LedgerReason | None = None
    error_class: str | None = None
    observed_characters: int | None = None
    limit_characters: int | None = None


@dataclass(frozen=True, slots=True)
class OwnedFlowFinding:
    """A finding paired with the single producer work item that owns it."""

    owner: FlowWorkRef
    finding: Finding


@dataclass(frozen=True, slots=True)
class FlowBatch:
    """All sanitized flow outputs for one analyzer invocation."""

    findings: tuple[OwnedFlowFinding, ...] = ()
    work: tuple[FlowWorkResult, ...] = ()


@dataclass(frozen=True, slots=True)
class _SinkHit:
    source_kind: str
    transport: TransportKind
    destination: DestinationClass
    line: int = 1


@dataclass(frozen=True, slots=True)
class _Reference:
    scope: str
    relative: str = field(repr=False)
    line: int = 1


@dataclass(slots=True)
class _HandlerBudget:
    seen: set[str] = field(default_factory=set)
    counted: set[str] = field(default_factory=set)
    aggregate_characters: int = 0


@dataclass(slots=True)
class _UserConfigUse:
    origins: set[str] = field(default_factory=set, repr=False)
    has_other_use: bool = False


def capture_handler(registration: HookRegistration, handler: dict[str, object]) -> HandlerFlowInput:
    """Copy only analysis-relevant handler fields into a repr-hidden record."""
    command = handler.get("command")
    raw_args = handler.get("args") if "args" in handler else None
    args = (
        tuple(value for value in raw_args if isinstance(value, str))
        if isinstance(raw_args, list)
        else None
    )
    url = handler.get("url")
    raw_headers = handler.get("headers")
    headers = (
        tuple(
            (key, value)
            for key, value in raw_headers.items()
            if isinstance(key, str) and isinstance(value, str)
        )
        if isinstance(raw_headers, dict)
        else ()
    )
    raw_allowed = handler.get("allowedEnvVars")
    allowed = (
        frozenset(value for value in raw_allowed if isinstance(value, str))
        if isinstance(raw_allowed, list)
        else frozenset()
    )
    return HandlerFlowInput(
        registration=registration,
        command=command if isinstance(command, str) else None,
        args=args,
        url=url if isinstance(url, str) else None,
        headers=headers,
        allowed_env_vars=allowed,
    )


def _user_config_environment_name(key: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]", "_", key).upper()
    return f"CLAUDE_PLUGIN_OPTION_{suffix}"


def build_user_config_profile(value: object) -> UserConfigProfile:
    """Extract sensitive setting identities without retaining manifest prose."""
    if not isinstance(value, dict):
        return UserConfigProfile()
    keys: set[str] = set()
    environment_names: set[str] = set()
    for raw_key, raw_spec in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_spec, dict):
            continue
        if raw_spec.get("sensitive") is not True:
            continue
        keys.add(raw_key)
        environment_names.add(_user_config_environment_name(raw_key))
    return UserConfigProfile(frozenset(keys), frozenset(environment_names))


def _destination_for_url(url: str | None) -> DestinationClass:
    if not url:
        return DestinationClass.DYNAMIC_UNKNOWN
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError:
        return DestinationClass.DYNAMIC_UNKNOWN
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or "$" in parsed.netloc
        or "${" in parsed.netloc
    ):
        return DestinationClass.DYNAMIC_UNKNOWN
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return DestinationClass.LOOPBACK
    if _is_numeric_loopback(normalized):
        return DestinationClass.LOOPBACK
    try:
        address = _normalized_ip_address(normalized)
    except ValueError:
        return DestinationClass.PUBLIC_REMOTE
    if address.is_loopback:
        return DestinationClass.LOOPBACK
    if address.is_link_local:
        return DestinationClass.LINK_LOCAL_REMOTE
    if address.is_private:
        return DestinationClass.PRIVATE_REMOTE
    return DestinationClass.PUBLIC_REMOTE


def _header_environment_source(
    handler: HandlerFlowInput, profile: UserConfigProfile | None
) -> str | None:
    if not handler.allowed_env_vars:
        return None
    destination = _destination_for_url(handler.url)
    for header_name, value in handler.headers:
        for match in _ENV_REFERENCE.finditer(value):
            variable = match.group(1) or match.group(2)
            sensitive_plugin_value = bool(
                profile and variable in profile.sensitive_environment_names
            )
            if variable in handler.allowed_env_vars and (
                _sensitive_environment_name(variable, profile) or sensitive_plugin_value
            ):
                if profile and variable in profile.sensitive_environment_names:
                    key = next(
                        (
                            candidate
                            for candidate in profile.authentication_only_keys
                            if _user_config_environment_name(candidate) == variable
                        ),
                        None,
                    )
                    if (
                        key is not None
                        and header_name.casefold() == "authorization"
                        and destination
                        not in {
                            DestinationClass.DYNAMIC_UNKNOWN,
                            DestinationClass.LOOPBACK,
                        }
                    ):
                        continue
                    return "plugin_sensitive_user_config"
                return "ambient_credential_environment"
    return None


def _normalized_executable(value: str) -> str:
    executable = PurePosixPath(value.replace("\\", "/")).name.lower()
    return executable[:-4] if executable.endswith(".exe") else executable


def _sensitive_environment_name(name: str, profile: UserConfigProfile | None = None) -> bool:
    return bool(_SENSITIVE_ENV_TOKEN.search(name) or _SENSITIVE_ENV_SEGMENT.search(name)) or bool(
        profile and name in profile.sensitive_environment_names
    )


def _environment_names(value: str) -> tuple[str, ...]:
    names = [match.group(1) or match.group(2) for match in _ENV_REFERENCE.finditer(value)]
    names.extend(match.group(1) for match in _POWERSHELL_ENV_REFERENCE.finditer(value))
    names.extend(match.group(1) for match in _CMD_ENV_REFERENCE.finditer(value))
    return tuple(dict.fromkeys(names))


def _sensitive_path(value: str, *, expand_shell: bool) -> bool:
    candidate = value.strip().lstrip("@").replace("\\", "/")
    if not expand_shell and ("$" in candidate or "%" in candidate or candidate.startswith("~")):
        return False
    if expand_shell:
        candidate = re.sub(r"^(?:\$HOME|\$\{HOME\}|~)(?=/)", "/home/user", candidate)
    lowered = candidate.lower()
    if lowered.endswith(".env.example") or "/.env.example" in lowered:
        return False
    return (
        lowered == ".env"
        or any(
            marker in lowered
            for marker in (
                "/.ssh/id_",
                "/.aws/credentials",
                "/.azure/accesstokens.json",
                "/.config/gh/hosts.yml",
                "/.config/gcloud/application_default_credentials.json",
                "/.config/rclone/rclone.conf",
                "/.docker/config.json",
                "/.kube/config",
                "/.npmrc",
                "/.bash_history",
                "/.zsh_history",
                "/.claude/settings.json",
                "/.gnupg/",
                "/credentials",
            )
        )
        or lowered.endswith(("/.env", ".pem", ".key"))
    )


def _value_taint(
    value: str,
    *,
    expand_shell: bool,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
    include_sensitive_path: bool = True,
) -> str | None:
    taints = _value_taints(
        value,
        expand_shell=expand_shell,
        variables=variables,
        profile=profile,
        include_sensitive_path=include_sensitive_path,
    )
    return taints[0] if taints else None


def _value_taints(
    value: str,
    *,
    expand_shell: bool,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
    include_sensitive_path: bool = True,
) -> tuple[str, ...]:
    """Return every distinct taint class present without order-dependent loss."""
    taints: list[str] = []
    if include_sensitive_path and _sensitive_path(value, expand_shell=expand_shell):
        taints.append("sensitive_local_file")
    for key in _USER_CONFIG_REFERENCE.findall(value):
        if profile and key in profile.sensitive_keys:
            taints.append("plugin_sensitive_user_config")
    if expand_shell:
        for name in _environment_names(value):
            if name in variables:
                taints.append(variables[name])
            elif _sensitive_environment_name(name, profile):
                taints.append(
                    "plugin_sensitive_user_config"
                    if profile and name in profile.sensitive_environment_names
                    else "ambient_credential_environment"
                )
    return tuple(dict.fromkeys(taints))


def _curl_operand_taints(
    option: str,
    value: str,
    *,
    expand_shell: bool,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
) -> tuple[str, ...]:
    """Classify a curl option value according to whether curl reads a file."""
    taints: list[str] = []
    file_value: str | None = None
    if option in {"--upload-file", "-T"}:
        file_value = value
    elif option in {"-d", "--data", "--data-ascii", "--data-binary", "--json"}:
        if value.startswith("@"):
            file_value = value[1:]
    elif option == "--data-urlencode":
        if value.startswith("@"):
            file_value = value[1:]
        else:
            named_file = re.fullmatch(r"[^=]+@(.+)", value, re.DOTALL)
            if named_file is not None:
                file_value = named_file.group(1)
    elif option in {"-F", "--form"}:
        marker = re.search(r"(?:^|=)[@<]([^;]+)", value)
        if marker is not None:
            file_value = marker.group(1)
    elif option in {"-H", "--header", "--proxy-header"} and value.startswith("@"):
        file_value = value[1:]
    if file_value and file_value != "-" and _sensitive_path(file_value, expand_shell=expand_shell):
        taints.append("sensitive_local_file")
    taints.extend(
        _value_taints(
            value,
            expand_shell=expand_shell,
            variables=variables,
            profile=profile,
            include_sensitive_path=False,
        )
    )
    return tuple(dict.fromkeys(taints))


def _shell_stdin_redirection_taint(
    words: tuple[str, ...],
    *,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
) -> tuple[bool, str | None]:
    """Return whether shell stdin is redirected and any proven source taint."""
    for index, word in enumerate(words):
        value: str | None = None
        if word in {"<", "0<"}:
            value = words[index + 1] if index + 1 < len(words) else None
        elif word.startswith("0<") and not word.startswith("0<<"):
            value = word[2:]
        elif word.startswith("<") and not word.startswith("<<"):
            value = word[1:]
        if value is not None:
            return (
                True,
                _value_taint(
                    value,
                    expand_shell=True,
                    variables=variables,
                    profile=profile,
                ),
            )
    return False, None


def _authentication_only_user_config_value(
    value: str,
    *,
    expand_shell: bool,
    profile: UserConfigProfile | None,
) -> bool:
    if profile is None or not profile.authentication_only_keys:
        return False
    referenced_keys = set(_USER_CONFIG_REFERENCE.findall(value))
    if expand_shell:
        environment_names = set(_environment_names(value))
        referenced_keys.update(
            key
            for key in profile.sensitive_keys
            if _user_config_environment_name(key) in environment_names
        )
    sensitive_references = referenced_keys & profile.sensitive_keys
    return bool(sensitive_references) and sensitive_references <= set(
        profile.authentication_only_keys
    )


def _is_authorization_header(value: str) -> bool:
    name, separator, _field_value = value.partition(":")
    return bool(separator) and name.strip().casefold() == "authorization"


def _destination_for_host(host: str | None) -> DestinationClass:
    if not host or "$" in host or "%" in host:
        return DestinationClass.DYNAMIC_UNKNOWN
    value = host.rsplit("@", 1)[-1].strip("[]").rstrip(".").lower()
    if value == "localhost" or value.endswith(".localhost"):
        return DestinationClass.LOOPBACK
    if _is_numeric_loopback(value):
        return DestinationClass.LOOPBACK
    try:
        address = _normalized_ip_address(value)
    except ValueError:
        return DestinationClass.PUBLIC_REMOTE
    if address.is_loopback:
        return DestinationClass.LOOPBACK
    if address.is_link_local:
        return DestinationClass.LINK_LOCAL_REMOTE
    if address.is_private:
        return DestinationClass.PRIVATE_REMOTE
    return DestinationClass.PUBLIC_REMOTE


def _is_numeric_loopback(value: str) -> bool:
    if not re.fullmatch(r"127(?:\.\d{1,3}){0,3}", value):
        return False
    return all(int(part) <= 255 for part in value.split("."))


def _normalized_ip_address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Normalize IPv4-mapped IPv6 consistently across supported Python patch releases."""
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


_CURL_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "-A",
        "-b",
        "-c",
        "-d",
        "-D",
        "-e",
        "-E",
        "-F",
        "-H",
        "-K",
        "-m",
        "-o",
        "-P",
        "-Q",
        "-r",
        "-T",
        "-u",
        "-U",
        "-w",
        "-x",
        "-X",
        "--cacert",
        "--capath",
        "--cert",
        "--cert-type",
        "--ciphers",
        "--connect-timeout",
        "--connect-to",
        "--cookie",
        "--cookie-jar",
        "--data",
        "--data-ascii",
        "--data-binary",
        "--data-raw",
        "--data-urlencode",
        "--dump-header",
        "--form",
        "--form-string",
        "--header",
        "--interface",
        "--json",
        "--key",
        "--limit-rate",
        "--local-port",
        "--max-filesize",
        "--max-redirs",
        "--max-time",
        "--noproxy",
        "--oauth2-bearer",
        "--output",
        "--pass",
        "--preproxy",
        "--proxy",
        "--proxy1.0",
        "--proxy-header",
        "--proxy-user",
        "--range",
        "--referer",
        "--request",
        "--resolve",
        "--retry",
        "--retry-delay",
        "--retry-max-time",
        "--socks4",
        "--socks4a",
        "--socks5",
        "--socks5-hostname",
        "--tls-max",
        "--tls-user",
        "--upload-file",
        "--url",
        "--user",
        "--user-agent",
        "--write-out",
    }
)
_CURL_SOURCE_OPTIONS: Final[frozenset[str]] = frozenset(
    {
        "-d",
        "--data",
        "--data-ascii",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
        "-F",
        "--form",
        "--form-string",
        "--upload-file",
        "-T",
        "-H",
        "--header",
        "--proxy-header",
        "-b",
        "--cookie",
        "--json",
        "--oauth2-bearer",
        "--referer",
        "-u",
        "--user",
    }
)
_CURL_SHORT_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    option for option in _CURL_VALUE_OPTIONS if len(option) == 2 and option.startswith("-")
)


def _curl_groups(words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if not words:
        return ()
    groups: list[tuple[str, ...]] = []
    current = [words[0]]
    for word in words[1:]:
        if word in {"--next", "-:"}:
            groups.append(tuple(current))
            current = [words[0]]
        else:
            current.append(word)
    groups.append(tuple(current))
    return tuple(groups)


def _curl_option_at(
    words: tuple[str, ...],
    index: int,
) -> tuple[str, str, int] | None:
    word = words[index]
    option, equals, inline_value = word.partition("=")
    if option in _CURL_VALUE_OPTIONS:
        if equals:
            return option, inline_value, index + 1
        value = words[index + 1] if index + 1 < len(words) else ""
        return option, value, min(len(words), index + 2)
    if len(word) > 2 and word.startswith("-") and not word.startswith("--"):
        for offset, short_name in enumerate(word[1:], start=1):
            short_option = f"-{short_name}"
            if short_option not in _CURL_SHORT_VALUE_OPTIONS:
                continue
            attached_value = word[offset + 1 :]
            if attached_value:
                return short_option, attached_value, index + 1
            value = words[index + 1] if index + 1 < len(words) else ""
            return short_option, value, min(len(words), index + 2)
    return None


def _curl_short_flags_before_value(word: str) -> tuple[str, ...]:
    """Return clustered flags which precede the first value-taking option."""
    if len(word) < 2 or not word.startswith("-") or word.startswith("--"):
        return ()
    flags: list[str] = []
    for short_name in word[1:]:
        short_option = f"-{short_name}"
        if short_option in _CURL_SHORT_VALUE_OPTIONS:
            break
        flags.append(short_option)
    return tuple(flags)


def _curl_transfer_urls(words: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = []
    index = 1
    while index < len(words):
        word = words[index]
        parsed_option = _curl_option_at(words, index)
        if parsed_option is not None:
            option, value, index = parsed_option
            if option == "--url":
                candidates.append(value)
            continue
        if not word.startswith("-"):
            candidates.append(word)
        index += 1
    return tuple(candidate for candidate in candidates if _http_transfer_target(candidate))


def _curl_has_route_override(words: tuple[str, ...]) -> bool:
    """Return whether curl may route a nominal destination through another host."""
    value_options = {
        "--connect-to",
        "--preproxy",
        "--proxy",
        "--proxy1.0",
        "--resolve",
        "--socks4",
        "--socks4a",
        "--socks5",
        "--socks5-hostname",
        "-x",
    }
    index = 1
    while index < len(words):
        word = words[index]
        if "-L" in _curl_short_flags_before_value(word):
            return True
        parsed_option = _curl_option_at(words, index)
        if parsed_option is not None:
            option, _value, index = parsed_option
            if option in value_options:
                return True
            continue
        option = word.partition("=")[0]
        if option in {"--location", "--location-trusted", "-L"}:
            return True
        index += 1
    return False


def _http_transfer_target(value: str) -> bool:
    normalized = value.casefold()
    return normalized.startswith(("http://", "https://")) or "$" in value or "%" in value


def _split_shell(source: str) -> tuple[tuple[str, str | None, int], ...]:
    """Split a bounded shell subset while ignoring quoted separators and comments."""
    result: list[tuple[str, str | None, int]] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    comment = False
    line = 1
    start_line = 1
    index = 0
    while index < len(source):
        character = source[index]
        if comment:
            if character == "\n":
                comment = False
                text = "".join(current).strip()
                if text:
                    result.append((text, None, start_line))
                current = []
                line += 1
                start_line = line
            index += 1
            continue
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            current.append(character)
            escaped = True
            index += 1
            continue
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            if character == "\n":
                line += 1
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
            index += 1
            continue
        if character == "#" and (not current or current[-1].isspace()):
            comment = True
            index += 1
            continue
        operator: str | None = None
        width = 1
        if source.startswith("&&", index):
            operator, width = "&&", 2
        elif character == "|":
            operator = "|"
        elif character == ";":
            operator = ";"
        elif character == "\n":
            operator = None
        if operator is not None or character in ";\n":
            text = "".join(current).strip()
            if text:
                result.append((text, operator, start_line))
            current = []
            if character == "\n":
                line += 1
            start_line = line
            index += width
            continue
        current.append(character)
        index += 1
    text = "".join(current).strip()
    if text:
        result.append((text, None, start_line))
    return tuple(result)


def _shell_words(segment: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(segment, comments=True, posix=True))
    except ValueError:
        return ()


def _assignment_taint(
    segment: str,
    *,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
) -> tuple[str, str] | None:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", segment, re.DOTALL)
    if match is None:
        return None
    name, expression = match.groups()
    taint = _value_taint(
        expression,
        expand_shell=True,
        variables=variables,
        profile=profile,
    )
    if taint is None and "$(" in expression:
        if any(
            _sensitive_path(token, expand_shell=True)
            for token in re.split(r"[\s<>()]+", expression)
        ):
            taint = "sensitive_local_file"
        else:
            for env_name in _environment_names(expression):
                if _sensitive_environment_name(env_name, profile):
                    taint = "ambient_credential_environment"
                    break
    return (name, taint) if taint is not None else (name, "")


def _curl_explicit_http_proxy_destination(
    words: tuple[str, ...],
) -> tuple[bool, bool, DestinationClass | None]:
    """Return explicit-proxy presence, HTTP activity, and destination."""
    configured = False
    active_http_proxy = False
    destination: DestinationClass | None = None
    bypass_all = False
    socks_options = {"--socks4", "--socks4a", "--socks5", "--socks5-hostname"}
    index = 1
    while index < len(words):
        parsed_option = _curl_option_at(words, index)
        if parsed_option is None:
            index += 1
            continue
        option, value, index = parsed_option
        if option == "--noproxy":
            bypass_all = value.strip() == "*"
            continue
        if option in socks_options:
            configured = True
            active_http_proxy = False
            destination = None
            continue
        if option not in {"-x", "--proxy", "--proxy1.0"}:
            continue
        configured = True
        normalized = value.strip()
        if not normalized:
            active_http_proxy = False
            destination = None
            continue
        if "$" in normalized or "%" in normalized:
            active_http_proxy = True
            destination = DestinationClass.DYNAMIC_UNKNOWN
            continue
        candidate = normalized if "://" in normalized else f"http://{normalized}"
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            active_http_proxy = True
            destination = DestinationClass.DYNAMIC_UNKNOWN
            continue
        active_http_proxy = parsed.scheme.casefold() in {"http", "https"}
        destination = _destination_for_host(parsed.hostname) if active_http_proxy else None
    if bypass_all:
        return True, False, None
    return configured, active_http_proxy, destination


def _curl_hit(
    words: tuple[str, ...],
    *,
    stdin_taint: str | None,
    expand_shell: bool,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
) -> _SinkHit | None:
    for group in _curl_groups(words):
        urls = _curl_transfer_urls(group)
        transfers = tuple((url, _destination_for_url(url)) for url in urls)
        route_override = _curl_has_route_override(group)
        outbound = (
            tuple((url, DestinationClass.DYNAMIC_UNKNOWN) for url, _destination in transfers)
            if route_override
            else tuple(
                (url, destination)
                for url, destination in transfers
                if destination is not DestinationClass.LOOPBACK
            )
        )
        if not outbound:
            continue
        origins = tuple(_static_http_origin(url) for url in urls)
        one_static_origin = (
            bool(origins) and None not in origins and len(set(origins)) == 1 and not route_override
        )
        origin_sources: list[str] = []
        proxy_sources: list[str] = []
        index = 1
        while index < len(group):
            parsed_option = _curl_option_at(group, index)
            if parsed_option is None:
                index += 1
                continue
            option, value, index = parsed_option
            if option not in _CURL_SOURCE_OPTIONS:
                continue
            sources = proxy_sources if option == "--proxy-header" else origin_sources
            if value in {"-", "@-"} and (
                option not in {"-H", "--header", "--proxy-header"} or value == "@-"
            ):
                if stdin_taint is not None:
                    sources.append(stdin_taint)
                continue
            auth_only_header = (
                option in {"-H", "--header"}
                and _is_authorization_header(value)
                and one_static_origin
                and _authentication_only_user_config_value(
                    value,
                    expand_shell=expand_shell,
                    profile=profile,
                )
            )
            for taint in _curl_operand_taints(
                option,
                value,
                expand_shell=expand_shell,
                variables=variables,
                profile=profile,
            ):
                if taint == "plugin_sensitive_user_config" and auth_only_header:
                    continue
                sources.append(taint)
        for url, destination in outbound:
            for taint in _value_taints(
                url,
                expand_shell=expand_shell,
                variables=variables,
                profile=profile,
                include_sensitive_path=False,
            ):
                origin_sources.append(taint)
            if origin_sources:
                return _SinkHit(origin_sources[0], TransportKind.HTTP, destination)
        if proxy_sources:
            configured, active_http_proxy, proxy_destination = (
                _curl_explicit_http_proxy_destination(group)
            )
            if not configured:
                return _SinkHit(
                    proxy_sources[0],
                    TransportKind.HTTP,
                    outbound[0][1],
                )
            if (
                active_http_proxy
                and proxy_destination is not None
                and proxy_destination is not DestinationClass.LOOPBACK
            ):
                return _SinkHit(
                    proxy_sources[0],
                    TransportKind.HTTP,
                    proxy_destination,
                )
    return None


def _unwrap_shell_command(words: tuple[str, ...]) -> tuple[str, ...]:
    """Remove supported process wrappers without joining or reparsing argv."""
    current = words
    for _hop in range(8):
        if not current:
            return ()
        executable = _normalized_executable(current[0])
        index = 1
        if executable == "env":
            value_options = {"-C", "-u", "--chdir", "--unset"}
            flag_options = {"-0", "-i", "-v", "--debug", "--ignore-environment", "--null"}
            split_command: tuple[str, ...] | None = None
            while index < len(current):
                value = current[index]
                if value == "--":
                    index += 1
                    break
                option = value.partition("=")[0]
                if option in {"-S", "--split-string"}:
                    split_value = (
                        value.partition("=")[2]
                        if "=" in value
                        else current[index + 1]
                        if index + 1 < len(current)
                        else ""
                    )
                    following_index = index + 1 if "=" in value else index + 2
                    try:
                        split_words = tuple(shlex.split(split_value, comments=False, posix=True))
                    except ValueError:
                        return ()
                    if not split_words:
                        return ()
                    split_command = (*split_words, *current[following_index:])
                    break
                if option in value_options:
                    index += 1 if "=" in value else 2
                    continue
                if (
                    value in flag_options
                    or re.fullmatch(r"-(?:C|u).+", value, re.DOTALL)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value, re.DOTALL)
                ):
                    index += 1
                    continue
                if value.startswith("-"):
                    return ()
                break
            if split_command is not None:
                current = split_command
                continue
        elif executable == "builtin":
            if index < len(current) and current[index] == "--":
                index += 1
        elif executable == "command":
            if index < len(current) and current[index] == "--":
                index += 1
            while index < len(current) and current[index] == "-p":
                index += 1
        elif executable == "nohup":
            if index < len(current) and current[index] == "--":
                index += 1
        elif executable == "sudo":
            value_options = {
                "-C",
                "-D",
                "-g",
                "-h",
                "-p",
                "-R",
                "-T",
                "-u",
                "--chdir",
                "--close-from",
                "--command-timeout",
                "--group",
                "--host",
                "--prompt",
                "--role",
                "--type",
                "--user",
            }
            flag_options = {
                "-A",
                "-b",
                "-E",
                "-e",
                "-H",
                "-i",
                "-K",
                "-k",
                "-n",
                "-P",
                "-S",
                "-s",
                "-V",
                "-v",
                "--askpass",
                "--background",
                "--edit",
                "--help",
                "--login",
                "--non-interactive",
                "--preserve-env",
                "--remove-timestamp",
                "--reset-timestamp",
                "--shell",
                "--stdin",
                "--validate",
                "--version",
            }
            while index < len(current):
                value = current[index]
                if value == "--":
                    index += 1
                    break
                if not value.startswith("-"):
                    break
                option = value.partition("=")[0]
                if option in value_options:
                    index += 1 if "=" in value else 2
                elif value in flag_options:
                    index += 1
                elif re.fullmatch(r"-(?:C|D|g|h|p|R|T|u).+", value, re.DOTALL):
                    index += 1
                else:
                    return ()
        elif executable == "timeout":
            value_options = {"-k", "-s", "--kill-after", "--signal"}
            flag_options = {"--foreground", "--preserve-status", "--verbose"}
            while index < len(current):
                value = current[index]
                if value == "--":
                    index += 1
                    break
                if not value.startswith("-"):
                    break
                option = value.partition("=")[0]
                if option in value_options:
                    index += 1 if "=" in value else 2
                elif value in flag_options:
                    index += 1
                elif re.fullmatch(r"-(?:k|s).+", value, re.DOTALL):
                    index += 1
                else:
                    return ()
            if index < len(current):
                index += 1
        elif executable == "exec":
            while index < len(current):
                value = current[index]
                if value == "--":
                    index += 1
                    break
                if value == "-a":
                    index += 2
                    continue
                if value in {"-c", "-l"}:
                    index += 1
                    continue
                if value.startswith("-"):
                    return ()
                break
        else:
            return current
        current = current[index:]
    return ()


def _option_aware_operands(
    words: tuple[str, ...],
    *,
    value_options: frozenset[str],
) -> tuple[str, ...]:
    """Return operands after a bounded leading-option parse."""
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if not word.startswith("-") or word == "-":
            break
        option = word.partition("=")[0]
        if option in value_options:
            index += 1 if "=" in word else 2
        elif any(
            option.startswith(short) and len(option) > len(short)
            for short in value_options
            if short.startswith("-") and not short.startswith("--")
        ):
            index += 1
        else:
            index += 1
    return words[index:]


def _ssh_data_bearing_option_values(
    words: tuple[str, ...],
    *,
    value_options: frozenset[str],
) -> tuple[str, ...]:
    """Return leading SSH option values that are transmitted to the server."""
    values: list[str] = []
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--" or not word.startswith("-") or word == "-":
            break
        option, equals, inline_value = word.partition("=")
        if option == "-l":
            value = inline_value if equals else (words[index + 1] if index + 1 < len(words) else "")
            if value:
                values.append(value)
            index += 1 if equals else 2
            continue
        if word.startswith("-l"):
            values.append(word[2:])
            index += 1
            continue
        if option == "-o":
            value = inline_value if equals else (words[index + 1] if index + 1 < len(words) else "")
            index += 1 if equals else 2
        elif word.startswith("-o"):
            value = word[2:].removeprefix("=")
            index += 1
        else:
            if option in value_options:
                index += 1 if equals else 2
            elif any(
                option.startswith(short) and len(option) > len(short)
                for short in value_options
                if short.startswith("-") and not short.startswith("--")
            ):
                index += 1
            else:
                index += 1
            continue
        user = re.fullmatch(r"(?i:user)(?:\s+|=)(.+)", value, re.DOTALL)
        if user is not None:
            values.append(user.group(1))
    return tuple(dict.fromkeys(values))


def _ssh_proxy_option_values(
    words: tuple[str, ...],
    *,
    value_options: frozenset[str],
    configuration_name: str,
    short_option: str | None = None,
) -> tuple[str, ...]:
    """Return leading SSH proxy option values for one configuration key."""
    values: list[str] = []
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--" or not word.startswith("-") or word == "-":
            break
        option, equals, inline_value = word.partition("=")
        if short_option is not None and option == short_option:
            value = inline_value if equals else (words[index + 1] if index + 1 < len(words) else "")
            if value:
                values.append(value)
            index += 1 if equals else 2
            continue
        if short_option is not None and word.startswith(short_option):
            value = word[len(short_option) :].removeprefix("=")
            if value:
                values.append(value)
            index += 1
            continue
        if option == "-o":
            value = inline_value if equals else (words[index + 1] if index + 1 < len(words) else "")
            index += 1 if equals else 2
        elif word.startswith("-o"):
            value = word[2:].removeprefix("=")
            index += 1
        else:
            if option in value_options:
                index += 1 if equals else 2
            elif any(
                option.startswith(short) and len(option) > len(short)
                for short in value_options
                if short.startswith("-") and not short.startswith("--")
            ):
                index += 1
            else:
                index += 1
            continue
        configured = re.fullmatch(
            rf"(?i:{re.escape(configuration_name)})(?:\s+|=)(.+)",
            value,
            re.DOTALL,
        )
        if configured is not None:
            values.append(configured.group(1))
    return tuple(values)


def _ssh_jump_host(value: str) -> str | None:
    """Extract one ProxyJump hop's host without attributing its user to the target."""
    candidate = value.strip().rsplit("@", 1)[-1]
    if candidate.startswith("["):
        closing = candidate.find("]")
        return candidate[1:closing] if closing > 1 else None
    host, separator, _port = candidate.rpartition(":")
    return host if separator and host else candidate or None


def _host_from_endpoint(value: str) -> str | None:
    """Extract a host from bracketed or conventional host:port/path syntax."""
    candidate = value.rsplit("@", 1)[-1]
    if candidate.startswith("["):
        closing = candidate.find("]")
        return candidate[1:closing] if closing > 1 else None
    host, separator, _remainder = candidate.partition(":")
    return host if separator and host else None


def _scp_remote_host(value: str) -> str | None:
    if value.casefold().startswith(("rsync://", "scp://", "ssh://")):
        try:
            return urlsplit(value).hostname
        except ValueError:
            return None
    return _host_from_endpoint(value)


def _socat_remote_host(words: tuple[str, ...]) -> str | None:
    for word in words[1:]:
        match = re.match(
            r"(?i)^(?:(?:OPENSSL|SSL|TCP|TCP4|TCP6)(?:-CONNECT)?):(.+)$",
            word,
        )
        if match is not None:
            return _host_from_endpoint(match.group(1))
    return None


def _dev_socket_redirection_host(source: str) -> str | None:
    """Return an unquoted shell redirection target's /dev/tcp or /dev/udp host."""
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(source):
        character = source[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if quote is not None:
            if character == "\\" and quote != "'":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character == "\\":
            escaped = True
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character != ">":
            index += 1
            continue
        cursor = index + 1
        if cursor < len(source) and source[cursor] == ">":
            cursor += 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        target: list[str] = []
        target_quote: str | None = None
        target_escaped = False
        while cursor < len(source):
            candidate = source[cursor]
            if target_escaped:
                target.append(candidate)
                target_escaped = False
            elif candidate == "\\" and target_quote != "'":
                target_escaped = True
            elif target_quote is not None:
                if candidate == target_quote:
                    target_quote = None
                else:
                    target.append(candidate)
            elif candidate in {"'", '"'}:
                target_quote = candidate
            elif candidate.isspace() or candidate in {";", "&", "|"}:
                break
            else:
                target.append(candidate)
            cursor += 1
        match = re.fullmatch(r"/dev/(?:tcp|udp)/([^/]+)/[^/]+", "".join(target), re.I)
        if match is not None:
            return match.group(1)
        index = max(index + 1, cursor)
    return None


def _wget_transfer_urls(words: tuple[str, ...]) -> tuple[str, ...]:
    """Collect wget transfer targets without mistaking option values for URLs."""
    value_options = {
        "-O",
        "--header",
        "--output-document",
        "--password",
        "--post-data",
        "--post-file",
        "--proxy-password",
        "--proxy-user",
        "--referer",
        "--user",
        "--user-agent",
    }
    candidates: list[str] = []
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            candidates.extend(words[index + 1 :])
            break
        option, equals, _inline_value = word.partition("=")
        if option in value_options:
            index += 1 if equals else 2
            continue
        if len(word) > 2 and word[:2] == "-O":
            index += 1
            continue
        if not word.startswith("-"):
            candidates.append(word)
        index += 1
    return tuple(candidate for candidate in candidates if _http_transfer_target(candidate))


def _cat_output_taint(
    words: tuple[str, ...],
    *,
    stdin_taint: str | None,
    expand_shell: bool,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
) -> str | None:
    """Return the source read by cat, excluding shell redirection syntax."""
    has_file_operand = False
    skip_next = False
    for word in words[1:]:
        if skip_next:
            skip_next = False
            continue
        if word in {"<", ">", ">>", "0<", "1>", "1>>", "2>", "2>>"}:
            skip_next = True
            continue
        if word.startswith(("<", ">")) or word.startswith("-"):
            continue
        has_file_operand = True
        source = _value_taint(
            word,
            expand_shell=expand_shell,
            variables=variables,
            profile=profile,
        )
        if source is not None:
            return source
    return stdin_taint if not has_file_operand else None


def _command_hit(
    words: tuple[str, ...],
    raw_segment: str,
    *,
    stdin_taint: str | None,
    expand_shell: bool,
    variables: dict[str, str],
    profile: UserConfigProfile | None,
) -> _SinkHit | None:
    if not words:
        return None
    if expand_shell:
        redirected, redirected_taint = _shell_stdin_redirection_taint(
            words,
            variables=variables,
            profile=profile,
        )
        if redirected:
            stdin_taint = redirected_taint
    words = _unwrap_shell_command(words)
    if not words:
        return None
    executable = _normalized_executable(words[0])
    if executable == "curl":
        return _curl_hit(
            words,
            stdin_taint=stdin_taint,
            expand_shell=expand_shell,
            variables=variables,
            profile=profile,
        )
    if executable == "wget":
        destination = next(
            (
                classified
                for url in _wget_transfer_urls(words)
                if (classified := _destination_for_url(url)) is not DestinationClass.LOOPBACK
            ),
            None,
        )
        if destination is None:
            return None
        index = 1
        while index < len(words):
            option, equals, inline_value = words[index].partition("=")
            if option not in {
                "--header",
                "--password",
                "--post-data",
                "--post-file",
                "--proxy-password",
                "--proxy-user",
                "--referer",
                "--user",
                "--user-agent",
            }:
                index += 1
                continue
            value = inline_value if equals else (words[index + 1] if index + 1 < len(words) else "")
            if not equals:
                index += 1
            if option == "--post-file":
                if value == "-" and stdin_taint:
                    return _SinkHit(stdin_taint, TransportKind.HTTP, destination)
                if _sensitive_path(value, expand_shell=expand_shell):
                    return _SinkHit(
                        "sensitive_local_file",
                        TransportKind.HTTP,
                        destination,
                    )
            source = _value_taint(
                value,
                expand_shell=expand_shell,
                variables=variables,
                profile=profile,
                include_sensitive_path=False,
            )
            if source:
                return _SinkHit(source, TransportKind.HTTP, destination)
            index += 1
        return None
    if executable in {"scp", "sftp", "rsync"}:
        value_options = (
            frozenset({"-b", "-c", "-D", "-F", "-i", "-J", "-l", "-o", "-P", "-S", "-X"})
            if executable == "scp"
            else frozenset(
                {"-B", "-b", "-c", "-D", "-F", "-i", "-J", "-l", "-o", "-P", "-R", "-S", "-X"}
            )
            if executable == "sftp"
            else frozenset()
        )
        operands = _option_aware_operands(
            words,
            value_options=value_options,
        )
        if len(operands) < 2:
            return None
        destination_word = operands[-1]
        remote_host = _scp_remote_host(destination_word)
        if remote_host is None:
            return None
        source = next(
            (
                taint
                for value in operands[:-1]
                if (
                    taint := _value_taint(
                        value,
                        expand_shell=expand_shell,
                        variables=variables,
                        profile=profile,
                    )
                )
            ),
            None,
        )
        if source:
            return _SinkHit(source, TransportKind.SSH, _destination_for_host(remote_host))
        return None
    if executable in {"nc", "ncat", "netcat", "socat", "ssh", "mail", "mailx"}:
        if executable in {"mail", "mailx"}:
            source = stdin_taint or next(
                (
                    taint
                    for value in words[1:]
                    if (
                        taint := _value_taint(
                            value,
                            expand_shell=expand_shell,
                            variables=variables,
                            profile=profile,
                        )
                    )
                ),
                None,
            )
            return (
                _SinkHit(source, TransportKind.MAIL, DestinationClass.PUBLIC_REMOTE)
                if source
                else None
            )
        value_options = (
            frozenset(
                {
                    "-B",
                    "-b",
                    "-c",
                    "-D",
                    "-E",
                    "-e",
                    "-F",
                    "-i",
                    "-J",
                    "-L",
                    "-l",
                    "-m",
                    "-O",
                    "-o",
                    "-p",
                    "-Q",
                    "-R",
                    "-S",
                    "-W",
                    "-w",
                }
            )
            if executable == "ssh"
            else frozenset({"-P", "-X", "-i", "-p", "-q", "-s", "-w", "-x"})
        )
        operands = _option_aware_operands(words, value_options=value_options)
        sink_host = operands[0] if operands else None
        if executable == "socat":
            sink_host = _socat_remote_host(words)
        if executable == "ssh":
            for proxy_command in _ssh_proxy_option_values(
                words,
                value_options=value_options,
                configuration_name="ProxyCommand",
            ):
                if proxy_command.strip().casefold() == "none":
                    continue
                proxy_hits = _analyze_shell(
                    proxy_command,
                    event_taint=None,
                    profile=profile,
                    variables=variables,
                )
                if proxy_hits:
                    return proxy_hits[0]

            jump_words = words
            if expand_shell:
                jump_words = _unwrap_shell_command(
                    _shell_words(_mask_inert_shell_text(raw_segment))
                )
            for proxy_jump in _ssh_proxy_option_values(
                jump_words,
                value_options=value_options,
                configuration_name="ProxyJump",
                short_option="-J",
            ):
                for jump_hop in proxy_jump.split(","):
                    jump_source = _value_taint(
                        jump_hop,
                        expand_shell=expand_shell,
                        variables=variables,
                        profile=profile,
                        include_sensitive_path=False,
                    )
                    if jump_source is not None:
                        jump_destination = _destination_for_host(_ssh_jump_host(jump_hop))
                        if jump_destination is DestinationClass.LOOPBACK:
                            continue
                        return _SinkHit(
                            jump_source,
                            TransportKind.SSH,
                            jump_destination,
                        )
        source = stdin_taint
        if source is None and executable == "ssh":
            source_words = words
            if expand_shell:
                source_words = _unwrap_shell_command(
                    _shell_words(_mask_inert_shell_text(raw_segment))
                )
            source_values = (
                *_option_aware_operands(source_words, value_options=value_options),
                *_ssh_data_bearing_option_values(
                    source_words,
                    value_options=value_options,
                ),
            )
            source = next(
                (
                    taint
                    for value in source_values
                    if (
                        taint := _value_taint(
                            value,
                            expand_shell=expand_shell,
                            variables=variables,
                            profile=profile,
                            include_sensitive_path=False,
                        )
                    )
                ),
                None,
            )
            if expand_shell:
                if source is None:
                    for substitution in re.finditer(
                        r"\$\(([^()]*)\)", " ".join(source_values), re.DOTALL
                    ):
                        substitution_words = _unwrap_shell_command(
                            _shell_words(substitution.group(1))
                        )
                        if (
                            not substitution_words
                            or _normalized_executable(substitution_words[0]) != "cat"
                        ):
                            continue
                        _redirected, substitution_stdin_taint = _shell_stdin_redirection_taint(
                            substitution_words,
                            variables=variables,
                            profile=profile,
                        )
                        source = _cat_output_taint(
                            substitution_words,
                            stdin_taint=substitution_stdin_taint,
                            expand_shell=True,
                            variables=variables,
                            profile=profile,
                        )
                        if source is not None:
                            break
        if source is None:
            return None
        return _SinkHit(
            source,
            TransportKind.SSH if executable == "ssh" else TransportKind.TCP,
            _destination_for_host(sink_host),
        )
    if executable in {"dig", "host", "nslookup"}:
        source = _value_taint(
            raw_segment,
            expand_shell=expand_shell,
            variables=variables,
            profile=profile,
        )
        if source is None and any(
            _sensitive_path(token, expand_shell=expand_shell)
            for token in re.split(r"[\s<>()]+", raw_segment)
        ):
            source = "sensitive_local_file"
        return (
            _SinkHit(source, TransportKind.DNS, DestinationClass.PUBLIC_REMOTE) if source else None
        )
    if executable == "rclone":
        rclone_operands = _option_aware_operands(
            words,
            value_options=frozenset({"--config"}),
        )
        operation_index = next(
            (
                index
                for index, value in enumerate(rclone_operands)
                if value in {"copy", "copyto", "move", "moveto", "sync"}
            ),
            None,
        )
        operation_operands = (
            rclone_operands[operation_index + 1 :] if operation_index is not None else ()
        )
        destination_word = operation_operands[1] if len(operation_operands) >= 2 else ""
        remote_destination = bool(
            re.fullmatch(r"[A-Za-z0-9_.-]+:.+", destination_word)
            and not re.match(r"^[A-Za-z]:[/\\]", destination_word)
        )
        source_candidates = [
            inline_value if equals else words[index + 1]
            for index, word in enumerate(words[:-1])
            for option, equals, inline_value in (word.partition("="),)
            if option == "--config"
        ]
        source_candidates.extend(operation_operands[:1])
        source = next(
            (
                taint
                for value in source_candidates
                if (
                    taint := _value_taint(
                        value,
                        expand_shell=expand_shell,
                        variables=variables,
                        profile=profile,
                    )
                )
            ),
            None,
        )
        if source and remote_destination:
            return _SinkHit(
                source,
                TransportKind.OBJECT_STORE,
                DestinationClass.PUBLIC_REMOTE,
            )
    if executable == "aws":
        operands = _option_aware_operands(
            words,
            value_options=frozenset(
                {
                    "--ca-bundle",
                    "--cli-connect-timeout",
                    "--cli-read-timeout",
                    "--color",
                    "--endpoint-url",
                    "--output",
                    "--profile",
                    "--region",
                }
            ),
        )
    else:
        operands = ()
    if len(operands) >= 4 and operands[0] == "s3" and operands[1] in {"cp", "mv", "sync"}:
        source = _value_taint(
            operands[2],
            expand_shell=expand_shell,
            variables=variables,
            profile=profile,
        )
        if source and operands[3].casefold().startswith("s3://"):
            return _SinkHit(source, TransportKind.OBJECT_STORE, DestinationClass.PUBLIC_REMOTE)
    if executable == "gcloud":
        operands = _option_aware_operands(
            words,
            value_options=frozenset(
                {
                    "--account",
                    "--billing-project",
                    "--configuration",
                    "--project",
                }
            ),
        )
        if len(operands) >= 4 and operands[:2] == ("storage", "cp"):
            source = _value_taint(
                operands[2],
                expand_shell=expand_shell,
                variables=variables,
                profile=profile,
            )
            if source and operands[3].casefold().startswith("gs://"):
                return _SinkHit(
                    source,
                    TransportKind.OBJECT_STORE,
                    DestinationClass.PUBLIC_REMOTE,
                )
    if executable == "az":
        operands = _option_aware_operands(
            words,
            value_options=frozenset({"--subscription"}),
        )
        if len(operands) >= 3 and operands[:3] == ("storage", "blob", "upload"):
            source_value: str | None = None
            for index, value in enumerate(operands[3:]):
                option, equals, inline_value = value.partition("=")
                if option not in {"--file", "-f"}:
                    continue
                absolute_index = index + 3
                source_value = (
                    inline_value
                    if equals
                    else operands[absolute_index + 1]
                    if absolute_index + 1 < len(operands)
                    else None
                )
                break
            source = (
                _value_taint(
                    source_value,
                    expand_shell=expand_shell,
                    variables=variables,
                    profile=profile,
                )
                if source_value is not None
                else None
            )
            if source:
                return _SinkHit(
                    source,
                    TransportKind.OBJECT_STORE,
                    DestinationClass.PUBLIC_REMOTE,
                )
    dev_socket_host = _dev_socket_redirection_host(raw_segment)
    if dev_socket_host is not None:
        source = stdin_taint
        if executable == "cat":
            source = _cat_output_taint(
                words,
                stdin_taint=stdin_taint,
                expand_shell=expand_shell,
                variables=variables,
                profile=profile,
            )
        if source is not None:
            return _SinkHit(
                source,
                TransportKind.TCP,
                _destination_for_host(dev_socket_host),
            )
    return None


def _nested_shell(command: str, args: tuple[str, ...]) -> str | None:
    executable = _normalized_executable(command)
    options = (
        {"-c"}
        if executable in {"bash", "sh", "zsh"}
        else {"-command", "-c"}
        if executable in {"powershell", "pwsh"}
        else {"/c"}
        if executable == "cmd"
        else set()
    )
    for index, value in enumerate(args[:-1]):
        lowered = value.lower()
        clustered_posix_command = bool(
            executable in {"bash", "sh", "zsh"} and re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", value)
        )
        if lowered in options or clustered_posix_command:
            return args[index + 1]
    return None


def _analyze_shell(
    source: str,
    *,
    event_taint: str | None,
    profile: UserConfigProfile | None,
    variables: dict[str, str] | None = None,
) -> list[_SinkHit]:
    hits: list[_SinkHit] = []
    variables = dict(variables or {})
    pipeline_taint: str | None = None
    pipeline_active = False
    for segment, following_operator, line in _split_shell(source):
        words = _shell_words(segment)
        command_variables = variables
        if not words:
            pipeline_taint = None
            pipeline_active = False
            continue
        if words[0] == "export" and len(words) > 1:
            exported = tuple(
                assignment
                for value in words[1:]
                if (assignment := _assignment_taint(value, variables=variables, profile=profile))
                is not None
            )
            if len(exported) == len(words) - 1:
                for name, taint in exported:
                    if taint:
                        variables[name] = taint
                    else:
                        variables.pop(name, None)
                pipeline_taint = None
                pipeline_active = False
                continue
        leading_assignments: list[tuple[str, str]] = []
        command_index = 0
        while command_index < len(words):
            assignment = _assignment_taint(
                words[command_index],
                variables=variables,
                profile=profile,
            )
            if assignment is None:
                break
            leading_assignments.append(assignment)
            command_index += 1
        if leading_assignments and command_index < len(words):
            command_variables = dict(variables)
            for name, taint in leading_assignments:
                if taint:
                    command_variables[name] = taint
                else:
                    command_variables.pop(name, None)
            words = words[command_index:]
        else:
            assignment = _assignment_taint(segment, variables=variables, profile=profile)
            if assignment is not None:
                name, taint = assignment
                if taint:
                    variables[name] = taint
                else:
                    variables.pop(name, None)
                pipeline_taint = None
                pipeline_active = False
                continue
        effective_words = _unwrap_shell_command(words)
        if not effective_words:
            pipeline_taint = None
            pipeline_active = False
            continue
        executable = _normalized_executable(effective_words[0])
        nested = _nested_shell(effective_words[0], effective_words[1:])
        stdin_taint = pipeline_taint if pipeline_active else event_taint
        if nested is not None:
            for nested_hit in _analyze_shell(
                nested,
                event_taint=stdin_taint,
                profile=profile,
                variables=command_variables,
            ):
                hits.append(
                    _SinkHit(
                        nested_hit.source_kind,
                        nested_hit.transport,
                        nested_hit.destination,
                        line,
                    )
                )
        else:
            command_hit = _command_hit(
                effective_words,
                segment,
                stdin_taint=stdin_taint,
                expand_shell=True,
                variables=command_variables,
                profile=profile,
            )
            if command_hit is not None and command_hit.destination is not DestinationClass.LOOPBACK:
                hits.append(
                    _SinkHit(
                        command_hit.source_kind,
                        command_hit.transport,
                        command_hit.destination,
                        line,
                    )
                )
        output_taint: str | None = None
        if executable == "cat":
            if len(effective_words) == 1:
                output_taint = stdin_taint
            else:
                output_taint = next(
                    (
                        _value_taint(
                            word,
                            expand_shell=True,
                            variables=command_variables,
                            profile=profile,
                        )
                        for word in effective_words[1:]
                        if _value_taint(
                            word,
                            expand_shell=True,
                            variables=command_variables,
                            profile=profile,
                        )
                    ),
                    None,
                )
        elif executable == "jq" and "transcript_path" in segment:
            output_taint = None
        elif executable in {"echo", "printf"}:
            output_taint = _value_taint(
                segment,
                expand_shell=True,
                variables=command_variables,
                profile=profile,
            )
        elif following_operator == "|":
            output_taint = stdin_taint
        pipeline_active = following_operator == "|"
        pipeline_taint = output_taint if pipeline_active else None
    return hits


def _analyze_command(
    handler: HandlerFlowInput,
    *,
    event_taint: str | None,
    profile: UserConfigProfile | None,
) -> list[_SinkHit]:
    if handler.command is None:
        return []
    if handler.args is None:
        return _analyze_shell(handler.command, event_taint=event_taint, profile=profile)
    words = _unwrap_shell_command((handler.command, *handler.args))
    if not words:
        return []
    nested = _nested_shell(words[0], words[1:])
    if nested is not None:
        return _analyze_shell(nested, event_taint=event_taint, profile=profile)
    hit = _command_hit(
        words,
        "",
        stdin_taint=event_taint,
        expand_shell=False,
        variables={},
        profile=profile,
    )
    return [hit] if hit is not None and hit.destination is not DestinationClass.LOOPBACK else []


def _plugin_source(source_kind: str) -> bool:
    return source_kind.startswith("plugin_") or source_kind.startswith("marketplace_plugin_")


def _references_in_text(source: str) -> tuple[_Reference, ...]:
    references: list[_Reference] = []
    previous_offset = 0
    line = 1
    for match in _BUNDLE_REFERENCE.finditer(source):
        scope, relative = match.groups()
        line += source.count("\n", previous_offset, match.start())
        previous_offset = match.start()
        references.append(_Reference(scope.lower(), relative, line))
    previous_offset = 0
    line = 1
    for match in _CD_BUNDLE_REFERENCE.finditer(source):
        braced_scope, plain_scope, relative = match.groups()
        scope = braced_scope or plain_scope
        line += source.count("\n", previous_offset, match.start())
        previous_offset = match.start()
        references.append(_Reference(scope.lower(), relative, line))
    return tuple(dict.fromkeys(references))


def _reference_from_token(value: str, line: int) -> _Reference | None:
    match = _BUNDLE_REFERENCE.fullmatch(value)
    if match is None:
        return None
    scope, relative = match.groups()
    return _Reference(scope.lower(), relative, line)


def _shell_entrypoint_references(source: str, *, depth: int = 0) -> tuple[_Reference, ...]:
    references: list[_Reference] = []
    pending_root_scope: str | None = None
    for segment, following_operator, line in _split_shell(source):
        words = _shell_words(segment)
        if not words:
            pending_root_scope = None
            continue
        original_executable = _normalized_executable(words[0])
        if original_executable in {".", "source", "exec"} and len(words) > 1:
            original_operand = words[1]
            if ("$" in original_operand or "%" in original_operand) and _BUNDLE_REFERENCE.fullmatch(
                original_operand
            ) is None:
                pending_root_scope = None
                continue
        effective_words = _unwrap_shell_command(words)
        if not effective_words:
            pending_root_scope = None
            continue
        executable = effective_words[0]
        arguments = effective_words[1:]
        normalized = _normalized_executable(executable)
        direct = _reference_from_token(executable, line)
        if direct is not None:
            references.append(direct)
            pending_root_scope = None
            continue
        if normalized == "cd" and arguments:
            root_match = re.fullmatch(
                r"\$(?:\{CLAUDE_(PLUGIN_ROOT|PROJECT_DIR)\}|"
                r"CLAUDE_(PLUGIN_ROOT|PROJECT_DIR))/?",
                arguments[0],
            )
            pending_root_scope = (
                (root_match.group(1) or root_match.group(2)).lower()
                if root_match is not None and following_operator == "&&"
                else None
            )
            continue
        if pending_root_scope is not None:
            candidates = (
                arguments
                if normalized in {"bash", "node", "python", "python3", "sh", "zsh"}
                else (executable,)
            )
            relative = next(
                (
                    value.removeprefix("./")
                    for value in candidates
                    if value
                    and not value.startswith("-")
                    and not value.startswith("/")
                    and "${" not in value
                    and "$" not in value
                ),
                None,
            )
            if relative is not None:
                references.append(_Reference(pending_root_scope, relative, line))
            pending_root_scope = None
        nested = _nested_shell(executable, arguments)
        if nested is not None and depth < _MAX_WRAPPER_HOPS:
            references.extend(_shell_entrypoint_references(nested, depth=depth + 1))
            continue
        operand: str | None = None
        if normalized in {".", "exec", "source", "node", "python", "python3"}:
            operand = next((value for value in arguments if not value.startswith("-")), None)
        elif normalized in {"bash", "sh", "zsh"}:
            operand = next((value for value in arguments if not value.startswith("-")), None)
        if operand is not None and (reference := _reference_from_token(operand, line)):
            references.append(reference)
    return tuple(dict.fromkeys(references))


def _mask_inert_shell_text(source: str) -> str:
    """Mask single-quoted and escaped shell text while preserving executable regions."""
    output = list(source)
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote == "'":
            if character == "'":
                quote = None
            if character != "\n":
                output[index] = " "
            continue
        if escaped:
            if character != "\n":
                output[index] = " "
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote == '"':
            if character == '"':
                quote = None
            continue
        if character == "'":
            output[index] = " "
            quote = "'"
        elif character == '"':
            quote = '"'
    return "".join(output)


def _shell_payload_unmodeled(source: str) -> bool:
    """Reject reachable shell grammar outside the supported simple-command subset."""
    control_words = {
        "case",
        "do",
        "done",
        "elif",
        "else",
        "esac",
        "fi",
        "for",
        "function",
        "if",
        "select",
        "then",
        "until",
        "while",
    }
    for segment, _operator, _line in _split_shell(source):
        executable_text = _mask_inert_shell_text(segment)
        if "`" in executable_text:
            return True
        for substitution in re.finditer(r"\$\(([^()]*)\)", executable_text, re.DOTALL):
            substituted_words = _shell_words(substitution.group(1))
            if not substituted_words or _normalized_executable(substituted_words[0]) != "cat":
                return True
        words = _shell_words(segment)
        if not words:
            continue
        command_words = words
        while command_words and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*",
            command_words[0],
            re.DOTALL,
        ):
            command_words = command_words[1:]
        if not command_words:
            continue
        normalized = _normalized_executable(command_words[0])
        if normalized in control_words or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", segment):
            return True
        if normalized == "eval":
            return True
        if normalized in {".", "exec", "source"} and len(command_words) > 1:
            operand = command_words[1]
            if normalized in {".", "source"} and operand.startswith(("./", "../")):
                return True
            if ("$" in operand or "%" in operand) and _BUNDLE_REFERENCE.fullmatch(operand) is None:
                return True
        effective = _unwrap_shell_command(command_words)
        if not effective:
            return True
        effective_executable = effective[0]
        if ("$" in effective_executable or "%" in effective_executable) and (
            _BUNDLE_REFERENCE.fullmatch(effective_executable) is None
        ):
            return True
        nested = _nested_shell(effective[0], effective[1:])
        if nested is not None and _shell_payload_unmodeled(nested):
            return True
        executable = _normalized_executable(effective[0])
        if executable in {"python", "python3"} and any(
            value == "-c" or value.startswith("-c") for value in effective[1:]
        ):
            return True
        if executable == "node" and any(
            value in {"-e", "--eval"} or value.startswith("-e=") or value.startswith("--eval=")
            for value in effective[1:]
        ):
            return True
    return False


def _handler_references(handler: HandlerFlowInput) -> tuple[_Reference, ...]:
    references: list[_Reference] = []
    if handler.args is None and handler.command is not None:
        references.extend(_shell_entrypoint_references(handler.command))
    for encoded in handler.registration.entrypoint_references:
        scope, separator, relative = encoded.partition(":")
        if separator:
            references.append(_Reference(scope, relative))
    return tuple(dict.fromkeys(references))


def _unsafe_entrypoint(handler: HandlerFlowInput) -> bool:
    sources = tuple(
        value for value in (handler.command, *(handler.args or ())) if isinstance(value, str)
    )
    if any("\x00" in value for value in sources):
        return True
    joined = "\n".join(sources)
    if "${CLAUDE_PLUGIN_DATA}" in joined:
        return True
    for reference in _references_in_text(joined):
        relative = reference.relative
        if (
            "${" in relative
            or "!/" in relative
            or "\\" in relative
            or relative.startswith("/")
            or any(part == ".." for part in relative.split("/"))
        ):
            return True
    command = handler.command or ""
    if handler.args is not None:
        effective_words = _unwrap_shell_command((command, *handler.args))
        effective_executable = effective_words[0] if effective_words else command
        if ("$" in effective_executable or "%" in effective_executable) and (
            _BUNDLE_REFERENCE.fullmatch(effective_executable) is None
        ):
            return True
        executable = _normalized_executable(command)
        if executable == "eval":
            return True
        if executable in {".", "source"} and handler.args:
            operand = handler.args[0]
            if ("$" in operand or "%" in operand) and _BUNDLE_REFERENCE.fullmatch(operand) is None:
                return True
        if executable in {"python", "python3"} and any(
            value == "-c" or value.startswith("-c") for value in handler.args
        ):
            return True
        if executable == "node" and any(
            value in {"-e", "--eval"} or value.startswith("-e=") or value.startswith("--eval=")
            for value in handler.args
        ):
            return True
        nested = _nested_shell(command, handler.args)
        if nested is not None:
            return _unsafe_entrypoint(
                HandlerFlowInput(registration=handler.registration, command=nested)
            )
        for value in (command, *handler.args):
            if "${CLAUDE_" not in value:
                continue
            matches = tuple(_BUNDLE_REFERENCE.finditer(value))
            if len(matches) != 1 or matches[0].start() != 0 or matches[0].end() != len(value):
                return True
        if _normalized_executable(command) in {
            "bash",
            "node",
            "python",
            "python3",
            "sh",
            "zsh",
        }:
            interpreter_operand = next(
                (value for value in handler.args if not value.startswith("-")),
                None,
            )
            if interpreter_operand is not None and "${CLAUDE_" not in interpreter_operand:
                if (
                    interpreter_operand.startswith(("./", "/", "\\\\"))
                    or re.match(r"^[A-Za-z]:[\\/]", interpreter_operand)
                    or "/" in interpreter_operand
                    or "\\" in interpreter_operand
                ):
                    return True
        return False
    if handler.args is None:
        if _shell_payload_unmodeled(command):
            return True
        if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", command.strip()):
            return True
        documented_cd_targets = {
            target
            for reference in _references_in_text(command)
            for target in (reference.relative, f"./{reference.relative}")
        }
        segments = _split_shell(command)
        for segment, _operator, _line in segments:
            words = _shell_words(segment)
            if not words:
                continue
            executable = words[0]
            if executable in documented_cd_targets:
                continue
            normalized = _normalized_executable(executable)
            if normalized == "eval":
                return True
            if normalized in {".", "source"} and len(words) > 1:
                if ("$" in words[1] or "%" in words[1]) and _BUNDLE_REFERENCE.fullmatch(
                    words[1]
                ) is None:
                    return True
            if normalized in {"python", "python3"} and any(
                value == "-c" or value.startswith("-c") for value in words[1:]
            ):
                return True
            if normalized == "node" and any(
                value in {"-e", "--eval"} or value.startswith("-e=") or value.startswith("--eval=")
                for value in words[1:]
            ):
                return True
            if normalized in {
                "curl",
                "wget",
                "scp",
                "sftp",
                "rsync",
                "nc",
                "ncat",
                "netcat",
                "socat",
                "ssh",
                "mail",
                "mailx",
                "dig",
                "host",
                "nslookup",
                "aws",
                "cat",
                "echo",
                "printf",
                "jq",
                "npm",
                "npx",
                "cp",
                "cd",
                "source",
                ".",
                "python",
                "python3",
                "node",
                "bash",
                "sh",
                "zsh",
                "powershell",
                "pwsh",
                "cmd",
            }:
                continue
            if (
                executable.startswith(("./", "/", "\\\\"))
                or re.match(r"^[A-Za-z]:[\\/]", executable)
                or ("/" in executable and not executable.startswith("${CLAUDE_"))
            ):
                return True
    return False


def _key_parts(path: str) -> tuple[str, tuple[str, ...]]:
    if "!/" in path:
        archive, member = path.rsplit("!/", 1)
        return f"{archive}!/", tuple(part for part in member.split("/") if part)
    return "", tuple(part for part in path.split("/") if part)


def _resolved_reference(
    registration: HookRegistration,
    reference: _Reference,
    *,
    base_path: str | None = None,
) -> str | None:
    if registration.execution_root is None:
        return None
    if reference.relative == "invalid":
        return None
    if reference.scope == "component":
        if base_path is None:
            return None
        relative = reference.relative
        if (
            not relative.startswith(("./", "../"))
            or "\x00" in relative
            or "\\" in relative
            or "!/" in relative
            or relative.startswith("/")
        ):
            return None
        root_namespace, root_parts = _key_parts(registration.execution_root)
        base_namespace, base_parts = _key_parts(base_path)
        if root_namespace != base_namespace or base_parts[: len(root_parts)] != root_parts:
            return None
        resolved_parts = list(base_parts[:-1])
        for part in relative.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                if len(resolved_parts) <= len(root_parts):
                    return None
                resolved_parts.pop()
                continue
            if len(part) > 1 and part[1] == ":":
                return None
            resolved_parts.append(part)
        if not resolved_parts or tuple(resolved_parts[: len(root_parts)]) != root_parts:
            return None
        member = "/".join(resolved_parts)
        return f"{root_namespace}{member}" if root_namespace else member
    if reference.scope == "plugin_root" and not _plugin_source(registration.source_kind):
        return None
    if reference.scope == "project_dir" and _plugin_source(registration.source_kind):
        return None
    raw_relative = reference.relative
    relative = raw_relative.strip("/")
    parts = tuple(part for part in relative.split("/") if part not in {"", "."})
    if (
        not parts
        or "\x00" in relative
        or "\\" in relative
        or "!/" in relative
        or raw_relative.startswith("/")
        or any(part == ".." or (len(part) > 1 and part[1] == ":") for part in parts)
        or any("${" in part for part in parts)
    ):
        return None
    namespace, root_parts = _key_parts(registration.execution_root)
    member = "/".join((*root_parts, *parts))
    return f"{namespace}{member}" if namespace else member


def _python_call_name(node: ast.Call, aliases: dict[str, str]) -> str | None:
    name = resolve_dotted_name(node.func)
    return apply_import_aliases(name, aliases) if name is not None else None


def _python_string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _python_environment_taint(
    node: ast.expr,
    aliases: dict[str, str],
    profile: UserConfigProfile | None,
) -> str | None:
    name: str | None = None
    if isinstance(node, ast.Subscript):
        base = resolve_dotted_name(node.value)
        if base is not None and apply_import_aliases(base, aliases) == "os.environ":
            name = _python_string(node.slice)
    elif isinstance(node, ast.Call):
        call_name = _python_call_name(node, aliases)
        if call_name in {"os.getenv", "os.environ.get"}:
            name = _python_string(node.args[0]) if node.args else None
    if name is None or not _sensitive_environment_name(name, profile):
        return None
    if profile and name in profile.sensitive_environment_names:
        return "plugin_sensitive_user_config"
    return "ambient_credential_environment"


def _python_sensitive_file_read(node: ast.expr, aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    call_name = _python_call_name(node, aliases)
    if call_name == "open" and node.args:
        path = _python_string(node.args[0])
        return path is not None and _sensitive_path(path, expand_shell=False)
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in {"read", "read_text", "read_bytes"}:
        return False
    receiver = node.func.value
    if not isinstance(receiver, ast.Call):
        return False
    receiver_name = _python_call_name(receiver, aliases)
    if receiver_name not in {"open", "pathlib.Path"} or not receiver.args:
        return False
    path = _python_string(receiver.args[0])
    return path is not None and _sensitive_path(path, expand_shell=False)


def _python_expr_taint(
    node: ast.expr,
    *,
    aliases: dict[str, str],
    variables: dict[str, str],
    event_taint: str | None,
    profile: UserConfigProfile | None,
) -> str | None:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in variables:
            return variables[child.id]
        if not isinstance(child, ast.expr):
            continue
        environment = _python_environment_taint(child, aliases, profile)
        if environment is not None:
            return environment
        if _python_sensitive_file_read(child, aliases):
            return "sensitive_local_file"
        if isinstance(child, ast.Call):
            call_name = _python_call_name(child, aliases)
            if call_name in {"sys.stdin.read", "sys.stdin.readline"} and event_taint:
                return event_taint
            if call_name == "json.load" and child.args and event_taint:
                source_name = resolve_dotted_name(child.args[0])
                if (
                    source_name is not None
                    and apply_import_aliases(source_name, aliases) == "sys.stdin"
                ):
                    return event_taint
    return None


def _python_targets(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets: tuple[ast.expr, ...]
    if isinstance(node, ast.Assign):
        targets = tuple(node.targets)
    else:
        targets = (node.target,)
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return tuple(names)


def _python_sink_arguments(node: ast.Call) -> tuple[ast.expr, ...]:
    return (*node.args, *(keyword.value for keyword in node.keywords))


def _python_destination(node: ast.Call, sink_name: str) -> DestinationClass:
    positional_index = 1 if sink_name == "requests.request" else 0
    url_node: ast.expr | None = (
        node.args[positional_index] if len(node.args) > positional_index else None
    )
    for keyword in node.keywords:
        if keyword.arg == "url":
            url_node = keyword.value
            break
    return _destination_for_url(_python_string(url_node))


def _python_is_unmodeled(node: ast.Call, aliases: dict[str, str]) -> bool:
    name = _python_call_name(node, aliases)
    if name is not None and name.split(".", 1)[0] in {"socket", "urllib3"}:
        return True
    if name in {"eval", "exec", "compile"}:
        return True
    if name in {"__import__", "importlib.import_module"}:
        return not node.args or _python_string(node.args[0]) is None
    if name in {
        "os.system",
        "os.popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }:
        return True
    if name in {
        "httpx.delete",
        "httpx.head",
        "httpx.options",
        "httpx.request",
        "requests.delete",
        "requests.head",
        "requests.options",
    }:
        return True
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
        receiver_name = _python_call_name(node.func.value, aliases)
        if receiver_name in {"httpx.AsyncClient", "httpx.Client", "requests.Session"}:
            return True
    return False


def _analyze_python_payload(
    content: str,
    path: str,
    *,
    event_taint: str | None,
    profile: UserConfigProfile | None,
    python_ast_cache_key: str | None,
) -> tuple[list[_SinkHit], bool]:
    parsed = get_python_ast(python_ast_cache_key, content, path)
    if parsed.tree is None:
        return [], True
    unsupported_nodes = (
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.AsyncWith,
        ast.ClassDef,
        ast.For,
        ast.FunctionDef,
        ast.If,
        ast.Lambda,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
        ast.comprehension,
    )
    if any(isinstance(node, unsupported_nodes) for node in ast.walk(parsed.tree)):
        return [], True
    aliases = parsed.import_aliases
    calls = tuple(node for node in ast.walk(parsed.tree) if isinstance(node, ast.Call))
    if any(_python_is_unmodeled(node, aliases) for node in calls):
        return [], True
    session_variables: set[str] = set()
    for assignment in ast.walk(parsed.tree):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        value = assignment.value
        if not isinstance(value, ast.Call):
            continue
        if _python_call_name(value, aliases) not in {
            "httpx.AsyncClient",
            "httpx.Client",
            "requests.Session",
        }:
            continue
        session_variables.update(_python_targets(assignment))
    if any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in session_variables
        and call.func.attr
        in {"delete", "get", "head", "options", "patch", "post", "put", "request", "stream"}
        for call in calls
    ):
        return [], True
    relevant_nodes = tuple(
        sorted(
            (
                node
                for node in ast.walk(parsed.tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Call))
            ),
            key=lambda node: (
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0),
                0 if isinstance(node, (ast.Assign, ast.AnnAssign)) else 1,
            ),
        )
    )
    variables: dict[str, str] = {}
    hits: list[_SinkHit] = []
    network_sinks = {
        "httpx.get",
        "httpx.patch",
        "httpx.post",
        "httpx.put",
        "requests.get",
        "requests.patch",
        "requests.post",
        "requests.put",
        "requests.request",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
    }
    for node in relevant_nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            taint = (
                _python_expr_taint(
                    value,
                    aliases=aliases,
                    variables=variables,
                    event_taint=event_taint,
                    profile=profile,
                )
                if value is not None
                else None
            )
            for target in _python_targets(node):
                if taint is None:
                    variables.pop(target, None)
                else:
                    variables[target] = taint
            continue
        sink_name = _python_call_name(node, aliases)
        if sink_name not in network_sinks:
            continue
        source = next(
            (
                taint
                for argument in _python_sink_arguments(node)
                if (
                    taint := _python_expr_taint(
                        argument,
                        aliases=aliases,
                        variables=variables,
                        event_taint=event_taint,
                        profile=profile,
                    )
                )
            ),
            None,
        )
        destination = _python_destination(node, sink_name)
        if source is not None and destination is not DestinationClass.LOOPBACK:
            hits.append(_SinkHit(source, TransportKind.HTTP, destination, node.lineno))
    return hits, False


def _strip_javascript_comments(source: str) -> tuple[str, bool]:
    output = list(source)
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = 0
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if character == "\n":
                line_comment = False
            else:
                output[index] = " "
            index += 1
            continue
        if block_comment:
            if character == "*" and following == "/":
                output[index] = output[index + 1] = " "
                block_comment = False
                index += 2
            else:
                if character != "\n":
                    output[index] = " "
                index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"', "`"}:
            quote = character
            index += 1
            continue
        if character == "/" and following == "/":
            output[index] = output[index + 1] = " "
            line_comment = True
            index += 2
            continue
        if character == "/" and following == "*":
            output[index] = output[index + 1] = " "
            block_comment = True
            index += 2
            continue
        index += 1
    return "".join(output), quote is None and not block_comment


def _javascript_statements(source: str) -> tuple[tuple[str, int], ...]:
    statements: list[tuple[str, int]] = []
    start = 0
    start_line = 1
    current_line = 1
    quote: str | None = None
    escaped = False
    depth = 0
    for index, character in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {"'", '"', "`"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == ";" and depth == 0:
            raw_statement = source[start:index]
            statement = raw_statement.strip()
            if statement:
                leading = len(raw_statement) - len(raw_statement.lstrip())
                statements.append((statement, start_line + raw_statement[:leading].count("\n")))
            start = index + 1
            start_line = current_line
        if character == "\n":
            if quote is None and depth == 0:
                raw_statement = source[start:index]
                statement = raw_statement.strip()
                if statement:
                    leading = len(raw_statement) - len(raw_statement.lstrip())
                    statements.append((statement, start_line + raw_statement[:leading].count("\n")))
                start = index + 1
                start_line = current_line + 1
            current_line += 1
    raw_statement = source[start:]
    statement = raw_statement.strip()
    if statement:
        leading = len(raw_statement) - len(raw_statement.lstrip())
        statements.append((statement, start_line + raw_statement[:leading].count("\n")))
    return tuple(statements)


def _mask_javascript_strings(source: str) -> str:
    output = list(source)
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote is not None:
            if character != "\n":
                output[index] = " "
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            output[index] = " "
            quote = character
    return "".join(output)


def _javascript_structure_valid(source: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for character in source:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}" and (not stack or stack.pop() != pairs[character]):
            return False
    return quote is None and not stack


def _javascript_literal(value: str) -> str | None:
    value = value.strip()
    if len(value) < 2 or value[0] not in {"'", '"', "`"} or value[-1] != value[0]:
        return None
    if value[0] == "`" and "${" in value:
        return None
    return value[1:-1]


def _javascript_first_argument(arguments: str) -> str:
    quote: str | None = None
    escaped = False
    depth = 0
    for index, character in enumerate(arguments):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {"'", '"', "`"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            return arguments[:index].strip()
    return arguments.strip()


_JAVASCRIPT_HTTP_METHOD_CALL: Final[str] = (
    r"(?<![\w$.])(?:(?P<fetch>(?:(?:globalThis|global)\s*\.\s*)?fetch)|"
    r"(?P<client>[A-Za-z_$][\w$]*)\s*(?:\?\.\s*|\.\s*)"
    r"(?P<method>get|patch|post|put|delete|head|options|request|patchForm|postForm|putForm))"
    r"\s*(?:\?\.\s*)?\("
)
_JAVASCRIPT_CALLABLE_CALL: Final[str] = r"(?<![\w$.])(?P<client>[A-Za-z_$][\w$]*)\s*(?:\?\.\s*)?\("
_JAVASCRIPT_CLIENT_FACTORY_CALL: Final[str] = (
    r"(?<![\w$.])(?P<client>[A-Za-z_$][\w$]*)\s*\.\s*"
    r"(?P<factory>create|extend)\s*\("
)


def _javascript_unwrap_outer_parentheses(expression: str) -> str:
    """Remove enclosing parentheses with one bounded linear scan."""
    left = 0
    right = len(expression)
    while left < right and expression[left].isspace():
        left += 1
    while right > left and expression[right - 1].isspace():
        right -= 1

    quote: str | None = None
    escaped = False
    openings: list[int] = []
    closing_for: dict[int, int] = {}
    for index, character in enumerate(expression):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character == "(":
            openings.append(index)
        elif character == ")" and openings:
            closing_for[openings.pop()] = index

    while left < right and expression[left] == "(" and closing_for.get(left) == right - 1:
        left += 1
        right -= 1
        while left < right and expression[left].isspace():
            left += 1
        while right > left and expression[right - 1].isspace():
            right -= 1
    return expression[left:right]


def _javascript_http_client_expression(
    expression: str,
    aliases: dict[str, str],
) -> str | None:
    """Resolve an exact CommonJS axios/got expression or one-hop alias."""
    value = _javascript_unwrap_outer_parentheses(expression)
    required_client = re.fullmatch(
        r"\(*\s*require\(\s*['\"](axios|got)['\"]\s*\)\s*\)*"
        r"(?:\s*\.\s*default)?",
        value,
    )
    if required_client is not None:
        return required_client.group(1)
    if re.fullmatch(r"[A-Za-z_$][\w$]*", value):
        return aliases.get(value)
    return None


def _javascript_contains_http_client_require(expression: str) -> bool:
    masked = _mask_javascript_strings(expression)
    for match in re.finditer(
        r"(?<![\w$])require\(\s*['\"](?:axios|got)['\"]\s*\)",
        expression,
    ):
        if masked[match.start()] != " ":
            return True
    return False


def _javascript_known_nonclient_require_expression(expression: str) -> bool:
    value = _javascript_unwrap_outer_parentheses(expression)
    return bool(
        re.fullmatch(
            r"\(*\s*require\(\s*['\"]axios['\"]\s*\)\s*\)*"
            r"\s*\.\s*mergeConfig",
            value,
        )
    )


def _javascript_http_client_factory_expression(
    expression: str,
    aliases: dict[str, str],
) -> bool:
    factory = re.fullmatch(
        r"([A-Za-z_$][\w$]*)\s*\.\s*(create|extend)\s*\(.*\)",
        expression.strip(),
        re.DOTALL,
    )
    if factory is None:
        return False
    name, method = factory.groups()
    return (aliases.get(name), method) in {("axios", "create"), ("got", "extend")}


def _javascript_top_level_parts(value: str, separator: str = ",") -> tuple[str, ...] | None:
    """Split JavaScript expressions at one top-level separator."""
    start = 0
    quote: str | None = None
    escaped = False
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    parts: list[str] = []
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}":
            if not stack or stack.pop() != pairs[character]:
                return None
        elif character == separator and not stack:
            parts.append(value[start:index].strip())
            start = index + 1
    if quote is not None or stack:
        return None
    parts.append(value[start:].strip())
    return tuple(parts)


def _javascript_declaration_has_multiple_declarators(value: str) -> bool | None:
    """Detect real declaration commas while respecting TS generics and regexes."""
    quote: str | None = None
    escaped = False
    regex_literal = False
    regex_character_class = False
    can_start_regex = False
    angle_depth = 0
    angle_kind: str | None = None
    angle_invalid = False
    in_type_annotation = False
    initializer_started = False
    ambiguous_angle_operator = False
    identifier: list[str] = []
    new_callee_state = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}

    def finish_identifier() -> None:
        nonlocal new_callee_state
        if not identifier:
            return
        token = "".join(identifier)
        identifier.clear()
        if not initializer_started:
            new_callee_state = 0
        elif new_callee_state in {1, 3}:
            new_callee_state = 2
        elif token == "new":
            new_callee_state = 1
        else:
            new_callee_state = 0

    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
                can_start_regex = False
            continue
        if regex_literal:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "[":
                regex_character_class = True
            elif character == "]":
                regex_character_class = False
            elif character == "/" and not regex_character_class:
                regex_literal = False
                can_start_regex = False
            continue
        if not stack and not angle_depth and (character.isalnum() or character in "_$"):
            identifier.append(character)
            can_start_regex = False
            continue
        finish_identifier()
        if character in {"'", '"', "`"}:
            quote = character
            new_callee_state = 0
            continue
        if character == "/" and can_start_regex:
            regex_literal = True
            regex_character_class = False
            new_callee_state = 0
            continue
        if character in "([{":
            stack.append(character)
            can_start_regex = True
            new_callee_state = 0
            continue
        if character in ")]}":
            if not stack or stack.pop() != pairs[character]:
                return None
            can_start_regex = False
            new_callee_state = 0
            continue
        if character.isspace():
            continue
        if stack:
            new_callee_state = 0
            if character == "/":
                can_start_regex = True
            else:
                can_start_regex = character in "=(:,!&|?+-*%^~<>"
            continue
        if angle_depth:
            new_callee_state = 0
            if character == "<":
                angle_depth += 1
            elif character == ">":
                angle_depth -= 1
                if angle_depth == 0 and angle_invalid:
                    return None
                if angle_depth == 0 and angle_kind == "initializer":
                    following = index + 1
                    while following < len(value) and value[following].isspace():
                        following += 1
                    if following >= len(value) or value[following] != "(":
                        return None
                    angle_kind = None
                elif angle_depth == 0:
                    angle_kind = None
            elif character == "=" and (index + 1 >= len(value) or value[index + 1] != ">"):
                angle_invalid = True
            can_start_regex = character in "=(:,!&|?+-*%^~<>"
            continue
        if character == ":" and not initializer_started:
            in_type_annotation = True
            new_callee_state = 0
        elif character == "=" and not initializer_started:
            if in_type_annotation and index + 1 < len(value) and value[index + 1] == ">":
                can_start_regex = True
                continue
            initializer_started = True
            in_type_annotation = False
            new_callee_state = 0
        elif character == "<":
            if in_type_annotation:
                angle_depth = 1
                angle_kind = "type"
                angle_invalid = False
                can_start_regex = True
                new_callee_state = 0
                continue
            if initializer_started and new_callee_state == 2:
                angle_depth = 1
                angle_kind = "initializer"
                angle_invalid = False
                can_start_regex = True
                new_callee_state = 0
                continue
            if initializer_started:
                ambiguous_angle_operator = True
            new_callee_state = 0
        if character == ",":
            return True
        if character == "." and new_callee_state == 2:
            new_callee_state = 3
        else:
            new_callee_state = 0
        if character == "/":
            can_start_regex = True
        else:
            can_start_regex = character in "=(:,!&|?+-*%^~<>"
    finish_identifier()
    if quote is not None or regex_literal or stack or angle_depth:
        return None
    if ambiguous_angle_operator:
        return None
    return False


def _javascript_object_properties(expression: str) -> dict[str, str] | None:
    """Return effective last-key-wins properties for a static object literal."""
    value = expression.strip()
    if not (value.startswith("{") and value.endswith("}")):
        return None
    raw_properties = _javascript_top_level_parts(value[1:-1])
    if raw_properties is None:
        return None
    properties: dict[str, str] = {}
    for raw_property in raw_properties:
        if not raw_property:
            continue
        if raw_property.startswith(("...", "[")):
            return None
        key_value = _javascript_top_level_parts(raw_property, ":")
        if key_value is None or len(key_value) != 2:
            return None
        raw_key, raw_value = key_value
        key = _javascript_literal(raw_key) or raw_key
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", key):
            return None
        properties[key] = raw_value
    return properties


_AXIOS_WIRE_FIELDS: Final[tuple[str, ...]] = ("data", "headers", "params", "auth")
_AXIOS_UNSUPPORTED_ROUTING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter",
        "beforeRedirect",
        "httpAgent",
        "httpsAgent",
        "paramsSerializer",
        "socketPath",
        "transformRequest",
        "transport",
    }
)
_GOT_WIRE_FIELDS: Final[tuple[str, ...]] = (
    "body",
    "json",
    "form",
    "headers",
    "searchParams",
    "username",
    "password",
    "cookieJar",
)
_GOT_UNSUPPORTED_ROUTING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "agent",
        "createConnection",
        "dnsLookup",
        "hooks",
        "lookup",
        "prefixUrl",
        "request",
        "socketPath",
    }
)
_JAVASCRIPT_MAX_WIRE_OBJECT_DEPTH: Final[int] = 32


def _javascript_wire_expression(
    properties: dict[str, str],
    fields: tuple[str, ...],
    *,
    authorization_overridden: bool = False,
) -> tuple[str, bool]:
    values: list[str] = []
    for field_name in fields:
        if field_name not in properties:
            continue
        excluded_keys = (
            frozenset({"authorization"})
            if field_name == "headers" and authorization_overridden
            else frozenset()
        )
        normalized, modeled = _javascript_effective_wire_value(
            properties[field_name],
            excluded_keys=excluded_keys,
        )
        if not modeled:
            return "", False
        if normalized:
            values.append(normalized)
    return "\n".join(values), True


def _javascript_effective_wire_value(
    expression: str,
    *,
    excluded_keys: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """Normalize bounded static wire objects with last-key-wins semantics."""
    if not _javascript_wire_object_depth_within_limit(expression):
        return "", False
    return _javascript_effective_wire_value_bounded(
        expression,
        excluded_keys=excluded_keys,
    )


def _javascript_wire_object_depth_within_limit(expression: str) -> bool:
    quote: str | None = None
    escaped = False
    depth = 0
    for character in expression:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character == "{":
            depth += 1
            if depth > _JAVASCRIPT_MAX_WIRE_OBJECT_DEPTH:
                return False
        elif character == "}":
            depth -= 1
    return True


def _javascript_effective_wire_value_bounded(
    expression: str,
    *,
    excluded_keys: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """Walk a preflight-bounded object; rescans are capped by the depth limit."""
    value = expression.strip()
    if not (value.startswith("{") and value.endswith("}")):
        return value, True
    properties = _javascript_object_properties(value)
    if properties is None:
        return "", False
    values: list[str] = []
    for key, nested_value in properties.items():
        if key.casefold() in excluded_keys:
            continue
        normalized, modeled = _javascript_effective_wire_value_bounded(nested_value)
        if not modeled:
            return "", False
        if normalized:
            values.append(normalized)
    return "\n".join(values), True


def _javascript_boolean(value: str) -> bool | None:
    normalized = value.strip()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _javascript_axios_destination(url: str | None) -> DestinationClass:
    """Classify HTTP(S) and Axios-absolute protocol-relative URLs."""
    if url is not None and url.startswith("//"):
        return _destination_for_url(f"http:{url}")
    return _destination_for_url(url)


def _javascript_axios_proxy_semantics(
    expression: str,
) -> tuple[DestinationClass | None, str, bool]:
    """Return an explicit Axios proxy route and proxy-auth source."""
    if expression.strip() == "false":
        return None, "", True
    properties = _javascript_object_properties(expression)
    if properties is None or not set(properties) <= {
        "auth",
        "host",
        "port",
        "protocol",
    }:
        return None, "", False
    host = _javascript_literal(properties.get("host", ""))
    if host is None:
        return None, "", False
    if "protocol" in properties:
        protocol = _javascript_literal(properties["protocol"])
        if protocol not in {"http", "https", "http:", "https:"}:
            return None, "", False
    if "port" in properties:
        port = properties["port"].strip()
        port_literal = _javascript_literal(port)
        if not re.fullmatch(r"\d+", port_literal if port_literal is not None else port):
            return None, "", False
    proxy_auth, auth_modeled = _javascript_effective_wire_value(properties.get("auth", ""))
    if not auth_modeled:
        return None, "", False
    return _destination_for_host(host), proxy_auth, True


def _javascript_axios_request_semantics(
    url_expression: str,
    properties: dict[str, str],
) -> tuple[DestinationClass, str, bool]:
    """Resolve a supported Axios request route and wire-bearing config."""
    if set(properties) & _AXIOS_UNSUPPORTED_ROUTING_FIELDS:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    url = _javascript_literal(url_expression)
    if url is None:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False

    base_url: str | None = None
    base_destination: DestinationClass | None = None
    if "baseURL" in properties:
        base_url = _javascript_literal(properties["baseURL"])
        base_destination = _javascript_axios_destination(base_url)
        if base_url is None or base_destination is DestinationClass.DYNAMIC_UNKNOWN:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False

    allow_absolute_urls = True
    if "allowAbsoluteUrls" in properties:
        parsed_allow_absolute_urls = _javascript_boolean(properties["allowAbsoluteUrls"])
        if parsed_allow_absolute_urls is None:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False
        allow_absolute_urls = parsed_allow_absolute_urls

    proxy_destination: DestinationClass | None = None
    proxy_source = ""
    if "proxy" in properties:
        proxy_destination, proxy_source, proxy_modeled = _javascript_axios_proxy_semantics(
            properties["proxy"]
        )
        if not proxy_modeled:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False

    url_destination = _javascript_axios_destination(url)
    url_is_absolute = url_destination is not DestinationClass.DYNAMIC_UNKNOWN
    if base_destination is not None and (not url_is_absolute or not allow_absolute_urls):
        destination = base_destination
    elif url_is_absolute:
        destination = url_destination
    else:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    if proxy_destination is not None and destination is DestinationClass.LOOPBACK:
        destination = proxy_destination

    auth_properties = (
        _javascript_object_properties(properties["auth"]) if "auth" in properties else None
    )
    wire_source, wire_modeled = _javascript_wire_expression(
        properties,
        _AXIOS_WIRE_FIELDS,
        authorization_overridden=auth_properties is not None,
    )
    if not wire_modeled:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    source_expression = "\n".join(
        value
        for value in (
            wire_source,
            proxy_source,
        )
        if value
    )
    return destination, source_expression, True


def _javascript_axios_callable_semantics(
    arguments: str,
) -> tuple[DestinationClass, str, bool]:
    argument_parts = _javascript_top_level_parts(arguments)
    if not argument_parts or not argument_parts[0] or len(argument_parts) > 2:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    direct_url = _javascript_literal(argument_parts[0])
    if direct_url is not None:
        if len(argument_parts) == 1:
            properties: dict[str, str] = {}
        else:
            parsed_properties = _javascript_object_properties(argument_parts[1])
            if parsed_properties is None:
                return DestinationClass.DYNAMIC_UNKNOWN, "", False
            properties = parsed_properties
        return _javascript_axios_request_semantics(argument_parts[0], properties)
    if len(argument_parts) != 1:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    object_properties = _javascript_object_properties(argument_parts[0])
    if object_properties is None or "url" not in object_properties:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    return _javascript_axios_request_semantics(
        object_properties["url"],
        object_properties,
    )


def _javascript_got_callable_semantics(
    arguments: str,
) -> tuple[DestinationClass, str, bool]:
    argument_parts = _javascript_top_level_parts(arguments)
    if not argument_parts or not argument_parts[0] or len(argument_parts) > 2:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    url = _javascript_literal(argument_parts[0])
    destination = _destination_for_url(url)
    if url is None or destination is DestinationClass.DYNAMIC_UNKNOWN:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    if len(argument_parts) == 1:
        properties: dict[str, str] = {}
    else:
        parsed_properties = _javascript_object_properties(argument_parts[1])
        if parsed_properties is None:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False
        properties = parsed_properties
    if set(properties) & _GOT_UNSUPPORTED_ROUTING_FIELDS:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    wire_source, wire_modeled = _javascript_wire_expression(properties, _GOT_WIRE_FIELDS)
    if not wire_modeled:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    return destination, wire_source, True


def _javascript_callable_semantics(
    arguments: str,
    client_kind: str,
) -> tuple[DestinationClass, str, bool]:
    """Return client-specific destination/source semantics for callable clients."""
    if client_kind == "axios":
        return _javascript_axios_callable_semantics(arguments)
    if client_kind == "got":
        return _javascript_got_callable_semantics(arguments)
    return DestinationClass.DYNAMIC_UNKNOWN, "", False


def _javascript_method_semantics(
    arguments: str,
    *,
    client_kind: str,
    method: str,
) -> tuple[DestinationClass, str, bool]:
    """Return client-specific semantics for axios/got shortcut methods."""
    argument_parts = _javascript_top_level_parts(arguments)
    if not argument_parts or not argument_parts[0]:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    if client_kind == "got":
        if method not in {"delete", "get", "head", "patch", "post", "put"}:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False
        return _javascript_got_callable_semantics(arguments)
    if client_kind != "axios":
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    if method == "request":
        if len(argument_parts) != 1:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False
        return _javascript_axios_callable_semantics(arguments)
    if method not in {
        "delete",
        "get",
        "head",
        "options",
        "patch",
        "patchForm",
        "post",
        "postForm",
        "put",
        "putForm",
    }:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False

    body_expression = ""
    config_index = 1
    if method in {"patch", "patchForm", "post", "postForm", "put", "putForm"}:
        if len(argument_parts) >= 2:
            body_expression = argument_parts[1]
        config_index = 2
    if len(argument_parts) > config_index + 1:
        return DestinationClass.DYNAMIC_UNKNOWN, "", False
    properties: dict[str, str] = {}
    if len(argument_parts) > config_index:
        parsed_properties = _javascript_object_properties(argument_parts[config_index])
        if parsed_properties is None:
            return DestinationClass.DYNAMIC_UNKNOWN, "", False
        properties = parsed_properties
    destination, config_source, modeled = _javascript_axios_request_semantics(
        argument_parts[0],
        properties,
    )
    source_expression = "\n".join(value for value in (body_expression, config_source) if value)
    return destination, source_expression, modeled


def _javascript_call_arguments(statement: str, opening: int) -> str | None:
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(opening, len(statement)):
        character = statement[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return statement[opening + 1 : index]
    return None


def _javascript_environment_taint(expression: str, profile: UserConfigProfile | None) -> str | None:
    masked = _mask_javascript_strings(expression)
    names = re.findall(
        r"\bprocess\s*\.\s*env\s*\.\s*([A-Za-z_$][\w$]*)",
        masked,
    )
    for match in re.finditer(
        r"\bprocess\s*\.\s*env\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
        expression,
    ):
        if masked[match.start() : match.start() + len("process")] == "process":
            names.append(match.group(1))
    names.extend(
        match.group(1)
        for match in re.finditer(
            r"`(?:\\.|[^`])*?\$\{\s*process\s*\.\s*env\s*\.\s*"
            r"([A-Za-z_$][\w$]*)[^}]*\}(?:\\.|[^`])*?`",
            expression,
            re.DOTALL,
        )
    )
    for name in names:
        if not _sensitive_environment_name(name, profile):
            continue
        if profile and name in profile.sensitive_environment_names:
            return "plugin_sensitive_user_config"
        return "ambient_credential_environment"
    return None


def _javascript_expr_taint(
    expression: str,
    *,
    variables: dict[str, str],
    event_taint: str | None,
    profile: UserConfigProfile | None,
) -> str | None:
    masked = _mask_javascript_strings(expression)
    for match in re.finditer(r"(?<![\w$])([A-Za-z_$][\w$]*)(?![\w$])", masked):
        if (taint := variables.get(match.group(1))) is not None:
            return taint
    environment = _javascript_environment_taint(expression, profile)
    if environment is not None:
        return environment
    for match in re.finditer(r"\breadFileSync\s*\(", masked):
        arguments = _javascript_call_arguments(expression, match.end() - 1)
        if arguments is None:
            continue
        operand = _javascript_first_argument(arguments)
        if operand == "0" and event_taint:
            return event_taint
        path = _javascript_literal(operand)
        if path is not None and _sensitive_path(path, expand_shell=False):
            return "sensitive_local_file"
    return None


def _javascript_is_unmodeled(source: str) -> bool:
    masked = _mask_javascript_strings(source)
    if re.search(r"(?<![\w$])(?:eval|Function)\s*\(", masked):
        return True
    if re.search(
        r"(?<![\w$])(?:catch|class|do|for|function|if|switch|try|while|with)\b|=>",
        masked,
    ):
        return True
    if re.search(
        r"(?m)^\s*(?:import|export)\s+(?:[^;\n]*?\s+from\s+)?"
        r"['\"](?!\.{1,2}/)",
        source,
    ):
        return True
    unsupported_client_shapes = (
        r"(?:const|let|var)\s*\{[^}\n]*\}\s*=\s*"
        r"require\(\s*['\"](?:axios|got)['\"]\s*\)",
        r"(?:\(\s*)?require\(\s*['\"](?:axios|got)['\"]\s*\)\s*"
        r"(?:\)\s*)?(?:\.\s*default\s*)?\(",
        r"require\(\s*['\"](?:axios|got)['\"]\s*\)\s*\.\s*"
        r"(?:create|extend)\s*\(",
    )
    for pattern in unsupported_client_shapes:
        for match in re.finditer(pattern, source):
            if masked[match.start()] != " ":
                return True
    for match in re.finditer(r"(?<![\w$])(?:import|require)\s*\(", masked):
        arguments = _javascript_call_arguments(source, match.end() - 1)
        literal = (
            _javascript_literal(_javascript_first_argument(arguments))
            if arguments is not None
            else None
        )
        if literal is None or literal in {
            "child_process",
            "http",
            "https",
            "node:child_process",
            "node:http",
            "node:https",
        }:
            return True
    if re.search(
        r"(?<![\w$])(?:child_process\s*\.\s*)?"
        r"(?:exec|execFile|execFileSync|execSync|fork|spawn|spawnSync)\s*\(",
        masked,
    ):
        return True
    return False


def _javascript_statement_hits(
    statement: str,
    *,
    start_line: int,
    variables: dict[str, str],
    http_client_aliases: dict[str, str],
    event_taint: str | None,
    profile: UserConfigProfile | None,
) -> tuple[list[_SinkHit], bool]:
    """Analyze one reachable statement against its pre-execution state."""
    masked = _mask_javascript_strings(statement)
    for factory_match in re.finditer(_JAVASCRIPT_CLIENT_FACTORY_CALL, masked):
        factory_name = factory_match.group("client")
        if (
            http_client_aliases.get(factory_name),
            factory_match.group("factory"),
        ) in {("axios", "create"), ("got", "extend")}:
            return [], True

    hits: list[_SinkHit] = []
    for match in re.finditer(_JAVASCRIPT_HTTP_METHOD_CALL, masked):
        client_name = match.group("client")
        if client_name is not None and client_name not in http_client_aliases:
            continue
        arguments = _javascript_call_arguments(statement, match.end() - 1)
        if arguments is None:
            return [], True
        if client_name is None:
            source_expression = arguments
            destination = _destination_for_url(
                _javascript_literal(_javascript_first_argument(arguments))
            )
            modeled = True
        else:
            destination, source_expression, modeled = _javascript_method_semantics(
                arguments,
                client_kind=http_client_aliases[client_name],
                method=match.group("method"),
            )
        if not modeled:
            return [], True
        source_kind = _javascript_expr_taint(
            source_expression,
            variables=variables,
            event_taint=event_taint,
            profile=profile,
        )
        if source_kind is None or destination is DestinationClass.LOOPBACK:
            continue
        sink_line = start_line + statement[: match.start()].count("\n")
        hits.append(_SinkHit(source_kind, TransportKind.HTTP, destination, sink_line))

    for match in re.finditer(_JAVASCRIPT_CALLABLE_CALL, masked):
        client_name = match.group("client")
        if client_name not in http_client_aliases:
            continue
        arguments = _javascript_call_arguments(statement, match.end() - 1)
        if arguments is None:
            return [], True
        destination, source_expression, modeled = _javascript_callable_semantics(
            arguments,
            http_client_aliases[client_name],
        )
        if not modeled:
            return [], True
        source_kind = _javascript_expr_taint(
            source_expression,
            variables=variables,
            event_taint=event_taint,
            profile=profile,
        )
        if source_kind is None or destination is DestinationClass.LOOPBACK:
            continue
        sink_line = start_line + statement[: match.start()].count("\n")
        hits.append(_SinkHit(source_kind, TransportKind.HTTP, destination, sink_line))
    return hits, False


def _analyze_javascript_payload(
    content: str,
    *,
    event_taint: str | None,
    profile: UserConfigProfile | None,
) -> tuple[list[_SinkHit], bool]:
    source, valid = _strip_javascript_comments(content)
    if not valid or not _javascript_structure_valid(source) or _javascript_is_unmodeled(source):
        return [], True
    variables: dict[str, str] = {}
    http_client_aliases: dict[str, str] = {}
    hits: list[_SinkHit] = []
    for statement, start_line in _javascript_statements(source):
        declaration = re.match(
            r"^(?:const|let|var)\s+(.*)$",
            statement,
            re.DOTALL,
        )
        if declaration is not None:
            multiple_declarators = _javascript_declaration_has_multiple_declarators(
                declaration.group(1)
            )
            if multiple_declarators is not False:
                return [], True
        assignment = re.match(
            r"^(?:const|let|var)\s+([A-Za-z_$][\w$]*)"
            r"(?:\s*:\s*[^=;]+)?\s*=\s*(.*)$",
            statement,
            re.DOTALL,
        )
        mutation = re.match(
            r"^([A-Za-z_$][\w$]*)\s*"
            r"(&&=|\|\|=|\?\?=|>>>=|<<=|>>=|\*\*=|[+\-*/%&|^]=|=(?!=|>))\s*(.*)$",
            statement,
            re.DOTALL,
        )
        rhs_expression = (
            assignment.group(2)
            if assignment is not None
            else mutation.group(3)
            if mutation is not None
            else None
        )
        mutation_name = mutation.group(1) if mutation is not None else None
        mutation_operator = mutation.group(2) if mutation is not None else None
        previous_kind = (
            http_client_aliases.get(mutation_name) if mutation_name is not None else None
        )
        rhs_reachable = not (mutation_operator in {"||=", "??="} and previous_kind is not None)

        resolved_rhs_kind = (
            _javascript_http_client_expression(rhs_expression, http_client_aliases)
            if rhs_expression is not None
            else None
        )
        if rhs_reachable and rhs_expression is not None:
            if _javascript_http_client_factory_expression(
                rhs_expression,
                http_client_aliases,
            ):
                return [], True
            if (
                resolved_rhs_kind is None
                and _javascript_contains_http_client_require(rhs_expression)
                and not _javascript_known_nonclient_require_expression(rhs_expression)
            ):
                return [], True
            if (
                mutation_operator in {"&&=", "||=", "??="}
                and previous_kind is None
                and resolved_rhs_kind is not None
            ):
                return [], True

        if rhs_reachable:
            statement_hits, unmodeled = _javascript_statement_hits(
                statement,
                start_line=start_line,
                variables=variables,
                http_client_aliases=http_client_aliases,
                event_taint=event_taint,
                profile=profile,
            )
            if unmodeled:
                return [], True
            hits.extend(statement_hits)

        if assignment is not None:
            alias_name = assignment.group(1)
            if resolved_rhs_kind is None:
                http_client_aliases.pop(alias_name, None)
            else:
                http_client_aliases[alias_name] = resolved_rhs_kind
        elif mutation is not None and mutation_name is not None:
            updated_kind: str | None
            if mutation_operator in {"||=", "??="} and previous_kind is not None:
                updated_kind = previous_kind
            elif mutation_operator in {"=", "&&="}:
                updated_kind = resolved_rhs_kind
            else:
                updated_kind = None
            if updated_kind is None:
                http_client_aliases.pop(mutation_name, None)
            else:
                http_client_aliases[mutation_name] = updated_kind

        variable_assignment = assignment
        if mutation is not None and mutation_operator == "=":
            variable_assignment = mutation
        if variable_assignment is not None:
            name = variable_assignment.group(1)
            expression = (
                variable_assignment.group(2)
                if variable_assignment is assignment
                else variable_assignment.group(3)
            )
            taint = _javascript_expr_taint(
                expression,
                variables=variables,
                event_taint=event_taint,
                profile=profile,
            )
            if taint is None:
                variables.pop(name, None)
            else:
                variables[name] = taint
    return hits, False


def _javascript_local_references(content: str) -> tuple[_Reference, ...]:
    source, valid = _strip_javascript_comments(content)
    if not valid:
        return ()
    masked = _mask_javascript_strings(source)
    unresolved: list[tuple[int, str]] = []
    for match in re.finditer(r"(?<![\w$])(?:import|require)\s*\(", masked):
        arguments = _javascript_call_arguments(source, match.end() - 1)
        if arguments is None:
            continue
        relative = _javascript_literal(_javascript_first_argument(arguments))
        if relative is None or not relative.startswith(("./", "../")):
            continue
        unresolved.append((match.start(), relative))
    static_import = re.compile(
        r"(?m)^\s*(?:import|export)\s+(?:[^;\n]*?\s+from\s+)?"
        r"(?P<quote>['\"])(?P<path>\.{1,2}/[^'\"]+)(?P=quote)"
    )
    for match in static_import.finditer(source):
        relative = match.group("path")
        unresolved.append((match.start(), relative))
    unresolved.sort(key=lambda item: item[0])
    references: list[_Reference] = []
    previous_offset = 0
    line = 1
    for offset, relative in unresolved:
        line += source.count("\n", previous_offset, offset)
        previous_offset = offset
        references.append(_Reference("component", relative, line))
    return tuple(dict.fromkeys(references))


def _deduplicated_references(
    registration: HookRegistration,
    references: tuple[_Reference, ...],
    cache: dict[str, str],
    *,
    base_path: str | None = None,
) -> tuple[_Reference, ...]:
    """Keep the first reference to each equivalent resolved component edge."""
    unique: list[_Reference] = []
    seen: set[tuple[str, ...]] = set()
    for reference in references:
        path = _resolved_reference(registration, reference, base_path=base_path)
        if (
            path is not None
            and reference.scope == "component"
            and path not in cache
            and not PurePosixPath(path).suffix
            and f"{path}.js" in cache
        ):
            path = f"{path}.js"
        key = (
            ("resolved", path)
            if path is not None
            else (
                "unresolved",
                reference.scope,
                reference.relative,
            )
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(reference)
    return tuple(unique)


class _TraversalSession:
    """One cache-only traversal session with globally unique component work."""

    def __init__(self, cache: dict[str, str], python_ast_cache_key: str | None) -> None:
        self.cache = cache
        self.python_ast_cache_key = python_ast_cache_key
        self.work: OrderedDict[FlowWorkRef, FlowWorkResult] = OrderedDict()

    def record(self, result: FlowWorkResult) -> None:
        existing = self.work.get(result.ref)
        if existing is None or (
            existing.outcome is LedgerOutcome.COMPLETED
            and result.outcome is not LedgerOutcome.COMPLETED
        ):
            self.work[result.ref] = result

    def fail_activation(
        self,
        registration: HookRegistration,
        reason: LedgerReason,
        *,
        error_class: str = "BundledHookFlowError",
        observed_characters: int | None = None,
        limit_characters: int | None = None,
    ) -> None:
        line = max(1, registration.source_line)
        ref = FlowWorkRef(registration.source_path, line, line)
        self.record(
            FlowWorkResult(
                ref,
                LedgerOutcome.FAILED,
                reason,
                error_class=error_class,
                observed_characters=observed_characters,
                limit_characters=limit_characters,
            )
        )

    def visit(
        self,
        document: DocumentFlowInput,
        handler: HandlerFlowInput,
        ordinal: int,
        reference: _Reference,
        *,
        event_taint: str | None,
        profile: UserConfigProfile | None,
        budget: _HandlerBudget,
        depth: int,
        stack: tuple[str, ...],
        chain: tuple[tuple[str, str], ...],
        base_path: str | None = None,
    ) -> list[OwnedFlowFinding]:
        path = _resolved_reference(
            handler.registration,
            reference,
            base_path=base_path,
        )
        if path is None:
            self.fail_activation(handler.registration, LedgerReason.UNMODELED_PAYLOAD)
            return []
        if (
            reference.scope == "component"
            and path not in self.cache
            and not PurePosixPath(path).suffix
            and f"{path}.js" in self.cache
        ):
            path = f"{path}.js"
        if path in stack:
            self.fail_activation(
                handler.registration,
                LedgerReason.UNMODELED_PAYLOAD,
                error_class="BundledHookReferenceCycle",
            )
            return []
        if depth > _MAX_WRAPPER_HOPS:
            self.fail_activation(
                handler.registration,
                LedgerReason.DEPTH_LIMIT,
                error_class="BundledHookDepthLimit",
            )
            return []
        if path not in budget.seen:
            if len(budget.seen) >= _MAX_REFERENCED_COMPONENTS:
                self.fail_activation(
                    handler.registration,
                    LedgerReason.COMPONENT_LIMIT,
                    error_class="BundledHookComponentLimit",
                )
                return []
            budget.seen.add(path)
        content = self.cache.get(path)
        if content is None:
            self.record(
                FlowWorkResult(
                    FlowWorkRef(path),
                    LedgerOutcome.FAILED,
                    LedgerReason.MISSING_FILE_CACHE,
                    error_class="MissingBundledHookPayload",
                )
            )
            return []
        if "\x00" in content:
            self.record(
                FlowWorkResult(
                    FlowWorkRef(path),
                    LedgerOutcome.FAILED,
                    LedgerReason.BINARY_CONTENT,
                    error_class="BinaryBundledHookPayload",
                )
            )
            return []
        if len(content) > MAX_FILE_CHARS:
            self.record(
                FlowWorkResult(
                    FlowWorkRef(path),
                    LedgerOutcome.FAILED,
                    LedgerReason.SIZE_LIMIT,
                    error_class="BundledHookPayloadSizeLimit",
                    observed_characters=len(content),
                    limit_characters=MAX_FILE_CHARS,
                )
            )
            return []
        if path not in budget.counted:
            if budget.aggregate_characters + len(content) > _MAX_AGGREGATE_PAYLOAD_CHARS:
                self.fail_activation(
                    handler.registration,
                    LedgerReason.AGGREGATE_BUDGET,
                    error_class="BundledHookAggregateBudget",
                    observed_characters=budget.aggregate_characters + len(content),
                    limit_characters=_MAX_AGGREGATE_PAYLOAD_CHARS,
                )
                return []
            budget.counted.add(path)
            budget.aggregate_characters += len(content)
        owner = FlowWorkRef(path)
        self.record(FlowWorkResult(owner, LedgerOutcome.COMPLETED))
        content_digest = f"sha256:{sha256(content.encode()).hexdigest()}"
        next_chain = (*chain, (path, content_digest))
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in {".sh", ".bash", ".zsh", ".py", ".js", ".mjs", ".cjs", ".ts"}:
            self.record(
                FlowWorkResult(
                    owner,
                    LedgerOutcome.FAILED,
                    LedgerReason.UNMODELED_PAYLOAD,
                    error_class="UnsupportedBundledHookPayload",
                )
            )
            return []
        findings: list[OwnedFlowFinding] = []
        unmodeled = False
        if suffix == ".py":
            hits, unmodeled = _analyze_python_payload(
                content,
                path,
                event_taint=event_taint,
                profile=profile,
                python_ast_cache_key=self.python_ast_cache_key,
            )
        elif suffix in {".js", ".mjs", ".cjs", ".ts"}:
            hits, unmodeled = _analyze_javascript_payload(
                content,
                event_taint=event_taint,
                profile=profile,
            )
        else:
            unmodeled = _shell_payload_unmodeled(content)
            hits = (
                []
                if unmodeled
                else _analyze_shell(content, event_taint=event_taint, profile=profile)
            )
        if unmodeled:
            self.record(
                FlowWorkResult(
                    owner,
                    LedgerOutcome.FAILED,
                    LedgerReason.UNMODELED_PAYLOAD,
                    error_class="UnmodeledBundledHookPayload",
                )
            )
            return []
        for hit in hits:
            findings.append(
                OwnedFlowFinding(
                    owner,
                    _bh2_finding(
                        document,
                        handler,
                        ordinal,
                        source_kind=hit.source_kind,
                        transport=hit.transport,
                        destination=hit.destination,
                        sink_path=path,
                        sink_line=hit.line,
                        component_identities=next_chain,
                    ),
                )
            )
        child_references: tuple[_Reference, ...] = ()
        if suffix in {".sh", ".bash", ".zsh"}:
            child_references = _shell_entrypoint_references(content)
        elif suffix in {".js", ".mjs", ".cjs", ".ts"}:
            child_references = _javascript_local_references(content)
        for child in _deduplicated_references(
            handler.registration,
            child_references,
            self.cache,
            base_path=path,
        ):
            findings.extend(
                self.visit(
                    document,
                    handler,
                    ordinal,
                    child,
                    event_taint=event_taint,
                    profile=profile,
                    budget=budget,
                    depth=depth + 1,
                    stack=(*stack, path),
                    chain=next_chain,
                    base_path=path,
                )
            )
        return findings

    def analyze_handler(
        self,
        document: DocumentFlowInput,
        handler: HandlerFlowInput,
        ordinal: int,
        *,
        event_taint: str | None,
        profile: UserConfigProfile | None,
    ) -> list[OwnedFlowFinding]:
        references = _deduplicated_references(
            handler.registration,
            _handler_references(handler),
            self.cache,
        )
        if _unsafe_entrypoint(handler):
            self.fail_activation(handler.registration, LedgerReason.UNMODELED_PAYLOAD)
            return []
        if not references:
            return []
        budget = _HandlerBudget()
        findings: list[OwnedFlowFinding] = []
        for reference in references:
            findings.extend(
                self.visit(
                    document,
                    handler,
                    ordinal,
                    reference,
                    event_taint=event_taint,
                    profile=profile,
                    budget=budget,
                    depth=0,
                    stack=(),
                    chain=(),
                )
            )
        return findings


def _chain_digest(
    document: DocumentFlowInput,
    handler: HandlerFlowInput,
    ordinal: int,
    *,
    source_kind: str,
    transport: TransportKind,
    destination: DestinationClass,
    component_identities: tuple[tuple[str, str], ...] = (),
) -> str:
    fields = [
        _SCHEMA,
        "BH2_CHAIN",
        document.source_path,
        document.content_digest,
        handler.registration.chain_digest,
        str(ordinal),
        source_kind,
        transport.value,
        destination.value,
    ]
    for path, content_digest in component_identities:
        fields.extend((path, content_digest))
    return f"sha256:{sha256(chr(0).join(fields).encode()).hexdigest()}"


def _bh2_finding(
    document: DocumentFlowInput,
    handler: HandlerFlowInput,
    ordinal: int,
    *,
    source_kind: str,
    transport: TransportKind,
    destination: DestinationClass,
    sink_path: str | None = None,
    sink_line: int | None = None,
    component_identities: tuple[tuple[str, str], ...] = (),
) -> Finding:
    digest = _chain_digest(
        document,
        handler,
        ordinal,
        source_kind=source_kind,
        transport=transport,
        destination=destination,
        component_identities=component_identities,
    )
    evidence: dict[str, object] = {
        "schema": _SCHEMA,
        "claude_semantics_snapshot": _SEMANTICS_SNAPSHOT,
        "source_kind": document.source_kind,
        "declaration_roles": ",".join(document.declaration_roles),
        "activation_lifetime": document.activation_lifetime,
        "runtime_status": "runnable",
        "chain_digest": digest,
        "transport_kind": transport.value,
        "destination_class": destination.value,
        "sensitive_source_kind": source_kind,
    }
    if sink_path is not None:
        evidence["payload_component"] = sink_path
        evidence["component_count"] = len(component_identities)
    return Finding(
        rule_id="BH2",
        message="Bundled hook can send sensitive runtime data to an outbound destination.",
        severity="CRITICAL",
        confidence=1.0,
        file=sink_path or document.source_path,
        start_line=max(1, sink_line or handler.registration.source_line),
        category="Bundled Execution Surface",
        pattern="Bundled Hook Data Exfiltration",
        explanation=(
            "A runnable bundled hook contains a correlated sensitive-source-to-outbound-sink flow."
        ),
        remediation="Remove the sensitive source-to-outbound-sink flow.",
        tags=["bundled-execution-surface", "structural"],
        matched_text=digest,
        finding=digest,
        evidence=evidence,
    )


def _static_http_origin(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or "$" in parsed.netloc
        or "%" in parsed.netloc
        or "{" in parsed.netloc
        or "}" in parsed.netloc
    ):
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    return f"{parsed.scheme}://{hostname.rstrip('.').casefold()}:{port or default_port}"


def _config_key_occurs(value: str, key: str) -> bool:
    if f"${{user_config.{key}}}" in value:
        return True
    environment_name = _user_config_environment_name(key)
    return environment_name in _environment_names(value)


def _curl_user_config_use(
    words: tuple[str, ...],
    key: str,
) -> tuple[bool, frozenset[str] | None]:
    occurrences = False
    origins: set[str] = set()
    only_proven_authorization = True
    for group in _curl_groups(words):
        group_authorization = False
        group_occurrence = False
        index = 1
        while index < len(group):
            word = group[index]
            parsed_option = _curl_option_at(group, index)
            if parsed_option is not None:
                option, value, index = parsed_option
                if not _config_key_occurs(value, key):
                    continue
                occurrences = True
                group_occurrence = True
                if option in {"-H", "--header"} and _is_authorization_header(value):
                    group_authorization = True
                else:
                    only_proven_authorization = False
                continue
            if _config_key_occurs(word, key):
                occurrences = True
                group_occurrence = True
                only_proven_authorization = False
            index += 1
        if group_occurrence and _curl_has_route_override(group):
            only_proven_authorization = False
        if not group_authorization:
            continue
        group_origins = tuple(_static_http_origin(url) for url in _curl_transfer_urls(group))
        if not group_origins or None in group_origins:
            only_proven_authorization = False
            continue
        origins.update(origin for origin in group_origins if origin is not None)
    if not occurrences or not only_proven_authorization:
        return occurrences, None
    return True, frozenset(origins)


def _record_config_command(
    command: str,
    args: tuple[str, ...] | None,
    profile: UserConfigProfile,
    uses: dict[str, _UserConfigUse],
) -> None:
    nested = _nested_shell(command, args or ()) if args is not None else None
    if nested is not None:
        _record_config_command(nested, None, profile, uses)
        return
    word_sets = (
        ((command, *args),)
        if args is not None
        else tuple(
            words
            for segment, _operator, _line in _split_shell(command)
            if (words := _shell_words(segment))
        )
    )
    for words in word_sets:
        effective_words = _unwrap_shell_command(words)
        is_curl = bool(effective_words) and _normalized_executable(effective_words[0]) == "curl"
        for key in profile.sensitive_keys:
            use = uses[key]
            if is_curl:
                occurs, origins = _curl_user_config_use(effective_words, key)
            else:
                occurs = any(_config_key_occurs(word, key) for word in words)
                origins = None
            if not occurs:
                continue
            if origins is None:
                use.has_other_use = True
            else:
                use.origins.update(origins)


def _record_config_http(
    handler: HandlerFlowInput,
    profile: UserConfigProfile,
    uses: dict[str, _UserConfigUse],
) -> None:
    origin = _static_http_origin(handler.url)
    for key in profile.sensitive_keys:
        environment_name = _user_config_environment_name(key)
        for header_name, value in handler.headers:
            if not _config_key_occurs(value, key):
                continue
            use = uses[key]
            is_runtime_value = (
                environment_name not in _environment_names(value)
                or environment_name not in handler.allowed_env_vars
            )
            if origin is None or header_name.casefold() != "authorization" or is_runtime_value:
                use.has_other_use = True
            else:
                use.origins.add(origin)


def _record_reachable_config_uses(
    handler: HandlerFlowInput,
    profile: UserConfigProfile,
    uses: dict[str, _UserConfigUse],
    cache: dict[str, str],
    python_ast_cache_key: str | None,
) -> None:
    """Prepass reachable cached payloads under the same bounds as flow traversal."""
    budget = _HandlerBudget()

    def disqualify_authentication_only() -> None:
        for use in uses.values():
            use.has_other_use = True

    def visit(
        reference: _Reference,
        *,
        depth: int,
        stack: tuple[str, ...],
        base_path: str | None = None,
    ) -> None:
        path = _resolved_reference(handler.registration, reference, base_path=base_path)
        if path is None:
            disqualify_authentication_only()
            return
        if (
            reference.scope == "component"
            and path not in cache
            and not PurePosixPath(path).suffix
            and f"{path}.js" in cache
        ):
            path = f"{path}.js"
        if path in stack or depth > _MAX_WRAPPER_HOPS:
            disqualify_authentication_only()
            return
        if path not in budget.seen:
            if len(budget.seen) >= _MAX_REFERENCED_COMPONENTS:
                disqualify_authentication_only()
                return
            budget.seen.add(path)
        content = cache.get(path)
        if content is None or "\x00" in content or len(content) > MAX_FILE_CHARS:
            disqualify_authentication_only()
            return
        if path not in budget.counted:
            if budget.aggregate_characters + len(content) > _MAX_AGGREGATE_PAYLOAD_CHARS:
                disqualify_authentication_only()
                return
            budget.counted.add(path)
            budget.aggregate_characters += len(content)
        suffix = PurePosixPath(path).suffix.lower()
        children: tuple[_Reference, ...] = ()
        if suffix in {".sh", ".bash", ".zsh"}:
            if _shell_payload_unmodeled(content):
                disqualify_authentication_only()
                return
            _record_config_command(content, None, profile, uses)
            children = _shell_entrypoint_references(content)
        elif suffix == ".py":
            _hits, unmodeled = _analyze_python_payload(
                content,
                path,
                event_taint=None,
                profile=profile,
                python_ast_cache_key=python_ast_cache_key,
            )
            if unmodeled:
                disqualify_authentication_only()
                return
            for key in profile.sensitive_keys:
                if _config_key_occurs(content, key):
                    uses[key].has_other_use = True
        elif suffix in {".js", ".mjs", ".cjs", ".ts"}:
            _hits, unmodeled = _analyze_javascript_payload(
                content,
                event_taint=None,
                profile=profile,
            )
            if unmodeled:
                disqualify_authentication_only()
                return
            for key in profile.sensitive_keys:
                if _config_key_occurs(content, key):
                    uses[key].has_other_use = True
            children = _javascript_local_references(content)
        else:
            disqualify_authentication_only()
            return
        for child in _deduplicated_references(
            handler.registration,
            children,
            cache,
            base_path=path,
        ):
            visit(
                child,
                depth=depth + 1,
                stack=(*stack, path),
                base_path=path,
            )

    if _unsafe_entrypoint(handler):
        disqualify_authentication_only()
        return
    for reference in _deduplicated_references(
        handler.registration,
        _handler_references(handler),
        cache,
    ):
        visit(reference, depth=0, stack=())


def _root_wide_user_config_profiles(
    documents: tuple[DocumentFlowInput, ...],
    profiles: dict[str, UserConfigProfile],
    local_file_cache: dict[str, str],
    python_ast_cache_key: str | None,
) -> dict[str, UserConfigProfile]:
    uses_by_root = {
        root: {key: _UserConfigUse() for key in profile.sensitive_keys}
        for root, profile in profiles.items()
    }
    for document in documents:
        for handler in document.handlers:
            registration = handler.registration
            if (
                not registration.runnable
                or registration.event_status != "known"
                or registration.handler_status != "supported"
            ):
                continue
            root = registration.execution_root or ""
            profile = profiles.get(root)
            uses = uses_by_root.get(root)
            if profile is None or uses is None:
                continue
            if registration.handler_type == "command" and handler.command is not None:
                _record_config_command(handler.command, handler.args, profile, uses)
                _record_reachable_config_uses(
                    handler,
                    profile,
                    uses,
                    local_file_cache,
                    python_ast_cache_key,
                )
            elif registration.handler_type == "http":
                _record_config_http(handler, profile, uses)
    result: dict[str, UserConfigProfile] = {}
    for root, profile in profiles.items():
        uses = uses_by_root[root]
        authentication_only = frozenset(
            key for key, use in uses.items() if not use.has_other_use and len(use.origins) == 1
        )
        result[root] = UserConfigProfile(
            profile.sensitive_keys,
            profile.sensitive_environment_names,
            authentication_only,
        )
    return result


def analyze_documents(
    documents: tuple[DocumentFlowInput, ...],
    *,
    local_file_cache: dict[str, str],
    user_config_by_root: dict[str, UserConfigProfile] | None = None,
    python_ast_cache_key: str | None = None,
) -> FlowBatch:
    """Analyze sorted hook documents without reading beyond the local cache."""
    profiles = _root_wide_user_config_profiles(
        documents,
        user_config_by_root or {},
        local_file_cache,
        python_ast_cache_key,
    )
    traversal = _TraversalSession(local_file_cache, python_ast_cache_key)
    findings: list[OwnedFlowFinding] = []
    for document in documents:
        owner = FlowWorkRef(document.source_path)
        for ordinal, handler in enumerate(document.handlers):
            registration = handler.registration
            if (
                not registration.runnable
                or registration.event_status != "known"
                or registration.handler_status != "supported"
            ):
                continue
            profile = profiles.get(registration.execution_root or "")
            if registration.handler_type == "http":
                destination = _destination_for_url(handler.url)
                if destination is DestinationClass.LOOPBACK:
                    continue
                source_kind = _EVENT_SOURCES.get(registration.event)
                if source_kind is None:
                    source_kind = _header_environment_source(handler, profile)
                if source_kind is None:
                    continue
                findings.append(
                    OwnedFlowFinding(
                        owner,
                        _bh2_finding(
                            document,
                            handler,
                            ordinal,
                            source_kind=source_kind,
                            transport=TransportKind.HTTP,
                            destination=destination,
                        ),
                    )
                )
                continue
            if registration.handler_type != "command":
                continue
            event_taint = _EVENT_SOURCES.get(registration.event)
            for hit in _analyze_command(
                handler,
                event_taint=event_taint,
                profile=profile,
            ):
                findings.append(
                    OwnedFlowFinding(
                        owner,
                        _bh2_finding(
                            document,
                            handler,
                            ordinal,
                            source_kind=hit.source_kind,
                            transport=hit.transport,
                            destination=hit.destination,
                            sink_line=registration.source_line + hit.line - 1,
                        ),
                    )
                )
            findings.extend(
                traversal.analyze_handler(
                    document,
                    handler,
                    ordinal,
                    event_taint=event_taint,
                    profile=profile,
                )
            )
    failed_components = {
        result.ref
        for result in traversal.work.values()
        if result.outcome is not LedgerOutcome.COMPLETED
    }
    return FlowBatch(
        tuple(finding for finding in findings if finding.owner not in failed_components),
        tuple(traversal.work.values()),
    )
