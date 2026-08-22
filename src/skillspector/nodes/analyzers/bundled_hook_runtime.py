# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded runtime normalization for Claude Code hook declarations.

The public analyzer owns discovery and cache semantics.  This module is deliberately
pure: it converts one event, matcher group, and handler into a payload-free record
used by BH1 aggregation.  It never executes hooks or follows referenced payloads.
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Final
from urllib.parse import urlsplit

_SCHEMA: Final = "skillspector.bundled_hook.v1"
_KNOWN_HANDLER_TYPES: Final = frozenset({"command", "http", "mcp_tool", "prompt", "agent"})

_ALL_HANDLER_EVENTS: Final = frozenset(
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
_COMMAND_HTTP_MCP_EVENTS: Final = frozenset(
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
        "PostCompact",
        "PreCompact",
        "SessionEnd",
        "StopFailure",
        "SubagentStart",
        "WorktreeCreate",
        "WorktreeRemove",
    }
)
_COMMAND_MCP_EVENTS: Final = frozenset({"SessionStart", "Setup"})
_KNOWN_EVENTS: Final = _ALL_HANDLER_EVENTS | _COMMAND_HTTP_MCP_EVENTS | _COMMAND_MCP_EVENTS
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
_TOOL_IF_EVENTS: Final = frozenset(
    {
        "PermissionDenied",
        "PermissionRequest",
        "PostToolUse",
        "PostToolUseFailure",
        "PreToolUse",
    }
)
_CONTROL_OR_INPUT_EVENTS: Final = frozenset(
    {
        "Elicitation",
        "ElicitationResult",
        "PermissionDenied",
        "PermissionRequest",
        "PreToolUse",
        "Stop",
        "SubagentStop",
        "TaskCompleted",
        "TeammateIdle",
        "UserPromptExpansion",
        "UserPromptSubmit",
    }
)
_SKILL_SOURCE_KINDS: Final = frozenset(
    {
        "marketplace_plugin_skill",
        "plugin_default_skill",
        "plugin_manifest_skill",
        "plugin_root_skill",
        "project_skill",
        "root_skill",
    }
)
_TRANSPORT_EXECUTABLES: Final = frozenset(
    {
        "aws",
        "az",
        "curl",
        "dig",
        "gcloud",
        "host",
        "mail",
        "mailx",
        "nc",
        "ncat",
        "netcat",
        "nslookup",
        "rclone",
        "rsync",
        "scp",
        "sftp",
        "ssh",
        "socat",
        "wget",
    }
)
_PERMISSION_RULE: Final = re.compile(r"^([A-Za-z0-9_:\-]+)\((.*)\)$", re.DOTALL)
_GENERAL_EXACT_MATCHER: Final = re.compile(r"^[A-Za-z0-9_\- ,|]+$")
_NARROW_EXACT_MATCHER: Final = re.compile(r"^[A-Za-z0-9_|]+$")
_ENTRYPOINT_TOKEN: Final = re.compile(
    r"\$\{CLAUDE_(PLUGIN_ROOT|PROJECT_DIR)\}/([A-Za-z0-9_./@%+=,:~-]+)"
)
_ENTRYPOINT_PLACEHOLDER: Final = re.compile(r"\$\{CLAUDE_(PLUGIN_ROOT|PROJECT_DIR)\}")
_SUBSTITUTION: Final = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\.[^{}]+)?\}")
_SENSITIVE_FIELDS_BY_EVENT: Final = {
    "UserPromptSubmit": frozenset({"prompt"}),
    "UserPromptExpansion": frozenset({"prompt", "command_args"}),
    "PreToolUse": frozenset({"tool_input"}),
    "PermissionRequest": frozenset({"tool_input"}),
    "PermissionDenied": frozenset({"tool_input", "reason"}),
    "PostToolUse": frozenset({"tool_input", "tool_response"}),
    "PostToolUseFailure": frozenset({"tool_input", "error"}),
    "PostToolBatch": frozenset({"tool_calls"}),
    "MessageDisplay": frozenset({"delta"}),
    "TaskCreated": frozenset({"task_subject", "task_description"}),
    "TaskCompleted": frozenset({"task_subject", "task_description"}),
    "Stop": frozenset({"last_assistant_message"}),
    "SubagentStop": frozenset({"last_assistant_message"}),
    "StopFailure": frozenset({"error", "error_details", "last_assistant_message"}),
    "PreCompact": frozenset({"custom_instructions"}),
    "PostCompact": frozenset({"compact_summary"}),
    "Elicitation": frozenset({"message", "requested_schema"}),
    "ElicitationResult": frozenset({"content"}),
}
_MAX_STRUCTURE_NODES: Final = 2048


@dataclass(frozen=True)
class HookRegistration:
    """Payload-free runtime classification for one declared handler."""

    event: str = field(repr=False)
    event_status: str
    matcher_kind: str
    matcher_effective: str = field(repr=False)
    handler_type: str
    handler_status: str
    handler_digest: str
    if_rule_present: bool
    if_status: str
    if_arguments_proven: bool
    runnable: bool
    runtime_status: str
    once: bool
    async_: bool
    async_rewake: bool
    command_mode: str
    args_present: bool
    executable_is_literal: bool = field(repr=False)
    shell_effective: str
    activation_lifetime: str
    source_kind: str
    source_path: str = field(repr=False)
    source_line: int
    chain_digest: str
    matches_all: bool
    watch_path_count: int
    ambient: bool
    known_transport: bool
    http_destination: str
    mcp_sensitive_forward: bool
    execution_root: str | None = field(repr=False)
    entrypoint_references: tuple[str, ...] = field(repr=False)


def _digest(domain: str, value: str) -> str:
    payload = f"{_SCHEMA}\0{domain}\0{value}".encode()
    return f"sha256:{sha256(payload).hexdigest()}"


def _canonical_handler(handler: dict[str, object]) -> str:
    return json.dumps(
        handler,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _supported_types(event: str) -> frozenset[str]:
    if event in _ALL_HANDLER_EVENTS:
        return _KNOWN_HANDLER_TYPES
    if event in _COMMAND_HTTP_MCP_EVENTS:
        return frozenset({"command", "http", "mcp_tool"})
    if event in _COMMAND_MCP_EVENTS:
        return frozenset({"command", "mcp_tool"})
    return frozenset()


def _required_fields_valid(handler_type: str, handler: dict[str, object]) -> bool:
    if handler_type == "command":
        command = handler.get("command")
        return isinstance(command, str) and command != ""
    if handler_type == "http":
        url = handler.get("url")
        return isinstance(url, str) and bool(url.strip())
    if handler_type == "mcp_tool":
        return isinstance(handler.get("server"), str) and isinstance(handler.get("tool"), str)
    if handler_type in {"prompt", "agent"}:
        return isinstance(handler.get("prompt"), str)
    return False


def _split_exact_matcher(matcher: str, *, narrow: bool) -> str:
    separator = r"\|" if narrow else r"[|,]"
    values = [value.strip() for value in re.split(separator, matcher)]
    return ",".join(value for value in values if value)


def _matcher_semantics(event: str, matcher_group: dict[str, object]) -> tuple[str, str, bool, int]:
    if event in _NO_MATCHER_EVENTS:
        return "ignored", "broad", True, 0

    present = "matcher" in matcher_group
    matcher = matcher_group.get("matcher")
    if event == "FileChanged":
        if not present or matcher == "":
            return "broad", "broad", True, 0
        if not isinstance(matcher, str):
            return "invalid", "unconfirmed", False, 0
        watch_path_count = len([part for part in matcher.split("|") if part])
        return "literal", matcher, matcher == "*", watch_path_count

    if not present:
        return "broad", "broad", True, 0
    if not isinstance(matcher, str):
        return "invalid", "unconfirmed", False, 0
    if matcher in {"", "*"}:
        return "broad", "broad", True, 0

    narrow = event == "StopFailure"
    exact_pattern = _NARROW_EXACT_MATCHER if narrow else _GENERAL_EXACT_MATCHER
    if exact_pattern.fullmatch(matcher):
        return "exact_list", _split_exact_matcher(matcher, narrow=narrow), False, 0
    return "regex", matcher, matcher in {".*", "^.*$"}, 0


def _handler_identity(handler: dict[str, object]) -> tuple[str, str, str]:
    raw_type = handler.get("type")
    if not isinstance(raw_type, str):
        return "unknown", "invalid", _digest("handler", _canonical_handler(handler))
    handler_type = raw_type if raw_type in _KNOWN_HANDLER_TYPES else "unknown"
    if handler_type == "unknown":
        return handler_type, "unknown", _digest("handler", _canonical_handler(handler))
    status = "supported" if _required_fields_valid(handler_type, handler) else "invalid"
    return handler_type, status, _digest("handler", _canonical_handler(handler))


def _command_semantics(
    handler_type: str, handler: dict[str, object]
) -> tuple[str, bool, bool, str, bool]:
    if handler_type != "command":
        return "none", False, False, "none", True
    args_present = "args" in handler
    if args_present:
        args = handler.get("args")
        args_valid = isinstance(args, list) and all(isinstance(value, str) for value in args)
        return "exec", True, True, "none", args_valid
    shell = handler.get("shell")
    if "shell" in handler and (not isinstance(shell, str) or shell not in {"bash", "powershell"}):
        return "shell", False, False, "unconfirmed", False
    return "shell", False, False, shell if isinstance(shell, str) else "default", True


def _plugin_source(source_kind: str) -> bool:
    return source_kind.startswith("plugin_") or source_kind.startswith("marketplace_plugin_")


def _if_semantics(
    event: str,
    matcher_kind: str,
    matcher_effective: str,
    handler: dict[str, object],
) -> tuple[bool, str, bool]:
    if "if" not in handler:
        return False, "absent", True
    if event not in _TOOL_IF_EVENTS:
        return True, "non_tool_dormant", False

    raw_rule = handler.get("if")
    if not isinstance(raw_rule, str):
        return True, "fail_open", True
    parsed = _PERMISSION_RULE.fullmatch(raw_rule)
    if parsed is None:
        return True, "fail_open", True
    rule_tool, argument_rule = parsed.groups()

    if matcher_kind == "regex":
        return True, "fail_open", True
    if matcher_kind in {"invalid"}:
        return True, "fail_open", True
    if matcher_kind in {"broad", "ignored"}:
        return (
            True,
            "all_tool" if argument_rule == "*" else "compatible_conditional",
            True,
        )

    matcher_tools = set(matcher_effective.split(","))
    if rule_tool not in matcher_tools:
        return True, "disjoint", False
    return (
        True,
        "all_tool" if argument_rule == "*" else "compatible_conditional",
        True,
    )


def _normalized_executable(value: str) -> str:
    executable = PurePosixPath(value.replace("\\", "/")).name.lower()
    return executable[:-4] if executable.endswith(".exe") else executable


def _env_split_string_source(words: tuple[str, ...]) -> str | None:
    """Return an env -S command string before ordinary argv unwrapping loses it."""
    env_index: int | None = None
    for index, word in enumerate(words):
        if _normalized_executable(word) != "env":
            continue
        if index == 0:
            env_index = index
            break
        executable, consumed = _unwrap_executable(words[: index + 1])
        if executable is None and consumed == index + 1:
            env_index = index
            break
    if env_index is None:
        return None
    assignment = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
    index = env_index + 1
    while index < len(words):
        value = words[index]
        option, equals, inline_value = value.partition("=")
        if option in {"-S", "--split-string"}:
            return inline_value if equals else words[index + 1] if index + 1 < len(words) else ""
        if option in {"-C", "-u", "--chdir", "--unset"}:
            index += 1 if equals else 2
            continue
        if value == "--":
            return None
        if value.startswith("-") or assignment.match(value):
            index += 1
            continue
        return None
    return None


def _nested_interpreter_sources(command: str, args: list[str]) -> tuple[str, ...]:
    executable = _normalized_executable(command)
    option_names: tuple[str, ...]
    if executable in {"bash", "sh", "zsh"}:
        option_names = ("-c",)
    elif executable in {"powershell", "pwsh"}:
        option_names = ("-command", "-c")
    elif executable == "cmd":
        option_names = ("/c",)
    else:
        return ()
    for index, value in enumerate(args[:-1]):
        if value.lower() in option_names:
            return (args[index + 1],)
    return ()


def _known_command_transport(handler: dict[str, object], command_mode: str) -> bool:
    command = handler.get("command")
    if not isinstance(command, str):
        return False
    if command_mode == "exec":
        args = handler.get("args")
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            return False
        words = (command, *args)
        split_source = _env_split_string_source(words)
        if split_source is not None:
            return not split_source or _known_shell_transport(split_source, depth=1)
        effective = _effective_argv(words)
        if effective is None:
            return False
        executable, args = effective
        if _normalized_executable(executable) in _TRANSPORT_EXECUTABLES:
            return True
        return any(
            _known_shell_transport(source, depth=1)
            for source in _nested_interpreter_sources(executable, list(args))
        )
    return _known_shell_transport(command)


def _shell_segments(source: str) -> tuple[str, ...]:
    """Split simple shell command boundaries without treating quoted text as code."""
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    comment = False
    for character in source:
        if comment:
            if character == "\n":
                comment = False
                if current:
                    segments.append("".join(current))
                    current = []
            continue
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\" and quote != "'":
            current.append(character)
            escaped = True
            continue
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
            continue
        if character == "#" and (not current or current[-1].isspace()):
            comment = True
            continue
        if character in ";|&()\n":
            if current:
                segments.append("".join(current))
                current = []
            continue
        current.append(character)
    if current:
        segments.append("".join(current))
    return tuple(segments)


def _shell_words(source: str) -> tuple[tuple[tuple[str, ...], ...], bool]:
    parsed: list[tuple[str, ...]] = []
    malformed = False
    for segment in _shell_segments(source):
        try:
            parsed.append(tuple(shlex.split(segment, comments=True, posix=True)))
        except ValueError:
            malformed = True
    return tuple(parsed), malformed


def _unwrap_executable(words: tuple[str, ...]) -> tuple[str | None, int]:
    """Return the effective command word and its index after documented wrappers."""
    assignment = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
    control = {"do", "elif", "else", "fi", "if", "then", "until", "while"}
    index = 0
    while index < len(words) and (assignment.match(words[index]) or words[index] in control):
        index += 1
    while index < len(words):
        word = _normalized_executable(words[index])
        if word == "builtin":
            index += 1
            if index < len(words) and words[index] == "--":
                index += 1
            continue
        if word == "command":
            index += 1
            if index < len(words) and words[index] == "--":
                index += 1
            while index < len(words) and words[index] == "-p":
                index += 1
            continue
        if word == "nohup":
            index += 1
            if index < len(words) and words[index] == "--":
                index += 1
            continue
        if word == "exec":
            index += 1
            while index < len(words):
                value = words[index]
                if value == "--":
                    index += 1
                    break
                if value == "-a":
                    index += 2
                    continue
                if value in {"-c", "-l"}:
                    index += 1
                    continue
                break
            continue
        if word == "env":
            index += 1
            while index < len(words):
                value = words[index]
                option = value.split("=", 1)[0]
                if value == "--":
                    index += 1
                    break
                if option in {"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}:
                    index += 1 if "=" in value else 2
                elif value.startswith("--unset=") or value.startswith("--chdir="):
                    index += 1
                elif value.startswith("--split-string="):
                    index += 1
                elif value.startswith("-") or assignment.match(value):
                    index += 1
                else:
                    break
            continue
        if word == "sudo":
            index += 1
            consuming = {
                "--chdir",
                "--chroot",
                "--close-from",
                "--command-timeout",
                "--group",
                "--host",
                "--login-class",
                "--other-user",
                "--prompt",
                "--role",
                "--type",
                "--user",
                "-C",
                "-D",
                "-g",
                "-h",
                "-p",
                "-r",
                "-R",
                "-t",
                "-T",
                "-u",
                "-U",
                "-c",
            }
            while index < len(words) and words[index].startswith("-"):
                value = words[index]
                if value == "--":
                    index += 1
                    break
                option = value.split("=", 1)[0]
                index += 1
                if option in consuming and "=" not in value:
                    index += 1
            continue
        if word == "timeout":
            index += 1
            consuming = {"--kill-after", "--signal", "-k", "-s"}
            while index < len(words) and words[index].startswith("-"):
                value = words[index]
                if value == "--":
                    index += 1
                    break
                option = value.split("=", 1)[0]
                index += 1
                if option in consuming and "=" not in value:
                    index += 1
            if index < len(words):
                index += 1
            continue
        return words[index], index
    return None, index


def _effective_argv(words: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
    """Return one option-normalized executable and its literal argument vector."""
    executable, index = _unwrap_executable(words)
    if executable is None:
        return None
    return executable, words[index + 1 :]


def _shell_command_executables(source: str) -> tuple[str, ...]:
    executables: list[str] = []
    word_groups, _malformed = _shell_words(source)
    for words in word_groups:
        effective = _effective_argv(words)
        if effective is not None:
            executables.append(_normalized_executable(effective[0]))
    return tuple(executables)


def _known_shell_transport(source: str, *, depth: int = 0) -> bool:
    word_groups, malformed = _shell_words(source)
    if malformed:
        return True
    for words in word_groups:
        split_source = _env_split_string_source(words)
        if split_source is not None:
            if not split_source or _known_shell_transport(split_source, depth=depth + 1):
                return True
            continue
        effective = _effective_argv(words)
        if effective is None:
            continue
        executable, args = effective
        normalized = "." if executable == "." else _normalized_executable(executable)
        if normalized in _TRANSPORT_EXECUTABLES:
            return True
        if depth >= 2:
            continue
        nested = _nested_interpreter_sources(executable, list(args))
        if any(_known_shell_transport(value, depth=depth + 1) for value in nested):
            return True
    return False


def _http_destination(handler: dict[str, object]) -> str:
    url = handler.get("url")
    if not isinstance(url, str):
        return "unconfirmed"
    if "${" in url or "$" in url:
        return "dynamic"
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError:
        return "dynamic"
    if not hostname:
        return "dynamic"
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return "loopback"
    try:
        address = ipaddress.ip_address(normalized)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if address.is_loopback:
            return "loopback"
    except ValueError:
        pass
    return "remote"


def _contains_sensitive_substitution(value: object, event: str) -> bool:
    sensitive_fields = _SENSITIVE_FIELDS_BY_EVENT.get(event, frozenset())
    if not sensitive_fields:
        return False
    pending: list[object] = [value]
    seen = 0
    while pending and seen < _MAX_STRUCTURE_NODES:
        current = pending.pop()
        seen += 1
        if isinstance(current, str):
            if any(match.group(1) in sensitive_fields for match in _SUBSTITUTION.finditer(current)):
                return True
        elif isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return bool(pending)


def _safe_entrypoint_reference(scope: str, relative: str) -> str:
    parsed = PurePosixPath(relative.strip())
    normalized = parsed.as_posix()
    if (
        "\x00" in relative
        or "\\" in relative
        or "${" in relative
        or parsed.is_absolute()
        or ".." in parsed.parts
        or any(len(part) >= 2 and part[1] == ":" for part in parsed.parts)
        or normalized in {"", "."}
    ):
        return f"{scope.lower()}:invalid"
    return f"{scope.lower()}:{normalized}"


def _references_in_value(value: str) -> tuple[str, ...]:
    match = _ENTRYPOINT_TOKEN.fullmatch(value)
    if match is not None:
        return (_safe_entrypoint_reference(*match.groups()),)
    scopes = tuple(dict.fromkeys(_ENTRYPOINT_PLACEHOLDER.findall(value)))
    return tuple(f"{scope.lower()}:invalid" for scope in scopes)


def _invalid_placeholder_references(values: tuple[str, ...]) -> tuple[str, ...]:
    scopes = tuple(
        dict.fromkeys(scope for value in values for scope in _ENTRYPOINT_PLACEHOLDER.findall(value))
    )
    return tuple(f"{scope.lower()}:invalid" for scope in scopes)


def _interpreter_entrypoint_operands(
    executable: str, arguments: tuple[str, ...]
) -> tuple[tuple[str, ...], bool]:
    """Return code-loading operands, plus whether option parsing was complete."""
    normalized = _normalized_executable(executable)
    operands: list[str] = []
    index = 0

    if normalized in {"python", "python3"}:
        value_options = {"--check-hash-based-pycs", "-W", "-X"}
        flag_options = {
            "--help",
            "--help-all",
            "--help-env",
            "--help-xoptions",
            "--version",
            "-b",
            "-B",
            "-d",
            "-E",
            "-h",
            "-i",
            "-I",
            "-O",
            "-OO",
            "-P",
            "-q",
            "-R",
            "-s",
            "-S",
            "-u",
            "-v",
            "-V",
            "-x",
        }
        while index < len(arguments):
            value = arguments[index]
            if value == "--":
                return ((*operands, *arguments[index + 1 : index + 2]), True)
            if value in {"-c", "-m"} or value.startswith(("-c=", "-m=")):
                return tuple(operands), True
            option = value.split("=", 1)[0]
            if option in value_options:
                if "=" in value or (option in {"-W", "-X"} and value != option):
                    index += 1
                elif index + 1 < len(arguments):
                    index += 2
                else:
                    return tuple(operands), False
                continue
            if value in flag_options or (value.startswith(("-W", "-X")) and len(value) > 2):
                index += 1
                continue
            if value.startswith("-"):
                return tuple(operands), False
            operands.append(value)
            return tuple(operands), True
        return tuple(operands), True

    if normalized == "node":
        code_value_options = {
            "--experimental-loader",
            "--import",
            "--loader",
            "--require",
            "-r",
        }
        value_options = code_value_options | {
            "--conditions",
            "--diagnostic-dir",
            "--env-file",
            "--env-file-if-exists",
            "--icu-data-dir",
            "--openssl-config",
            "--redirect-warnings",
            "--report-directory",
            "--report-filename",
            "--title",
        }
        flag_options = {
            "--check",
            "--experimental-strip-types",
            "--experimental-transform-types",
            "--frozen-intrinsics",
            "--help",
            "--no-addons",
            "--no-deprecation",
            "--no-warnings",
            "--preserve-symlinks",
            "--preserve-symlinks-main",
            "--test",
            "--trace-deprecation",
            "--trace-warnings",
            "--version",
            "-c",
            "-h",
            "-v",
        }
        while index < len(arguments):
            value = arguments[index]
            if value == "--":
                return ((*operands, *arguments[index + 1 : index + 2]), True)
            if value in {"--eval", "--print", "-e", "-p"} or value.startswith(
                ("--eval=", "--print=", "-e=", "-p=")
            ):
                return tuple(operands), True
            option = value.split("=", 1)[0]
            if option in value_options:
                if "=" in value:
                    option_value = value.split("=", 1)[1]
                    index += 1
                elif index + 1 < len(arguments):
                    option_value = arguments[index + 1]
                    index += 2
                else:
                    return tuple(operands), False
                if option in code_value_options:
                    operands.append(option_value)
                continue
            if value in flag_options or value.startswith(
                ("--inspect=", "--inspect-brk=", "--stack-trace-limit=")
            ):
                index += 1
                continue
            if value.startswith("-"):
                return tuple(operands), False
            operands.append(value)
            return tuple(operands), True
        return tuple(operands), True

    return (), True


def _shell_entrypoint_references(source: str, *, depth: int = 0) -> tuple[str, ...]:
    references: list[str] = []
    pending_root_scope: str | None = None
    word_groups, _malformed = _shell_words(source)
    for words in word_groups:
        effective = _effective_argv(words)
        if effective is None:
            references.extend(_invalid_placeholder_references(words))
            continue
        executable, effective_arguments = effective
        arguments = list(effective_arguments)
        normalized = "." if executable == "." else _normalized_executable(executable)
        direct = _references_in_value(executable)
        if direct:
            references.extend(direct)
            pending_root_scope = None
            continue
        if normalized == "cd" and arguments:
            root_match = re.fullmatch(r"\$\{CLAUDE_(PLUGIN_ROOT|PROJECT_DIR)\}/?", arguments[0])
            pending_root_scope = root_match.group(1) if root_match else None
            continue
        if pending_root_scope is not None:
            candidates = arguments if normalized in {"node", "python", "python3"} else [executable]
            relative = next(
                (
                    value.removeprefix("./")
                    for value in candidates
                    if value
                    and not value.startswith("-")
                    and not value.startswith("/")
                    and "${" not in value
                ),
                None,
            )
            if relative is not None:
                references.append(_safe_entrypoint_reference(pending_root_scope, relative))
            pending_root_scope = None
        operand_sources: list[str] = []
        modeled = True
        if normalized in {".", "source"}:
            operand_sources = arguments[:1]
        elif normalized in {"node", "python", "python3"}:
            operands, modeled = _interpreter_entrypoint_operands(executable, tuple(arguments))
            operand_sources = list(operands)
        elif normalized in {"bash", "sh", "zsh", "powershell", "pwsh", "cmd"}:
            nested = _nested_interpreter_sources(executable, arguments)
            if nested and depth < 2:
                for nested_source in nested:
                    references.extend(_shell_entrypoint_references(nested_source, depth=depth + 1))
            else:
                operand_sources = [value for value in arguments if not value.startswith("-")][:1]
        for operand in operand_sources:
            references.extend(_references_in_value(operand))
        if not modeled:
            references.extend(_invalid_placeholder_references(tuple(arguments)))
    return tuple(dict.fromkeys(references))


def _entrypoint_references(handler: dict[str, object], command_mode: str) -> tuple[str, ...]:
    if command_mode == "none":
        return ()
    command = handler.get("command")
    if not isinstance(command, str):
        return ()
    if command_mode == "shell":
        return _shell_entrypoint_references(command)
    args = handler.get("args")
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        return _references_in_value(command)
    words = (command, *args)
    split_source = _env_split_string_source(words)
    if split_source is not None:
        if not split_source:
            return _invalid_placeholder_references(words)
        return _shell_entrypoint_references(split_source, depth=1)
    effective = _effective_argv(words)
    if effective is None:
        return _invalid_placeholder_references(words)
    executable, effective_arguments = effective
    references = list(_references_in_value(executable))
    normalized = _normalized_executable(executable)
    if normalized in {"bash", "cmd", "powershell", "pwsh", "sh", "zsh"}:
        nested = _nested_interpreter_sources(executable, list(effective_arguments))
        if nested:
            for nested_source in nested:
                references.extend(_shell_entrypoint_references(nested_source, depth=1))
        else:
            operand = next(
                (value for value in effective_arguments if not value.startswith("-")), None
            )
            if operand is not None:
                references.extend(_references_in_value(operand))
    elif normalized in {"node", "python", "python3"}:
        operands, modeled = _interpreter_entrypoint_operands(executable, effective_arguments)
        for operand in operands:
            references.extend(_references_in_value(operand))
        if not modeled:
            references.extend(_invalid_placeholder_references(effective_arguments))
    return tuple(dict.fromkeys(references))


def normalize_registration(
    event: str,
    matcher_group: dict[str, object],
    handler: dict[str, object],
    *,
    source_kind: str,
    activation_lifetime: str,
    source_line: int,
    source_path: str = "",
    execution_root: str | None = None,
    runtime_confirmed: bool = True,
) -> HookRegistration:
    """Normalize one hook registration without retaining executable payload text."""
    if source_kind == "project_agent" and event == "Stop":
        event = "SubagentStop"
    event_status = "known" if event in _KNOWN_EVENTS else "unknown"
    matcher_kind, matcher_effective, matches_all, watch_path_count = _matcher_semantics(
        event, matcher_group
    )
    handler_type, handler_status, handler_digest = _handler_identity(handler)
    if event_status == "known" and handler_status == "supported":
        if handler_type not in _supported_types(event):
            handler_status = "unsupported"

    command_mode, args_present, executable_is_literal, shell_effective, args_valid = (
        _command_semantics(handler_type, handler)
    )
    if handler_status == "supported" and not args_valid:
        handler_status = "invalid"

    if_rule_present, if_status, if_runnable = _if_semantics(
        event, matcher_kind, matcher_effective, handler
    )
    valid_runtime = (
        event_status == "known" and handler_status == "supported" and matcher_kind != "invalid"
    )
    runnable = valid_runtime and if_runnable
    runtime_status = "runnable" if runnable else "unconfirmed"
    if valid_runtime and if_status in {"non_tool_dormant", "disjoint"}:
        runtime_status = "dormant"
    elif valid_runtime and if_status == "fail_open":
        runtime_status = "fail_open"

    if (
        handler_type == "command"
        and command_mode == "shell"
        and _plugin_source(source_kind)
        and isinstance(handler.get("command"), str)
        and "${user_config." in str(handler["command"])
    ):
        runnable = False
        runtime_status = "rejected"

    once = source_kind in _SKILL_SOURCE_KINDS and handler.get("once") is True
    async_rewake = handler_type == "command" and handler.get("asyncRewake") is True
    async_ = handler_type == "command" and (handler.get("async") is True or async_rewake)
    if not runtime_confirmed:
        runnable = False
        runtime_status = "unconfirmed"
    ambient = runnable and matches_all
    known_transport = handler_type == "command" and _known_command_transport(handler, command_mode)
    http_destination = _http_destination(handler) if handler_type == "http" else "none"
    mcp_sensitive_forward = handler_type == "mcp_tool" and _contains_sensitive_substitution(
        handler.get("input"), event
    )
    entrypoint_references = _entrypoint_references(handler, command_mode)

    chain_digest = _digest(
        "registration",
        "\0".join(
            (
                source_kind,
                activation_lifetime,
                str(source_line),
                event,
                event_status,
                matcher_kind,
                matcher_effective,
                handler_type,
                handler_status,
                handler_digest,
                if_status,
                command_mode,
                execution_root or "",
                *entrypoint_references,
            )
        ),
    )
    return HookRegistration(
        event=event,
        event_status=event_status,
        matcher_kind=matcher_kind,
        matcher_effective=matcher_effective,
        handler_type=handler_type,
        handler_status=handler_status,
        handler_digest=handler_digest,
        if_rule_present=if_rule_present,
        if_status=if_status,
        if_arguments_proven=False,
        runnable=runnable,
        runtime_status=runtime_status,
        once=once,
        async_=async_,
        async_rewake=async_rewake,
        command_mode=command_mode,
        args_present=args_present,
        executable_is_literal=executable_is_literal,
        shell_effective=shell_effective,
        activation_lifetime=activation_lifetime,
        source_kind=source_kind,
        source_path=source_path,
        source_line=max(1, source_line),
        chain_digest=chain_digest,
        matches_all=matches_all,
        watch_path_count=watch_path_count,
        ambient=ambient,
        known_transport=known_transport,
        http_destination=http_destination,
        mcp_sensitive_forward=mcp_sensitive_forward,
        execution_root=execution_root,
        entrypoint_references=entrypoint_references,
    )


def _key_parts(path: str) -> tuple[str, tuple[str, ...]]:
    if "!/" in path:
        archive, member = path.rsplit("!/", 1)
        return f"{archive}!/", tuple(part for part in member.split("/") if part)
    return "", tuple(part for part in path.split("/") if part)


def _key_from_parts(namespace: str, parts: tuple[str, ...]) -> str:
    member = "/".join(parts)
    return f"{namespace}{member}" if namespace else member


def _join_cache_key(root: str, relative: str) -> str:
    namespace, root_parts = _key_parts(root)
    relative_parts = tuple(part for part in relative.split("/") if part)
    return _key_from_parts(namespace, (*root_parts, *relative_parts))


def entrypoint_is_resolved(registration: HookRegistration, known_paths: set[str]) -> bool:
    """Resolve a placeholder target within its project, plugin, or archive root."""
    references = registration.entrypoint_references
    if not references:
        return True
    root = registration.execution_root
    if root is None:
        return False
    for reference in references:
        scope, _, relative = reference.partition(":")
        if relative == "invalid":
            return False
        if scope == "project_dir" and _plugin_source(registration.source_kind):
            return False
        if scope == "plugin_root" and not _plugin_source(registration.source_kind):
            return False
        if _join_cache_key(root, relative) not in known_paths:
            return False
    return True


def registration_severity(registration: HookRegistration, known_paths: set[str]) -> str:
    """Classify one normalized registration for BH1 aggregation."""
    high = (
        (
            registration.event_status == "known"
            and registration.handler_status in {"unknown", "invalid"}
        )
        or registration.known_transport
        or registration.http_destination in {"remote", "dynamic"}
        or registration.mcp_sensitive_forward
        or not entrypoint_is_resolved(registration, known_paths)
    )
    if high:
        return "HIGH"
    if registration.once or not registration.runnable or registration.event_status == "unknown":
        return "LOW"
    if (
        registration.ambient
        or registration.http_destination == "loopback"
        or registration.event in _CONTROL_OR_INPUT_EVENTS
    ):
        return "MEDIUM"
    return "LOW"
