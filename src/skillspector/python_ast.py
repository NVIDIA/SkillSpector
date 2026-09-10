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

"""Shared, per-scan Python AST parsing and import-alias metadata.

The graph prewarms this module's cache before its analyzer branches fan out.
Consumers must treat returned ASTs as read-only; keeping parsing, syntax-error
handling, and import aliases together lets later scope-aware resolution extend
one stable interface.
"""

from __future__ import annotations

import ast
import codecs
import posixpath
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import RLock
from uuid import uuid4

# Keep this in sync with the existing static-analyzer size gate.  It lives here
# so prewarming does not parse files that AST consumers will skip anyway.
MAX_PYTHON_AST_SOURCE_CHARS = 1_000_000
# AST nodes can be substantially larger than their source.  Limit the total
# source retained as parsed trees for any one scan; files beyond this budget
# use the existing on-demand behavior rather than retaining unbounded memory.
MAX_PYTHON_AST_CACHE_SOURCE_CHARS = 8_000_000
PYTHON_SOURCE_EXTENSIONS = frozenset({".py", ".pyw"})
MAX_PYTHON_SHEBANG_CHARS = 512
_MAX_PYTHON_CONSUMED_SOURCE_SPELLING_CHARS = 4_096
# Linux used a 128-byte ``BINPRM_BUF_SIZE`` through 5.0 and has used 256
# bytes since 5.1.  Supported Python runtimes still run on both kernel lines,
# so classifications must agree with both truncation boundaries.
_LINUX_SHEBANG_BUFFER_SIZES = (128, 256)
_TRUSTED_ENV_PATHS = frozenset({"/bin/env", "/usr/bin/env"})
_PYTHON_INTERPRETER_BASENAME = re.compile(
    r"(?:python(?:[0-9]+(?:\.[0-9]+)*(?:d?m?u?|t?d?)"
    r"(?:-(?:intel64|32|dbg))?)?"
    r"|pypy(?:[0-9]+(?:\.[0-9]+)*)?)\Z"
)
_ENV_GNU_SPLIT_SHORT_PREFIX = re.compile(r"-[iv]*S")
_ENV_FREEBSD_SPLIT_SHORT_PREFIX = re.compile(r"-[iv-]*S")
_ENV_VARIABLE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_ENV_CHARACTER_ESCAPES = {
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}
_ENV_GNU_SHORT_FLAGS = frozenset({"i", "v"})
_ENV_FREEBSD_SHORT_FLAGS = frozenset({"-", "i", "v"})
_ENV_DARWIN_SHORT_FLAGS = frozenset({"-", "i", "v"})
_ENV_GNU_SHORT_OPERANDS = frozenset({"C", "a", "u"})
_ENV_FREEBSD_SHORT_OPERANDS = frozenset({"C", "L", "P", "U", "u"})
_ENV_DARWIN_SHORT_OPERANDS = frozenset({"C", "P", "u"})
_ENV_SPLIT_LONG_FLAGS = frozenset(
    {
        "--debug",
        "--ignore-environment",
        "--list-signal-handling",
    }
)
_ENV_SPLIT_LONG_OPERANDS = frozenset({"--argv0", "--chdir", "--env0-from", "--unset"})
_ENV_SPLIT_LONG_OPTIONAL_OPERANDS = frozenset(
    {
        "--block-signal",
        "--default-signal",
        "--ignore-signal",
    }
)
_ENV_SPLIT_LONG_OPTIONS = (
    _ENV_SPLIT_LONG_FLAGS
    | _ENV_SPLIT_LONG_OPERANDS
    | _ENV_SPLIT_LONG_OPTIONAL_OPERANDS
    | {"--split-string"}
)
_MAX_ENV_PARSE_STATES = MAX_PYTHON_SHEBANG_CHARS * 4


class PythonSourceClassification(StrEnum):
    """Confidence in Python source identity derived from bounded metadata."""

    PYTHON = "python"
    NON_PYTHON = "non_python"
    AMBIGUOUS = "ambiguous"


class _EnvPlatform(StrEnum):
    """Supported ``env`` grammar whose branches must remain coherent."""

    GNU = "gnu"
    FREEBSD = "freebsd"
    DARWIN = "darwin"


class _PythonInspectEnvironment(StrEnum):
    """Static knowledge about PYTHONINSPECT supplied by the env launcher.

    Ambient inherited variables are deliberately UNMENTIONED: without an
    explicit launcher action the scanner does not assume a TTY-driven REPL.
    Environment-file and login-class sources are UNKNOWN until a later exact
    assignment determines the final value.
    """

    UNMENTIONED = "unmentioned"
    ABSENT_OR_EMPTY = "absent_or_empty"
    NONEMPTY = "nonempty"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class _EnvArgument:
    """One split argument with projected text and runtime-expansion offsets."""

    text: str
    dynamic_offsets: tuple[int, ...] = ()
    retained_when_empty: bool = True


@dataclass(frozen=True, slots=True)
class _EnvExecution:
    """One selected env utility and its arguments before the script path."""

    utility: str
    arguments: tuple[_EnvArgument, ...] = ()
    python_inspect: _PythonInspectEnvironment = _PythonInspectEnvironment.UNMENTIONED


def _has_python_source_extension(path: str) -> bool:
    """Return whether the basename has an authoritative Python extension."""
    basename = path.replace("\\", "/").rsplit("/", 1)[-1]
    dot = basename.rfind(".")
    suffix = basename[dot:].casefold() if dot >= 0 else ""
    return suffix in PYTHON_SOURCE_EXTENSIONS


def _bounded_shebang_line(content: str | bytes | None) -> str | None:
    """Return one bounded shebang line, excluding its line ending."""
    if content is None:
        return None
    sample = content[: MAX_PYTHON_SHEBANG_CHARS + 1]
    if isinstance(sample, bytes):
        terminators = [offset for marker in (b"\n", b"\0") if (offset := sample.find(marker)) >= 0]
    else:
        terminators = [offset for marker in ("\n", "\0") if (offset := sample.find(marker)) >= 0]
    terminator = min(terminators, default=-1)
    if terminator < 0:
        if len(sample) > MAX_PYTHON_SHEBANG_CHARS:
            return None
        line = sample
    else:
        if terminator > MAX_PYTHON_SHEBANG_CHARS:
            return None
        line = sample[:terminator]
    if isinstance(line, bytes):
        decoded = line.decode("utf-8", errors="surrogateescape")
    else:
        decoded = line
    return decoded


def _has_overlong_shebang(content: str | bytes | None) -> bool:
    """Return whether a shebang line extends beyond the inspection bound."""
    if content is None:
        return False
    if isinstance(content, bytes):
        if not content.startswith(b"#!"):
            return False
        bytes_sample = content[: MAX_PYTHON_SHEBANG_CHARS + 1]
        return len(bytes_sample) > MAX_PYTHON_SHEBANG_CHARS and bytes_sample.find(b"\n") < 0
    if not content.startswith("#!"):
        return False
    text_sample = content[: MAX_PYTHON_SHEBANG_CHARS + 1]
    return len(text_sample) > MAX_PYTHON_SHEBANG_CHARS and text_sample.find("\n") < 0


def _linux_truncated_shebang(
    content: str | bytes | None,
    buffer_bytes: int,
) -> tuple[bool, bytes | None]:
    """Return one Linux ``BINPRM_BUF_SIZE`` view when its line is truncated."""
    if content is None:
        return False, None
    if isinstance(content, bytes):
        if not content.startswith(b"#!"):
            return False, None
        encoded = content[: buffer_bytes + 1]
    else:
        if not content.startswith("#!"):
            return False, None
        sample = content[: buffer_bytes + 1]
        try:
            encoded = sample.encode("utf-8", errors="surrogateescape")
        except UnicodeEncodeError:
            # An in-memory string that cannot be faithfully mapped back to
            # source bytes has no single kernel interpretation.
            return True, None
    terminators = [offset for marker in (b"\n", b"\0") if (offset := encoded.find(marker)) >= 0]
    line = encoded[: min(terminators)] if terminators else encoded
    if len(line) < buffer_bytes:
        return False, None
    # Linux reserves the last byte of BINPRM_BUF_SIZE as the exclusive end
    # marker.  Bytes at ``buffer_bytes - 1`` and later are therefore absent
    # from the interpreter line parsed by the kernel.
    return True, encoded[: buffer_bytes - 1] + b"\n"


def _is_python_interpreter(command: str) -> bool:
    """Return whether an absolute or PATH-resolved command names Python."""
    basename = command.rsplit("/", 1)[-1]
    return _PYTHON_INTERPRETER_BASENAME.fullmatch(basename) is not None


def _is_python_interpreter_filesystem_alias(command: str) -> bool:
    """Return whether a case/Unicode-insensitive volume may resolve Python."""
    basename = command.rsplit("/", 1)[-1]
    normalized = unicodedata.normalize("NFD", basename).casefold()
    return _PYTHON_INTERPRETER_BASENAME.fullmatch(normalized) is not None


def _is_trusted_env_filesystem_alias(command: str) -> bool:
    """Return whether filesystem/path aliases may resolve trusted ``env``."""
    if command.startswith("/"):
        # ``normpath`` intentionally preserves exactly two leading slashes;
        # collapse them here because that spelling is implementation-defined
        # and must remain a possible trusted-env branch.
        command = "/" + posixpath.normpath("/" + command.lstrip("/")).lstrip("/")
    normalized = unicodedata.normalize("NFD", command).casefold()
    return any(
        normalized == unicodedata.normalize("NFD", trusted).casefold()
        for trusted in _TRUSTED_ENV_PATHS
    )


def _split_env_arguments(value: str, platform: _EnvPlatform) -> list[_EnvArgument] | None:
    """Apply the bounded ``env -S`` grammar and preserve runtime uncertainty."""
    arguments: list[_EnvArgument] = []
    current: list[str] = []
    argument_started = False
    dynamic_offsets: list[int] = []
    single_quoted = False
    double_quoted = False
    index = 0

    def finish_argument() -> None:
        nonlocal argument_started, current, dynamic_offsets
        if argument_started or dynamic_offsets:
            arguments.append(
                _EnvArgument(
                    "".join(current),
                    tuple(dynamic_offsets),
                    retained_when_empty=argument_started,
                )
            )
            current = []
            argument_started = False
            dynamic_offsets = []

    while index < len(value):
        character = value[index]
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            argument_started = True
            index += 1
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            argument_started = True
            index += 1
            continue
        if character in " \t\n\r\v\f" and not single_quoted and not double_quoted:
            finish_argument()
            index += 1
            continue
        if character == "#" and not argument_started and not dynamic_offsets:
            break
        if character == "\\":
            if single_quoted and (index + 1 >= len(value) or value[index + 1] not in {"\\", "'"}):
                current.append(character)
                argument_started = True
                index += 1
                continue
            if index + 1 >= len(value):
                return None
            escaped = value[index + 1]
            if escaped in {'"', "#", "$", "'", "\\"}:
                current.append(escaped)
                argument_started = True
                index += 2
                continue
            if escaped in " \t\n\r\v\f" and platform is not _EnvPlatform.GNU:
                # FreeBSD env accepts escaped literal whitespace.  GNU env
                # rejects this spelling, but the successful BSD branch still
                # establishes execution intent and must not be skipped.
                current.append(escaped)
                argument_started = True
                index += 2
                continue
            if escaped == "_":
                if double_quoted:
                    current.append(" ")
                    argument_started = True
                else:
                    finish_argument()
                index += 2
                continue
            if escaped == "c":
                if double_quoted:
                    return None
                break
            replacement = _ENV_CHARACTER_ESCAPES.get(escaped)
            if replacement is None:
                return None
            current.append(replacement)
            argument_started = True
            index += 2
            continue
        if character == "$" and not single_quoted:
            variable = _ENV_VARIABLE.match(value, index)
            if variable is None:
                return None
            # Expansion values and even the presence of an unquoted standalone
            # argument depend on the runtime environment.  The projected text
            # remains useful for syntax validation, but callers must retain the
            # uncertainty rather than treating the empty branch as definitive.
            dynamic_offsets.append(len(current))
            index = variable.end()
            continue
        current.append(character)
        argument_started = True
        index += 1

    if single_quoted or double_quoted:
        return None
    finish_argument()
    return arguments


def _dynamic_env_argument_role(argument: _EnvArgument, platform: _EnvPlatform) -> str:
    """Return the only stable role available before a dynamic expansion."""
    first_dynamic = argument.dynamic_offsets[0]
    static_prefix = argument.text[:first_dynamic]
    if not static_prefix:
        return "ambiguous"
    if first_dynamic > 0 and not argument.text.startswith("-") and "=" in argument.text:
        return "assignment"
    if argument.text.startswith("-") and (
        platform is not _EnvPlatform.GNU or not argument.text.startswith("--")
    ):
        short_flags = {
            _EnvPlatform.GNU: _ENV_GNU_SHORT_FLAGS,
            _EnvPlatform.FREEBSD: _ENV_FREEBSD_SHORT_FLAGS,
            _EnvPlatform.DARWIN: _ENV_DARWIN_SHORT_FLAGS,
        }[platform]
        short_operands = {
            _EnvPlatform.GNU: _ENV_GNU_SHORT_OPERANDS,
            _EnvPlatform.FREEBSD: _ENV_FREEBSD_SHORT_OPERANDS,
            _EnvPlatform.DARWIN: _ENV_DARWIN_SHORT_OPERANDS,
        }[platform]
        cluster = static_prefix[1:]
        for position, option in enumerate(cluster):
            if option in short_flags:
                continue
            if option == "S":
                return "ambiguous"
            if option in short_operands:
                has_static_operand = bool(cluster[position + 1 :] or argument.text[first_dynamic:])
                return "option" if has_static_operand else "branching_option"
            return "invalid"
    if platform is _EnvPlatform.GNU and argument.text.startswith("--") and "=" in static_prefix:
        resolved = _resolve_env_long_option(argument.text)
        if resolved is None:
            return "invalid"
        option, attached_operand = resolved
        if option == "--split-string":
            return "ambiguous"
        if attached_operand is not None and option not in _ENV_SPLIT_LONG_FLAGS:
            return "option"
        return "invalid"
    return "ambiguous"


def _fixed_dynamic_utility(argument: _EnvArgument, *, options_ended: bool = False) -> str | None:
    """Return a utility basename fixed after all dynamic path segments."""
    first_dynamic = argument.dynamic_offsets[0]
    static_prefix = argument.text[:first_dynamic]
    static_suffix = argument.text[argument.dynamic_offsets[-1] :]
    if (
        not static_prefix
        or (static_prefix.startswith("-") and not options_ended)
        or "/" not in static_suffix
    ):
        return None
    basename = argument.text.rsplit("/", 1)[-1]
    return basename or None


def _resolve_env_long_option(argument: str) -> tuple[str, str | None] | None:
    """Resolve a GNU long option, including an unambiguous abbreviation."""
    option_name, separator, attached_operand = argument.partition("=")
    if option_name in _ENV_SPLIT_LONG_OPTIONS:
        resolved = option_name
    else:
        matches = [option for option in _ENV_SPLIT_LONG_OPTIONS if option.startswith(option_name)]
        if len(matches) != 1:
            return None
        resolved = matches[0]
    return resolved, attached_operand if separator else None


def _required_operand_continuations(
    arguments: tuple[_EnvArgument, ...], operand_index: int
) -> set[int]:
    """Return positions after consuming every viable required-operand branch."""
    continuations: set[int] = set()
    index = operand_index
    while index < len(arguments):
        argument = arguments[index]
        continuations.add(index + 1)
        if not argument.dynamic_offsets or argument.retained_when_empty:
            break
        # A standalone unquoted ${VAR} is absent when VAR is unset.  In that
        # branch the option consumes the next retained argument instead.
        index += 1
    return continuations


def _unset_operand_continuations(
    arguments: tuple[_EnvArgument, ...], operand_index: int
) -> tuple[set[int], set[int], bool]:
    """Return valid/invalid unset branches plus a missing-operand branch."""
    continuations: set[int] = set()
    invalid_continuations: set[int] = set()
    missing = operand_index >= len(arguments)
    index = operand_index
    while index < len(arguments):
        argument = arguments[index]
        can_succeed = "=" not in argument.text and bool(argument.text or argument.dynamic_offsets)
        if can_succeed:
            continuations.add(index + 1)
        else:
            invalid_continuations.add(index + 1)
        if argument.dynamic_offsets:
            # A substituted value can contain '='.  A retained dynamic-only
            # argument can also be empty; both make unsetenv reject before the
            # utility is selected.
            invalid_continuations.add(index + 1)
        if not argument.dynamic_offsets or argument.retained_when_empty:
            break
        index += 1
        if index >= len(arguments):
            missing = True
    return continuations, invalid_continuations, missing


@dataclass(frozen=True, slots=True)
class _EnvParseState:
    """One coherent platform parse, including GNU's deferred env actions."""

    pending: tuple[_EnvArgument, ...]
    index: int = 0
    options_ended: bool = False
    split_count: int = 0
    clear_environment: bool = False
    env0_from_file: bool = False
    invalid_unset: bool = False
    python_inspect: _PythonInspectEnvironment = _PythonInspectEnvironment.UNMENTIONED
    python_inspect_unset: bool = False
    login_environment: bool = False


def _nested_env_split_state(
    state: _EnvParseState,
    payload: _EnvArgument,
    tail_index: int,
    platform: _EnvPlatform,
) -> _EnvParseState | None:
    """Build a restarted parser state for one static nested split."""
    if payload.dynamic_offsets:
        return None
    split_arguments = _split_env_arguments(payload.text, platform)
    if split_arguments is None:
        return None
    return _EnvParseState(
        pending=tuple(split_arguments) + state.pending[tail_index:],
        split_count=state.split_count + 1,
        clear_environment=state.clear_environment,
        env0_from_file=state.env0_from_file,
        invalid_unset=state.invalid_unset,
        python_inspect=state.python_inspect,
        python_inspect_unset=state.python_inspect_unset,
        login_environment=state.login_environment,
    )


def _python_inspect_after_env_operand(
    platform: _EnvPlatform,
    option: str,
    operand: str,
    *,
    current: _PythonInspectEnvironment,
    unset_requested: bool,
    login_environment: bool,
) -> tuple[_PythonInspectEnvironment, bool, bool]:
    """Apply one exact env option that can determine launcher environment state."""
    if option == "u" and operand == "PYTHONINSPECT":
        unset_requested = True
        current = (
            _PythonInspectEnvironment.UNKNOWN
            if platform is _EnvPlatform.FREEBSD and login_environment
            else _PythonInspectEnvironment.ABSENT_OR_EMPTY
        )
    elif platform is _EnvPlatform.FREEBSD and option in {"L", "U"}:
        # FreeBSD constructs the login environment after clear/unset actions,
        # irrespective of the options' textual order.
        login_environment = True
        current = _PythonInspectEnvironment.UNKNOWN
    return current, unset_requested, login_environment


def _short_env_parse_states(
    state: _EnvParseState,
    cluster: str,
    platform: _EnvPlatform,
) -> tuple[list[_EnvParseState], bool, bool]:
    """Parse one short-option branch, retaining ambiguity and rejection."""
    pending = state.pending
    index = state.index
    clear_environment = state.clear_environment
    python_inspect = state.python_inspect
    python_inspect_unset = state.python_inspect_unset
    login_environment = state.login_environment
    short_flags = {
        _EnvPlatform.GNU: _ENV_GNU_SHORT_FLAGS,
        _EnvPlatform.FREEBSD: _ENV_FREEBSD_SHORT_FLAGS,
        _EnvPlatform.DARWIN: _ENV_DARWIN_SHORT_FLAGS,
    }[platform]
    short_operands = {
        _EnvPlatform.GNU: _ENV_GNU_SHORT_OPERANDS,
        _EnvPlatform.FREEBSD: _ENV_FREEBSD_SHORT_OPERANDS,
        _EnvPlatform.DARWIN: _ENV_DARWIN_SHORT_OPERANDS,
    }[platform]
    for position, option in enumerate(cluster):
        if option in short_flags:
            if option == "i":
                clear_environment = True
                python_inspect = (
                    _PythonInspectEnvironment.UNKNOWN
                    if platform is _EnvPlatform.FREEBSD and login_environment
                    else _PythonInspectEnvironment.ABSENT_OR_EMPTY
                )
            continue
        if option == "0":
            # Both implementations reject combining null-delimited output
            # with a utility, so this branch cannot execute source payload.
            return [], False, True
        if option == "S":
            if position + 1 < len(cluster):
                payload = _EnvArgument(cluster[position + 1 :])
                tail_index = index + 1
            elif index + 1 < len(pending):
                payload = pending[index + 1]
                tail_index = index + 2
            else:
                return [], False, True
            nested_state = _nested_env_split_state(
                _EnvParseState(
                    pending=pending,
                    index=index,
                    options_ended=state.options_ended,
                    split_count=state.split_count,
                    clear_environment=clear_environment,
                    env0_from_file=state.env0_from_file,
                    invalid_unset=state.invalid_unset,
                    python_inspect=python_inspect,
                    python_inspect_unset=python_inspect_unset,
                    login_environment=login_environment,
                ),
                payload,
                tail_index,
                platform,
            )
            if nested_state is None:
                return [], bool(payload.dynamic_offsets), not payload.dynamic_offsets
            return [nested_state], False, False
        if option not in short_operands:
            return [], False, True
        if position + 1 < len(cluster):
            operand = cluster[position + 1 :]
            if option == "u" and (not operand or "=" in operand):
                # BSD getopt accepts GNU-looking ``--unset=NAME`` as the
                # short spelling ``-u nset=NAME``.  env then rejects that
                # statically invalid variable name instead of executing the
                # following utility.
                if platform is not _EnvPlatform.GNU:
                    return [], False, True
                return (
                    [
                        _EnvParseState(
                            pending=pending,
                            index=index + 1,
                            split_count=state.split_count,
                            clear_environment=clear_environment,
                            env0_from_file=state.env0_from_file,
                            invalid_unset=True,
                            python_inspect=python_inspect,
                            python_inspect_unset=python_inspect_unset,
                            login_environment=login_environment,
                        )
                    ],
                    False,
                    False,
                )
            (
                next_python_inspect,
                next_python_inspect_unset,
                next_login_environment,
            ) = _python_inspect_after_env_operand(
                platform,
                option,
                operand,
                current=python_inspect,
                unset_requested=python_inspect_unset,
                login_environment=login_environment,
            )
            return (
                [
                    _EnvParseState(
                        pending=pending,
                        index=index + 1,
                        split_count=state.split_count,
                        clear_environment=clear_environment,
                        env0_from_file=state.env0_from_file,
                        invalid_unset=state.invalid_unset,
                        python_inspect=next_python_inspect,
                        python_inspect_unset=next_python_inspect_unset,
                        login_environment=next_login_environment,
                    )
                ],
                False,
                False,
            )
        if option == "u":
            continuations, invalid_continuations, missing = _unset_operand_continuations(
                pending, index + 1
            )
        else:
            continuations = _required_operand_continuations(pending, index + 1)
            invalid_continuations = set()
            missing = not continuations
        states: list[_EnvParseState] = []
        for continuation in continuations:
            operand = pending[continuation - 1].text
            (
                next_python_inspect,
                next_python_inspect_unset,
                next_login_environment,
            ) = _python_inspect_after_env_operand(
                platform,
                option,
                operand,
                current=python_inspect,
                unset_requested=python_inspect_unset,
                login_environment=login_environment,
            )
            states.append(
                _EnvParseState(
                    pending=pending,
                    index=continuation,
                    split_count=state.split_count,
                    clear_environment=clear_environment,
                    env0_from_file=state.env0_from_file,
                    invalid_unset=state.invalid_unset,
                    python_inspect=next_python_inspect,
                    python_inspect_unset=next_python_inspect_unset,
                    login_environment=next_login_environment,
                )
            )
        if platform is _EnvPlatform.GNU:
            states.extend(
                _EnvParseState(
                    pending=pending,
                    index=continuation,
                    split_count=state.split_count,
                    clear_environment=clear_environment,
                    env0_from_file=state.env0_from_file,
                    invalid_unset=True,
                    python_inspect=python_inspect,
                    python_inspect_unset=python_inspect_unset,
                    login_environment=login_environment,
                )
                for continuation in invalid_continuations
            )
        return (
            states,
            False,
            missing or (bool(invalid_continuations) and platform is not _EnvPlatform.GNU),
        )
    return (
        [
            _EnvParseState(
                pending=pending,
                index=index + 1,
                split_count=state.split_count,
                clear_environment=clear_environment,
                env0_from_file=state.env0_from_file,
                invalid_unset=state.invalid_unset,
                python_inspect=python_inspect,
                python_inspect_unset=python_inspect_unset,
                login_environment=login_environment,
            )
        ],
        False,
        False,
    )


def _env_split_executions(
    arguments: list[_EnvArgument],
    platform: _EnvPlatform,
    *,
    clear_environment: bool = False,
) -> tuple[set[_EnvExecution], bool, bool]:
    """Return viable executions, runtime ambiguity, and reachable nonexecution."""
    stack = [
        _EnvParseState(
            pending=tuple(arguments),
            clear_environment=clear_environment,
            python_inspect=(
                _PythonInspectEnvironment.ABSENT_OR_EMPTY
                if clear_environment
                else _PythonInspectEnvironment.UNMENTIONED
            ),
        )
    ]
    seen: set[_EnvParseState] = set()
    executions: set[_EnvExecution] = set()
    ambiguous = False
    nonexecuting = False

    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        if len(seen) > _MAX_ENV_PARSE_STATES:
            return executions, True, nonexecuting

        pending = state.pending
        index = state.index
        options_ended = state.options_ended
        split_count = state.split_count
        if index >= len(pending):
            # With no utility selected from the inspected arguments, env sees
            # the kernel-appended artifact as its utility.  That branch does
            # not establish Python execution intent and must survive beside a
            # different runtime branch that did select Python.
            nonexecuting = True
            continue
        if split_count > MAX_PYTHON_SHEBANG_CHARS:
            ambiguous = True
            continue

        argument_record = pending[index]
        argument = argument_record.text
        if argument_record.dynamic_offsets:
            if options_ended and "=" in argument:
                # Once option parsing has ended, a statically present '='
                # makes this an assignment even when its name begins with '-'.
                stack.append(replace(state, index=index + 1, options_ended=True))
                continue
            dynamic_role = _dynamic_env_argument_role(argument_record, platform)
            if dynamic_role == "assignment":
                stack.append(replace(state, index=index + 1, options_ended=True))
                continue
            if not options_ended and dynamic_role == "option":
                stack.append(replace(state, index=index + 1, options_ended=False))
                continue
            if not options_ended and dynamic_role == "branching_option":
                # A nonempty expansion is an attached operand.  If every
                # expansion is empty, the bare option consumes the next
                # runtime-present argument instead.
                stack.append(replace(state, index=index + 1, options_ended=False))
                stack.extend(
                    replace(state, index=continuation, options_ended=False)
                    for continuation in _required_operand_continuations(pending, index + 1)
                )
                continue
            if not options_ended and dynamic_role == "invalid":
                nonexecuting = True
                continue
            fixed_utility = _fixed_dynamic_utility(argument_record, options_ended=options_ended)
            if fixed_utility is not None:
                if state.invalid_unset and not (
                    platform is _EnvPlatform.GNU
                    and state.clear_environment
                    and not state.env0_from_file
                ):
                    nonexecuting = True
                else:
                    executions.add(
                        _EnvExecution(
                            fixed_utility,
                            pending[index + 1 :],
                            python_inspect=state.python_inspect,
                        )
                    )
                # A runtime '=' can instead make this token an assignment.
                # Continue that neutral/following-utility branch explicitly.
                stack.append(replace(state, index=index + 1, options_ended=True))
                continue
            ambiguous = True
            continue

        if not options_ended:
            if argument == "--":
                next_index = index + 1
                clear_after_terminator = state.clear_environment
                if (
                    platform is _EnvPlatform.GNU
                    and next_index < len(pending)
                    and pending[next_index].text == "-"
                    and not pending[next_index].dynamic_offsets
                ):
                    # GNU accepts its legacy lone ``-`` ignore-environment
                    # spelling immediately after getopt's ``--`` terminator.
                    # This check belongs here rather than in the general
                    # options-ended path: a second ``-`` or one following an
                    # assignment is the selected utility instead.
                    next_index += 1
                    clear_after_terminator = True
                stack.append(
                    replace(
                        state,
                        index=next_index,
                        options_ended=True,
                        clear_environment=clear_after_terminator,
                        python_inspect=(
                            _PythonInspectEnvironment.ABSENT_OR_EMPTY
                            if clear_after_terminator
                            else state.python_inspect
                        ),
                    )
                )
                continue
            if argument == "-":
                # GNU stops parsing after this legacy -i spelling; FreeBSD's
                # getopt treats it as an ordinary flag and keeps scanning.
                stack.append(
                    replace(
                        state,
                        index=index + 1,
                        options_ended=platform is _EnvPlatform.GNU,
                        clear_environment=True,
                        python_inspect=(
                            _PythonInspectEnvironment.UNKNOWN
                            if platform is _EnvPlatform.FREEBSD and state.login_environment
                            else _PythonInspectEnvironment.ABSENT_OR_EMPTY
                        ),
                    )
                )
                continue
            if platform is _EnvPlatform.GNU and argument.startswith("--"):
                resolved = _resolve_env_long_option(argument)
                if resolved is None:
                    nonexecuting = True
                    continue
                option, attached_operand = resolved
                if option in _ENV_SPLIT_LONG_FLAGS:
                    if attached_operand is None:
                        stack.append(
                            replace(
                                state,
                                index=index + 1,
                                options_ended=False,
                                clear_environment=(
                                    state.clear_environment or option == "--ignore-environment"
                                ),
                                python_inspect=(
                                    _PythonInspectEnvironment.ABSENT_OR_EMPTY
                                    if option == "--ignore-environment"
                                    else state.python_inspect
                                ),
                            )
                        )
                    else:
                        nonexecuting = True
                elif option == "--split-string":
                    if attached_operand is not None:
                        payload = _EnvArgument(attached_operand)
                        tail_index = index + 1
                    elif index + 1 < len(pending):
                        payload = pending[index + 1]
                        tail_index = index + 2
                    else:
                        payload = None
                    if payload is None:
                        nonexecuting = True
                    else:
                        nested_state = _nested_env_split_state(state, payload, tail_index, platform)
                        if nested_state is None:
                            if payload.dynamic_offsets:
                                ambiguous = True
                            else:
                                nonexecuting = True
                        else:
                            stack.append(nested_state)
                elif option in _ENV_SPLIT_LONG_OPERANDS:
                    env0_from_file = state.env0_from_file or option == "--env0-from"
                    if attached_operand is not None:
                        if option != "--unset" or (
                            attached_operand and "=" not in attached_operand
                        ):
                            python_inspect = state.python_inspect
                            python_inspect_unset = state.python_inspect_unset
                            login_environment = state.login_environment
                            if option == "--env0-from":
                                python_inspect = (
                                    _PythonInspectEnvironment.ABSENT_OR_EMPTY
                                    if python_inspect_unset
                                    else _PythonInspectEnvironment.UNKNOWN
                                )
                            elif option == "--unset":
                                (
                                    python_inspect,
                                    python_inspect_unset,
                                    login_environment,
                                ) = _python_inspect_after_env_operand(
                                    platform,
                                    "u",
                                    attached_operand,
                                    current=python_inspect,
                                    unset_requested=python_inspect_unset,
                                    login_environment=login_environment,
                                )
                            stack.append(
                                replace(
                                    state,
                                    index=index + 1,
                                    options_ended=False,
                                    env0_from_file=env0_from_file,
                                    python_inspect=python_inspect,
                                    python_inspect_unset=python_inspect_unset,
                                    login_environment=login_environment,
                                )
                            )
                        elif platform is _EnvPlatform.GNU:
                            stack.append(
                                replace(
                                    state,
                                    index=index + 1,
                                    options_ended=False,
                                    env0_from_file=env0_from_file,
                                    invalid_unset=True,
                                )
                            )
                        else:
                            nonexecuting = True
                    else:
                        if option == "--unset":
                            (
                                continuations,
                                invalid_continuations,
                                missing,
                            ) = _unset_operand_continuations(pending, index + 1)
                        else:
                            continuations = _required_operand_continuations(pending, index + 1)
                            invalid_continuations = set()
                            missing = not continuations
                        nonexecuting = (
                            nonexecuting
                            or missing
                            or (bool(invalid_continuations) and platform is not _EnvPlatform.GNU)
                        )
                        for continuation in continuations:
                            operand = pending[continuation - 1].text
                            python_inspect = state.python_inspect
                            python_inspect_unset = state.python_inspect_unset
                            login_environment = state.login_environment
                            if option == "--env0-from":
                                python_inspect = (
                                    _PythonInspectEnvironment.ABSENT_OR_EMPTY
                                    if python_inspect_unset
                                    else _PythonInspectEnvironment.UNKNOWN
                                )
                            elif option == "--unset":
                                (
                                    python_inspect,
                                    python_inspect_unset,
                                    login_environment,
                                ) = _python_inspect_after_env_operand(
                                    platform,
                                    "u",
                                    operand,
                                    current=python_inspect,
                                    unset_requested=python_inspect_unset,
                                    login_environment=login_environment,
                                )
                            stack.append(
                                replace(
                                    state,
                                    index=continuation,
                                    options_ended=False,
                                    env0_from_file=env0_from_file,
                                    python_inspect=python_inspect,
                                    python_inspect_unset=python_inspect_unset,
                                    login_environment=login_environment,
                                )
                            )
                        if platform is _EnvPlatform.GNU:
                            stack.extend(
                                replace(
                                    state,
                                    index=continuation,
                                    options_ended=False,
                                    env0_from_file=env0_from_file,
                                    invalid_unset=True,
                                )
                                for continuation in invalid_continuations
                            )
                else:
                    # GNU optional long-option operands are accepted only with '='.
                    stack.append(replace(state, index=index + 1, options_ended=False))

                continue
            if argument.startswith("-"):
                short_states, short_ambiguity, short_nonexecution = _short_env_parse_states(
                    state, argument[1:], platform
                )
                stack.extend(short_states)
                ambiguous = ambiguous or short_ambiguity
                nonexecuting = nonexecuting or short_nonexecution
                continue
        if "=" in argument:
            # GNU's putenv-backed implementation accepts an empty variable
            # name, while the BSD setenv-backed implementations reject it.
            # Retain both outcomes so GNU-only execution remains incomplete.
            name, _, value = argument.partition("=")
            if name or platform is _EnvPlatform.GNU:
                stack.append(
                    replace(
                        state,
                        index=index + 1,
                        options_ended=True,
                        python_inspect=(
                            _PythonInspectEnvironment.NONEMPTY
                            if name == "PYTHONINSPECT" and value
                            else _PythonInspectEnvironment.ABSENT_OR_EMPTY
                            if name == "PYTHONINSPECT"
                            else state.python_inspect
                        ),
                        python_inspect_unset=(
                            False if name == "PYTHONINSPECT" else state.python_inspect_unset
                        ),
                        login_environment=(
                            False if name == "PYTHONINSPECT" else state.login_environment
                        ),
                    )
                )
            else:
                nonexecuting = True
            continue
        if argument:
            if state.invalid_unset and not (
                platform is _EnvPlatform.GNU
                and state.clear_environment
                and not state.env0_from_file
            ):
                nonexecuting = True
            else:
                executions.add(
                    _EnvExecution(
                        argument,
                        pending[index + 1 :],
                        python_inspect=state.python_inspect,
                    )
                )
        else:
            nonexecuting = True

    return executions, ambiguous, nonexecuting


_PYTHON_NO_OPERAND_SHORT_OPTIONS = frozenset("bBdEiIOPqRsStuvx")
_PYTHON_TERMINAL_SHORT_OPTIONS = frozenset("?hV")
_PYTHON_VERSION_DEPENDENT_SHORT_OPTIONS = frozenset("IPq")
_PYTHON_HASH_BASED_PYCS_VALUES = frozenset({"always", "default", "never"})
_PYTHON_ENCODING_COOKIE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")
_PYTHON_BLANK_OR_COMMENT_LINE = re.compile(rb"^[ \t\f]*(?:[#\r\n]|$)")
_PYTHON_PHYSICAL_LINE_END = re.compile(rb"\r\n?|\n")


def _python_script_argument_executes_source(
    argument: _EnvArgument, source_path: str
) -> bool | None:
    """Resolve a selected Python script argument against the analyzed artifact."""
    if argument.dynamic_offsets:
        return None
    if not argument.text:
        return False
    # The caller's working directory is not part of bundle analysis.  Even an
    # identically spelled relative argument can therefore select a different
    # script when the inspected artifact itself was invoked by absolute path.
    if not posixpath.isabs(argument.text) or not posixpath.isabs(source_path):
        return None
    # Shebang argv uses POSIX path semantics: a backslash is a literal filename
    # character, and collapsing ``..`` can cross a symlink into a different
    # script.  Dot/repeated-separator normalization is safe only when neither
    # spelling contains a parent traversal.
    if ".." in argument.text.split("/") or ".." in source_path.split("/"):
        return None
    normalized_argument = posixpath.normpath(argument.text)
    normalized_source = posixpath.normpath(source_path)
    if normalized_argument == normalized_source:
        return True
    if (
        unicodedata.normalize("NFD", normalized_argument).casefold()
        == unicodedata.normalize("NFD", normalized_source).casefold()
    ):
        # Default macOS volumes resolve case and canonical Unicode variants to
        # the same artifact, while case-sensitive Unix volumes may not.
        return None
    argument_basename = normalized_argument.rsplit("/", 1)[-1]
    source_basename = normalized_source.rsplit("/", 1)[-1]
    if argument_basename and unicodedata.normalize("NFD", argument_basename).casefold() == (
        unicodedata.normalize("NFD", source_basename).casefold()
    ):
        # The caller's working directory, an absolute bundle root, or an alias
        # can make this the same file, but that identity is not statically known.
        return None
    # Any selected Python script receives the kernel-appended artifact in argv
    # and can load/execute it (the stdlib ``trace.py`` entrypoint is one concrete
    # example).  Without inspecting that external script, non-self paths are
    # therefore unresolved rather than proof that this source is not executed.
    return None


def _implicit_python_source_path_can_be_an_option(source_path: str) -> bool:
    """Return whether one valid relative invocation starts the path with ``-``."""
    return any(component.startswith("-") for component in source_path.split("/") if component)


def _python_command_source_spelling_may_execute(source_path: str) -> bool:
    """Return whether a bare ``-c`` source spelling is not provably inert."""
    basename = source_path.rsplit("/", 1)[-1]
    candidates = tuple(dict.fromkeys((source_path, basename)))
    for candidate in candidates:
        if len(candidate) > _MAX_PYTHON_CONSUMED_SOURCE_SPELLING_CHARS:
            return True
        try:
            parsed = ast.parse(candidate, filename="<python-c-source-path>", mode="exec")
        except (SyntaxError, ValueError):
            continue
        except (MemoryError, RecursionError):
            return True
        if not all(
            isinstance(statement, ast.Pass)
            or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, (ast.Constant, ast.Name))
            )
            for statement in parsed.body
        ):
            return True
    return False


def _python_arguments_execute_appended_script(
    arguments: tuple[_EnvArgument, ...],
    source_path: str,
    *,
    environment_inspect: _PythonInspectEnvironment = _PythonInspectEnvironment.UNMENTIONED,
) -> bool | None:
    """Classify whether Python consumes all arguments before the script path."""
    index = 0
    version_dependent = False
    forced_interactive = False
    ignore_environment = False

    def merge_version_branches(result: bool | None) -> bool | None:
        if version_dependent and result is True:
            return None
        return result

    while index < len(arguments):
        argument = arguments[index]
        if argument.dynamic_offsets:
            return None
        value = argument.text
        if value == "--":
            tail = arguments[index + 1 :]
            if not tail:
                return merge_version_branches(True)
            first = tail[0]
            if first.dynamic_offsets:
                # A retained expansion selects a runtime path; an unquoted
                # empty expansion can instead expose a later/appended path.
                return None
            return merge_version_branches(
                _python_script_argument_executes_source(first, source_path)
            )
        if value == "-":
            # Stdin is the selected program, but the kernel-appended artifact
            # remains in ``sys.argv``.  Caller-supplied stdin can therefore
            # load and execute that path just like a supplied ``-c`` command
            # or external Python script.
            return None
        if value in {
            "--help",
            "--help-all",
            "--help-env",
            "--help-xoptions",
            "--version",
        }:
            return False
        if value == "--check-hash-based-pycs":
            if index + 1 >= len(arguments):
                # The kernel-appended invocation spelling becomes this option's
                # operand.  A source whose basename is one of the accepted values
                # can be invoked from its containing directory by that basename;
                # Python then falls through to stdin, which can recover the path
                # from ``_imp.check_hash_based_pycs`` and execute it.  Preserve
                # case-insensitive filesystem aliases as the same viable branch.
                source_basename = source_path.rsplit("/", 1)[-1]
                return (
                    None if source_basename.casefold() in _PYTHON_HASH_BASED_PYCS_VALUES else False
                )
            if arguments[index + 1].dynamic_offsets:
                return None
            if arguments[index + 1].text not in _PYTHON_HASH_BASED_PYCS_VALUES:
                return False
            version_dependent = True
            index += 2
            continue
        if value.startswith("--check-hash-based-pycs="):
            return False
        if not value.startswith("-"):
            return merge_version_branches(
                _python_script_argument_executes_source(argument, source_path)
            )
        if value.startswith("--"):
            return None

        cluster = value[1:]
        position = 0
        while position < len(cluster):
            option = cluster[position]
            if option == "i":
                forced_interactive = True
            if option in {"E", "I"}:
                ignore_environment = True
            if option in _PYTHON_VERSION_DEPENDENT_SHORT_OPTIONS:
                version_dependent = True
                position += 1
                continue
            if option in _PYTHON_NO_OPERAND_SHORT_OPTIONS:
                position += 1
                continue
            if option in _PYTHON_TERMINAL_SHORT_OPTIONS:
                return False
            if option in {"c", "m"}:
                # A bare option consumes the kernel-appended source path as
                # command/module text and does not load the file.  An attached
                # or explicitly supplied command/module leaves that path in
                # argv, where arbitrary code may read and execute it.
                if (
                    forced_interactive
                    or (
                        environment_inspect
                        in {
                            _PythonInspectEnvironment.NONEMPTY,
                            _PythonInspectEnvironment.UNKNOWN,
                        }
                        and not ignore_environment
                    )
                    or position + 1 < len(cluster)
                    or index + 1 < len(arguments)
                ):
                    return None
                if option == "m" or _python_command_source_spelling_may_execute(source_path):
                    return None
                return False
            if option in {"W", "X"}:
                if position + 1 < len(cluster):
                    position = len(cluster)
                    continue
                if index + 1 >= len(arguments):
                    # The appended artifact becomes the option operand and
                    # Python falls through to stdin.  Stdin code can recover
                    # that path from ``sys.warnoptions`` or ``sys._xoptions``
                    # and execute it, so applicability remains unresolved.
                    return None
                if arguments[index + 1].dynamic_offsets:
                    return None
                index += 1
                position = len(cluster)
                continue
            return None
        index += 1
    if _implicit_python_source_path_can_be_an_option(source_path):
        return None
    return merge_version_branches(True)


def _record_source_execution(
    execution: _EnvExecution,
    execution_kinds: set[bool],
    source_path: str,
    *,
    allow_filesystem_aliases: bool = False,
) -> bool:
    """Record a viable source interpreter and return unresolved Python argv."""
    if not _is_python_interpreter(execution.utility) and not (
        allow_filesystem_aliases and _is_python_interpreter_filesystem_alias(execution.utility)
    ):
        execution_kinds.add(False)
        return False
    executes_script = _python_arguments_execute_appended_script(
        execution.arguments,
        source_path,
        environment_inspect=execution.python_inspect,
    )
    if executes_script is True:
        execution_kinds.add(True)
    elif executes_script is False:
        execution_kinds.add(False)
    return executes_script is None


def _source_classification_from_executions(
    execution_kinds: set[bool], *, ambiguous: bool
) -> PythonSourceClassification:
    """Merge successful interpreter identities and unresolved argv branches."""
    if ambiguous or len(execution_kinds) > 1:
        return PythonSourceClassification.AMBIGUOUS
    if execution_kinds == {True}:
        return PythonSourceClassification.PYTHON
    return PythonSourceClassification.NON_PYTHON


def _darwin_shebang_tokens(command_line: str) -> list[str]:
    """Tokenize the XNU shebang branch, where ``#`` ends the line."""
    xnu_line = command_line.partition("#")[0].strip(" \t")
    return re.split(r"[ \t]+", xnu_line) if xnu_line else []


def _env_split_payload(command_line: str, platform: _EnvPlatform) -> tuple[str, bool] | None:
    """Extract an outer split string and its already-applied clear-env flag."""
    short_pattern = (
        _ENV_GNU_SPLIT_SHORT_PREFIX
        if platform is _EnvPlatform.GNU
        else _ENV_FREEBSD_SPLIT_SHORT_PREFIX
    )
    short_prefix = short_pattern.match(command_line)
    if short_prefix is not None:
        return command_line[short_prefix.end() :], "i" in short_prefix.group()
    if platform is _EnvPlatform.GNU:
        resolved = _resolve_env_long_option(command_line)
        if resolved is not None:
            option, attached_operand = resolved
            if option == "--split-string" and attached_operand is not None:
                return attached_operand, False
    return None


def _classify_python_source_platforms(
    path: str, content: str | bytes | None = None
) -> PythonSourceClassification:
    """Classify Python execution intent from a path and bounded metadata.

    Normal Python source extensions are authoritative.  Other paths qualify
    only through a short shebang naming a Python interpreter directly
    or through a conventional system ``env`` launcher.  Runtime-dependent
    ``env -S`` expansion is retained as ambiguous rather than guessed.
    """
    if _has_python_source_extension(path):
        return PythonSourceClassification.PYTHON

    line = _bounded_shebang_line(content)
    if line is None:
        return (
            PythonSourceClassification.AMBIGUOUS
            if _has_overlong_shebang(content)
            else PythonSourceClassification.NON_PYTHON
        )
    if not line.startswith("#!"):
        return PythonSourceClassification.NON_PYTHON
    command_line = line[2:].lstrip(" \t")
    if not command_line.startswith("/"):
        return PythonSourceClassification.NON_PYTHON
    tokens = re.split(r"[ \t]+", command_line.strip(" \t"))
    if not tokens or not tokens[0]:
        return PythonSourceClassification.NON_PYTHON
    darwin_tokens = _darwin_shebang_tokens(command_line)
    if tokens[0] not in _TRUSTED_ENV_PATHS:
        execution_kinds: set[bool] = set()
        opaque_argument = command_line[len(tokens[0]) :].strip(" \t")
        opaque_execution = _EnvExecution(
            tokens[0],
            (_EnvArgument(opaque_argument),) if opaque_argument else (),
        )
        execution_ambiguity = _record_source_execution(opaque_execution, execution_kinds, path)
        if (
            darwin_tokens
            and darwin_tokens[0].startswith("/")
            and _is_trusted_env_filesystem_alias(darwin_tokens[0])
        ):
            darwin_executions, darwin_ambiguity, darwin_nonexecution = _env_split_executions(
                [_EnvArgument(argument) for argument in darwin_tokens[1:]],
                _EnvPlatform.DARWIN,
            )
            if darwin_nonexecution:
                execution_kinds.add(False)
            for darwin_execution in darwin_executions:
                execution_ambiguity = (
                    _record_source_execution(
                        darwin_execution,
                        execution_kinds,
                        path,
                        allow_filesystem_aliases=True,
                    )
                    or execution_ambiguity
                )
            execution_ambiguity = execution_ambiguity or darwin_ambiguity
        elif darwin_tokens and darwin_tokens[0].startswith("/"):
            darwin_execution = _EnvExecution(
                darwin_tokens[0],
                tuple(_EnvArgument(argument) for argument in darwin_tokens[1:]),
            )
            execution_ambiguity = (
                _record_source_execution(
                    darwin_execution,
                    execution_kinds,
                    path,
                    allow_filesystem_aliases=True,
                )
                or execution_ambiguity
            )
        return _source_classification_from_executions(
            execution_kinds, ambiguous=execution_ambiguity
        )

    env_command_line = command_line[len(tokens[0]) :].lstrip(" \t")
    execution_kinds = set()
    runtime_ambiguity = False
    for platform in (_EnvPlatform.GNU, _EnvPlatform.FREEBSD):
        split = _env_split_payload(env_command_line, platform)
        initial_clear_environment = False
        if split is not None:
            split_payload, initial_clear_environment = split
            env_arguments = _split_env_arguments(split_payload, platform)
            if env_arguments is None:
                execution_kinds.add(False)
                continue
            if any(argument.dynamic_offsets for argument in env_arguments):
                # A valid env -S substitution can alter token boundaries,
                # option operands, assignment roles, or the selected utility.
                # Analyze the source, but never certify every runtime branch
                # as Python without knowing the environment.
                return PythonSourceClassification.AMBIGUOUS
        elif env_command_line:
            # Linux and FreeBSD kernels pass the entire optional shebang
            # argument opaquely.  Feed that one argv item through this
            # platform's option/assignment parser even when another platform
            # recognized an outer split form: a GNU-only spelling, for
            # example, is an invalid non-executing branch on FreeBSD.
            env_arguments = [_EnvArgument(env_command_line)]
        else:
            continue
        platform_executions, platform_ambiguity, platform_nonexecution = _env_split_executions(
            env_arguments,
            platform,
            clear_environment=initial_clear_environment,
        )
        if platform_nonexecution:
            execution_kinds.add(False)
        for execution in platform_executions:
            runtime_ambiguity = (
                _record_source_execution(execution, execution_kinds, path) or runtime_ambiguity
            )
        runtime_ambiguity = runtime_ambiguity or platform_ambiguity

    # XNU tokenizes all interpreter-line arguments, with '#' ending the line,
    # instead of passing one opaque optional argument.  Preserve each selected
    # utility's argv so a preceding Python option (for example ``-I``) is not
    # confused with a preceding script name that would prevent this file from
    # being executed.
    if darwin_tokens and darwin_tokens[0] in _TRUSTED_ENV_PATHS:
        darwin_executions, darwin_ambiguity, darwin_nonexecution = _env_split_executions(
            [_EnvArgument(argument) for argument in darwin_tokens[1:]],
            _EnvPlatform.DARWIN,
        )
        if darwin_nonexecution:
            execution_kinds.add(False)
        for execution in darwin_executions:
            runtime_ambiguity = (
                _record_source_execution(
                    execution,
                    execution_kinds,
                    path,
                    allow_filesystem_aliases=True,
                )
                or runtime_ambiguity
            )
        runtime_ambiguity = runtime_ambiguity or darwin_ambiguity

    return _source_classification_from_executions(execution_kinds, ambiguous=runtime_ambiguity)


def classify_python_source(
    path: str, content: str | bytes | None = None
) -> PythonSourceClassification:
    """Classify Python source across supported interpreter-line semantics."""
    full_line = _classify_python_source_platforms(path, content)
    if _has_python_source_extension(path):
        return full_line
    platform_views = {full_line}
    for buffer_bytes in _LINUX_SHEBANG_BUFFER_SIZES:
        was_truncated, linux_content = _linux_truncated_shebang(content, buffer_bytes)
        if not was_truncated:
            continue
        if linux_content is None:
            return PythonSourceClassification.AMBIGUOUS
        platform_views.add(_classify_python_source_platforms(path, linux_content))
    return full_line if len(platform_views) == 1 else PythonSourceClassification.AMBIGUOUS


def is_python_source(path: str, content: str | bytes | None = None) -> bool:
    """Return whether bounded metadata definitively identifies Python source."""
    return classify_python_source(path, content) is PythonSourceClassification.PYTHON


def may_be_python_source(path: str, content: str | bytes | None = None) -> bool:
    """Return whether Python analysis is required, including ambiguous execution metadata."""
    return classify_python_source(path, content) is not PythonSourceClassification.NON_PYTHON


def resolve_python_source_classification(
    path: str,
    content: str | bytes | None = None,
    *,
    source_classifications: Mapping[str, PythonSourceClassification | str] | None = None,
    raw_file_cache: Mapping[str, bytes] | None = None,
) -> PythonSourceClassification:
    """Resolve one source identity, preferring cached byte-derived classification.

    Build-context classification is the canonical decision shared by analyzer
    branches.  Falling back keeps direct analyzer/unit invocations compatible,
    while raw bytes preserve kernel shebang limits that a lossy text projection
    can shift.
    """
    if source_classifications is not None:
        cached = source_classifications.get(path)
        if cached is not None:
            try:
                return PythonSourceClassification(cached)
            except (TypeError, ValueError):
                return PythonSourceClassification.AMBIGUOUS
    raw_content = raw_file_cache.get(path) if raw_file_cache is not None else None
    return classify_python_source(path, raw_content if raw_content is not None else content)


def _normalize_python_source_encoding(encoding: bytes) -> str:
    """Apply CPython's bounded UTF-8 and Latin-1 cookie normalization."""
    original = encoding.decode("ascii")
    normalized = original[:12].lower().replace("_", "-")
    if normalized == "utf-8" or normalized.startswith("utf-8-"):
        return "utf-8"
    if normalized in {"latin-1", "iso-8859-1", "iso-latin-1"} or normalized.startswith(
        ("latin-1-", "iso-8859-1-", "iso-latin-1-")
    ):
        return "iso-8859-1"
    return original


def _normalize_python_source_newlines(content: bytes) -> bytes:
    """Apply CPython's raw universal-newline pass before source decoding."""
    if b"\r" not in content:
        return content
    return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _first_two_python_source_lines(content: bytes) -> tuple[bytes, bytes]:
    """Return at most two universal-newline physical lines."""
    first_match = _PYTHON_PHYSICAL_LINE_END.search(content)
    if first_match is None:
        return content, b""
    first_end = first_match.end()
    second_match = _PYTHON_PHYSICAL_LINE_END.search(content, first_end)
    if second_match is None:
        return content[:first_end], content[first_end:]
    return content[:first_end], content[first_end : second_match.end()]


def _python_source_cookie(line: bytes, *, bom_found: bool) -> str | None:
    """Resolve one bytes-level PEP 263 cookie using CPython 3.14 semantics."""
    match = _PYTHON_ENCODING_COOKIE.match(line)
    if match is None:
        return None
    encoding = _normalize_python_source_encoding(match.group(1))
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise SyntaxError(f"unknown encoding: {encoding}") from exc
    if bom_found:
        if encoding != "utf-8":
            raise SyntaxError("encoding problem: utf-8")
        return "utf-8-sig"
    return encoding


def _validate_python_source_prefix(content: bytes, encoding: str) -> None:
    """Reject a prefix that CPython cannot decode under the selected encoding."""
    try:
        content.decode(encoding)
    except UnicodeDecodeError as exc:
        raise SyntaxError("invalid or missing encoding declaration") from exc


def _detect_python_source_encoding(content: bytes) -> str:
    """Detect PEP 263 encoding independently of the scanner host's Python."""
    first, second = _first_two_python_source_lines(content)
    bom_found = first.startswith(codecs.BOM_UTF8)
    if bom_found:
        first = first[len(codecs.BOM_UTF8) :]
    default = "utf-8-sig" if bom_found else "utf-8"
    if not first:
        return default

    encoding = _python_source_cookie(first, bom_found=bom_found)
    if encoding is not None:
        _validate_python_source_prefix(first, encoding)
        return encoding
    if _PYTHON_BLANK_OR_COMMENT_LINE.match(first) is None:
        _validate_python_source_prefix(first, default)
        return default
    if not second:
        _validate_python_source_prefix(first, default)
        return default

    encoding = _python_source_cookie(second, bom_found=bom_found)
    if encoding is not None:
        _validate_python_source_prefix(first + second, encoding)
        return encoding
    _validate_python_source_prefix(first + second, default)
    return default


def decode_python_source(content: bytes) -> str:
    """Decode raw source with stable CPython 3.14 PEP 263 rules."""
    if b"\0" in content:
        raise SyntaxError("Python source contains a null byte")
    normalized = _normalize_python_source_newlines(content)
    encoding = _detect_python_source_encoding(normalized)
    decoded = normalized.decode(encoding)
    if "\0" in decoded:
        raise SyntaxError("Python source decoder produced a null character")
    # ``str`` AST consumers and report byte accounting require scalar Unicode;
    # CPython likewise rejects source decoders that produce lone surrogates.
    decoded.encode("utf-8")
    return decoded


@dataclass(frozen=True, slots=True)
class ParsedPythonFile:
    """One Python source file's shared parse result and import aliases.

    ``tree`` is ``None`` when parsing failed.  The failed result is cached just
    like a successful one so every consumer can apply its own fallback policy
    without reparsing the same malformed source.
    """

    tree: ast.Module | None
    import_aliases: dict[str, str]
    lines: list[str]
    content: str
    parse_error: str | None = None

    @property
    def is_parseable(self) -> bool:
        """Return whether this result contains a usable Python AST."""
        return self.tree is not None


PythonAstCache = dict[str, ParsedPythonFile]


@dataclass(slots=True)
class _RuntimePythonAstCache:
    """Per-scan LRU of parsed files with an aggregate source-size budget."""

    entries: OrderedDict[str, ParsedPythonFile]
    source_characters: int = 0


# AST nodes are intentionally kept outside LangGraph state: ``ast.Module`` is
# not checkpoint-serializable.  State carries a UUID cache key, while this
# process-local registry keeps one scan's parsed trees available to all of its
# parallel analyzer branches.  Completed scans release their entry in report.
_MAX_RUNTIME_AST_CACHES = 32
_runtime_ast_caches: OrderedDict[str, _RuntimePythonAstCache] = OrderedDict()
_runtime_ast_cache_lock = RLock()


def _remember_runtime_ast_cache(cache_key: str, cache: _RuntimePythonAstCache) -> None:
    """Store a cache under the lock and bound abandoned scan entries."""
    _runtime_ast_caches[cache_key] = cache
    _runtime_ast_caches.move_to_end(cache_key)
    while len(_runtime_ast_caches) > _MAX_RUNTIME_AST_CACHES:
        _runtime_ast_caches.popitem(last=False)


def _cache_runtime_entry(
    cache: _RuntimePythonAstCache, filename: str, parsed: ParsedPythonFile
) -> None:
    """Store one parsed source, evicting least-recent entries to stay bounded."""
    old = cache.entries.pop(filename, None)
    if old is not None:
        cache.source_characters -= len(old.content)

    source_characters = len(parsed.content)
    if source_characters > MAX_PYTHON_AST_CACHE_SOURCE_CHARS:
        return
    while (
        cache.entries
        and cache.source_characters + source_characters > MAX_PYTHON_AST_CACHE_SOURCE_CHARS
    ):
        _, evicted = cache.entries.popitem(last=False)
        cache.source_characters -= len(evicted.content)
    if cache.source_characters + source_characters <= MAX_PYTHON_AST_CACHE_SOURCE_CHARS:
        cache.entries[filename] = parsed
        cache.source_characters += source_characters


def build_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map locally bound names to their fully-qualified import paths.

    ``from pathlib import Path`` becomes ``{"Path": "pathlib.Path"}``, while
    ``import pathlib as pl`` becomes ``{"pl": "pathlib"}``.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{module}.{alias.name}" if module else alias.name
    return aliases


def parse_python_source(content: str, filename: str) -> ParsedPythonFile:
    """Parse *content* once and retain its aliases or a structured parse failure."""
    lines = content.splitlines()
    try:
        tree = ast.parse(content, filename=filename)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return ParsedPythonFile(
            tree=None,
            import_aliases={},
            lines=lines,
            content=content,
            parse_error=type(exc).__name__,
        )
    return ParsedPythonFile(
        tree=tree,
        import_aliases=build_import_aliases(tree),
        lines=lines,
        content=content,
    )


def build_python_ast_cache(
    components: Iterable[str],
    file_cache: Mapping[str, str],
    *,
    raw_file_cache: Mapping[str, bytes] | None = None,
    source_classifications: Mapping[str, PythonSourceClassification | str] | None = None,
    max_source_chars: int = MAX_PYTHON_AST_SOURCE_CHARS,
    max_cache_source_chars: int = MAX_PYTHON_AST_CACHE_SOURCE_CHARS,
    clock: Callable[[], float] = time.monotonic,
    started_at: float | None = None,
    deadline: float | None = None,
    runtime_limitations: list[tuple[str, float]] | None = None,
) -> PythonAstCache:
    """Preparse eligible Python files within one scan's aggregate cache budget."""
    cache: PythonAstCache = {}
    source_characters = 0
    effective_started_at = clock() if started_at is None else started_at

    def _expired(path: str) -> bool:
        if deadline is None:
            return False
        now = clock()
        if now < deadline:
            return False
        if runtime_limitations is not None and not runtime_limitations:
            runtime_limitations.append((path, max(0.0, now - effective_started_at)))
        return True

    for path in components:
        if _expired(path):
            break
        content = file_cache.get(path)
        if content is None:
            continue
        source_classification = resolve_python_source_classification(
            path,
            content,
            source_classifications=source_classifications,
            raw_file_cache=raw_file_cache,
        )
        if _expired(path):
            break
        if (
            source_classification is PythonSourceClassification.NON_PYTHON
            or len(content) > max_source_chars
            or source_characters + len(content) > max_cache_source_chars
        ):
            continue
        cache[path] = parse_python_source(content, path)
        source_characters += len(content)
        if _expired(path):
            break
    return cache


def prewarm_python_ast_cache(
    components: Iterable[str],
    file_cache: Mapping[str, str],
    *,
    raw_file_cache: Mapping[str, bytes] | None = None,
    source_classifications: Mapping[str, PythonSourceClassification | str] | None = None,
    max_source_chars: int = MAX_PYTHON_AST_SOURCE_CHARS,
    max_cache_source_chars: int = MAX_PYTHON_AST_CACHE_SOURCE_CHARS,
    clock: Callable[[], float] = time.monotonic,
    started_at: float | None = None,
    deadline: float | None = None,
    runtime_limitations: list[tuple[str, float]] | None = None,
) -> str | None:
    """Preparse one scan's eligible Python files and return its runtime cache key."""
    cache = build_python_ast_cache(
        components,
        file_cache,
        raw_file_cache=raw_file_cache,
        source_classifications=source_classifications,
        max_source_chars=max_source_chars,
        max_cache_source_chars=max_cache_source_chars,
        clock=clock,
        started_at=started_at,
        deadline=deadline,
        runtime_limitations=runtime_limitations,
    )
    if not cache:
        return None

    cache_key = uuid4().hex
    with _runtime_ast_cache_lock:
        _remember_runtime_ast_cache(
            cache_key,
            _RuntimePythonAstCache(
                entries=OrderedDict(cache.items()),
                source_characters=sum(len(parsed.content) for parsed in cache.values()),
            ),
        )
    return cache_key


def get_python_ast(cache_key: str | None, content: str, filename: str) -> ParsedPythonFile:
    """Return a scan's prewarmed result, or parse for standalone analyzer use.

    If a checkpoint resumes in a new process, the cache key has no registry
    entry.  The lock recreates and fills it once per source before parallel
    analyzer branches can observe it.
    """
    if cache_key is None:
        return parse_python_source(content, filename)

    with _runtime_ast_cache_lock:
        cache = _runtime_ast_caches.get(cache_key)
        if cache is None:
            cache = _RuntimePythonAstCache(entries=OrderedDict())
            _remember_runtime_ast_cache(cache_key, cache)
        else:
            _runtime_ast_caches.move_to_end(cache_key)
        cached = cache.entries.get(filename)
        if cached is not None and cached.content == content:
            cache.entries.move_to_end(filename)
            return cached
        parsed = parse_python_source(content, filename)
        _cache_runtime_entry(cache, filename, parsed)
        return parsed


def clear_python_ast_cache(cache_key: str | None) -> None:
    """Release one scan's process-local parsed trees after its analyzer phase."""
    if cache_key is None:
        return
    with _runtime_ast_cache_lock:
        _runtime_ast_caches.pop(cache_key, None)
