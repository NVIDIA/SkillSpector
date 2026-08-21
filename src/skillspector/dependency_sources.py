# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic dependency-source redirection analysis.

The analyzer models package-manager configuration locally. It does not contact
registries, infer ownership/reputation, or trust explanatory prose.
"""

from __future__ import annotations

import configparser
import re
import shlex
import tomllib
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import PurePosixPath

from skillspector.models import Finding

_URL_RE = re.compile(
    r"(?:https?|ssh|git\+https?|git\+ssh|sparse\+https)://[^\s'\"<>]+",
    re.IGNORECASE,
)
_VARIABLE_RE = re.compile(
    r"(?<!\\)\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_ASSIGNMENT_WORD_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>'[^']*'|\"(?:\\.|[^\"\\])*\"|[^\s]*)"
)
_FUNCTION_DECLARATION_RE = re.compile(
    r"^\s*(?:function\s+(?P<bash>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\(\s*\))?"
    r"|(?P<posix>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\))(?P<rest>.*)$"
)
_SENSITIVE_QUERY_KEY = re.compile(r"(?:auth|credential|key|pass|secret|signature|token)", re.I)
_SHELL_SUFFIXES = frozenset({".sh", ".bash", ".zsh"})
_SHELL_SHEBANG_RE = re.compile(r"^#![^\n]*(?:^|/|\s)(?:ba|z|da|k)?sh(?:\s|$)", re.I)

Assignments = dict[str, list[tuple[int, str | None]]]

_CANONICAL_DESTINATIONS: dict[str, frozenset[str]] = {
    "npm": frozenset({"https://registry.npmjs.org/"}),
    "yarn": frozenset(
        {
            "https://registry.npmjs.org/",
            "https://registry.yarnpkg.com/",
        }
    ),
    "pip": frozenset({"https://pypi.org/simple/"}),
    "poetry": frozenset({"https://pypi.org/simple/"}),
    "maven": frozenset(
        {
            "https://repo.maven.apache.org/maven2/",
            "https://repo1.maven.org/maven2/",
        }
    ),
    "cargo": frozenset(
        {
            "sparse+https://index.crates.io/",
            "https://github.com/rust-lang/crates.io-index/",
        }
    ),
}


@dataclass(frozen=True)
class SourceChange:
    """One dependency-source trust-boundary change."""

    ecosystem: str
    operation: str
    surface: str
    scope: str | None
    destination: str
    file: str
    line: int
    matched_text: str


@dataclass(frozen=True)
class _HeredocRegion:
    target: str
    body: str
    declaration_line: int
    start_line: int
    end_line: int
    expand_variables: bool
    complete: bool


@dataclass(frozen=True)
class _ShellHeredocSpec:
    """One statically bounded heredoc declaration on a shell command line."""

    delimiter: str
    strip_tabs: bool
    expand_variables: bool
    input_fd: int
    segment: int
    command_depth: int


@dataclass(frozen=True)
class _HeredocBody:
    """The completed body associated with one ordered heredoc declaration."""

    spec: _ShellHeredocSpec
    body: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _ShellWord:
    """One statically tokenized shell word with its raw assignment shape."""

    raw: str
    value: str
    assignment: tuple[str, str] | None


_HEREDOC_WORD_BOUNDARIES = frozenset(";|&<>()")


def _strip_shell_comment(value: str) -> str:
    """Remove an unquoted shell comment without interpreting the command."""
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
        elif character == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _brace_delta(value: str) -> int:
    """Count shell grouping braces while ignoring quotes and parameter expansion."""
    quote: str | None = None
    escaped = False
    parameter_depth = 0
    delta = 0
    index = 0
    while index < len(value):
        character = value[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            index += 1
            continue
        if quote is not None:
            index += 1
            continue
        if character == "#" and (index == 0 or value[index - 1].isspace()):
            break
        if character == "$" and value[index : index + 2] == "${":
            parameter_depth += 1
            index += 2
            continue
        if character == "}" and parameter_depth:
            parameter_depth -= 1
        elif character == "{":
            delta += 1
        elif character == "}":
            delta -= 1
        index += 1
    return delta


def _command_segment_body(segment: str, *, allow_case_arm: bool = False) -> str:
    """Remove bounded shell-control wrappers around one simple command."""
    candidate = segment.strip().removesuffix("}").strip()
    candidate = candidate.lstrip("{").lstrip()
    keyword = re.match(r"^(?:then|do|else)\b\s*(?P<rest>.*)$", candidate)
    if keyword:
        candidate = keyword.group("rest")
    if allow_case_arm:
        case_arm = re.match(r"^[A-Za-z0-9_.*?|/-]+\)\s*(?P<rest>.+)$", candidate)
        if case_arm:
            candidate = case_arm.group("rest")
    return candidate.strip()


def _leading_assignments(segment: str) -> tuple[list[tuple[str, str]], str]:
    """Return leading assignment words and the remaining simple command."""
    candidate = segment.strip()
    position = 0
    export = re.match(r"export\b\s*", candidate)
    if export and _ASSIGNMENT_WORD_RE.match(candidate, export.end()):
        position = export.end()

    assignments: list[tuple[str, str]] = []
    while match := _ASSIGNMENT_WORD_RE.match(candidate, position):
        assignments.append((match.group("name"), match.group("value")))
        position = match.end()
        if position >= len(candidate):
            break
        if not candidate[position].isspace():
            return [], candidate
        position += len(candidate[position:]) - len(candidate[position:].lstrip())
    remainder = candidate[position:].strip()
    if (
        export
        and assignments
        and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in remainder.split())
    ):
        remainder = ""
    return assignments, remainder


def _assignment_from_word(word: str) -> tuple[str, str] | None:
    """Return a static ``NAME=value`` operand."""
    name, separator, value = word.partition("=")
    if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        return None
    return name, value


def _raw_shell_words(value: str) -> list[str] | None:
    """Split shell words while retaining quoting and command substitutions."""
    words: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    substitution_depth = 0
    index = 0
    while index < len(value):
        character = value[index]
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
        if character in {'"', "'", "`"}:
            quote = None if quote == character else character if quote is None else quote
            current.append(character)
            index += 1
            continue
        if quote is None and value[index : index + 2] == "$(":
            current.extend(("$", "("))
            substitution_depth += 1
            index += 2
            continue
        if quote is None and substitution_depth and character == "(":
            substitution_depth += 1
        elif quote is None and substitution_depth and character == ")":
            substitution_depth -= 1
        if character.isspace() and quote is None and substitution_depth == 0:
            if current:
                words.append("".join(current))
                current = []
            index += 1
            continue
        current.append(character)
        index += 1
    if escaped or quote is not None or substitution_depth:
        return None
    if current:
        words.append("".join(current))
    return words


def _shell_words(value: str) -> list[_ShellWord] | None:
    """Tokenize one bounded segment without losing shell word boundaries."""
    raw_words = _raw_shell_words(value)
    if raw_words is None:
        return None
    words: list[_ShellWord] = []
    for raw in raw_words:
        assignment = _assignment_from_word(raw)
        try:
            normalized = shlex.split(raw, comments=False, posix=True)
        except ValueError:
            return None
        token = normalized[0] if len(normalized) == 1 else raw
        if re.search(r"(?:\\[$`]|'[^']*[$`][^']*')", raw):
            # Quote removal must not turn a literal dollar/backtick into an
            # expandable value when the destination is resolved later.
            token = raw
        words.append(_ShellWord(raw=raw, value=token, assignment=assignment))
    return words


_STATIC_REDIRECTION_TARGET = r"(?:'[^'\n]+'|\"[^\"$`\n]+\"|[-A-Za-z0-9_./:+@%=,]+)"
_STATIC_SUBSHELL_REDIRECTIONS = re.compile(
    rf"(?:\d*>&(?:\d+|-)|\d*>>?\s*{_STATIC_REDIRECTION_TARGET})"
    rf"(?:\s*(?:\d*>&(?:\d+|-)|\d*>>?\s*{_STATIC_REDIRECTION_TARGET}))*\s*"
)


def _strip_outer_subshell(value: str) -> str:
    """Strip bounded outer ``(...)`` wrappers around static command lists."""
    candidate = value.strip()
    for _ in range(8):
        # ``((...))`` is arithmetic syntax in the supported shells, not two
        # nested subshells. Requiring whitespace distinguishes ``( (...) )``.
        if not candidate.startswith("(") or candidate.startswith("(("):
            return candidate

        quote: str | None = None
        escaped = False
        depth = 0
        closing_index: int | None = None
        for index, character in enumerate(candidate):
            if escaped:
                escaped = False
                continue
            if character == "\\" and quote != "'":
                escaped = True
                continue
            if character in {'"', "'", "`"}:
                quote = None if quote == character else character if quote is None else quote
                continue
            if quote is not None:
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    return candidate
                if depth == 0:
                    closing_index = index
                    break
        if closing_index is None:
            return candidate

        tail = candidate[closing_index + 1 :].strip()
        if tail and _STATIC_SUBSHELL_REDIRECTIONS.fullmatch(tail) is None:
            return candidate

        inner = candidate[1:closing_index].strip()
        if inner.endswith(";") and not inner.endswith(r"\;"):
            without_terminator = inner[:-1].rstrip()
            if without_terminator.endswith(";"):
                return candidate
            inner = without_terminator
        if not inner:
            return candidate
        candidate = inner
    return candidate


def _prepared_shell_segment(segment: str) -> str:
    """Remove bounded control, prompt, and subshell wrappers from a segment."""
    candidate = _command_segment_body(segment, allow_case_arm=True)
    candidate = re.sub(r"^(?:[$>]\s+)", "", candidate)
    return _strip_outer_subshell(candidate)


def _persistent_environment_assignments(segment: str) -> list[tuple[str, str]]:
    """Return assignments from an assignment-only or ``export`` command."""
    words = _shell_words(_prepared_shell_segment(segment))
    if not words:
        return []

    if words[0].value == "export" and words[0].assignment is None:
        index = 1
        if index < len(words) and words[index].value == "--":
            index += 1
        elif index < len(words) and words[index].value.startswith("-"):
            return []
        assignments: list[tuple[str, str]] = []
        for word in words[index:]:
            assignment = word.assignment or _assignment_from_word(word.value)
            if assignment is not None:
                assignments.append(assignment)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", word.value) is None:
                return []
        return assignments

    if any(word.assignment is None for word in words):
        return []
    return [word.assignment for word in words if word.assignment is not None]


def _consume_assignment_words(
    words: list[_ShellWord], index: int, *, utility_operands: bool = False
) -> tuple[int, list[tuple[str, str]]]:
    assignments: list[tuple[str, str]] = []
    while index < len(words):
        assignment = words[index].assignment
        if utility_operands and assignment is None:
            assignment = _assignment_from_word(words[index].value)
        if assignment is None:
            break
        assignments.append(assignment)
        index += 1
    return index, assignments


def _consume_env_wrapper(
    words: list[_ShellWord], index: int
) -> tuple[int, list[tuple[str, str]], bool, set[str]] | None:
    """Consume a bounded subset of static ``env`` options and assignments."""
    clear_environment = False
    unset_names: set[str] = set()
    while index < len(words) and words[index].value.startswith("-"):
        option = words[index].value
        if option == "--":
            index += 1
            break
        if option in {"-i", "--ignore-environment"}:
            clear_environment = True
            index += 1
            continue
        if option in {"-u", "--unset"}:
            if index + 1 >= len(words):
                return None
            unset_names.add(words[index + 1].value)
            index += 2
            continue
        if option in {"-C", "--chdir", "--argv0"}:
            if index + 1 >= len(words):
                return None
            index += 2
            continue
        if option.startswith("-u") and len(option) > 2:
            unset_names.add(option[2:])
            index += 1
            continue
        if option.startswith("--unset="):
            unset_names.add(option.split("=", 1)[1])
            index += 1
            continue
        if (
            (option.startswith("-C") and len(option) > 2)
            or option.startswith("--chdir=")
            or option.startswith("--argv0=")
        ):
            index += 1
            continue
        return None
    index, assignments = _consume_assignment_words(words, index, utility_operands=True)
    return (index, assignments, clear_environment, unset_names) if index < len(words) else None


def _consume_sudo_wrapper(words: list[_ShellWord], index: int) -> int | None:
    """Consume static sudo execution options, rejecting informational modes."""
    no_argument = {
        "-E",
        "-H",
        "-S",
        "-b",
        "-n",
        "--background",
        "--non-interactive",
        "--preserve-env",
        "--set-home",
        "--stdin",
    }
    with_argument = {
        "-C",
        "-D",
        "-R",
        "-T",
        "-g",
        "-h",
        "-p",
        "-r",
        "-t",
        "-u",
        "--chdir",
        "--chroot",
        "--close-from",
        "--group",
        "--host",
        "--prompt",
        "--role",
        "--type",
        "--user",
    }
    while index < len(words) and words[index].value.startswith("-"):
        option = words[index].value
        if option == "--":
            return index + 1
        if option in no_argument or option.startswith("--preserve-env="):
            index += 1
            continue
        if option in with_argument:
            if index + 1 >= len(words):
                return None
            index += 2
            continue
        if re.match(r"^-[CDRTghprtu].+", option) or re.match(
            r"^--(?:chdir|chroot|close-from|group|host|prompt|role|type|user)=.+", option
        ):
            index += 1
            continue
        return None
    return index if index < len(words) else None


def _consume_command_wrapper(words: list[_ShellWord], index: int) -> int | None:
    """Consume execution-preserving options for the shell ``command`` builtin."""
    while index < len(words) and words[index].value.startswith("-"):
        option = words[index].value
        if option == "--":
            index += 1
            break
        if option == "-p":
            index += 1
            continue
        return None
    return index if index < len(words) else None


def _normalize_executable_command(
    segment: str,
) -> tuple[list[str], list[tuple[str, str]]] | None:
    """Normalize common static execution wrappers around one simple command."""
    words = _shell_words(_prepared_shell_segment(segment))
    if words is None:
        return None
    index, assignments = _consume_assignment_words(words, 0)

    for _ in range(8):
        if index >= len(words):
            return [], assignments
        wrapper = words[index].value
        if wrapper == "env":
            consumed = _consume_env_wrapper(words, index + 1)
            if consumed is None:
                return None
            index, wrapper_assignments, clear_environment, unset_names = consumed
            if clear_environment:
                assignments.clear()
            if unset_names:
                assignments = [
                    assignment for assignment in assignments if assignment[0] not in unset_names
                ]
            assignments.extend(wrapper_assignments)
        elif wrapper == "sudo":
            consumed_index = _consume_sudo_wrapper(words, index + 1)
            if consumed_index is None:
                return None
            index = consumed_index
            index, wrapper_assignments = _consume_assignment_words(
                words, index, utility_operands=True
            )
            assignments.extend(wrapper_assignments)
        elif wrapper == "command":
            consumed_index = _consume_command_wrapper(words, index + 1)
            if consumed_index is None:
                return None
            index = consumed_index
        else:
            break
    else:
        return None

    if (
        index >= len(words)
        or re.fullmatch(r"(?:npm|yarn|pip3?|python3?|poetry|mvn|cargo)", words[index].value, re.I)
        is None
    ):
        return None
    return [word.value for word in words[index:]], assignments


def _environment_source_details(name: str) -> tuple[str, str, str | None] | None:
    normalized = name.upper()
    if normalized == "NPM_CONFIG_REGISTRY":
        return "npm", "replace", None
    if normalized == "PIP_INDEX_URL":
        return "pip", "replace", None
    if normalized == "PIP_EXTRA_INDEX_URL":
        return "pip", "add", None
    cargo = re.fullmatch(r"CARGO_REGISTRIES_([A-Z0-9_]+)_INDEX", normalized)
    if cargo:
        return "cargo", "add", cargo.group(1).lower()
    return None


def _command_environment_ecosystem(command: list[str]) -> str | None:
    if command and command[0].lower() == "npm":
        return "npm"
    if command and command[0].lower() in {"pip", "pip3"}:
        return "pip"
    if (
        len(command) >= 3
        and command[0].lower() in {"python", "python3"}
        and command[1] == "-m"
        and command[2].lower() in {"pip", "pip3"}
    ):
        return "pip"
    if command and command[0].lower() == "cargo":
        return "cargo"
    return None


def _command_destination(word: str) -> str | None:
    """Keep one argv destination without inventing boundaries inside strings."""
    if any(character.isspace() for character in word) and "$(" not in word and "`" not in word:
        return None
    return word


def _command_source_specs(
    command: list[str],
) -> list[tuple[str, str, str, str | None, str]]:
    """Return dependency-source changes from one normalized argv vector."""
    if not command:
        return []
    lowered = [word.lower() for word in command]
    specs: list[tuple[str, str, str, str | None, str]] = []

    if len(command) >= 5 and lowered[:3] == ["npm", "config", "set"]:
        key = command[3]
        scope: str | None = None
        if key.lower() == "registry":
            pass
        elif re.fullmatch(r"@[\w.-]+:registry", key, re.I):
            scope = key.rsplit(":", 1)[0]
        else:
            return specs
        destination = _command_destination(command[4])
        if destination is not None:
            specs.append(("npm", "replace", "npm config set", scope, destination))
        return specs

    if (
        len(command) >= 5
        and lowered[:3] == ["yarn", "config", "set"]
        and lowered[3] in {"registry", "npmregistryserver"}
    ):
        destination = _command_destination(command[4])
        if destination is not None:
            specs.append(("yarn", "replace", "yarn config set", None, destination))
        return specs

    pip_args: list[str] | None = None
    if lowered[0] in {"pip", "pip3"}:
        pip_args = command[1:]
    elif (
        len(command) >= 3
        and lowered[0] in {"python", "python3"}
        and lowered[1] == "-m"
        and lowered[2] in {"pip", "pip3"}
    ):
        pip_args = command[3:]
    if pip_args is not None:
        lowered_args = [word.lower() for word in pip_args]
        if len(pip_args) >= 4 and lowered_args[:2] == ["config", "set"]:
            key = lowered_args[2].removeprefix("global.")
            destination = _command_destination(pip_args[3])
            if destination is not None and key in {"index-url", "extra-index-url"}:
                specs.append(
                    (
                        "pip",
                        "add" if key == "extra-index-url" else "replace",
                        "pip config set",
                        None,
                        destination,
                    )
                )
            return specs
        for index, argument in enumerate(pip_args):
            lowered_argument = argument.lower()
            option: str | None = None
            option_destination: str | None = None
            for candidate in ("--extra-index-url", "--index-url", "-i"):
                if lowered_argument == candidate and index + 1 < len(pip_args):
                    option = candidate
                    option_destination = pip_args[index + 1]
                    break
                if lowered_argument.startswith(candidate + "="):
                    option = candidate
                    option_destination = argument.split("=", 1)[1]
                    break
            if option is None or option_destination is None:
                continue
            destination = _command_destination(option_destination)
            if destination is not None:
                extra = option == "--extra-index-url"
                specs.append(
                    (
                        "pip",
                        "add" if extra else "replace",
                        "pip --extra-index-url" if extra else "pip --index-url",
                        None,
                        destination,
                    )
                )
        return specs

    if len(command) >= 5 and lowered[:3] == ["poetry", "source", "add"]:
        index = 3
        while index < len(command) and command[index].startswith("-"):
            index += 1
        if index + 1 < len(command):
            destination = _command_destination(command[index + 1])
            if destination is not None and re.fullmatch(r"[\w.-]+", command[index]):
                specs.append(
                    (
                        "poetry",
                        "add",
                        "poetry source add",
                        command[index],
                        destination,
                    )
                )
        return specs

    if (
        len(command) >= 4
        and lowered[:2] == ["poetry", "config"]
        and lowered[2].startswith("repositories.")
    ):
        scope = command[2].split(".", 1)[1]
        destination = _command_destination(command[3])
        if destination is not None and re.fullmatch(r"[\w.-]+", scope):
            specs.append(("poetry", "add", "poetry config repositories", scope, destination))
        return specs

    if lowered[0] == "mvn":
        prefix = "-Dmaven.repo.remote="
        for argument in command[1:]:
            if argument.lower().startswith(prefix.lower()):
                destination = _command_destination(argument[len(prefix) :])
                if destination is not None:
                    specs.append(("maven", "replace", "Maven CLI repository", None, destination))
        return specs

    return specs


def _normalize_heredoc_word(word: str) -> tuple[str, bool] | None:
    """Apply bounded shell quote removal to one static heredoc word."""
    if not word or word.startswith("#"):
        return None
    delimiter: list[str] = []
    quoted = False
    index = 0
    while index < len(word):
        character = word[index]
        if character == "$" and index + 1 < len(word) and word[index + 1] == "(":
            # The surrounding scanner deliberately fails open for command and
            # arithmetic substitutions instead of accepting a partial prefix.
            return None
        if character == "$" and index + 1 < len(word) and word[index + 1] == "'":
            # Support the common static subset of Bash ANSI-C quoting. Escape
            # decoding is intentionally rejected rather than approximated.
            end = word.find("'", index + 2)
            if end < 0 or "\\" in word[index + 2 : end]:
                return None
            delimiter.append(word[index + 2 : end])
            quoted = True
            index = end + 1
            continue
        if character == "$" and index + 1 < len(word) and word[index + 1] == '"':
            # Treat Bash locale quoting like ordinary quoting. A translated
            # delimiter that no longer matches the literal terminator leaves the
            # shell input incomplete, so masking the literal complete form is the
            # conservative inert-data result.
            quoted = True
            index += 1
            continue
        if character == "'":
            end = word.find("'", index + 1)
            if end < 0:
                return None
            delimiter.append(word[index + 1 : end])
            quoted = True
            index = end + 1
            continue
        if character == '"':
            quoted = True
            index += 1
            while index < len(word) and word[index] != '"':
                if word[index] == "\\":
                    if index + 1 >= len(word):
                        return None
                    escaped = word[index + 1]
                    if escaped in {"$", "`", '"', "\\"}:
                        delimiter.append(escaped)
                    else:
                        delimiter.extend(("\\", escaped))
                    index += 2
                else:
                    delimiter.append(word[index])
                    index += 1
            if index >= len(word):
                return None
            index += 1
            continue
        if character == "\\":
            if index + 1 >= len(word):
                return None
            delimiter.append(word[index + 1])
            quoted = True
            index += 2
            continue
        if character == "`" or character.isspace() or character in _HEREDOC_WORD_BOUNDARIES:
            return None
        delimiter.append(character)
        index += 1
    normalized = "".join(delimiter)
    return (normalized, quoted) if normalized else None


def _function_context(content: str, data_lines: set[int]) -> tuple[set[int], dict[str, set[str]]]:
    """Locate function definitions and variables they may assign, without executing them."""
    lines = content.splitlines()
    function_lines: set[int] = set()
    assigned_by_function: dict[str, set[str]] = {}
    index = 0
    while index < len(lines):
        line_number = index + 1
        if line_number in data_lines:
            index += 1
            continue
        declaration = _FUNCTION_DECLARATION_RE.match(_strip_shell_comment(lines[index]))
        if not declaration:
            index += 1
            continue
        name = declaration.group("bash") or declaration.group("posix") or ""
        rest = declaration.group("rest")
        opening_index = index if rest.lstrip().startswith("{") else None
        if opening_index is None:
            candidate = index + 1
            while candidate < len(lines) and not _strip_shell_comment(lines[candidate]).strip():
                candidate += 1
            if candidate >= len(lines) or not _strip_shell_comment(
                lines[candidate]
            ).lstrip().startswith("{"):
                index += 1
                continue
            opening_index = candidate

        function_lines.add(line_number)
        depth = 0
        cursor = opening_index
        assigned_names: set[str] = set()
        while cursor < len(lines):
            function_lines.add(cursor + 1)
            fragment = rest if cursor == index else lines[cursor]
            depth += _brace_delta(fragment)
            for _, segment in _shell_parts(fragment):
                assignment_words, remainder = _leading_assignments(
                    _command_segment_body(segment, allow_case_arm=True)
                )
                if not remainder:
                    assigned_names.update(name for name, _ in assignment_words)
            cursor += 1
            if depth <= 0:
                break
        assigned_by_function.setdefault(name, set()).update(assigned_names)
        index = max(index + 1, cursor)
    return function_lines, assigned_by_function


def _literal_assignments(content: str) -> Assignments:
    """Collect definite top-level assignments without evaluating shell syntax.

    Heredoc data and function bodies are inert at their physical location, so
    their assignment-shaped text is ignored. Assignments in conditional or
    iterative control flow are recorded as ambiguous so they cannot silently
    make an earlier possible destination appear canonical.
    """
    assignments: Assignments = {}
    heredoc_data_lines = _heredoc_data_lines(content)
    function_lines, assigned_by_function = _function_context(content, heredoc_data_lines)
    control_depth = 0
    for line_number, line in enumerate(content.splitlines(), 1):
        if line_number in heredoc_data_lines or line_number in function_lines:
            continue
        for separator, segment in _shell_parts(line):
            stripped = _command_segment_body(segment, allow_case_arm=bool(control_depth))
            if re.match(r"^(?:fi|done|esac)\b", stripped):
                control_depth = max(0, control_depth - 1)
                continue
            control = re.match(
                r"^(?P<keyword>if|elif|case|for|while|until|select)\b\s*(?P<rest>.*)$",
                stripped,
            )
            if control:
                keyword = control.group("keyword")
                if keyword != "elif":
                    control_depth += 1
                if keyword == "case":
                    case_arm = re.match(r"^[^)]*\)\s*(?P<rest>.+)$", control.group("rest"))
                    if not case_arm:
                        continue
                    stripped = case_arm.group("rest")
                elif keyword in {"if", "elif", "while", "until"}:
                    stripped = control.group("rest")
                else:
                    continue

            assignment_words, call_candidate = _leading_assignments(stripped)
            if assignment_words and not call_candidate:
                for name, raw_value in assignment_words:
                    value = _strip_shell_comment(raw_value).strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                        value = value[1:-1]
                    resolved_value: str | None = value
                    if (
                        control_depth
                        or separator in {"&&", "||", "|", "|&"}
                        or not value
                        or "$" in value
                        or "`" in value
                    ):
                        resolved_value = None
                    assignments.setdefault(name, []).append((line_number, resolved_value))
                continue

            call = re.match(r"^(?:command\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b", call_candidate)
            if call and call.group("name") in assigned_by_function:
                for name in assigned_by_function[call.group("name")]:
                    assignments.setdefault(name, []).append((line_number, None))
    return assignments


def _resolve_value(value: str, assignments: Assignments, use_line: int) -> tuple[str, bool]:
    """Resolve simple variables from the latest literal assignment before use."""
    resolved = _strip_shell_comment(value).strip().strip(";,)")
    single_quoted = len(resolved) >= 2 and resolved[0] == resolved[-1] == "'"
    if single_quoted:
        literal = resolved[1:-1]
        dynamic = bool("$" in literal or "`" in literal)
        return (
            "unresolved" if dynamic or not literal else literal,
            not dynamic and bool(literal),
        )
    if len(resolved) >= 2 and resolved[0] == resolved[-1] == '"':
        resolved = resolved[1:-1]

    def replacement(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain") or ""
        prior = [assigned for line, assigned in assignments.get(name, []) if line < use_line]
        return prior[-1] if prior and prior[-1] is not None else match.group(0)

    resolved = _VARIABLE_RE.sub(replacement, resolved).strip().strip("\"'")
    dynamic = bool("$" in resolved or "`" in resolved)
    return ("unresolved" if dynamic or not resolved else resolved, not dynamic and bool(resolved))


def _normalize_destination(destination: str) -> str:
    """Normalize a URL for comparison with built-in canonical endpoints."""
    if destination == "unresolved":
        return destination
    try:
        parsed = urllib.parse.urlsplit(destination)
    except ValueError:
        return destination.rstrip("/") + "/"
    if not parsed.scheme or not parsed.hostname:
        return destination.rstrip("/") + "/"
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return destination.rstrip("/") + "/"
    if port and not (
        (scheme in {"https", "sparse+https", "git+https"} and port == 443)
        or (scheme == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"
    path = re.sub(r"/+", "/", parsed.path or "/")
    if not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunsplit((scheme, hostname, path, "", ""))


def redact_url(destination: str) -> str:
    """Remove URL credentials and sensitive query values from report evidence."""
    if destination == "unresolved":
        return destination
    try:
        parsed = urllib.parse.urlsplit(destination)
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or not parsed.hostname:
        if "@" in destination or _SENSITIVE_QUERY_KEY.search(destination.partition("?")[2]):
            return "<redacted-url>"
        return destination
    hostname = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if port:
        hostname = f"{hostname}:{port}"
    if parsed.username is not None or parsed.password is not None:
        hostname = f"***@{hostname}"
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = [
        (key, "***" if _SENSITIVE_QUERY_KEY.search(key) else value) for key, value in query
    ]
    return urllib.parse.urlunsplit(
        (parsed.scheme, hostname, parsed.path, urllib.parse.urlencode(safe_query), "")
    )


def redact_text(text: str) -> str:
    """Redact every URL-like token in source evidence."""

    def replacement(match: re.Match[str]) -> str:
        raw = match.group(0)
        suffix = ""
        while raw and raw[-1] in ".,;)]}":
            suffix = raw[-1] + suffix
            raw = raw[:-1]
        return redact_url(raw) + suffix

    return _URL_RE.sub(replacement, text)


def _is_canonical(ecosystem: str, destination: str) -> bool:
    normalized = _normalize_destination(destination)
    return normalized in _CANONICAL_DESTINATIONS[ecosystem]


def _line_for(content: str, needle: str, default: int = 1) -> int:
    for index, line in enumerate(content.splitlines(), 1):
        if needle and needle in line:
            return index
    return default


def _add_change(
    changes: list[SourceChange],
    *,
    ecosystem: str,
    operation: str,
    surface: str,
    scope: str | None,
    raw_destination: str,
    file: str,
    line: int,
    matched_text: str,
    assignments: Assignments,
) -> None:
    destination, resolved = _resolve_value(raw_destination, assignments, line)
    if resolved and _is_canonical(ecosystem, destination):
        return
    changes.append(
        SourceChange(
            ecosystem=ecosystem,
            operation=operation,
            surface=surface,
            scope=scope,
            destination=destination,
            file=file,
            line=line,
            matched_text=matched_text,
        )
    )


def _add_environment_assignment_changes(
    changes: list[SourceChange],
    assignment_words: list[tuple[str, str]],
    *,
    file: str,
    line: int,
    matched_text: str,
    assignments: Assignments,
    required_ecosystem: str | None = None,
) -> None:
    """Add one finding for every supported dependency-source assignment."""
    effective_reversed: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for name, raw_destination in reversed(assignment_words):
        if name in seen_names:
            continue
        seen_names.add(name)
        effective_reversed.append((name, raw_destination))

    for name, raw_destination in reversed(effective_reversed):
        details = _environment_source_details(name)
        if details is None:
            continue
        ecosystem, operation, scope = details
        if required_ecosystem is not None and ecosystem != required_ecosystem:
            continue
        _add_change(
            changes,
            ecosystem=ecosystem,
            operation=operation,
            surface="environment variable",
            scope=scope,
            raw_destination=raw_destination,
            file=file,
            line=line,
            matched_text=matched_text,
            assignments=assignments,
        )


def _parse_npmrc(
    content: str, file: str, start_line: int, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        match = re.match(r"(?P<key>(?:@[\w.-]+:)?registry)\s*=\s*(?P<value>.+)$", stripped, re.I)
        if not match:
            continue
        scope = match.group("key").split(":", 1)[0] if match.group("key").startswith("@") else None
        _add_change(
            changes,
            ecosystem="npm",
            operation="replace",
            surface=".npmrc",
            scope=scope,
            raw_destination=match.group("value"),
            file=file,
            line=start_line + offset,
            matched_text=line,
            assignments=assignments,
        )
    return changes


def _parse_yarnrc(
    content: str, file: str, start_line: int, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    current_scope: str | None = None
    scope_indent = -1
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        indent = len(line) - len(line.lstrip())
        scope_match = re.match(r"(?P<scope>[\w.-]+):\s*$", stripped)
        if scope_match and "npmScopes" not in stripped and indent > 0:
            current_scope = scope_match.group("scope")
            scope_indent = indent
            continue
        if current_scope and indent <= scope_indent:
            current_scope = None
        match = re.match(
            r"(?P<key>registry|npmRegistryServer)\s*(?::|\s)\s*(?P<value>.+)$",
            stripped,
            re.I,
        )
        if not match:
            continue
        _add_change(
            changes,
            ecosystem="yarn",
            operation="replace",
            surface=".yarnrc.yml" if file.lower().endswith((".yml", ".yaml")) else ".yarnrc",
            scope=current_scope,
            raw_destination=match.group("value"),
            file=file,
            line=start_line + offset,
            matched_text=line,
            assignments=assignments,
        )
    return changes


def _parse_pip_config(
    content: str, file: str, start_line: int, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    section: str | None = None
    for offset, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        match = re.match(r"(?P<key>index-url|extra-index-url)\s*=\s*(?P<value>.+)$", stripped, re.I)
        if not match:
            continue
        key = match.group("key").lower()
        _add_change(
            changes,
            ecosystem="pip",
            operation="add" if key == "extra-index-url" else "replace",
            surface="pip config",
            scope=section,
            raw_destination=match.group("value"),
            file=file,
            line=start_line + offset,
            matched_text=line,
            assignments=assignments,
        )
    return changes


def _parse_poetry(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return changes
    poetry = parsed.get("tool", {}).get("poetry", {})
    if not isinstance(poetry, dict):
        return changes
    sources = poetry.get("source", [])
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        return changes
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            continue
        destination = str(source["url"])
        _add_change(
            changes,
            ecosystem="poetry",
            operation="add",
            surface="pyproject.toml source",
            scope=str(source.get("name")) if source.get("name") is not None else None,
            raw_destination=destination,
            file=file,
            line=_line_for(content, destination),
            matched_text=next(
                (line for line in content.splitlines() if destination in line), destination
            ),
            assignments=assignments,
        )
    return changes


def _parse_maven(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return changes

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for element in root.iter():
        if local_name(element.tag) not in {"mirror", "repository", "pluginRepository"}:
            continue
        values = {local_name(child.tag): (child.text or "").strip() for child in element}
        destination = values.get("url")
        if not destination:
            continue
        is_mirror = local_name(element.tag) == "mirror"
        _add_change(
            changes,
            ecosystem="maven",
            operation="replace" if is_mirror else "add",
            surface="settings.xml mirror" if is_mirror else "Maven repository",
            scope=values.get("mirrorOf") or values.get("id"),
            raw_destination=destination,
            file=file,
            line=_line_for(content, destination),
            matched_text=next(
                (line for line in content.splitlines() if destination in line), destination
            ),
            assignments=assignments,
        )
    return changes


def _parse_cargo(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    try:
        parsed = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return changes
    sources = parsed.get("source", {})
    if isinstance(sources, dict):
        for name, source in sources.items():
            if not isinstance(source, dict):
                continue
            replacement = source.get("replace-with")
            if isinstance(replacement, str):
                target = sources.get(replacement, {})
                destination = target.get("registry") if isinstance(target, dict) else None
                raw_destination = str(destination) if destination else "unresolved"
                _add_change(
                    changes,
                    ecosystem="cargo",
                    operation="replace",
                    surface="Cargo source.replace-with",
                    scope=str(name),
                    raw_destination=raw_destination,
                    file=file,
                    line=_line_for(content, "replace-with"),
                    matched_text=next(
                        (line for line in content.splitlines() if "replace-with" in line),
                        "replace-with",
                    ),
                    assignments=assignments,
                )
            elif isinstance(source.get("registry"), str):
                destination = str(source["registry"])
                _add_change(
                    changes,
                    ecosystem="cargo",
                    operation="add" if name != "crates-io" else "replace",
                    surface="Cargo source registry",
                    scope=str(name),
                    raw_destination=destination,
                    file=file,
                    line=_line_for(content, destination),
                    matched_text=next(
                        (line for line in content.splitlines() if destination in line), destination
                    ),
                    assignments=assignments,
                )
    registries = parsed.get("registries", {})
    if isinstance(registries, dict):
        for name, registry in registries.items():
            if not isinstance(registry, dict) or not isinstance(registry.get("index"), str):
                continue
            destination = str(registry["index"])
            _add_change(
                changes,
                ecosystem="cargo",
                operation="add",
                surface="Cargo registry index",
                scope=str(name),
                raw_destination=destination,
                file=file,
                line=_line_for(content, destination),
                matched_text=next(
                    (line for line in content.splitlines() if destination in line), destination
                ),
                assignments=assignments,
            )
    return changes


def _redirection_word(line: str, start: int) -> tuple[str, int, bool]:
    """Read one redirection word without accepting a static prefix."""
    index = start
    while index < len(line) and line[index].isspace():
        index += 1
    word_start = index
    quote: str | None = None
    escaped = False
    substitution_depth = 0
    dynamic = False
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote is not None:
            if character == quote:
                quote = None
            index += 1
            continue
        if substitution_depth:
            if character in {'"', "'"}:
                quote = character
            elif line[index : index + 2] == "$(":
                substitution_depth += 1
                index += 2
                continue
            elif character == "(":
                substitution_depth += 1
            elif character == ")":
                substitution_depth -= 1
            index += 1
            continue
        if line[index : index + 2] == "$(":
            dynamic = True
            substitution_depth = 1
            index += 2
            continue
        if character == "`":
            dynamic = True
            quote = "`"
            index += 1
            continue
        if character in {'"', "'"}:
            quote = character
            index += 1
            continue
        if character.isspace() or character in _HEREDOC_WORD_BOUNDARIES:
            break
        index += 1
    malformed = escaped or quote is not None or substitution_depth > 0
    return line[word_start:index], index, dynamic or malformed


def _arithmetic_end(line: str, start: int) -> int:
    """Skip one balanced ``((...))`` or ``$((...))`` arithmetic expression."""
    opener_length = 3 if line[start : start + 3] == "$((" else 2
    depth = 2
    index = start + opener_length
    quote: str | None = None
    escaped = False
    while index < len(line) and depth:
        character = line[index]
        if escaped:
            escaped = False
        elif character == "\\" and quote != "'":
            escaped = True
        elif character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
        elif quote is None and character == "(":
            depth += 1
        elif quote is None and character == ")":
            depth -= 1
        index += 1
    return index


def _redirection_start(line: str, operator_index: int) -> int:
    """Return the start of an adjacent shell IO number, if one exists."""
    start = operator_index
    while start > 0 and line[start - 1] in "0123456789":
        start -= 1
    if start > 0 and not (line[start - 1].isspace() or line[start - 1] in ";|&()"):
        return operator_index
    return start


def _redirection_fd(line: str, operator_index: int, default: int) -> int:
    """Return an adjacent shell IO number, or the operator's default fd."""
    start = _redirection_start(line, operator_index)
    if start == operator_index:
        return default
    normalized = line[start:operator_index].lstrip("0") or "0"
    return int(normalized) if len(normalized) <= 6 else -1


def _static_redirection_target(raw: str, dynamic: bool) -> str | None:
    """Normalize one static output-redirection target without expanding it."""
    if dynamic or not raw:
        return None
    try:
        words = shlex.split(raw, comments=False, posix=True)
    except ValueError:
        return None
    return words[0] if len(words) == 1 else None


def _scan_shell_redirections(
    line: str,
) -> tuple[
    list[_ShellHeredocSpec],
    dict[tuple[int, int], str | None],
    dict[tuple[int, int], int | None],
    str,
    bool,
]:
    """Scan heredoc and stdout-file redirects in one linear lexical pass."""
    specs: list[_ShellHeredocSpec] = []
    stdout_targets: dict[tuple[int, int], str | None] = {}
    stdin_heredocs: dict[tuple[int, int], int | None] = {}
    command_characters = list(line)
    valid_heredocs = True
    quote: str | None = None
    escaped = False
    segment = 0
    command_depth = 0
    return_quote: str | None = None
    return_segment = 0
    index = 0

    def mask_command_redirection(start: int, end: int) -> None:
        if command_depth == 0 and segment == 0:
            command_characters[start:end] = [" "] * (end - start)

    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote == '"' and line[index : index + 2] == "$(":
            # A command substitution inside double quotes has its own shell
            # grammar; heredoc bodies there remain data, not executable lines.
            if command_depth == 0:
                return_quote = quote
                return_segment = segment
                segment = 0
            quote = None
            command_depth += 1
            index += 2
            continue
        if character in {'"', "'", "`"}:
            quote = None if quote == character else character if quote is None else quote
            index += 1
            continue
        if quote is not None:
            index += 1
            continue
        if character == "#" and (index == 0 or line[index - 1].isspace()):
            break
        if line[index : index + 3] == "$((" or line[index : index + 2] == "((":
            index = _arithmetic_end(line, index)
            continue
        if line[index : index + 2] == "$(":
            if command_depth == 0:
                return_quote = None
                return_segment = segment
                segment = 0
            command_depth += 1
            index += 2
            continue
        if command_depth and character == "(":
            command_depth += 1
            index += 1
            continue
        if command_depth and character == ")":
            command_depth -= 1
            index += 1
            if command_depth == 0:
                quote = return_quote
                segment = return_segment
                return_quote = None
            continue
        if character == "&" and line[index : index + 2] == "&>":
            stdout_targets[(command_depth, segment)] = None
            cursor = index + (3 if line[index : index + 3] == "&>>" else 2)
            _, end, _ = _redirection_word(line, cursor)
            mask_command_redirection(index, end)
            index = end
            continue
        if character in ";|&":
            pair = line[index : index + 2]
            segment += 1
            index += 2 if pair in {"&&", "||", "|&"} else 1
            continue
        if line[index : index + 2] == "<<" and (
            (index == 0 or line[index - 1] != "<") and line[index : index + 3] != "<<<"
        ):
            cursor = index + 2
            strip_tabs = cursor < len(line) and line[cursor] == "-"
            if strip_tabs:
                cursor += 1
            raw_word, end, dynamic = _redirection_word(line, cursor)
            partial_before_parenthesis = end < len(line) and line[end] == "("
            normalized = (
                None if dynamic or partial_before_parenthesis else _normalize_heredoc_word(raw_word)
            )
            if normalized is None:
                valid_heredocs = False
            else:
                delimiter, quoted = normalized
                input_fd = _redirection_fd(line, index, 0)
                spec_index = len(specs)
                specs.append(
                    _ShellHeredocSpec(
                        delimiter=delimiter,
                        strip_tabs=strip_tabs,
                        expand_variables=not quoted,
                        input_fd=input_fd,
                        segment=segment,
                        command_depth=command_depth,
                    )
                )
                if input_fd == 0:
                    stdin_heredocs[(command_depth, segment)] = spec_index
            mask_command_redirection(_redirection_start(line, index), end)
            index = max(end, cursor)
            continue
        if character == ">" and (index == 0 or line[index - 1] not in "<>"):
            cursor = index + (2 if line[index : index + 2] == ">>" else 1)
            fd = _redirection_fd(line, index, 1)
            supported_file_redirect = True
            if cursor < len(line) and line[cursor] in "&|":
                supported_file_redirect = False
                cursor += 1
            raw_target, end, dynamic = _redirection_word(line, cursor)
            if fd == 1:
                stdout_targets[(command_depth, segment)] = (
                    _static_redirection_target(raw_target, dynamic)
                    if supported_file_redirect
                    else None
                )
            mask_command_redirection(_redirection_start(line, index), end)
            index = max(end, cursor)
            continue
        if character == "<" and (index == 0 or line[index - 1] not in "<>"):
            if line[index : index + 2] == "<<":
                # Here-strings were excluded from the heredoc branch above.
                cursor = index + 3
            else:
                cursor = index + (2 if line[index : index + 2] in {"<&", "<>"} else 1)
            fd = _redirection_fd(line, index, 0)
            raw_target, end, _ = _redirection_word(line, cursor)
            if fd == 0:
                stdin_heredocs[(command_depth, segment)] = None
            elif fd == 1:
                stdout_targets[(command_depth, segment)] = None
            mask_command_redirection(_redirection_start(line, index), end)
            index = max(end, cursor if raw_target else cursor)
            continue
        index += 1
    if not valid_heredocs:
        specs = []
        stdin_heredocs = {}
    return specs, stdout_targets, stdin_heredocs, "".join(command_characters), valid_heredocs


def _ordered_heredoc_bodies(
    lines: list[str], header_index: int, specs: list[_ShellHeredocSpec]
) -> tuple[list[_HeredocBody], int, bool]:
    """Bind sequential heredoc bodies to their declarations in shell order."""
    bodies: list[_HeredocBody] = []
    body_index = header_index + 1
    for spec in specs:
        end = body_index
        while end < len(lines):
            terminator = lines[end].lstrip("\t") if spec.strip_tabs else lines[end]
            if terminator == spec.delimiter:
                break
            end += 1
        if end >= len(lines):
            return bodies, len(lines), False
        body_lines = lines[body_index:end]
        if spec.strip_tabs:
            body_lines = [line.lstrip("\t") for line in body_lines]
        bodies.append(
            _HeredocBody(
                spec=spec,
                body="\n".join(body_lines),
                start_line=body_index + 1,
                end_line=end + 1,
            )
        )
        body_index = end + 1
    return bodies, body_index, True


def _cat_reads_stdin(command_text: str) -> bool:
    """Return whether a bounded simple ``cat`` consumes its stdin."""
    parts = _shell_parts(command_text)
    if not parts:
        return False
    words = _shell_words(parts[0][1])
    if not words or words[0].value != "cat":
        return False

    operands: list[_ShellWord] = []
    parse_options = True
    informational = {"--help", "--version"}
    long_options = {
        "--number-nonblank",
        "--number",
        "--show-all",
        "--show-ends",
        "--show-nonprinting",
        "--show-tabs",
        "--squeeze-blank",
    }
    for word in words[1:]:
        value = word.value
        if parse_options and value == "--":
            parse_options = False
            continue
        if parse_options and value in informational:
            return False
        if parse_options and value in long_options:
            continue
        if parse_options and value.startswith("-") and value != "-":
            if re.fullmatch(r"-[AbEenstTuv]+", value) is None:
                return False
            continue
        operands.append(word)

    if not operands:
        return True
    if any(word.value == "-" for word in operands):
        return True
    # A dynamic operand can still resolve to the conventional stdin marker.
    return any("$" in word.raw or "`" in word.raw for word in operands)


def _generated_cat_heredoc(
    line: str,
    command_text: str,
    specs: list[_ShellHeredocSpec],
    stdout_targets: dict[tuple[int, int], str | None],
    stdin_heredocs: dict[tuple[int, int], int | None],
) -> tuple[str, int] | None:
    """Return the target and effective stdin heredoc for a simple ``cat``."""
    if re.match(r"^\s*cat\b", line) is None or not _cat_reads_stdin(command_text):
        return None
    target = stdout_targets.get((0, 0))
    if target is None:
        return None
    spec_index = stdin_heredocs.get((0, 0))
    if spec_index is None or spec_index >= len(specs):
        return None
    return target, spec_index


def _heredocs(content: str) -> list[_HeredocRegion]:
    """Return generated-config heredocs using ordered, linear redirection scans."""
    lines = content.splitlines()
    regions: list[_HeredocRegion] = []
    index = 0
    while index < len(lines):
        specs, stdout_targets, stdin_heredocs, command_text, valid = _scan_shell_redirections(
            lines[index]
        )
        if not valid or not specs:
            index += 1
            continue
        bodies, next_index, complete = _ordered_heredoc_bodies(lines, index, specs)
        if not complete:
            # Avoid repeated suffix scans; executable command parsing remains
            # fail-open because the unmatched body is not added to data lines.
            break
        generated = _generated_cat_heredoc(
            lines[index], command_text, specs, stdout_targets, stdin_heredocs
        )
        if generated is not None:
            target, spec_index = generated
            selected = bodies[spec_index]
            regions.append(
                _HeredocRegion(
                    target=target,
                    body=selected.body,
                    declaration_line=index + 1,
                    start_line=selected.start_line,
                    end_line=selected.end_line,
                    expand_variables=selected.spec.expand_variables,
                    complete=True,
                )
            )
        index = next_index
    return regions


def _shell_heredoc_specs(line: str) -> list[tuple[str, bool]]:
    """Return ordered static heredoc delimiters declared by one shell line."""
    specs, _, _, _, valid = _scan_shell_redirections(line)
    if not valid:
        return []
    return [(spec.delimiter, spec.strip_tabs) for spec in specs]


def _heredoc_data_lines(content: str) -> set[int]:
    """Return all complete shell heredoc body and terminator lines in one pass."""
    lines = content.splitlines()
    data_lines: set[int] = set()
    index = 0
    while index < len(lines):
        specs, _, _, _, valid = _scan_shell_redirections(lines[index])
        if not valid or not specs:
            index += 1
            continue
        bodies, next_index, complete = _ordered_heredoc_bodies(lines, index, specs)
        for body in bodies:
            data_lines.update(range(body.start_line, body.end_line + 1))
        if not complete:
            # Fail open for the unmatched body while retaining already completed
            # bodies from earlier declarations on the same command line.
            return data_lines
        index = next_index
    return data_lines


def _parse_generated_configs(
    content: str, file: str, assignments: Assignments
) -> list[SourceChange]:
    changes: list[SourceChange] = []
    heredoc_data_lines = _heredoc_data_lines(content)
    for region in _heredocs(content):
        if not region.complete or region.declaration_line in heredoc_data_lines:
            continue
        lower = region.target.lower()
        region_assignments = assignments if region.expand_variables else {}
        if lower.endswith(".npmrc"):
            changes.extend(_parse_npmrc(region.body, file, region.start_line, region_assignments))
        elif lower.endswith(".yarnrc") or lower.endswith((".yarnrc.yml", ".yarnrc.yaml")):
            changes.extend(_parse_yarnrc(region.body, file, region.start_line, region_assignments))
        elif lower.endswith(("pip.conf", "pip.ini")):
            changes.extend(
                _parse_pip_config(region.body, file, region.start_line, region_assignments)
            )
        elif lower.endswith(("settings.xml", "pom.xml")):
            generated = _parse_maven(region.body, file, region_assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=region.start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
        elif lower.endswith("pyproject.toml"):
            generated = _parse_poetry(region.body, file, region_assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=region.start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
        elif ".cargo/" in lower and lower.endswith(("/config", "/config.toml")):
            generated = _parse_cargo(region.body, file, region_assignments)
            changes.extend(
                SourceChange(
                    ecosystem=change.ecosystem,
                    operation=change.operation,
                    surface=f"generated {change.surface}",
                    scope=change.scope,
                    destination=change.destination,
                    file=change.file,
                    line=region.start_line + change.line - 1,
                    matched_text=change.matched_text,
                )
                for change in generated
            )
    return changes


def _shell_parts(line: str) -> list[tuple[str | None, str]]:
    """Split shell command lists while retaining the preceding control operator."""
    parts: list[tuple[str | None, str]] = []
    current: list[str] = []
    separator: str | None = None
    quote: str | None = None
    escaped = False
    substitution_depth = 0
    grouping_depth = 0
    index = 0
    while index < len(line):
        character = line[index]
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
        if character in {'"', "'", "`"}:
            quote = None if quote == character else character if quote is None else quote
            current.append(character)
            index += 1
            continue
        if quote is None and line[index : index + 2] == "$(":
            current.extend(("$", "("))
            substitution_depth += 1
            index += 2
            continue
        if quote is None and substitution_depth and character == "(":
            substitution_depth += 1
        elif quote is None and substitution_depth and character == ")":
            substitution_depth -= 1
        elif quote is None and character == "(":
            grouping_depth += 1
        elif quote is None and character == ")" and grouping_depth:
            grouping_depth -= 1
        if (
            quote is None
            and substitution_depth == 0
            and character == "#"
            and (not current or current[-1].isspace())
        ):
            break
        pair = line[index : index + 2]
        delimiter = pair if pair in {"&&", "||", "|&"} else character
        if (
            quote is None
            and substitution_depth == 0
            and grouping_depth == 0
            and (character in {";", "|"} or pair in {"&&", "||", "|&"})
        ):
            segment = "".join(current).strip()
            if segment:
                parts.append((separator, segment))
            current = []
            separator = delimiter
            index += 2 if pair in {"&&", "||", "|&"} else 1
            continue
        current.append(character)
        index += 1
    segment = "".join(current).strip()
    if segment:
        parts.append((separator, segment))
    return parts


def _shell_segments(line: str) -> list[str]:
    """Split executable shell command lists without evaluating shell syntax."""
    segments: list[str] = []
    for _, segment in _shell_parts(line):
        unwrapped = _strip_outer_subshell(segment)
        if unwrapped != segment:
            segments.extend(_shell_segments(unwrapped))
        else:
            segments.append(segment)
    return segments


def _parse_commands(content: str, file: str, assignments: Assignments) -> list[SourceChange]:
    changes: list[SourceChange] = []
    heredoc_data_lines = _heredoc_data_lines(content)
    for line_number, line in enumerate(content.splitlines(), 1):
        if line_number in heredoc_data_lines:
            continue
        for segment in _shell_segments(line):
            _add_environment_assignment_changes(
                changes,
                _persistent_environment_assignments(segment),
                file=file,
                line=line_number,
                matched_text=line,
                assignments=assignments,
            )

            normalized = _normalize_executable_command(segment)
            if normalized is None:
                continue
            command_candidate, command_assignments = normalized
            command_ecosystem = _command_environment_ecosystem(command_candidate)
            if command_ecosystem is not None:
                _add_environment_assignment_changes(
                    changes,
                    command_assignments,
                    file=file,
                    line=line_number,
                    matched_text=line,
                    assignments=assignments,
                    required_ecosystem=command_ecosystem,
                )
            for ecosystem, operation, surface, scope, destination in _command_source_specs(
                command_candidate
            ):
                _add_change(
                    changes,
                    ecosystem=ecosystem,
                    operation=operation,
                    surface=surface,
                    scope=scope,
                    raw_destination=destination,
                    file=file,
                    line=line_number,
                    matched_text=line,
                    assignments=assignments,
                )
    return changes


def _markdown_shell_content(content: str) -> str:
    """Keep actionable shell fences while blanking prose and preserving lines."""
    output: list[str] = []
    in_shell = False
    for line in content.splitlines():
        fence = re.match(r"^\s*```\s*([\w+-]*)", line)
        if fence:
            language = fence.group(1).lower()
            if in_shell:
                in_shell = False
            else:
                in_shell = language in {"bash", "sh", "shell", "zsh", "console"}
            output.append("")
        else:
            output.append(line if in_shell else "")
    return "\n".join(output)


def _changes_for_file(content: str, file: str, *, executable: bool = False) -> list[SourceChange]:
    normalized = file.replace("\\", "/")
    lower = normalized.lower()
    name = PurePosixPath(normalized).name.lower()
    assignments = _literal_assignments(content)
    changes: list[SourceChange] = []
    if name == ".npmrc":
        changes.extend(_parse_npmrc(content, file, 1, assignments))
    elif name == ".yarnrc":
        changes.extend(_parse_yarnrc(content, file, 1, assignments))
    elif name in {".yarnrc.yml", ".yarnrc.yaml"}:
        changes.extend(_parse_yarnrc(content, file, 1, assignments))
    elif name in {"pip.conf", "pip.ini"}:
        # ConfigParser validates basic INI structure without executing interpolation.
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(content)
        except configparser.Error:
            pass
        changes.extend(_parse_pip_config(content, file, 1, assignments))
    elif name == "pyproject.toml":
        changes.extend(_parse_poetry(content, file, assignments))
    elif name in {"settings.xml", "pom.xml"}:
        changes.extend(_parse_maven(content, file, assignments))
    elif name in {"config", "config.toml"} and "/.cargo/" in f"/{lower}":
        changes.extend(_parse_cargo(content, file, assignments))

    suffix = PurePosixPath(normalized).suffix.lower()
    is_script = suffix in _SHELL_SUFFIXES or (
        not suffix and executable and bool(_SHELL_SHEBANG_RE.search(content[:256]))
    )
    actionable = _markdown_shell_content(content) if name in {"skill.md", "readme.md"} else content
    if is_script or actionable != content:
        command_assignments = _literal_assignments(actionable) or assignments
        changes.extend(_parse_generated_configs(actionable, file, command_assignments))
        changes.extend(_parse_commands(actionable, file, command_assignments))
    return changes


def _finding(change: SourceChange, *, local_only: bool) -> Finding:
    destination = redact_url(change.destination)
    matched_text = redact_text(change.matched_text)
    resolved = change.destination != "unresolved"
    scope = change.scope or "global"
    tags = ["supply-chain", "dependency-source"]
    evidence: dict[str, object] = {
        "ecosystem": change.ecosystem,
        "operation": change.operation,
        "surface": change.surface,
        "scope": scope,
        "destination": destination,
        "destination_status": "resolved" if resolved else "unresolved",
    }
    if local_only:
        tags.append("local-only")
        evidence["local_only"] = True
    return Finding(
        rule_id="SC10",
        message=(
            f"{change.ecosystem} dependency source {change.operation} changes the "
            f"trust boundary to {destination}."
        ),
        severity="HIGH",
        confidence=1.0,
        file=change.file,
        start_line=change.line,
        category="Supply Chain",
        pattern="Dependency Source Redirection",
        finding=matched_text[:200],
        explanation=(
            "Dependency resolution is redirected away from a canonical default, adds another "
            "source, or uses a destination that cannot be resolved statically."
        ),
        remediation=(
            "Review the destination and configuration scope as a dependency trust-boundary "
            "change, and keep the intended source explicit and reviewable."
        ),
        tags=tags,
        context=matched_text,
        matched_text=matched_text[:200],
        evidence=evidence,
    )


def analyze_dependency_sources(
    components: list[str],
    file_cache: dict[str, str],
    component_metadata: list[dict[str, object]] | None = None,
) -> list[Finding]:
    """Return deterministic HIGH findings for dependency-source trust changes."""
    local_only_paths = {
        str(metadata.get("path", ""))
        for metadata in component_metadata or []
        if metadata.get("local_only") is True
    }
    executable_paths = {
        str(metadata.get("path", ""))
        for metadata in component_metadata or []
        if metadata.get("executable") is True
    }
    changes: list[SourceChange] = []
    for file in components:
        content = file_cache.get(file)
        if content is None or "\x00" in content[:8192]:
            continue
        changes.extend(_changes_for_file(content, file, executable=file in executable_paths))

    findings: list[Finding] = []
    seen: set[tuple[object, ...]] = set()
    for change in changes:
        key = (
            change.ecosystem,
            change.operation,
            change.surface,
            change.scope,
            change.destination,
            change.file,
            change.line,
        )
        if key in seen:
            continue
        seen.add(key)
        findings.append(_finding(change, local_only=change.file in local_only_paths))
    return findings
