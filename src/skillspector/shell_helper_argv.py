# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite explicit argv forwarding through a proved generated shell helper.

The caller owns generated-file identity, PATH selection, complete call-site
enumeration and mutation checks. This module only interprets a small shell
grammar over supplied text; it never executes commands or reads files. A
``printf '%q '`` value uses equivalent POSIX quoting, not Bash's original bytes.
Unsupported syntax or any unresolved execution path refuses the entire proof.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, replace

ReadWord = Callable[[str, int, Callable[[], None]], tuple[str | None, int, bool]]
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MAX_CHARS = 131_072
_MAX_STATES = 128
_MAX_ARGS = 128
_MAX_STEPS = 8192
_SHELLS = frozenset({"bash", "/bin/bash", "sh", "/bin/sh"})
_SPECIAL = frozenset(
    {
        "alias",
        "builtin",
        "declare",
        "enable",
        "eval",
        "export",
        "hash",
        "read",
        "readonly",
        "set",
        "source",
        ".",
        "trap",
        "typeset",
        "unalias",
        "unset",
    }
)
_BUILTINS = frozenset(
    {"printf", "echo", "command", "local", "shift", "exit", "return", "true", "false", ":"}
)
_SPECIAL_VARIABLES = frozenset(
    {
        "_",
        "ENV",
        "IFS",
        "PATH",
        "SHELLOPTS",
        "EUID",
        "UID",
        "PPID",
        "RANDOM",
        "SRANDOM",
        "SECONDS",
        "LINENO",
        "OPTIND",
        "OPTARG",
        "GROUPS",
        "DIRSTACK",
        "FUNCNAME",
        "PIPESTATUS",
        "CDPATH",
        "PWD",
        "OLDPWD",
        "EPOCHSECONDS",
        "EPOCHREALTIME",
        "HISTCMD",
        "EXECIGNORE",
        "LD_PRELOAD",
    }
)


def is_managed_shell_variable(name: str) -> bool:
    """Native or execution-setting names that cannot be ordinary bindings."""
    return name.startswith("BASH") or name in _SPECIAL_VARIABLES


class _UnresolvedError(Exception):
    """The optional proof cannot account for every explicit call."""


@dataclass(frozen=True, slots=True)
class _Token:
    raw: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Node:
    kind: str
    words: tuple[_Token, ...] = ()
    blocks: tuple[tuple[_Node, ...], ...] = ()
    labels: tuple[tuple[str, ...], ...] = ()
    connector: str = ";"
    redirects: tuple[tuple[str, _Token], ...] = ()


class _Parser:
    def __init__(
        self,
        text: str,
        read_word: ReadWord,
        tick: Callable[[], None],
        *,
        output_redirects: bool = False,
    ) -> None:
        self.tokens: list[_Token] = []
        self.cursor = 0
        self.tick = tick
        self.output_redirects = output_redirects
        cursor = 0
        while cursor < len(text):
            tick()
            if text[cursor] in " \t\r":
                cursor += 1
                continue
            if text.startswith("\\\n", cursor) or text.startswith("\\\r\n", cursor):
                cursor += 3 if text.startswith("\\\r\n", cursor) else 2
                continue
            if text[cursor] == "#":
                stop = text.find("\n", cursor)
                cursor = len(text) if stop < 0 else stop
                continue
            operator = next(
                (
                    item
                    for item in (
                        ";;",
                        "&&",
                        "||",
                        ">&",
                        ">>",
                        "<<",
                        ";",
                        "\n",
                        "(",
                        ")",
                        "|",
                        "&",
                        ">",
                        "<",
                    )
                    if text.startswith(item, cursor)
                ),
                None,
            )
            if operator is not None:
                stop = cursor + len(operator)
            elif text[cursor] in "{}" and (
                cursor + 1 == len(text) or text[cursor + 1] in " \t\r\n;"
            ):
                stop = cursor + 1
            else:
                _, stop, limited = read_word(text, cursor, tick)
                if limited or stop <= cursor or stop > len(text):
                    raise _UnresolvedError
            self.tokens.append(_Token(text[cursor:stop], cursor, stop))
            cursor = stop

    def peek(self, offset: int = 0) -> str:
        index = self.cursor + offset
        return self.tokens[index].raw if index < len(self.tokens) else ""

    def take(self, expected: str | None = None) -> _Token:
        self.tick()
        if self.cursor >= len(self.tokens) or expected is not None and self.peek() != expected:
            raise _UnresolvedError
        token = self.tokens[self.cursor]
        self.cursor += 1
        return token

    def sequence(self, stops: frozenset[str] = frozenset(), depth: int = 0) -> tuple[_Node, ...]:
        if depth > 16:
            raise _UnresolvedError
        nodes: list[_Node] = []
        connector = ";"
        while self.peek() and self.peek() not in stops:
            if self.peek() in {";", "\n"}:
                self.take()
                continue
            node = self.node(depth)
            nodes.append(replace(node, connector=connector))
            connector = ";"
            if self.peek() in {"&&", "||"}:
                connector = self.take().raw
                while self.peek() == "\n":
                    self.take()
                if not self.peek() or self.peek() in stops | {";", "&&", "||", ";;"}:
                    raise _UnresolvedError
            elif self.peek() and self.peek() not in stops and self.peek() not in {";", "\n"}:
                raise _UnresolvedError
        return tuple(nodes)

    def node(self, depth: int) -> _Node:
        if self.peek() == "if":
            self.take()
            blocks: list[tuple[_Node, ...]] = []
            while True:
                condition = self.sequence(frozenset({"then"}), depth + 1)
                self.take("then")
                if not condition:
                    raise _UnresolvedError
                body = self.sequence(frozenset({"elif", "else", "fi"}), depth + 1)
                if not body:
                    raise _UnresolvedError
                blocks.extend((condition, body))
                if self.peek() != "elif":
                    break
                self.take()
            if self.peek() == "else":
                self.take()
                blocks.append(self.sequence(frozenset({"fi"}), depth + 1))
                if not blocks[-1]:
                    raise _UnresolvedError
            else:
                blocks.append(())
            self.take("fi")
            return _Node("if", blocks=tuple(blocks))
        if self.peek() == "case":
            self.take()
            selector = self.take()
            self.take("in")
            labels: list[tuple[str, ...]] = []
            blocks = []
            while self.peek() != "esac":
                while self.peek() in {";", "\n"}:
                    self.take()
                if self.peek() == "esac":
                    break
                if self.peek() == "(":
                    self.take()
                patterns = [self.take().raw]
                while self.peek() == "|":
                    self.take()
                    patterns.append(self.take().raw)
                if any(
                    pattern != "*" and not re.fullmatch(r"[A-Za-z0-9_.+-]+", pattern)
                    for pattern in patterns
                ):
                    raise _UnresolvedError
                self.take(")")
                labels.append(tuple(patterns))
                blocks.append(self.sequence(frozenset({";;", "esac"}), depth + 1))
                if self.peek() == ";;":
                    self.take()
                elif self.peek() != "esac":
                    raise _UnresolvedError
            self.take("esac")
            return _Node("case", (selector,), tuple(blocks), tuple(labels))
        if _NAME.fullmatch(self.peek()) and self.peek(1) == "(" and self.peek(2) == ")":
            name = self.take()
            self.take("(")
            self.take(")")
            while self.peek() == "\n":
                self.take()
            self.take("{")
            body = self.sequence(frozenset({"}"}), depth + 1)
            if not body:
                raise _UnresolvedError
            self.take("}")
            return _Node("function", (name,), (body,))
        words: list[_Token] = []
        redirects: list[tuple[str, _Token]] = []
        if self.peek() == "[[":
            while self.peek() and self.peek() != "]]":
                words.append(self.take())
            words.append(self.take("]]"))
        else:
            while self.peek() and self.peek() not in {
                ";",
                "\n",
                "&&",
                "||",
                ";;",
                "}",
                "|",
                "&",
                "(",
                ")",
            }:
                if self.peek() in {">", ">&"} or self.output_redirects and self.peek() == ">>":
                    redirect = self.take()
                    descriptor = "1"
                    if words and words[-1].end == redirect.start and words[-1].raw.isdecimal():
                        descriptor = words.pop().raw
                        if descriptor not in {"1", "2"}:
                            raise _UnresolvedError
                    destination = self.take()
                    if self.output_redirects:
                        redirects.append((descriptor + redirect.raw, destination))
                    elif (redirect.raw, destination.raw) not in {
                        (">", "/dev/null"),
                        (">&", "1"),
                        (">&", "2"),
                    }:
                        raise _UnresolvedError
                    continue
                if self.peek() in {"<", "<<", ">>", "{"}:
                    raise _UnresolvedError
                words.append(self.take())
        if not words:
            raise _UnresolvedError
        return _Node("command", tuple(words), redirects=tuple(redirects))


@dataclass(frozen=True, slots=True)
class _SerializedArgv:
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CommandText:
    """Equivalent command syntax whose exact Bash-emitted bytes are unknown."""

    normalized: str


@dataclass(frozen=True, slots=True)
class ShellHelperOutput:
    """Literal stdout or an exact file read; no archive bytes are fabricated."""

    text: str | None = None
    path: str | None = None


_Value = str | _SerializedArgv | _CommandText | None


def _value_size(value: _Value) -> int:
    if isinstance(value, _SerializedArgv):
        return sum(map(len, value.arguments))
    if isinstance(value, _CommandText):
        return len(value.normalized)
    return len(value) if value is not None else 0


@dataclass(slots=True)
class _State:
    values: dict[str, _Value]
    scopes: list[dict[str, _Value]]
    argv: tuple[str | _CommandText | None, ...]
    status: bool | None = True
    flow: str = ""
    output: tuple[ShellHelperOutput, ...] = ()
    errexit: bool = False
    nounset: bool = False

    def copy(self) -> _State:
        return _State(
            self.values.copy(),
            [scope.copy() for scope in self.scopes],
            self.argv,
            self.status,
            self.flow,
            self.output,
            self.errexit,
            self.nounset,
        )

    def lookup(self, name: str) -> _Value:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return self.values.get(name)

    def assign(self, name: str, value: _Value) -> None:
        if is_managed_shell_variable(name) or _value_size(value) > _MAX_CHARS:
            raise _UnresolvedError
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return
        self.values[name] = value


def _word(raw: str, state: _State, tick: Callable[[], None]) -> str | _CommandText | None:
    """Expand only literal text, quoted known parameters and one suffix trim."""
    serialized = re.fullmatch(r'"exec \$\{([A-Za-z_][A-Za-z0-9_]*)% \}"', raw)
    if serialized is not None and isinstance(value := state.lookup(serialized[1]), _SerializedArgv):
        for argument in value.arguments:
            for _ in range(0, len(argument) + 1, 256):
                tick()
        normalized = "exec " + " ".join(shlex.quote(argument) for argument in value.arguments)
        if len(normalized) > _MAX_CHARS:
            raise _UnresolvedError
        return _CommandText(normalized)
    parts: list[str] = []
    quote = ""
    cursor = 0
    next_check = 0
    unknown = False
    while cursor < len(raw):
        if cursor >= next_check:
            tick()
            next_check = cursor + 256
        character = raw[cursor]
        if quote == "'":
            if character == "'":
                quote = ""
            else:
                parts.append(character)
            cursor += 1
            continue
        if character in "'\"" and (not quote or quote == character):
            quote = "" if quote else character
            cursor += 1
            continue
        if character == "\\":
            if cursor + 1 >= len(raw):
                raise _UnresolvedError
            following = raw[cursor + 1]
            if following == "\n":
                cursor += 2
                continue
            if following == "\r" and raw[cursor + 2 : cursor + 3] == "\n":
                cursor += 3
                continue
            if quote == '"' and following not in '$`"\\':
                parts.append("\\")
            parts.append(following)
            cursor += 2
            continue
        if character == "$":
            if quote != '"':
                raise _UnresolvedError  # Unquoted expansion can split or glob.
            if raw.startswith("$*", cursor):
                if any(not isinstance(argument, str) for argument in state.argv):
                    unknown = True
                else:
                    parts.append(" ".join(state.argv))
                cursor += 2
                continue
            expanded = re.match(
                r"\$\{([1-9][0-9]{0,2}|[A-Za-z_][A-Za-z0-9_]*)(:-([-A-Za-z0-9_.]+)|##\*/)\}",
                raw[cursor:],
            )
            if expanded is not None:
                name = expanded[1]
                value = (
                    state.argv[int(name) - 1]
                    if name.isdecimal() and int(name) <= len(state.argv)
                    else None
                    if name.isdecimal()
                    else state.lookup(name)
                )
                if value is None:
                    unknown = True
                elif not isinstance(value, str):
                    raise _UnresolvedError
                elif expanded[2] == "##*/":
                    parts.append(value.rpartition("/")[2])
                else:
                    parts.append(value or expanded[3])
                cursor += expanded.end()
                continue
            parameter = re.match(
                r"\$(?:([1-9]|[A-Za-z_][A-Za-z0-9_]*)|\{([1-9][0-9]{0,2}|[A-Za-z_][A-Za-z0-9_]*)(% )?\})",
                raw[cursor:],
            )
            if parameter is None:
                raise _UnresolvedError
            name = parameter[1] or parameter[2]
            if name.isdecimal():
                index = int(name) - 1
                if index >= len(state.argv) and state.nounset:
                    raise _UnresolvedError
                value = state.argv[index] if index < len(state.argv) else ""
            else:
                value = state.lookup(name)
            if value is None:
                unknown = True
            elif isinstance(value, _SerializedArgv):
                # %q chooses Bash-specific bytes. Only the exact reparse above
                # may use an equivalent representation; comparisons, copying,
                # case selectors and extra quote embedding cannot inspect it.
                raise _UnresolvedError
            elif isinstance(value, _CommandText):
                if parameter[3] is not None or cursor != 1 or raw != '"' + parameter[0] + '"':
                    raise _UnresolvedError
                return value
            else:
                for _ in range(0, len(value), 256):
                    tick()
                parts.append(value[:-1] if parameter[3] and value.endswith(" ") else value)
            cursor += parameter.end()
            continue
        if character == "`" or not quote and character in "*?[~{}":
            raise _UnresolvedError
        parts.append(character)
        cursor += 1
    if quote:
        raise _UnresolvedError
    value = "".join(parts)
    if len(value) > _MAX_CHARS or "\x00" in value:
        raise _UnresolvedError
    return None if unknown else value


class _Interpreter:
    def __init__(self, helper_name: str, tick: Callable[[], None], *, helper: bool = False) -> None:
        self.helper_name = helper_name
        self.tick = tick
        self.helper = helper
        self.functions: dict[str, tuple[_Node, ...]] = {}
        self.active: list[str] = []
        self.calls: list[tuple[str | _CommandText, ...]] = []
        self.commands: dict[tuple[int, int], list[str]] = {}
        self.allow_definitions = True

    def sequence(self, nodes: tuple[_Node, ...], states: list[_State]) -> list[_State]:
        for node in nodes:
            output: list[_State] = []
            for state in states:
                self.tick()
                if (
                    state.flow
                    or node.connector == "&&"
                    and state.status is False
                    or node.connector == "||"
                    and state.status is True
                ):
                    output.append(state)
                    continue
                if node.connector in {"&&", "||"} and state.status is None:
                    skipped = state.copy()
                    skipped.status = node.connector == "||"
                    output.append(skipped)
                output.extend(self.node(node, state))
            if len(output) > _MAX_STATES:
                raise _UnresolvedError
            states = output
        return states

    def node(self, node: _Node, state: _State) -> list[_State]:
        if node.kind == "function":
            name = node.words[0].raw
            if (
                not self.allow_definitions
                or state.scopes
                or name in self.functions
                or name == self.helper_name
                or name in _SPECIAL | _SHELLS | _BUILTINS
            ):
                raise _UnresolvedError
            self.functions[name] = node.blocks[0]
            return [state]
        self.allow_definitions = False
        if node.kind == "if":
            pending = [state]
            output: list[_State] = []
            for index in range(0, len(node.blocks) - 1, 2):
                following: list[_State] = []
                for branch in self.sequence(node.blocks[index], pending):
                    if branch.flow:
                        output.append(branch)
                    else:
                        if branch.status is not False:
                            output.extend(self.sequence(node.blocks[index + 1], [branch.copy()]))
                        if branch.status is not True:
                            following.append(branch.copy())
                pending = following
            if not node.blocks[-1]:
                for branch in pending:
                    branch.status = True  # No matching if branch succeeds.
            output.extend(self.sequence(node.blocks[-1], pending))
            return output
        if node.kind == "case":
            value = _word(node.words[0].raw, state, self.tick)
            if isinstance(value, _CommandText):
                raise _UnresolvedError
            output = []
            covered = False
            for labels, body in zip(node.labels, node.blocks, strict=True):
                if value is None or value in labels or "*" in labels:
                    branch = state.copy()
                    if not body:
                        branch.status = True
                    output.extend(self.sequence(body, [branch]))
                    if value is not None or "*" in labels:
                        covered = True
                        break
            if not covered:
                state.status = True
                output.append(state)
            return output
        return self.command(node.words, state)

    def arguments(self, words: tuple[_Token, ...], state: _State) -> tuple[str | _CommandText, ...]:
        arguments: list[str | _CommandText] = []
        for word in words:
            if word.raw == '"$@"':
                for value in state.argv:
                    for _ in range(0, _value_size(value) + 1, 256):
                        self.tick()
                arguments.extend(state.argv)
            else:
                value = _word(word.raw, state, self.tick)
                if value is None:
                    raise _UnresolvedError
                arguments.append(value)
        if len(arguments) > _MAX_ARGS or sum(map(_value_size, arguments)) > _MAX_CHARS:
            raise _UnresolvedError
        return tuple(arguments)

    def predicate(self, words: tuple[_Token, ...], state: _State) -> bool | None:
        if words[-1].raw != "]]":
            raise _UnresolvedError
        results: list[bool | None] = []
        terms: list[list[_Token]] = [[]]
        for word in words[1:-1]:
            if word.raw == "&&":
                terms.append([])
            else:
                terms[-1].append(word)
        for term in terms:
            if len(term) != 3 or term[1].raw not in {"==", "=", "!=", "-eq", "-ne"}:
                raise _UnresolvedError
            left = _word(term[0].raw, state, self.tick)
            pattern = re.fullmatch(r"\*([-A-Za-z0-9_.]+)\*", term[2].raw)
            if pattern is not None and term[1].raw in {"==", "=", "!="}:
                if isinstance(left, _CommandText):
                    raise _UnresolvedError
                results.append(
                    None if left is None else (pattern[1] in left) != (term[1].raw == "!=")
                )
                continue
            right = _word(term[2].raw, state, self.tick)
            if isinstance(left, _CommandText) or isinstance(right, _CommandText):
                raise _UnresolvedError
            if left is None or right is None:
                results.append(None)
            elif term[1].raw in {"-eq", "-ne"}:
                if not re.fullmatch(r"-?(?:0|[1-9][0-9]{0,17})", left) or not re.fullmatch(
                    r"-?(?:0|[1-9][0-9]{0,17})", right
                ):
                    raise _UnresolvedError
                results.append((int(left) == int(right)) == (term[1].raw == "-eq"))
            else:
                results.append((left == right) != (term[1].raw == "!="))
        return False if False in results else None if None in results else True

    def command(self, words: tuple[_Token, ...], state: _State) -> list[_State]:
        if words[0].raw == "[[":
            state.status = self.predicate(words, state)
            return [state]
        assignments = [
            re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", word.raw, re.DOTALL) for word in words
        ]
        if all(assignments):
            for assignment in assignments:
                assert assignment is not None
                value = _word(assignment[2], state, self.tick)
                if isinstance(value, _CommandText):
                    raise _UnresolvedError
                state.assign(assignment[1], value)
            state.status = True
            return [state]
        if any(assignments):
            raise _UnresolvedError
        if words[0].raw == "local":
            if not state.scopes or len(words) < 2:
                raise _UnresolvedError
            for word in words[1:]:
                if not _NAME.fullmatch(word.raw) or is_managed_shell_variable(word.raw):
                    raise _UnresolvedError
                state.scopes[-1][word.raw] = None
            state.status = True
            return [state]
        arguments = self.arguments(words, state)
        if not arguments:
            raise _UnresolvedError
        command, *operands = arguments
        if not isinstance(command, str):
            raise _UnresolvedError
        if command == "command":
            if not operands or not isinstance(operands[0], str) or operands[0].startswith("-"):
                raise _UnresolvedError
            command, *operands = operands
            bypass_function = True
        else:
            bypass_function = False
        if command in _SPECIAL:
            raise _UnresolvedError
        if command in self.functions and not bypass_function:
            if command in self.active or len(self.active) >= 8:
                raise _UnresolvedError
            self.active.append(command)
            previous_argv = state.argv
            state.argv = tuple(operands)
            state.scopes.append({})
            output = self.sequence(self.functions[command], [state])
            self.active.pop()
            for branch in output:
                branch.scopes.pop()
                branch.argv = previous_argv
                if branch.flow == "return":
                    branch.flow = ""
            return output
        if command == self.helper_name:
            if self.helper:
                raise _UnresolvedError
            self.calls.append(tuple(operands))
            if len(self.calls) > _MAX_STATES:
                raise _UnresolvedError
            state.status = None
        elif command in _SHELLS:
            if not self.helper or len(operands) != 2 or operands[0] != "-c" or len(words) != 3:
                raise _UnresolvedError
            span = (words[2].start, words[2].end)
            self.commands.setdefault(span, []).append(
                operands[1].normalized if isinstance(operands[1], _CommandText) else operands[1]
            )
            state.status = None
        elif any(isinstance(value, _CommandText) for value in operands):
            raise _UnresolvedError
        elif command == "printf" and operands and operands[0] == "-v":
            if len(operands) < 3 or not _NAME.fullmatch(operands[1]) or operands[2] != "%q ":
                raise _UnresolvedError
            state.assign(operands[1], _SerializedArgv(tuple(operands[3:] or [""])))
            state.status = True
        elif command in {"exit", "return"}:
            if len(operands) > 1 or operands and not re.fullmatch(r"[0-9]{1,3}", operands[0]):
                raise _UnresolvedError
            if command == "return" and not state.scopes:
                raise _UnresolvedError
            state.status = int(operands[0]) % 256 == 0 if operands else state.status
            state.flow = command
        elif command == "shift":
            if len(operands) > 1 or operands and not re.fullmatch(r"[0-9]{1,3}", operands[0]):
                raise _UnresolvedError
            count = int(operands[0]) if operands else 1
            if count > len(state.argv):
                raise _UnresolvedError
            state.argv = state.argv[count:]
            state.status = True
        elif command in {"true", "false", ":"}:
            state.status = command != "false"
        else:
            if command == "printf":
                format_args = operands[1:] if operands and operands[0] == "--" else operands
                if (
                    not format_args
                    or format_args[0].startswith("-")
                    or re.search(r"%[^%]*n", format_args[0].replace("%%", ""))
                ):
                    raise _UnresolvedError
            elif command != "echo":
                raise _UnresolvedError
            state.status = None
        return [state]


def reconstruct_helper_commands(
    helper_text: str,
    caller_texts: tuple[str, ...],
    helper_name: str,
    read_word: ReadWord,
    check_runtime: Callable[[], None],
) -> dict[tuple[int, int], tuple[tuple[str, tuple[str, ...]], ...]] | None:
    """Return separate command variants indexed by helper-relative word spans.

    Each variant contains normalized command text and the helper invocation's
    argv before ``shift``, excluding its executable. Serialized command argv
    elements use equivalent normalized quoting, not Bash's emitted bytes.
    ``None`` means the caller
    must retain incomplete inspection. An empty mapping proves no observed
    finite call reaches a shell consumer, only when the caller's independent
    call-site and generated-file proof is complete.
    """
    if (
        not caller_texts
        or len(caller_texts) > 64
        or not re.fullmatch(r"[A-Za-z0-9_.+-]+", helper_name)
        or helper_name in _SPECIAL | _SHELLS | _BUILTINS
        or sum(map(len, (helper_text, *caller_texts))) > _MAX_CHARS
        or any("\x00" in text for text in (helper_text, *caller_texts))
    ):
        return None
    steps = 0

    def tick() -> None:
        nonlocal steps
        check_runtime()
        steps += 1
        if steps > _MAX_STEPS:
            raise _UnresolvedError

    try:
        helper_parser = _Parser(helper_text, read_word, tick)
        helper = helper_parser.sequence()
        if helper_parser.peek():
            raise _UnresolvedError
        invocations: list[tuple[str | _CommandText, ...]] = []
        for caller_text in caller_texts:
            parser = _Parser(caller_text, read_word, tick)
            caller = parser.sequence()
            if parser.peek():
                raise _UnresolvedError
            interpreter = _Interpreter(helper_name, tick)
            interpreter.sequence(caller, [_State({}, [], ())])
            invocations.extend(interpreter.calls)
        if not invocations or len(invocations) > _MAX_STATES:
            raise _UnresolvedError
        output: dict[tuple[int, int], list[tuple[str, tuple[str, ...]]]] = {}
        for argv in dict.fromkeys(invocations):
            interpreter = _Interpreter(helper_name, tick, helper=True)
            interpreter.sequence(helper, [_State({}, [], argv)])
            for span, commands in interpreter.commands.items():
                for command in commands:
                    variant = (
                        command,
                        tuple(
                            value.normalized if isinstance(value, _CommandText) else value
                            for value in argv
                        ),
                    )
                    if variant not in output.setdefault(span, []):
                        output[span].append(variant)
        if sum(map(len, output.values())) > _MAX_STATES:
            raise _UnresolvedError
        return {span: tuple(variants) for span, variants in output.items()}
    except _UnresolvedError:
        return None


class _OutputInterpreter(_Interpreter):
    """Track stdout only for finite printf/cat helpers and proved log paths."""

    def __init__(
        self,
        tick: Callable[[], None],
        writable_paths: frozenset[str],
        program: tuple[_Node, ...],
        written_paths: set[str] | None = None,
    ) -> None:
        super().__init__("", tick, helper=True)
        self.writable_paths = writable_paths
        self.tested_status: set[int] = set()
        self.written_paths = written_paths

        def collect(nodes: tuple[_Node, ...], *, tested: bool = False) -> None:
            for index, node in enumerate(nodes):
                self.tick()
                if tested or index + 1 < len(nodes) and nodes[index + 1].connector in {"&&", "||"}:
                    self.tested_status.add(id(node))
                for block_index, block in enumerate(node.blocks):
                    condition = (
                        node.kind == "if"
                        and block_index < len(node.blocks) - 1
                        and block_index % 2 == 0
                    )
                    collect(block, tested=tested or condition)

        collect(program)

    def node(self, node: _Node, state: _State) -> list[_State]:
        if node.kind != "command":
            if node.kind == "function":
                raise _UnresolvedError
            return super().node(node, state)
        if node.words[0].raw == "[[":
            states = super().node(node, state)
            if state.errexit and state.status is None and id(node) not in self.tested_status:
                raise _UnresolvedError
            if state.errexit and state.status is False and id(node) not in self.tested_status:
                state.flow = "exit"
            return states
        head = _word(node.words[0].raw, state, self.tick)
        stdout = True
        for redirect, word in node.redirects:
            destination = _word(word.raw, state, self.tick)
            if redirect in {"1>", "1>>", "2>", "2>>"}:
                if destination not in self.writable_paths and destination != "/dev/null":
                    raise _UnresolvedError
                if destination != "/dev/null" and self.written_paths is not None:
                    self.written_paths.add(destination)
                if redirect.startswith("1"):
                    stdout = False
            elif redirect == "1>&" and destination == "2":
                stdout = False
            else:
                raise _UnresolvedError
        if head == "set":
            if (
                tuple(word.raw for word in node.words[1:])
                not in {("-euo", "pipefail"), ("-Eeuo", "pipefail")}
                or node.redirects
            ):
                raise _UnresolvedError
            state.status = True
            state.errexit = True
            state.nounset = True
            return [state]
        if head == "printf":
            arguments = [_word(word.raw, state, self.tick) for word in node.words[1:]]
            if not arguments or not isinstance(arguments[0], str):
                raise _UnresolvedError
            template, *values = arguments
            # The format is always parsed, even for a discarded log, so a
            # variable-setting %n or escape-driven format cannot hide effects.
            if (
                template.startswith("-")
                or re.fullmatch(r"(?:[^%\\]|%[%s]|\\[nt\\])*", template) is None
            ):
                raise _UnresolvedError
            if stdout:
                if any(not isinstance(value, str) for value in values):
                    raise _UnresolvedError
                tokens = re.split(r"(%s|%%|\\n|\\t|\\\\)", template)
                count = tokens.count("%s")
                if count == 0 and values:
                    raise _UnresolvedError
                output = []
                for index in range(0, max(1, len(values)), max(1, count)):
                    cursor = index
                    for token in tokens:
                        self.tick()
                        if token == "%s":
                            output.append(values[cursor] if cursor < len(values) else "")
                            cursor += 1
                        else:
                            output.append(
                                {"%%": "%", r"\n": "\n", r"\t": "\t", r"\\": "\\"}.get(token, token)
                            )
                self._append(state, ShellHelperOutput(text="".join(output)))
            state.status = True
            return [state]
        if head in {"cat", "/bin/cat", "/usr/bin/cat"}:
            if not stdout:
                # Discarded reads leave no artifact for the caller to validate.
                raise _UnresolvedError
            arguments = self.arguments(node.words[1:], state)
            if (
                len(arguments) != 1
                or not isinstance(arguments[0], str)
                or not arguments[0].startswith("/")
            ):
                raise _UnresolvedError
            self._append(state, ShellHelperOutput(path=arguments[0]))
            state.status = True
            return [state]
        if node.redirects or head not in {"exit", "true", ":"}:
            raise _UnresolvedError
        return super().node(node, state)

    def _append(self, state: _State, value: ShellHelperOutput) -> None:
        if (
            len(state.output) >= _MAX_ARGS
            or sum(len(item.text or item.path or "") for item in (*state.output, value))
            > _MAX_CHARS
        ):
            raise _UnresolvedError
        state.output += (value,)


def reconstruct_helper_output(
    helper_text: str,
    argv: tuple[str | None, ...],
    environment: dict[str, str | None],
    writable_paths: frozenset[str],
    read_word: ReadWord,
    check_runtime: Callable[[], None],
    *,
    written_paths: set[str] | None = None,
) -> tuple[tuple[ShellHelperOutput, ...], ...] | None:
    """Enumerate possible successful stdout values without opening a file.

    The caller must prove helper identity, ordinary shell startup, and writable
    log ownership. A file-read piece retains its identity until the caller maps
    it to a known artifact. Unknown arguments explore both conditional paths;
    unsupported effects or a possibly failing path refuse the entire proof.
    Failure output cannot disappear when a caller runs with errexit disabled.
    """
    if (
        len(helper_text) > _MAX_CHARS
        or "\x00" in helper_text
        or len(argv) > _MAX_ARGS
        or len(environment) > _MAX_ARGS
        or any(not _NAME.fullmatch(name) or is_managed_shell_variable(name) for name in environment)
        or sum(len(value or "") for value in (*argv, *environment.values(), *writable_paths))
        > _MAX_CHARS
        or any("\x00" in (value or "") for value in (*argv, *environment.values(), *writable_paths))
    ):
        return None
    steps = 0

    def tick() -> None:
        nonlocal steps
        check_runtime()
        steps += 1
        if steps > _MAX_STEPS:
            raise _UnresolvedError

    try:
        parser = _Parser(helper_text, read_word, tick, output_redirects=True)
        program = parser.sequence()
        if parser.peek():
            raise _UnresolvedError
        states = _OutputInterpreter(tick, writable_paths, program, written_paths).sequence(
            program, [_State(environment.copy(), [], argv)]
        )
        if any(state.status is not True for state in states):
            raise _UnresolvedError
        return tuple(dict.fromkeys(state.output for state in states))
    except _UnresolvedError:
        return None
