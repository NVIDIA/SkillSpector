# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, invocation-aware effects of already loaded shell programs.

Definitions do not execute their bodies. Imports and calls retain ordered
overrides, arguments and dynamic local scope. Unknown branches are both audited;
unsupported syntax refuses the optional source-identity proof. Nothing here
executes a process or reads a file from the runtime filesystem.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from skillspector.shell_helper_argv import (
    _Node,
    _Parser,
    _Token,
    _UnresolvedError,
    is_managed_shell_variable,
)
from skillspector.shell_sources import readonly_file_arguments

if TYPE_CHECKING:
    from skillspector.shell_sources import ShellSourceResolver


@dataclass(frozen=True, slots=True)
class ShellEffectCommand:
    argv: tuple[str | None, ...]
    environment: dict[str, str | None]
    cwd: str | None
    source_path: str
    offset: int
    redirects: tuple[tuple[str, str | None], ...] = ()
    stdin: str | None = None
    stdout: str | None = None
    uncertain: bool = False
    identity_only: bool = False
    child_inspected: bool = False
    child_status: bool | None = None
    redirect_open: bool = False
    fail_stop: bool = False
    interpreter: str | None = None


@dataclass(frozen=True, slots=True)
class ShellEffectResult:
    stdout: str | None = None
    status: bool | None = None
    writes: tuple[str, ...] = ()
    script_path: str | None = None
    native_identity: bool = False


NativeEffect = Callable[[ShellEffectCommand], ShellEffectResult | None]
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)\Z", re.DOTALL)
SHELL_PID_MARKER = "\ue102skillspector_shell_pid\ue103"
_QUOTED_PARAMETER_REMOVAL = re.compile(
    r'"\$\{[A-Za-z_][A-Za-z0-9_]{0,127}(?:#{1,2}|%{1,2})'
    r'"\$[A-Za-z_][A-Za-z0-9_]{0,127}"[-A-Za-z0-9_./]*\}"'
)
_MAX_STEPS = 65_536
_MAX_STATES = 128
_MAX_DEPTH = 16


def _symbolic_path_matches(left: str, right: str) -> bool:
    """Keep a decimal PID domain distinct from an exact/fresh path."""
    if SHELL_PID_MARKER not in left:
        left, right = right, left
    if SHELL_PID_MARKER not in left:
        return left == right
    if SHELL_PID_MARKER in right or left.count(SHELL_PID_MARKER) != 1:
        return True
    prefix, suffix = left.split(SHELL_PID_MARKER)
    if not right.startswith(prefix) or not right.endswith(suffix):
        return False
    end = len(right) - len(suffix)
    digits = right[len(prefix) : end] if end >= len(prefix) else ""
    return bool(digits) and digits.isascii() and digits.isdecimal()


@dataclass(frozen=True, slots=True)
class _Program:
    path: str
    text: str
    tokens: tuple[_Token, ...]
    heredocs: dict[int, tuple[str, bool]]


@dataclass(frozen=True, slots=True)
class _Function:
    program: _Program
    tokens: tuple[_Token, ...]


@dataclass(slots=True)
class _Frame:
    values: dict[str, str | tuple[str, ...] | None]
    argv: tuple[str | None, ...]
    zero: str | None
    cwd: str | None = "."
    functions: dict[str, _Function] = field(default_factory=dict)
    scopes: list[dict[str, str | tuple[str, ...] | None]] = field(default_factory=list)
    status: bool | None = True
    flow: str = ""
    output: str | None = ""
    uncertain: bool = False
    traps: dict[str, tuple[_Program, str]] = field(default_factory=dict)
    exported: set[str] = field(default_factory=set)
    errexit: bool = False
    checked: bool = False
    required_path_output: bool = False
    asynchronous: bool = False

    def copy(self) -> _Frame:
        return _Frame(
            self.values.copy(),
            self.argv,
            self.zero,
            self.cwd,
            self.functions.copy(),
            [item.copy() for item in self.scopes],
            self.status,
            self.flow,
            self.output,
            self.uncertain,
            self.traps.copy(),
            self.exported.copy(),
            self.errexit,
            self.checked,
            self.required_path_output,
            self.asynchronous,
        )

    def lookup(self, name: str):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return self.values.get(name)

    def assign(self, name: str, value, *, local: bool = False) -> None:
        if not _NAME.fullmatch(name) or is_managed_shell_variable(name):
            raise _UnresolvedError
        if local:
            if not self.scopes:
                raise _UnresolvedError
            self.scopes[-1][name] = value
            return
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return
        self.values[name] = value


class _EffectsParser(_Parser):
    """Reuse shell tokens/control nodes; defer parsing uncalled function bodies."""

    def __init__(self, program: _Program, tick, tokens=None):
        self.program = program
        self.tokens = list(program.tokens if tokens is None else tokens)
        self.cursor = 0
        self.tick = tick
        self.output_redirects = True

    def sequence(self, stops=frozenset(), depth=0):
        if depth > _MAX_DEPTH:
            raise _UnresolvedError
        nodes = []
        connector = ";"
        while self.peek() and self.peek() not in stops:
            if self.peek() in {";", "\n"}:
                self.take()
                continue
            node = self.node(depth)
            if node.kind in {"group", "subshell"}:
                redirects = []
                while self.peek():
                    descriptor = "1"
                    if self.peek().isdecimal() and self.peek(1) in {">", ">>", ">&"}:
                        if self.tokens[self.cursor].end != self.tokens[self.cursor + 1].start:
                            break
                        descriptor = self.take().raw
                    elif self.peek() not in {">", ">>", ">&"}:
                        break
                    operator = self.take().raw
                    destination = self.take()
                    redirects.append((descriptor + operator, destination))
                if redirects:
                    node = _Node("redirect", blocks=((node,),), redirects=tuple(redirects))
            if self.peek() == "|":
                blocks = [(node,)]
                while self.peek() == "|":
                    self.take()
                    while self.peek() == "\n":
                        self.take()
                    blocks.append((self.node(depth + 1),))
                node = _Node("pipeline", blocks=tuple(blocks))
            if self.peek() == "&":
                self.take()
                node = _Node("background", blocks=((node,),))
            nodes.append(replace(node, connector=connector))
            connector = ";"
            if self.peek() in {"&&", "||"}:
                connector = self.take().raw
                while self.peek() == "\n":
                    self.take()
                if not self.peek() or self.peek() in stops | {";", "&&", "||", ";;"}:
                    raise _UnresolvedError
            elif self.peek() and self.peek() not in stops | {";", "\n"}:
                raise _UnresolvedError
        return tuple(nodes)

    def node(self, depth):
        if self.peek() == "!":
            self.take()
            return _Node("negate", blocks=((self.node(depth + 1),),))
        if self.peek() == "(" and self.peek(1) == "(":
            self.take("(")
            self.take("(")
            words = []
            while self.peek() and not (self.peek() == ")" and self.peek(1) == ")"):
                words.append(self.take())
            self.take(")")
            self.take(")")
            return _Node("arithmetic", tuple(words))
        if _NAME.fullmatch(self.peek()) and self.peek(1) == "(" and self.peek(2) == ")":
            name = self.take()
            self.take("(")
            self.take(")")
            while self.peek() == "\n":
                self.take()
            self.take("{")
            body = []
            nesting = 1
            while nesting:
                token = self.take()
                nesting += (token.raw == "{") - (token.raw == "}")
                if nesting:
                    body.append(token)
            return _Node("function", (name, *body))
        if self.peek() in {"(", "{"}:
            opening = self.take().raw
            closing = ")" if opening == "(" else "}"
            body = self.sequence(frozenset({closing}), depth + 1)
            self.take(closing)
            return _Node("subshell" if opening == "(" else "group", blocks=(body,))
        if self.peek() in {"while", "until"}:
            kind = self.take().raw
            condition = self.sequence(frozenset({"do"}), depth + 1)
            self.take("do")
            body = self.sequence(frozenset({"done"}), depth + 1)
            self.take("done")
            return _Node(kind, blocks=(condition, body))
        if self.peek() == "for":
            self.take()
            words = [self.take()]
            if self.peek() == "in":
                words.append(self.take())
                while self.peek() and self.peek() not in {";", "\n"}:
                    words.append(self.take())
            while self.peek() in {";", "\n"}:
                self.take()
            self.take("do")
            body = self.sequence(frozenset({"done"}), depth + 1)
            self.take("done")
            return _Node("for", tuple(words), (body,))
        if self.peek() in {"if", "case"}:
            return super().node(depth)
        words = []
        redirects = []
        process_inputs = []
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
                ")",
            }:
                if self.peek() in {">", ">>", ">&", "<", "<<", "<<<", "<&"}:
                    operator = self.take()
                    descriptor = "0" if operator.raw.startswith("<") else "1"
                    if words and words[-1].end == operator.start and words[-1].raw.isdecimal():
                        descriptor = words.pop().raw
                    if operator.raw == "<" and self.peek() == "<" and self.peek(1) == "(":
                        marker = self.take("<")
                        opening = self.take("(")
                        if descriptor != "0" or redirects or marker.end != opening.start:
                            raise _UnresolvedError
                        process_inputs.append(self.sequence(frozenset({")"}), depth + 1))
                        self.take(")")
                        redirects.append(("0<process", marker))
                        continue
                    destination = self.take()
                    redirects.append((descriptor + operator.raw, destination))
                elif self.peek() == "(":
                    # Only simple indexed arrays have non-command parentheses.
                    if not words or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=", words[-1].raw):
                        raise _UnresolvedError
                    words.append(self.take())
                    while self.peek() and self.peek() != ")":
                        words.append(self.take())
                    words.append(self.take(")"))
                elif self.peek() == "{":
                    raise _UnresolvedError
                else:
                    words.append(self.take())
        if not words and not redirects:
            raise _UnresolvedError
        if process_inputs and len(redirects) != 1:
            raise _UnresolvedError
        return _Node(
            "command", tuple(words), blocks=tuple(process_inputs), redirects=tuple(redirects)
        )


class _EffectAudit:
    def __init__(
        self,
        resolver,
        target,
        read_source,
        native_effect,
        *,
        preserve_only=False,
        preserve_state=None,
    ):
        from skillspector.shell_sources import SAFE_PATH_MARKER

        self.resolver = resolver
        self.target = target
        self.read_source = read_source if read_source is not None else resolver.context.sources.get
        self.native_effect = native_effect
        self.preserve_only = preserve_only
        self.preserve_state = preserve_state
        self.path_tail = SAFE_PATH_MARKER + "_PATH_TAIL"
        self.symbolic_names = (SAFE_PATH_MARKER, SHELL_PID_MARKER)
        self.steps = 0
        self.depth = 0
        self.temporary = 0
        self.programs = {}
        self.active_redirects: list[str] = []

    def tick(self):
        self.resolver.check_runtime()
        self.steps += 1
        if self.steps > _MAX_STEPS:
            raise _UnresolvedError

    def program(self, path, text):
        if (
            len(text) > 262_144
            or "\x00" in text
            or self.path_tail in text
            or SHELL_PID_MARKER in text
        ):
            raise _UnresolvedError
        key = path, text
        if key in self.programs:
            return self.programs[key]
        if len(self.programs) >= 128:
            raise _UnresolvedError
        tokens, heredocs, pending = [], {}, []
        cursor = 0
        while cursor < len(text):
            self.tick()
            if text[cursor] in " \t\r":
                cursor += 1
                continue
            if text.startswith(("\\\n", "\\\r\n"), cursor):
                cursor += 3 if text.startswith("\\\r\n", cursor) else 2
                continue
            if text[cursor] == "#":
                end = text.find("\n", cursor)
                cursor = len(text) if end < 0 else end
                continue
            if text.startswith("<<", cursor) and not text.startswith("<<<", cursor):
                header = self.resolver.heredoc_header(text, cursor, len(text))
                if header is None or len(pending) >= 32:
                    raise _UnresolvedError
                item, stop = header
                tokens.extend((_Token("<<", cursor, cursor + 2), _Token(item[0], cursor + 2, stop)))
                pending.append((cursor + 2, item))
                cursor = stop
                continue
            operator = next(
                (
                    item
                    for item in (
                        ";;",
                        "&&",
                        "||",
                        ">&",
                        "<&",
                        ">>",
                        "<<<",
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
                cursor + 1 == len(text) or text[cursor + 1] in " \t\r\n;|&()"
            ):
                stop = cursor + 1
            else:
                removal = _QUOTED_PARAMETER_REMOVAL.match(
                    text, cursor, min(len(text), cursor + 512)
                )
                if removal is not None and (
                    removal.end() == len(text) or text[removal.end()] in " \t\r\n;|&()<>"
                ):
                    # The shared multi-shell reader also models ksh's inherited
                    # quote transitions. This complete Bash value-only shape
                    # has independent inner pattern quotes. Keep its boundary
                    # exact; evaluating the removal still requires a value proof.
                    stop = removal.end()
                else:
                    _, stop, limited = self.resolver.read_word(text, cursor, self.tick)
                    if limited or stop <= cursor:
                        raise _UnresolvedError
            tokens.append(_Token(text[cursor:stop], cursor, stop))
            cursor = stop
            if operator == "\n" and pending:
                for owner, item in pending:
                    stop = self.resolver.heredoc_bodies(text, cursor, len(text), (item,), self.tick)
                    if stop is None:
                        raise _UnresolvedError
                    last = stop - int(text[stop - 1 : stop] == "\n")
                    body_end = max(cursor, text.rfind("\n", cursor, last) + 1)
                    body = text[cursor:body_end]
                    if item[1]:
                        body = "\n".join(line.lstrip("\t") for line in body.split("\n"))
                    heredocs[owner] = body, item[2]
                    cursor = stop
                pending.clear()
        if pending:
            raise _UnresolvedError
        program = _Program(path, text, tuple(tokens), heredocs)
        self.programs[key] = program
        return program

    def parse(self, program, tokens=None):
        parser = _EffectsParser(program, self.tick, tokens)
        result = parser.sequence()
        if parser.peek():
            raise _UnresolvedError
        return result

    def path(self, value, frame):
        if value is None or "\x00" in value or "\\" in value:
            return None
        if value.startswith("/"):
            return posixpath.normpath(value)
        return (
            posixpath.normpath(posixpath.join(frame.cwd, value)) if frame.cwd is not None else None
        )

    def disjoint(self, value):
        from skillspector.shell_sources import SAFE_PATH_MARKER

        if value is None:
            return False
        if self.target is None:
            # Collection delegates every known write to the authoritative
            # inventory callback; an unknown destination is still refused.
            return True
        if (
            value.startswith("/tmp/")
            and SAFE_PATH_MARKER in value.split("/")[2]
            and ".." not in value.split("/")
        ):
            return True
        canonical = posixpath.normpath(value)
        if canonical in {".", "/", self.target} or self.target.startswith(canonical + "/"):
            return False
        if SHELL_PID_MARKER in canonical:
            target = self.target
            while target:
                self.tick()
                if _symbolic_path_matches(canonical, target):
                    return False
                target = posixpath.dirname(target)
        return not _symbolic_path_matches(
            canonical.rpartition("/")[2], self.target.rpartition("/")[2]
        )

    def word(self, raw, frame, program, *, assignment=False, final_assignment=False):
        parts, quote, cursor, unknown = [], "", 0, False
        while cursor < len(raw):
            self.tick()
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
                if cursor + 1 == len(raw):
                    raise _UnresolvedError
                following = raw[cursor + 1]
                if quote == '"' and following not in '$`"\\\n':
                    parts.append("\\")
                if following != "\n":
                    parts.append(following)
                cursor += 2
                continue
            if raw.startswith("$(", cursor):
                if raw.startswith("$((", cursor):
                    arithmetic = re.match(r"\$\(\(([0-9 +*()-]+)\)\)", raw[cursor:])
                    if arithmetic is None:
                        raise _UnresolvedError
                    # Constant arithmetic is irrelevant to path effects.
                    unknown = True
                    cursor += arithmetic.end()
                    continue
                if not assignment and not quote:
                    # Substitution output can split into several argv words
                    # and expand globs, just like an unquoted parameter.
                    raise _UnresolvedError
                end = self.resolver.substitution_end(raw, cursor, len(raw))
                if end is None:
                    raise _UnresolvedError
                child = frame.copy()
                child.output, child.flow = "", ""
                child.traps = {}
                # Bash command substitution clears errexit unless explicitly
                # enabled inside; unsupported inherit_errexit settings refuse.
                child.errexit = False
                child.required_path_output = False
                nested = self.program(program.path, raw[cursor + 2 : end - 1])
                nodes = self.parse(nested)
                sole_value = (
                    cursor == 0
                    and end == len(raw)
                    or cursor == 1
                    and raw[:1] == raw[-1:] == '"'
                    and end == len(raw) - 1
                )
                if (
                    final_assignment
                    and frame.errexit
                    and not frame.checked
                    and sole_value
                    and len(nodes) == 1
                    and nodes[0].kind == "command"
                    and not nodes[0].redirects
                    and nodes[0].words
                    and nodes[0].words[0].raw in {"mktemp", "/bin/mktemp", "/usr/bin/mktemp"}
                ):
                    # A failed sole mktemp is the assignment's own status,
                    # so its parent cannot consume a fabricated temp path.
                    child.errexit = True
                if (
                    final_assignment
                    and frame.errexit
                    and not frame.checked
                    and sole_value
                    and len(nodes) == 2
                    and all(node.kind == "command" and not node.redirects for node in nodes)
                    and len(nodes[0].words) == 2
                    and nodes[0].words[0].raw == "cd"
                    and tuple(word.raw for word in nodes[1].words) == ("pwd",)
                    and nodes[1].connector == "&&"
                    and not {"cd", "pwd"}.intersection(frame.functions)
                ):
                    # In this exact final assignment, failed cd skips pwd and
                    # fails the parent assignment. Only a successful path can
                    # reach its later consumer; no other child effects borrow it.
                    child.required_path_output = True
                branches = self.run(nested, nodes, [child])
                self.apply_traps(branches)
                values = {
                    branch.output.rstrip("\n") if branch.output is not None else None
                    for branch in branches
                }
                value = next(iter(values)) if len(values) == 1 else None
                if value is None:
                    unknown = True
                else:
                    parts.append(value)
                cursor = end
                continue
            if character == "$":
                parameter = re.match(
                    r"\$(?:([@*#?$!0-9]|[A-Za-z_][A-Za-z0-9_]*)|\{([^{}]+)\})", raw[cursor:]
                )
                if parameter is None:
                    if cursor + 1 == len(raw) or quote == '"' and raw[cursor + 1 :] == '"':
                        parts.append("$")
                        cursor += 1
                        continue
                    raise _UnresolvedError
                name = parameter[1] or parameter[2]
                if name == "BASH_SOURCE[0]":
                    value = program.path
                elif name == "0":
                    value = frame.zero
                elif name == "#":
                    value = str(len(frame.argv))
                elif name == "?":
                    # A failed condition records truth, not its exact exit
                    # code. It cannot justify skipping an rc==2 branch.
                    value = "0" if frame.status is True else None
                elif name in {"@", "*"}:
                    if name == "@" and not assignment:
                        # Embedded quoted @ can produce several argv words.
                        # Only arguments() owns the exact standalone "$@".
                        raise _UnresolvedError
                    value = (
                        " ".join(frame.argv)
                        if all(isinstance(item, str) for item in frame.argv)
                        else None
                    )
                elif name == "$":
                    value = SHELL_PID_MARKER
                elif name == "!":
                    value = None
                elif name == "!#" and self.preserve_only:
                    # The last positional value is data, not an unevaluated
                    # program. Its unknown value still cannot select a path,
                    # executable, or source in the existing dispatch checks.
                    value = None
                elif re.fullmatch(r"[0-9]{1,16}", name):
                    position = int(name)
                    value = (
                        frame.zero
                        if position == 0
                        else frame.argv[position - 1]
                        if position <= len(frame.argv)
                        else ""
                    )
                else:
                    match = re.fullmatch(
                        r"([A-Za-z_][A-Za-z0-9_]*|[0-9]{1,16})(?::([-+?])(.*))?", name, re.DOTALL
                    )
                    array = re.fullmatch(
                        r"(#?)([A-Za-z_][A-Za-z0-9_]*)\[(0|[1-9][0-9]{0,2}|@)\]", name
                    )
                    if array:
                        if array[1] and array[3] != "@":
                            raise _UnresolvedError
                        items = frame.lookup(array[2])
                        value = (
                            str(len(items))
                            if isinstance(items, tuple) and array[1] and array[3] == "@"
                            else items[int(array[3])]
                            if isinstance(items, tuple)
                            and array[3].isdecimal()
                            and int(array[3]) < len(items)
                            else None
                        )
                    elif match:
                        if match[1].isdecimal():
                            position = int(match[1])
                            value = (
                                frame.zero
                                if position == 0
                                else frame.argv[position - 1]
                                if position <= len(frame.argv)
                                else ""
                            )
                        else:
                            value = frame.lookup(match[1])
                        unused_word = value is not None and (
                            match[2] in {"-", "?"} and bool(value) or match[2] == "+" and not value
                        )
                        if (
                            match[2]
                            and not unused_word
                            and re.fullmatch(r"[-A-Za-z0-9_./]*", match[3]) is None
                        ):
                            # Parameter fallback words have their own shell
                            # expansions. Only a proven unused word may skip
                            # those effects, including diagnostic messages.
                            raise _UnresolvedError
                        if match[2] and value is not None:
                            if match[2] == "-" and not value:
                                value = match[3]
                            elif match[2] == "+":
                                value = match[3] if value else ""
                            elif match[2] == "?" and not value:
                                raise _UnresolvedError
                    else:
                        raise _UnresolvedError
                if not assignment and not quote:
                    # Field splitting and pathname expansion need an argv proof.
                    raise _UnresolvedError
                if value is None:
                    unknown = True
                elif not isinstance(value, str):
                    raise _UnresolvedError
                else:
                    parts.append(value)
                cursor += parameter.end()
                continue
            if (
                character == "`"
                or not quote
                and (character == "~" or not assignment and character in "*?[]{}")
            ):
                raise _UnresolvedError
            parts.append(character)
            cursor += 1
        if quote:
            raise _UnresolvedError
        result = "".join(parts)
        if len(result) > 131_072:
            raise _UnresolvedError
        return None if unknown else result

    def arguments(self, words, frame, program):
        output = []
        for token in words:
            if token.raw == '"$@"':
                output.extend(frame.argv)
                continue
            array = re.fullmatch(r'"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}"', token.raw)
            if array:
                items = frame.lookup(array[1])
                if not isinstance(items, tuple):
                    raise _UnresolvedError
                output.extend(items)
                continue
            output.append(self.word(token.raw, frame, program))
        if len(output) > 128:
            raise _UnresolvedError
        return tuple(output)

    def run(self, program, nodes, frames):
        self.depth += 1
        try:
            if self.depth > _MAX_DEPTH:
                raise _UnresolvedError
            for index, node in enumerate(nodes):
                output = []
                for frame in frames:
                    self.tick()
                    if (
                        frame.flow
                        or node.connector == "&&"
                        and frame.status is False
                        or node.connector == "||"
                        and frame.status is True
                    ):
                        output.append(frame)
                        continue
                    if node.connector in {"&&", "||"} and frame.status is None:
                        skipped = frame.copy()
                        skipped.status = node.connector == "||"
                        skipped.uncertain = frame.uncertain = True
                        output.append(skipped)
                    previous_checked = frame.checked
                    frame.checked |= index + 1 < len(nodes) and nodes[index + 1].connector in {
                        "&&",
                        "||",
                    }
                    branches = self.node(program, node, frame)
                    for branch in branches:
                        branch.checked = previous_checked
                    output.extend(branches)
                if len(output) > _MAX_STATES:
                    raise _UnresolvedError
                frames = self.join(output) if self.preserve_only else output
            return frames
        finally:
            self.depth -= 1

    def join(self, frames):
        """Join only data facts; possible effects already reached the callback."""
        if len(frames) > _MAX_STATES:
            raise _UnresolvedError
        groups = {}
        for frame in frames:
            self.tick()
            groups.setdefault(frame.flow, []).append(frame)
        output = []
        for alternatives in groups.values():
            first = alternatives[0]
            if len(alternatives) == 1 or first.flow == "exit":
                # Exited alternatives cannot call later commands, but their
                # EXIT traps must retain each original late-bound function.
                output.extend(alternatives)
                continue
            if any(
                branch.functions != first.functions
                or branch.traps != first.traps
                or branch.exported != first.exported
                or len(branch.argv) != len(first.argv)
                or [set(scope) for scope in branch.scopes] != [set(scope) for scope in first.scopes]
                for branch in alternatives[1:]
            ):
                # Unknown code bindings or local-scope lookup order need a
                # separate identity proof, not a fallback to native commands.
                raise _UnresolvedError

            def equal(values):
                value = values[0]
                return value if all(item == value for item in values[1:]) else None

            def bindings(mappings):
                result = {}
                for name in set().union(*mappings):
                    self.tick()
                    result[name] = equal([mapping.get(name) for mapping in mappings])
                return result

            joined = first.copy()
            joined.values = bindings([branch.values for branch in alternatives])
            joined.scopes = [
                bindings([branch.scopes[index] for branch in alternatives])
                for index in range(len(first.scopes))
            ]
            joined.argv = tuple(
                equal([branch.argv[index] for branch in alternatives])
                for index in range(len(first.argv))
            )
            for name in ("zero", "cwd", "status", "output"):
                setattr(joined, name, equal([getattr(branch, name) for branch in alternatives]))
            for name in ("uncertain", "checked", "asynchronous"):
                setattr(joined, name, any(getattr(branch, name) for branch in alternatives))
            for name in ("errexit", "required_path_output"):
                setattr(joined, name, all(getattr(branch, name) for branch in alternatives))
            output.append(joined)
        return output

    def node(self, program, node, frame):
        if node.kind == "function":
            if node.words[0].raw in {
                "local",
                "export",
                "unset",
                "test",
                "source",
                "return",
                "exit",
                "break",
                "continue",
                "shift",
                "command",
                "trap",
            }:
                raise _UnresolvedError
            frame.functions[node.words[0].raw] = _Function(program, node.words[1:])
            frame.status = True
            return [frame]
        if node.kind == "arithmetic":
            text = " ".join(token.raw for token in node.words)
            if text == "$#":
                frame.status = bool(frame.argv)
            elif re.fullmatch(r"0|[1-9][0-9]{0,15}", text):
                frame.status = bool(int(text))
            else:
                raise _UnresolvedError
            return [frame]
        if node.kind == "negate":
            frame.checked = True
            branches = self.run(program, node.blocks[0], [frame])
            for branch in branches:
                branch.status = None if branch.status is None else not branch.status
            return branches
        if node.kind == "if":
            remaining, result = [frame], []
            previous_checked = frame.checked
            for index in range(0, len(node.blocks) - 1, 2):
                next_remaining = []
                for condition in remaining:
                    condition.checked = True
                for condition in self.run(program, node.blocks[index], remaining):
                    condition.checked = previous_checked
                    if condition.status is not False:
                        accepted = condition.copy()
                        accepted.uncertain |= condition.status is None
                        result.extend(self.run(program, node.blocks[index + 1], [accepted]))
                    if condition.status is not True:
                        condition.uncertain |= condition.status is None
                        next_remaining.append(condition)
                remaining = next_remaining
            if not node.blocks[-1]:
                for branch in remaining:
                    branch.status = True
            return result + self.run(program, node.blocks[-1], remaining)
        if node.kind == "case":
            value = self.word(node.words[0].raw, frame, program)
            if value is not None and any(marker in value for marker in self.symbolic_names):
                value = None
            result = []
            for labels, body in zip(node.labels, node.blocks, strict=True):
                if value is None or any(fnmatch.fnmatchcase(value, label) for label in labels):
                    branch = frame.copy()
                    branch.uncertain |= value is None
                    if not body:
                        branch.status = True
                    result.extend(self.run(program, body, [branch]))
                    if value is not None:
                        return result
            if value is None:
                frame.uncertain = True
            frame.status = True
            return result + [frame]
        if node.kind == "redirect":
            redirects = tuple(
                (operator, self.word(token.raw, frame, program))
                for operator, token in node.redirects
            )
            request = ShellEffectCommand(
                (":",),
                {},
                frame.cwd,
                program.path,
                node.redirects[0][1].start,
                redirects,
                uncertain=frame.uncertain,
                fail_stop=frame.errexit and not frame.checked,
            )
            opened = self.begin_redirects(request, frame)
            previous_output = frame.output
            frame.output = ""
            branches = self.run(program, node.blocks[0], [frame])
            self.end_redirects(opened)
            for branch in branches:
                output = branch.output
                branch.output = previous_output
                completion = replace(
                    request,
                    stdout=output,
                    uncertain=branch.uncertain,
                    child_inspected=True,
                    child_status=branch.status,
                )
                self.native(program, branch, completion, child=True)
            return branches
        if node.kind in {"group", "subshell", "background"}:
            if node.kind == "group":
                return self.run(program, node.blocks[0], [frame])
            nested = frame.copy()
            nested.traps = {}
            nested.uncertain |= node.kind == "background"
            nested.asynchronous |= node.kind == "background"
            branches = self.run(program, node.blocks[0], [nested])
            self.apply_traps(branches)
            frame.status = (
                True
                if node.kind == "background"
                else branches[0].status
                if len(branches) == 1
                else None
            )
            if node.kind != "background":
                values = {branch.output for branch in branches}
                frame.output = values.pop() if len(values) == 1 else None
            return [frame]
        if node.kind == "pipeline":
            for body in node.blocks:
                # Pipeline members share files, not shell-variable changes.
                nested = frame.copy()
                nested.traps = {}
                nested.uncertain = True
                nested.asynchronous = True
                nested.checked = True
                self.apply_traps(self.run(program, body, [nested]))
            frame.output, frame.status = None, None
            return [frame]
        if node.kind == "for":
            name = node.words[0].raw
            if self.preserve_only and len(node.words) == 3:
                array = re.fullmatch(r'"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}"', node.words[2].raw)
                if array is not None and frame.lookup(array[1]) is None:
                    return self.unknown_for(program, node, frame)
            values = (
                self.arguments(node.words[2:], frame, program)
                if len(node.words) > 1
                else frame.argv
            )
            if len(values) > 128:
                raise _UnresolvedError
            if not values:
                frame.status = True
            branches = [frame]
            for value in values:
                for branch in branches:
                    branch.assign(name, value)
                branches = self.run(program, node.blocks[0], branches)
                for branch in branches:
                    if branch.flow in {"break", "continue"}:
                        raise _UnresolvedError
            return branches
        if node.kind in {"while", "until"}:
            branches, completed = [frame], []
            previous_checked = frame.checked
            for iteration in range(128):
                pending = []
                for branch in branches:
                    branch.checked = True
                for branch in self.run(program, node.blocks[0], branches):
                    branch.checked = previous_checked
                    if branch.status is None:
                        raise _UnresolvedError
                    if branch.status == (node.kind == "until"):
                        # The terminating condition is not the loop's exit
                        # status. Retain uncertainty after any body ran.
                        branch.status = True if iteration == 0 else None
                        completed.append(branch)
                    else:
                        pending.append(branch)
                if not pending:
                    return completed
                branches = self.run(program, node.blocks[1], pending)
                remaining = []
                for branch in branches:
                    if branch.flow == "break":
                        branch.flow = ""
                        completed.append(branch)
                    elif branch.flow in {"return", "exit"}:
                        completed.append(branch)
                    else:
                        if branch.flow == "continue":
                            branch.flow = ""
                        remaining.append(branch)
                branches = remaining
                if not branches:
                    return completed
            raise _UnresolvedError
        return self.command(program, node, frame)

    def unknown_for(self, program, node, frame):
        """Audit any finite array length without inventing its data values."""
        if self.preserve_state is None:
            raise _UnresolvedError
        zero = frame.copy()
        zero.status = True
        completed = [zero]
        entry = frame.copy()
        entry.assign(node.words[0].raw, None)
        entry.uncertain = True
        entry.status = None
        entry.output = None
        for _ in range(_MAX_DEPTH):
            self.tick()
            previous_effects = self.preserve_state()
            body = entry.copy()
            body.assign(node.words[0].raw, None)
            continuing = []
            for branch in self.run(program, node.blocks[0], [body]):
                if branch.flow in {"return", "exit"}:
                    completed.append(branch)
                elif branch.flow == "break":
                    branch.flow = ""
                    completed.append(branch)
                else:
                    if branch.flow == "continue":
                        branch.flow = ""
                    continuing.append(branch)
            if not continuing:
                return self.join(completed)
            # Include the earlier entry before another iteration. Changed
            # data becomes unknown before it can choose the next effects.
            (successor,) = self.join([entry, *continuing])
            successor.status = None
            successor.output = None
            if successor == entry and previous_effects == self.preserve_state():
                return self.join([*completed, successor])
            entry = successor
        raise _UnresolvedError

    def condition(self, raw, frame, program, depth=0, *, extended=True):
        self.tick()
        if depth > _MAX_DEPTH or not raw or any(item in {"(", ")"} for item in raw):
            raise _UnresolvedError
        if not extended and (len(raw) > 4 or any(item in {"&&", "||"} for item in raw)):
            raise _UnresolvedError
        # Bash binds ! more tightly than &&, and && more tightly than ||.
        # Split the outer logical expression before interpreting its terms.
        for separator in ("||", "&&"):
            if separator in raw:
                at = raw.index(separator)
                left = self.condition(raw[:at], frame, program, depth + 1, extended=extended)
                if separator == "&&" and left is False or separator == "||" and left is True:
                    return left
                right_frame = frame.copy()
                right_frame.uncertain |= left is None
                right = self.condition(
                    raw[at + 1 :], right_frame, program, depth + 1, extended=extended
                )
                if separator == "&&":
                    return (
                        False
                        if left is False or right is False
                        else True
                        if left is True and right is True
                        else None
                    )
                return (
                    True
                    if left is True or right is True
                    else False
                    if left is False and right is False
                    else None
                )
        if raw[:1] == ("!",):
            value = self.condition(raw[1:], frame, program, depth + 1, extended=extended)
            return None if value is None else not value
        if len(raw) == 1:
            value = self.word(raw[0], frame, program, assignment=extended)
            return bool(value) if value is not None else None
        if extended and len(raw) == 3 and raw[1] == "=~":
            for operand in (raw[0], raw[2]):
                self.word(operand, frame, program, assignment=True)
            return None
        values = tuple(self.word(item, frame, program, assignment=extended) for item in raw)
        if frame.asynchronous and (
            len(values) == 2
            and values[0] not in {"-n", "-z"}
            or len(values) == 3
            and values[1] in {"-nt", "-ot", "-ef"}
        ):
            # An asynchronous child may observe files after later parent
            # writes. The current artifact snapshot cannot prune its branches.
            raise _UnresolvedError
        if (
            len(values) == 2
            and values[0] in {"-e", "-f", "-s", "-x", "-L", "-d"}
            and self.native_effect is not None
        ):
            path = self.path(values[1], frame)
            if path is not None and self.conflicts_with_redirect(path):
                raise _UnresolvedError
            request = ShellEffectCommand(
                ("test", *values), {}, frame.cwd, program.path, 0, uncertain=frame.uncertain
            )
            result = self.native_effect(request)
            if result is None or result.writes or result.script_path is not None:
                return None
            return result.status
        if len(values) == 2 and values[0] in {"-n", "-z"}:
            return None if values[1] is None else bool(values[1]) == (values[0] == "-n")
        if len(values) == 3 and values[1] in {
            "=",
            "==",
            "!=",
            "-eq",
            "-ne",
            "-gt",
            "-ge",
            "-lt",
            "-le",
        }:
            left, operator, right = values
            if left is None or right is None:
                return None
            if any(marker in left or marker in right for marker in self.symbolic_names):
                # Fresh-path and PID placeholders describe runtime names,
                # not literal strings that can select a single branch.
                return None
            if operator in {"=", "==", "!="}:
                if not extended:
                    return (left == right) != (operator == "!=")
                # A fully quoted RHS is literal; mixed quoted/unquoted pieces
                # preserve quoting per character and cannot use the flattened
                # value as either a plain string or an unqualified pattern.
                literal = (
                    re.fullmatch(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"", raw[2], re.DOTALL) is not None
                )
                if not literal and (
                    any(character in raw[2] for character in "'\"\\")
                    or any(character in right for character in "[]()|")
                ):
                    return None
                matched = left == right if literal else fnmatch.fnmatchcase(left, right)
                return matched != (operator == "!=")
            if not re.fullmatch(r"-?(?:0|[1-9][0-9]{0,15})", left) or not re.fullmatch(
                r"-?(?:0|[1-9][0-9]{0,15})", right
            ):
                return None
            a, b = int(left), int(right)
            return {
                "-eq": a == b,
                "-ne": a != b,
                "-gt": a > b,
                "-ge": a >= b,
                "-lt": a < b,
                "-le": a <= b,
            }[operator]
        return None

    def process_input(self, program, node, frame):
        words = [token.raw for token in node.words]
        if len(node.blocks) != 1 or tuple(operator for operator, _ in node.redirects) != (
            "0<process",
        ):
            raise _UnresolvedError
        if words and words[0].startswith("IFS="):
            separator = words.pop(0)[4:]
            if (
                not re.fullmatch(r"(['\"])[!-~]\1", separator)
                or separator[1] in "'\"\\$`"
                or words[:1] != ["read"]
            ):
                raise _UnresolvedError
        if words[:2] == ["read", "-r"]:
            names = words[2:]
        elif len(words) == 3 and words[:2] == ["mapfile", "-t"]:
            names = words[2:]
        else:
            raise _UnresolvedError
        if (
            not names
            or len(names) > 128
            or words[0] in frame.functions
            or any(not _NAME.fullmatch(name) or is_managed_shell_variable(name) for name in names)
        ):
            raise _UnresolvedError
        # Process substitution runs concurrently. Restrict its effects to
        # copied functions and pure builtins, without snapshot-dependent reads.
        child = frame.copy()
        child.traps = {}
        child.output = ""
        child.uncertain = True
        child.asynchronous = True
        child.errexit = False
        child.checked = True
        self.apply_traps(self.run(program, node.blocks[0], [child]))
        # A successful consumer does not prove that the producer completed.
        # Keep values and status unknown until a separate output proof exists.
        for name in names:
            frame.assign(name, None)
        frame.status = None
        return [frame]

    def command(self, program, node, frame):
        if node.blocks:
            return self.process_input(program, node, frame)
        if frame.asynchronous and node.redirects:
            raise _UnresolvedError
        words = list(node.words)
        environment = {
            name: value if isinstance(value := frame.lookup(name), str) else None
            for name in frame.exported
        }
        prefix = []
        assignment_frame = frame.copy()
        assignment_substitution = False
        while words and (assignment := _ASSIGN.fullmatch(words[0].raw)):
            token = words.pop(0)
            name, raw = assignment.groups()
            assignment_substitution |= "$(" in raw.replace("$((", "") or "`" in raw
            if words[:1] and words[0].raw == "(":
                closing = next((index for index, word in enumerate(words) if word.raw == ")"), -1)
                if raw or closing < 0:
                    raise _UnresolvedError
                value = self.arguments(words[1:closing], assignment_frame, program)
                del words[: closing + 1]
            else:
                value = self.word(
                    raw,
                    assignment_frame,
                    program,
                    assignment=True,
                    final_assignment=len(node.words) == 1 and not node.redirects,
                )
            if is_managed_shell_variable(name):
                if name != "PATH" or not words or not isinstance(value, str):
                    raise _UnresolvedError
            prefix.append((name, value))
            environment[name] = value if isinstance(value, str) else None
            if name == "PATH":
                assignment_frame.values[name] = value
            else:
                assignment_frame.assign(name, value)
        redirects, stdin = [], None
        for operator, token in node.redirects:
            if operator == "0<<":
                body, literal = program.heredocs[token.start]
                if not literal:
                    # Expandable heredocs can execute substitutions. Refuse
                    # until their distinct quote rules have an effects proof.
                    raise _UnresolvedError
                stdin = body
            else:
                value = self.word(token.raw, frame, program)
                if operator == "0<<<":
                    stdin = None if value is None else value + "\n"
                else:
                    redirects.append((operator, value))
        if not words:
            if redirects:
                self.native(
                    program,
                    frame,
                    ShellEffectCommand(
                        (":",),
                        environment,
                        frame.cwd,
                        program.path,
                        node.redirects[0][1].start,
                        tuple(redirects),
                    ),
                )
            for name, value in prefix:
                frame.assign(name, value)
            frame.status = None if assignment_substitution else True
            return [frame]
        # Redirections happen before dispatch, including tests and builtins.
        for operator, value in redirects:
            if operator in {"1>&", "2>&", "0<&"} and value in {"0", "1", "2", "-"}:
                continue
            if (
                ">" in operator
                and value != "/dev/null"
                and not self.disjoint(self.path(value, frame))
            ):
                raise _UnresolvedError
        raw = tuple(word.raw for word in words)
        if raw[0] in {"[[", "[", "test"}:
            if prefix or redirects:
                raise _UnresolvedError
            closing = "]]" if raw[0] == "[[" else "]" if raw[0] == "[" else None
            if closing is not None and raw[-1] != closing:
                raise _UnresolvedError
            frame.status = self.condition(
                raw[1:-1] if closing else raw[1:], frame, program, extended=raw[0] == "[["
            )
            return [frame]
        if raw[0] in {"local", "export", "unset"}:
            if prefix or redirects:
                raise _UnresolvedError
            local_words = words[1:]
            if raw[0] == "local" and local_words[:1] and local_words[0].raw == "-a":
                local_words = local_words[1:]
                if (
                    len(local_words) >= 3
                    and local_words[1].raw == "("
                    and local_words[-1].raw == ")"
                ):
                    assignment = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=", local_words[0].raw)
                    if assignment is None:
                        raise _UnresolvedError
                    frame.assign(
                        assignment[1], self.arguments(local_words[2:-1], frame, program), local=True
                    )
                    frame.status = True
                    return [frame]
            for token in local_words:
                assignment = _ASSIGN.fullmatch(token.raw)
                if assignment:
                    name, value = (
                        assignment[1],
                        self.word(assignment[2], frame, program, assignment=True),
                    )
                elif _NAME.fullmatch(token.raw):
                    name, value = (
                        token.raw,
                        None if raw[0] in {"local", "unset"} else frame.lookup(token.raw),
                    )
                else:
                    raise _UnresolvedError
                frame.assign(name, value, local=raw[0] == "local")
                if raw[0] == "export":
                    frame.exported.add(name)
                elif raw[0] == "unset":
                    frame.exported.discard(name)
            frame.status = True
            return [frame]
        argv = self.arguments(words, frame, program)
        name = argv[0]
        if name is None:
            raise _UnresolvedError
        if name in frame.functions:
            if prefix:
                raise _UnresolvedError
            function = frame.functions[name]
            previous = frame.argv
            previous_output = frame.output
            redirect_cwd = frame.cwd
            request = ShellEffectCommand(
                argv,
                environment,
                redirect_cwd,
                program.path,
                words[0].start,
                tuple(redirects),
                stdin,
                uncertain=frame.uncertain,
                fail_stop=frame.errexit and not frame.checked,
            )
            opened = self.begin_redirects(request, frame)
            frame.argv = argv[1:]
            frame.output = ""
            frame.scopes.append({})
            branches = self.run(
                function.program, self.parse(function.program, function.tokens), [frame]
            )
            self.end_redirects(opened)
            for branch in branches:
                branch.scopes.pop()
                branch.argv = previous
                if branch.flow == "return":
                    branch.flow = ""
                output = branch.output
                branch.output = previous_output
                completion = replace(
                    request,
                    stdout=output,
                    uncertain=branch.uncertain,
                    child_inspected=True,
                    child_status=branch.status,
                )
                self.native(program, branch, completion, child=True)
            return branches
        if name in {"source", "."}:
            if frame.asynchronous or prefix or redirects or len(argv) < 2 or argv[1] is None:
                raise _UnresolvedError
            path = self.path(argv[1], frame)
            if path is not None and self.conflicts_with_redirect(path):
                raise _UnresolvedError
            text = self.read_source(path) if path is not None else None
            if text is None:
                raise _UnresolvedError
            imported = self.program(path, text)
            previous = frame.argv
            if len(argv) > 2:
                frame.argv = argv[2:]
            branches = self.run(imported, self.parse(imported), [frame])
            for branch in branches:
                branch.argv = previous
                if branch.flow == "return":
                    branch.flow = ""
            return branches
        if name in {"return", "exit", "break", "continue"}:
            if prefix or redirects or len(argv) > 2:
                raise _UnresolvedError
            frame.flow = name
            frame.status = (
                frame.status
                if len(argv) == 1
                else argv[1] == "0"
                if argv[1] in {"0", "1", "2"}
                else None
            )
            return [frame]
        if name == "shift":
            amount = "1" if len(argv) == 1 else argv[1] if len(argv) == 2 else None
            if (
                amount is None
                or not amount.isdecimal()
                or not 0 < int(amount) <= len(frame.argv)
                or prefix
                or redirects
            ):
                raise _UnresolvedError
            frame.argv = frame.argv[int(amount) :]
            frame.status = True
            return [frame]
        if name == "command":
            if len(argv) > 1 and argv[1] not in {"-v", "-V"}:
                # command bypasses shell functions but preserves native argv.
                argv = argv[1:]
                name = argv[0]
        if name == "printf" and len(argv) >= 4 and argv[1] == "-v":
            if prefix or redirects or argv[2] is None:
                raise _UnresolvedError
            frame.assign(argv[2], self.printf(argv[3:]))
            frame.status = True
            return [frame]
        if name in {"bash", "/bin/bash", "/usr/bin/bash", "sh", "/bin/sh", "/usr/bin/sh"}:
            if frame.asynchronous:
                raise _UnresolvedError
            if (
                name in {"bash", "/bin/bash", "/usr/bin/bash"}
                and len(argv) >= 2
                and argv[1] is not None
                and not argv[1].startswith("-")
            ):
                if self.native_effect is None:
                    raise _UnresolvedError
                request = ShellEffectCommand(
                    argv[1:],
                    environment,
                    frame.cwd,
                    program.path,
                    words[0].start,
                    tuple(redirects),
                    stdin,
                    uncertain=frame.uncertain,
                    interpreter=name,
                )
                self.native(program, frame, request)
                return [frame]
            if "/" not in name and environment.get("PATH", self.path_tail) != self.path_tail:
                if self.native_effect is None:
                    raise _UnresolvedError
                identity_request = ShellEffectCommand(
                    argv,
                    environment,
                    frame.cwd,
                    program.path,
                    words[0].start,
                    uncertain=frame.uncertain,
                    identity_only=True,
                )
                identity = self.native_effect(identity_request)
                if (
                    identity is None
                    or not identity.native_identity
                    or identity.script_path is not None
                    or identity.writes
                ):
                    raise _UnresolvedError
            if len(argv) < 3 or argv[1] != "-c" or argv[2] is None:
                raise _UnresolvedError
            redirect_request = ShellEffectCommand(
                argv,
                environment,
                frame.cwd,
                program.path,
                words[0].start,
                tuple(redirects),
                stdin,
                uncertain=frame.uncertain,
                fail_stop=frame.errexit and not frame.checked,
            )
            opened = self.begin_redirects(redirect_request, frame)
            child = _Frame(
                environment,
                argv[4:],
                argv[3] if len(argv) > 3 and argv[3] is not None else name,
                frame.cwd,
                uncertain=frame.uncertain,
                exported=set(environment),
            )
            nested = self.program(program.path, argv[2])
            branches = self.run(nested, self.parse(nested), [child])
            self.apply_traps(branches)
            self.end_redirects(opened)
            frame.status = branches[0].status if len(branches) == 1 else None
            outputs = {branch.output for branch in branches}
            output = outputs.pop() if len(outputs) == 1 else None
            request = replace(
                redirect_request, stdout=output, child_inspected=True, child_status=frame.status
            )
            self.native(program, frame, request, child=True)
            return [frame]
        if name == "trap":
            if (
                prefix
                or redirects
                or len(argv) != 3
                or argv[1] is None
                or argv[2] not in {"EXIT", "0"}
            ):
                raise _UnresolvedError
            if argv[1] == "-":
                frame.traps.pop("EXIT", None)
            else:
                frame.traps["EXIT"] = program, argv[1]
            frame.status = True
            return [frame]
        stdout = self.printf(argv[1:]) if name == "printf" else None
        request = ShellEffectCommand(
            argv,
            environment,
            frame.cwd,
            program.path,
            words[0].start,
            tuple(redirects),
            stdin,
            stdout,
            frame.uncertain,
        )
        self.native(program, frame, request)
        return [frame]

    def printf(self, arguments):
        if not arguments or arguments[0] is None:
            raise _UnresolvedError
        format_string, *values = arguments
        if re.search(r"%(?![%sq])", format_string) or re.search(r"\\(?![nrt\\])", format_string):
            raise _UnresolvedError
        if "%q" in format_string or any(value is None for value in values):
            return None
        output, cursor, argument = [], 0, 0
        while cursor < len(format_string):
            self.tick()
            character = format_string[cursor]
            if character == "\\":
                output.append(
                    {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}[format_string[cursor + 1]]
                )
                cursor += 2
            elif character == "%":
                if format_string[cursor + 1] == "%":
                    output.append("%")
                else:
                    output.append(values[argument] if argument < len(values) else "")
                    argument += 1
                cursor += 2
            else:
                output.append(character)
                cursor += 1
        # printf repeats a format for remaining arguments; prove a single
        # simple %s conversion only, rather than guessing complex repetition.
        if argument < len(values):
            if format_string == "%s\\n":
                output.extend(value + "\n" for value in values[argument:])
            elif format_string == "%s":
                output.extend(values[argument:])
            else:
                raise _UnresolvedError
        result = "".join(output)
        if len(result) > 131_072:
            raise _UnresolvedError
        return result

    def apply_traps(self, frames):
        for frame in frames:
            if "EXIT" in frame.traps:
                program, text = frame.traps.pop("EXIT")
                child = frame.copy()
                child.flow = ""
                trap = self.program(program.path, text)
                self.run(trap, self.parse(trap), [child])
                # The trap may append to stdout or replace the shell's exit
                # status. Until those alternatives are joined, neither can
                # justify exact generated bytes or prune a failure branch.
                frame.output = None
                frame.status = None

    def conflicts_with_redirect(self, path):
        for target in self.active_redirects:
            while target:
                self.tick()
                if _symbolic_path_matches(path, target):
                    return True
                parent = posixpath.dirname(target)
                if parent == target:
                    break
                target = parent
        return False

    def begin_redirects(self, request, frame):
        if frame.asynchronous and request.redirects:
            raise _UnresolvedError
        paths = [
            self.path(value, frame)
            for operator, value in request.redirects
            if operator in {"1>", "1>>", "2>", "2>>"} and value != "/dev/null"
        ]
        if any(
            path is None or not self.disjoint(path) or self.conflicts_with_redirect(path)
            for path in paths
        ):
            raise _UnresolvedError
        if paths and self.native_effect is not None:
            # Open happens before the body. Reads of the active destination
            # are separately refused because stdout is accumulated, not streamed.
            opening = replace(
                request, stdout=None, child_inspected=True, child_status=None, redirect_open=True
            )
            result = self.native_effect(opening)
            if (
                result is None
                or result.script_path is not None
                or any(
                    not self.disjoint(path) or self.conflicts_with_redirect(path)
                    for path in result.writes
                )
            ):
                raise _UnresolvedError
        self.active_redirects.extend(paths)
        return len(paths)

    def end_redirects(self, count):
        if count:
            del self.active_redirects[-count:]

    def asynchronous_reader(self, request):
        name = request.argv[0]
        if name is None:
            return None
        basename = name.rpartition("/")[2]
        if name not in {basename, "/bin/" + basename, "/usr/bin/" + basename}:
            return None
        if name == basename and request.environment.get("PATH", self.path_tail) != self.path_tail:
            return None
        args = request.argv[1:]
        if basename == "cat" and not args:
            return basename
        if (
            basename == "df"
            and len(args) == 3
            and args[:2] == ("-B1", "--output=avail")
            and args[2] is not None
            and not args[2].startswith("-")
        ):
            return basename
        if (
            basename == "awk"
            and len(args) == 1
            and args[0] is not None
            and re.fullmatch(
                r"[ \t]*NR[ \t]*==[ \t]*2[ \t]*\{[ \t]*print[ \t]+\$1[ \t]*;?[ \t]*\}[ \t]*",
                args[0],
            )
        ):
            return basename
        return None

    def native(self, program, frame, request, *, child=False):
        asynchronous_reader = (
            self.asynchronous_reader(request) if frame.asynchronous and not child else None
        )
        if frame.asynchronous and (
            request.interpreter is not None
            or request.redirects
            or not (child and request.child_inspected)
            and asynchronous_reader is None
            and request.argv[0] not in {":", "true", "false", "printf", "echo", "set"}
        ):
            raise _UnresolvedError
        if not request.child_inspected:
            request = replace(
                request,
                fail_stop=(frame.errexit and not frame.checked)
                or (
                    frame.required_path_output
                    and request.interpreter is None
                    and request.argv[0] == "cd"
                ),
            )
        redirect_frame = replace(frame, cwd=request.cwd)
        argument_conflict = False
        # Reached child bodies already audited their reads. Completion only
        # publishes their output; its invocation arguments are not new reads.
        arguments = () if child and request.child_inspected else request.argv
        for value in arguments:
            path = self.path(value, redirect_frame)
            if path is not None and self.conflicts_with_redirect(path):
                argument_conflict = True
                break
        if argument_conflict and asynchronous_reader != "df":
            if (
                request.child_inspected
                or request.interpreter is not None
                or self.native_effect is None
                or len(request.argv) != 4
                or request.argv[0] not in {"df", "/bin/df", "/usr/bin/df"}
                or request.argv[1:3] != ("-B1", "--output=avail")
                or request.argv[3] is None
            ):
                raise _UnresolvedError
            identity = self.native_effect(
                replace(request, redirects=(), stdin=None, stdout=None, identity_only=True)
            )
            if (
                identity is None
                or not identity.native_identity
                or identity.script_path is not None
                or identity.writes
                or identity.stdout is not None
            ):
                raise _UnresolvedError
            # Proven native df reads filesystem metadata, not the bytes of an
            # active output beneath that path. Its result remains unknown.
        for operator, value in request.redirects:
            if operator in {"1>&", "2>&", "0<&"} and value in {"0", "1", "2", "-"}:
                continue
            path = self.path(value, redirect_frame)
            if path is not None and self.conflicts_with_redirect(path):
                raise _UnresolvedError
            if ">" in operator and value != "/dev/null" and not self.disjoint(path):
                raise _UnresolvedError
        if frame.asynchronous:
            # Proven builtin/reader identities require no mutable artifact
            # snapshot. Stdin filters and disk metadata keep unknown output.
            result = (
                ShellEffectResult()
                if asynchronous_reader is not None
                else self.native_default(replace(request, environment={}), frame, child=child)
            )
            if result is not None:
                result = replace(result, native_identity=not child)
        else:
            result = (
                self.native_effect(request)
                if self.native_effect is not None
                else self.native_default(request, frame, child=child)
            )
        if result is None or any(
            not self.disjoint(path) or self.conflicts_with_redirect(path) for path in result.writes
        ):
            raise _UnresolvedError
        if result.stdout is not None and SHELL_PID_MARKER in result.stdout:
            result = replace(result, stdout=None)
        if request.argv[0] == "set" and (result.native_identity or self.native_effect is None):
            if request.argv[1:] in {("-euo", "pipefail"), ("-Eeuo", "pipefail"), ("-e",)}:
                frame.errexit = True
            elif request.argv[1:] == ("+e",):
                frame.errexit = False
        if request.argv[0] == "cd" and result.native_identity:
            if len(request.argv) != 2:
                raise _UnresolvedError
            if result.status is True and request.fail_stop:
                frame.cwd = self.path(request.argv[1], frame)
            elif result.status is not False:
                # Unknown success may retain the old directory or change it.
                # Pure later computations can continue with unknown output;
                # relative reads/writes cannot claim either destination.
                frame.cwd = None
                result = replace(result, status=None)
        if result.script_path is not None:
            text = self.read_source(result.script_path)
            if text is None:
                raise _UnresolvedError
            nested = self.program(result.script_path, text)
            opened = self.begin_redirects(request, frame)
            zero = request.argv[0] if request.interpreter is not None else result.script_path
            nested_frame = _Frame(
                request.environment.copy(),
                request.argv[1:],
                zero,
                frame.cwd,
                uncertain=frame.uncertain,
                exported=set(request.environment),
            )
            branches = self.run(nested, self.parse(nested), [nested_frame])
            self.apply_traps(branches)
            self.end_redirects(opened)
            outputs = {branch.output for branch in branches}
            result = replace(
                result,
                stdout=outputs.pop() if len(outputs) == 1 else None,
                status=branches[0].status if len(branches) == 1 else None,
            )
            completion = replace(
                request, stdout=result.stdout, child_inspected=True, child_status=result.status
            )
            result = self.native_effect(completion) if self.native_effect is not None else result
            if (
                result is None
                or result.script_path is not None
                or any(
                    not self.disjoint(path) or self.conflicts_with_redirect(path)
                    for path in result.writes
                )
            ):
                raise _UnresolvedError
        native_name = request.argv[0].rpartition("/")[2] if request.argv[0] is not None else ""
        if (
            not request.child_inspected
            and asynchronous_reader is None
            and native_name not in {":", "true", "false", "set", "cd", "pwd", "dirname"}
        ):
            # Native I/O success is not an exact status fact when a caller
            # can continue after failure. Do not prune its explicit else/||.
            result = replace(result, status=True if request.fail_stop else None)
        frame.status = result.status
        if request.fail_stop and result.status is False:
            frame.flow = "exit"
        if not any(operator.startswith("1>") for operator, _ in request.redirects):
            frame.output = (
                frame.output + result.stdout
                if frame.output is not None and result.stdout is not None
                else None
            )

    def native_default(self, request, frame, *, child=False):
        argv = request.argv
        if child:
            return ShellEffectResult(request.stdout, request.child_status)
        name = argv[0]
        if name is None:
            return None
        basename = name.rpartition("/")[2]
        if name not in {
            basename,
            "/bin/" + basename,
            "/usr/bin/" + basename,
        } or self.path_tail not in request.environment.get("PATH", self.path_tail):
            return None
        if request.environment.get("PATH", self.path_tail) != self.path_tail:
            # A supplied prefix can replace any native command. The callback
            # must prove its exact identity before using native semantics.
            return None
        args = argv[1:]
        if basename in {":", "true", "false"} and not args:
            return ShellEffectResult("", basename != "false")
        if basename == "set" and args in {
            ("-euo", "pipefail"),
            ("-Eeuo", "pipefail"),
            ("-e",),
            ("+e",),
            ("-a",),
            ("+a",),
        }:
            return ShellEffectResult("", True)
        if basename == "pwd" and not args:
            return ShellEffectResult(None if frame.cwd is None else frame.cwd + "\n", True)
        if basename == "cd" and len(args) == 1:
            destination = self.path(args[0], frame)
            directories = {"."}
            for path in self.resolver.context.sources:
                directory = posixpath.dirname(path)
                while directory:
                    directories.add(directory)
                    directory = posixpath.dirname(directory)
            return ShellEffectResult(
                "", True if destination in directories else None, native_identity=True
            )
        if (
            basename == "dirname"
            and len(args) == 1
            and args[0] is not None
            and not args[0].startswith("-")
        ):
            return ShellEffectResult(posixpath.dirname(args[0]) or ".", True)
        if (
            basename == "mktemp"
            and len(args) in {1, 2}
            and args[-1] is not None
            and (len(args) == 1 or args[0] == "-d")
        ):
            from skillspector.shell_sources import SAFE_PATH_MARKER

            template = args[-1]
            if re.fullmatch(r"/tmp/[-A-Za-z0-9_.]+X{3,}[-A-Za-z0-9_.]*", template) is None:
                return None
            self.temporary += 1
            path = re.sub(r"X{3,}", SAFE_PATH_MARKER + str(self.temporary), template)
            return ShellEffectResult(path + "\n", True, (path,))
        if basename == "echo":
            known = all(value is not None for value in args) and not any(
                value.startswith("-") or "\\" in value for value in args if value is not None
            )
            return ShellEffectResult(" ".join(args) + "\n" if known else None, True)
        if basename == "printf":
            return ShellEffectResult(request.stdout, True)
        if basename == "file":
            return (
                ShellEffectResult(None, None) if readonly_file_arguments(args, self.tick) else None
            )
        if basename in {
            "cat",
            "head",
            "tail",
            "grep",
            "egrep",
            "fgrep",
            "cut",
            "wc",
            "stat",
            "sha256sum",
            "sha1sum",
            "md5sum",
            "cmp",
            "diff",
            "readlink",
            "ls",
            "sleep",
            "wait",
            "id",
            "date",
        }:
            return ShellEffectResult(
                request.stdin if basename == "cat" and not args else None, None
            )
        if basename in {
            "rm",
            "unlink",
            "rmdir",
            "cp",
            "mv",
            "install",
            "ln",
            "tee",
            "truncate",
            "touch",
            "mkdir",
            "mkfifo",
            "chmod",
            "chown",
            "chgrp",
        }:
            if any(value is None for value in args):
                return None
            if any(
                value.startswith("-") and not re.fullmatch(r"--?[A-Za-z]+|--", value)
                for value in args
            ):
                return None
            destinations = tuple(
                self.path(value, frame) for value in args if not value.startswith("-")
            )
            if any(not self.disjoint(path) for path in destinations):
                return None
            return ShellEffectResult(None, None, destinations)
        return None


def _audit_prefix(
    resolver: ShellSourceResolver,
    target_path: str | None,
    use_offset: int,
    *,
    read_source: Callable[[str], str | None] | None = None,
    native_effect: NativeEffect | None = None,
    preserve_effect: NativeEffect | None = None,
    preserve_state: Callable[[], object] | None = None,
) -> bool:
    from skillspector.shell_sources import (
        MAX_SHELL_SOURCE_CHARS,
        MAX_SHELL_SOURCE_FILES,
        SAFE_PATH_MARKER,
        _canonical_path,
    )

    if (
        target_path is not None
        and _canonical_path(target_path) != target_path
        or not 0 <= use_offset <= len(resolver.content)
    ):
        return False
    if preserve_effect is not None and (
        target_path not in resolver.context.sources
        or read_source is None
        or native_effect is not None
    ):
        return False
    if preserve_state is not None and preserve_effect is None:
        return False
    if (
        len(resolver.context.sources) > MAX_SHELL_SOURCE_FILES
        or sum(map(len, resolver.context.sources.values())) > MAX_SHELL_SOURCE_CHARS
        or SHELL_PID_MARKER in resolver.context.path
        or any(SHELL_PID_MARKER in path for path in resolver.context.sources)
        or any(
            SAFE_PATH_MARKER in source or SHELL_PID_MARKER in source
            for source in resolver.context.sources.values()
        )
        or any(SHELL_PID_MARKER in value.text for value in resolver.inherited_values.values())
        or any(
            is_managed_shell_variable(name) and name != "PATH" for name in resolver.inherited_values
        )
    ):
        return False
    audit = _EffectAudit(
        resolver,
        target_path,
        read_source,
        preserve_effect if preserve_effect is not None else native_effect,
        preserve_only=preserve_effect is not None,
        preserve_state=preserve_state,
    )
    try:
        program = audit.program(resolver.context.path, resolver.content)
        # A consumer boundary must coincide with a complete preceding program.
        tokens = tuple(token for token in program.tokens if token.start < use_offset)
        if tokens and tokens[-1].end > use_offset:
            token = tokens[-1]
            prefix = resolver.content[token.start : use_offset]
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*="?\$\([ \t\r\n]*', prefix) is None:
                return False
            # The read is the first executable piece of this assignment.
            # Other substitution/argument prefixes need an ordered word proof.
            tokens = tokens[:-1]
        values = {name: value.text for name, value in resolver.inherited_values.items()}
        values.setdefault("PATH", audit.path_tail)
        frame = _Frame(
            values,
            (),
            resolver.context.path if resolver.source_identity else None,
            exported=set(values),
        )
        audit.run(program, audit.parse(program, tokens), [frame])
        return True
    except (KeyError, _UnresolvedError):
        return False


def source_effects_preserve(
    resolver: ShellSourceResolver,
    target_path: str,
    use_offset: int,
    *,
    read_source: Callable[[str], str | None] | None = None,
    native_effect: NativeEffect | None = None,
    preserve_effect: NativeEffect | None = None,
    preserve_state: Callable[[], object] | None = None,
) -> bool:
    """Prove prior reached code preserves a relative cached source, or refuse.

    The callback owns native identity and an optional shared artifact inventory.
    For ``uncertain`` commands it must invalidate possible output destinations,
    never publish bytes from just one branch. ``script_path`` requires a proved
    executable identity; the returned script is then inspected in a child scope.

    ``preserve_effect`` instead opts into a target-only may-effect proof for an
    uploaded source. It requires an authoritative ``read_source`` and cannot be
    combined with ``native_effect``. The callback must accumulate all possible
    writes and protect current executable/source dependencies across branches;
    a successful result authorizes original target bytes only, never generated
    artifact facts. Unknown data and joined bindings cannot supply code or paths.
    Unknown array loops additionally require ``preserve_state``: an immutable
    snapshot of every callback/source identity and write fact, whose exact
    equality proves that another iteration cannot discover different effects.
    """
    return _audit_prefix(
        resolver,
        target_path,
        use_offset,
        read_source=read_source,
        native_effect=native_effect,
        preserve_effect=preserve_effect,
        preserve_state=preserve_state,
    )


def source_effects_collect(
    resolver: ShellSourceResolver,
    use_offset: int,
    *,
    read_source: Callable[[str], str | None],
    native_effect: NativeEffect,
) -> bool:
    """Collect an ordered prefix into a caller-owned artifact inventory.

    Only a true result permits using the callback's generated-file facts.
    Unknown effects and destinations refuse the entire collection.
    """
    if read_source is None or native_effect is None:
        return False
    return _audit_prefix(
        resolver, None, use_offset, read_source=read_source, native_effect=native_effect
    )
