# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Bash comment and quoted-operand bounds for the diagnostic reader."""

from __future__ import annotations

import re
import time
from array import array
from bisect import bisect_left
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import tree_sitter_bash
from tree_sitter import Language, Node, Parser, Tree

from skillspector.shell_helper_argv import is_managed_shell_variable
from skillspector.shell_sources import bundled_script_sibling

_LANGUAGE = Language(tree_sitter_bash.language())
_SHEBANG = re.compile(r"^#![ \t]*(?:/[^ \t\r\n]*/|/)(?:env[ \t]+)?(?:bash|sh)(?:[ \t\r\n]|$)")
_CANDIDATES = re.compile(r"""[rRdDeE$'"`\\]""")


@dataclass(frozen=True, slots=True)
class BashSyntax:
    comments: tuple[tuple[int, int], ...]
    quote_limits: dict[int, int]


def _tree(raw: bytes, deadline: float) -> Tree | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    # The pinned binding's progress callback is unsafe on supported CPython
    # versions. Use the native timeout and a fresh parser for each attempt.
    return Parser(_LANGUAGE, timeout_micros=max(1, int(remaining * 1e6))).parse(raw)


# A Markdown example can start by supplying an angle-bracket metavar as
# its command or assignment value. Only the syntax check gets a stand-in.
# The diagnostic reader still sees every original byte, command and operand.
_EXAMPLE_METAVAR = re.compile(
    rb"^(?:(?:export[ \t]+)?[A-Za-z_][A-Za-z0-9_]*=)?"
    rb"(<[A-Za-z_][A-Za-z0-9_ .-]{0,63}>)(?=[\s/]|$)"
)


def complete_shell_fences(
    content: str, ranges: list[tuple[int, int]], check_runtime: Callable[[], None]
) -> tuple[tuple[int, int], ...]:
    """Grant a local EOF only to closed, syntax-checked shell examples."""
    proven = []
    deadline = time.monotonic() + 2.0
    remaining = 1_000_000
    for start, end in ranges:
        check_runtime()
        remaining -= end - start
        if remaining < 0 or time.monotonic() >= deadline:
            break
        try:
            raw = content[start:end].encode()
            if b"\0" in raw:
                continue
            tree = _tree(raw, deadline)
            if (
                tree is not None
                and tree.root_node.has_error
                and (marker := _EXAMPLE_METAVAR.match(raw))
            ):
                left, right = marker.span(1)
                # Leave closing markers, quoted here-doc delimiters and every
                # other operator intact; invalid syntax elsewhere remains invalid.
                tree = _tree(raw[:left] + b"_" * (right - left) + raw[right:], deadline)
        except (UnicodeError, ValueError, TypeError, OverflowError):
            continue
        if (
            tree is not None
            and tree.root_node.type == "program"
            and tree.root_node.end_byte == len(raw)
            and not tree.root_node.has_error
        ):
            proven.append((start, end))
    check_runtime()
    return tuple(proven)


def _arithmetic_base_markers(
    root: Node, source: bytes, check_runtime: Callable[[], None]
) -> list[tuple[int, int]]:
    """The grammar omits Bash's base# prefix before an expanded arithmetic value."""
    markers = []
    pending = [root]
    remaining = 100_000
    while pending:
        check_runtime()
        node = pending.pop()
        remaining -= 1
        if remaining < node.child_count + len(pending) or node.is_missing:
            return []
        if node.is_error:
            start, end = node.start_byte, node.end_byte
            if (
                end - start > 3
                or re.fullmatch(rb"(?:[2-9]|[1-5][0-9]|6[0-4])#", source[start:end]) is None
                or source[end : end + 1] != b"$"
            ):
                return []
            parent = node.parent
            while parent is not None and parent.type in {
                "binary_expression",
                "parenthesized_expression",
            }:
                parent = parent.parent
            if parent is None or parent.type != "arithmetic_expansion":
                return []
            markers.append((start, end))
        else:
            pending.extend(child for child in node.children if child.has_error or child.is_missing)
    return markers


def bash_syntax(content: str, check_runtime: Callable[[], None]) -> BashSyntax | None:
    """Prove source locations, without deciding what a command does at runtime."""
    check_runtime()
    if len(content) > 1_000_000 or "\x00" in content:
        return None
    if not _SHEBANG.match(content):
        return None
    deadline = time.monotonic() + 2.0
    try:
        raw = content.encode()
        tree = _tree(raw, min(deadline, time.monotonic() + 1.0))
        if tree is not None and tree.root_node.has_error:
            markers = _arithmetic_base_markers(tree.root_node, raw, check_runtime)
            if not markers or time.monotonic() >= deadline:
                return None
            # This spelling is opaque to the shell-ownership view. Keep the
            # original expansion, command text, offsets and findings unchanged.
            grammar_view = bytearray(raw)
            for start, end in markers:
                grammar_view[start:end] = b" " * (end - start)
            tree = _tree(bytes(grammar_view), deadline)
    except (UnicodeError, ValueError, TypeError, OverflowError):
        return None
    check_runtime()
    if tree is None or time.monotonic() >= deadline:
        return None
    root = tree.root_node
    if root.type != "program" or root.has_error or root.end_byte != len(raw):
        return None

    # UTF-8 byte offsets differ from the scanner's Python character offsets.
    boundaries = array("I")
    if not content.isascii():
        count = 0
        for index, character in enumerate(content):
            if not index % 4096:
                check_runtime()
                if time.monotonic() >= deadline:
                    return None
            boundaries.append(count)
            ordinal = ord(character)
            count += 1 + (ordinal > 0x7F) + (ordinal > 0x7FF) + (ordinal > 0xFFFF)
        boundaries.append(count)

    def offset(pos: int) -> int:
        return bisect_left(boundaries, pos) if boundaries else pos

    comments: list[tuple[int, int]] = []
    limits: dict[int, int] = {}
    pending = [(root, 0)]
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if not visited % 128:
            check_runtime()
            if time.monotonic() >= deadline:
                return None
        if depth > 128 or visited + len(pending) + node.child_count > 100_000:
            return None
        if node.type == "comment":
            comments.append((offset(node.start_byte), offset(node.end_byte)))
        elif node.type == "string" and node.parent is not None:
            # Only direct, final argv values can end at this quote. In particular
            # this is never a command name, adjacent shell word or joined value.
            parent = node.parent
            if parent.type == "command" and parent.end_byte == node.end_byte:
                end = offset(node.end_byte) - 1
                if content[end : end + 1] == '"':
                    for child in node.named_children:
                        if child.type in {"string_content", "simple_expansion", "expansion"}:
                            start, stop = offset(child.start_byte), offset(child.end_byte)
                            # Parameter operators may contain an executable substitution.
                            # Their inner bytes retain their separate lexical owner.
                            if child.type == "expansion":
                                stop = min(start + 1, stop)
                            for match in _CANDIDATES.finditer(content, start, stop):
                                limits[match.start()] = end
                                if len(limits) > 100_000:
                                    return None
                                if not len(limits) % 1024:
                                    check_runtime()
                                    if time.monotonic() >= deadline:
                                        return None
        pending.extend((child, depth + 1) for child in reversed(node.named_children))
    return BashSyntax(tuple(comments), limits)


_PATH_OPERAND = re.compile(rb"/[a-zA-Z0-9_./-]+")
# A plain external command cannot change shell bindings in its parent.
# Ambiguous shell syntax or a process-mutating builtin in another example
# leaves the original diagnostic intact.
_BINDING_BUILTINS = frozenset(
    "alias bg bind break builtin caller cd command compgen complete compopt continue coproc declare "
    "dirs disown enable eval exec exit export fc fg getopts hash help history jobs kill let local "
    "logout mapfile popd pushd read readarray readonly return set shift shopt source suspend test "
    "times trap type typeset ulimit umask unalias unset wait printf case do done elif else esac fi for "
    "function if in select then time until while".split()
)
_SCOPE_HEAD = re.compile(r"[A-Za-z_][A-Za-z_0-9.-]{0,127}")
_SCOPE_ARG = re.compile(
    r"""(?:[a-zA-Z0-9_./:+=,%*@-]+|<[a-zA-Z_][a-zA-Z_0-9]*>|'[^'\r\n]*'|"[^"\\$`\r\n]*"|"\$[A-Za-z_][A-Za-z_0-9]*")(?=[ \t]|$)"""
)


def _ordinary_binding(name: str) -> bool:
    # Multi-character environment controls can affect shell startup or the
    # dynamic loader used by external utilities (LD_*, DYLD_*, LC_*, etc.).
    return (len(name) == 1 or name.islower()) and not is_managed_shell_variable(name)


def plain_shell_neighbor(content: str, check_runtime: Callable[[], None]) -> bool:
    """Accept simple external argv and literal scalar assignments in a neighboring example."""
    if len(content) > 16_384 or not content.isascii():
        return False
    for raw_line in content.split("\n"):
        check_runtime()
        line = raw_line.strip(" \t")
        if not line or line.startswith("#"):
            continue
        head = _SCOPE_HEAD.match(line)
        if head is None or head.group() in _BINDING_BUILTINS:
            return False
        end = head.end()
        if line[end : end + 1] == "=":
            if (
                not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", head.group())
                or not _ordinary_binding(head.group())
                or _PATH_OPERAND.fullmatch(line[end + 1 :].encode()) is None
            ):
                return False
            continue
        while end < len(line):
            if line[end] not in " \t":
                return False
            while end < len(line) and line[end] in " \t":
                end += 1
            if line[end : end + 1] == "#":
                break
            arg = _SCOPE_ARG.match(line, end)
            if arg is None:
                return False
            end = arg.end()
    return True


_NATURAL = re.compile(rb"(?:0|[1-9][0-9]{0,8}|0[xX][0-9A-Fa-f]{1,8})")
# Space padding can split an unquoted result into words. A literal numeric
# prefix, the radix flag and up to 32 zero-padded digits produce one operand.
_INTEGER_FORMAT = re.compile(
    rb"(?P<prefix>0[xX]?)?%(?P<radix>#)?(?:0(?:[1-9]|[12][0-9]|3[0-2]))?(?P<conversion>[diuxXo])"
)


def may_have_formatted_loop(content: str) -> bool:
    """Cheap trigger only; the complete native shell tree proves the role."""
    return len(content) <= 1_000_000 and all(
        marker in content for marker in ("for", "$(", "printf", "%")
    )


def _integer_format(node: Node) -> bool:
    raw = node.text
    if node.type in {"raw_string", "string"}:
        quote = b"'" if node.type == "raw_string" else b'"'
        if not (raw.startswith(quote) and raw.endswith(quote)):
            return False
        raw = raw[1:-1]
    elif node.type != "word":
        return False
    match = _INTEGER_FORMAT.fullmatch(raw)
    if match is None:
        return False
    prefix, radix, conversion = match.groups()
    if prefix and radix:
        return False
    if prefix in (b"0x", b"0X"):
        return conversion in (b"x", b"X")
    if prefix == b"0":
        return conversion == b"o"
    return not radix or conversion in (b"x", b"X", b"o")


def _unquote_expansion(node: Node) -> Node:
    if node.type == "string" and node.named_child_count == 1:
        inner = node.named_children[0]
        if node.text == b'"' + inner.text + b'"':
            return inner
    return node


def _bounded_integer_operand(node: Node, bounds: dict[bytes, tuple[int, int]]) -> bool:
    operand = _unquote_expansion(node)
    if operand.type in {"simple_expansion", "expansion"} and operand.named_child_count == 1:
        name = operand.named_children[0]
        return (
            name.type == "variable_name"
            and name.text in bounds
            and operand.text in (b"$" + name.text, b"${" + name.text + b"}")
        )
    return (
        operand.type in {"arithmetic_expansion", "number"}
        and _numeric_range(operand, bounds) is not None
    )


def _numeric_range(
    node: Node, bounds: dict[bytes, tuple[int, int]], depth: int = 0
) -> tuple[int, int] | None:
    """Bound pure integer arithmetic without Bash's recursive evaluation of strings."""
    if depth > 16:
        return None
    if node.type == "number" and _NATURAL.fullmatch(node.text):
        number = int(node.text, 0)
        return (number, number) if number <= 0x7FFFFFFF else None
    if node.type == "variable_name":
        return bounds.get(node.text)
    if node.type == "arithmetic_expansion" and node.named_child_count == 1:
        return _numeric_range(node.named_children[0], bounds, depth + 1)
    if node.type != "binary_expression" or node.child_count != 3:
        return None
    left, operator, right = node.children
    a, b = _numeric_range(left, bounds, depth + 1), _numeric_range(right, bounds, depth + 1)
    if a is None or b is None:
        return None
    if operator.type == "+":
        lo, hi = a[0] + b[0], a[1] + b[1]
    elif operator.type == "-":
        lo, hi = a[0] - b[1], a[1] - b[0]
    elif operator.type == "*":
        lo, hi = a[0] * b[0], a[1] * b[1]
    else:
        return None
    return (lo, hi) if 0 <= lo <= hi <= 0x7FFFFFFF else None


def _substitution_command(node: Node) -> Node | None:
    if node.type != "command_substitution" or node.child_count not in (3, 4):
        return None
    opener, command, *terminator, closer = node.children
    if opener.type != "$(" or closer.type != ")" or (terminator and terminator[0].type != ";"):
        return None
    return command if command.type == "command" else None


def _loop_domain(node: Node) -> tuple[int, int] | None:
    values = node.children_by_field_name("value")
    if len(values) == 1 and values[0].type == "command_substitution":
        command = _substitution_command(values[0])
        if command is None or command.named_child_count != 3:
            return None
        head, first, last = command.named_children
        if (
            head.type == "command_name"
            and head.text == b"seq"
            and all(
                part.type == "number" and re.fullmatch(rb"0|[1-9][0-9]{0,6}", part.text)
                for part in (first, last)
            )
        ):
            low, high = int(first.text), int(last.text)
            if 0 <= high - low < 128:
                return low, high
    elif 0 < len(values) <= 128 and all(
        node.type == "number" and _NATURAL.fullmatch(node.text) for node in values
    ):
        # Loop words are strings. Only spellings that Bash itself reads as
        # plain integers may feed arithmetic expansion.
        numbers = [int(node.text, 0) for node in values]
        if max(numbers) <= 0x7FFFFFFF:
            return min(numbers), max(numbers)
    return None


def finite_numeric_operands(
    content: str, check_runtime: Callable[[], None]
) -> list[tuple[int, int]]:
    """Prove formatted numeric argv in a complete, bounded sequence of display/read loops."""
    if (
        len(content) > 16_384
        or not may_have_formatted_loop(content)
        or not content.isascii()
        or (content.startswith("#!") and _SHEBANG.match(content) is None)
    ):
        return []
    if "\\" + "\n" in content or "\r" in content or "\0" in content:
        return []
    check_runtime()
    try:
        tree = _tree(content.encode(), time.monotonic() + 0.25)
    except (ValueError, TypeError, OverflowError):
        return []
    if tree is None or tree.root_node.has_error or tree.root_node.end_byte != len(content):
        return []
    operands = []
    definitions: set[bytes] = set()
    paths: set[bytes] = set()

    def bind(name: Node | None) -> bool:
        if (
            name is None
            or name.type != "variable_name"
            or name.text in definitions
            or not _ordinary_binding(name.text.decode())
        ):
            return False
        definitions.add(name.text)
        return True

    def walk(node: Node, bounds: dict[bytes, tuple[int, int]], depth: int) -> bool:
        check_runtime()
        if depth > 16:
            return False
        kind = node.type
        if kind in {"program", "do_group"}:
            # Background lists do not establish a dominating assignment.
            return all(
                walk(child, bounds, depth + 1)
                if child.is_named
                else child.type in {"do", "done", ";"}
                for child in node.children
            )
        if kind == "comment":
            return True
        if kind == "for_statement":
            variable, body = node.child_by_field_name("variable"), node.child_by_field_name("body")
            domain = _loop_domain(node)
            return (
                body is not None
                and domain is not None
                and bind(variable)
                and walk(body, bounds | {variable.text: domain}, depth + 1)
            )
        if kind == "variable_assignment":
            name, value = node.child_by_field_name("name"), node.child_by_field_name("value")
            if not bind(name) or value is None or node.text != name.text + b"=" + value.text:
                return False
            if value.type == "word" and _PATH_OPERAND.fullmatch(value.text):
                if depth != 1:
                    return False
                paths.add(name.text)
                return True
            domain = _numeric_range(value, bounds)
            if domain is None:
                return False
            bounds[name.text] = domain
            return True
        if kind == "pipeline":
            # An outer display may translate line breaks from the inner loop.
            return (
                node.child_count == 3
                and node.children[1].type == "|"
                and node.children[0].type == "for_statement"
                and re.fullmatch(
                    rb"tr[ \t]+'[^'\r\n]{1,16}'[ \t]+'[^'\r\n]{1,16}'", node.children[2].text
                )
                is not None
                and walk(node.children[0], bounds.copy(), depth + 1)
            )
        if kind != "command":
            return False
        name = node.child_by_field_name("name")
        args = node.children_by_field_name("argument")
        if name is None or node.named_child_count != len(args) + 1:
            return False
        if name.text == b"echo" and len(args) == 1 and args[0].type == "string":
            for part in args[0].named_children:
                if part.type == "string_content":
                    continue
                if part.type == "simple_expansion" and part.text[1:] in bounds:
                    continue
                if (
                    part.type != "command_substitution"
                    or part.named_child_count != 1
                    or not walk(part.named_children[0], bounds.copy(), depth + 1)
                ):
                    return False
            return True
        # A fully spelled external utility consumes a fixed pathname and a
        # numeric operand. Neither can select the executable or add an option.
        if (
            name.text == b"sudo"
            and len(args) == 3
            and re.fullmatch(rb"[A-Za-z0-9_][A-Za-z0-9_.-]*", args[0].text)
        ):
            args = args[1:]
        elif not _PATH_OPERAND.fullmatch(name.text) or len(args) != 2:
            return False
        path, scalar = args
        if not (
            (path.type == "word" and _PATH_OPERAND.fullmatch(path.text))
            or (path.type == "simple_expansion" and path.text[1:] in paths)
        ):
            return False
        command = _substitution_command(_unquote_expansion(scalar))
        if command is None or command.named_child_count != 3:
            return False
        parts = command.named_children
        if (
            parts[0].type != "command_name"
            or parts[0].text != b"printf"
            or not _integer_format(parts[1])
            or not _bounded_integer_operand(parts[2], bounds)
        ):
            return False
        operands.append((scalar.start_byte, scalar.end_byte))
        return True

    return operands if walk(tree.root_node, {}, 0) else []


@dataclass(frozen=True, slots=True)
class ShellProgramCaller:
    start: int
    end: int
    program: str | None


@dataclass(frozen=True, slots=True)
class ShellProgramForwarding:
    argument: tuple[int, int]
    callers: tuple[ShellProgramCaller, ...]
    scopes: tuple[tuple[int, int], ...]
    closed: bool


def _literal_argument(node: Node) -> str | None:
    raw = node.text
    if len(raw) > 2048:
        return None
    if node.type == "raw_string" and raw[:1] == raw[-1:] == b"'":
        return raw[1:-1].decode()
    if node.type == "string" and raw[:1] == raw[-1:] == b'"':
        if (
            all(child.type == "string_content" for child in node.named_children)
            and b"\\" not in raw
        ):
            return raw[1:-1].decode()
    if node.type in {"word", "number", "command_name"} and re.fullmatch(
        rb"[a-zA-Z0-9_./:+-]+", raw
    ):
        return raw.decode()
    return None


_DURATION = re.compile(r"[0-9]{1,6}(?:\.[0-9]{1,3})?[smhd]?")


def _single_argv_slot(word: Node) -> bool:
    if _literal_argument(word) is not None:
        return True
    if word.type != "string":
        return False
    for child in word.named_children:
        if child.type in {"string_content", "command_substitution", "arithmetic_expansion"}:
            continue
        if child.type in {"expansion", "simple_expansion"} and re.fullmatch(
            rb"\$(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]|\{(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)\})", child.text
        ):
            continue
        return False
    return True


def _forwarding_parameters(command: Node) -> tuple[Node, int, int, str] | None:
    """Recognize a single positional script, optionally behind a fixed timeout."""
    if command.type != "command" or not 3 <= command.named_child_count <= 10:
        return None
    words = command.named_children
    if (
        command.child_by_field_name("name") != words[0]
        or len(command.children_by_field_name("argument")) != len(words) - 1
    ):
        return None
    duration_index, default = 0, ""
    if _literal_argument(words[0]) in {"timeout", "/bin/timeout", "/usr/bin/timeout"}:
        opts = words[1:-3]
        if not opts:
            return None
        duration = opts[-1]
        options = [_literal_argument(word) for word in opts[:-1]]
        while options:
            flag = options.pop(0)
            if flag in {"--foreground", "--preserve-status"}:
                continue
            if (
                flag in {"-s", "--signal"}
                and options
                and re.fullmatch(r"[A-Z]{1,12}|[0-9]{1,2}", options.pop(0) or "")
            ):
                continue
            return None
        if not _DURATION.fullmatch(_literal_argument(duration) or ""):
            number = re.fullmatch(
                rb'"\$([1-9])"|"\$\{([1-9])(?::-([0-9]{1,6}[smhd]?))?\}"', duration.text
            )
            if number is None:
                return None
            duration_index = int(number[1] or number[2])
            default = (number[3] or b"").decode()
        words = words[-3:]
    if (
        len(words) != 3
        or _literal_argument(words[0])
        not in {"sh", "bash", "/bin/sh", "/bin/bash", "/usr/bin/bash", "/usr/bin/sh"}
        or _literal_argument(words[1]) != "-c"
    ):
        return None
    number = re.fullmatch(rb'"\$([1-9])"|"\$\{([1-9])\}"', words[2].text)
    if number is None:
        return None
    return words[2], int(number[1] or number[2]), duration_index, default


def _forwarded_argument(
    command: Node, index: int, duration_index: int, default: str
) -> tuple[Node, str | None, bool] | None:
    args = command.children_by_field_name("argument")
    if len(args) > 32 or command.named_child_count != len(args) + 1 or index > len(args):
        return None
    if any(not _single_argv_slot(word) for word in args[: max(index, duration_index)]):
        return None
    if duration_index:
        duration = (
            _literal_argument(args[duration_index - 1]) if duration_index <= len(args) else default
        )
        if duration is None or not _DURATION.fullmatch(duration or default):
            return None
    operand = args[index - 1]
    program = _literal_argument(operand)
    if program is None:
        # A quoted scalar has one argv position. Locate the expansion that
        # actually supplies unknown command text in that position.
        if operand.type != "string":
            return None
        origin = next(
            (child for child in operand.named_children if child.type in _SHELL_EXPANSIONS), None
        )
        if origin is None:
            return None
    else:
        origin = operand
    # An unrelated expansion can execute another command or change shell state
    # (e.g. in arithmetic). A quoted scalar read still occupies exactly one slot.
    other_args = [arg for arg in args if arg != operand]
    inert = all(
        _literal_argument(arg) is not None
        or (
            _single_argv_slot(arg)
            and all(
                c.type in {"string_content", "simple_expansion", "expansion"}
                for c in arg.named_children
            )
        )
        for arg in other_args
    )
    return origin, program, inert


_SHELL_EXPANSIONS = {
    "expansion",
    "simple_expansion",
    "command_substitution",
    "arithmetic_expansion",
}
_BINDING_COMMANDS = frozenset(
    ". alias builtin complete declare enable eval exec export hash mapfile read readarray "
    "readonly shopt source trap typeset unalias unset".split()
)


def positional_shell_callers(
    content: str, start: int, end: int, check_runtime: Callable[[], None]
) -> ShellProgramForwarding | None:
    """Trace an ordinary function forwarding one quoted positional shell program."""
    if (
        len(content) > 131_072
        or "-c" not in content
        or ("(" not in content and "function" not in content)
        or "\0" in content
        or (
            content.startswith("#!")
            and not re.fullmatch(
                r"#![ \t]*(?:/bin/(?:ba)?sh|/usr/bin/(?:ba)?sh|/usr/bin/env[ \t]+(?:ba)?sh)[ \t]*",
                content.partition("\n")[0],
            )
        )
    ):
        return None
    deadline = time.monotonic() + 0.5
    try:
        raw = content.encode()
        tree = _tree(raw, deadline)
    except (UnicodeError, TypeError, ValueError, OverflowError):
        return None
    if (
        tree is None
        or tree.root_node.has_error
        or tree.root_node.descendant_count > 65_536
        or tree.root_node.end_byte != len(raw)
    ):
        return None
    begin, finish = len(content[:start].encode()), len(content[:end].encode())
    if not begin < finish:
        return None
    node = tree.root_node.descendant_for_byte_range(begin, finish)
    while node is not None and node.type != "function_definition":
        node = node.parent
    if node is None or node.parent != tree.root_node:
        return None
    owner = node
    named, body = owner.child_by_field_name("name"), owner.child_by_field_name("body")
    if (
        named is None
        or owner.child_by_field_name("redirect") is not None
        or not re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]{0,63}", named.text)
        or body is None
        or body.type != "compound_statement"
        or body.named_child_count != 1
        or any(not child.is_named and child.type not in {"{", "}", ";"} for child in body.children)
    ):
        return None
    command = body.named_children[0]
    if command.type == "redirected_statement":
        for redirect in command.children_by_field_name("redirect"):
            if redirect.type != "file_redirect" or not re.fullmatch(
                rb"(?:[12]?>[ \t]*/dev/null|[12]?>&[12])", redirect.text
            ):
                return None
        command = command.child_by_field_name("body")
    if command is None or not (command.start_byte <= begin < finish <= command.end_byte):
        return None
    params = _forwarding_parameters(command)
    if params is None:
        return None
    argument, index, duration_index, default = params
    if not begin <= argument.start_byte < argument.end_byte <= finish:
        return None
    name = named.text.decode()
    protected = {name, "bash", "sh", "timeout"}
    if name in {"sh", "bash", "timeout", "command_not_found_handle"} or name in _BINDING_BUILTINS:
        return None
    # Only the lexical calls in an entirely closed top-level script can clear
    # a declaration. In a larger script a demonstrably dynamic direct call may
    # refine its location, even when other possible callers remain unknown.
    calls: list[ShellProgramCaller] = []
    known_commands = set()
    scopes = [(owner.start_byte, owner.end_byte)]
    uncertain_scopes: list[tuple[int, int]] = []
    closed = True
    pending = [tree.root_node]

    def offset(byte: int) -> int:
        return byte if len(raw) == len(content) else len(raw[:byte].decode())

    while pending:
        check_runtime()
        n = pending.pop()
        pending.extend(reversed(n.named_children))
        if time.monotonic() >= deadline:
            return None
        if n.type == "function_definition" and n != owner:
            fn = n.child_by_field_name("name")
            if fn is None or fn.text.decode() in protected:
                return None
        if n.type in {"word", "raw_string", "string", "concatenation"} and n.text in {
            named.text,
            b"'" + named.text + b"'",
            b'"' + named.text + b'"',
        }:
            parent = n.parent
            direct = parent is not None and parent.type == "command_name"
            if not direct and n != named:
                closed = False
        if n.type in {"command", "variable_assignment"} and n != command:
            target = n.child_by_field_name("name")
            if target is None:
                return None
            spelling = _literal_argument(target)
            if n.type == "variable_assignment":
                var = target.text.decode()
                hazard = is_managed_shell_variable(var) or var.startswith(("LD_", "DYLD_"))
            else:
                hazard = spelling is None or spelling in _BINDING_COMMANDS
                if spelling == "command":
                    first = n.child_by_field_name("argument")
                    hazard = first is None or _literal_argument(first) not in {"-v", "-V"}
            if hazard:
                closed = False
                scope = n.parent
                while scope is not None and scope.type not in {
                    "command_substitution",
                    "subshell",
                    "program",
                }:
                    scope = scope.parent
                if scope is not None:
                    if len(uncertain_scopes) >= 128:
                        return None
                    uncertain_scopes.append((n.start_byte, scope.end_byte))
            if n.type == "command" and spelling == name:
                if target.text != named.text or n.start_byte < owner.end_byte or len(calls) >= 128:
                    return None
                if any(a < n.start_byte < b for a, b in uncertain_scopes):
                    # A later binding change cannot invalidate an earlier,
                    # independently established unresolved call, but it does
                    # prevent a closed-world claim on subsequent calls.
                    closed = False
                    continue
                caller = _forwarded_argument(n, index, duration_index, default)
                if caller is None:
                    return None
                span, literal, fixed_other_args = caller
                closed = closed and fixed_other_args
                calls.append(
                    ShellProgramCaller(offset(span.start_byte), offset(span.end_byte), literal)
                )
                known_commands.add(n.id)
                scopes.append((n.start_byte, n.end_byte))
    if not calls:
        return None

    def accounted_statement(statement: Node) -> bool:
        if statement == owner or statement.type == "comment" or statement.id in known_commands:
            return True
        if statement.type != "variable_assignment":
            return False
        binding = statement.child_by_field_name("name")
        value = statement.child_by_field_name("value")
        if binding is None or value is None or not _ordinary_binding(binding.text.decode()):
            return False
        captured = _substitution_command(_unquote_expansion(value))
        return captured is not None and captured.id in known_commands

    closed = closed and all(accounted_statement(child) for child in tree.root_node.named_children)
    return ShellProgramForwarding(
        (offset(argument.start_byte), offset(argument.end_byte)),
        tuple(calls),
        tuple((offset(low), offset(high)) for low, high in scopes),
        closed,
    )


@dataclass(frozen=True, slots=True)
class SourcedEval:
    caller_span: tuple[int, int]
    definition_path: str
    eval_span: tuple[int, int]
    script_relative: bool


def _forwarded_eval(
    body: Node, supplied: list[Node], check_runtime: Callable[[], None]
) -> tuple[int, Node] | None:
    """Prove a straight-line positional argument reaches eval in the selected arm."""
    variables: dict[bytes, tuple[int, bytes | None]] = {}
    positional = rb'(?:"\$([1-9])"|"\$\{([1-9])(?::-([A-Za-z0-9_.-]+))?\}")'
    statements = [node for node in body.named_children if node.type != "comment"]
    if not 1 <= len(statements) <= 16 or body.type != "compound_statement":
        return None
    for node in statements[:-1]:
        check_runtime()
        if node.type != "declaration_command" or node.children[0].type != "local":
            return None
        for binding in node.named_children:
            name = binding.child_by_field_name("name")
            value = binding.child_by_field_name("value")
            if name is None or value is None or binding.type != "variable_assignment":
                return None
            match = re.fullmatch(positional, value.text)
            if not match or name.text in variables or not _ordinary_binding(name.text.decode()):
                return None
            variables[name.text] = (int(match[1] or match[2]), match[3])
    node = statements[-1]
    if node.type == "case_statement":
        selector = node.child_by_field_name("value")
        var = re.fullmatch(rb'"\$([A-Za-z_]\w*)"', selector.text) if selector is not None else None
        if not var or var[1] not in variables:
            return None
        position, default = variables[var[1]]
        value = _literal_argument(supplied[position - 1]) if position <= len(supplied) else ""
        if value is None:
            return None
        selected_value = value.encode() or default or b""
        for case in node.named_children:
            check_runtime()
            if case.type != "case_item":
                continue
            labels = case.children_by_field_name("value")
            if not labels or any(
                not re.fullmatch(rb"[A-Za-z0-9_.-]+|\*", label.text) for label in labels
            ):
                return None
            if any(label.text in {b"*", selected_value} for label in labels):
                actions = [item for item in case.named_children if item not in labels]
                if len(actions) != 1 or not case.text.rstrip().endswith(b";;"):
                    return None
                node = actions[0]
                break
    if node.type == "subshell" and node.named_child_count == 1:
        node = node.named_children[0]
    if (
        node.type != "command"
        or node.named_child_count != 2
        or node.named_children[0].text != b"eval"
    ):
        return None
    argument = node.child_by_field_name("argument")
    if argument is None:
        return None
    match = re.fullmatch(positional, argument.text)
    if match is not None:
        return int(match[1] or match[2]), argument
    var = re.fullmatch(rb'"\$([A-Za-z_]\w*)"', argument.text)
    return (variables[var[1]][0], argument) if var and var[1] in variables else None


def _script_directory(value: Node) -> bool:
    """Match a directory derived from the original script's positional identity."""
    value = _unquote_expansion(value)
    if value.type != "command_substitution" or value.named_child_count != 1:
        return False
    sequence = value.named_children[0]
    if sequence.type != "list" or [n.type for n in sequence.children] != [
        "command",
        "&&",
        "command",
    ]:
        return False
    cd, pwd = sequence.named_children
    cd_words, pwd_words = cd.named_children, pwd.named_children
    if [word.text for word in pwd_words] not in ([b"pwd"], [b"pwd", b"-P"]):
        return False
    if [word.text for word in cd_words[:-1]] not in ([b"cd"], [b"cd", b"--"]):
        return False
    directory = cd_words[-1]
    if directory.type != "string" or directory.named_child_count != 1:
        return False
    command = _substitution_command(directory.named_children[0])
    if command is None:
        return False
    words = command.named_children
    return [word.text for word in words[:-1]] in ([b"dirname"], [b"dirname", b"--"]) and words[
        -1
    ].text == b'"${BASH_SOURCE[0]}"'


def sourced_eval_argument(
    caller: str, path: str, sources: Mapping[str, str], check_runtime: Callable[[], None]
) -> SourcedEval | None:
    """Find an original script-relative library function consuming a runtime shell program.

    Only declarative libraries and a straight-line caller prefix establish a binding.
    A working-directory-relative include establishes only a possible callable identity.
    This adds an uncertainty witness without granting any completion or removing findings.
    """
    if (
        len(caller) > 131_072
        or sources.get(path) != caller
        or not re.search(r"(?m)(?:^|;)[ \t]*(?:source|\.)[ \t]", caller)
    ):
        return None
    deadline = time.monotonic() + 0.3
    remaining_bytes = 256_000

    def parse(text: str) -> Node | None:
        nonlocal remaining_bytes
        check_runtime()
        if len(text) > remaining_bytes or (
            text.startswith("#!")
            and not re.fullmatch(
                r"#![ \t]*(?:/bin/(?:ba)?sh|/usr/bin/(?:ba)?sh|/usr/bin/env[ \t]+(?:ba)?sh)[ \t]*",
                text.partition("\n")[0],
            )
        ):
            return None
        raw = text.encode()
        remaining_bytes -= len(raw)
        if remaining_bytes < 0 or time.monotonic() >= deadline or b"\0" in raw:
            return None
        try:
            tree = _tree(raw, deadline)
        except (ValueError, OverflowError):
            return None
        if tree is None or tree.root_node.has_error or tree.root_node.end_byte != len(raw):
            return None
        if tree.root_node.descendant_count > 65_536:
            return None
        return tree.root_node

    root = parse(caller)
    if root is None or root.child_count > 512:
        return None
    directories: set[bytes] = set()
    assigned: set[bytes] = set()
    definitions: dict[bytes, tuple[str, Node]] = {}
    all_sources_script_relative = True
    loaded: set[str] = set()
    for node in root.children:
        check_runtime()
        if time.monotonic() >= deadline:
            return None
        if node.type in {"comment", ";"}:
            continue
        if node.type == "variable_assignment":
            name = node.child_by_field_name("name")
            value = node.child_by_field_name("value")
            if (
                name is None
                or value is None
                or not re.fullmatch(rb"[A-Za-z_]\w*", name.text)
                or is_managed_shell_variable(name.text.decode())
                or name.text.startswith((b"LD_", b"DYLD_"))
            ):
                return None
            if name.text in directories or name.text in assigned:
                return None
            assigned.add(name.text)
            if _script_directory(value):
                directories.add(name.text)
            elif _literal_argument(value) is None:
                return None
            continue
        if node.type != "command" or node.named_children[0] != node.child_by_field_name("name"):
            return None
        name = node.named_children[0].text
        args = node.children_by_field_name("argument")
        # Backgrounded imports, pipeline segments and conditional substitutions
        # cannot introduce a function into the current shell unconditionally.
        following = node.next_sibling
        if following is not None and following.type == "&":
            return None
        if name == b":" and not args and node.named_child_count == 1:
            continue
        if name in {b"source", b"."}:
            if len(loaded) >= 12 or len(args) != 1:
                return None
            ref = re.fullmatch(
                rb'"\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)\})/([\w./-]+)"', args[0].text
            )
            script_relative = bool(ref and (ref[1] or ref[2]) in directories)
            literal = _literal_argument(args[0])
            if script_relative and ref:
                suffix = ref[3].decode()
            elif literal and literal.startswith("./"):
                suffix = literal[2:]
            else:
                return None
            source_path = bundled_script_sibling(path, suffix)
            if source_path is None:
                return None
            if not script_relative and suffix != source_path and suffix in sources:
                return None
            library = sources.get(source_path)
            if source_path in loaded or library is None:
                return None
            imported = parse(library)
            if imported is None or imported.child_count > 512:
                return None
            loaded.add(source_path)
            # A working-directory-relative source may replace any binding in the
            # current shell, including an earlier helper or the next include root.
            all_sources_script_relative = all_sources_script_relative and script_relative
            for item in imported.children:
                check_runtime()
                if item.type in {"comment", ";"}:
                    continue
                symbol = item.child_by_field_name("name")
                body = item.child_by_field_name("body")
                if item.type != "function_definition" or symbol is None or body is None:
                    return None
                if not re.fullmatch(rb"[A-Za-z_]\w*", symbol.text) or symbol.text in {
                    b"source",
                    b"eval",
                    b"cd",
                    b"dirname",
                    b"pwd",
                    b"local",
                }:
                    return None
                definitions[symbol.text] = (source_path, body)
            continue
        if (
            name not in definitions
            or len(args) > 16
            or not all(_single_argv_slot(arg) for arg in args)
        ):
            return None
        source_path, body = definitions[name]
        reached = _forwarded_eval(body, args, check_runtime)
        if reached is not None:
            slot, sink = reached
            if slot <= len(args):
                argument = args[slot - 1]
                var = re.fullmatch(rb'"\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)\})"', argument.text)
                if var and (var[1] or var[2]) not in assigned:
                    raw = caller.encode()
                    source_raw = sources[source_path].encode()
                    witness = argument if all_sources_script_relative else node.named_children[0]
                    return SourcedEval(
                        (
                            len(raw[: witness.start_byte].decode()),
                            len(raw[: witness.end_byte].decode()),
                        ),
                        source_path,
                        (
                            len(source_raw[: sink.start_byte].decode()),
                            len(source_raw[: sink.end_byte].decode()),
                        ),
                        all_sources_script_relative,
                    )
        # An intervening function may change the current shell's bindings.
        return None
    return None
