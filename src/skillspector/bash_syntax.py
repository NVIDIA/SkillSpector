# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Bash comment and quoted-operand bounds for the diagnostic reader."""

from __future__ import annotations

import re
import time
from array import array
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass

import tree_sitter_bash
from tree_sitter import Language, Node, Parser, Tree

from skillspector.shell_helper_argv import is_managed_shell_variable

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


def _loop_domain(node: Node) -> tuple[int, int] | None:
    values = node.children_by_field_name("value")
    if len(values) == 1 and values[0].type == "command_substitution":
        match = re.fullmatch(
            rb"\$\(seq[ \t]+(0|[1-9][0-9]{0,6})[ \t]+(0|[1-9][0-9]{0,6})\)", values[0].text
        )
        if match:
            low, high = int(match[1]), int(match[2])
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
        "$(printf" not in content
        or "for " not in content
        or len(content) > 16_384
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
        if scalar.type != "command_substitution" or scalar.named_child_count != 1:
            return False
        command = scalar.named_children[0]
        parts = command.named_children
        if (
            command.type != "command"
            or scalar.text != b"$(" + command.text + b")"
            or len(parts) != 3
            or parts[0].text != b"printf"
            or parts[1].text != b"0x%x"
            or parts[2].type != "arithmetic_expansion"
            or _numeric_range(parts[2], bounds) is None
        ):
            return False
        operands.append((scalar.start_byte, scalar.end_byte))
        return True

    return operands if walk(tree.root_node, {}, 0) else []
