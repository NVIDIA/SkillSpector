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
