# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded JavaScript syntax ownership, without executing or rewriting source.

The concrete syntax tree is not a binding, type, or ECMAScript early-error
proof. Consumers must establish their own narrower ownership contracts.
"""

from __future__ import annotations

import re
import time
from array import array
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

import tree_sitter_javascript
from tree_sitter import Language, Node, Parser, Tree

MAX_JAVASCRIPT_SOURCE_CHARS = 1_000_000
_MAX_NODES = 100_000
_MAX_DEPTH = 128
_MAX_SECONDS = 2.0
_LANGUAGE = Language(tree_sitter_javascript.language())
NODE_SHEBANG = re.compile(r"#![ \t]*(?:\S*/)?(?:env[ \t]+)?node(?:js)?(?=[ \t\r\n]|$)")


@dataclass(frozen=True, slots=True)
class JavaScriptProgram:
    """Read-only syntax and exact character offsets for one complete source."""

    content: str
    root_node: Node
    nodes: tuple[Node, ...]
    _multibyte_ends: array
    _extra_bytes: array

    def span(self, node: Node) -> tuple[int, int]:
        def offset(byte_offset: int) -> int:
            index = bisect_right(self._multibyte_ends, byte_offset)
            return byte_offset - (self._extra_bytes[index - 1] if index else 0)

        return offset(node.start_byte), offset(node.end_byte)

    def text(self, node: Node) -> str:
        start, end = self.span(node)
        return self.content[start:end]


def parse_javascript_source(
    content: str, check_runtime: Callable[[], None]
) -> JavaScriptProgram | None:
    """Return complete syntax, or no ownership on malformed/over-budget input.

    The pinned native timeout bounds parsing without Python callbacks: the
    upstream progress callback crashes on Python 3.12/3.13. Caller expiry is
    checked immediately before and after that bounded native operation.
    """
    check_runtime()
    if len(content) > MAX_JAVASCRIPT_SOURCE_CHARS or "\x00" in content:
        return None
    deadline = time.monotonic() + _MAX_SECONDS
    try:
        source = content.encode("utf-8")
    except UnicodeEncodeError:
        return None
    check_runtime()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    tree: Tree | None
    try:
        # Zero disables the native timeout, so round a positive budget up to
        # at least one microsecond. A fresh parser cannot resume timed-out work.
        parser = Parser(_LANGUAGE, timeout_micros=max(1, int(remaining * 1_000_000)))
        tree = parser.parse(source)
    except (ValueError, TypeError, OverflowError):
        tree = None
    check_runtime()
    if tree is None or time.monotonic() >= deadline:
        return None
    root = tree.root_node
    if root.type != "program" or root.has_error or root.end_byte != len(source):
        return None

    nodes: list[Node] = []
    pending = [(root, 0)]
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited % 256 == 0:
            check_runtime()
            if time.monotonic() >= deadline:
                return None
        if visited > _MAX_NODES or depth > _MAX_DEPTH or node.is_error or node.is_missing:
            return None
        if node.type == "hash_bang_line" and (
            node.start_byte != 0 or NODE_SHEBANG.match(content) is None
        ):
            return None
        if _invalid_optional_target(node, check_runtime):
            return None
        if node.is_named:
            nodes.append(node)
        # Bound child allocation before materializing a wide CST level.
        if visited + len(pending) + node.child_count > _MAX_NODES:
            return None
        pending.extend((child, depth + 1) for child in reversed(node.children))

    multibyte_ends = array("I")
    extra_bytes = array("I")
    byte_offset = 0
    extra = 0
    for index, character in enumerate(content):
        if index % 4096 == 0:
            check_runtime()
            if time.monotonic() >= deadline:
                return None
        codepoint = ord(character)
        width = 1 + (codepoint > 0x7F) + (codepoint > 0x7FF) + (codepoint > 0xFFFF)
        byte_offset += width
        if width > 1:
            extra += width - 1
            multibyte_ends.append(byte_offset)
            extra_bytes.append(extra)
    check_runtime()
    return JavaScriptProgram(content, root, tuple(nodes), multibyte_ends, extra_bytes)


def _invalid_optional_target(node: Node, check_runtime: Callable[[], None]) -> bool:
    """Reject optional chains in targets that the permissive CST admits.

    Parentheses end a chain; computed indices and call arguments are separate
    expressions. This is a targeted validity check, not all ECMAScript early
    errors or binding validation.
    """
    depth = 0
    if node.type in {
        "assignment_expression",
        "augmented_assignment_expression",
        "for_in_statement",
    }:
        target = node.child_by_field_name("left")
    elif node.type == "update_expression":
        target = node.child_by_field_name("argument")
    elif node.type == "new_expression":
        target = node.child_by_field_name("constructor")
    elif node.type == "call_expression":
        arguments = node.child_by_field_name("arguments")
        target = (
            node.child_by_field_name("function")
            if arguments is not None and arguments.type == "template_string"
            else None
        )
    else:
        return False
    if node.type in {
        "assignment_expression",
        "augmented_assignment_expression",
        "update_expression",
        "for_in_statement",
    }:
        while target is not None and target.type == "parenthesized_expression":
            check_runtime()
            depth += 1
            if depth > _MAX_DEPTH:
                return True
            children = [child for child in target.named_children if child.type != "comment"]
            target = children[0] if len(children) == 1 else None
    while target is not None:
        check_runtime()
        depth += 1
        if depth > _MAX_DEPTH:
            return True
        if target.child_by_field_name("optional_chain") is not None:
            return True
        if target.type in {"member_expression", "subscript_expression"}:
            target = target.child_by_field_name("object")
        elif target.type == "call_expression":
            target = target.child_by_field_name("function")
        else:
            break
    return False
