# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser-proven JavaScript template literal spans for static analyzers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import esprima  # type: ignore[import-untyped]

from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

JavaScriptTemplateSpans = tuple[tuple[int, ...], tuple[int, ...]]

_SHELL_EXECUTION_CALLS = frozenset(
    {
        "exec",
        "execfile",
        "execfilesync",
        "execsync",
        "eval",
        "execa",
        "spawn",
        "spawnfile",
        "spawnfilesync",
        "spawnsync",
        "system",
    }
)
_SHELL_EXECUTION_MODULES = frozenset({"child_process", "execa", "shelljs", "zx"})


def javascript_template_spans(
    content: str,
    check_runtime: Callable[[], None],
) -> JavaScriptTemplateSpans | None:
    """Return template literal ranges only when a complete script parses.

    Unsupported or malformed JavaScript returns ``None`` so callers retain
    their conservative parse bounds. A file with a template and a shell
    execution call also keeps those bounds because this parser does not
    establish whether that template supplies the executed command.
    """
    if "`" not in content or len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return None
    check_runtime()
    options = {"range": True}
    try:
        try:
            program = esprima.parseScript(content, options)
        except esprima.Error:
            program = esprima.parseModule(content, options)
    except (esprima.Error, RecursionError, ValueError, TypeError):
        return None
    check_runtime()

    templates: list[tuple[int, int, bool]] = []
    for node in _walk_nodes(program, check_runtime):
        if node.type == "TemplateLiteral":
            source_range = getattr(node, "range", None)
            if not isinstance(source_range, list) or len(source_range) != 2:
                return None
            start, end = source_range
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end > len(content)
                or start >= end
                or content[start] != "`"
                or content[end - 1] != "`"
            ):
                # Esprima offsets must map back to the exact source. A mismatch
                # (for example after an astral Unicode character) proves no
                # ownership for this artifact.
                return None
            templates.append((start, end, bool(node.expressions)))

    if not templates:
        return None

    if _has_shell_execution_call(program, check_runtime):
        return None

    # Nested template literals occur inside interpolation expressions. The
    # outer literal owns all of those bytes, so retain only disjoint outermost
    # spans for the shell parser's offset-based ownership checks.
    starts: list[int] = []
    ends: list[int] = []
    for start, end, _dynamic in sorted(templates):
        if starts and start < ends[-1]:
            if end <= ends[-1]:
                continue
            return None
        starts.append(start)
        ends.append(end)
    return tuple(starts), tuple(ends)


def _walk_nodes(value: Any, check_runtime: Callable[[], None]) -> Iterator[Any]:
    pending = [value]
    visited = 0
    while pending:
        visited += 1
        if visited % 256 == 0:
            check_runtime()
        current = pending.pop()
        if isinstance(current, (list, tuple)):
            pending.extend(reversed(current))
        elif hasattr(current, "type"):
            yield current
            children = [
                child
                for name, child in vars(current).items()
                if name not in {"type", "range", "loc", "tokens", "comments"}
            ]
            pending.extend(reversed(children))


def _has_shell_execution_call(program: Any, check_runtime: Callable[[], None]) -> bool:
    for node in _walk_nodes(program, check_runtime):
        if node.type == "ImportDeclaration" and _is_shell_execution_module(
            getattr(node.source, "value", "")
        ):
            return True
        if node.type in {"CallExpression", "NewExpression"}:
            if _is_shell_execution_callee(node.callee):
                return True
            if (
                node.callee.type == "Identifier"
                and node.callee.name == "require"
                and node.arguments
                and _is_shell_execution_module(getattr(node.arguments[0], "value", ""))
            ):
                return True
        if node.type == "TaggedTemplateExpression":
            tag = node.tag
            if _is_shell_execution_callee(tag) or (tag.type == "Identifier" and tag.name == "$"):
                return True
    return False


def _is_shell_execution_module(module: Any) -> bool:
    if not isinstance(module, str):
        return False
    normalized = module.removeprefix("node:")
    return any(
        normalized == name or normalized.startswith(f"{name}/") for name in _SHELL_EXECUTION_MODULES
    )


def _is_shell_execution_callee(callee: Any) -> bool:
    if callee.type == "Identifier":
        return callee.name.casefold() in _SHELL_EXECUTION_CALLS
    if callee.type == "MemberExpression":
        property_name = getattr(callee.property, "name", "") or getattr(
            callee.property, "value", ""
        )
        return property_name.casefold() in _SHELL_EXECUTION_CALLS
    return False
