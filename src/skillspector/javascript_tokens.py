# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded lexical ownership for Node assertion template-message delimiters.

This is not a JavaScript parser or a shell-sink/dataflow model. Only a direct
diagnostic argument of one unaliased built-in assertion import may borrow
ownership. Unsupported source keeps its original conservative shell reading.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import esprima  # type: ignore[import-untyped]

from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

_MAX_SOURCE_CHARS = 1_000_000
_MAX_TOKENS = 100_000
_MAX_NESTING = 128
_LINE_ENDS = "\r\n\u2028\u2029"
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_TEMPLATE_VALUE = re.compile(
    r"\s*[A-Za-z_$][A-Za-z0-9_$]*(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*\s*\Z"
)
_NODE_SHEBANG = re.compile(r"#![ \t]*(?:\S*/)?(?:env[ \t]+)?node(?:js)?(?=[ \t\r\n]|$)")
_ASSERT_MODULES = frozenset({"node:assert", "node:assert/strict"})
_ASSERT_MODULE_ALIASES = _ASSERT_MODULES | {"assert", "assert/strict"}
_MESSAGE_ARGUMENTS = {
    "ok": 2,
    "equal": 3,
    "notEqual": 3,
    "strictEqual": 3,
    "notStrictEqual": 3,
    "deepEqual": 3,
    "notDeepEqual": 3,
    "deepStrictEqual": 3,
    "notDeepStrictEqual": 3,
    "match": 3,
    "doesNotMatch": 3,
    "throws": 3,
    "doesNotThrow": 3,
}
_EXPRESSION_PREFIX_WORDS = frozenset(
    {
        "return",
        "throw",
        "case",
        "delete",
        "void",
        "typeof",
        "new",
        "yield",
        "await",
        "in",
        "instanceof",
        "else",
        "do",
    }
)
_OPERATORS = frozenset(
    {
        "===",
        "!==",
        ">>>",
        "**=",
        "&&=",
        "||=",
        "??=",
        "...",
        "=>",
        "==",
        "!=",
        "<=",
        ">=",
        "&&",
        "||",
        "??",
        "++",
        "--",
        "**",
        "+=",
        "-=",
        "*=",
        "%=",
        "&=",
        "|=",
        "^=",
        "<<",
        ">>",
        "?.",
    }
)


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    start: int
    end: int
    depth: int
    interpolation_names: tuple[str, ...] = ()


def _tokens(
    content: str, check_runtime: Callable[[], None]
) -> tuple[list[_Token], dict[int, int]] | None:
    """Tokenize the limited lexical forms needed to prove message boundaries."""
    tokens: list[_Token] = []
    pairs: dict[int, int] = {}
    stack: list[tuple[str, int]] = []
    cursor = 0
    expression_start = True
    next_check = 0

    def check(position: int) -> None:
        nonlocal next_check
        if position >= next_check:
            check_runtime()
            next_check = position + 4096

    if content.startswith("#!"):
        if _NODE_SHEBANG.match(content) is None:
            return None
        cursor = content.find("\n")
        if cursor < 0:
            return None
    while cursor < len(content):
        check(cursor)
        character = content[cursor]
        if character.isspace():
            cursor += 1
            continue
        if character == "\x00" or len(tokens) >= _MAX_TOKENS:
            return None
        if content.startswith("//", cursor):
            while cursor < len(content) and content[cursor] not in _LINE_ENDS:
                check(cursor)
                cursor += 1
            continue
        if content.startswith("/*", cursor):
            cursor += 2
            while cursor < len(content) and not content.startswith("*/", cursor):
                check(cursor)
                cursor += 1
            if cursor == len(content):
                return None
            cursor += 2
            continue
        start = cursor
        kind = "punctuation"
        interpolation_names: list[str] = []
        if character in "'\"`":
            kind = "template" if character == "`" else "string"
            cursor += 1
            while cursor < len(content):
                check(cursor)
                current = content[cursor]
                if current == "\\":
                    cursor += 2
                    if cursor <= len(content) and content[cursor - 1 : cursor + 1] == "\r\n":
                        cursor += 1
                    continue
                if current == character:
                    cursor += 1
                    break
                if kind == "string" and current in _LINE_ENDS:
                    return None
                if kind == "template" and content.startswith("${", cursor):
                    value_start = cursor + 2
                    cursor = value_start
                    while cursor < len(content) and content[cursor] != "}":
                        check(cursor)
                        # Calls, computed members, comments, nested templates,
                        # and other effectful expressions never gain ownership.
                        if not content[cursor].isascii():
                            return None
                        cursor += 1
                    value = content[value_start:cursor]
                    if cursor == len(content) or _TEMPLATE_VALUE.fullmatch(value) is None:
                        return None
                    interpolation_names.extend(_IDENTIFIER.findall(value))
                    cursor += 1
                    continue
                cursor += 1
            else:
                return None
            expression_start = False
        elif character == "/":
            # Division versus a statement-level regexp needs more grammar.
            # Only a clear expression prefix establishes regexp ownership.
            if not expression_start:
                return None
            kind = "regexp"
            cursor += 1
            in_class = False
            while cursor < len(content):
                check(cursor)
                current = content[cursor]
                if current in _LINE_ENDS:
                    return None
                if current == "\\":
                    if cursor + 1 == len(content) or content[cursor + 1] in _LINE_ENDS:
                        return None
                    cursor += 2
                    continue
                if current == "[":
                    if in_class:
                        # Unicode-set nested classes require a different grammar.
                        return None
                    in_class = True
                elif current == "]":
                    in_class = False
                elif current == "/" and not in_class:
                    cursor += 1
                    break
                cursor += 1
            else:
                return None
            flags_start = cursor
            while cursor < len(content) and content[cursor].isascii() and content[cursor].isalpha():
                check(cursor)
                cursor += 1
            flags = content[flags_start:cursor]
            if len(set(flags)) != len(flags) or set(flags) - set("dgimsuy"):
                return None
            expression_start = False
        elif (identifier := _IDENTIFIER.match(content, cursor)) is not None:
            kind = "identifier"
            cursor = identifier.end()
            previous = content[tokens[-1].start : tokens[-1].end] if tokens else ""
            expression_start = identifier[0] in _EXPRESSION_PREFIX_WORDS and previous not in {
                ".",
                "?.",
            }
        elif character.isascii() and character.isdigit():
            kind = "number"
            cursor += 1
            while cursor < len(content) and content[cursor] in "0123456789abcdefABCDEFxXoObBn._":
                check(cursor)
                cursor += 1
            expression_start = False
        elif character in "()[]{};,:.?=<>!~+-*%&|^":
            width = 1
            if content[cursor : cursor + 3] in _OPERATORS:
                width = 3
            elif content[cursor : cursor + 2] in _OPERATORS:
                width = 2
            cursor += width
            operator = content[start:cursor]
            if operator == "!" and not expression_start:
                # TypeScript's postfix assertion leaves division possible.
                return None
            if operator in "([{":
                if len(stack) >= _MAX_NESTING:
                    return None
                stack.append((operator, len(tokens)))
            elif operator in ")]}":
                if not stack or stack[-1][0] != {"}": "{", ")": "(", "]": "["}[operator]:
                    return None
                _, opening = stack.pop()
                pairs[opening] = len(tokens)
            expression_start = operator not in {")", "]", "}", ".", "?.", "++", "--"}
        else:
            # Escaped/Unicode identifiers, JSX, and unknown lexical forms are
            # not proof of a Node assertion receiver or template boundary.
            return None
        tokens.append(_Token(kind, start, cursor, len(stack), tuple(interpolation_names)))
    check_runtime()
    return None if stack else (tokens, pairs)


def assertion_template_shell_text(content: str, check_runtime: Callable[[], None]) -> str:
    """Project only proven Node assertion message delimiters for shell completeness.

    Literal bodies and interpolations remain byte-for-byte present. Findings
    must still be produced from the original source. Unknown calls, tagged or
    concatenated templates, ambiguous imports, and receiver mutation/shadowing
    receive no exemption. Runtime interruption propagates to the caller.

    This assumes the standard built-in module. It cannot prove the absence of
    arbitrary runtime monkeypatching or object getters.
    """
    if len(content) > _MAX_SOURCE_CHARS or "node:assert" not in content or "`" not in content:
        return content
    parsed = _tokens(content, check_runtime)
    if parsed is None:
        return content
    tokens, pairs = parsed
    values = [content[token.start : token.end] for token in tokens]
    imports: list[tuple[str, int, int]] = []
    for index, token in enumerate(tokens):
        if index % 256 == 0:
            check_runtime()
        if token.kind == "identifier" and (
            values[index] == "require"
            or values[index] == "import"
            and values[index + 1 : index + 2] == ["("]
        ):
            return content
        if token.kind == "string" and index and values[index - 1] in {"import", "from"}:
            # Escaped module names can alias the same mutable built-in object.
            if "\\" in values[index]:
                return content
        if values[index] != "import" or token.depth:
            continue
        following = values[index : index + 7]
        if len(following) >= 5 and following[2] == "from" and following[4] == ";":
            binding_index, module_index = index + 1, index + 3
        elif (
            len(following) == 7
            and following[1:3] == ["*", "as"]
            and following[4] == "from"
            and following[6] == ";"
        ):
            binding_index, module_index = index + 3, index + 5
        else:
            continue
        if (
            tokens[binding_index].kind == "identifier"
            and tokens[module_index].kind == "string"
            and values[module_index][1:-1] in _ASSERT_MODULES
        ):
            imports.append((values[binding_index], binding_index, module_index))
    if len(imports) != 1:
        return content
    binding, binding_index, module_index = imports[0]
    if any(
        token.kind == "string"
        and index != module_index
        and values[index][1:-1] in _ASSERT_MODULE_ALIASES
        or binding in token.interpolation_names
        for index, token in enumerate(tokens)
    ):
        return content
    delimiters: list[int] = []
    for index, token in enumerate(tokens):
        if index % 256 == 0:
            check_runtime()
        if token.kind != "identifier" or values[index] != binding or index == binding_index:
            continue
        if (index and values[index - 1] in {".", "?.", "new"}) or values[index + 1 : index + 2] != [
            "."
        ]:
            return content
        if (
            index + 3 >= len(tokens)
            or values[index + 2] not in _MESSAGE_ARGUMENTS
            or values[index + 3] != "("
        ):
            return content
        opening = index + 3
        closing = pairs.get(opening)
        if closing is None:
            return content
        arguments: list[tuple[int, int]] = []
        begin = cursor = opening + 1
        while cursor < closing:
            check_runtime()
            if cursor in pairs:
                cursor = pairs[cursor] + 1
                continue
            if values[cursor] == ",":
                arguments.append((begin, cursor))
                begin = cursor + 1
            cursor += 1
        if begin < closing:
            arguments.append((begin, closing))
        if len(arguments) != _MESSAGE_ARGUMENTS[values[index + 2]] or any(
            start == end or values[start] == "..." for start, end in arguments
        ):
            continue
        start, end = arguments[-1]
        if end == start + 1 and tokens[start].kind == "template":
            delimiters.extend((tokens[start].start, tokens[start].end - 1))
    if not delimiters:
        return content
    output = list(content)
    for position in delimiters:
        output[position] = " "
    check_runtime()
    return "".join(output)


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
        property_name = getattr(callee.property, "name", None)
        if property_name is None:
            property_name = getattr(callee.property, "value", None)
        return isinstance(property_name, str) and property_name.casefold() in _SHELL_EXECUTION_CALLS
    return False
