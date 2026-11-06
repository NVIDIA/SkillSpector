# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded lexical ownership for TypeScript template literals.

This is not a TypeScript parser or a shell-sink/dataflow model. It proves
only template-literal byte ranges in complete TypeScript source, and it
keeps the original conservative shell reading for anything it cannot prove.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from skillspector.javascript_tokens import (
    _SHELL_EXECUTION_CALLS as _JAVASCRIPT_EXECUTION_CALLS,
)
from skillspector.javascript_tokens import (
    _SHELL_EXECUTION_MODULES,
    JavaScriptTemplateSpans,
)
from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

_MAX_NESTING = 128
_MAX_TEMPLATES = 100_000
_LINE_ENDS = "\r\n\u2028\u2029"
_TYPESCRIPT_SHEBANG = re.compile(
    r"#![ \t]*(?:\S*/)?(?:env[ \t]+(?:-S[ \t]+)?)?"
    r"(?:node|nodejs|ts-node|tsx|bun|deno)(?=[ \t\r\n]|$)",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
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
_CONTROL_WORDS = frozenset({"if", "while", "for", "with", "switch", "catch"})
_INVALID_ASSIGNMENT_KEYWORDS = frozenset(
    {
        "const",
        "let",
        "var",
        "function",
        "class",
        "interface",
        "type",
        "enum",
        "namespace",
        "declare",
        "abstract",
    }
)
_REGEX_DIVISION_PUNCTUATION = frozenset("([{,;:?=<>!~+-*%&|^") | {">=", "=>"}
_TAG_PUNCTUATION = frozenset({")", "]", ".", ">", "!"})
_MAX_UNICODE_ESCAPE = 0x10FFFF
_SURROGATE_START = 0xD800
_SURROGATE_END = 0xDFFF
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_MODULE_KEYWORDS = frozenset({"from", "import", "require"})
_FORM_KEYWORDS = frozenset({"import", "from", "as"})
_SHELL_EXECUTION_CALLS = _JAVASCRIPT_EXECUTION_CALLS
# Lowercase ``function`` is the declaration keyword; ``Function``, which the
# casefolded comparison would otherwise fold together with it, is the
# constructor an executable template can reach.
_DECLARATION_KEYWORD = "function"


def _decode_string_source(source: str, check_runtime: Callable[[], None]) -> str | None:
    """Return the value a quoted source slice spells, or ``None`` when it cannot
    be decoded completely.

    Escape sequences are resolved here, never evaluated: ``\\xNN``, ``\\uNNNN``,
    ``\\u{...}``, the single-character escapes, a line continuation, and an
    escaped quote or backslash. Malformed, truncated, lone-surrogate, or
    unknown escapes return ``None`` so the caller keeps the conservative read.
    """
    output: list[str] = []
    cursor = 0
    length = len(source)
    while cursor < length:
        character = source[cursor]
        if character != "\\":
            output.append(character)
            cursor += 1
            continue
        cursor += 1
        if cursor >= length:
            return None
        mark = source[cursor]
        # ``\r`` opens the CRLF continuation below, not the simple map.
        if mark in "\r\n\u2028\u2029":
            if mark == "\r" and cursor + 1 < length and source[cursor + 1] == "\n":
                cursor += 1
            cursor += 1
            continue
        if mark == "0":
            following = source[cursor + 1] if cursor + 1 < length else ""
            if following.isdigit():
                return None
            output.append("\0")
            cursor += 1
            continue
        if mark in _HEX_DIGITS and mark not in "xu":
            return None
        if mark == "x":
            digits = source[cursor + 1 : cursor + 3]
            if len(digits) != 2 or not all(d in _HEX_DIGITS for d in digits):
                return None
            output.append(chr(int(digits, 16)))
            cursor += 3
            continue
        if mark == "u":
            if source.startswith("{", cursor + 1):
                close = source.find("}", cursor + 2)
                if close < 0:
                    return None
                digits = source[cursor + 2 : close]
                if not 1 <= len(digits) <= 6 or not all(d in _HEX_DIGITS for d in digits):
                    return None
                code = int(digits, 16)
                if code > _MAX_UNICODE_ESCAPE or _SURROGATE_START <= code <= _SURROGATE_END:
                    return None
                output.append(chr(code))
                cursor = close + 1
                continue
            digits = source[cursor + 1 : cursor + 5]
            if len(digits) != 4 or not all(d in _HEX_DIGITS for d in digits):
                return None
            code = int(digits, 16)
            if _SURROGATE_START <= code <= _SURROGATE_END:
                return None
            output.append(chr(code))
            cursor += 5
            continue
        output.append(mark)
        cursor += 1
    check_runtime()
    return "".join(output)


def _is_shell_execution_module(value: str) -> bool:
    normalized = value.removeprefix("node:")
    return any(
        normalized == name or normalized.startswith(f"{name}/") for name in _SHELL_EXECUTION_MODULES
    )


def typescript_template_spans(
    content: str,
    check_runtime: Callable[[], None],
) -> JavaScriptTemplateSpans | None:
    """Return proven template literal ranges for complete TypeScript source."""
    if "`" not in content or len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return None
    check_runtime()
    templates: list[tuple[int, int]] = []
    next_check = 0
    last_kind: str | None = None
    last_value = ""
    last_closed_control = False
    paren_stack: list[bool] = []
    square_stack: list[int] = []
    delimiters: list[str] = []
    spans: list[tuple[int, int]] = []
    values: dict[int, str | None] = {}
    callee: str | None = None

    def check(position: int) -> None:
        nonlocal next_check
        if position >= next_check:
            check_runtime()
            next_check = position + 4096

    def execution_name(word: str) -> bool:
        """Whether *word*, as written, names a shell execution call.

        The list is lowercase, so the comparison folds case; only the exact
        lowercase ``function`` keyword is read as a declaration instead.
        """
        return word.casefold() in _SHELL_EXECUTION_CALLS and word != _DECLARATION_KEYWORD

    def brace_member() -> bool | None:
        """Whether a template right after ``{`` sits where a command runs.

        Returns None when the brace is not proven to belong to an execution
        call, which keeps the template unproven rather than owned.
        """
        if last_kind == "identifier":
            return execution_name(last_value) or None
        if last_kind == "punct" and last_value == ".":
            return True
        return None

    def regex_allowed() -> bool:
        if last_kind is None:
            return True
        if last_kind == "identifier":
            return last_value.casefold() in _EXPRESSION_PREFIX_WORDS
        if last_kind in {"number", "string", "template", "regex", "postfix"}:
            return False
        if last_kind != "punct":
            return True
        if last_value == ")":
            return last_closed_control
        if last_value == "]":
            return False
        if last_value == "}":
            return True
        if last_value == ".":
            return False
        if last_value == ">":
            return False
        return last_value in _REGEX_DIVISION_PUNCTUATION

    def string_value(span: tuple[int, int]) -> str | None:
        start, end = span
        if start not in values:
            values[start] = _decode_string_source(content[start:end], check_runtime)
        return values[start]

    def shell_execution_member(span: tuple[int, int]) -> bool:
        """Whether a member name, computed or dotted, names a shell execution call."""
        if last_kind == "punct" and last_value == ".":
            return content[span[0] : span[1]].casefold() in _SHELL_EXECUTION_CALLS
        value = string_value(span)
        return value is None or execution_name(value)

    def template_member_value(span: tuple[int, int]) -> str | None:
        start, end = span
        if start + 1 == end:
            return ""
        return _decode_string_source(content[start + 1 : end - 1], check_runtime)

    def module_string(span: tuple[int, int]) -> bool:
        value = string_value(span)
        if value is None:
            return _is_shell_execution_module(content[span[0] : span[1]])
        return _is_shell_execution_module(value)

    def tagged_template() -> bool:
        # ``import ... from`` is a module clause, not a tag; ``import()`` and a
        # bare ``import`` tag are not.
        if last_kind == "identifier" and last_value not in _FORM_KEYWORDS:
            return last_value.casefold() not in _EXPRESSION_PREFIX_WORDS
        if last_kind in {"number", "string", "template", "regex", "postfix"}:
            return True
        return last_kind == "punct" and last_value in _TAG_PUNCTUATION

    def skip_string(cursor: int, quote: str) -> int | None:
        value_start = cursor + 1
        cursor = value_start
        while cursor < len(content):
            check(cursor)
            character = content[cursor]
            if character == "\\":
                cursor += 2
                continue
            if character == quote:
                spans.append((value_start, cursor))
                return cursor + 1
            if character in _LINE_ENDS:
                return None
            cursor += 1
        return None

    def skip_regex(cursor: int) -> int | None:
        cursor += 1
        in_class = False
        while cursor < len(content):
            check(cursor)
            character = content[cursor]
            if character in _LINE_ENDS:
                return None
            if character == "\\":
                if cursor + 1 >= len(content) or content[cursor + 1] in _LINE_ENDS:
                    return None
                cursor += 2
                continue
            if character == "[":
                if in_class:
                    return None
                in_class = True
            elif character == "]":
                in_class = False
            elif character == "/" and not in_class:
                cursor += 1
                while (
                    cursor < len(content)
                    and content[cursor].isascii()
                    and content[cursor].isalpha()
                ):
                    check(cursor)
                    cursor += 1
                return cursor
            cursor += 1
        return None

    def skip_template(cursor: int, depth: int) -> int | None:
        nonlocal callee
        if depth >= _MAX_NESTING or len(templates) >= _MAX_TEMPLATES:
            return None
        start = cursor
        cursor += 1
        while cursor < len(content):
            check(cursor)
            character = content[cursor]
            if character == "\\":
                cursor += 2
                continue
            if character == "`":
                end = cursor + 1
                templates.append((start, end))
                return end
            if content.startswith("${", cursor):
                # The substitution brace is recorded on the shared delimiter
                # stack so a later top-level ``}`` cannot close it.
                delimiters.append("{")
                next_cursor = scan_code(cursor + 2, depth + 1, interpolation=True)
                if next_cursor is None:
                    return None
                cursor = next_cursor
                callee = None
                continue
            cursor += 1
        return None

    def scan_code(cursor: int, depth: int, *, interpolation: bool = False) -> int | None:
        nonlocal last_kind, last_value, last_closed_control, callee
        if depth >= _MAX_NESTING:
            return None
        outer_callee = callee
        callee = None
        braces = 0
        paren_boundary = 0
        paren_floor = 0
        square_floor = 0
        if interpolation:
            paren_boundary = paren_floor = len(paren_stack)
            square_floor = len(square_stack)
        while cursor < len(content):
            check(cursor)
            character = content[cursor]
            if character.isspace():
                cursor += 1
                continue
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
            if character in "'\"":
                end = skip_string(cursor, character)
                if end is None:
                    return None
                span = spans[-1]
                if last_kind == "punct" and last_value in {"[", "."}:
                    if shell_execution_member(span):
                        return None
                elif (
                    last_kind == "identifier"
                    and last_value.casefold() in _MODULE_KEYWORDS
                    and module_string(span)
                ):
                    return None
                elif (
                    last_kind == "punct"
                    and last_value == "("
                    and callee is not None
                    and callee.casefold() in _MODULE_KEYWORDS
                    and module_string(span)
                ):
                    return None
                cursor = end
                last_kind, last_value, last_closed_control = "string", "", False
                continue
            if character == "`":
                if tagged_template():
                    return None
                end = skip_template(cursor, depth + 1)
                if end is None:
                    return None
                if last_kind == "punct" and last_value == "[":
                    member = template_member_value((cursor, end))
                    if member is None or execution_name(member):
                        return None
                elif (
                    last_kind == "punct"
                    and last_value == "("
                    and callee is not None
                    and callee.casefold() in _MODULE_KEYWORDS
                ):
                    module = template_member_value((cursor, end))
                    if module is None or _is_shell_execution_module(module):
                        return None
                elif last_kind == "punct" and last_value == "{":
                    # A brace body such as ``cp.exec({ command: `...` })`` is a
                    # call argument; one we cannot place stays unproven.
                    member = brace_member()
                    if not member:
                        return None
                cursor = end
                last_kind, last_value, last_closed_control = "template", "", False
                continue
            if character == "/":
                if last_kind == "punct" and last_value == ">":
                    return None
                if regex_allowed():
                    end = skip_regex(cursor)
                    if end is None:
                        return None
                    cursor = end
                    last_kind, last_value, last_closed_control = "regex", "", False
                    continue
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", "/", False
                continue
            if character == "(":
                paren_stack.append(
                    last_kind == "identifier" and last_value.casefold() in _CONTROL_WORDS
                )
                callee = last_value if last_kind == "identifier" else None
                delimiters.append(character)
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", "(", False
                continue
            if character == ")":
                if (
                    len(paren_stack) <= paren_floor
                    or not delimiters
                    or delimiters[-1] != "("
                    or delimiters.count("[") > delimiters.count("]")
                ):
                    return None
                delimiters.pop()
                last_closed_control = paren_stack.pop()
                callee = None
                cursor += 1
                last_kind, last_value = "punct", ")"
                continue
            if character == "{":
                braces += 1
                delimiters.append("{")
                paren_stack.append(paren_boundary)
                callee = None
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", "{", False
                continue
            if character == "}":
                if not delimiters or delimiters[-1] != "{":
                    return None
                if interpolation and braces == 0:
                    if len(paren_stack) != paren_boundary or len(square_stack) != square_floor:
                        return None
                    delimiters.pop()
                    callee = outer_callee
                    return cursor + 1
                delimiters.pop()
                if interpolation:
                    # A block cannot open inside ``[ ... ]``; that bracket
                    # would otherwise stay pending while the block closes.
                    if delimiters.count("[") > delimiters.count("]"):
                        return None
                    braces -= 1
                else:
                    if len(square_stack) != square_floor:
                        return None
                    if paren_stack.pop() != paren_boundary:
                        return None
                    braces = 0
                callee = None
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", "}", False
                continue
            if character == "[":
                square_stack.append(cursor)
                delimiters.append("[")
                callee = None
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", "[", False
                continue
            if character == "]":
                if (
                    len(square_stack) <= square_floor
                    or not delimiters
                    or delimiters[-1] != "["
                    or delimiters.count("(") > delimiters.count(")")
                ):
                    return None
                delimiters.pop()
                square_stack.pop()
                callee = None
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", "]", False
                continue
            if character == ",":
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", ",", False
                continue
            if character == ".":
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", ".", False
                continue
            if character == "!":
                cursor += 1
                if cursor < len(content) and content[cursor] == "=":
                    cursor += 1
                    if cursor < len(content) and content[cursor] == "=":
                        cursor += 1
                    last_kind, last_value, last_closed_control = "punct", "!", False
                    continue
                postfix = last_kind in {"identifier", "number", "string", "template", "regex"} or (
                    last_kind == "punct" and last_value in {")", "]", "}"}
                )
                last_kind, last_value, last_closed_control = (
                    ("postfix", "!", False) if postfix else ("punct", "!", False)
                )
                continue
            if character in "+-" and cursor + 1 < len(content) and content[cursor + 1] == character:
                postfix = last_kind in {"identifier", "number", "string", "template", "regex"} or (
                    last_kind == "punct" and last_value in {")", "]"}
                )
                cursor += 2
                callee = None
                last_kind, last_value, last_closed_control = (
                    ("postfix", character * 2, False)
                    if postfix
                    else ("punct", character * 2, False)
                )
                continue
            if character == "=" and cursor + 1 < len(content) and content[cursor + 1] == ">":
                cursor += 2
                last_kind, last_value, last_closed_control = "punct", "=>", False
                continue
            if character == ">" and cursor + 1 < len(content) and content[cursor + 1] == "=":
                cursor += 2
                last_kind, last_value, last_closed_control = "punct", ">=", False
                continue
            if character in _REGEX_DIVISION_PUNCTUATION:
                if (
                    character == "="
                    and last_kind == "identifier"
                    and last_value.casefold() in _INVALID_ASSIGNMENT_KEYWORDS
                ):
                    return None
                cursor += 1
                last_kind, last_value, last_closed_control = "punct", character, False
                continue
            identifier = _IDENTIFIER.match(content, cursor)
            if identifier is not None:
                word = identifier.group(0)
                # ``if (...)`` reaches a statement head, so only a control
                # keyword followed by an argument list is a call here.
                if execution_name(word) and not (
                    word.casefold() in _CONTROL_WORDS and content.startswith("(", identifier.end())
                ):
                    return None
                callee = None
                cursor = identifier.end()
                last_kind, last_value, last_closed_control = "identifier", word, False
                continue
            if character.isdigit():
                cursor += 1
                while cursor < len(content) and (
                    content[cursor].isalnum() or content[cursor] in "._"
                ):
                    check(cursor)
                    cursor += 1
                last_kind, last_value, last_closed_control = "number", "", False
                continue
            cursor += 1
            last_kind, last_value, last_closed_control = "punct", character, False
        if interpolation or braces or delimiters:
            return None
        return cursor

    cursor = 0
    if content.startswith("#!"):
        if _TYPESCRIPT_SHEBANG.match(content) is None:
            return None
        newline = content.find("\n")
        if newline < 0:
            return None
        cursor = newline + 1
    if scan_code(cursor, 0) is None:
        return None
    if paren_stack or square_stack:
        return None
    if not templates:
        return None
    starts: list[int] = []
    ends: list[int] = []
    for start, end in sorted(templates):
        if starts and start < ends[-1]:
            if end <= ends[-1]:
                continue
            return None
        starts.append(start)
        ends.append(end)
    return tuple(starts), tuple(ends)
