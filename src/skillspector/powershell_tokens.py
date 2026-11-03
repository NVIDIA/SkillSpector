# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded lexical ownership of PowerShell strings and comments.

This does not execute PowerShell or establish command data flow. It only
prevents host-language quoting from extending a shell word into later code.
The caller must separately inspect string values and embedded shell calls.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

_MAX_DEPTH = 32
_MAX_SPANS = 65_536
_ESCAPES = {
    "0": "\x00",
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}
_DYNAMIC_CODE = re.compile(
    r"\b(?:Invoke-Expression|iex|InvokeScript|Set-Alias|New-Alias)\b"
    r"|\[\s*scriptblock\s*\]\s*::\s*Create\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PowerShellString:
    start: int
    end: int
    value: str
    decoded: bool
    invoked: bool
    operand_start: int | None = None


@dataclass(frozen=True)
class PowerShellSource:
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    strings: tuple[PowerShellString, ...]
    clause_starts: tuple[int, ...]
    continuations: tuple[tuple[int, int], ...] = ()
    invocations: tuple[tuple[int, int], ...] = ()

    def owns(self, position: int) -> bool:
        index = bisect_right(self.starts, position) - 1
        return index >= 0 and position < self.ends[index]

    def next_boundary(self, position: int, fallback: int) -> int:
        index = bisect_right(self.starts, position) - 1
        if index >= 0 and position < self.ends[index]:
            return self.ends[index]
        index += 1
        return self.starts[index] if index < len(self.starts) else fallback


class _UnprovenPowerShellError(Exception):
    pass


def powershell_source(
    content: str,
    check_runtime: Callable[[], None],
    *,
    unresolved_executables: list[int] | None = None,
) -> PowerShellSource | None:
    """Return only whole-source lexical proof; malformed input proves nothing."""
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return None
    spans: list[tuple[int, int]] = []
    strings: list[PowerShellString] = []
    clause_starts: set[int] = {0}
    continuations: list[tuple[int, int]] = []
    invocations: list[tuple[int, int]] = []
    checked = -1024

    def executable_boundary(start: int, end: int) -> None:
        if end < len(content) and not (content[end].isspace() or content[end] in ";|&)}"):
            # Falling back to shell quoting cannot establish coverage of a
            # PowerShell method or index expression that selects executable code.
            if unresolved_executables is not None:
                unresolved_executables.append(start)
            raise _UnprovenPowerShellError

    def check(cursor: int, depth: int) -> None:
        nonlocal checked
        if (
            depth > _MAX_DEPTH
            or len(spans)
            + len(strings)
            + len(clause_starts)
            + len(continuations)
            + len(invocations)
            > _MAX_SPANS
        ):
            raise _UnprovenPowerShellError
        if cursor - checked >= 1024:
            check_runtime()
            checked = cursor

    def own(start: int, end: int) -> None:
        if start < end:
            spans.append((start, end))

    def variable(start: int, depth: int) -> int:
        cursor = start + 2
        while cursor < len(content):
            check(cursor, depth)
            if content[cursor] == "`":
                cursor += 2
            elif content[cursor] == "}":
                return cursor + 1
            else:
                cursor += 1
        raise _UnprovenPowerShellError

    def quoted(start: int, depth: int, *, here: bool = False, invoked_context: bool = False) -> int:
        span_start = len(spans)
        quote_at = start + int(here)
        quote = content[quote_at]
        cursor = quote_at + 1
        if here:
            if content.startswith("\r\n", cursor):
                cursor += 2
            elif content[cursor : cursor + 1] == "\n":
                cursor += 1
            else:
                raise _UnprovenPowerShellError
        region_start = start
        value: list[str] = []
        decoded = False
        dynamic = False
        while cursor < len(content):
            check(cursor, depth)
            character = content[cursor]
            if not here and character in "‘’“”":
                # PowerShell also accepts mixed ASCII/typographic quote
                # delimiters. Do not claim bytes past an unproven closer.
                raise _UnprovenPowerShellError
            closing = (
                content.startswith(quote + "@", cursor)
                and (cursor == 0 or content[cursor - 1] == "\n")
                if here
                else character == quote
            )
            if closing:
                if not here and content[cursor : cursor + 2] == quote * 2:
                    value.append(quote)
                    decoded = True
                    cursor += 2
                    continue
                end = cursor + (2 if here else 1)
                if (
                    here
                    and end < len(content)
                    and not (content[end].isspace() or content[end] in ";|&),.+-*/[")
                ):
                    raise _UnprovenPowerShellError
                own(region_start, end)
                # A quoted command name after the PowerShell call operator
                # remains an executable candidate with its following operands.
                previous = start - 1
                parentheses = 0
                while previous >= 0 and content[previous] in " \t\r\n":
                    if previous % 1024 == 0:
                        check_runtime()
                    previous -= 1
                while previous >= 0 and content[previous] == "(":
                    parentheses += 1
                    if parentheses > _MAX_DEPTH:
                        raise _UnprovenPowerShellError
                    previous -= 1
                    if previous >= 0 and content[previous] == "$":
                        previous -= 1
                    while previous >= 0 and content[previous] in " \t\r\n":
                        if previous % 1024 == 0:
                            check_runtime()
                        previous -= 1
                invoked = (
                    invoked_context
                    or previous >= 0
                    and (
                        content[previous] == "&"
                        or (
                            content[previous] == "."
                            and (
                                previous == 0
                                or content[previous - 1].isspace()
                                or content[previous - 1] in ";({"
                            )
                        )
                    )
                )
                operand_start: int | None = end
                if invoked and parentheses:
                    suffix = end
                    for _ in range(parentheses):
                        while suffix < len(content) and content[suffix] in " \t\r\n":
                            check(suffix, depth)
                            suffix += 1
                        if content[suffix : suffix + 1] != ")":
                            # A concatenation or method call can still build
                            # the executable name. No literal-name proof.
                            operand_start = None
                            break
                        suffix += 1
                    else:
                        operand_start = suffix
                if (
                    invoked
                    and operand_start is not None
                    and operand_start < len(content)
                    and not content[operand_start].isspace()
                    and content[operand_start] not in ";|&"
                ):
                    operand_start = None
                if here and value and value[-1] == "\n":
                    # The physical newline before the footer belongs to the
                    # delimiter. Preserve any preceding body or escaped newline.
                    value.pop()
                    if content[cursor - 2 : cursor] == "\r\n" and value and value[-1] == "\r":
                        value.pop()
                strings.append(
                    PowerShellString(start, end, "".join(value), decoded, invoked, operand_start)
                )
                if invoked and dynamic and operand_start is not None:
                    invocations.append((start, operand_start))
                if invoked_context and operand_start is None and unresolved_executables is not None:
                    unresolved_executables.append(start)
                if invoked:
                    del spans[span_start:]
                return end
            if quote == '"' and character == "`":
                if cursor + 1 >= len(content):
                    raise _UnprovenPowerShellError
                escaped = content[cursor + 1]
                if here and escaped in "\r\n":
                    # An escaped physical newline does not take the native
                    # footer-recognition path. Do not claim that boundary.
                    raise _UnprovenPowerShellError
                if escaped == "u":
                    if content[cursor + 2 : cursor + 3] != "{":
                        raise _UnprovenPowerShellError
                    close = content.find("}", cursor + 3, cursor + 10)
                    digits = content[cursor + 3 : close] if close >= 0 else ""
                    if not digits or any(c not in "0123456789abcdefABCDEF" for c in digits):
                        raise _UnprovenPowerShellError
                    number = int(digits, 16)
                    if number > 0x10FFFF:
                        raise _UnprovenPowerShellError
                    value.append(chr(number))
                    cursor = close + 1
                else:
                    value.append(_ESCAPES.get(escaped, escaped))
                    cursor += 2
                decoded = True
                continue
            if quote == '"' and content.startswith("$(", cursor):
                dynamic = True
                # Preserve executable subexpressions as independent code. The
                # outer value stays dynamic, never a trusted literal command.
                own(region_start, cursor + 2)
                clause_starts.add(cursor + 2)
                end = code(cursor + 2, depth + 1, closes_parenthesis=True)
                value.append("$VALUE")
                cursor = end
                region_start = end - 1
                continue
            if quote == '"' and content.startswith("${", cursor):
                dynamic = True
                end = variable(cursor, depth)
                value.append(content[cursor:end])
                cursor = end
                continue
            if quote == '"' and character == "$":
                dynamic = True
            value.append(character)
            cursor += 1
        raise _UnprovenPowerShellError

    def code(cursor: int, depth: int, *, closes_parenthesis: bool = False) -> int:
        parentheses = 0
        while cursor < len(content):
            check(cursor, depth)
            character = content[cursor]
            if character == "&" or (
                character == "."
                and (cursor == 0 or content[cursor - 1].isspace() or content[cursor - 1] in ";({")
            ):
                operand = cursor + 1
                gap_comments: list[tuple[int, int]] = []
                while operand < len(content):
                    check(operand, depth)
                    if content[operand].isspace():
                        operand += 1
                    elif content.startswith("<#", operand):
                        end = content.find("#>", operand + 2)
                        if end < 0:
                            raise _UnprovenPowerShellError
                        gap_comments.append((operand, end + 2))
                        operand = end + 2
                    else:
                        break
                opening = operand + int(content.startswith("$(", operand))
                if content[operand : operand + 1] in {"'", '"'}:
                    spans.extend(gap_comments)
                    cursor = quoted(operand, depth + 1, invoked_context=True)
                    continue
                if content[opening : opening + 1] == "(":
                    spans.extend(gap_comments)
                    clause_starts.add(opening + 1)
                    first_string = len(strings)
                    end = code(opening + 1, depth + 1, closes_parenthesis=True)
                    executable_boundary(operand, end)
                    # A proven literal here-string already carries this call's
                    # operands. Dynamic values retain their own invocation.
                    if not any(
                        string.invoked
                        and string.operand_start == end
                        and content[string.start : string.start + 2] in {"@'", '@"'}
                        for string in strings[first_string:]
                    ):
                        invocations.append((operand, end))
                    cursor = end
                    continue
                variable_command = re.match(
                    r"\$(?:[A-Za-z_][A-Za-z0-9_:]*|\{[A-Za-z_][A-Za-z0-9_:]*\})",
                    content[operand : operand + 512],
                )
                if variable_command is not None:
                    spans.extend(gap_comments)
                    end = operand + variable_command.end()
                    suffix = re.match(
                        r"(?:\[[0-9]+\]|\.[A-Za-z_][A-Za-z0-9_]*|[\\/][A-Za-z0-9_.-]+)*",
                        content[end : end + 512],
                    )
                    end += suffix.end()
                    executable_boundary(operand, end)
                    invocations.append((operand, end))
                    cursor = end
                    continue
                if content[operand : operand + 1] == "$":
                    if unresolved_executables is not None:
                        unresolved_executables.append(operand)
                    raise _UnprovenPowerShellError
            if content.startswith("<#", cursor):
                start = cursor
                cursor += 2
                # PowerShell block comments end at the FIRST #>. An apparent
                # nested opener cannot hide code following that terminator.
                while cursor < len(content) and not content.startswith("#>", cursor):
                    check(cursor, depth)
                    cursor += 1
                if cursor == len(content):
                    raise _UnprovenPowerShellError
                cursor += 2
                own(start, cursor)
                clause_starts.add(cursor)
                continue
            if character == "#" and (cursor == 0 or content[cursor - 1] in " \t\r\n;|&(){}"):
                start = cursor
                while cursor < len(content) and content[cursor] not in "\r\n":
                    check(cursor, depth)
                    cursor += 1
                own(start, cursor)
                clause_starts.add(cursor)
                continue
            if character in "'\"":
                cursor = quoted(cursor, depth + 1)
                continue
            if character == "@" and content[cursor + 1 : cursor + 2] in {"'", '"'}:
                cursor = quoted(cursor, depth + 1, here=True)
                continue
            if character == "`":
                if cursor + 1 >= len(content):
                    raise _UnprovenPowerShellError
                # Only a continuation between words can be projected to
                # spaces without changing command identity. Joining an
                # escaped command name remains outside this lexical proof.
                if (cursor == 0 or content[cursor - 1] in " \t") and (
                    content.startswith("`\n", cursor) or content.startswith("`\r\n", cursor)
                ):
                    end = cursor + (3 if content[cursor + 1] == "\r" else 2)
                    continuations.append((cursor, end))
                    own(cursor, end)
                    cursor = end
                    continue
                # Bare escaped words can name native commands (b`ash), not
                # merely data. Until their command identity is reconstructed,
                # keep the original conservative traversal for this source.
                raise _UnprovenPowerShellError
            if content.startswith("${", cursor):
                cursor = variable(cursor, depth)
                continue
            if character in "‘’“”":
                # PowerShell accepts typographic delimiters too. Until this
                # helper proves their exact mixed-quote rules, retain fallback.
                raise _UnprovenPowerShellError
            if character in ";|&({\r\n":
                clause_starts.add(cursor + 1)
            if closes_parenthesis:
                if character == "(":
                    parentheses += 1
                    if parentheses + depth > _MAX_DEPTH:
                        raise _UnprovenPowerShellError
                elif character == ")":
                    if not parentheses:
                        return cursor + 1
                    parentheses -= 1
            cursor += 1
        if closes_parenthesis:
            raise _UnprovenPowerShellError
        return cursor

    check_runtime()
    try:
        code(0, 0)
        check(len(content), 0)
    except _UnprovenPowerShellError:
        return None
    check_runtime()
    if _DYNAMIC_CODE.search(content):
        return None
    spans.sort()
    if any(end > next_start for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False)):
        return None
    return PowerShellSource(
        tuple(start for start, _ in spans),
        tuple(end for _, end in spans),
        tuple(strings),
        tuple(sorted(clause_starts)),
        tuple(continuations),
        tuple(invocations),
    )
