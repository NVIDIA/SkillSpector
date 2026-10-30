# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded lexical ownership for Node assertion template-message delimiters.

This is not a JavaScript parser or a shell-sink/dataflow model. Only a direct
diagnostic argument of one unaliased built-in assertion import may borrow
ownership. Unsupported source keeps its original conservative shell reading.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser

_MAX_SOURCE_CHARS = 1_000_000
_MAX_TOKENS = 100_000
_MAX_NESTING = 128
_LINE_ENDS = "\r\n\u2028\u2029"
_BrowserBinding = tuple[str, int, str]
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


def _nested_template_code(
    content: str, check_runtime: Callable[[], None]
) -> tuple[str, list[tuple[int, int, tuple[int, ...]]], set[int]] | None:
    """Expose template expressions to the existing lexer, retaining exact offsets."""
    if len(content) > _MAX_SOURCE_CHARS:
        return None
    output = list(content)
    templates: list[tuple[int, int, tuple[int, ...]]] = []
    expression_openers: set[int] = set()
    next_check = 0

    def check(position: int) -> None:
        nonlocal next_check
        if position >= next_check:
            check_runtime()
            next_check = position + 4096

    def code(cursor: int, depth: int, *, interpolation: bool = False) -> int | None:
        if depth >= _MAX_NESTING:
            return None
        braces = 0
        while cursor < len(content):
            check(cursor)
            character = content[cursor]
            if character in "'\"":
                quote = character
                cursor += 1
                while cursor < len(content) and content[cursor] != quote:
                    check(cursor)
                    if content[cursor] in _LINE_ENDS:
                        return None
                    cursor += 2 if content[cursor] == "\\" else 1
                if cursor >= len(content):
                    return None
            elif content.startswith("//", cursor):
                while cursor < len(content) and content[cursor] not in _LINE_ENDS:
                    check(cursor)
                    cursor += 1
                continue
            elif content.startswith("/*", cursor):
                cursor += 2
                while cursor < len(content) and not content.startswith("*/", cursor):
                    check(cursor)
                    cursor += 1
                if cursor == len(content):
                    return None
                cursor += 2
                continue
            elif character == "/":
                # Regexp/division ambiguity cannot establish nested ownership.
                return None
            elif character == "`":
                end = template(cursor, depth + 1)
                if end is None:
                    return None
                cursor = end
                continue
            elif character == "{":
                braces += 1
                if braces + depth >= _MAX_NESTING:
                    return None
            elif character == "}":
                if not braces and interpolation:
                    return cursor
                braces -= 1
                if braces < 0:
                    return None
            cursor += 1
        return None if interpolation or braces else cursor

    def template(start: int, depth: int) -> int | None:
        if depth >= _MAX_NESTING or len(templates) >= _MAX_TOKENS:
            return None
        delimiters = [start]
        expression_openers.add(start)
        output[start] = "("
        cursor = start + 1
        while cursor < len(content):
            check(cursor)
            if content[cursor] == "`":
                output[cursor] = ")"
                delimiters.append(cursor)
                templates.append((start, cursor + 1, tuple(delimiters)))
                return cursor + 1
            if content.startswith("${", cursor):
                end = code(cursor + 2, depth + 1, interpolation=True)
                if end is None:
                    return None
                expression_openers.add(cursor)
                output[cursor], output[cursor + 1], output[end] = "(", " ", ")"
                delimiters.extend((cursor, cursor + 1, end))
                cursor = end + 1
                continue
            if content[cursor] == "\\":
                # Decoding an escape needs a separate source/value proof.
                return None
            if content[cursor] not in _LINE_ENDS:
                output[cursor] = " "
            cursor += 1
        return None

    if code(0, 0) is None:
        return None
    check_runtime()
    return "".join(output), templates, expression_openers


def browser_escaped_list_shell_text(content: str, check_runtime: Callable[[], None]) -> str:
    """Own only JSON-backed list values passed through a local escaping formatter.

    This proves explicit source consumers, not that a browser program is inert.
    As with assertion messages, ordinary built-ins are assumed; arbitrary runtime
    getters/monkeypatching are outside this proof. Explicit aliases, mutation,
    shadowing, unknown consumers, and executable expressions never gain ownership.
    Only backticks change. Raw findings and interpolation bytes remain untouched.
    """
    if "JSON.parse" not in content or "replaceAll" not in content or "innerHTML" not in content:
        return content
    nested = _nested_template_code(content, check_runtime)
    if nested is None:
        return content
    code, templates, expression_openers = nested
    parsed = _tokens(code, check_runtime)
    if parsed is None:
        return content
    tokens, pairs = parsed
    values = [code[token.start : token.end] for token in tokens]
    positions = [token.start for token in tokens]
    reverse = {end: start for start, end in pairs.items()}
    template_spans = {
        bisect_left(positions, start): (bisect_left(positions, end), delimiters)
        for start, end, delimiters in templates
    }
    for start in template_spans:
        if start and (
            tokens[start - 1].kind == "identifier"
            and values[start - 1] != "return"
            or values[start - 1] in {")", "]"}
        ):
            # A tag receives a strings array before its interpolation values.
            # Projecting it as an ordinary call would shift parameter sources.
            return content
    fields: dict[int, int] = {}
    for start, (_, delimiters) in template_spans.items():
        for position in delimiters[1:-1:3]:
            fields[bisect_left(positions, position)] = start
    budget = _MAX_TOKENS * 4

    def step(amount: int = 1) -> bool:
        nonlocal budget
        previous = budget
        budget -= amount
        if budget // 256 != previous // 256:
            check_runtime()
        return budget > 0

    def parts(start: int, end: int) -> list[tuple[int, int]]:
        result = []
        begin = cursor = start
        while cursor < end and step():
            if values[cursor] == ",":
                result.append((begin, cursor))
                begin = cursor + 1
            cursor = pairs.get(cursor, cursor) + 1
        if begin < end:
            result.append((begin, end))
        return result

    def arguments(opening: int) -> list[tuple[int, int]]:
        return parts(opening + 1, pairs[opening])

    enclosing: list[int | None] = []
    blocks: list[tuple[int, int]] = []
    stack: list[int] = []
    block_stack: list[int] = []
    for index, value in enumerate(values):
        if not step():
            return content
        while stack and pairs[stack[-1]] == index:
            opening = stack.pop()
            if values[opening] == "{":
                block_stack.pop()
        enclosing.append(stack[-1] if stack else None)
        blocks.append(
            (block_stack[-1], pairs[block_stack[-1]]) if block_stack else (0, len(tokens))
        )
        if index in pairs:
            stack.append(index)
            if value == "{":
                block_stack.append(index)

    functions: dict[str, tuple[int, int, int, list[str]]] = {}
    constants: dict[str, list[tuple[int, int, int, int, int]]] = {}
    arrows: dict[int, tuple[int, int, int, list[str]]] = {}
    native_methods = {
        "map",
        "filter",
        "find",
        "some",
        "flatMap",
        "forEach",
        "join",
        "replaceAll",
        "trim",
        "toLowerCase",
        "includes",
        "getElementById",
        "querySelectorAll",
        "addEventListener",
    }
    protected = {"String", "JSON", "document", "Object"}
    forbidden = {"constructor", "prototype", "__proto__", "globalThis", "window", "self", "with"}
    for index, value in enumerate(values):
        if not step():
            return content
        if value in forbidden:
            return content
        if value in protected:
            expected = {"String": ["("], "JSON": ["."], "document": ["."], "Object": ["."]}[value]
            if values[index + 1 : index + 1 + len(expected)] != expected or (
                index and values[index - 1] in {"function", ".", "?.", "const", "let", "var"}
            ):
                return content
        if value == "JSON" and values[index + 2 : index + 4] not in (
            ["parse", "("],
            ["stringify", "("],
        ):
            return content
        if value == "document" and values[index + 2 : index + 4] != ["getElementById", "("]:
            return content
        if value == "Object" and values[index + 2 : index + 4] != ["entries", "("]:
            return content
        if value in native_methods and index and values[index - 1] in {".", "?."}:
            if values[index + 1 : index + 2] != ["("]:
                return content
        if values[index + 1 : index + 2] == [":"] and (
            value in native_methods
            or tokens[index].kind == "string"
            and value[1:-1] in native_methods
        ):
            return content
        if value == "]" and values[index + 1 : index + 2] in ([":"], ["="]):
            return content
        if value == "function":
            opening = index + 2
            closing = pairs.get(opening)
            body = -1 if closing is None else closing + 1
            end = pairs.get(body)
            if (
                tokens[index].depth
                or end is None
                or values[body] != "{"
                or tokens[index + 1].kind != "identifier"
                or values[index + 1] in functions
                or len(functions) >= _MAX_NESTING
            ):
                return content
            params = arguments(opening)
            names = []
            for begin, stop in params:
                if (
                    tokens[begin].kind != "identifier"
                    or values[begin] in protected
                    or values[begin] in names
                ):
                    return content
                if stop != begin + 1 and not (
                    stop == begin + 3
                    and values[begin + 1] == "="
                    and tokens[begin + 2].kind == "string"
                ):
                    return content
                names.append(values[begin])
            functions[values[index + 1]] = index + 1, body, end, names
        if value == "const":
            if (
                index + 2 >= len(tokens)
                or tokens[index + 1].kind != "identifier"
                or values[index + 2] != "="
            ):
                return content
            stop = index + 3
            while stop < len(tokens) and values[stop] != ";" and step():
                stop = pairs.get(stop, stop) + 1
            if stop == len(tokens):
                return content
            constants.setdefault(values[index + 1], []).append(
                (index + 1, index + 3, stop, *blocks[index])
            )
        if value == "=>":
            if index and values[index - 1] == ")":
                opening = reverse[index - 1]
                begin, stop = opening + 1, index - 1
                call = enclosing[opening]
            elif index and tokens[index - 1].kind == "identifier":
                begin, stop = index - 1, index
                call = enclosing[begin]
            else:
                return content
            if begin < stop and values[begin] in {"{", "["} and pairs.get(begin) == stop - 1:
                begin, stop = begin + 1, stop - 1
            names = []
            for lower, upper in parts(begin, stop):
                if (
                    upper != lower + 1
                    or tokens[lower].kind != "identifier"
                    or values[lower] in protected
                    or values[lower] in names
                ):
                    return content
                names.append(values[lower])
            body = index + 1
            end = pairs.get(body, body) + 1
            if body not in pairs:
                while end < len(tokens) and values[end] not in {",", ";", ")", "]", "}"} and step():
                    end = pairs.get(end, end) + 1
            if call is not None:
                arrows[index] = call, body, end, names

    arrows_by_call: dict[int, list[tuple[int, int, int, list[str]]]] = {}
    arrows_by_body: dict[tuple[int, int], tuple[int, int, int, list[str]]] = {}
    arrows_by_name: dict[str, list[tuple[int, int, int, list[str]]]] = {}
    for row in arrows.values():
        arrows_by_call.setdefault(row[0], []).append(row)
        arrows_by_body[row[1:3]] = row
        for name in row[3]:
            arrows_by_name.setdefault(name, []).append(row)

    def literal(index: int) -> str | None:
        value = values[index]
        return value[1:-1] if tokens[index].kind == "string" and "\\" not in value else None

    # Verify the actual text escaping operation, with no aliases or extra body.
    escapes: set[str] = set()
    for name, (_, body, end, params) in functions.items():
        if len(params) != 1:
            continue
        sequence = values[body + 1 : end]
        prefix = ["return", "String", "(", params[0], "??"]
        cursor = body + 1
        if sequence[:5] != prefix or literal(cursor + 5) != "" or values[cursor + 6] != ")":
            continue
        cursor += 7
        for before, after in [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;")]:
            if (
                values[cursor : cursor + 3] != [".", "replaceAll", "("]
                or literal(cursor + 3) != before
                or values[cursor + 4 : cursor + 5] != [","]
                or literal(cursor + 5) != after
                or values[cursor + 6 : cursor + 7] != [")"]
            ):
                break
            cursor += 7
        else:
            if values[cursor:end] == [";"]:
                escapes.add(name)

    def source(start: int, end: int) -> str:
        return content[tokens[start].start : tokens[end - 1].end]

    def template_fields(start: int) -> list[tuple[int, int]]:
        delimiters = template_spans[start][1]
        return [
            (bisect_left(positions, p) + 1, pairs[bisect_left(positions, p)])
            for p in delimiters[1:-1:3]
        ]

    def text_slot(template: int, field: int, *, inert: bool = False) -> bool:
        """The selected HTML fragment occurs in ordinary literal markup text."""
        marker = "SKILLSPECTOR_SELECTED_FRAGMENT"
        if marker in content:
            return False
        _, delimiters = template_spans[template]
        if not step(1 + (delimiters[-1] - delimiters[0]) // 32):
            return False
        chunks = []
        cursor = delimiters[0] + 1
        for dollar, _, close in zip(
            delimiters[1:-1:3], delimiters[2:-1:3], delimiters[3:-1:3], strict=True
        ):
            chunks.append(content[cursor:dollar])
            chunks.append(marker if bisect_left(positions, dollar) == field else "VALUE")
            cursor = close + 1
        chunks.append(content[cursor : delimiters[-1]])

        class Slot(HTMLParser):
            valid = True
            found = False
            context = ""

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if inert and (
                    tag not in {"div", "span", "ul", "li", "p", "strong", "em", "b", "code"}
                    or any(name not in {"class", "id", "title"} for name, _ in attrs)
                ):
                    self.valid = False
                if tag in {"script", "style", "iframe", "object", "embed", "textarea", "title"}:
                    self.context = tag
                if any(value is not None and marker in value for _, value in attrs):
                    self.valid = False

            def handle_endtag(self, tag: str) -> None:
                if tag == self.context:
                    self.context = ""

            def handle_data(self, data: str) -> None:
                if marker in data:
                    self.found = True
                    self.valid = self.valid and not self.context

        parser = Slot(convert_charrefs=False)
        check_runtime()
        parser.feed("".join(chunks))
        parser.close()
        check_runtime()
        return parser.valid and parser.found

    # The formatter has one escaped empty branch and one map/escape/join branch.
    formatters: set[str] = set()
    for name, (_, body, end, params) in functions.items():
        if len(params) != 2:
            continue
        items, empty = params
        for escape in escapes:
            if not step(end - body + 1):
                return content
            compact = re.sub(r"\s+", "", source(body + 1, end))
            head = f"if(!{items}||!{items}.length){{return"
            if not compact.startswith(head):
                continue
            returns = [index for index in range(body + 1, end) if values[index] == "return"]
            if len(returns) != 2:
                continue
            first, second = (index + 1 for index in returns)
            if first not in template_spans or second not in template_spans:
                continue
            first_end, _ = template_spans[first]
            second_end, _ = template_spans[second]
            if values[first_end:second] != [";", "}", "return"] or values[second_end:end] != [";"]:
                continue
            a, b = template_fields(first), template_fields(second)
            if len(a) != 1 or len(b) != 1:
                continue
            if values[a[0][0] : a[0][1]] != [escape, "(", empty, ")"] or not text_slot(
                first, a[0][0] - 1, inert=True
            ):
                continue
            begin, stop = b[0]
            if values[begin : begin + 4] != [items, ".", "map", "("]:
                continue
            call = begin + 3
            args = arguments(call)
            if len(args) != 1:
                continue
            matching = arrows_by_call.get(call, [])
            if len(matching) != 1:
                continue
            _, inner, inner_end, names = matching[0]
            if (
                len(names) != 1
                or inner not in template_spans
                or template_spans[inner][0] != inner_end
            ):
                continue
            item = names[0]
            field = template_fields(inner)
            if len(field) != 1 or not text_slot(inner, field[0][0] - 1, inert=True):
                continue
            expression = values[field[0][0] : field[0][1]]
            simple = [escape, "(", item, ")"]
            fallback = [
                escape,
                "(",
                "typeof",
                item,
                "===",
                "'string'",
                "?",
                item,
                ":",
                "JSON",
                ".",
                "stringify",
                "(",
                item,
                ")",
                ")",
            ]
            if expression != simple and expression != fallback:
                continue
            close = pairs[call] + 1
            if (
                values[close : close + 3] != [".", "join", "("]
                or literal(close + 3) != ""
                or values[close + 4 : stop] != [")"]
            ):
                continue
            if text_slot(second, begin - 1, inert=True):
                formatters.add(name)
    if not formatters:
        return content

    # A closed callable vocabulary rules out explicit escapes of JSON values
    # into arbitrary code. Native method spelling alone never proves a selected
    # receiver: ordinary() below separately traces that receiver back to JSON.
    callbacks: set[int] = set()
    function_parameter_openers = {declaration + 1 for declaration, _, _, _ in functions.values()}
    defaults = {
        a + 1
        for opening in function_parameter_openers
        for a, b in arguments(opening)
        if b == a + 3 and values[a + 1] == "="
    }
    declaration_indices = {row[0] for rows in constants.values() for row in rows}
    mutated: set[str] = set()
    member_writes: list[int] = []
    for index, value in enumerate(values):
        if not step():
            return content
        if value == "=" and index - 1 not in declaration_indices and index not in defaults:
            if index < 3 or values[index - 2] != ".":
                return content
            member_writes.append(index)
            if tokens[index - 3].kind == "identifier":
                mutated.add(values[index - 3])
        if (
            value.endswith("=") and value not in {"=", "==", "===", "!=", "!==", "<=", ">=", "=>"}
        ) or value in {"++", "--", "delete", "new", "class", "import", "export", "let", "var"}:
            return content
        if value != "(" or tokens[index].start in expression_openers:
            continue
        closing = pairs[index]
        if index in function_parameter_openers or values[closing + 1 : closing + 2] == ["=>"]:
            continue
        previous = values[index - 1] if index else ""
        if previous == "if":
            continue
        if index and tokens[index - 1].kind == "identifier":
            if index > 1 and values[index - 2] == ".":
                receiver = values[index - 3] if index > 2 else ""
                if not (
                    previous in native_methods
                    or receiver == "JSON"
                    and previous in {"parse", "stringify"}
                    or receiver == "Object"
                    and previous == "entries"
                ):
                    return content
                args = arguments(index)
                if previous == "replaceAll" and (
                    len(args) != 2 or any(b != a + 1 or literal(a) is None for a, b in args)
                ):
                    return content
                if previous in {
                    "map",
                    "filter",
                    "find",
                    "some",
                    "flatMap",
                    "forEach",
                    "addEventListener",
                }:
                    slot = 1 if previous == "addEventListener" else 0
                    if len(args) <= slot:
                        return content
                    a, b = args[slot]
                    if b == a + 1 and values[a] in functions:
                        callbacks.add(a)
                    elif index not in arrows_by_call:
                        return content
            elif previous not in functions and previous != "String":
                return content
        elif previous in {")", "]", "?."}:
            return content
    for index, value in enumerate(values):
        if not step():
            return content
        if value in functions and index != functions[value][0]:
            if index and values[index - 1] in {".", "?."}:
                return content
            if values[index + 1 : index + 2] != ["("] and index not in callbacks:
                return content

    selected_names = escapes | formatters
    for index, value in enumerate(values):
        if not step():
            return content
        if value in selected_names:
            if index == functions[value][0]:
                continue
            if value in escapes and index in callbacks:
                continue
            if (
                values[index + 1 : index + 2] != ["("]
                or index
                and values[index - 1] in {".", "?.", "function", "new"}
            ):
                return content
        if (
            tokens[index].kind == "string"
            and literal(index) in selected_names | protected | native_methods
            and index >= 2
            and values[index - 1] == "["
            and values[index + 1 : index + 2] == ["]"]
            and (tokens[index - 2].kind == "identifier" or values[index - 2] in {")", "]", "}"})
        ):
            # Protect computed member references, not text arguments such as a
            # DOM element whose ID happens to be "filter" or "map".
            return content

    def receiver_start(end: int, depth: int = 0) -> int | None:
        if depth >= _MAX_NESTING or end <= 0 or not step():
            return None
        last = end - 1
        if last in reverse:
            opening = reverse[last]
            if opening and (
                tokens[opening - 1].kind == "identifier" or values[opening - 1] in {")", "]"}
            ):
                return receiver_start(opening, depth + 1)
            return opening
        if tokens[last].kind not in {"identifier", "string", "number"}:
            return None
        if last >= 2 and values[last - 1] in {".", "?."}:
            return receiver_start(last - 1, depth + 1)
        return last

    # Resolve the nearest explicit binding, including callback/function
    # parameters. A same-spelled outer DOM or JSON constant is not that value.
    binding_rows: dict[_BrowserBinding, tuple[int, int, int, int]] = {}
    bindings_by_name: dict[str, list[_BrowserBinding]] = {}
    for name, rows in constants.items():
        for declaration, a, b, lower, upper in rows:
            key = ("const", declaration, name)
            binding_rows[key] = a, b, lower, upper
            bindings_by_name.setdefault(name, []).append(key)
    for declaration, body, stop, params in functions.values():
        for slot, parameter in enumerate(params):
            key = ("function", declaration, parameter)
            binding_rows[key] = slot, declaration, body + 1, stop
            bindings_by_name.setdefault(parameter, []).append(key)
    for call, body, stop, params in arrows.values():
        for slot, parameter in enumerate(params):
            key = ("arrow", body, parameter)
            binding_rows[key] = slot, call, body, stop
            bindings_by_name.setdefault(parameter, []).append(key)

    def binding(name: str, index: int) -> _BrowserBinding | None:
        owner = None
        width = len(tokens) + 1
        for key in bindings_by_name.get(name, []):
            if not step():
                return None
            _, _, lower, upper = binding_rows[key]
            if lower <= index < upper:
                size = upper - lower
                if size == width:
                    return None
                if size < width:
                    owner, width = key, size
        return owner

    calls_by_function: dict[int, list[int]] = {}
    for index, value in enumerate(values):
        if not step():
            return content
        if value in functions and index != functions[value][0]:
            declaration = functions[value][0]
            if binding(value, index) is not None:
                return content
            calls_by_function.setdefault(declaration, []).append(index)

    resolving: set[tuple[str, int, str]] = set()
    dependencies: set[_BrowserBinding] = set()

    def ordinary(start: int, end: int, env: frozenset[str] = frozenset(), depth: int = 0) -> bool:
        if start >= end or depth >= 32 or not step():
            return False
        if start in template_spans and template_spans[start][0] == end:
            return all(ordinary(a, b, env, depth + 1) for a, b in template_fields(start))
        if values[start] == "(" and pairs.get(start) == end - 1:
            return ordinary(start + 1, end - 1, env, depth + 1)
        cursor = start
        while cursor < end:
            if not step():
                return False
            if values[cursor] in {"===", "!==", "==", "!=", "&&", "||", "??"}:
                return ordinary(start, cursor, env, depth + 1) and ordinary(
                    cursor + 1, end, env, depth + 1
                )
            cursor = pairs.get(cursor, cursor) + 1
        if values[start] in {"!", "typeof"}:
            return ordinary(start + 1, end, env, depth + 1)
        cursor = start + 1
        value = values[start]
        if value in {"{", "["} and start in pairs:
            closing = pairs[start]
            if closing >= end:
                return False
            for a, b in parts(start + 1, closing):
                if values[start] == "{" and a + 1 < b and values[a + 1] == ":":
                    if tokens[a].kind != "identifier" or values[a] in forbidden | native_methods:
                        return False
                    a += 2
                elif value == "[" and values[a] == "...":
                    a += 1
                if not ordinary(a, b, env, depth + 1):
                    return False
            cursor = closing + 1
        elif value == "Object" and values[start + 1 : start + 4] == [".", "entries", "("]:
            opening = start + 3
            args = arguments(opening)
            if len(args) != 1 or not ordinary(*args[0], env, depth + 1):
                return False
            cursor = pairs[opening] + 1
        elif value == "JSON" and values[start + 1 : start + 4] == [".", "parse", "("]:
            opening = start + 3
            args = arguments(opening)
            if len(args) != 1:
                return False
            a, b = args[0]
            if (
                values[a : a + 4] != ["document", ".", "getElementById", "("]
                or literal(a + 4) is None
                or values[a + 5 : b] != [")", ".", "textContent"]
            ):
                return False
            cursor = pairs[opening] + 1
        elif start + 1 < end and values[start + 1] == "(" and value in functions:
            declaration, body, stop, params = functions[value]
            opening = start + 1
            args = arguments(opening)
            key = "call", declaration, value
            if (
                key in resolving
                or len(args) != len(params)
                or values[body + 1 : body + 2] != ["return"]
                or values[stop - 1 : stop] != [";"]
            ):
                return False
            if not all(ordinary(a, b, env, depth + 1) for a, b in args):
                return False
            resolving.add(key)
            valid = ordinary(body + 2, stop - 1, frozenset(params), depth + 1)
            resolving.remove(key)
            if not valid:
                return False
            cursor = pairs[opening] + 1
        elif (
            tokens[start].kind in {"string", "number"}
            or value in {"true", "false", "null", "undefined"}
            or value in env
        ):
            pass
        elif tokens[start].kind == "identifier":
            if value in mutated:
                return False
            key = binding(value, start)
            if key is None or key in resolving:
                return False
            dependencies.add(key)
            a, b, _, _ = binding_rows[key]
            resolving.add(key)
            if key[0] == "const":
                valid = ordinary(a, b, frozenset(), depth + 1)
            elif key[0] == "arrow":
                call = b
                begin = receiver_start(call - 2)
                valid = (
                    begin is not None
                    and values[call - 1] in {"map", "filter", "find", "some", "flatMap", "forEach"}
                    and ordinary(begin, call - 2, frozenset(), depth + 1)
                )
            else:
                # Every explicit invocation must provide an ordinary argument.
                # A recursive unchanged forwarding edge adds no new source.
                valid = True
                sources = 0
                for call in calls_by_function.get(key[1], []):
                    if not step():
                        valid = False
                        break
                    if call in callbacks:
                        opening = enclosing[call]
                        begin = None if opening is None else receiver_start(opening - 2)
                        good = (
                            a == 0
                            and opening is not None
                            and begin is not None
                            and values[opening - 1]
                            in {"map", "filter", "find", "some", "flatMap", "forEach"}
                            and ordinary(begin, opening - 2, frozenset(), depth + 1)
                        )
                    else:
                        args = arguments(call + 1)
                        if a >= len(args):
                            params = arguments(key[1] + 1)
                            lower, upper = params[a]
                            good = upper == lower + 3 and literal(lower + 2) is not None
                        else:
                            lower, upper = args[a]
                            if upper == lower + 1 and binding(values[lower], lower) == key:
                                continue
                            good = ordinary(lower, upper, frozenset(), depth + 1)
                    sources += 1
                    if not good:
                        valid = False
                        break
                valid = valid and sources > 0
            resolving.remove(key)
            if not valid:
                return False
        else:
            return False
        while cursor < end and step():
            if (
                values[cursor] == "["
                and pairs.get(cursor) == cursor + 2
                and tokens[cursor + 1].kind == "number"
            ):
                cursor += 3
                continue
            if values[cursor] not in {".", "?."} or cursor + 1 >= end:
                return False
            member = values[cursor + 1]
            if tokens[cursor + 1].kind != "identifier" or member in forbidden:
                return False
            cursor += 2
            if cursor < end and values[cursor] == "(":
                if member not in {
                    "map",
                    "filter",
                    "find",
                    "some",
                    "flatMap",
                    "join",
                    "trim",
                    "toLowerCase",
                }:
                    return False
                args = arguments(cursor)
                if member in {"trim", "toLowerCase"}:
                    if args:
                        return False
                elif member == "join":
                    if len(args) > 1 or any(b != a + 1 or literal(a) is None for a, b in args):
                        return False
                else:
                    matches = arrows_by_call.get(cursor, [])
                    if len(args) != 1:
                        return False
                    a, b = args[0]
                    if member in {"filter", "find", "some"} and b == a + 1 and a in callbacks:
                        # These native operations retain existing members;
                        # dependency closure separately audits the predicate.
                        cursor = pairs[cursor] + 1
                        continue
                    if len(matches) != 1:
                        return False
                    _, body, stop, names = matches[0]
                    if member in {"filter", "find", "some"}:
                        # A simple predicate can select only existing JSON
                        # members. Its values need not become part of output.
                        if not step(stop - body + 1) or any(
                            values[i] not in {".", "?.", "!", "===", "!==", "==", "!=", "&&", "||"}
                            and tokens[i].kind not in {"identifier", "string", "number"}
                            for i in range(body, stop)
                        ):
                            return False
                    elif not ordinary(body, stop, env | frozenset(names), depth + 1):
                        return False
                cursor = pairs[cursor] + 1
        return cursor == end

    def display(start: int, end: int) -> bool:
        escaped = False
        for _ in range(_MAX_NESTING):
            if not step():
                return False
            if values[end : end + 3] == [".", "join", "("]:
                args = arguments(end + 2)
                if len(args) != 1 or args[0][1] != args[0][0] + 1 or literal(args[0][0]) != "":
                    return False
                end = pairs[end + 2] + 1
                continue
            parent = enclosing[start]
            if parent in fields and start == parent + 1 and pairs[parent] == end:
                template = fields[parent]
                if not text_slot(template, parent):
                    return False
                start, end = template, template_spans[template][0]
                continue
            arrow = arrows_by_body.get((start, end))
            if arrow is not None:
                call = arrow[0]
                begin = receiver_start(call - 2)
                if values[call - 1] != "map" or begin is None or not ordinary(begin, call - 2):
                    return False
                start, end = begin, pairs[call] + 1
                continue
            if parent is not None and parent and values[parent - 1] in formatters:
                args = arguments(parent)
                if (
                    not args
                    or args[0] != (start, end)
                    or len(args) > 2
                    or any(b != a + 1 or literal(a) is None for a, b in args[1:])
                ):
                    return False
                escaped = True
                start, end = parent - 1, pairs[parent] + 1
                continue
            if (
                start >= 4
                and values[start - 3 : start] == [".", "innerHTML", "="]
                and values[end : end + 1] == [";"]
            ):
                return escaped and dom(start - 4, start - 3) == "element"
            return False
        return False

    def dom(start: int, end: int, depth: int = 0) -> str | None:
        if depth >= 32 or start >= end or not step():
            return None
        cursor = start + 1
        kind = None
        if values[start : start + 4] == ["document", ".", "getElementById", "("]:
            if literal(start + 4) is None or values[start + 5 : start + 6] != [")"]:
                return None
            cursor, kind = start + 6, "element"
        elif tokens[start].kind == "identifier":
            key = binding(values[start], start)
            if key is None:
                return None
            dependencies.add(key)
            a, b, _, _ = binding_rows[key]
            if key[0] == "const":
                kind = dom(a, b, depth + 1)
            elif key[0] == "arrow" and a == 0 and values[b - 1] == "forEach":
                begin = receiver_start(b - 2)
                if begin is not None and dom(begin, b - 2, depth + 1) == "list":
                    kind = "element"
        if kind is None:
            return None
        if cursor < end and values[cursor : cursor + 3] == [".", "querySelectorAll", "("]:
            args = arguments(cursor + 2)
            if (
                kind != "element"
                or len(args) != 1
                or args[0][1] != args[0][0] + 1
                or literal(args[0][0]) is None
            ):
                return None
            cursor, kind = pairs[cursor + 2] + 1, "list"
        return kind if cursor == end else None

    member_values: list[tuple[_BrowserBinding, int, int]] = []
    for index in member_writes:
        begin = receiver_start(index - 2)
        if begin is None:
            return content
        if (
            values[index - 1] in {"innerHTML", "textContent", "value"}
            and dom(begin, index - 2) == "element"
        ):
            continue
        if begin != index - 3:
            return content
        owner = binding(values[begin], begin)
        if owner is None or owner[0] != "const":
            return content
        a, b, _, _ = binding_rows[owner]
        if values[a] != "{" or pairs.get(a) != b - 1:
            return content
        stop = index + 1
        while stop < len(tokens) and values[stop] != ";" and step():
            stop = pairs.get(stop, stop) + 1
        if stop == len(tokens) or budget <= 0:
            return content
        member_values.append((owner, index + 1, stop))

    for index in callbacks:
        if values[index] not in escapes:
            continue
        call = enclosing[index]
        if call is None:
            return content
        begin = receiver_start(call - 2)
        if begin is None or values[call - 1] != "map" or not ordinary(begin, call - 2):
            return content
    output = list(content)
    changed = False
    for start, (end, delimiters) in template_spans.items():
        if not step():
            return content
        expressions = template_fields(start)
        if not expressions or any(
            _TEMPLATE_VALUE.fullmatch(source(a, b)) is None for a, b in expressions
        ):
            continue
        if not display(start, end) or not all(ordinary(a, b) for a, b in expressions):
            continue
        output[delimiters[0]] = output[delimiters[-1]] = " "
        changed = True

    # Follow explicit aliases and helper returns that carry a proven value.
    # Unknown native-looking receivers must not acquire it through a callback.
    def dependency(index: int) -> bool:
        if (
            tokens[index].kind != "identifier"
            or index
            and values[index - 1] in {".", "?."}
            or values[index + 1 : index + 2] == [":"]
        ):
            return False
        if values[index] == "document":
            # A second literal lookup can reacquire the selected element even
            # without referring to its local constant binding.
            return True
        key = binding(values[index], index)
        if key is None and values[index] in functions:
            key = ("call", functions[values[index]][0], values[index])
        return key in dependencies

    def carries(start: int, end: int) -> bool:
        return step(end - start + 1) and any(dependency(i) for i in range(start, end))

    for _ in range(_MAX_NESTING):
        before = len(dependencies)
        for name, rows in constants.items():
            for declaration, a, b, _, _ in rows:
                if carries(a, b):
                    dependencies.add(("const", declaration, name))
        for owner, a, b in member_values:
            if carries(a, b):
                dependencies.add(owner)
        for name, (declaration, a, b, params) in functions.items():
            if carries(a, b):
                dependencies.add(("call", declaration, name))
            for call in calls_by_function.get(declaration, []):
                if not step():
                    return content
                if call in callbacks:
                    opening = enclosing[call]
                    begin = None if opening is None else receiver_start(opening - 2)
                    if begin is not None and carries(begin, opening - 2):
                        dependencies.update(
                            ("function", declaration, parameter) for parameter in params
                        )
                else:
                    for parameter, (lower, upper) in zip(params, arguments(call + 1), strict=False):
                        if carries(lower, upper):
                            dependencies.add(("function", declaration, parameter))
        for call, body, _, params in arrows.values():
            begin = receiver_start(call - 2) if call >= 3 and values[call - 2] == "." else None
            if begin is not None and carries(begin, call - 2):
                dependencies.update(("arrow", body, parameter) for parameter in params)
        if budget <= 0:
            return content
        if len(dependencies) == before:
            break
    else:
        return content
    for opening in pairs:
        if values[opening] != "(" or opening < 3 or values[opening - 2] != ".":
            continue
        member = values[opening - 1]
        if member not in native_methods:
            continue
        closing = pairs[opening]
        if not step(closing - opening + 1):
            return content
        if not any(dependency(i) for i in range(opening + 1, closing)):
            continue
        begin = receiver_start(opening - 2)
        if begin is None or not (
            ordinary(begin, opening - 2) or dom(begin, opening - 2) is not None
        ):
            return content
    check_runtime()
    return "".join(output) if changed and budget > 0 else content
