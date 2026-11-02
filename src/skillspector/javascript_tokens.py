# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded JavaScript ownership for shell-completeness checks.

Assertion messages require a proved receiver. General template ownership also
requires a complete syntax tree and retains conservative shell-execution checks.
Original source bytes remain the input to security matching.
"""

from __future__ import annotations

import re
from bisect import bisect_left
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
        "function",
        "execa",
        "spawn",
        "spawnfile",
        "spawnfilesync",
        "spawnsync",
        "system",
        "run",
    }
)
_SHELL_EXECUTION_MODULES = frozenset({"child_process", "execa", "shelljs", "zx"})


def _javascript_parse_text(
    content: str, check_runtime: Callable[[], None]
) -> tuple[str, tuple[int, ...], frozenset[tuple[int, int]]] | None:
    """Adapt supported host syntax without changing any source offsets.

    Esprima predates hashbangs and optional chaining. Removing optional guards
    only overapproximates the calls inspected here; this text is never executed
    or used to resolve values. Token ranges prevent edits to strings or regexes.
    """
    projected = content
    if content.startswith("#!"):
        if _NODE_SHEBANG.match(content) is None:
            return None
        projected = "//" + content[2:]
    if "?." not in projected:
        return projected, (), frozenset()

    visited = 0

    def bounded_token(token: Any) -> Any:
        nonlocal visited
        visited += 1
        if visited > _MAX_TOKENS:
            raise ValueError("JavaScript token limit")
        if visited % 256 == 0:
            check_runtime()
        return token

    try:
        tokens = esprima.tokenize(projected, {"range": True}, bounded_token)
    except (esprima.Error, RecursionError, ValueError, TypeError, IndexError):
        return None
    check_runtime()
    output = list(projected)
    optional_positions: list[int] = []
    parentheses: list[int] = []
    parenthesized: set[tuple[int, int]] = set()
    for index, token in enumerate(tokens):
        if index % 256 == 0:
            check_runtime()
        if token.type != "Punctuator":
            continue
        if token.value == "(":
            parentheses.append(index)
        elif token.value == ")" and parentheses:
            opener = parentheses.pop()
            if opener + 1 < index:
                parenthesized.add((tokens[opener + 1].range[0], tokens[index - 1].range[1]))
    for index, token in enumerate(tokens[:-2]):
        if index % 256 == 0:
            check_runtime()
        following, member = tokens[index + 1 : index + 3]
        if token.type != "Punctuator" or token.value != "?":
            continue
        start, end = token.range
        if (
            following.type != "Punctuator"
            or following.value != "."
            or following.range != [end, end + 1]
            or projected[start : end + 1] != "?."
        ):
            continue
        if member.type == "Punctuator" and member.value in {"[", "("}:
            output[start : end + 1] = "  "
        elif member.type in {"Identifier", "Keyword"}:
            output[start] = " "
        else:
            return None
        optional_positions.append(start)
    return "".join(output), tuple(optional_positions), frozenset(parenthesized)


def _invalid_optional_context(
    program: Any,
    positions: tuple[int, ...],
    parenthesized: frozenset[tuple[int, int]],
    tick: Callable[[], None],
) -> bool:
    """Optional chains cannot be assignment targets, constructors or tags."""
    if not positions:
        return False

    def contains(start: int, end: int) -> bool:
        index = bisect_left(positions, start)
        return index < len(positions) and positions[index] < end

    def optional_chain(target: Any) -> bool:
        # Computed indices and call arguments are independent expressions.
        # Parentheses also end a chain: (a?.b).c is a valid assignment target.
        while target.type in {"MemberExpression", "CallExpression"}:
            tick()
            if target.type == "MemberExpression":
                prefix = target.object
                end = target.property.range[0]
            else:
                prefix = target.callee
                end = target.arguments[0].range[0] if target.arguments else target.range[1]
            if contains(prefix.range[1], end):
                return True
            if tuple(prefix.range) in parenthesized:
                return False
            target = prefix
        return False

    for node in _walk_nodes(program, tick):
        target = None
        if node.type == "AssignmentExpression":
            target = node.left
        elif node.type == "UpdateExpression":
            target = node.argument
        elif node.type == "NewExpression":
            target = node.callee
            end = node.arguments[0].range[0] if node.arguments else node.range[1]
            if contains(target.range[1], end):
                return True
            if tuple(target.range) in parenthesized:
                continue
        elif node.type == "TaggedTemplateExpression":
            target = node.tag
            if tuple(target.range) in parenthesized:
                continue
        if target is not None and optional_chain(target):
            return True
    return False


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
    prepared = _javascript_parse_text(content, check_runtime)
    if prepared is None:
        return None
    parse_text, optional_positions, parenthesized = prepared
    options = {"range": True}
    try:
        try:
            program = esprima.parseScript(parse_text, options)
        except esprima.Error:
            program = esprima.parseModule(parse_text, options)
    except (esprima.Error, RecursionError, ValueError, TypeError):
        return None
    check_runtime()
    if _invalid_optional_context(program, optional_positions, parenthesized, check_runtime):
        return None

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

    if _has_unproven_assertion_template_call(content, program, templates, check_runtime):
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
    global_objects = {"globalThis", "global", "window", "self"}
    computed_callables: set[str] = set()
    aliases: dict[str, list[str]] = {}

    def unknown_member(node: Any) -> bool:
        return (
            getattr(node, "type", None) == "MemberExpression"
            and node.computed
            and getattr(node.property, "type", None) != "Literal"
        )

    for node in _walk_nodes(program, check_runtime):
        if node.type == "VariableDeclarator":
            target, value = node.id, node.init
        elif node.type == "AssignmentExpression" and node.operator == "=":
            target, value = node.left, node.right
        else:
            continue
        if getattr(target, "type", None) != "Identifier":
            continue
        if getattr(value, "type", None) == "Identifier":
            aliases.setdefault(value.name, []).append(target.name)
        elif unknown_member(value):
            computed_callables.add(target.name)
    # Overapproximate aliases across scopes and assignments. A name may have
    # several possible values; none may hide a possible evaluator selection.
    for names in (global_objects, computed_callables):
        pending = list(names)
        while pending:
            check_runtime()
            for alias in aliases.get(pending.pop(), ()):
                if alias not in names:
                    names.add(alias)
                    pending.append(alias)

    def unproven_callable(callee: Any) -> bool:
        while getattr(callee, "type", None) == "MemberExpression":
            check_runtime()
            if unknown_member(callee):
                return True
            callee = callee.object
        return getattr(callee, "type", None) == "Identifier" and callee.name in computed_callables

    literal_keys: set[int] = set()
    for node in _walk_nodes(program, check_runtime):
        if node.type == "ObjectExpression":
            # Object metadata keys are names, not references to evaluators.
            for prop in node.properties:
                if prop.type == "Property" and not prop.computed and not prop.shorthand:
                    literal_keys.add(id(prop.key))
        if node.type == "ObjectPattern":
            for prop in node.properties:
                if prop.type == "Property" and (
                    getattr(prop.key, "value", None) in {"eval", "Function"}
                    or prop.computed and getattr(prop.key, "type", None) != "Literal"
                ):
                    return True
        if (
            node.type == "Identifier"
            and node.name in {"eval", "Function"}
            and id(node) not in literal_keys
        ):
            # Aliasing an evaluator does not make its input ordinary text.
            return True
        if (
            node.type == "MemberExpression"
            and node.computed
            and getattr(node.property, "type", None) == "Literal"
            and getattr(node.property, "value", None) in {"eval", "Function"}
        ):
            return True
        if unknown_member(node):
            receiver = node.object
            while getattr(receiver, "type", None) == "MemberExpression":
                check_runtime()
                receiver = receiver.object
            if getattr(receiver, "type", None) == "Identifier" and receiver.name in global_objects:
                return True
        if node.type == "ImportDeclaration" and _is_shell_execution_module(
            getattr(node.source, "value", "")
        ):
            return True
        if node.type in {"CallExpression", "NewExpression"}:
            if unproven_callable(node.callee):
                return True
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
            if unproven_callable(tag):
                return True
            if _is_shell_execution_callee(tag) or (tag.type == "Identifier" and tag.name == "$"):
                return True
    return False


def _has_unproven_assertion_template_call(
    content: str,
    program: Any,
    templates: list[tuple[int, int, bool]],
    check_runtime: Callable[[], None],
) -> bool:
    projected = assertion_template_shell_text(content, check_runtime)
    if "node:assert" in content and projected == content:
        return True
    unprojected_templates = sorted(
        (start, end)
        for start, end, _dynamic in templates
        if projected[start] == content[start] or projected[end - 1] == content[end - 1]
    )
    template_starts = [start for start, _end in unprojected_templates]
    for node in _walk_nodes(program, check_runtime):
        if node.type != "CallExpression" or not _is_assertion_like_callee(node.callee):
            continue
        for argument in node.arguments:
            argument_range = getattr(argument, "range", None)
            if not isinstance(argument_range, list) or len(argument_range) != 2:
                continue
            argument_start, argument_end = argument_range
            index = bisect_left(template_starts, argument_start)
            if (
                index < len(unprojected_templates)
                and template_starts[index] < argument_end
                and unprojected_templates[index][1] <= argument_end
            ):
                return True
    return False


def _is_assertion_like_callee(callee: Any) -> bool:
    if callee.type == "Identifier":
        return callee.name.casefold() in {"assert", "check"}
    if callee.type != "MemberExpression":
        return False
    property_name = getattr(callee.property, "name", None)
    if property_name is None:
        property_name = getattr(callee.property, "value", None)
    return isinstance(property_name, str) and property_name in _MESSAGE_ARGUMENTS


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
