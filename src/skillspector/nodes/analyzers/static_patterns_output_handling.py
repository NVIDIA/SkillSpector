# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static patterns: output handling (OH1–OH3). Node and analyze() in one module.

Detects patterns where model output is used without validation (OH1),
output crosses security context boundaries (OH2), or output size/rate
is unbounded (OH3).

Framework: LLM05.
"""

from __future__ import annotations

import ast
import re

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    build_import_aliases,
    get_context,
    get_context_from_lines,
    get_line_number,
    get_source_segment,
    resolve_call_name,
    resolve_dynamic_import_call,
)
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_output_handling"

_SUBPROCESS_OUTPUT_NAMES = frozenset(
    {"response", "output", "result", "answer", "completion", "reply", "generated"}
)
_SUBPROCESS_EXECUTION_KEYWORDS = {
    "call": frozenset({"args", "executable"}),
    "run": frozenset({"args", "input", "executable"}),
    "Popen": frozenset({"args", "executable"}),
    "check_output": frozenset({"args", "input", "executable"}),
    "check_call": frozenset({"args", "executable"}),
    "getoutput": frozenset({"cmd"}),
    "getstatusoutput": frozenset({"cmd"}),
}
_SUBPROCESS_CALLS = frozenset(_SUBPROCESS_EXECUTION_KEYWORDS)
_SUBPROCESS_FALLBACK_MAX_CHARS = 1_000
_SUBPROCESS_FALLBACK_PATTERN = re.compile(
    rf"""
    \bsubprocess\s*\.\s*(?:{"|".join(sorted(_SUBPROCESS_CALLS))})\s*\(
    [^)]{{0,{_SUBPROCESS_FALLBACK_MAX_CHARS}}}?
    (?<![-\w'"])(?:{"|".join(sorted(_SUBPROCESS_OUTPUT_NAMES))})(?!\w)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EXEC_OUTPUT_PATTERN = r"exec\s*\(\s*(?:response|output|result|answer|completion|reply|generated)"
_JAVASCRIPT_FILE_TYPES = frozenset({"javascript", "typescript"})
_JAVASCRIPT_EXTENSIONS = frozenset({".cjs", ".cts", ".js", ".jsx", ".mjs", ".mts", ".ts", ".tsx"})
_JAVASCRIPT_REGEXP_FLAGS = frozenset("dgimsuvy")
_JAVASCRIPT_REGEXP_LOOKBACK_CHARS = 4_096
_JAVASCRIPT_LINE_TERMINATORS = "\r\n\u2028\u2029"
_JAVASCRIPT_EXPRESSION_PREFIX_CHARACTERS = frozenset("=([{,:;!?&|+-*%^~<>")
_JAVASCRIPT_EXPRESSION_PREFIX_KEYWORDS = frozenset(
    {
        "case",
        "delete",
        "do",
        "else",
        "in",
        "instanceof",
        "new",
        "return",
        "throw",
        "typeof",
        "void",
    }
)
_JAVASCRIPT_MUTATION_TRIVIA = r"(?:\s|/\*(?:[^*]|\*(?!/))*\*/|//[^\r\n]*(?:\r\n?|\n|$))*"
_JAVASCRIPT_ASSIGNMENT_OPERATOR = r"(?:\?\?=|&&=|\|\|=|\*\*=|>>>=|>>=|<<=|[+\-*/%&|^]?=(?!=|>))"
_JAVASCRIPT_IDENTIFIER = r"(?:[$_]|[^\W\d])[\w$]*"
_JAVASCRIPT_REGEXP_LITERAL = r"/(?:\\[^\r\n]|[^/\\\r\n]){1,4096}/[dgimsuvy]*"
_JAVASCRIPT_REGEXP_CONSTRUCTOR = r"(?:new\s+)?\bRegExp\s*\([^\r\n)]{0,4096}\)"
_JAVASCRIPT_REGEXP_INSTANCE = rf"(?:{_JAVASCRIPT_REGEXP_LITERAL}|{_JAVASCRIPT_REGEXP_CONSTRUCTOR})"
_JAVASCRIPT_REGEXP_PROTOTYPE = r"\bRegExp\s*(?:\.\s*prototype\b|\[\s*['\"`]prototype['\"`]\s*\])"
_JAVASCRIPT_REGEXP_PROTOTYPE_FROM_INSTANCE = (
    rf"(?:\b(?:Object|Reflect)\s*\.\s*getPrototypeOf\s*\(\s*"
    rf"(?:\(\s*)*{_JAVASCRIPT_REGEXP_INSTANCE}(?:\s*\))*\s*\)|"
    rf"(?:\(\s*)*{_JAVASCRIPT_REGEXP_INSTANCE}(?:\s*\))*\s*\.\s*"
    r"(?:__proto__\b|constructor\s*\.\s*prototype\b))"
)
_JAVASCRIPT_REGEXP_PROTOTYPE_EXPRESSION = (
    rf"(?:{_JAVASCRIPT_REGEXP_PROTOTYPE}|"
    rf"{_JAVASCRIPT_REGEXP_PROTOTYPE_FROM_INSTANCE})"
)
_JAVASCRIPT_UNICODE_ESCAPE_PATTERN = re.compile(
    r"\\(?:u(?:\{(?P<braced>[0-9a-fA-F]+)\}|(?P<fixed>[0-9a-fA-F]{4}))|"
    r"x(?P<hex>[0-9a-fA-F]{2}))"
)
_JAVASCRIPT_SIMPLE_STRING_CONCAT_PATTERN = re.compile(
    r"(?P<left_quote>['\"])(?P<left>[A-Za-z_$]+)(?P=left_quote)\s*\+\s*"
    r"(?P<right_quote>['\"])(?P<right>[A-Za-z_$]+)(?P=right_quote)"
)
_JAVASCRIPT_COMMENT_PATTERN = re.compile(r"/\*[\s\S]*?\*/|//[^\r\n]*")
_JAVASCRIPT_EXEC_PROPERTY_ALIAS_PATTERN = re.compile(
    rf"\b(?:const|let|var)\s+(?P<name>{_JAVASCRIPT_IDENTIFIER})\s*=\s*['\"`]exec['\"`]"
)
_JAVASCRIPT_REGEXP_INSTANCE_ALIAS_PATTERN = re.compile(
    rf"\b(?:const|let|var)\s+(?P<target>{_JAVASCRIPT_IDENTIFIER})\s*=\s*"
    rf"{_JAVASCRIPT_REGEXP_INSTANCE}(?=\s*(?:[;,\r\n]|$))"
)
_JAVASCRIPT_IDENTIFIER_ALIAS_PATTERN = re.compile(
    rf"\b(?:const|let|var)\s+(?P<target>{_JAVASCRIPT_IDENTIFIER})\s*=\s*"
    rf"(?P<source>{_JAVASCRIPT_IDENTIFIER})\b(?=\s*(?:[;,\r\n]|$))"
)
_JAVASCRIPT_IMPORT_ALIAS_PATTERN = re.compile(
    rf"\b(?P<source>{_JAVASCRIPT_IDENTIFIER})\s+as\s+"
    rf"(?P<target>{_JAVASCRIPT_IDENTIFIER})\b"
)

# OH1: Unvalidated Output Injection — model output used directly in dangerous sinks
OH1_PATTERNS = [
    # Python: output piped into exec/eval. Subprocess calls are inspected via AST below.
    (_EXEC_OUTPUT_PATTERN, 0.9),
    (r"eval\s*\(\s*(?:response|output|result|answer|completion|reply|generated)", 0.9),
    (r"os\.system\s*\(\s*(?:response|output|result|answer|completion)", 0.85),
    (r"os\.popen\s*\(\s*(?:response|output|result|answer|completion)", 0.85),
    # Web: output injected into HTML without sanitization
    (r"innerHTML\s*=\s*(?:response|output|result|answer|completion)", 0.8),
    (r"document\.write\s*\(\s*(?:response|output|result|answer|completion)", 0.8),
    (r"\.html\s*\(\s*(?:response|output|result|answer|completion)", 0.7),
    (r"dangerouslySetInnerHTML\s*=\s*\{", 0.65),
    # SQL: output concatenated into queries
    (
        r"(?:execute|cursor\.execute|query)\s*\([^)]*(?:\+|%|\.format|f['\"])\s*.*?(?:response|output|result)",
        0.85,
    ),
    (r"f['\"](?:SELECT|INSERT|UPDATE|DELETE)\s+.*?\{(?:response|output|result)", 0.9),
    # Shell: output in command strings
    (
        r"(?:run|execute|shell)\s+(?:the\s+)?(?:generated|model|llm|ai)\s+(?:output|response|code|command)",
        0.8,
    ),
    (
        r"(?:pipe|pass|feed)\s+(?:the\s+)?(?:output|response|result)\s+(?:directly\s+)?(?:to|into)\s+(?:the\s+)?(?:shell|terminal|command|interpreter)",
        0.85,
    ),
    # Markdown/template injection
    (
        r"(?:use|insert|embed)\s+(?:the\s+)?(?:raw|unfiltered|unescaped|unsanitized)\s+(?:output|response)",
        0.8,
    ),
]

# OH2: Cross-Context Output — output from one context used in another
OH2_PATTERNS = [
    (
        r"(?:pass|forward|relay|send|pipe)\s+(?:the\s+)?(?:output|response|result)\s+(?:from\s+\w+\s+)?(?:to|into)\s+(?:another|different|separate|external)\s+(?:context|agent|service|system|session)",
        0.75,
    ),
    (
        r"(?:share|transfer|propagate)\s+(?:the\s+)?(?:output|response|context|state)\s+(?:across|between|to\s+other)\s+(?:sessions?|contexts?|agents?|services?)",
        0.75,
    ),
    (
        r"(?:inject|insert|embed)\s+(?:the\s+)?(?:output|response)\s+(?:from\s+\w+\s+)?(?:into|as)\s+(?:the\s+)?(?:system\s+prompt|instructions?|context)",
        0.85,
    ),
    (
        r"(?:use|include)\s+(?:the\s+)?(?:previous|other|external)\s+(?:agent|model|llm)(?:'s)?\s+(?:output|response)\s+(?:as|in|for)\s+(?:input|context|prompt)",
        0.8,
    ),
    (
        r"(?:cross[_-]?context|cross[_-]?session|cross[_-]?agent)\s+(?:output|data|state)\s+(?:sharing|transfer|flow)",
        0.8,
    ),
    (
        r"(?:take|use)\s+(?:the\s+)?(?:output|result)\s+(?:and\s+)?(?:run|execute|eval)\s+(?:it\s+)?(?:in|on|against)\s+(?:a\s+)?(?:different|another|new)\s+(?:environment|context|system)",
        0.8,
    ),
]

# OH3: Unbounded Output — output size or rate not bounded
OH3_PATTERNS = [
    (
        r"(?:no|without|disable)\s+(?:output\s+)?(?:length|size|token)\s+(?:limit|cap|maximum|restriction)",
        0.75,
    ),
    (r"max[_-]?tokens?\s*=\s*(?:None|float\s*\(\s*['\"]inf['\"]|math\.inf|999999|1000000)", 0.8),
    (
        r"(?:generate|produce|output)\s+(?:as\s+much|unlimited|unbounded|infinite)\s+(?:text|content|output|tokens?)",
        0.8,
    ),
    (r"(?:no|without)\s+(?:output\s+)?(?:truncation|trimming|cutting)", 0.6),
    (
        r"(?:repeat|loop|generate)\s+(?:the\s+)?(?:output|response)\s+(?:indefinitely|forever|continuously|endlessly)",
        0.8,
    ),
    (
        r"(?:keep|continue)\s+(?:generating|producing|outputting)\s+(?:until|unless)\s+(?:stopped|killed|interrupted)",
        0.75,
    ),
    (r"(?:stream|emit)\s+(?:output|tokens?|response)\s+(?:without\s+(?:limit|bound|end))", 0.75),
    (r"(?:flood|spam|fill)\s+(?:the\s+)?(?:output|log|console|terminal|channel)", 0.8),
    (r"max[_-]?(?:output[_-]?)?length\s*=\s*(?:None|0|-1|float\s*\(\s*['\"]inf)", 0.75),
]


def _contains_output_name(node: ast.AST) -> bool:
    """Return whether *node* references a model-output-like identifier.

    Constants and keyword names are deliberately excluded. In particular, a
    subprocess command containing the literal CLI flag ``"--output"`` or the
    keyword ``capture_output=True`` must not be treated as model-generated data.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id.casefold() in _SUBPROCESS_OUTPUT_NAMES:
            return True
        if isinstance(child, ast.Attribute) and child.attr.casefold() in _SUBPROCESS_OUTPUT_NAMES:
            return True
    return False


def _is_javascript_source(file_path: str, file_type: str) -> bool:
    """Return whether analyzer inputs identify JavaScript or TypeScript source."""
    suffix_start = file_path.rfind(".")
    suffix = file_path[suffix_start:].casefold() if suffix_start >= 0 else ""
    return file_type in _JAVASCRIPT_FILE_TYPES or suffix in _JAVASCRIPT_EXTENSIONS


def _javascript_unicode_escape_value(match: re.Match[str]) -> str:
    """Decode a bounded JavaScript Unicode or hexadecimal escape for scanning."""
    digits = match.group("braced") or match.group("fixed") or match.group("hex")
    significant = digits.lstrip("0") or "0"
    if len(significant) > 6:
        return match.group(0)
    value = int(significant, 16)
    if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
        return match.group(0)
    return chr(value)


def _javascript_mutation_scan_variants(content: str) -> tuple[str, ...]:
    """Return conservative source variants for mutation signal matching.

    The raw variant prevents comment-like text inside regexp literals from
    hiding later code. The comment-collapsed variant recognizes mutations with
    comments inserted between JavaScript tokens. Identifier and string escapes
    plus simple constant string concatenations are normalized in both.
    """
    normalized = _JAVASCRIPT_UNICODE_ESCAPE_PATTERN.sub(_javascript_unicode_escape_value, content)
    for _ in range(4):
        folded = _JAVASCRIPT_SIMPLE_STRING_CONCAT_PATTERN.sub(
            lambda match: f'"{match.group("left")}{match.group("right")}"',
            normalized,
        )
        if folded == normalized:
            break
        normalized = folded
    without_comments = _JAVASCRIPT_COMMENT_PATTERN.sub(" ", normalized)
    return (normalized,) if without_comments == normalized else (normalized, without_comments)


def _javascript_expand_aliases(
    seeds: set[str],
    contents: tuple[str, ...],
    *,
    include_imports: bool = True,
) -> frozenset[str]:
    """Propagate simple identifier and import aliases from known seed names."""
    aliases = set(seeds)
    aliases_by_source: dict[str, set[str]] = {}
    for content in contents:
        patterns = [_JAVASCRIPT_IDENTIFIER_ALIAS_PATTERN]
        if include_imports:
            patterns.append(_JAVASCRIPT_IMPORT_ALIAS_PATTERN)
        for pattern in patterns:
            for match in pattern.finditer(content):
                aliases_by_source.setdefault(match.group("source"), set()).add(
                    match.group("target")
                )

    pending = list(aliases)
    while pending:
        source = pending.pop()
        for target in aliases_by_source.get(source, ()):
            if target not in aliases:
                aliases.add(target)
                pending.append(target)
    return frozenset(aliases)


def _javascript_exec_property_aliases(contents: tuple[str, ...]) -> frozenset[str]:
    """Collect simple local or imported aliases whose constant value is ``exec``."""
    seeds = {
        match.group("name")
        for content in contents
        for match in _JAVASCRIPT_EXEC_PROPERTY_ALIAS_PATTERN.finditer(content)
    }
    return _javascript_expand_aliases(seeds, contents)


def _javascript_regexp_prototype_receiver(
    contents: tuple[str, ...],
) -> str:
    """Build a pattern for direct and simply aliased RegExp prototype receivers."""
    constructor_aliases = _javascript_expand_aliases(
        {"RegExp"}, contents, include_imports=False
    ) - {"RegExp"}
    needs_instance_aliases = any(
        any(hint in content for hint in ("getPrototypeOf", "__proto__", "constructor"))
        for content in contents
    )
    instance_seeds = (
        {
            match.group("target")
            for content in contents
            for match in _JAVASCRIPT_REGEXP_INSTANCE_ALIAS_PATTERN.finditer(content)
        }
        if needs_instance_aliases
        else set()
    )
    instance_aliases = _javascript_expand_aliases(instance_seeds, contents)

    prototype_expressions = [_JAVASCRIPT_REGEXP_PROTOTYPE_EXPRESSION]
    if constructor_aliases:
        constructors = "|".join(re.escape(alias) for alias in sorted(constructor_aliases))
        prototype_expressions.append(
            rf"\b(?:{constructors})\b\s*(?:\.\s*prototype\b|"
            r"\[\s*['\"`]prototype['\"`]\s*\])"
        )
    if instance_aliases:
        instances = "|".join(re.escape(alias) for alias in sorted(instance_aliases))
        instance = rf"\b(?:{instances})\b"
        prototype_expressions.extend(
            (
                rf"\b(?:Object|Reflect)\s*\.\s*getPrototypeOf\s*\(\s*{instance}\s*\)",
                rf"{instance}\s*\.\s*(?:__proto__\b|constructor\s*\.\s*prototype\b)",
            )
        )
    prototype_expression = rf"(?:{'|'.join(prototype_expressions)})"

    prototype_alias_pattern = re.compile(
        rf"\b(?:const|let|var)\s+(?P<target>{_JAVASCRIPT_IDENTIFIER})\s*=\s*"
        rf"{prototype_expression}(?=\s*(?:[;,\r\n]|$))"
    )
    prototype_seeds = {
        match.group("target")
        for content in contents
        for match in prototype_alias_pattern.finditer(content)
    }
    prototype_aliases = _javascript_expand_aliases(prototype_seeds, contents)
    receivers = [prototype_expression]
    if prototype_aliases:
        aliases = "|".join(re.escape(alias) for alias in sorted(prototype_aliases))
        receivers.append(rf"\b(?:{aliases})\b")
    return rf"(?:{'|'.join(receivers)})"


def _javascript_mutates_regexp_exec(
    content: str,
    receiver: str,
    property_aliases: frozenset[str],
) -> bool:
    """Return whether *content* mutates ``exec`` on a known RegExp prototype."""
    property_keys = [r"['\"`]exec['\"`]"]
    computed_properties: list[str] = []
    if property_aliases:
        aliases = "|".join(re.escape(alias) for alias in sorted(property_aliases))
        property_keys.append(rf"\b(?:{aliases})\b")
        computed_properties.append(rf"\[\s*(?:{aliases})\s*\]")
    property_key = rf"(?:{'|'.join(property_keys)})"
    member = r"(?:\.\s*exec\b|\[\s*['\"`]exec['\"`]\s*\]"
    if computed_properties:
        member += rf"|{'|'.join(computed_properties)}"
    member += ")"
    wrapped_receiver = rf"(?:\(\s*)*{receiver}(?:\s*\))*"

    patterns = (
        rf"{wrapped_receiver}{_JAVASCRIPT_MUTATION_TRIVIA}{member}"
        rf"{_JAVASCRIPT_MUTATION_TRIVIA}{_JAVASCRIPT_ASSIGNMENT_OPERATOR}",
        rf"\bdelete{_JAVASCRIPT_MUTATION_TRIVIA}{wrapped_receiver}"
        rf"{_JAVASCRIPT_MUTATION_TRIVIA}{member}",
        rf"\b(?:Object|Reflect)\s*\.\s*(?:defineProperty|set)\s*\(\s*"
        rf"{wrapped_receiver}\s*,\s*{property_key}\s*,",
        rf"\bObject\s*\.\s*(?:assign|defineProperties)\s*\(\s*"
        rf"{wrapped_receiver}\s*,\s*\{{[^}}\r\n]{{0,4096}}"
        rf"(?:\bexec\s*:|['\"`]exec['\"`]\s*:"
        rf"|\[\s*{property_key}\s*\]\s*:)",
        rf"{wrapped_receiver}{_JAVASCRIPT_MUTATION_TRIVIA}\.\s*"
        rf"__define(?:Getter|Setter)__\s*\(\s*{property_key}\s*,",
    )
    return any(re.search(pattern, content, re.MULTILINE) for pattern in patterns)


def _javascript_regexp_exec_mutation_possible(contents: tuple[str, ...]) -> bool:
    """Return whether visible code may replace the method used by regexp literals.

    Prototype acquisition and mutation may occur in different components, so
    the graph node evaluates these signals across the complete scanned skill.
    The public ``analyze`` helper applies the same rule within a single source.
    """
    if not contents:
        return False
    if not any(
        any(
            hint in content
            for hint in ("RegExp", "getPrototypeOf", "__proto__", "constructor", "\\u", "\\x")
        )
        for content in contents
    ):
        return False
    if not any(
        any(
            hint in content
            for hint in (
                "=",
                "delete",
                "defineProperty",
                "defineProperties",
                "assign",
                "set",
                "__define",
            )
        )
        for content in contents
    ):
        return False

    scan_contents = tuple(
        dict.fromkeys(
            variant
            for content in contents
            for variant in _javascript_mutation_scan_variants(content)
        )
    )
    property_aliases = _javascript_exec_property_aliases(scan_contents)
    receiver = _javascript_regexp_prototype_receiver(scan_contents)
    return any(
        _javascript_mutates_regexp_exec(content, receiver, property_aliases)
        for content in scan_contents
    )


class _OutputHandlingPatternAdapter:
    """Bind whole-skill mutation context to the generic static runner contract."""

    ANALYZER_ID = "static_patterns_output_handling"

    def __init__(self, regexp_exec_mutation_possible: bool) -> None:
        self._regexp_exec_mutation_possible = regexp_exec_mutation_possible

    def analyze(self, *, content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
        """Analyze one component with whole-skill RegExp mutation context."""
        return analyze(
            content,
            file_path,
            file_type,
            regexp_exec_mutation_possible=self._regexp_exec_mutation_possible,
        )


def _skip_javascript_whitespace_backward(content: str, index: int, floor: int) -> int:
    """Skip JavaScript whitespace before *index*, but deliberately not comments.

    Recognizing comments without a JavaScript lexer is unsafe because ``/*``
    and ``//`` are both valid text inside regexp character classes. Treating
    those sequences as trivia can skip into a preceding regexp and make an
    unrelated ``exec(output)`` call look like ``RegExp.prototype.exec``.
    Comment-separated receivers therefore fail closed as OH1 findings.
    """
    while index > floor and content[index - 1].isspace():
        index -= 1
    return index


def _is_javascript_character_escaped(content: str, index: int, floor: int) -> bool:
    """Return whether the character at *index* has an odd backslash prefix."""
    backslashes = 0
    cursor = index - 1
    while cursor >= floor and content[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _is_javascript_identifier_part(character: str) -> bool:
    """Conservatively return whether *character* may continue a JS identifier.

    Python exposes Unicode ``XID_Continue`` through ``str.isidentifier()``,
    while JavaScript uses ``ID_Continue`` and can support a newer Unicode
    version. Treat an otherwise-unknown non-ASCII character as an identifier
    part so version or normalization differences fail closed instead of
    splitting an identifier such as ``x\u037areturn`` at the ``return`` suffix.
    """
    if character.isspace():
        return False
    return ord(character) > 0x7F or character == "$" or ("a" + character).isidentifier()


def _javascript_braced_unicode_escape_ends_at(content: str, index: int, floor: int) -> bool:
    """Return whether a ``\\u{...}`` escape ends immediately before *index*."""
    if index <= floor or content[index - 1] != "}":
        return False

    cursor = index - 2
    while cursor >= floor and content[cursor] in "0123456789abcdefABCDEF":
        cursor -= 1
    if cursor == index - 2:
        return False
    if cursor < floor:
        return True
    if content[cursor] != "{":
        return False
    if cursor - 2 < floor:
        return True
    return content[cursor - 2 : cursor] == "\\u"


def _javascript_regexp_opening_has_unambiguous_line_prefix(
    content: str, opening_slash: int, floor: int
) -> bool:
    """Reject an opening candidate when earlier line syntax makes it ambiguous.

    A slash immediately before ``g.exec`` may be division, and the nearest
    preceding slash may then be the *closing* delimiter of another regexp.
    Only accept a candidate when there is no earlier code slash on its line.
    Slashes inside ordinary quoted strings are ignored; comments and template
    literals deliberately fail closed because they need a full JS lexer.
    """
    last_line_break = max(
        content.rfind(terminator, floor, opening_slash)
        for terminator in _JAVASCRIPT_LINE_TERMINATORS
    )
    if last_line_break >= floor:
        line_start = last_line_break + 1
    elif floor == 0 or content[floor - 1] in _JAVASCRIPT_LINE_TERMINATORS:
        line_start = floor
    else:
        return False

    quote: str | None = None
    escaped = False
    for character in content[line_start:opening_slash]:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue

        if character in {'"', "'"}:
            quote = character
        elif character in {"`", "/"}:
            return False

    return quote is None


def _find_javascript_regexp_opening_slash(
    content: str, closing_slash: int, floor: int
) -> int | None:
    """Find a regexp literal's opening slash without mistaking class slashes."""
    in_character_class = False
    cursor = closing_slash - 1
    while cursor >= floor:
        character = content[cursor]
        if character in _JAVASCRIPT_LINE_TERMINATORS:
            return None

        if character in "/[]" and not _is_javascript_character_escaped(content, cursor, floor):
            if character == "]":
                in_character_class = True
            elif character == "[" and in_character_class:
                in_character_class = False
            elif character == "/" and not in_character_class:
                return cursor if cursor + 1 < closing_slash else None
        cursor -= 1
    return None


def _javascript_expression_can_start_at(content: str, index: int, floor: int) -> bool:
    """Conservatively validate that a JavaScript expression may start at *index*."""
    cursor = _skip_javascript_whitespace_backward(content, index, floor)
    if cursor == floor:
        return floor == 0

    previous = content[cursor - 1]
    if previous in _JAVASCRIPT_EXPRESSION_PREFIX_CHARACTERS:
        if previous == ">":
            return cursor - floor >= 2 and content[cursor - 2] == "="
        if previous in "+-" and cursor - floor >= 2 and content[cursor - 2] == previous:
            return False
        if previous == "!":
            operator_start = cursor - 1
            while True:
                prefix_end = _skip_javascript_whitespace_backward(content, operator_start, floor)
                if prefix_end <= floor or content[prefix_end - 1] != "!":
                    break
                operator_start = prefix_end - 1
            return _javascript_expression_can_start_at(content, operator_start, floor)
        return True
    if not _is_javascript_identifier_part(previous):
        return False

    token_start = cursor - 1
    while token_start > floor and _is_javascript_identifier_part(content[token_start - 1]):
        token_start -= 1
    token = content[token_start:cursor]
    if _javascript_braced_unicode_escape_ends_at(content, token_start, floor):
        return False
    token_prefix = _skip_javascript_whitespace_backward(content, token_start, floor)
    if token_prefix == floor and floor > 0:
        # The bounded window may start inside an identifier or after a property
        # accessor. Without the preceding lexical context, treating a suffix
        # such as ``return`` as a keyword could suppress an unrelated sink.
        return False
    if token_prefix > floor and content[token_prefix - 1] in ".#":
        return False
    return token in _JAVASCRIPT_EXPRESSION_PREFIX_KEYWORDS


def _is_javascript_regexp_literal_exec(
    content: str,
    match: re.Match[str],
    file_path: str,
    file_type: str,
) -> bool:
    """Return whether an ``exec`` match is called on a JavaScript regexp literal.

    This bounded backward recognizer handles whitespace, optional chaining,
    and parentheses around the literal. It deliberately rejects comments and
    a parenthesized function argument such as ``makeRunner(/x/).exec(output)``.
    Contexts that require matching an earlier control header remain findings.
    """
    if not _is_javascript_source(file_path, file_type):
        return False
    if content[match.start() : match.start() + 4] != "exec":
        # The surrounding OH1 pattern is case-insensitive, but JavaScript
        # property names are not. Only the built-in lowercase method is safe.
        return False

    floor = max(0, match.start() - _JAVASCRIPT_REGEXP_LOOKBACK_CHARS)
    cursor = _skip_javascript_whitespace_backward(content, match.start(), floor)
    if cursor <= floor or content[cursor - 1] != ".":
        return False
    cursor = _skip_javascript_whitespace_backward(content, cursor - 1, floor)

    if cursor > floor and content[cursor - 1] == "?":
        cursor = _skip_javascript_whitespace_backward(content, cursor - 1, floor)

    closing_parentheses = 0
    while cursor > floor and content[cursor - 1] == ")":
        closing_parentheses += 1
        cursor = _skip_javascript_whitespace_backward(content, cursor - 1, floor)

    while cursor > floor and content[cursor - 1] in _JAVASCRIPT_REGEXP_FLAGS:
        cursor -= 1
    if cursor <= floor or content[cursor - 1] != "/":
        return False

    opening_slash = _find_javascript_regexp_opening_slash(content, cursor - 1, floor)
    if opening_slash is None:
        return False
    if not _javascript_regexp_opening_has_unambiguous_line_prefix(content, opening_slash, floor):
        return False
    if not _javascript_expression_can_start_at(content, opening_slash, floor):
        return False

    wrapper_start = opening_slash
    for _ in range(closing_parentheses):
        wrapper_start = _skip_javascript_whitespace_backward(content, wrapper_start, floor)
        if wrapper_start <= floor or content[wrapper_start - 1] != "(":
            return False
        wrapper_start -= 1

    if closing_parentheses and not _javascript_expression_can_start_at(
        content, wrapper_start, floor
    ):
        return False

    return True


def _subprocess_execution_arguments(node: ast.Call, method_name: str) -> list[ast.expr]:
    """Return subprocess arguments that can supply executed content."""
    execution_keywords = _SUBPROCESS_EXECUTION_KEYWORDS.get(method_name)
    if execution_keywords is None:
        return []

    arguments = [node.args[0]] if node.args else []
    arguments.extend(
        keyword.value for keyword in node.keywords if keyword.arg in execution_keywords
    )
    return arguments


def _analyze_subprocess_fallback(
    content: str, file_path: str, tag: list[str]
) -> list[AnalyzerFinding]:
    """Conservatively detect subprocess sinks when Python AST analysis is unavailable."""
    return [
        AnalyzerFinding(
            rule_id="OH1",
            message="Unvalidated Output Injection",
            severity=Severity.HIGH,
            location=Location(
                file=file_path,
                start_line=get_line_number(content, match.start()),
            ),
            confidence=0.85,
            tags=tag,
            context=get_context(content, match.start()),
            matched_text=match.group(0)[:200],
        )
        for match in _SUBPROCESS_FALLBACK_PATTERN.finditer(content)
    ]


def _analyze_python_subprocess_calls(
    content: str, file_path: str, tag: list[str]
) -> list[AnalyzerFinding]:
    """Detect output-like values used as Python subprocess command arguments."""
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        # Static pattern analysis also runs over partial/generated Python files.
        # Retain best-effort subprocess coverage without failing the analyzer.
        return _analyze_subprocess_fallback(content, file_path, tag)

    aliases = build_import_aliases(tree)
    lines = content.splitlines()
    findings: list[AnalyzerFinding] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        call_name = resolve_call_name(node, aliases)
        if call_name is None:
            call_name = resolve_dynamic_import_call(node, aliases)
        if call_name is None or not call_name.startswith("subprocess."):
            continue

        _, _, method_name = call_name.partition(".")
        execution_arguments = _subprocess_execution_arguments(node, method_name)
        if (
            method_name not in _SUBPROCESS_CALLS
            or not execution_arguments
            or not any(_contains_output_name(argument) for argument in execution_arguments)
        ):
            continue

        lineno = getattr(node, "lineno", 1)
        end_lineno = getattr(node, "end_lineno", None)
        findings.append(
            AnalyzerFinding(
                rule_id="OH1",
                message="Unvalidated Output Injection",
                severity=Severity.HIGH,
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=0.95,
                tags=tag,
                context=get_context_from_lines(lines, lineno),
                matched_text=get_source_segment(lines, lineno, end_lineno),
            )
        )

    return findings


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    *,
    regexp_exec_mutation_possible: bool | None = None,
) -> list[AnalyzerFinding]:
    """Analyze content for output handling patterns (OH1–OH3)."""
    findings: list[AnalyzerFinding] = []

    if regexp_exec_mutation_possible is None:
        regexp_exec_mutation_possible = _is_javascript_source(
            file_path, file_type
        ) and _javascript_regexp_exec_mutation_possible((content,))

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.OUTPUT_HANDLING.value]

    for pattern, confidence in OH1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            if (
                pattern == _EXEC_OUTPUT_PATTERN
                and not regexp_exec_mutation_possible
                and _is_javascript_regexp_literal_exec(content, match, file_path, file_type)
            ):
                continue
            line_num = get_line_number(content, match.start())
            adj = (
                min(1.0, confidence + 0.1)
                if file_type in ("python", "javascript", "shell")
                else confidence
            )
            findings.append(
                AnalyzerFinding(
                    rule_id="OH1",
                    message="Unvalidated Output Injection",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=adj,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    if file_type == "python":
        subprocess_findings = _analyze_python_subprocess_calls(content, file_path, tag)
    else:
        # Other file types can contain embedded Python snippets, so preserve the
        # analyzer's previous best-effort subprocess coverage for those files.
        subprocess_findings = _analyze_subprocess_fallback(content, file_path, tag)
    findings.extend(subprocess_findings)

    for pattern, confidence in OH2_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="OH2",
                    message="Cross-Context Output",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    for pattern, confidence in OH3_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="OH3",
                    message="Unbounded Output",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run output_handling patterns and return findings."""
    components: list[str] = state.get("components") or []
    file_cache: dict[str, str] = state.get("file_cache") or {}
    javascript_contents: list[str] = []
    has_uninspected_javascript = False
    for path in components:
        if not _is_javascript_source(path, ""):
            continue
        content = file_cache.get(path)
        if (
            content is None
            or len(content) > static_runner.MAX_FILE_CHARS
            or "\x00" in content[:512]
        ):
            has_uninspected_javascript = True
        else:
            javascript_contents.append(content)
    adapter = _OutputHandlingPatternAdapter(
        has_uninspected_javascript
        or _javascript_regexp_exec_mutation_possible(tuple(javascript_contents))
    )
    response = static_runner.run_static_patterns_with_ledger(state, [adapter])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
