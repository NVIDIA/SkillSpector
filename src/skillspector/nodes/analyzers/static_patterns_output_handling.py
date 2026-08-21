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
import sys

from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, parse_python_source
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import (
    get_complete_source_segment,
    get_context,
    get_context_from_lines,
    get_line_number,
    resolve_call_name,
    resolve_dynamic_import_call,
)
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_output_handling"
USES_PYTHON_AST = True

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


def _javascript_whitespace_crosses_possible_line_comment(
    content: str, whitespace_start: int, whitespace_end: int, floor: int
) -> bool:
    """Return whether a backward whitespace walk may have entered a line comment.

    A line comment ends at a JavaScript line terminator. After walking backward
    across that terminator, an accepted expression-prefix character or keyword
    at the end of the comment must not validate the following slash as a regexp
    literal. This includes Annex B's legacy ``<!--`` opener and line-start
    ``-->`` closer. Ordinary quoted strings are tracked so comment lookalikes on
    the preceding line do not fail closed. Definite comment openers do. A prior
    unquoted slash only becomes ambiguous if later quoting prevents this small
    scanner from proving that a subsequent ``//`` is outside a regexp. Lines
    that inherit a multiline string, template, or block-comment state and
    truncated lines also fail closed.
    """
    whitespace = content[whitespace_start:whitespace_end]
    if not any(terminator in whitespace for terminator in _JAVASCRIPT_LINE_TERMINATORS):
        return False

    last_line_break = max(
        content.rfind(terminator, floor, whitespace_start)
        for terminator in _JAVASCRIPT_LINE_TERMINATORS
    )
    if last_line_break >= floor:
        line_start = last_line_break + 1
    elif floor == 0 or content[floor - 1] in _JAVASCRIPT_LINE_TERMINATORS:
        line_start = floor
    else:
        return True

    line_prefix = content[line_start:whitespace_start]
    if "`" in line_prefix or "*/" in line_prefix:
        return True
    if line_prefix.lstrip().startswith("-->"):
        return True
    if last_line_break >= floor:
        terminator_start = last_line_break
        while (
            terminator_start > floor
            and content[terminator_start - 1] in _JAVASCRIPT_LINE_TERMINATORS
        ):
            terminator_start -= 1
        if _is_javascript_character_escaped(content, terminator_start, floor):
            return True

    quote: str | None = None
    escaped = False
    saw_unquoted_slash = False
    cursor = line_start
    while cursor < whitespace_start:
        character = content[cursor]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {'"', "'"}:
            if saw_unquoted_slash:
                return True
            quote = character
        elif character == "`":
            return True
        elif content.startswith("<!--", cursor, whitespace_start):
            return True
        elif character == "/":
            if cursor + 1 < whitespace_start and content[cursor + 1] in {"/", "*"}:
                return True
            saw_unquoted_slash = True
        cursor += 1

    return quote is not None


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
    if _javascript_whitespace_crosses_possible_line_comment(content, cursor, index, floor):
        return False
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

    This is a syntactic classification of the ordinary built-in method. Raw
    cross-file source cannot establish whether, or when, another component
    mutates JavaScript intrinsics; correlating those strings here would turn
    uncertainty into an unverified HIGH finding at this call site. Prototype
    mutation belongs in a separate parser-backed rule with data-flow context.
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
            complete_match=match.group(0),
        )
        for match in _SUBPROCESS_FALLBACK_PATTERN.finditer(content)
    ]


def _analyze_python_subprocess_calls(
    content: str,
    file_path: str,
    tag: list[str],
    python_ast: ParsedPythonFile | None = None,
) -> list[AnalyzerFinding]:
    """Detect output-like values used as Python subprocess command arguments."""
    if python_ast is None:
        python_ast = parse_python_source(content, file_path)
    tree = python_ast.tree
    if tree is None:
        # Static pattern analysis also runs over partial/generated Python files.
        # Retain best-effort subprocess coverage without failing the analyzer.
        return _analyze_subprocess_fallback(content, file_path, tag)

    aliases = python_ast.import_aliases
    lines = python_ast.lines
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
        complete_match = get_complete_source_segment(lines, lineno, end_lineno)
        findings.append(
            AnalyzerFinding(
                rule_id="OH1",
                message="Unvalidated Output Injection",
                severity=Severity.HIGH,
                location=Location(file=file_path, start_line=lineno, end_line=end_lineno),
                confidence=0.95,
                tags=tag,
                context=get_context_from_lines(lines, lineno),
                matched_text=complete_match[:200],
                complete_match=complete_match,
            )
        )

    return findings


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    *,
    python_ast: ParsedPythonFile | None = None,
) -> list[AnalyzerFinding]:
    """Analyze content for output handling patterns (OH1–OH3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.OUTPUT_HANDLING.value]

    for pattern, confidence in OH1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
            if pattern == _EXEC_OUTPUT_PATTERN and _is_javascript_regexp_literal_exec(
                content, match, file_path, file_type
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
                    complete_match=match.group(0),
                )
            )
    if file_type == "python":
        subprocess_findings = _analyze_python_subprocess_calls(content, file_path, tag, python_ast)
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
                    complete_match=match.group(0),
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
                    complete_match=match.group(0),
                )
            )
    return findings


def node(state: SkillspectorState) -> AnalyzerNodeResponse:
    """Run output_handling patterns and return findings."""
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
