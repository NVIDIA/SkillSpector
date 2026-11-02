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

"""Static patterns: system prompt leakage (P6–P8). Node and analyze() in one module.

Detects patterns that directly extract system prompts (P6), indirectly
extract them via side channels (P7), or exfiltrate prompts via tool
calls (P8).

Framework: LLM07.
"""

from __future__ import annotations

import ast
import re
import sys
from bisect import bisect_right
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

from skillspector.artifacts import SecurityTextView
from skillspector.logging_config import get_logger
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import (
    MAX_PYTHON_AST_SOURCE_CHARS,
    parse_python_source,
    peek_python_ast,
)
from skillspector.state import AnalyzerNodeResponse, SkillspectorState

from . import static_runner
from .common import LOGICAL_LINE_BREAK, get_context, get_line_number
from .pattern_defaults import PatternCategory

logger = get_logger(__name__)

ANALYZER_ID = "static_patterns_system_prompt_leakage"

_scan_python_ast_cache_key: ContextVar[str | None] = ContextVar(
    "system_prompt_leakage_python_ast_cache_key", default=None
)

_P6_LOCAL_RETURN = re.compile(
    r"return\s+(?:prompts?|instructions?|rules?|guidelines?|directives?)", re.IGNORECASE
)

_NESTED_PYTHON_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

# P6: Direct Prompt Extraction — instructions that directly reveal system prompts
P6_PATTERNS = [
    (
        r"(?:print|output|show|display|reveal|expose|return|echo)\s+(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompts?|instructions?|rules?|guidelines?|directives?)",
        0.85,
    ),
    (
        r"(?:what\s+(?:are|is)\s+your|tell\s+me\s+your)\s+(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?|directives?)",
        0.8,
    ),
    (
        r"(?:output|print|display)\s+(?:the\s+)?(?:text|content|message)\s+(?:above|before)\s+(?:this|the\s+user)",
        0.8,
    ),
    (
        r"(?:share|disclose|leak|give\s+me)\s+(?:your\s+)?(?:internal|hidden|system|secret)\s+(?:prompt|instructions?|rules?|config)",
        0.85,
    ),
    (
        r"(?:dump|extract|copy)\s+(?:the\s+)?(?:system\s+)?(?:prompt|message|instructions?)\s+(?:to|into|as)",
        0.8,
    ),
    (
        r"(?:begin|start)\s+(?:your\s+)?(?:response|output|reply)\s+with\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
    (
        r"(?:first|before\s+anything)\s+(?:output|print|show|display)\s+(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
    (
        r"(?:verbatim|exactly|word\s+for\s+word)\s+(?:repeat|output|print)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.9,
    ),
]

# P7: Indirect Prompt Extraction — side-channel or inference-based extraction
P7_PATTERNS = [
    (
        r"(?:repeat|recite|reproduce)\s+(?:everything|all|the\s+text)\s+(?:above|before|prior|preceding)",
        0.8,
    ),
    (
        r"(?:summarize|paraphrase|rephrase|translate)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?)",
        0.75,
    ),
    (
        r"(?:translate|convert)\s+(?:your\s+)?(?:instructions?|rules?|prompt)\s+(?:into|to)\s+(?:\w+)",
        0.75,
    ),
    (
        r"(?:what\s+(?:were|was)\s+you\s+told|how\s+were\s+you\s+(?:instructed|configured|programmed|set\s+up))",
        0.7,
    ),
    (
        r"(?:if\s+you\s+had\s+to\s+(?:describe|explain)|describe\s+how\s+you(?:'re|\s+are)\s+(?:configured|set\s+up|instructed))",
        0.7,
    ),
    (
        r"(?:encode|encrypt|rot13|base64|reverse)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)",
        0.85,
    ),
    (
        r"(?:write|create)\s+(?:a\s+)?(?:poem|song|story|acrostic)\s+(?:about|using|from)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.75,
    ),
    (
        r"(?:format|present|rearrange)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:as|in)\s+(?:a\s+)?(?:list|json|yaml|csv|table|markdown)",
        0.8,
    ),
    (
        r"(?:spell\s+out|enumerate)\s+(?:each|every|all)\s+(?:of\s+your\s+)?(?:rules?|instructions?|constraints?|guidelines?)",
        0.75,
    ),
    (
        r"(?:ignore|skip)\s+(?:the\s+)?(?:last|final|end)\s+(?:instruction|rule)\s+and\s+(?:output|print|show)\s+(?:the\s+)?(?:rest|previous|first)",
        0.8,
    ),
]

# P8: Prompt Exfiltration via Tool — system prompt sent via tool calls
P8_PATTERNS = [
    (
        r"(?:write|save|store|log|dump)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:to|into|in)\s+(?:a\s+)?(?:file|disk|log|database|storage)",
        0.85,
    ),
    (
        r"(?:send|post|upload|transmit)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:to|via)\s+(?:https?://|external|remote|webhook|api|server)",
        0.9,
    ),
    (
        r"(?:include|embed|attach|append)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:in|to)\s+(?:the\s+)?(?:file|output|response|request|payload|email|message)",
        0.8,
    ),
    (
        r"(?:log|record|track|monitor)\s+(?:all\s+)?(?:system\s+)?(?:prompts?|instructions?)\s+(?:to|in|at)\s+(?:external|remote|server)",
        0.85,
    ),
    (
        r"(?:copy|clone|duplicate)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|config)\s+(?:to|into)\s+(?:a\s+)?(?:new|separate|external)",
        0.8,
    ),
    (
        r"(?:pipe|redirect|forward)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:to|through|via)\s+(?:curl|wget|fetch|http|api|tool)",
        0.85,
    ),
    (
        r"(?:use|call|invoke)\s+(?:the\s+)?(?:write|file|network|http|fetch|curl)\s+tool\s+(?:to|and)\s+(?:save|send|export)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        0.85,
    ),
]

_BENIGN_OUTPUT_RULES_HEADING = "## Output Rules (Both Modes)"
_LOGICAL_BREAK = rf"(?:{LOGICAL_LINE_BREAK.pattern})"
_BENIGN_PRINT_RULES_TAXONOMY = re.compile(
    rf"(?:\A|{_LOGICAL_BREAK})[ \t]*[\"'`]{{0,3}}[ \t]*"
    r"(?:single-class[ \t]+selectors[ \t]+are[ \t]+honored[ \t]+(?:—|--|-)[ \t]+)?"
    r"descendant[ \t]*/[ \t]*compound[ \t]*/[ \t]*"
    r"(?P<target>print[ \t]+rules)[ \t]+are"
    rf"(?:[ \t]+|[ \t]*{_LOGICAL_BREAK}[ \t]+)(?:not|never)[ \t]+evaluated"
    r"(?:[ \t]+\((?:avoids?|to[ \t]+avoid)[ \t]+over-stripping[ \t]+content"
    r"[ \t]+behind[ \t]+e\.g\.[ \t]+`?\.a[ \t]+\.b`?[ \t]+rules\))?"
    r"[ \t]*(?:[.!?][ \t]*)?[\"'`]{0,3}[ \t]*"
    rf"(?=\Z|{_LOGICAL_BREAK})",
    re.IGNORECASE,
)
_PRECEDING_DIRECTIVE = re.compile(
    r"\b(?:you|your|agents?|assistants?|models?|llms?|bots?|must|shall|should|"
    r"required|mandatory)\b"
    r"|\bbefore[ \t]+(?:replying|responding)\b"
    r"|\b(?:following|below|above|next|this|that|it|them|these|those|so|prior|"
    r"previous|preceding|everything|all|former|latter|content|text|output|"
    r"configuration|material)\b"
    r"|\bthe[ \t]+same\b"
    r"|\bwhat[ \t]+follows\b"
    r"|:[ \t]*$",
    re.IGNORECASE,
)
_NEXT_LINE_REFERENCE = re.compile(
    r"\b(?:it|them|this|these|those|so|same|above|below|prior|previous|"
    r"preceding|following|foregoing|everything|all|former|latter|content|text|"
    r"output|configuration|material)\b"
    r"|\b(?:the|this|that|these|those|same)[ \t]+(?:rules?|instructions?|prompts?|"
    r"guidelines?|directives?|operations?|actions?)\b"
    r"|\b(?:do|execute|perform|apply|follow|obey|use|print|output|show|display|"
    r"reveal|expose|return|echo|repeat|share|disclose|publish|provide|send|copy|"
    r"extract|dump|recite|summarize|translate|encode|write|save|forward|pipe)"
    r"[ \t]+that\b",
    re.IGNORECASE,
)


def _is_benign_output_rules_heading(content: str, match: re.Match[str], file_type: str) -> bool:
    """Return True only for the reported benign Markdown heading."""
    if file_type != "markdown" or match.group(0) != "Output Rules":
        return False
    line_start = content.rfind("\n", 0, match.start()) + 1
    line_end = content.find("\n", match.end())
    if line_end < 0:
        line_end = len(content)
    return content[line_start:line_end].strip() == _BENIGN_OUTPUT_RULES_HEADING


def _bounded_previous_nonblank_line(content: str, offset: int) -> tuple[str, bool]:
    """Return the prior nonblank logical line and whether it was complete."""
    window_start = max(0, offset - 512)
    parts = LOGICAL_LINE_BREAK.split(content[window_start:offset])
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].strip():
            return parts[index], index > 0 or window_start == 0
    return "", window_start == 0


def _bounded_next_nonblank_line(content: str, offset: int) -> tuple[str, bool]:
    """Return the next nonblank logical line and whether it was complete."""
    window_end = min(len(content), offset + 512)
    window = content[offset:window_end]
    cursor = 0
    for line_break in LOGICAL_LINE_BREAK.finditer(window):
        line = window[cursor : line_break.start()]
        if line.strip():
            return line, True
        cursor = line_break.end()
    if window_end == len(content):
        return window[cursor:], True
    return "", False


def _is_benign_print_rules_taxonomy(content: str, match: re.Match[str]) -> bool:
    """Return True only for a bounded declarative selector-taxonomy clause."""
    window_start = max(0, match.start() - 256)
    window_end = min(len(content), match.end() + 256)
    for candidate in _BENIGN_PRINT_RULES_TAXONOMY.finditer(content, window_start, window_end):
        if candidate.span("target") != match.span():
            continue
        candidate_end = candidate.end()
        if (
            candidate_end != len(content)
            and LOGICAL_LINE_BREAK.match(content, candidate_end) is None
        ):
            continue

        if candidate_end != len(content):
            line_break = LOGICAL_LINE_BREAK.match(content, candidate_end)
            assert line_break is not None
            next_line, next_complete = _bounded_next_nonblank_line(content, line_break.end())
            if not next_complete:
                continue
            if _NEXT_LINE_REFERENCE.search(next_line):
                continue

        if candidate.start() == 0:
            return True

        previous_line, previous_complete = _bounded_previous_nonblank_line(
            content, candidate.start()
        )
        if not previous_complete:
            return False
        return _PRECEDING_DIRECTIVE.search(previous_line) is None
    return False


def _leading_name(expression: ast.expr) -> ast.Name | None:
    """Return the name an expression starts with, such as ``rules`` in ``rules[0]``."""
    node: ast.expr | None = expression
    while node is not None and not isinstance(node, ast.Name):
        start = (node.lineno, node.col_offset)
        node = next(
            (
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.expr) and (child.lineno, child.col_offset) == start
            ),
            None,
        )
    return node


def _python_local_return_spans(
    content: str, file_path: str, check_runtime: Callable[[], None]
) -> frozenset[tuple[int, int]]:
    """Return source spans of Python ``return`` statements that return a local name.

    A span runs from the ``return`` keyword to the end of the name that starts
    the returned expression. Its source text must itself read ``return <target>``,
    so a normalized or reconstructed reading of other text never qualifies. The
    name must be a parameter or an assignment target of the same function, and
    not declared ``global`` or ``nonlocal`` there. Names from enclosing scopes or
    comprehensions, strings, comments, docstrings and source that does not parse
    keep their findings.
    """
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return frozenset()
    check_runtime()
    parsed = peek_python_ast(_scan_python_ast_cache_key.get(), content, file_path)
    if parsed is None:
        parsed = parse_python_source(content, file_path)
    check_runtime()
    if parsed.tree is None:
        return frozenset()
    # AST columns are UTF-8 byte offsets on Python's own lines, which split
    # only at CR, LF and CRLF (str.splitlines() also splits at form feeds).
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer(rb"\r\n?|\n", parsed.source_bytes))

    def offset(line: int, byte_column: int) -> int:
        byte_offset = line_starts[line - 1] + byte_column
        non_ascii = bisect_right(parsed.non_ascii_byte_ends, byte_offset)
        return byte_offset - parsed.non_ascii_extra_bytes[non_ascii]

    spans: set[tuple[int, int]] = set()
    visited = 0
    for function in ast.walk(parsed.tree):
        visited += 1
        if visited % 1024 == 0:
            check_runtime()
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        local_names = {
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            )
            if argument is not None
        }
        declared: set[str] = set()
        returns: list[tuple[ast.Return, ast.Name]] = []
        pending: list[ast.AST] = list(function.body)
        while pending:
            visited += 1
            if visited % 1024 == 0:
                check_runtime()
            node = pending.pop()
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(node.id)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                declared.update(node.names)
            elif isinstance(node, ast.Return) and node.value is not None:
                if (name := _leading_name(node.value)) is not None:
                    returns.append((node, name))
            if not isinstance(node, _NESTED_PYTHON_SCOPES):
                pending.extend(ast.iter_child_nodes(node))
        local_names -= declared
        for statement, name in returns:
            if name.id not in local_names or name.end_lineno is None or name.end_col_offset is None:
                continue
            start = offset(statement.lineno, statement.col_offset)
            end = offset(name.end_lineno, name.end_col_offset)
            if _P6_LOCAL_RETURN.fullmatch(content, start, end) is not None:
                spans.add((start, end))
    check_runtime()
    return frozenset(spans)


@dataclass
class _PreparedAnalysis:
    """Keep local-return proof anchored to the complete original source."""

    source: str
    check_runtime: Callable[[], None]
    python_file: bool = False
    _local_return_spans: dict[str, frozenset[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def analyze(
        self, content: str, file_path: str, file_type: str, source_view: SecurityTextView
    ) -> list[AnalyzerFinding]:
        return _analyze(content, file_path, file_type, self, source_view)

    def returns_local_value(
        self, match: re.Match[str], view: SecurityTextView, file_path: str
    ) -> bool:
        """Return whether the match is exactly a Python return of a local name."""
        if not self.python_file or _P6_LOCAL_RETURN.fullmatch(match.group(0)) is None:
            return False
        spans = self._local_return_spans.get(file_path)
        if spans is None:
            spans = _python_local_return_spans(self.source, file_path, self.check_runtime)
            self._local_return_spans[file_path] = spans
        start = view.source_offset(match.start())
        end = view.source_offset(match.end() - 1) + 1
        return (start, end) in spans


def prepare_analysis(
    content: str, file_type: str, check_runtime: Callable[[], None]
) -> _PreparedAnalysis:
    check_runtime()
    return _PreparedAnalysis(content, check_runtime, file_type == "python")


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    prepared = prepare_analysis(content, file_type, lambda: None)
    return _analyze(content, file_path, file_type, prepared, SecurityTextView("raw", content))


def _analyze(
    content: str,
    file_path: str,
    file_type: str,
    prepared: _PreparedAnalysis,
    source_view: SecurityTextView,
) -> list[AnalyzerFinding]:
    """Analyze content for system prompt leakage patterns (P6–P8)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.SYSTEM_PROMPT_LEAKAGE.value]

    for pattern, confidence in P6_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            if prepared.returns_local_value(match, source_view, file_path):
                continue
            if _is_benign_output_rules_heading(content, match, file_type):
                continue
            if _is_benign_print_rules_taxonomy(content, match):
                continue
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P6",
                    message="Direct Prompt Extraction",
                    severity=Severity.HIGH,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in P7_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P7",
                    message="Indirect Prompt Extraction",
                    severity=Severity.MEDIUM,
                    location=loc(line_num),
                    confidence=confidence,
                    tags=tag,
                    context=ctx(match.start()),
                    matched_text=match.group(0)[:200],
                    complete_match=match.group(0),
                )
            )
    for pattern, confidence in P8_PATTERNS:
        for match in static_runner.iter_paragraph_matches(
            pattern, content, re.IGNORECASE | re.MULTILINE
        ):
            line_num = get_line_number(content, match.start())
            findings.append(
                AnalyzerFinding(
                    rule_id="P8",
                    message="Prompt Exfiltration via Tool",
                    severity=Severity.HIGH,
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
    """Run system_prompt_leakage patterns and return findings."""
    token = _scan_python_ast_cache_key.set(state.get("python_ast_cache_key"))
    try:
        response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    finally:
        _scan_python_ast_cache_key.reset(token)
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
