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

# OH1: Unvalidated Output Injection — model output used directly in dangerous sinks
OH1_PATTERNS = [
    # Python: output piped into exec/eval. Subprocess calls are inspected via AST below.
    (r"exec\s*\(\s*(?:response|output|result|answer|completion|reply|generated)", 0.9),
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


def analyze(content: str, file_path: str, file_type: str) -> list[AnalyzerFinding]:
    """Analyze content for output handling patterns (OH1–OH3)."""
    findings: list[AnalyzerFinding] = []

    def loc(ln: int) -> Location:
        return Location(file=file_path, start_line=ln)

    def ctx(start: int) -> str:
        return get_context(content, start)

    tag = [PatternCategory.OUTPUT_HANDLING.value]

    for pattern, confidence in OH1_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
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
    response = static_runner.run_static_patterns_with_ledger(state, [sys.modules[__name__]])
    logger.info("%s: %d findings", ANALYZER_ID, len(response["findings"]))
    return response
