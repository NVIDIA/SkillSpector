# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find direct subprocess calls using a definitely truthy local name.

This companion deliberately recognizes only the straight-line form reported in
issue #475. Unsupported expressions and compound statements discard facts
rather than guessing about Python execution.
"""

from __future__ import annotations

import ast

from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, parse_python_source

from .common import get_context_from_lines, get_source_segment
from .pattern_defaults import PatternCategory

ANALYZER_ID = "static_patterns_tool_misuse"
USES_PYTHON_AST = True
BOUND_SHELL_EVIDENCE = "_tm1_bound_shell_value"
_MAX_TRUTH_DEPTH = 64
_MAX_FUNCTION_DEPTH = 64


def _truth_value(
    expression: ast.expr | None,
    facts: dict[str, bool],
    depth: int = 0,
) -> bool | None:
    """Return truth for a small, immutable, side-effect-free expression subset."""
    if expression is None or depth > _MAX_TRUTH_DEPTH:
        return None
    if isinstance(expression, ast.Constant):
        return bool(expression.value)
    if isinstance(expression, ast.Name):
        return facts.get(expression.id)
    if isinstance(expression, ast.Tuple):
        if not expression.elts:
            return False
        if any(isinstance(item, ast.Starred) for item in expression.elts):
            return None
        if any(_truth_value(item, facts, depth + 1) is None for item in expression.elts):
            return None
        return True
    if isinstance(expression, ast.UnaryOp):
        operand = _truth_value(expression.operand, facts, depth + 1)
        if operand is None:
            return None
        if isinstance(expression.op, ast.Not):
            return not operand
        if (
            isinstance(expression.op, (ast.UAdd, ast.USub))
            and isinstance(expression.operand, ast.Constant)
            and type(expression.operand.value) in (int, float, complex)
        ):
            return operand
    return None


def _bound_names(statement: ast.Import | ast.ImportFrom) -> set[str] | None:
    """Return names rebound by an import, or ``None`` for a wildcard import."""
    names: set[str] = set()
    for imported in statement.names:
        if imported.name == "*":
            return None
        names.add(imported.asname or imported.name.partition(".")[0])
    return names


def _is_direct_subprocess_call(call: ast.Call) -> bool:
    function = call.func
    return (
        isinstance(function, ast.Name)
        and function.id == "Popen"
        or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "subprocess"
        )
    )


def _has_named_expression(call: ast.Call) -> bool:
    return any(isinstance(node, ast.NamedExpr) for node in ast.walk(call))


def _annotation_is_passive(annotation: ast.expr) -> bool:
    """Accept only annotation spellings whose evaluation cannot rebind a name."""
    return all(
        isinstance(node, (ast.Name, ast.Constant, ast.Load)) for node in ast.walk(annotation)
    )


class _Analyzer:
    def __init__(self, file_path: str, lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.findings: list[AnalyzerFinding] = []

    def _inspect_call(self, call: ast.Call, facts: dict[str, bool]) -> None:
        if not _is_direct_subprocess_call(call) or _has_named_expression(call):
            return
        shell = next((item.value for item in call.keywords if item.arg == "shell"), None)
        if (
            not isinstance(shell, ast.Name)
            or shell.id.casefold().startswith("true")
            or facts.get(shell.id) is not True
        ):
            return
        line = getattr(call, "lineno", 1)
        end_line = getattr(call, "end_lineno", None)
        self.findings.append(
            AnalyzerFinding(
                rule_id="TM1",
                message="Tool Parameter Abuse",
                severity=Severity.HIGH,
                location=Location(file=self.file_path, start_line=line, end_line=end_line),
                confidence=0.8,
                tags=[PatternCategory.TOOL_MISUSE.value],
                context=get_context_from_lines(self.lines, line),
                matched_text=get_source_segment(self.lines, line, end_line),
                evidence={BOUND_SHELL_EVIDENCE: True},
            )
        )

    def _scan_assignment(
        self,
        targets: list[ast.expr],
        value: ast.expr,
        facts: dict[str, bool],
    ) -> None:
        if isinstance(value, ast.Call) and _is_direct_subprocess_call(value):
            self._inspect_call(value, facts)
            resolved = None
            safe_value = not _has_named_expression(value)
        else:
            resolved = _truth_value(value, facts)
            safe_value = resolved is not None

        if not safe_value or any(not isinstance(target, ast.Name) for target in targets):
            facts.clear()
            return
        for target in targets:
            assert isinstance(target, ast.Name)
            if resolved is None:
                facts.pop(target.id, None)
            else:
                facts[target.id] = resolved

    def _scan_block(self, statements: list[ast.stmt], *, depth: int = 0) -> None:
        facts: dict[str, bool] = {}
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if depth < _MAX_FUNCTION_DEPTH:
                    self._scan_block(statement.body, depth=depth + 1)
                facts.clear()
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                names = _bound_names(statement)
                if names is None:
                    facts.clear()
                else:
                    for name in names:
                        facts.pop(name, None)
            elif isinstance(statement, ast.Assign):
                self._scan_assignment(list(statement.targets), statement.value, facts)
            elif isinstance(statement, ast.AnnAssign):
                if statement.value is not None:
                    self._scan_assignment([statement.target], statement.value, facts)
                if not _annotation_is_passive(statement.annotation):
                    facts.clear()
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if _is_direct_subprocess_call(call) and not _has_named_expression(call):
                    self._inspect_call(call, facts)
                else:
                    facts.clear()
            elif isinstance(statement, ast.Pass) or (
                isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            ):
                continue
            else:
                facts.clear()

    def run(self, tree: ast.Module) -> list[AnalyzerFinding]:
        self._scan_block(tree.body)
        return sorted(self.findings, key=lambda finding: finding.location.start_line)


def analyze(
    content: str,
    file_path: str,
    file_type: str,
    *,
    python_ast: ParsedPythonFile | None = None,
) -> list[AnalyzerFinding]:
    """Find straight-line truthy names passed to subprocess ``shell``."""
    if file_type != "python":
        return []
    parsed = python_ast or parse_python_source(content, file_path)
    if parsed.tree is None:
        return []
    return _Analyzer(file_path, parsed.lines).run(parsed.tree)
