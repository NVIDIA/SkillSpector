# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find direct subprocess calls using a definitely truthy local name.

This companion recognizes the straight-line ordinary-Python form reported in
issue #475. Call arguments must be passive, and unsupported expressions or
compound statements discard facts rather than guessing about Python execution.
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
_DIRECT_CALL_NAMES = frozenset({"subprocess", "Popen"})


def _truth_value(
    expression: ast.expr | None,
    facts: dict[str, bool],
) -> bool | None:
    """Return truth for a small, immutable, side-effect-free expression subset."""
    if expression is None:
        return None

    resolved: dict[ast.expr, bool] = {}
    pending: list[tuple[ast.expr, bool]] = [(expression, False)]
    while pending:
        current, expanded = pending.pop()
        if isinstance(current, ast.Constant):
            resolved[current] = bool(current.value)
        elif isinstance(current, ast.Name):
            if current.id not in facts:
                return None
            resolved[current] = facts[current.id]
        elif isinstance(current, ast.Tuple):
            if not current.elts:
                resolved[current] = False
            elif any(isinstance(item, ast.Starred) for item in current.elts):
                return None
            elif all(_is_passive_argument(item) for item in current.elts):
                resolved[current] = True
            else:
                return None
        elif isinstance(current, ast.UnaryOp):
            if not isinstance(current.op, ast.Not) and not (
                isinstance(current.op, (ast.UAdd, ast.USub))
                and isinstance(current.operand, ast.Constant)
                and type(current.operand.value) in (bool, int, float, complex)
            ):
                return None
            if expanded:
                operand = resolved[current.operand]
                resolved[current] = not operand if isinstance(current.op, ast.Not) else operand
            else:
                pending.append((current, True))
                pending.append((current.operand, False))
        else:
            return None
    return resolved[expression]


def _update_trusted_names_from_import(
    statement: ast.Import | ast.ImportFrom,
    trusted_names: set[str],
) -> None:
    """Update only direct receiver names that the import actually binds."""
    if isinstance(statement, ast.Import):
        for imported in statement.names:
            bound = imported.asname or imported.name.partition(".")[0]
            if imported.name == "subprocess" and bound == "subprocess":
                trusted_names.add(bound)
            elif bound in trusted_names:
                trusted_names.discard(bound)
        return

    if any(imported.name == "*" for imported in statement.names):
        trusted_names.clear()
        return
    for imported in statement.names:
        bound = imported.asname or imported.name
        if (
            statement.level == 0
            and statement.module == "subprocess"
            and imported.name == "Popen"
            and bound == "Popen"
        ):
            trusted_names.add(bound)
        elif bound in trusted_names:
            trusted_names.discard(bound)


class _DirectBindingCollector:
    """Collect direct receiver bindings without entering nested scopes."""

    def __init__(self, tracked_names: set[str] | frozenset[str]) -> None:
        self.tracked_names = tracked_names
        self.bound: set[str] = set()
        self.mutated: set[str] = set()
        self.nonlocal_names: set[str] = set()

    @staticmethod
    def _function_header_nodes(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[ast.AST]:
        nodes: list[ast.AST] = [*node.decorator_list, *node.args.defaults]
        nodes.extend(item for item in node.args.kw_defaults if item is not None)
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        nodes.extend(
            argument.annotation for argument in arguments if argument.annotation is not None
        )
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            nodes.append(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            nodes.append(node.args.kwarg.annotation)
        if node.returns is not None:
            nodes.append(node.returns)
        nodes.extend(getattr(node, "type_params", ()))
        return nodes

    def visit(self, node: ast.AST) -> None:
        pending = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, ast.Name):
                if (
                    isinstance(current.ctx, (ast.Store, ast.Del))
                    and current.id in self.tracked_names
                ):
                    self.bound.add(current.id)
                continue
            if isinstance(current, (ast.Attribute, ast.Subscript)):
                if isinstance(current.ctx, (ast.Store, ast.Del)):
                    root: ast.expr = current.value
                    while isinstance(root, (ast.Attribute, ast.Subscript)):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in self.tracked_names:
                        self.mutated.add(root.id)
                pending.extend(ast.iter_child_nodes(current))
                continue
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if current.name in self.tracked_names:
                    self.bound.add(current.name)
                pending.extend(self._function_header_nodes(current))
                continue
            if isinstance(current, ast.ClassDef):
                if current.name in self.tracked_names:
                    self.bound.add(current.name)
                pending.extend(current.decorator_list)
                pending.extend(current.bases)
                pending.extend(keyword.value for keyword in current.keywords)
                continue
            if isinstance(current, ast.Lambda):
                pending.extend(current.args.defaults)
                pending.extend(item for item in current.args.kw_defaults if item is not None)
                continue
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                pending.append(current.elt)
                for generator in current.generators:
                    pending.append(generator.iter)
                    pending.extend(generator.ifs)
                continue
            if isinstance(current, ast.DictComp):
                pending.extend((current.key, current.value))
                for generator in current.generators:
                    pending.append(generator.iter)
                    pending.extend(generator.ifs)
                continue
            if isinstance(current, ast.Import):
                for imported in current.names:
                    bound = imported.asname or imported.name.partition(".")[0]
                    if bound in self.tracked_names:
                        self.bound.add(bound)
                continue
            if isinstance(current, ast.ImportFrom):
                if any(imported.name == "*" for imported in current.names):
                    self.bound.update(self.tracked_names)
                    continue
                for imported in current.names:
                    bound = imported.asname or imported.name
                    if bound in self.tracked_names:
                        self.bound.add(bound)
                continue
            if isinstance(current, ast.ExceptHandler):
                if isinstance(current.name, str) and current.name in self.tracked_names:
                    self.bound.add(current.name)
                pending.extend(ast.iter_child_nodes(current))
                continue
            if isinstance(current, (ast.Global, ast.Nonlocal)):
                self.nonlocal_names.update(current.names)
                continue
            if isinstance(current, ast.MatchAs):
                if isinstance(current.name, str) and current.name in self.tracked_names:
                    self.bound.add(current.name)
                if current.pattern is not None:
                    pending.append(current.pattern)
                continue
            if isinstance(current, ast.MatchStar):
                if isinstance(current.name, str) and current.name in self.tracked_names:
                    self.bound.add(current.name)
                continue
            if isinstance(current, ast.MatchMapping):
                if isinstance(current.rest, str) and current.rest in self.tracked_names:
                    self.bound.add(current.rest)
                pending.extend(current.patterns)
                continue
            pending.extend(ast.iter_child_nodes(current))


def _function_bound_direct_names(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    tracked_names: set[str],
) -> set[str]:
    """Return compile-time local receiver names for one function scope."""
    arguments = statement.args
    named = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    names = {argument.arg for argument in named}
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    collector = _DirectBindingCollector(tracked_names)
    for child in statement.body:
        collector.visit(child)
    return names.intersection(tracked_names).union(
        collector.bound.difference(collector.nonlocal_names)
    )


def _changed_direct_names(nodes: list[ast.AST], tracked_names: set[str]) -> set[str]:
    """Return receiver names explicitly rebound or mutated by current-scope nodes."""
    collector = _DirectBindingCollector(tracked_names)
    for node in nodes:
        collector.visit(node)
    return collector.bound.union(collector.mutated)


def _class_body_changed_direct_names(
    statement: ast.ClassDef,
    tracked_names: set[str],
) -> set[str]:
    """Return explicit class-execution effects on outer receiver objects."""

    def nested_classes(node: ast.AST) -> list[ast.ClassDef]:
        classes: list[ast.ClassDef] = []
        pending = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, ast.ClassDef):
                classes.append(current)
                continue
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            pending.extend(ast.iter_child_nodes(current))
        return classes

    affected: set[str] = set()
    pending_classes = [statement]
    while pending_classes:
        current_class = pending_classes.pop()
        declaration_collector = _DirectBindingCollector(tracked_names)
        for child in current_class.body:
            declaration_collector.visit(child)
        global_names = declaration_collector.nonlocal_names.intersection(tracked_names)

        local_direct: dict[str, bool] = {}
        affected.update(global_names.intersection(declaration_collector.bound))
        for child in current_class.body:
            collector = _DirectBindingCollector(tracked_names)
            collector.visit(child)
            affected.update(
                name
                for name in collector.mutated
                if name in global_names or local_direct.get(name, True)
            )
            affected.update(collector.bound.intersection(global_names))
            pending_classes.extend(nested_classes(child))

            local_bound = collector.bound.difference(global_names)
            if isinstance(child, ast.Import):
                for imported in child.names:
                    bound = imported.asname or imported.name.partition(".")[0]
                    if bound in local_bound:
                        local_direct[bound] = (
                            imported.name == "subprocess" and bound == "subprocess"
                        )
            elif isinstance(child, ast.ImportFrom):
                for imported in child.names:
                    bound = imported.asname or imported.name
                    if bound in local_bound:
                        local_direct[bound] = (
                            child.level == 0
                            and child.module == "subprocess"
                            and imported.name == "Popen"
                            and bound == "Popen"
                        )
            elif isinstance(child, ast.Assign):
                prior_local_direct = dict(local_direct)
                for name in local_bound:
                    local_direct[name] = False
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id in local_bound:
                        local_direct[target.id] = (
                            isinstance(child.value, ast.Name)
                            and child.value.id == target.id
                            and prior_local_direct.get(
                                child.value.id,
                                child.value.id in tracked_names,
                            )
                        )
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                prior_local_direct = dict(local_direct)
                for name in local_bound:
                    local_direct[name] = False
                if isinstance(child.target, ast.Name) and child.target.id in local_bound:
                    local_direct[child.target.id] = (
                        isinstance(child.value, ast.Name)
                        and child.value.id == child.target.id
                        and prior_local_direct.get(
                            child.value.id,
                            child.value.id in tracked_names,
                        )
                    )
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name in tracked_names and child.name not in global_names:
                    local_direct[child.name] = False
            elif isinstance(child, ast.Delete):
                for name in collector.bound:
                    local_direct.pop(name, None)
            elif local_bound:
                for name in local_bound:
                    local_direct.pop(name, None)
    return affected


def _is_direct_subprocess_call(call: ast.Call, trusted_names: set[str]) -> bool:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id == "Popen" and function.id in trusted_names
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "subprocess"
        and function.value.id in trusted_names
    )


def _is_passive_argument(expression: ast.expr) -> bool:
    """Return whether evaluation cannot invoke user-controlled Python code."""
    normal, hash_required, truth_required, numeric_required, integral_required = range(5)
    pending: list[tuple[ast.expr, int]] = [(expression, normal)]
    while pending:
        current, requirement = pending.pop()
        if isinstance(current, ast.Constant):
            if requirement == numeric_required and type(current.value) not in (
                bool,
                int,
                float,
                complex,
            ):
                return False
            if requirement == integral_required and type(current.value) not in (bool, int):
                return False
            continue
        if isinstance(current, ast.Name):
            if requirement != normal:
                return False
            continue
        if isinstance(current, ast.List):
            if requirement in (hash_required, numeric_required, integral_required) or any(
                isinstance(item, ast.Starred) for item in current.elts
            ):
                return False
            pending.extend((item, normal) for item in current.elts)
            continue
        if isinstance(current, ast.Tuple):
            if requirement in (numeric_required, integral_required):
                return False
            if any(isinstance(item, ast.Starred) for item in current.elts):
                return False
            nested_requirement = hash_required if requirement == hash_required else normal
            pending.extend((item, nested_requirement) for item in current.elts)
            continue
        if isinstance(current, ast.Dict):
            if requirement in (hash_required, numeric_required, integral_required) or any(
                key is None for key in current.keys
            ):
                return False
            pending.extend((key, hash_required) for key in current.keys if key is not None)
            pending.extend((value, normal) for value in current.values)
            continue
        if isinstance(current, ast.Set):
            if requirement in (hash_required, numeric_required, integral_required):
                return False
            pending.extend((item, hash_required) for item in current.elts)
            continue
        if isinstance(current, ast.UnaryOp):
            if isinstance(current.op, ast.Not):
                pending.append((current.operand, truth_required))
            elif isinstance(current.op, (ast.UAdd, ast.USub)):
                operand_requirement = (
                    integral_required if requirement == integral_required else numeric_required
                )
                pending.append((current.operand, operand_requirement))
            elif isinstance(current.op, ast.Invert):
                pending.append((current.operand, integral_required))
            else:
                return False
            continue
        if isinstance(current, ast.JoinedStr) and all(
            isinstance(item, ast.Constant) for item in current.values
        ):
            if requirement in (numeric_required, integral_required):
                return False
            continue
        return False
    return True


def _call_arguments_are_passive(call: ast.Call) -> bool:
    return all(_is_passive_argument(argument) for argument in call.args) and all(
        keyword.arg is not None and _is_passive_argument(keyword.value) for keyword in call.keywords
    )


def _annotation_is_passive(annotation: ast.expr) -> bool:
    """Accept only annotation spellings whose evaluation cannot rebind a name."""
    return all(
        isinstance(node, (ast.Name, ast.Constant, ast.Load)) for node in ast.walk(annotation)
    )


def _function_header_is_passive(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Reject definition-time expressions that could mutate tracked bindings."""
    if statement.decorator_list or getattr(statement, "type_params", []):
        return False
    defaults = (*statement.args.defaults, *(item for item in statement.args.kw_defaults if item))
    if any(not _is_passive_argument(default) for default in defaults):
        return False
    arguments = (
        *statement.args.posonlyargs,
        *statement.args.args,
        *statement.args.kwonlyargs,
    )
    annotations = [argument.annotation for argument in arguments if argument.annotation is not None]
    if statement.args.vararg is not None and statement.args.vararg.annotation is not None:
        annotations.append(statement.args.vararg.annotation)
    if statement.args.kwarg is not None and statement.args.kwarg.annotation is not None:
        annotations.append(statement.args.kwarg.annotation)
    if statement.returns is not None:
        annotations.append(statement.returns)
    return all(_annotation_is_passive(annotation) for annotation in annotations)


def _advance_trusted_names(statement: ast.stmt, trusted_names: set[str]) -> None:
    """Apply one statement's explicit receiver-binding effects."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        _update_trusted_names_from_import(statement, trusted_names)
        return
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not _function_header_is_passive(statement):
            trusted_names.clear()
        trusted_names.discard(statement.name)
        return
    if isinstance(statement, ast.Assign):
        changed = _changed_direct_names(
            [statement.value, *statement.targets],
            trusted_names,
        )
        preserved = {
            target.id
            for target in statement.targets
            if isinstance(target, ast.Name)
            and isinstance(statement.value, ast.Name)
            and statement.value.id == target.id
            and target.id in trusted_names
        }
        trusted_names.difference_update(changed.difference(preserved))
        return
    if isinstance(statement, ast.ClassDef):
        trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
        trusted_names.difference_update(_class_body_changed_direct_names(statement, trusted_names))
        return
    trusted_names.difference_update(_changed_direct_names([statement], trusted_names))


class _Analyzer:
    def __init__(self, file_path: str, lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.findings: list[AnalyzerFinding] = []

    def _inspect_call(self, call: ast.Call, facts: dict[str, bool]) -> None:
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
        trusted_names: set[str],
    ) -> None:
        if isinstance(value, ast.Call) and _is_direct_subprocess_call(value, trusted_names):
            resolved = None
            safe_value = _call_arguments_are_passive(value)
            if safe_value:
                self._inspect_call(value, facts)
        else:
            resolved = _truth_value(value, facts)
            safe_value = resolved is not None or _is_passive_argument(value)

        if not safe_value or any(not isinstance(target, ast.Name) for target in targets):
            facts.clear()
            trusted_names.difference_update(_changed_direct_names([value, *targets], trusted_names))
            return
        for target in targets:
            assert isinstance(target, ast.Name)
            if resolved is None:
                facts.pop(target.id, None)
            else:
                facts[target.id] = resolved
            preserves_binding = (
                isinstance(value, ast.Name) and value.id == target.id and value.id in trusted_names
            )
            if not preserves_binding:
                trusted_names.discard(target.id)

    def _scan_block(
        self,
        statements: list[ast.stmt],
        *,
        trusted_names: set[str] | None = None,
    ) -> None:
        trusted_names = set(_DIRECT_CALL_NAMES if trusted_names is None else trusted_names)
        facts: dict[str, bool] = {}
        last_invalidation_by_name: dict[str, int] = {}

        def last_invalidation(name: str) -> int:
            cached = last_invalidation_by_name.get(name)
            if cached is not None:
                return cached
            last = -1
            for candidate_index, candidate in enumerate(statements):
                probe = {name}
                _advance_trusted_names(candidate, probe)
                if name not in probe:
                    last = candidate_index
            last_invalidation_by_name[name] = last
            return last

        for index, statement in enumerate(statements):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                passive_header = _function_header_is_passive(statement)
                nested_trusted_names = {
                    name for name in trusted_names if last_invalidation(name) <= index
                }
                nested_trusted_names.difference_update(
                    _function_bound_direct_names(statement, nested_trusted_names)
                )
                nested_trusted_names.discard(statement.name)
                if not passive_header:
                    nested_trusted_names.clear()
                self._scan_block(statement.body, trusted_names=nested_trusted_names)
                if passive_header:
                    facts.pop(statement.name, None)
                else:
                    facts.clear()
                    trusted_names.clear()
                trusted_names.discard(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                facts.clear()
                _update_trusted_names_from_import(statement, trusted_names)
            elif isinstance(statement, ast.Assign):
                self._scan_assignment(
                    list(statement.targets),
                    statement.value,
                    facts,
                    trusted_names,
                )
            elif isinstance(statement, ast.AnnAssign):
                value = statement.value
                if (
                    isinstance(value, ast.Call)
                    and _is_direct_subprocess_call(value, trusted_names)
                    and _call_arguments_are_passive(value)
                ):
                    self._inspect_call(value, facts)
                facts.clear()
                trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
            elif isinstance(statement, (ast.AugAssign, ast.Delete)):
                facts.clear()
                trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if _is_direct_subprocess_call(call, trusted_names) and _call_arguments_are_passive(
                    call
                ):
                    self._inspect_call(call, facts)
                else:
                    facts.clear()
                    trusted_names.difference_update(_changed_direct_names([call], trusted_names))
            elif isinstance(statement, ast.Pass) or (
                isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            ):
                continue
            elif isinstance(statement, ast.ClassDef):
                facts.clear()
                trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
                trusted_names.difference_update(
                    _class_body_changed_direct_names(statement, trusted_names)
                )
            else:
                facts.clear()
                trusted_names.difference_update(_changed_direct_names([statement], trusted_names))

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
    """Find straight-line truthy names passed to direct subprocess calls."""
    if file_type != "python":
        return []
    parsed = python_ast or parse_python_source(content, file_path)
    if parsed.tree is None:
        return []
    return _Analyzer(file_path, parsed.lines).run(parsed.tree)
