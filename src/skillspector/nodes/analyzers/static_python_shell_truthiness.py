# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find direct subprocess calls using a definitely truthy local name.

This companion recognizes the straight-line ordinary-Python form reported in
issue #475. Arguments evaluated through ``shell=`` must be passive, and
unsupported expressions or compound statements discard facts rather than
guessing about Python execution.
"""

from __future__ import annotations

import ast

from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, parse_python_source

from .common import get_complete_source_segment, get_context_from_lines
from .pattern_defaults import PatternCategory

ANALYZER_ID = "static_patterns_tool_misuse"
USES_PYTHON_AST = True
BOUND_SHELL_EVIDENCE = "_tm1_bound_shell_value"
_DIRECT_CALL_NAMES = frozenset({"subprocess", "Popen"})
_CACHED_SUBPROCESS_API_SLOTS = frozenset(
    {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
)
BoundShellCallKey = tuple[int, int, int, int]


def _bound_shell_call_key(call: ast.Call) -> BoundShellCallKey:
    """Return a stable source key for one call node."""
    return (
        getattr(call, "lineno", 1),
        getattr(call, "col_offset", 0),
        getattr(call, "end_lineno", getattr(call, "lineno", 1)),
        getattr(call, "end_col_offset", getattr(call, "col_offset", 0)),
    )


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


def _direct_bound_names(node: ast.AST) -> set[str]:
    """Return names bound by *node* without entering deferred nested scopes."""
    candidates: set[str] = set()
    for current in ast.walk(node):
        if isinstance(current, ast.Name) and isinstance(current.ctx, (ast.Store, ast.Del)):
            candidates.add(current.id)
        elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.add(current.name)
        elif isinstance(current, ast.Import):
            candidates.update(
                imported.asname or imported.name.partition(".")[0] for imported in current.names
            )
        elif isinstance(current, ast.ImportFrom):
            candidates.update(
                imported.asname or imported.name
                for imported in current.names
                if imported.name != "*"
            )
        elif isinstance(current, ast.ExceptHandler) and isinstance(current.name, str):
            candidates.add(current.name)
        elif isinstance(current, (ast.MatchAs, ast.MatchStar)) and isinstance(
            current.name,
            str,
        ):
            candidates.add(current.name)
        elif isinstance(current, ast.MatchMapping) and isinstance(current.rest, str):
            candidates.add(current.rest)
    collector = _DirectBindingCollector(candidates)
    collector.visit(node)
    return collector.bound


def _function_parameter_names(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return names that may already hold unsafe values on body entry."""
    arguments = statement.args
    names = {
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    declarations = _DirectBindingCollector(set())
    for child in statement.body:
        declarations.visit(child)
    names.update(declarations.nonlocal_names)
    return names


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


def _nodes_contain_eager_call(nodes: list[ast.AST]) -> bool:
    """Return whether eager evaluation reaches a call outside deferred bodies."""
    pending = list(nodes)
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Call):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _function_header_is_passive(current):
                return True
            continue
        if isinstance(current, ast.ClassDef):
            if (
                not _class_header_is_passive(current)
                or _class_body_may_release_outer_value(current)
                or _nodes_contain_eager_call(list(current.body))
            ):
                return True
            continue
        if isinstance(current, ast.Lambda):
            defaults = (
                *current.args.defaults,
                *(item for item in current.args.kw_defaults if item is not None),
            )
            if any(not _is_passive_argument(default) for default in defaults):
                return True
            continue
        pending.extend(ast.iter_child_nodes(current))
    return False


def _class_body_may_release_outer_value(statement: ast.ClassDef) -> bool:
    """Return whether class execution stores to a declared outer name."""
    declarations = _DirectBindingCollector(set())
    for child in statement.body:
        declarations.visit(child)
    outer_names = declarations.nonlocal_names
    if outer_names:
        bindings = _DirectBindingCollector(outer_names)
        for child in statement.body:
            bindings.visit(child)
        if bindings.bound.intersection(outer_names):
            return True

    pending: list[ast.AST] = list(statement.body)
    while pending:
        current = pending.pop()
        if isinstance(current, ast.ClassDef):
            if _class_body_may_release_outer_value(current):
                return True
            continue
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return False


def _class_body_has_eager_effects(statement: ast.ClassDef) -> bool:
    """Return whether class execution can run user code or unsafe finalizers."""
    declarations = _DirectBindingCollector(set())
    for child in statement.body:
        declarations.visit(child)
    outer_names = declarations.nonlocal_names
    bound_names: set[str] = set()
    finalizer_safe_names: set[str] = set()

    for child in statement.body:
        if isinstance(child, (ast.Global, ast.Nonlocal, ast.Pass)) or (
            isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant)
        ):
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            releases_unsafe_value = (
                child.name in bound_names and child.name not in finalizer_safe_names
            )
            if (
                child.name in outer_names
                or not _function_header_is_passive(child)
                or releases_unsafe_value
            ):
                return True
            bound_names.add(child.name)
            finalizer_safe_names.add(child.name)
            continue
        if isinstance(child, ast.ClassDef):
            releases_unsafe_value = (
                child.name in bound_names and child.name not in finalizer_safe_names
            )
            if (
                child.name in outer_names
                or not _class_header_is_passive(child)
                or releases_unsafe_value
                or _class_body_has_eager_effects(child)
            ):
                return True
            bound_names.add(child.name)
            finalizer_safe_names.add(child.name)
            continue
        if isinstance(child, ast.Assign):
            if not all(isinstance(target, ast.Name) for target in child.targets):
                return True
            target_names = {target.id for target in child.targets if isinstance(target, ast.Name)}
            if target_names.intersection(outer_names):
                return True
            releases_unsafe_value = any(
                name in bound_names
                and name not in finalizer_safe_names
                and not (isinstance(child.value, ast.Name) and child.value.id == name)
                for name in target_names
            )
            if releases_unsafe_value or not _is_finalizer_safe_value(
                child.value,
                finalizer_safe_names,
            ):
                return True
            bound_names.update(target_names)
            finalizer_safe_names.update(target_names)
            continue
        if isinstance(child, ast.AnnAssign):
            if not isinstance(child.target, ast.Name) or not _annotation_is_passive(
                child.annotation
            ):
                return True
            if child.value is None:
                continue
            target_name = child.target.id
            if target_name in outer_names:
                return True
            releases_unsafe_value = (
                target_name in bound_names
                and target_name not in finalizer_safe_names
                and not (isinstance(child.value, ast.Name) and child.value.id == target_name)
            )
            if releases_unsafe_value or not _is_finalizer_safe_value(
                child.value,
                finalizer_safe_names,
            ):
                return True
            bound_names.add(target_name)
            finalizer_safe_names.add(target_name)
            continue
        if isinstance(child, ast.Assert) and _is_finalizer_safe_value(
            child.test,
            finalizer_safe_names,
        ):
            continue
        return True
    return False


def _class_deferred_receiver_trust(
    statement: ast.ClassDef,
    trusted_names: set[str],
) -> tuple[set[str], dict[int, set[str]]]:
    """Return final and observed-call outer trust for deferred class methods."""
    deferred = set(trusted_names)
    trusted_at_call_by_definition: dict[int, set[str]] = {}
    active_functions: dict[str, int] = {}
    declarations = _DirectBindingCollector(_DIRECT_CALL_NAMES)
    for child in statement.body:
        declarations.visit(child)
    global_names = declarations.nonlocal_names.intersection(_DIRECT_CALL_NAMES)

    for child_index, child in enumerate(statement.body):
        call = _passive_direct_call(child)
        if call is not None:
            assert isinstance(call.func, ast.Name)
            owner = active_functions.get(call.func.id)
            if owner is not None:
                trusted_at_call_by_definition.setdefault(owner, set()).update(deferred)

        if _nodes_contain_eager_call([child]):
            # Class-body expressions run before any method can be called and may
            # mutate the surrounding module/function receiver binding.
            deferred.clear()

        collector = _DirectBindingCollector(_DIRECT_CALL_NAMES)
        collector.visit(child)
        deferred.difference_update(collector.mutated)
        globally_bound = collector.bound.intersection(global_names)
        if isinstance(child, ast.Import):
            for imported in child.names:
                bound = imported.asname or imported.name.partition(".")[0]
                if bound not in globally_bound:
                    continue
                if imported.name == "subprocess" and bound == "subprocess":
                    deferred.add(bound)
                else:
                    deferred.discard(bound)
        elif isinstance(child, ast.ImportFrom):
            if any(imported.name == "*" for imported in child.names) and global_names:
                deferred.clear()
            for imported in child.names:
                bound = imported.asname or imported.name
                if bound not in globally_bound:
                    continue
                if (
                    child.level == 0
                    and child.module == "subprocess"
                    and imported.name == "Popen"
                    and bound == "Popen"
                ):
                    deferred.add(bound)
                else:
                    deferred.discard(bound)
        else:
            deferred.difference_update(globally_bound)

        changed_names = _direct_bound_names(child)
        for name in changed_names:
            active_functions.pop(name, None)
        if (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _function_header_is_passive(child)
            and _is_immediate_function(child)
        ):
            active_functions[child.name] = child_index
    return deferred, trusted_at_call_by_definition


def _is_direct_subprocess_syntax(call: ast.Call) -> bool:
    """Return whether a call uses one of the direct subprocess spellings."""
    function = call.func
    if isinstance(function, ast.Name):
        return function.id == "Popen"
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "subprocess"
    )


def _is_direct_subprocess_call(call: ast.Call, trusted_names: set[str]) -> bool:
    if not _is_direct_subprocess_syntax(call):
        return False
    function = call.func
    receiver = function.id if isinstance(function, ast.Name) else function.value.id
    return receiver in trusted_names


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


def _value_preserves_receiver_trust(
    expression: ast.expr,
    trusted_names: set[str],
    protocol_safe_names: set[str] | None = None,
) -> bool:
    """Return whether eager evaluation cannot replace a direct receiver."""
    if _is_passive_argument(expression):
        return True
    if isinstance(expression, ast.Call):
        return (
            _is_direct_subprocess_call(expression, trusted_names)
            and _call_arguments_are_passive(expression)
            and (
                protocol_safe_names is None
                or _call_arguments_are_protocol_safe(expression, protocol_safe_names)
            )
        )
    if isinstance(expression, (ast.Tuple, ast.List)):
        return all(
            not isinstance(item, ast.Starred)
            and _value_preserves_receiver_trust(
                item,
                trusted_names,
                protocol_safe_names,
            )
            for item in expression.elts
        )
    if isinstance(expression, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        values = [expression.elt]
    elif isinstance(expression, ast.DictComp):
        values = [expression.key, expression.value]
    else:
        return False
    for generator in expression.generators:
        if generator.is_async or not _is_simple_comprehension_target(generator.target):
            return False
        if protocol_safe_names is not None and (
            not _is_finalizer_safe_value(generator.iter, protocol_safe_names)
            or any(
                not _is_finalizer_safe_value(condition, protocol_safe_names)
                for condition in generator.ifs
            )
        ):
            return False
        values.extend((generator.iter, *generator.ifs))
    # Comprehensions have historically retained ownership for their direct
    # subprocess calls. Preserve that behavior while still rejecting generic
    # nested calls and receiver stores.
    return all(
        _value_preserves_receiver_trust(
            value,
            trusted_names,
            protocol_safe_names,
        )
        for value in values
    )


def _is_simple_comprehension_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return True
    return isinstance(target, (ast.Tuple, ast.List)) and all(
        _is_simple_comprehension_target(item) for item in target.elts
    )


def _shell_argument_is_captured_before_effects(call: ast.Call) -> bool:
    """Return whether evaluation reaches ``shell=`` without user-code effects.

    Python evaluates every positional argument, including starred expansions,
    before keyword arguments. Keyword values are then evaluated in their stored
    order. Effects after ``shell=`` cannot change the already captured value.
    """
    if any(not _is_passive_argument(argument) for argument in call.args):
        return False
    for keyword in call.keywords:
        if keyword.arg == "shell":
            return _is_passive_argument(keyword.value)
        if keyword.arg is None or not _is_passive_argument(keyword.value):
            return False
    return False


def _is_finalizer_safe_value(expression: ast.expr, safe_names: set[str]) -> bool:
    """Return whether releasing the resulting value cannot run user code."""
    if not _is_passive_argument(expression):
        return False
    return all(
        not isinstance(node, ast.Name) or node.id in safe_names for node in ast.walk(expression)
    )


def _call_arguments_are_protocol_safe(call: ast.Call, safe_names: set[str]) -> bool:
    """Return whether subprocess argument consumption cannot dispatch user code."""
    return all(_is_finalizer_safe_value(argument, safe_names) for argument in call.args) and all(
        keyword.arg is not None and _is_finalizer_safe_value(keyword.value, safe_names)
        for keyword in call.keywords
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


def _class_header_is_passive(statement: ast.ClassDef) -> bool:
    """Return whether evaluating a class header cannot rebind a receiver."""
    return not (
        statement.decorator_list
        or statement.bases
        or statement.keywords
        or getattr(statement, "type_params", [])
    )


def _is_immediate_function(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a direct call begins executing this function body."""
    if isinstance(statement, ast.AsyncFunctionDef):
        return False
    pending: list[ast.AST] = list(statement.body)
    while pending:
        current = pending.pop()
        if isinstance(current, (ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return True


def _passive_direct_call(statement: ast.stmt) -> ast.Call | None:
    """Return a directly evaluated simple-name call with passive arguments."""
    value: ast.expr | None = None
    if isinstance(statement, (ast.Expr, ast.Assign)):
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        value = statement.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and _call_arguments_are_passive(value)
    ):
        return value
    return None


def _advance_trusted_names(
    statement: ast.stmt,
    trusted_names: set[str],
    bound_names: set[str],
    finalizer_safe_names: set[str],
    unknown_unsafe_bindings: list[bool],
) -> None:
    """Apply one eager statement's receiver-trust and value-release effects."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        imported_names = _direct_bound_names(statement)
        releases_unsafe_value = any(
            name in bound_names and name not in finalizer_safe_names for name in imported_names
        )
        bound_names.update(imported_names)
        finalizer_safe_names.difference_update(imported_names)
        _update_trusted_names_from_import(statement, trusted_names)
        imports_unknown_names = isinstance(statement, ast.ImportFrom) and any(
            imported.name == "*" for imported in statement.names
        )
        if releases_unsafe_value or imports_unknown_names:
            unknown_unsafe_bindings[0] = True
        if releases_unsafe_value or unknown_unsafe_bindings[0]:
            trusted_names.clear()
        return
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        releases_unsafe_value = (
            statement.name in bound_names and statement.name not in finalizer_safe_names
        )
        if not _function_header_is_passive(statement) or releases_unsafe_value:
            trusted_names.clear()
            unknown_unsafe_bindings[0] = True
        bound_names.add(statement.name)
        finalizer_safe_names.discard(statement.name)
        trusted_names.discard(statement.name)
        return
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
        simple_targets = all(isinstance(target, ast.Name) for target in targets)
        result_is_finalizer_safe = _is_finalizer_safe_value(
            statement.value,
            finalizer_safe_names,
        )
        preserves_receiver_trust = simple_targets and _value_preserves_receiver_trust(
            statement.value,
            trusted_names,
            finalizer_safe_names,
        )
        releases_unsafe_value = simple_targets and any(
            target.id in bound_names
            and target.id not in finalizer_safe_names
            and not (isinstance(statement.value, ast.Name) and statement.value.id == target.id)
            for target in targets
            if isinstance(target, ast.Name)
        )
        if not preserves_receiver_trust or releases_unsafe_value:
            trusted_names.clear()
            finalizer_safe_names.clear()
            unknown_unsafe_bindings[0] = True
        if not simple_targets:
            bound_names.update(_direct_bound_names(statement))
            return
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
        for target in targets:
            assert isinstance(target, ast.Name)
            bound_names.add(target.id)
            if preserves_receiver_trust and not releases_unsafe_value and result_is_finalizer_safe:
                finalizer_safe_names.add(target.id)
            else:
                finalizer_safe_names.discard(target.id)
        return
    if isinstance(statement, ast.AnnAssign):
        value = statement.value
        target = statement.target
        simple_target = isinstance(target, ast.Name)
        preserves_receiver_trust = (
            simple_target
            and _annotation_is_passive(statement.annotation)
            and (
                value is None
                or _value_preserves_receiver_trust(
                    value,
                    trusted_names,
                    finalizer_safe_names,
                )
            )
        )
        releases_unsafe_value = (
            value is not None
            and simple_target
            and target.id in bound_names
            and target.id not in finalizer_safe_names
            and not (isinstance(value, ast.Name) and value.id == target.id)
        )
        result_is_finalizer_safe = value is not None and _is_finalizer_safe_value(
            value,
            finalizer_safe_names,
        )
        if not preserves_receiver_trust or releases_unsafe_value:
            trusted_names.clear()
            finalizer_safe_names.clear()
            unknown_unsafe_bindings[0] = True
        if value is not None:
            bound_names.update(_direct_bound_names(statement))
            if simple_target:
                if (
                    preserves_receiver_trust
                    and not releases_unsafe_value
                    and result_is_finalizer_safe
                ):
                    finalizer_safe_names.add(target.id)
                else:
                    finalizer_safe_names.discard(target.id)
        trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
        return
    if isinstance(statement, ast.ClassDef):
        releases_unsafe_value = (
            statement.name in bound_names and statement.name not in finalizer_safe_names
        )
        unsafe_class_execution = (
            not _class_header_is_passive(statement)
            or _class_body_has_eager_effects(statement)
            or releases_unsafe_value
        )
        if unsafe_class_execution:
            trusted_names.clear()
            unknown_unsafe_bindings[0] = True
            finalizer_safe_names.clear()
        else:
            finalizer_safe_names.discard(statement.name)
        trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
        trusted_names.difference_update(_class_body_changed_direct_names(statement, trusted_names))
        bound_names.update(_direct_bound_names(statement))
        return
    if isinstance(statement, ast.Expr):
        value = statement.value
        preserves_receiver_trust = _value_preserves_receiver_trust(
            value,
            trusted_names,
            finalizer_safe_names,
        )
        if not preserves_receiver_trust:
            trusted_names.clear()
            unknown_unsafe_bindings[0] = True
            finalizer_safe_names.clear()
        else:
            trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
        bound_names.update(_direct_bound_names(statement))
        return
    if isinstance(statement, ast.Assert):
        expressions = [statement.test]
        if statement.msg is not None:
            expressions.append(statement.msg)
        if not (
            _is_finalizer_safe_value(statement.test, finalizer_safe_names)
            and all(
                _value_preserves_receiver_trust(
                    item,
                    trusted_names,
                    finalizer_safe_names,
                )
                for item in expressions
            )
        ):
            trusted_names.clear()
            unknown_unsafe_bindings[0] = True
            finalizer_safe_names.clear()
        return
    if isinstance(statement, (ast.Global, ast.Nonlocal, ast.Pass)):
        return
    bound_names.update(_direct_bound_names(statement))
    finalizer_safe_names.clear()
    trusted_names.clear()
    unknown_unsafe_bindings[0] = True


class _CachedSubprocessState:
    """Track explicit writes to cached module slots independently of imports."""

    def __init__(self) -> None:
        self.module_names = {"subprocess"}
        self.changed_methods: set[str] = set()
        self.changed_popen = False
        self.effect_generation = 0
        self.protocol_unsafe = False
        self.bound_names: set[str] = set()
        self.safe_names: set[str] = set()

    def copy(self) -> _CachedSubprocessState:
        copied = _CachedSubprocessState()
        copied.module_names = set(self.module_names)
        copied.changed_methods = set(self.changed_methods)
        copied.changed_popen = self.changed_popen
        copied.effect_generation = self.effect_generation
        copied.protocol_unsafe = self.protocol_unsafe
        copied.bound_names = set(self.bound_names)
        copied.safe_names = set(self.safe_names)
        return copied

    def _invalidate(self) -> None:
        """An unmodeled eager effect can change receivers and restore old slots."""
        self.module_names.clear()
        self.changed_methods.clear()
        self.changed_popen = False
        self.safe_names.clear()
        self.effect_generation += 1

    def _passive_value(self, value: ast.expr) -> bool:
        pending = [value]
        while pending:
            current = pending.pop()
            if _is_passive_argument(current):
                continue
            if (
                isinstance(current, ast.Attribute)
                and isinstance(current.value, ast.Name)
                and current.value.id in self.module_names
            ):
                continue
            if isinstance(current, (ast.Tuple, ast.List)) and not any(
                isinstance(item, ast.Starred) for item in current.elts
            ):
                pending.extend(current.elts)
                continue
            return False
        return True

    def blocks(self, call: ast.Call) -> bool:
        # A later import or slot write cannot establish that an unknown earlier
        # effect left the cached module's protocol and finalizer bindings intact.
        if self.protocol_unsafe or self.effect_generation > 0:
            return False
        function = call.func
        if isinstance(function, ast.Name):
            return function.id == "Popen" and self.changed_popen
        return isinstance(function, ast.Attribute) and function.attr in self.changed_methods

    def advance(self, statement: ast.stmt) -> None:
        """Apply only explicit eager bindings; deferred bodies do not execute here."""
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            imported_names = _direct_bound_names(statement)
            if any(
                name in self.bound_names
                and name not in self.safe_names
                and name not in self.module_names
                for name in imported_names
            ):
                self._invalidate()
            ordinary_subprocess_import = (
                isinstance(statement, ast.Import)
                and all(imported.name == "subprocess" for imported in statement.names)
            ) or (
                isinstance(statement, ast.ImportFrom)
                and statement.level == 0
                and statement.module == "subprocess"
                and all(
                    imported.name == "Popen" and (imported.asname or imported.name) == "Popen"
                    for imported in statement.names
                )
            )
            if not ordinary_subprocess_import:
                self._invalidate()
            self.module_names.difference_update(imported_names)
            self.bound_names.update(imported_names)
            self.safe_names.difference_update(imported_names)
            if isinstance(statement, ast.Import):
                self.module_names.update(
                    imported.asname or "subprocess"
                    for imported in statement.names
                    if imported.name == "subprocess"
                )
            elif statement.level == 0 and statement.module == "subprocess":
                for imported in statement.names:
                    if imported.name == "Popen" and (imported.asname or imported.name) == "Popen":
                        self.changed_popen = "Popen" in self.changed_methods
            return
        if isinstance(statement, ast.ClassDef):
            if not _class_header_is_passive(statement):
                self._invalidate()
            class_state = self.copy()
            declarations = _DirectBindingCollector(set())
            pending_classes = [statement]
            while pending_classes:
                for child in pending_classes.pop().body:
                    declarations.visit(child)
                    if isinstance(child, ast.ClassDef):
                        pending_classes.append(child)
            local_bound = _direct_bound_names(statement).difference(declarations.nonlocal_names)
            for child in statement.body:
                local_bound.update(
                    _direct_bound_names(child).difference(declarations.nonlocal_names)
                )
            class_state.bound_names.difference_update(local_bound)
            class_state.safe_names.difference_update(local_bound)
            for child in statement.body:
                class_state.advance(child)
            self.protocol_unsafe |= class_state.protocol_unsafe
            if class_state.effect_generation != self.effect_generation:
                self.module_names.clear()
                self.changed_popen = False
                self.safe_names.clear()
                self.effect_generation = class_state.effect_generation
            self.changed_methods = class_state.changed_methods
            self.module_names.difference_update(declarations.nonlocal_names)
            self.module_names.update(
                class_state.module_names.intersection(declarations.nonlocal_names)
            )
            if "Popen" in declarations.nonlocal_names:
                self.changed_popen = class_state.changed_popen
            self.bound_names.update(
                class_state.bound_names.intersection(declarations.nonlocal_names)
            )
            self.safe_names.difference_update(declarations.nonlocal_names)
            self.safe_names.update(class_state.safe_names.intersection(declarations.nonlocal_names))
            if (
                statement.name in self.bound_names
                and statement.name not in self.safe_names
                and statement.name not in self.module_names
            ):
                self._invalidate()
            self.bound_names.add(statement.name)
            self.safe_names.discard(statement.name)
            self.module_names.discard(statement.name)
            return
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _function_header_is_passive(statement) or (
                statement.name in self.bound_names
                and statement.name not in self.safe_names
                and statement.name not in self.module_names
            ):
                self._invalidate()
            self.bound_names.add(statement.name)
            self.safe_names.discard(statement.name)
            self.module_names.discard(statement.name)
            return
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets, value = list(statement.targets), statement.value
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                if not _annotation_is_passive(statement.annotation):
                    self._invalidate()
                return
            targets, value = [statement.target], statement.value
        elif isinstance(statement, ast.AugAssign):
            targets = [statement.target]
        elif isinstance(statement, ast.Delete):
            self._invalidate()
            self.bound_names.difference_update(_direct_bound_names(statement))
            return
        else:
            if not (
                isinstance(statement, (ast.Global, ast.Nonlocal, ast.Pass))
                or isinstance(statement, (ast.Expr, ast.Return))
                and (statement.value is None or self._passive_value(statement.value))
            ):
                self._invalidate()
            return

        native_api_value = value is not None and any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.module_names
            and node.attr in _CACHED_SUBPROCESS_API_SLOTS
            and node.attr not in self.changed_methods
            or isinstance(node, ast.Name)
            and node.id == "Popen"
            and not self.changed_popen
            for node in ast.walk(value)
        )
        same_slot_identity = (
            len(targets) == 1
            and isinstance(targets[0], ast.Attribute)
            and isinstance(targets[0].value, ast.Name)
            and targets[0].value.id in self.module_names
            and isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id in self.module_names
            and targets[0].attr == value.attr
        )
        # A native callable capture or cross-slot store cannot establish a
        # custom replacement. Abstain without tracking callable value aliases.
        if native_api_value and not same_slot_identity:
            self._invalidate()
        if (
            isinstance(statement, ast.AugAssign)
            or value is not None
            and not self._passive_value(value)
        ):
            self._invalidate()
        rhs_names = (
            {node.id for node in ast.walk(value) if isinstance(node, ast.Name)}
            if value is not None
            else set()
        )
        before_names = rhs_names.intersection(self.module_names)
        before_methods = set(self.changed_methods)
        before_safe = rhs_names.intersection(self.safe_names)
        pending = [(target, value) for target in reversed(targets)]
        while pending:
            target, source = pending.pop()
            if isinstance(target, (ast.Tuple, ast.List)):
                if not isinstance(statement, ast.Delete) and not (
                    isinstance(source, (ast.Tuple, ast.List))
                    and len(target.elts) == len(source.elts)
                    and not any(isinstance(item, ast.Starred) for item in source.elts)
                    and not any(isinstance(item, ast.Starred) for item in target.elts)
                ):
                    self._invalidate()
                sources = (
                    list(source.elts)
                    if isinstance(source, (ast.Tuple, ast.List))
                    and len(target.elts) == len(source.elts)
                    and not any(isinstance(item, ast.Starred) for item in source.elts)
                    else [None] * len(target.elts)
                )
                pending.extend(reversed(list(zip(target.elts, sources, strict=True))))
            elif isinstance(target, ast.Starred):
                pending.append((target.value, None))
            elif isinstance(target, ast.Name):
                self_store = isinstance(source, ast.Name) and source.id == target.id
                releases_unsafe = (
                    target.id in self.bound_names
                    and target.id not in self.safe_names
                    and target.id not in self.module_names
                    and not self_store
                )
                safe_value = source is not None and _is_finalizer_safe_value(source, before_safe)
                self.module_names.discard(target.id)
                if isinstance(source, ast.Name) and source.id in before_names:
                    self.module_names.add(target.id)
                self.bound_names.add(target.id)
                self.safe_names.discard(target.id)
                if safe_value:
                    self.safe_names.add(target.id)
                if releases_unsafe:
                    self._invalidate()
            elif (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in self.module_names
            ):
                if target.attr not in _CACHED_SUBPROCESS_API_SLOTS or self.protocol_unsafe:
                    self.protocol_unsafe |= target.attr.startswith("__")
                    self._invalidate()
                    continue
                preserves_slot = (
                    isinstance(source, ast.Attribute)
                    and isinstance(source.value, ast.Name)
                    and source.value.id in before_names
                    and source.attr == target.attr
                )
                if preserves_slot and target.attr not in before_methods:
                    self.changed_methods.discard(target.attr)
                elif not preserves_slot and target.attr in before_methods:
                    # Releasing a previously replaced slot can run an arbitrary
                    # finalizer after the new store and restore its old value.
                    self._invalidate()
                else:
                    self.changed_methods.add(target.attr)
            else:
                self._invalidate()
        if isinstance(statement, ast.AnnAssign) and not _annotation_is_passive(
            statement.annotation
        ):
            self._invalidate()


class _Analyzer:
    def __init__(self, file_path: str, python_ast: ParsedPythonFile) -> None:
        self.file_path = file_path
        self.python_ast = python_ast
        self.lines = python_ast.lines
        self.findings: list[AnalyzerFinding] = []
        self.bound_shell_call_ownership: dict[BoundShellCallKey, bool] = {}
        self.emitted_shell_calls: set[BoundShellCallKey] = set()
        self.cached_replacement_by_call: dict[BoundShellCallKey, bool] = {}
        self.cached_subprocess = _CachedSubprocessState()

    def _record_cached_replacement(self, call: ast.Call) -> None:
        """Keep explicit cached-slot evidence distinct from receiver uncertainty."""
        shell = next((item.value for item in call.keywords if item.arg == "shell"), None)
        if not _is_direct_subprocess_syntax(call) or not isinstance(shell, ast.Name):
            return
        key = _bound_shell_call_key(call)
        blocked = self.cached_subprocess.blocks(call)
        self.cached_replacement_by_call[key] = (
            self.cached_replacement_by_call.get(key, True) and blocked
        )

    def _record_eager_cached_replacements(self, expression: ast.expr) -> None:
        """Visit eager expression order until an unmodeled effect or scope boundary."""
        pending: list[ast.expr | None] = [expression]
        while pending:
            current = pending.pop()
            if current is None:
                return
            if isinstance(current, ast.Call):
                function = current.func
                passive_lookup = isinstance(function, ast.Name) or (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id in self.cached_subprocess.module_names
                )
                if passive_lookup:
                    self._record_cached_replacement(current)
                children = [function, *current.args, *(item.value for item in current.keywords)]
                pending.append(None)
                pending.extend(reversed(children))
            elif isinstance(current, (ast.Name, ast.Constant)):
                continue
            elif isinstance(current, (ast.Tuple, ast.List)):
                pending.extend(reversed(current.elts))
            elif isinstance(current, ast.Attribute):
                if not (
                    isinstance(current.value, ast.Name)
                    and current.value.id in self.cached_subprocess.module_names
                ):
                    pending.append(None)
                pending.append(current.value)
            else:
                return

    def _record_bound_shell_call(self, call: ast.Call, trusted_names: set[str]) -> None:
        """Record whether the companion owns one supported bound-shell call."""
        shell = next((item.value for item in call.keywords if item.arg == "shell"), None)
        if not _is_direct_subprocess_syntax(call) or not isinstance(shell, ast.Name):
            return
        self._record_cached_replacement(call)
        key = _bound_shell_call_key(call)
        blocked = self.cached_subprocess.blocks(call)
        self.bound_shell_call_ownership[key] = bool(
            _is_direct_subprocess_call(call, trusted_names)
            and not blocked
            and _shell_argument_is_captured_before_effects(call)
        )

    def _inspect_call(self, call: ast.Call, facts: dict[str, bool]) -> None:
        shell = next((item.value for item in call.keywords if item.arg == "shell"), None)
        if (
            not isinstance(shell, ast.Name)
            or self.cached_subprocess.blocks(call)
            or shell.id.casefold().startswith("true")
            or facts.get(shell.id) is not True
        ):
            return
        self.emitted_shell_calls.add(_bound_shell_call_key(call))
        line = getattr(call, "lineno", 1)
        end_line = getattr(call, "end_lineno", None)
        start_byte_column = getattr(call, "col_offset", 0)
        end_byte_column = getattr(call, "end_col_offset", start_byte_column)
        start_column = self.python_ast.character_column(line, start_byte_column)
        end_column = self.python_ast.character_column(end_line or line, end_byte_column)
        complete_match = self.python_ast.source_segment(call)
        if complete_match is None:
            complete_match = get_complete_source_segment(self.lines, line, end_line)
        self.findings.append(
            AnalyzerFinding(
                rule_id="TM1",
                message="Tool Parameter Abuse",
                severity=Severity.HIGH,
                location=Location(
                    file=self.file_path,
                    start_line=line,
                    end_line=end_line,
                    start_column=start_column,
                    end_column=end_column,
                ),
                confidence=0.8,
                tags=[PatternCategory.TOOL_MISUSE.value],
                context=get_context_from_lines(
                    self.lines,
                    line,
                    column=start_column if start_column is not None else 0,
                ),
                matched_text=complete_match[:200],
                complete_match=complete_match,
                evidence={BOUND_SHELL_EVIDENCE: True},
            )
        )

    def _scan_assignment(
        self,
        targets: list[ast.expr],
        value: ast.expr,
        facts: dict[str, bool],
        trusted_names: set[str],
        bound_names: set[str],
        finalizer_safe_names: set[str],
        unknown_unsafe_bindings: list[bool],
    ) -> None:
        simple_targets = all(isinstance(target, ast.Name) for target in targets)
        releases_unsafe_value = simple_targets and any(
            target.id in bound_names
            and target.id not in finalizer_safe_names
            and not (isinstance(value, ast.Name) and value.id == target.id)
            for target in targets
            if isinstance(target, ast.Name)
        )
        result_is_finalizer_safe = _is_finalizer_safe_value(value, finalizer_safe_names)
        call_has_protocol_effects = False
        preserves_receiver_trust = simple_targets and _value_preserves_receiver_trust(
            value,
            trusted_names,
            finalizer_safe_names,
        )
        if isinstance(value, ast.Call):
            self._record_bound_shell_call(value, trusted_names)
        if isinstance(value, ast.Call) and _is_direct_subprocess_call(value, trusted_names):
            resolved = None
            safe_value = _call_arguments_are_passive(value)
            if _shell_argument_is_captured_before_effects(value):
                self._inspect_call(value, facts)
            if safe_value:
                call_has_protocol_effects = not _call_arguments_are_protocol_safe(
                    value,
                    finalizer_safe_names,
                )
        else:
            resolved = _truth_value(value, facts)
            safe_value = resolved is not None or _is_passive_argument(value)

        if not safe_value or not simple_targets:
            facts.clear()
            finalizer_safe_names.clear()
            for target in targets:
                if isinstance(target, ast.Name):
                    bound_names.add(target.id)
            if not preserves_receiver_trust:
                trusted_names.clear()
                unknown_unsafe_bindings[0] = True
            else:
                trusted_names.difference_update(
                    _changed_direct_names([value, *targets], trusted_names)
                )
            return
        if releases_unsafe_value:
            facts.clear()
            finalizer_safe_names.clear()
            trusted_names.clear()
            unknown_unsafe_bindings[0] = True
        if call_has_protocol_effects:
            facts.clear()
            finalizer_safe_names.clear()
            trusted_names.clear()
            unknown_unsafe_bindings[0] = True
        for target in targets:
            assert isinstance(target, ast.Name)
            bound_names.add(target.id)
            if releases_unsafe_value or call_has_protocol_effects or resolved is None:
                facts.pop(target.id, None)
            else:
                facts[target.id] = resolved
            if releases_unsafe_value or call_has_protocol_effects or not result_is_finalizer_safe:
                finalizer_safe_names.discard(target.id)
            else:
                finalizer_safe_names.add(target.id)
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
        initial_bound_names: set[str] | None = None,
        nested_function_trusted_names: set[str] | None = None,
        nested_function_trusted_at_call: dict[int, set[str]] | None = None,
        cached_subprocess: _CachedSubprocessState | None = None,
    ) -> None:
        previous_cached = self.cached_subprocess
        self.cached_subprocess = (cached_subprocess or previous_cached).copy()
        initial_cached = self.cached_subprocess.copy()
        trusted_names = set(_DIRECT_CALL_NAMES if trusted_names is None else trusted_names)
        facts: dict[str, bool] = {}
        bound_names = set(initial_bound_names or ())
        finalizer_safe_names: set[str] = set()
        unknown_unsafe_bindings = [False]

        last_invalidation_by_name: dict[str, int] = {}
        receiver_trust = set(trusted_names)
        receiver_bound_names = set(initial_bound_names or ())
        receiver_finalizer_safe_names: set[str] = set()
        receiver_unknown_unsafe_bindings = [False]
        for candidate_index, candidate in enumerate(statements):
            before = set(receiver_trust)
            _advance_trusted_names(
                candidate,
                receiver_trust,
                receiver_bound_names,
                receiver_finalizer_safe_names,
                receiver_unknown_unsafe_bindings,
            )
            for name in before.difference(receiver_trust):
                last_invalidation_by_name[name] = candidate_index

        trusted_at_call_by_definition: dict[int, set[str]] = {}
        receiver_trust = set(trusted_names)
        receiver_bound_names = set(initial_bound_names or ())
        receiver_finalizer_safe_names = set()
        receiver_unknown_unsafe_bindings = [False]
        active_functions: dict[str, int] = {}
        cached_at_call_by_definition: dict[int, _CachedSubprocessState] = {}
        deferred_cached = initial_cached.copy()
        for candidate_index, candidate in enumerate(statements):
            call = _passive_direct_call(candidate)
            if call is not None:
                assert isinstance(call.func, ast.Name)
                owner = active_functions.get(call.func.id)
                if owner is not None:
                    trusted_at_call_by_definition.setdefault(owner, set()).update(receiver_trust)
                    prior_cached = cached_at_call_by_definition.get(owner)
                    if prior_cached is None:
                        cached_at_call_by_definition[owner] = deferred_cached.copy()
                    else:
                        prior_cached.changed_methods.intersection_update(
                            deferred_cached.changed_methods
                        )
                        prior_cached.changed_popen &= deferred_cached.changed_popen
                        prior_cached.effect_generation = max(
                            prior_cached.effect_generation, deferred_cached.effect_generation
                        )

            changed_names = _direct_bound_names(candidate)
            for name in changed_names:
                active_functions.pop(name, None)
            if (
                isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _function_header_is_passive(candidate)
                and _is_immediate_function(candidate)
            ):
                active_functions[candidate.name] = candidate_index
            _advance_trusted_names(
                candidate,
                receiver_trust,
                receiver_bound_names,
                receiver_finalizer_safe_names,
                receiver_unknown_unsafe_bindings,
            )
            deferred_cached.advance(candidate)

        for index, statement in enumerate(statements):
            if isinstance(statement, (ast.Expr, ast.Return, ast.Assign, ast.AnnAssign)):
                if statement.value is not None:
                    self._record_eager_cached_replacements(statement.value)
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                passive_header = _function_header_is_passive(statement)
                if nested_function_trusted_names is None:
                    trusted_at_call = trusted_at_call_by_definition.get(index, set())
                    nested_trusted_names = set(trusted_names).union(trusted_at_call)
                    nested_trusted_names = {
                        name
                        for name in nested_trusted_names
                        if last_invalidation_by_name.get(name, -1) <= index
                        or name in trusted_at_call
                    }
                else:
                    nested_trusted_names = set(nested_function_trusted_names).union(
                        (nested_function_trusted_at_call or {}).get(index, set())
                    )
                nested_trusted_names.difference_update(
                    _function_bound_direct_names(statement, nested_trusted_names)
                )
                nested_trusted_names.discard(statement.name)
                if not passive_header:
                    nested_trusted_names.clear()
                nested_cached = cached_at_call_by_definition.get(index, deferred_cached).copy()
                nested_cached.module_names.difference_update(
                    _function_bound_direct_names(statement, nested_cached.module_names)
                )
                local_bound = _function_bound_direct_names(statement, nested_cached.bound_names)
                nested_cached.bound_names.difference_update(local_bound)
                nested_cached.safe_names.difference_update(local_bound)
                parameters = _function_parameter_names(statement)
                nested_cached.bound_names.update(parameters)
                nested_cached.safe_names.difference_update(parameters)
                self._scan_block(
                    statement.body,
                    trusted_names=nested_trusted_names,
                    initial_bound_names=_function_parameter_names(statement),
                    cached_subprocess=nested_cached,
                )
                releases_unsafe_value = (
                    statement.name in bound_names and statement.name not in finalizer_safe_names
                )
                if passive_header and not releases_unsafe_value:
                    facts.pop(statement.name, None)
                else:
                    facts.clear()
                    finalizer_safe_names.clear()
                    trusted_names.clear()
                    unknown_unsafe_bindings[0] = True
                bound_names.add(statement.name)
                finalizer_safe_names.discard(statement.name)
                trusted_names.discard(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                imported_names = _direct_bound_names(statement)
                releases_unsafe_value = any(
                    name in bound_names and name not in finalizer_safe_names
                    for name in imported_names
                )
                facts.clear()
                finalizer_safe_names.difference_update(imported_names)
                bound_names.update(imported_names)
                _update_trusted_names_from_import(statement, trusted_names)
                imports_unknown_names = isinstance(statement, ast.ImportFrom) and any(
                    imported.name == "*" for imported in statement.names
                )
                if releases_unsafe_value or imports_unknown_names:
                    unknown_unsafe_bindings[0] = True
                if releases_unsafe_value or unknown_unsafe_bindings[0]:
                    trusted_names.clear()
            elif isinstance(statement, ast.Assign):
                self._scan_assignment(
                    list(statement.targets),
                    statement.value,
                    facts,
                    trusted_names,
                    bound_names,
                    finalizer_safe_names,
                    unknown_unsafe_bindings,
                )
            elif isinstance(statement, ast.AnnAssign):
                value = statement.value
                target = statement.target
                simple_target = isinstance(target, ast.Name)
                releases_unsafe_value = (
                    value is not None
                    and simple_target
                    and target.id in bound_names
                    and target.id not in finalizer_safe_names
                    and not (isinstance(value, ast.Name) and value.id == target.id)
                )
                result_is_finalizer_safe = value is not None and _is_finalizer_safe_value(
                    value,
                    finalizer_safe_names,
                )
                preserves_receiver_trust = (
                    simple_target
                    and _annotation_is_passive(statement.annotation)
                    and (
                        value is None
                        or _value_preserves_receiver_trust(
                            value,
                            trusted_names,
                            finalizer_safe_names,
                        )
                    )
                )
                if isinstance(value, ast.Call):
                    self._record_bound_shell_call(value, trusted_names)
                if isinstance(value, ast.Call) and _is_direct_subprocess_call(value, trusted_names):
                    if _shell_argument_is_captured_before_effects(value):
                        self._inspect_call(value, facts)
                facts.clear()
                if not preserves_receiver_trust or releases_unsafe_value:
                    finalizer_safe_names.clear()
                if value is not None:
                    bound_names.update(_direct_bound_names(statement))
                    if (
                        simple_target
                        and preserves_receiver_trust
                        and not releases_unsafe_value
                        and result_is_finalizer_safe
                    ):
                        finalizer_safe_names.add(target.id)
                    elif simple_target:
                        finalizer_safe_names.discard(target.id)
                if not preserves_receiver_trust or releases_unsafe_value:
                    trusted_names.clear()
                    unknown_unsafe_bindings[0] = True
                else:
                    trusted_names.difference_update(
                        _changed_direct_names([statement], trusted_names)
                    )
            elif isinstance(statement, (ast.AugAssign, ast.Delete)):
                facts.clear()
                finalizer_safe_names.clear()
                bound_names.update(_direct_bound_names(statement))
                trusted_names.clear()
                unknown_unsafe_bindings[0] = True
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                self._record_bound_shell_call(call, trusted_names)
                direct_call = _is_direct_subprocess_call(call, trusted_names)
                if direct_call and _shell_argument_is_captured_before_effects(call):
                    self._inspect_call(call, facts)
                arguments_are_passive = _call_arguments_are_passive(call)
                if direct_call and arguments_are_passive:
                    if not _call_arguments_are_protocol_safe(call, finalizer_safe_names):
                        facts.clear()
                        finalizer_safe_names.clear()
                        trusted_names.clear()
                        unknown_unsafe_bindings[0] = True
                else:
                    facts.clear()
                    finalizer_safe_names.clear()
                    trusted_names.clear()
                    unknown_unsafe_bindings[0] = True
            elif isinstance(statement, (ast.Global, ast.Nonlocal, ast.Pass)) or (
                isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            ):
                continue
            elif isinstance(statement, ast.Expr):
                preserves_receiver_trust = _value_preserves_receiver_trust(
                    statement.value,
                    trusted_names,
                    finalizer_safe_names,
                )
                bound_names.update(_direct_bound_names(statement))
                if preserves_receiver_trust:
                    trusted_names.difference_update(
                        _changed_direct_names([statement], trusted_names)
                    )
                else:
                    facts.clear()
                    finalizer_safe_names.clear()
                    trusted_names.clear()
                    unknown_unsafe_bindings[0] = True
            elif isinstance(statement, ast.Assert):
                expressions = [statement.test]
                if statement.msg is not None:
                    expressions.append(statement.msg)
                preserves_receiver_trust = all(
                    _value_preserves_receiver_trust(
                        item,
                        trusted_names,
                        finalizer_safe_names,
                    )
                    for item in expressions
                ) and _is_finalizer_safe_value(statement.test, finalizer_safe_names)
                facts.clear()
                if not preserves_receiver_trust:
                    finalizer_safe_names.clear()
                    trusted_names.clear()
                    unknown_unsafe_bindings[0] = True
            elif isinstance(statement, ast.ClassDef):
                releases_unsafe_value = (
                    statement.name in bound_names and statement.name not in finalizer_safe_names
                )
                class_trusted_names = set(trusted_names)
                passive_class_header = _class_header_is_passive(statement)
                if not passive_class_header:
                    class_trusted_names.clear()
                method_outer_trust = set(
                    trusted_names
                    if nested_function_trusted_names is None
                    else nested_function_trusted_names
                )
                if nested_function_trusted_names is None:
                    method_outer_trust = {
                        name
                        for name in method_outer_trust
                        if last_invalidation_by_name.get(name, -1) <= index
                    }
                if not passive_class_header:
                    method_outer_trust.clear()
                method_trusted_names, method_trusted_at_call = _class_deferred_receiver_trust(
                    statement,
                    method_outer_trust,
                )
                self._scan_block(
                    statement.body,
                    trusted_names=class_trusted_names,
                    nested_function_trusted_names=method_trusted_names,
                    nested_function_trusted_at_call=method_trusted_at_call,
                    cached_subprocess=self.cached_subprocess,
                )
                facts.clear()
                bound_names.update(_direct_bound_names(statement))
                unsafe_class_execution = (
                    not passive_class_header
                    or releases_unsafe_value
                    or _class_body_has_eager_effects(statement)
                )
                if unsafe_class_execution:
                    finalizer_safe_names.clear()
                    trusted_names.clear()
                    unknown_unsafe_bindings[0] = True
                else:
                    finalizer_safe_names.discard(statement.name)
                    trusted_names.difference_update(
                        _changed_direct_names([statement], trusted_names)
                    )
                    trusted_names.difference_update(
                        _class_body_changed_direct_names(statement, trusted_names)
                    )
            else:
                facts.clear()
                finalizer_safe_names.clear()
                bound_names.update(_direct_bound_names(statement))
                trusted_names.clear()
                unknown_unsafe_bindings[0] = True
            self.cached_subprocess.advance(statement)

        self.cached_subprocess = previous_cached

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
    return _Analyzer(file_path, parsed).run(parsed.tree)


def bound_shell_call_analysis(
    file_path: str,
    python_ast: ParsedPythonFile,
) -> tuple[dict[BoundShellCallKey, bool], set[BoundShellCallKey], set[BoundShellCallKey]]:
    """Separate trust, detections, and affirmative cached-slot replacements."""
    if python_ast.tree is None:
        return {}, set(), set()
    analyzer = _Analyzer(file_path, python_ast)
    analyzer.run(python_ast.tree)
    return (
        dict(analyzer.bound_shell_call_ownership),
        set(analyzer.emitted_shell_calls),
        {key for key, replaced in analyzer.cached_replacement_by_call.items() if replaced},
    )


def bound_shell_call_state(
    file_path: str,
    python_ast: ParsedPythonFile,
) -> tuple[dict[BoundShellCallKey, bool], set[BoundShellCallKey]]:
    """Return receiver trust separately from affirmative companion detections."""
    ownership, emitted, _ = bound_shell_call_analysis(file_path, python_ast)
    return ownership, emitted


def bound_shell_call_ownership(
    file_path: str,
    python_ast: ParsedPythonFile,
) -> dict[BoundShellCallKey, bool]:
    """Return supported bound-shell calls and whether their receiver is trusted."""
    if python_ast.tree is None:
        return {}
    analyzer = _Analyzer(file_path, python_ast)
    analyzer.run(python_ast.tree)
    return dict(analyzer.bound_shell_call_ownership)
