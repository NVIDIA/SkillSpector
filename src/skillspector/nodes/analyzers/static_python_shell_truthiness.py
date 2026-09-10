# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find direct subprocess calls using a definitely truthy local name.

This companion deliberately recognizes only the straight-line form reported in
issue #475. Call arguments must be passive, and unsupported expressions or
compound statements (including class bodies) discard facts rather than guessing
about Python execution.
"""

from __future__ import annotations

import ast
import re
from array import array
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from skillspector.artifacts import (
    normalized_security_prefix,
    normalized_security_view,
    security_text_views,
)
from skillspector.models import AnalyzerFinding, Location, Severity
from skillspector.python_ast import ParsedPythonFile, parse_python_source

from .common import LINE_BREAK_CHARS, get_context_from_lines, get_source_segment
from .pattern_defaults import PatternCategory

ANALYZER_ID = "static_patterns_tool_misuse"
USES_PYTHON_AST = True
BOUND_SHELL_EVIDENCE = "_tm1_bound_shell_value"
BOUND_CALL_START_EVIDENCE = "_tm1_bound_call_start"
BOUND_CALL_END_EVIDENCE = "_tm1_bound_call_end"
BOUND_SHELL_ANCHOR_EVIDENCE = "_tm1_bound_shell_anchor"
BOUND_CANONICAL_FINGERPRINT_EVIDENCE = "_tm1_bound_canonical_fingerprint"
BOUND_NORMALIZED_VIEW_EVIDENCE = "_tm1_bound_normalized_view"
BOUND_CLASSIFICATION_MATCH_EVIDENCE = "_tm1_bound_classification_match"
BOUND_DIRECT_MATCH_END_EVIDENCE = "_tm1_bound_direct_match_end"
BOUND_POPEN_START_EVIDENCE = "_tm1_bound_popen_start"
BOUND_DIRECT_OWNER_START_EVIDENCE = "_tm1_bound_direct_owner_start"
BOUND_SHELL_VALUE_START_EVIDENCE = "_tm1_bound_shell_value_start"
BOUND_SHELL_VALUE_END_EVIDENCE = "_tm1_bound_shell_value_end"
DIRECT_LITERAL_METADATA_EVIDENCE = "_tm1_direct_literal_metadata"
_SOURCE_LINE_BREAK = re.compile(r"\r\n|\n|\r")
_DIRECT_CALL_NAMES = frozenset({"subprocess", "Popen"})
_DIRECT_CALLEE = re.compile(r"(?:subprocess\.\w+|Popen)", re.IGNORECASE)
_SHELL_KEYWORD_PREFIX = re.compile(r"shell\s*=\s*", re.IGNORECASE)
_MAX_CONTEXT_CHARS = 1024
_MAX_FINGERPRINT_CHARS = 200
_MAX_DIRECT_NAME_CHARS = len("subprocess") + 1


@dataclass(frozen=True, slots=True)
class DirectLiteralMetadata:
    """AST metadata attached only to an already-confirmed lexical finding."""

    call_start: int
    call_end: int | None
    shell_anchor: int
    match_fingerprint: str
    normalized_view: bool


@dataclass(frozen=True, slots=True)
class BoundShellMetadata:
    """Cap-independent source coordinates for one supported bound shell call."""

    call_start: int
    call_end: int | None
    shell_anchor: int
    value_start: int
    value_end: int
    popen_start: int | None
    canonical_fingerprint: str
    normalized_view: bool


@dataclass(slots=True)
class _BlockEffectState:
    """Monotonic fail-closed state for arbitrary effects in one lexical block."""

    arbitrary_effects_seen: bool = False


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
    bound_names: set[str],
    finalizer_safe_names: set[str],
    effect_state: _BlockEffectState,
) -> None:
    """Apply import-hook and sequential STORE effects to receiver trust."""
    if isinstance(statement, ast.Import):
        for imported in statement.names:
            # Each IMPORT_NAME can run a distinct finder/loader before its
            # corresponding STORE.  Once an arbitrary alias has run, retain a
            # monotonic taint so a later exact spelling cannot certify state
            # that the earlier hook may have poisoned.
            trusted_names.clear()
            bound = imported.asname or imported.name.partition(".")[0]
            releases_unsafe = bound in bound_names and bound not in finalizer_safe_names
            trusted_alias = imported.name == "subprocess" and bound == "subprocess"
            if not trusted_alias:
                effect_state.arbitrary_effects_seen = True
                finalizer_safe_names.clear()
            bound_names.add(bound)
            finalizer_safe_names.discard(bound)
            if trusted_alias and not releases_unsafe and not effect_state.arbitrary_effects_seen:
                trusted_names.add(bound)
                finalizer_safe_names.add(bound)
            if releases_unsafe:
                # STORE_NAME decrefs the displaced value after installing the
                # import result; its finalizer can overwrite even this exact
                # trusted binding.
                trusted_names.clear()
                finalizer_safe_names.clear()
                effect_state.arbitrary_effects_seen = True
        return
    # ImportFrom performs one arbitrary module import, followed by ordered
    # IMPORT_FROM/STORE pairs without another hook between them.
    trusted_names.clear()
    trusted_only = (
        statement.level == 0
        and statement.module == "subprocess"
        and bool(statement.names)
        and all(
            imported.name == "Popen" and (imported.asname or imported.name) == "Popen"
            for imported in statement.names
        )
    )
    if not trusted_only:
        effect_state.arbitrary_effects_seen = True
        finalizer_safe_names.clear()
    if any(imported.name == "*" for imported in statement.names):
        return
    for imported in statement.names:
        bound = imported.asname or imported.name
        releases_unsafe = bound in bound_names and bound not in finalizer_safe_names
        bound_names.add(bound)
        finalizer_safe_names.discard(bound)
        if (
            statement.level == 0
            and statement.module == "subprocess"
            and imported.name == "Popen"
            and bound == "Popen"
            and not releases_unsafe
            and not effect_state.arbitrary_effects_seen
        ):
            trusted_names.add(bound)
            finalizer_safe_names.add(bound)
        elif bound in trusted_names:
            trusted_names.discard(bound)
        if releases_unsafe:
            trusted_names.clear()
            finalizer_safe_names.clear()
            effect_state.arbitrary_effects_seen = True


class _DirectBindingCollector:
    """Collect lexical-scope bindings without recursive AST traversal."""

    def __init__(self, tracked_names: set[str] | frozenset[str]) -> None:
        self.tracked_names = tracked_names
        self.bound: set[str] = set()
        self.mutated: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def _function_header_nodes(
        self,
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
    """Return compile-time local direct names for one function scope."""
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
    """Return direct names rebound or explicitly mutated by current-scope nodes."""
    collector = _DirectBindingCollector(tracked_names)
    for node in nodes:
        collector.visit(node)
    return collector.bound.union(collector.mutated)


def _class_body_changed_direct_names(
    statement: ast.ClassDef,
    tracked_names: set[str],
) -> set[str]:
    """Return explicit class execution effects on module/global receivers."""

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

        # Class bodies use runtime ``LOAD_NAME`` lookup rather than function-style
        # compile-time locals. Track whether a class-local spelling still aliases
        # the genuine outer receiver so a later attribute store is classified
        # correctly (``subprocess = Proxy()`` differs from
        # ``subprocess = subprocess``).
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
                # A compound or expression-level binding may not execute on every
                # path. Forget class-local provenance so a subsequent rooted
                # mutation is conservatively treated as reaching the outer object.
                for name in local_bound:
                    local_direct.pop(name, None)
    return affected


def _normalized_direct_name(name: str) -> str | None:
    normalized = normalized_security_prefix(name, _MAX_DIRECT_NAME_CHARS).casefold()
    return normalized if normalized in {"subprocess", "popen"} else None


def _is_direct_subprocess_call(call: ast.Call, trusted_names: set[str]) -> bool:
    function = call.func
    if isinstance(function, ast.Name):
        canonical = _normalized_direct_name(function.id)
        return canonical == "popen" and function.id in trusted_names
    if not isinstance(function, ast.Attribute) or not isinstance(function.value, ast.Name):
        return False
    receiver = function.value.id
    canonical = _normalized_direct_name(receiver)
    return canonical == "subprocess" and receiver in trusted_names


def _is_passive_argument(expression: ast.expr) -> bool:
    """Return whether evaluating an argument cannot invoke user-controlled code."""
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


def _is_protocol_safe_argument(
    expression: ast.expr,
    protocol_safe_names: set[str],
) -> bool:
    """Accept only exact built-in values that cannot dispatch protocol hooks."""
    pending = [expression]
    while pending:
        current = pending.pop()
        if _is_constant_operator_expression(current):
            continue
        if isinstance(current, ast.Name):
            if current.id not in protocol_safe_names:
                return False
            continue
        if isinstance(current, (ast.List, ast.Tuple, ast.Set)):
            if any(isinstance(item, ast.Starred) for item in current.elts):
                return False
            pending.extend(current.elts)
            continue
        if isinstance(current, ast.Dict):
            if any(key is None for key in current.keys):
                return False
            pending.extend(key for key in current.keys if key is not None)
            pending.extend(current.values)
            continue
        if isinstance(current, ast.JoinedStr) and all(
            isinstance(item, ast.Constant) for item in current.values
        ):
            continue
        return False
    return True


def _call_arguments_are_protocol_safe(
    call: ast.Call,
    facts: dict[str, bool],
    protocol_safe_names: set[str],
) -> bool:
    """Return whether call execution cannot invoke hooks on supplied values."""
    if not all(_is_protocol_safe_argument(argument, protocol_safe_names) for argument in call.args):
        return False
    for keyword in call.keywords:
        if keyword.arg is None:
            return False
        if (
            keyword.arg == "shell"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in facts
        ):
            # A proven truth value is sufficient for ``shell`` itself: Python
            # tests only the outer value, so exact built-in container truth
            # does not inspect potentially opaque elements.
            continue
        if not _is_protocol_safe_argument(keyword.value, protocol_safe_names):
            return False
    return True


def _target_names(target: ast.expr) -> set[str]:
    """Return simple names stored or deleted through one assignment target."""
    names: set[str] = set()
    pending = [target]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Name):
            names.add(current.id)
        elif isinstance(current, ast.Starred):
            pending.append(current.value)
        elif isinstance(current, (ast.List, ast.Tuple)):
            pending.extend(current.elts)
    return names


def _current_scope_bound_names(statement: ast.stmt) -> set[str]:
    """Collect every name STORE reachable in this lexical scope."""
    candidates: set[str] = set()
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            candidates.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.add(node.name)
        elif isinstance(node, ast.Import):
            candidates.update(
                imported.asname or imported.name.partition(".")[0] for imported in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            candidates.update(
                imported.asname or imported.name for imported in node.names if imported.name != "*"
            )
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            candidates.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and isinstance(node.name, str):
            candidates.add(node.name)
        elif isinstance(node, ast.MatchMapping) and isinstance(node.rest, str):
            candidates.add(node.rest)
    collector = _DirectBindingCollector(candidates)
    collector.visit(statement)
    return collector.bound


def _statement_name_transitions(statement: ast.stmt) -> tuple[set[str], set[str]]:
    """Return direct name stores and deletes performed by one statement."""
    stored = _current_scope_bound_names(statement)
    deleted: set[str] = set()
    if isinstance(statement, ast.AnnAssign) and statement.value is None:
        stored.difference_update(_target_names(statement.target))
    if isinstance(statement, ast.Delete):
        for target in statement.targets:
            deleted.update(_target_names(target))
        stored.difference_update(deleted)
    return stored, deleted


def _statement_releases_unsafe_binding(
    statement: ast.stmt,
    bound_names: set[str],
    finalizer_safe_names: set[str],
) -> bool:
    """Return whether a STORE/DELETE can invoke an opaque prior finalizer."""
    stored, deleted = _statement_name_transitions(statement)
    unsafe_bound_names = bound_names.difference(finalizer_safe_names)
    if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name):
        # ``x = x`` keeps the loaded object alive across STORE_NAME, so the
        # displaced reference cannot be its final reference at that point.
        stored.difference_update(
            target.id
            for target in statement.targets
            if isinstance(target, ast.Name) and target.id == statement.value.id
        )
    if isinstance(statement, ast.ImportFrom) and any(
        imported.name == "*" for imported in statement.names
    ):
        # A wildcard import may overwrite any existing module binding.
        return bool(unsafe_bound_names)
    return bool((stored | deleted).intersection(unsafe_bound_names))


def _advance_finalizer_provenance(
    statement: ast.stmt,
    bound_names: set[str],
    finalizer_safe_names: set[str],
    *,
    invalidated: bool,
) -> None:
    """Track bindings whose later release cannot dispatch ``__del__`` hooks."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        # Import receiver handling applies each hook and STORE in runtime order
        # and updates these sets in place.
        return
    stored, deleted = _statement_name_transitions(statement)
    prior_safe_names = set(finalizer_safe_names)
    if invalidated:
        # Arbitrary evaluation can replace any currently known binding through
        # frame/global access.  Retain that it is bound, but no longer certify
        # the value released by a later STORE/DELETE.
        finalizer_safe_names.clear()

    new_safe_names: set[str] = set()
    if not invalidated:
        if isinstance(statement, ast.Assign) and all(
            isinstance(target, ast.Name) for target in statement.targets
        ):
            if _is_protocol_safe_argument(statement.value, prior_safe_names):
                new_safe_names.update(stored)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            _function_result_is_finalizer_safe(statement, prior_safe_names)
        ):
            new_safe_names.update(stored)
        elif isinstance(statement, ast.ClassDef) and _class_result_is_finalizer_safe(
            statement,
            prior_safe_names,
        ):
            new_safe_names.update(stored)

    finalizer_safe_names.difference_update(stored | deleted)
    bound_names.difference_update(deleted)
    bound_names.update(stored)
    finalizer_safe_names.update(new_safe_names)


def _is_immediate_function(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a direct call starts executing this body immediately."""
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


def _direct_call_value(statement: ast.stmt) -> ast.Call | None:
    """Return a call evaluated directly by this simple statement, if any."""
    value: ast.expr | None = None
    if isinstance(statement, (ast.Expr, ast.Assign)):
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        value = statement.value
    if not isinstance(value, ast.Call):
        return None
    return value


def _passive_direct_call(statement: ast.stmt) -> ast.Call | None:
    """Return a definitely evaluated simple-name call with passive arguments."""
    value = _direct_call_value(statement)
    if value is None:
        return None
    if not isinstance(value.func, ast.Name) or not _call_arguments_are_passive(value):
        return None
    return value


def _is_constant_operator_expression(expression: ast.expr) -> bool:
    """Accept operator trees whose operands are exact built-in constants."""
    pending = [expression]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Constant):
            continue
        if isinstance(current, ast.BinOp):
            pending.extend((current.left, current.right))
            continue
        if isinstance(current, ast.UnaryOp):
            pending.append(current.operand)
            continue
        return False
    return True


def _expression_preserves_receiver_trust(
    expression: ast.expr,
    trusted_names: set[str],
    facts: dict[str, bool],
    protocol_safe_names: set[str],
) -> bool:
    """Return whether evaluating an expression cannot mutate receiver globals."""
    if _is_passive_argument(expression) or _is_constant_operator_expression(expression):
        return True
    return (
        isinstance(expression, ast.Call)
        and _is_direct_subprocess_call(expression, trusted_names)
        and _call_arguments_are_protocol_safe(expression, facts, protocol_safe_names)
    )


def _statement_requires_receiver_barrier(
    statement: ast.stmt,
    trusted_names: set[str],
    facts: dict[str, bool],
    protocol_safe_names: set[str],
) -> bool:
    """Return whether statement evaluation may mutate either trusted receiver."""
    if isinstance(statement, ast.Assign):
        return any(not isinstance(target, ast.Name) for target in statement.targets) or not (
            _expression_preserves_receiver_trust(
                statement.value,
                trusted_names,
                facts,
                protocol_safe_names,
            )
        )
    if isinstance(statement, ast.AnnAssign):
        # Module annotations write through a replaceable ``__annotations__``
        # mapping, even when the value and target otherwise look passive.
        return True
    if isinstance(statement, ast.AugAssign):
        # In-place operators and attribute/subscript stores can invoke user code.
        return True
    if isinstance(statement, ast.Delete):
        return any(not isinstance(target, ast.Name) for target in statement.targets)
    if isinstance(statement, ast.Expr):
        return not _expression_preserves_receiver_trust(
            statement.value,
            trusted_names,
            facts,
            protocol_safe_names,
        )
    if isinstance(statement, (ast.Global, ast.Nonlocal, ast.Pass)):
        return False
    # Unsupported control flow and runtime statements are outside this
    # deliberately straight-line, side-effect-free subset.
    return True


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


def _function_initial_bound_names(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Return parameters and outer-scope declarations bound on body entry."""
    arguments = statement.args
    names = {
        argument.arg
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    declaration_collector = _DirectBindingCollector(frozenset())
    for child in statement.body:
        declaration_collector.visit(child)
    names.update(declaration_collector.nonlocal_names)
    return names


def _function_result_is_finalizer_safe(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    finalizer_safe_names: set[str],
) -> bool:
    """Return whether releasing the created function retains only safe values."""
    if not _function_header_is_passive(statement):
        return False
    retained = [
        *statement.args.defaults,
        *(item for item in statement.args.kw_defaults if item is not None),
    ]
    arguments = (
        *statement.args.posonlyargs,
        *statement.args.args,
        *statement.args.kwonlyargs,
    )
    retained.extend(
        argument.annotation for argument in arguments if argument.annotation is not None
    )
    if statement.args.vararg is not None and statement.args.vararg.annotation is not None:
        retained.append(statement.args.vararg.annotation)
    if statement.args.kwarg is not None and statement.args.kwarg.annotation is not None:
        retained.append(statement.args.kwarg.annotation)
    if statement.returns is not None:
        retained.append(statement.returns)
    return all(
        _is_protocol_safe_argument(expression, finalizer_safe_names) for expression in retained
    )


def _class_definition_is_passive(statement: ast.ClassDef) -> bool:
    """Accept class execution only when every evaluated form is passive."""
    pending = [statement]
    while pending:
        current = pending.pop()
        if (
            current.decorator_list
            or current.bases
            or current.keywords
            or getattr(current, "type_params", [])
        ):
            return False
        for child in current.body:
            if isinstance(child, ast.Pass):
                continue
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                # A later simple-looking class-body STORE can then replace an
                # outer opaque value and run its finalizer during class
                # execution.  Keep this boundary outside the passive subset.
                return False
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                continue
            if isinstance(child, ast.Assign) and all(
                isinstance(target, ast.Name) for target in child.targets
            ):
                if not isinstance(child.value, ast.Name) and (
                    _is_passive_argument(child.value)
                    or _is_constant_operator_expression(child.value)
                ):
                    continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                _function_header_is_passive(child)
            ):
                continue
            if isinstance(child, ast.ClassDef):
                pending.append(child)
                continue
            return False
    return True


def _class_result_is_finalizer_safe(
    statement: ast.ClassDef,
    finalizer_safe_names: set[str],
) -> bool:
    """Return whether releasing a passive class retains only safe members."""
    if not _class_definition_is_passive(statement):
        return False
    pending = [statement]
    while pending:
        current = pending.pop()
        for child in current.body:
            if isinstance(child, (ast.Pass, ast.Expr, ast.Global, ast.Nonlocal)):
                continue
            if isinstance(child, ast.Assign):
                if not _is_protocol_safe_argument(child.value, finalizer_safe_names):
                    return False
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _function_result_is_finalizer_safe(child, finalizer_safe_names):
                    return False
                continue
            if isinstance(child, ast.ClassDef):
                pending.append(child)
                continue
            return False
    return True


def _statement_requires_callable_barrier(
    statement: ast.stmt,
    trusted_names: set[str],
    facts: dict[str, bool],
    protocol_safe_names: set[str],
) -> bool:
    """Return whether statement evaluation may rebind tracked local callables."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        # Import finders, loaders, and imported module code are arbitrary
        # Python.  Even when the statement explicitly reestablishes a trusted
        # subprocess receiver, it cannot preserve a previously tracked local
        # function identity across that execution boundary.
        return True
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return not _function_header_is_passive(statement)
    if isinstance(statement, ast.ClassDef):
        return not _class_definition_is_passive(statement)
    return _statement_requires_receiver_barrier(
        statement,
        trusted_names,
        facts,
        protocol_safe_names,
    )


def _advance_trusted_names(
    statement: ast.stmt,
    trusted_names: set[str],
    facts: dict[str, bool],
    protocol_safe_names: set[str],
    bound_names: set[str],
    finalizer_safe_names: set[str],
    effect_state: _BlockEffectState,
) -> None:
    """Apply one statement's explicit receiver-binding effects."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        _update_trusted_names_from_import(
            statement,
            trusted_names,
            bound_names,
            finalizer_safe_names,
            effect_state,
        )
        return
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not _function_header_is_passive(statement):
            trusted_names.clear()
        trusted_names.discard(statement.name)
        return
    if isinstance(statement, ast.ClassDef):
        if not _class_definition_is_passive(statement):
            trusted_names.clear()
            return
        trusted_names.difference_update(_changed_direct_names([statement], trusted_names))
        trusted_names.difference_update(_class_body_changed_direct_names(statement, trusted_names))
        return
    if _statement_requires_receiver_barrier(
        statement,
        trusted_names,
        facts,
        protocol_safe_names,
    ):
        # The call-site prepass records an eligible local body before applying
        # this post-evaluation barrier, so a genuine call before a later
        # mutation remains reportable without trusting subsequent generations.
        trusted_names.clear()
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
    trusted_names.difference_update(_changed_direct_names([statement], trusted_names))


def _advance_value_provenance(
    statement: ast.stmt,
    facts: dict[str, bool],
    protocol_safe_names: set[str],
    trusted_names_before: set[str],
) -> None:
    """Apply one statement's straight-line truth and exact-value effects."""
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        facts.clear()
        protocol_safe_names.clear()
        return
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if _function_header_is_passive(statement):
            facts.pop(statement.name, None)
            protocol_safe_names.discard(statement.name)
        else:
            facts.clear()
            protocol_safe_names.clear()
        return
    if isinstance(statement, ast.Assign):
        if any(not isinstance(target, ast.Name) for target in statement.targets):
            facts.clear()
            protocol_safe_names.clear()
            return
        value = statement.value
        if (
            isinstance(value, ast.Call)
            and _is_direct_subprocess_call(value, trusted_names_before)
            and _call_arguments_are_passive(value)
        ):
            if not _call_arguments_are_protocol_safe(value, facts, protocol_safe_names):
                facts.clear()
                protocol_safe_names.clear()
                return
            resolved = None
            protocol_safe_value = False
        else:
            resolved = _truth_value(value, facts)
            if resolved is None and not _is_passive_argument(value):
                facts.clear()
                protocol_safe_names.clear()
                return
            protocol_safe_value = _is_protocol_safe_argument(value, protocol_safe_names)
        for target in statement.targets:
            assert isinstance(target, ast.Name)
            if resolved is None:
                facts.pop(target.id, None)
            else:
                facts[target.id] = resolved
            if protocol_safe_value:
                protocol_safe_names.add(target.id)
            else:
                protocol_safe_names.discard(target.id)
        return
    if isinstance(statement, ast.Expr):
        value = statement.value
        if (
            isinstance(value, ast.Call)
            and _is_direct_subprocess_call(value, trusted_names_before)
            and _call_arguments_are_passive(value)
            and _call_arguments_are_protocol_safe(value, facts, protocol_safe_names)
        ) or isinstance(value, ast.Constant):
            return
        facts.clear()
        protocol_safe_names.clear()
        return
    if isinstance(statement, (ast.Global, ast.Nonlocal, ast.Pass)):
        return
    facts.clear()
    protocol_safe_names.clear()


class _Analyzer:
    def __init__(
        self,
        file_path: str,
        content: str,
        *,
        emit_findings: bool = True,
        check_runtime: Callable[[], None] | None = None,
    ) -> None:
        self.file_path = file_path
        self.content = content
        self._emit_findings = emit_findings
        self._check_runtime = check_runtime
        self.lines = _SOURCE_LINE_BREAK.split(content)
        self._encoded_lines = tuple(line.encode("utf-8") for line in self.lines)
        self._ascii_lines = tuple(line.isascii() for line in self.lines)
        self._byte_boundaries: dict[int, array[int]] = {}
        self._line_starts = (0, *(match.end() for match in _SOURCE_LINE_BREAK.finditer(content)))
        self.findings: list[AnalyzerFinding] = []
        self._finding_keys: set[tuple[int, str]] = set()
        self.bound_shell_metadata: list[BoundShellMetadata] = []

    def _source_position(self, line: object, column: object) -> int | None:
        """Convert one AST UTF-8 byte coordinate to a character offset."""
        if (
            not isinstance(line, int)
            or not isinstance(column, int)
            or line < 1
            or line > len(self._encoded_lines)
        ):
            return None
        line_index = line - 1
        if self._ascii_lines[line_index]:
            start_character = column
        else:
            boundaries = self._byte_boundaries.get(line_index)
            if boundaries is None:
                boundaries = array("I", [0])
                byte_offset = 0
                for character in self.lines[line_index]:
                    byte_offset += len(character.encode("utf-8"))
                    boundaries.append(byte_offset)
                self._byte_boundaries[line_index] = boundaries
            start_character = bisect_left(boundaries, column)
            if start_character >= len(boundaries) or boundaries[start_character] != column:
                return None
        return self._line_starts[line - 1] + start_character

    def _source_start(self, node: ast.AST) -> int | None:
        """Return the exact character offset where an AST node starts."""
        return self._source_position(
            getattr(node, "lineno", None),
            getattr(node, "col_offset", None),
        )

    def _source_end(self, node: ast.AST) -> int | None:
        """Return the exact exclusive character offset where an AST node ends."""
        return self._source_position(
            getattr(node, "end_lineno", None),
            getattr(node, "end_col_offset", None),
        )

    def _canonical_fingerprint(
        self,
        call: ast.Call,
        shell_keyword: ast.keyword,
        shell: ast.expr,
    ) -> tuple[str, bool, int] | None:
        """Mirror the legacy direct match through ``shell=True`` for compaction."""
        call_start = self._source_start(call)
        shell_start = self._source_start(shell)
        if call_start is None or shell_start is None or shell_start < call_start:
            return None
        raw_callee = self._source_segment(call.func)
        canonical_start = call_start
        if (
            isinstance(call.func, ast.Attribute)
            and _normalized_direct_name(call.func.attr) == "popen"
            and not any(
                view.text.casefold() == "subprocess.popen"
                for view in security_text_views(raw_callee)
            )
        ):
            function_end = self._source_end(call.func)
            raw_method = re.search(r"(?P<method>\w+)\s*$", raw_callee)
            if function_end is not None and raw_method is not None:
                canonical_start = function_end - len(raw_method.group("method"))
        raw_canonical = self.content[canonical_start:shell_start] + "True"
        canonical = normalized_security_prefix(raw_canonical, _MAX_FINGERPRINT_CHARS)
        normalized = " ".join(canonical.strip().split())
        normalized_callee = normalized_security_view(raw_callee).text
        keyword_start = self._source_start(shell_keyword)
        raw_keyword = self.content[keyword_start:shell_start] if keyword_start is not None else ""
        normalized_keyword = normalized_security_view(raw_keyword).text
        normalization_exposed_direct_spelling = (
            _DIRECT_CALLEE.fullmatch(raw_callee) is None
            and _DIRECT_CALLEE.fullmatch(normalized_callee) is not None
            or _SHELL_KEYWORD_PREFIX.fullmatch(raw_keyword) is None
            and _SHELL_KEYWORD_PREFIX.fullmatch(normalized_keyword) is not None
        )
        return (
            sha256(f"TM1\x1f{normalized}".encode()).hexdigest(),
            normalization_exposed_direct_spelling,
            canonical_start,
        )

    def _bounded_context(self, call: ast.Call) -> str:
        """Retain useful local context without repeating an attacker-sized line."""
        call_start = self._source_start(call)
        if call_start is None:
            line = getattr(call, "lineno", 1)
            return get_context_from_lines(self.lines, line)[:_MAX_CONTEXT_CHARS]
        left = max(0, call_start - _MAX_CONTEXT_CHARS // 2)
        right = min(len(self.content), left + _MAX_CONTEXT_CHARS)
        left = max(0, right - _MAX_CONTEXT_CHARS)
        return self.content[left:right].rstrip(LINE_BREAK_CHARS)

    def _source_segment(self, call: ast.AST) -> str:
        """Extract one exact AST span without repeatedly rescanning the whole file."""
        line = getattr(call, "lineno", 1)
        end_line = getattr(call, "end_lineno", None)
        column = getattr(call, "col_offset", None)
        end_column = getattr(call, "end_col_offset", None)
        if (
            not isinstance(end_line, int)
            or not isinstance(column, int)
            or not isinstance(end_column, int)
            or line < 1
            or end_line < line
            or end_line > len(self._encoded_lines)
        ):
            return get_source_segment(self.lines, line, end_line)

        if line == end_line:
            encoded = self._encoded_lines[line - 1][column:end_column]
        else:
            encoded = b"\n".join(
                (
                    self._encoded_lines[line - 1][column:],
                    *self._encoded_lines[line : end_line - 1],
                    self._encoded_lines[end_line - 1][:end_column],
                )
            )
        return encoded.decode("utf-8", errors="replace")

    def _append_finding(
        self,
        call: ast.Call,
        shell_keyword: ast.keyword,
        shell: ast.expr,
    ) -> None:
        """Materialize one public bound-name finding."""
        line = getattr(call, "lineno", 1)
        end_line = getattr(call, "end_lineno", None)
        source_start = self._source_start(call)
        source_end = self._source_end(call)
        shell_anchor = self._source_start(shell_keyword)
        shell_start = self._source_start(shell)
        shell_end = self._source_end(shell)
        popen_start: int | None = None
        if (
            isinstance(call.func, ast.Attribute)
            and _normalized_direct_name(call.func.attr) == "popen"
        ):
            function_end = self._source_end(call.func)
            raw_function = self._source_segment(call.func)
            raw_method = re.search(r"(?P<method>\w+)\s*$", raw_function)
            if function_end is not None and raw_method is not None:
                popen_start = function_end - len(raw_method.group("method"))
        canonical = self._canonical_fingerprint(call, shell_keyword, shell)
        if (
            source_start is not None
            and shell_anchor is not None
            and shell_start is not None
            and shell_end is not None
            and canonical is not None
        ):
            self.bound_shell_metadata.append(
                BoundShellMetadata(
                    call_start=source_start,
                    call_end=source_end,
                    shell_anchor=shell_anchor,
                    value_start=shell_start,
                    value_end=shell_end,
                    popen_start=popen_start,
                    canonical_fingerprint=canonical[0],
                    normalized_view=canonical[1],
                )
            )
        if not self._emit_findings:
            return

        source = self._source_segment(call)
        evidence: dict[str, object] = {BOUND_SHELL_EVIDENCE: True}
        if source_start is not None:
            evidence[BOUND_CALL_START_EVIDENCE] = source_start
        if source_end is not None:
            evidence[BOUND_CALL_END_EVIDENCE] = source_end
        if shell_anchor is not None:
            evidence[BOUND_SHELL_ANCHOR_EVIDENCE] = shell_anchor
        if shell_start is not None:
            evidence[BOUND_SHELL_VALUE_START_EVIDENCE] = shell_start
        if shell_end is not None:
            evidence[BOUND_SHELL_VALUE_END_EVIDENCE] = shell_end
        if source_start is not None and shell_start is not None:
            evidence[BOUND_CLASSIFICATION_MATCH_EVIDENCE] = (
                self.content[source_start:shell_start] + "True"
            )[:200]
            evidence[BOUND_DIRECT_MATCH_END_EVIDENCE] = shell_start + len("True")
        if popen_start is not None:
            evidence[BOUND_POPEN_START_EVIDENCE] = popen_start
        canonical_fingerprint = (
            canonical[0]
            if canonical is not None
            else sha256(
                (
                    "TM1\x1f"
                    + " ".join(
                        normalized_security_prefix(source, _MAX_FINGERPRINT_CHARS).strip().split()
                    )
                ).encode()
            ).hexdigest()
        )
        finding_key = (line, canonical_fingerprint)
        if finding_key in self._finding_keys:
            return
        self._finding_keys.add(finding_key)
        if canonical is not None:
            fingerprint, is_normalized, direct_owner_start = canonical
            evidence[BOUND_CANONICAL_FINGERPRINT_EVIDENCE] = fingerprint
            evidence[BOUND_DIRECT_OWNER_START_EVIDENCE] = direct_owner_start
            if is_normalized:
                evidence[BOUND_NORMALIZED_VIEW_EVIDENCE] = True
        self.findings.append(
            AnalyzerFinding(
                rule_id="TM1",
                message="Tool Parameter Abuse",
                severity=Severity.HIGH,
                location=Location(file=self.file_path, start_line=line, end_line=end_line),
                confidence=0.8,
                tags=[PatternCategory.TOOL_MISUSE.value],
                context=self._bounded_context(call),
                matched_text=source[:200],
                evidence=evidence,
            )
        )

    def _inspect_call(self, call: ast.Call, facts: dict[str, bool]) -> None:
        shell_keyword = next((item for item in call.keywords if item.arg == "shell"), None)
        if shell_keyword is None:
            return
        shell = shell_keyword.value
        is_bound = isinstance(shell, ast.Name) and facts.get(shell.id) is True
        if not is_bound:
            return
        self._append_finding(call, shell_keyword, shell)

    def _scan_assignment(
        self,
        targets: list[ast.expr],
        value: ast.expr,
        facts: dict[str, bool],
        protocol_safe_names: set[str],
        trusted_names: set[str],
    ) -> None:
        if isinstance(value, ast.Call) and _is_direct_subprocess_call(value, trusted_names):
            resolved = None
            safe_value = _call_arguments_are_passive(value)
            protocol_safe_value = False
            if safe_value:
                self._inspect_call(value, facts)
                if not _call_arguments_are_protocol_safe(
                    value,
                    facts,
                    protocol_safe_names,
                ):
                    # The receiver and argument expressions were resolved
                    # safely, but subprocess may now dispatch ``__fspath__``,
                    # mapping, iterator, or other hooks on supplied objects.
                    facts.clear()
                    protocol_safe_names.clear()
                    trusted_names.clear()
                    return
        else:
            resolved = _truth_value(value, facts)
            safe_value = resolved is not None or _is_passive_argument(value)
            protocol_safe_value = _is_protocol_safe_argument(value, protocol_safe_names)

        if not safe_value or any(not isinstance(target, ast.Name) for target in targets):
            preserves_receiver_trust = _expression_preserves_receiver_trust(
                value,
                trusted_names,
                facts,
                protocol_safe_names,
            )
            facts.clear()
            protocol_safe_names.clear()
            if any(not isinstance(target, ast.Name) for target in targets) or not (
                preserves_receiver_trust
            ):
                trusted_names.clear()
            else:
                trusted_names.difference_update(
                    _changed_direct_names([value, *targets], trusted_names)
                )
            return
        for target in targets:
            assert isinstance(target, ast.Name)
            if resolved is None:
                facts.pop(target.id, None)
            else:
                facts[target.id] = resolved
            if protocol_safe_value:
                protocol_safe_names.add(target.id)
            else:
                protocol_safe_names.discard(target.id)
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
        initially_bound_names: set[str] | None = None,
    ) -> None:
        trusted_names = set(_DIRECT_CALL_NAMES if trusted_names is None else trusted_names)
        initial_bound_names = set(initially_bound_names or ())
        facts: dict[str, bool] = {}
        protocol_safe_names: set[str] = set()
        bound_names = set(initial_bound_names)
        finalizer_safe_names: set[str] = set()
        effect_state = _BlockEffectState()
        last_invalidation_by_name: dict[str, int] = {}
        receiver_trust = set(trusted_names)
        prepass_facts: dict[str, bool] = {}
        prepass_protocol_safe_names: set[str] = set()
        prepass_bound_names = set(initial_bound_names)
        prepass_finalizer_safe_names: set[str] = set()
        prepass_effect_state = _BlockEffectState()
        for candidate_index, candidate in enumerate(statements):
            release_hazard = _statement_releases_unsafe_binding(
                candidate,
                prepass_bound_names,
                prepass_finalizer_safe_names,
            )
            provenance_invalidated = release_hazard or _statement_requires_callable_barrier(
                candidate,
                receiver_trust,
                prepass_facts,
                prepass_protocol_safe_names,
            )
            before = set(receiver_trust)
            _advance_trusted_names(
                candidate,
                receiver_trust,
                prepass_facts,
                prepass_protocol_safe_names,
                prepass_bound_names,
                prepass_finalizer_safe_names,
                prepass_effect_state,
            )
            if release_hazard and not isinstance(candidate, (ast.Import, ast.ImportFrom)):
                receiver_trust.clear()
            _advance_value_provenance(
                candidate,
                prepass_facts,
                prepass_protocol_safe_names,
                before,
            )
            _advance_finalizer_provenance(
                candidate,
                prepass_bound_names,
                prepass_finalizer_safe_names,
                invalidated=provenance_invalidated,
            )
            if provenance_invalidated and not isinstance(candidate, (ast.Import, ast.ImportFrom)):
                prepass_effect_state.arbitrary_effects_seen = True
            for name in before.difference(receiver_trust):
                last_invalidation_by_name[name] = candidate_index

        def last_invalidation(name: str) -> int:
            return last_invalidation_by_name.get(name, -1)

        def calls_before_invalidation() -> dict[int, set[str]]:
            receiver_trust = set(trusted_names)
            trusted_by_definition: dict[int, set[str]] = {}
            active_functions: dict[str, int] = {}
            prepass_facts: dict[str, bool] = {}
            prepass_protocol_safe_names: set[str] = set()
            prepass_bound_names = set(initial_bound_names)
            prepass_finalizer_safe_names: set[str] = set()
            prepass_effect_state = _BlockEffectState()
            tracked_callable_names = {
                candidate.name
                for candidate in statements
                if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            for candidate_index, candidate in enumerate(statements):
                if self._check_runtime is not None:
                    self._check_runtime()
                release_hazard = _statement_releases_unsafe_binding(
                    candidate,
                    prepass_bound_names,
                    prepass_finalizer_safe_names,
                )
                callable_barrier = release_hazard or _statement_requires_callable_barrier(
                    candidate,
                    receiver_trust,
                    prepass_facts,
                    prepass_protocol_safe_names,
                )

                call = _passive_direct_call(candidate)
                if call is not None:
                    assert isinstance(call.func, ast.Name)
                    owner = active_functions.get(call.func.id)
                    if owner is not None:
                        trusted_by_definition.setdefault(owner, set()).update(receiver_trust)

                alias_owner: int | None = None
                alias_names: list[str] = []
                if (
                    isinstance(candidate, ast.Assign)
                    and isinstance(candidate.value, ast.Name)
                    and all(isinstance(target, ast.Name) for target in candidate.targets)
                ):
                    alias_owner = active_functions.get(candidate.value.id)
                    if alias_owner is not None:
                        alias_names = [
                            target.id
                            for target in candidate.targets
                            if isinstance(target, ast.Name)
                        ]

                if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (
                    _function_header_is_passive(candidate)
                ):
                    active_functions.clear()
                changed_callable_names = _changed_direct_names(
                    [candidate],
                    tracked_callable_names,
                )
                if isinstance(candidate, ast.ClassDef):
                    changed_callable_names.update(
                        _class_body_changed_direct_names(candidate, tracked_callable_names)
                    )
                for name in changed_callable_names:
                    active_functions.pop(name, None)

                if alias_owner is not None:
                    for name in alias_names:
                        tracked_callable_names.add(name)
                        active_functions[name] = alias_owner

                if (
                    isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _function_header_is_passive(candidate)
                    and _is_immediate_function(candidate)
                ):
                    active_functions[candidate.name] = candidate_index

                if callable_barrier:
                    active_functions.clear()
                trusted_names_before = set(receiver_trust)
                _advance_trusted_names(
                    candidate,
                    receiver_trust,
                    prepass_facts,
                    prepass_protocol_safe_names,
                    prepass_bound_names,
                    prepass_finalizer_safe_names,
                    prepass_effect_state,
                )
                if release_hazard and not isinstance(candidate, (ast.Import, ast.ImportFrom)):
                    receiver_trust.clear()
                _advance_value_provenance(
                    candidate,
                    prepass_facts,
                    prepass_protocol_safe_names,
                    trusted_names_before,
                )
                _advance_finalizer_provenance(
                    candidate,
                    prepass_bound_names,
                    prepass_finalizer_safe_names,
                    invalidated=callable_barrier,
                )
                if callable_barrier and not isinstance(candidate, (ast.Import, ast.ImportFrom)):
                    prepass_effect_state.arbitrary_effects_seen = True

            return trusted_by_definition

        trusted_at_call_by_definition = calls_before_invalidation()

        for index, statement in enumerate(statements):
            if self._check_runtime is not None:
                self._check_runtime()
            release_hazard = _statement_releases_unsafe_binding(
                statement,
                bound_names,
                finalizer_safe_names,
            )
            provenance_invalidated = release_hazard or _statement_requires_callable_barrier(
                statement,
                trusted_names,
                facts,
                protocol_safe_names,
            )
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                passive_header = _function_header_is_passive(statement)
                trusted_at_call = trusted_at_call_by_definition.get(index, set())
                nested_trusted_names = set(trusted_names).union(trusted_at_call)
                nested_trusted_names = {
                    name
                    for name in nested_trusted_names
                    if last_invalidation(name) <= index or name in trusted_at_call
                }
                nested_trusted_names.difference_update(
                    _function_bound_direct_names(statement, nested_trusted_names)
                )
                nested_trusted_names.discard(statement.name)
                if not passive_header:
                    nested_trusted_names.clear()
                if release_hazard:
                    # The function object is stored only after its header is
                    # evaluated.  Releasing an opaque prior binding can then
                    # mutate every global used by a later call to this body.
                    nested_trusted_names.clear()
                self._scan_block(
                    statement.body,
                    trusted_names=nested_trusted_names,
                    initially_bound_names=_function_initial_bound_names(statement),
                )
                if passive_header:
                    facts.pop(statement.name, None)
                    protocol_safe_names.discard(statement.name)
                else:
                    facts.clear()
                    protocol_safe_names.clear()
                    trusted_names.clear()
                trusted_names.discard(statement.name)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                # Import hooks execute arbitrary Python before bindings are
                # committed, so no truth fact survives an import boundary.
                facts.clear()
                protocol_safe_names.clear()
                _update_trusted_names_from_import(
                    statement,
                    trusted_names,
                    bound_names,
                    finalizer_safe_names,
                    effect_state,
                )
            elif isinstance(statement, ast.Assign):
                self._scan_assignment(
                    list(statement.targets),
                    statement.value,
                    facts,
                    protocol_safe_names,
                    trusted_names,
                )
            elif isinstance(statement, ast.AnnAssign):
                # Module annotations write through ``__annotations__`` and may
                # invoke a user-controlled mapping. Keep annotated bindings
                # outside this deliberately side-effect-free subset.
                value = statement.value
                if (
                    isinstance(value, ast.Call)
                    and _is_direct_subprocess_call(value, trusted_names)
                    and _call_arguments_are_passive(value)
                ):
                    self._inspect_call(value, facts)
                facts.clear()
                protocol_safe_names.clear()
                trusted_names.clear()
            elif isinstance(statement, ast.AugAssign):
                facts.clear()
                protocol_safe_names.clear()
                trusted_names.clear()
            elif isinstance(statement, ast.Delete):
                facts.clear()
                protocol_safe_names.clear()
                if any(not isinstance(target, ast.Name) for target in statement.targets):
                    trusted_names.clear()
                else:
                    trusted_names.difference_update(
                        _changed_direct_names(list(statement.targets), trusted_names)
                    )
            elif isinstance(statement, ast.Expr):
                value = statement.value
                preserves_receiver_trust = _expression_preserves_receiver_trust(
                    value,
                    trusted_names,
                    facts,
                    protocol_safe_names,
                )
                if (
                    isinstance(value, ast.Call)
                    and _is_direct_subprocess_call(value, trusted_names)
                    and _call_arguments_are_passive(value)
                ):
                    self._inspect_call(value, facts)
                    if not _call_arguments_are_protocol_safe(
                        value,
                        facts,
                        protocol_safe_names,
                    ):
                        facts.clear()
                        protocol_safe_names.clear()
                elif isinstance(value, ast.Constant):
                    continue
                else:
                    facts.clear()
                    protocol_safe_names.clear()
                if not preserves_receiver_trust:
                    trusted_names.clear()
            elif isinstance(statement, (ast.Global, ast.Nonlocal, ast.Pass)):
                continue
            elif isinstance(statement, ast.ClassDef):
                facts.clear()
                protocol_safe_names.clear()
                if not _class_definition_is_passive(statement):
                    trusted_names.clear()
                else:
                    trusted_names.difference_update(
                        _changed_direct_names([statement], trusted_names)
                    )
                    trusted_names.difference_update(
                        _class_body_changed_direct_names(statement, trusted_names)
                    )
            else:
                facts.clear()
                protocol_safe_names.clear()
                trusted_names.clear()

            if release_hazard and not isinstance(statement, (ast.Import, ast.ImportFrom)):
                facts.clear()
                protocol_safe_names.clear()
                trusted_names.clear()
            _advance_finalizer_provenance(
                statement,
                bound_names,
                finalizer_safe_names,
                invalidated=provenance_invalidated,
            )
            if provenance_invalidated and not isinstance(statement, (ast.Import, ast.ImportFrom)):
                effect_state.arbitrary_effects_seen = True

    def run(self, tree: ast.Module) -> list[AnalyzerFinding]:
        self._scan_block(tree.body)
        return sorted(self.findings, key=lambda finding: finding.location.start_line)


def bound_shell_metadata(
    parsed: ParsedPythonFile,
    file_path: str,
    *,
    check_runtime: Callable[[], None] | None = None,
) -> tuple[BoundShellMetadata, ...]:
    """Return every supported bound call independently of the finding cap."""
    if parsed.tree is None:
        return ()
    analyzer = _Analyzer(
        file_path,
        parsed.content,
        emit_findings=False,
        check_runtime=check_runtime,
    )
    analyzer.run(parsed.tree)
    return tuple(sorted(analyzer.bound_shell_metadata, key=lambda item: item.call_start))


def direct_literal_metadata(
    parsed: ParsedPythonFile,
    file_path: str,
    call_starts: set[int],
) -> dict[int, DirectLiteralMetadata]:
    """Enrich only retained lexical owners without creating budgeted findings."""
    if parsed.tree is None or not call_starts:
        return {}
    locator = _Analyzer(file_path, parsed.content)
    metadata: dict[int, DirectLiteralMetadata] = {}
    for node in ast.walk(parsed.tree):
        if not isinstance(node, ast.Call):
            continue
        call_start = locator._source_start(node)
        if call_start is None:
            continue
        lookup_coordinates = {call_start}
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and _normalized_direct_name(function.value.id) == "subprocess"
            and _normalized_direct_name(function.attr) == "popen"
        ):
            function_end = locator._source_end(function)
            raw_function = locator._source_segment(function)
            raw_method = re.search(r"(?P<method>\w+)\s*$", raw_function)
            if function_end is not None and raw_method is not None:
                lookup_coordinates.add(function_end - len(raw_method.group("method")))
        retained_coordinates = lookup_coordinates.intersection(call_starts)
        if not retained_coordinates:
            continue
        shell_keyword = next((item for item in node.keywords if item.arg == "shell"), None)
        if shell_keyword is None:
            continue
        shell = shell_keyword.value
        if not (
            isinstance(shell, ast.Constant) and type(shell.value) is bool and shell.value is True
        ):
            continue
        canonical = locator._canonical_fingerprint(node, shell_keyword, shell)
        shell_anchor = locator._source_start(shell_keyword)
        if canonical is None or shell_anchor is None:
            continue
        fingerprint, normalized_view, _ = canonical
        direct_metadata = DirectLiteralMetadata(
            call_start=call_start,
            call_end=locator._source_end(node),
            shell_anchor=shell_anchor,
            match_fingerprint=fingerprint,
            normalized_view=normalized_view,
        )
        for coordinate in retained_coordinates:
            metadata[coordinate] = direct_metadata
    return metadata


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
    return _Analyzer(file_path, parsed.content).run(parsed.tree)
