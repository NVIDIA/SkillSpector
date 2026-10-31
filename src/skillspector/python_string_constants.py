# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded literal values for simple Python f-string replacement fields.

This evaluates a small AST vocabulary on scanner-owned native values. It never
executes uploaded code. Ordinary standard-library string/URL behavior is assumed;
explicit binding changes, escaping helper references and unknown syntax refuse
the proof. The caller still owns string boundaries and shell-sink checks.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Callable
from urllib.parse import urlparse

from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS, parse_python_source
from skillspector.python_tokens import _PYTHON_OWNERSHIP_PARSE_LOCK

_MAX_WORK = 100_000
_MAX_DEPTH = 16
_MAX_VALUE_CHARS = 4096
_MAX_ITEMS = 32
_MAX_FIELDS = 1024
_MAX_RESULT_CHARS = 32_768
_PLAIN_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_REFLECTION = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "__builtins__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
    }
)


class _UnprovenError(Exception):
    pass


class _ProofLimitError(Exception):
    pass


class _Constants:
    def __init__(
        self,
        content: str,
        tree: ast.Module,
        check_runtime: Callable[[], None],
        stdlib_check: Callable[[str, Callable[[], None]], bool] | None,
    ) -> None:
        self.content = content
        self.tree = tree
        self.check_runtime = check_runtime
        self.stdlib_check = stdlib_check
        self.urlparse_allowed: bool | None = None
        self.work = 0
        self.parents: dict[ast.AST, ast.AST] = {}
        self.scopes: dict[ast.AST, ast.AST] = {}
        self.bindings: dict[tuple[ast.AST, str], list[ast.AST]] = {}
        self.scope_names: dict[ast.AST, set[str]] = {}
        self.references: dict[str, list[ast.Name]] = {}
        self.fields: list[ast.FormattedValue] = []
        self.active: set[ast.AST] = set()
        self.values: dict[ast.AST, object] = {}
        self.checked_callables: set[ast.AST] = set()
        self.line_starts = [0]
        self.offsets: dict[tuple[int, int], int] = {}
        for index, character in enumerate(content):
            if index % 4096 == 0:
                check_runtime()
            if character == "\n":
                self.line_starts.append(index + 1)
        pending: list[tuple[ast.AST, ast.AST]] = [(tree, tree)]
        while pending:
            node, scope = pending.pop()
            self.spend()
            self.scopes[node] = scope
            names: list[str] = []
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.append(node.id)
                else:
                    self.references.setdefault(node.id, []).append(node)
                    if node.id in _REFLECTION:
                        raise _UnprovenError
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.append(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.NamedExpr, ast.Match)):
                raise _UnprovenError
            if isinstance(node, ast.Attribute) and (
                node.attr
                in _REFLECTION
                | {
                    "__dict__",
                    "__code__",
                    "__defaults__",
                    "__kwdefaults__",
                    "__globals__",
                    "__closure__",
                    "__getattribute__",
                    "__setattr__",
                    "import_module",
                    "modules",
                    "meta_path",
                    "path_hooks",
                    "path_importer_cache",
                }
                or isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                raise _UnprovenError
            if "*" in names:
                raise _UnprovenError
            for name in names:
                self.bindings.setdefault((scope, name), []).append(node)
                self.scope_names.setdefault(scope, set()).add(name)
            if isinstance(node, ast.FormattedValue):
                self.fields.append(node)
            child_scope = node if isinstance(node, _SCOPES) else scope
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
                pending.append((child, child_scope))

    def spend(self, amount: int = 1) -> None:
        previous = self.work
        self.work += amount
        if self.work > _MAX_WORK:
            raise _ProofLimitError
        if previous // 256 != self.work // 256:
            self.check_runtime()

    def offset(self, line: int, column: int) -> int:
        key = (line, column)
        if key not in self.offsets:
            start = self.line_starts[line - 1]
            end = self.line_starts[line] if line < len(self.line_starts) else len(self.content)
            self.spend(end - start)
            prefix = self.content[start:end].encode("utf-8")[:column].decode("utf-8")
            self.offsets[key] = start + len(prefix)
        return self.offsets[key]

    def start(self, node: ast.AST) -> int:
        return self.offset(node.lineno, node.col_offset)

    def end(self, node: ast.AST) -> int:
        return self.offset(node.end_lineno, node.end_col_offset)

    def module_binding(self, node: ast.Name) -> ast.AST:
        scope = self.scopes[node]
        first = True
        while scope is not self.tree:
            self.spend()
            if (first or not isinstance(scope, ast.ClassDef)) and self.bindings.get(
                (scope, node.id)
            ):
                raise _UnprovenError
            first = False
            scope = self.scopes[scope]
        candidates = self.bindings.get((self.tree, node.id), [])
        if len(candidates) != 1:
            raise _UnprovenError
        return candidates[0]

    def callable_binding(self, node: ast.Name, use: int) -> ast.AST:
        owner = self.module_binding(node)
        if self.parents.get(owner) is not self.tree or self.end(owner) > use:
            raise _UnprovenError
        if owner not in self.checked_callables:
            for reference in self.references.get(node.id, []):
                self.spend()
                # Defaults and annotations execute outside a function's local
                # scope. Ambiguous shadows must not certify a callable escape.
                parent = self.parents.get(reference)
                if not isinstance(parent, ast.Call) or parent.func is not reference:
                    raise _UnprovenError
            self.checked_callables.add(owner)
        return owner

    def known(
        self, node: ast.AST, local: dict[str, object], local_names: set[str], use: int, depth: int
    ) -> object:
        self.spend()
        if depth > _MAX_DEPTH:
            raise _ProofLimitError
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, bool):
            if isinstance(node.value, str) and len(node.value) > _MAX_VALUE_CHARS:
                raise _UnprovenError
            if type(node.value) is int and abs(node.value) > _MAX_VALUE_CHARS:
                raise _UnprovenError
            return node.value
        if isinstance(node, ast.Name):
            if node.id in local_names:
                if node.id not in local:
                    raise _UnprovenError
                return local[node.id]
            owner = self.module_binding(node)
            assignment = self.parents.get(owner)
            if not (
                isinstance(assignment, (ast.Assign, ast.AnnAssign))
                and self.parents.get(assignment) is self.tree
                and (
                    isinstance(assignment, ast.Assign)
                    and assignment.targets == [owner]
                    or isinstance(assignment, ast.AnnAssign)
                    and assignment.target is owner
                )
                and self.end(assignment) <= use
            ):
                raise _UnprovenError
            if assignment in self.active:
                raise _UnprovenError
            if assignment not in self.values:
                self.active.add(assignment)
                try:
                    value = self.known(
                        assignment.value, {}, set(), self.start(assignment), depth + 1
                    )
                    # Mutable module containers can change through aliases; only
                    # the immutable scalar binding itself is used as evidence.
                    if type(value) not in (str, int, bool):
                        raise _UnprovenError
                    self.values[assignment] = value
                finally:
                    self.active.remove(assignment)
            return self.values[assignment]
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= _MAX_ITEMS:
            values = [self.known(item, local, local_names, use, depth + 1) for item in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.known(node.operand, local, local_names, use, depth + 1)
            if type(value) is int:
                return -value if isinstance(node.op, ast.USub) else value
            raise _UnprovenError
        if isinstance(node, ast.Subscript):
            value = self.known(node.value, local, local_names, use, depth + 1)
            if type(value) not in (str, list, tuple):
                raise _UnprovenError
            if isinstance(node.slice, ast.Slice):
                parts = [
                    self.known(item, local, local_names, use, depth + 1)
                    if item is not None
                    else None
                    for item in (node.slice.lower, node.slice.upper, node.slice.step)
                ]
                if any(item is not None and type(item) is not int for item in parts):
                    raise _UnprovenError
                index = slice(*parts)
            else:
                index = self.known(node.slice, local, local_names, use, depth + 1)
                if type(index) is not int:
                    raise _UnprovenError
            try:
                return value[index]
            except (IndexError, ValueError):
                raise _UnprovenError from None
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "path"
            and isinstance(node.value, ast.Call)
        ):
            call = node.value
            if not isinstance(call.func, ast.Name) or call.keywords or len(call.args) != 1:
                raise _UnprovenError
            owner = self.callable_binding(call.func, use)
            if not (
                isinstance(owner, ast.ImportFrom)
                and owner.module == "urllib.parse"
                and owner.level == 0
                and any(alias.name == "urlparse" and alias.asname is None for alias in owner.names)
                and call.func.id == "urlparse"
            ):
                raise _UnprovenError
            if self.urlparse_allowed is None:
                self.urlparse_allowed = (
                    self.stdlib_check is None
                    or self.stdlib_check("urllib", self.check_runtime) is True
                )
            if not self.urlparse_allowed:
                raise _UnprovenError
            value = self.known(call.args[0], local, local_names, use, depth + 1)
            if type(value) is not str:
                raise _UnprovenError
            try:
                return urlparse(value).path
            except ValueError:
                raise _UnprovenError from None
        if isinstance(node, ast.Call) and not node.keywords:
            if len(node.args) > _MAX_ITEMS:
                raise _UnprovenError
            arguments = [self.known(arg, local, local_names, use, depth + 1) for arg in node.args]
            if isinstance(node.func, ast.Attribute):
                value = self.known(node.func.value, local, local_names, use, depth + 1)
                if type(value) is not str:
                    raise _UnprovenError
                if (
                    node.func.attr in {"rstrip", "endswith"}
                    and len(arguments) == 1
                    and type(arguments[0]) is str
                ):
                    if node.func.attr == "rstrip":
                        return value.rstrip(arguments[0])
                    return value.endswith(arguments[0])
                if (
                    node.func.attr == "rsplit"
                    and len(arguments) == 2
                    and type(arguments[0]) is str
                    and type(arguments[1]) is int
                    and 0 <= arguments[1] < _MAX_ITEMS
                ):
                    try:
                        return value.rsplit(*arguments)
                    except ValueError:
                        raise _UnprovenError from None
                raise _UnprovenError
            if isinstance(node.func, ast.Name) and node.func.id not in local_names:
                function = self.callable_binding(node.func, use)
                if not isinstance(function, ast.FunctionDef) or function in self.active:
                    raise _UnprovenError
                args = function.args
                parameters = args.posonlyargs + args.args
                if (
                    function.decorator_list
                    or getattr(function, "type_params", [])
                    or args.defaults
                    or args.kwonlyargs
                    or args.vararg
                    or args.kwarg
                    or len(parameters) != len(arguments)
                ):
                    raise _UnprovenError
                bound = {arg.arg: value for arg, value in zip(parameters, arguments, strict=True)}
                names = self.scope_names.get(function, set())
                self.active.add(function)
                try:
                    returned, value = self.body(function.body, bound, names, use, depth + 1)
                finally:
                    self.active.remove(function)
                if returned:
                    return value
        raise _UnprovenError

    def body(
        self,
        statements: list[ast.stmt],
        local: dict[str, object],
        names: set[str],
        use: int,
        depth: int,
    ) -> tuple[bool, object]:
        if depth > _MAX_DEPTH:
            raise _ProofLimitError
        for statement in statements:
            self.spend()
            if isinstance(statement, ast.Return):
                return True, self.known(statement.value, local, names, use, depth + 1)
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                local[statement.targets[0].id] = self.known(
                    statement.value, local, names, use, depth + 1
                )
            elif isinstance(statement, ast.If):
                condition = self.known(statement.test, local, names, use, depth + 1)
                if type(condition) is not bool:
                    raise _UnprovenError
                returned, value = self.body(
                    statement.body if condition else statement.orelse, local, names, use, depth + 1
                )
                if returned:
                    return True, value
            elif (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and type(statement.value.value) is str
            ):
                continue  # A literal helper docstring has no execution effects.
            else:
                raise _UnprovenError
        return False, None

    def result(self) -> tuple[tuple[int, int, str], ...]:
        results = []
        result_chars = 0
        for field in self.fields:
            self.spend()
            if (
                not isinstance(field.value, ast.Name)
                or field.conversion != -1
                or field.format_spec is not None
            ):
                continue
            ancestor = self.parents.get(field)
            while ancestor is not None:
                self.spend()
                if isinstance(ancestor, (ast.FormattedValue, ast.ClassDef)):
                    # Nested fields belong to another expression; class bodies,
                    # defaults and comprehensions have distinct lookup rules.
                    break
                ancestor = self.parents.get(ancestor)
            else:
                ancestor = None
            if isinstance(ancestor, (ast.FormattedValue, ast.ClassDef)):
                continue
            start = self.start(field.value) - 1
            end = self.end(field.value) + 1
            if self.content[start:end] != "{" + field.value.id + "}":
                continue
            try:
                value = self.known(field.value, {}, set(), start, 0)
            except _UnprovenError:
                continue
            if (
                type(value) is str
                and _PLAIN_COMPONENT.fullmatch(value)
                and not value.startswith("-")
                and value not in {".", ".."}
            ):
                result_chars += len(value)
                if len(results) >= _MAX_FIELDS or result_chars > _MAX_RESULT_CHARS:
                    raise _ProofLimitError
                results.append((start, end, value))
        self.check_runtime()
        return tuple(sorted(results))


def python_constant_string_fields(
    content: str,
    check_runtime: Callable[[], None],
    *,
    stdlib_check: Callable[[str, Callable[[], None]], bool] | None = None,
) -> tuple[tuple[int, int, str], ...]:
    """Return exact character spans and known plain values for normal ``{NAME}`` fields.

    A bundle caller supplies ``stdlib_check`` to reject uploaded implementations
    of trusted imports. Without bundle context, ordinary stdlib identity applies.
    """
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS or re.search(r"\r(?!\n)", content):
        return ()
    check_runtime()
    with _PYTHON_OWNERSHIP_PARSE_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = parse_python_source(content, "<constant string fields>")
    check_runtime()
    if parsed.tree is None:
        return ()
    try:
        return _Constants(content, parsed.tree, check_runtime, stdlib_check).result()
    except (_UnprovenError, _ProofLimitError):
        return ()
