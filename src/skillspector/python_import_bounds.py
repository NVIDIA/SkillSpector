# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, in-memory import context for the optional Python string-value proof.

This does not execute imports or infer arbitrary data flow. A caller cannot
discard existing shell-operand uncertainty when a selected bundled helper can
execute its input. Unmatched external imports retain the scanner's existing
interprocedural limits; missing or ambiguous bundled evidence grants no proof.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath

from skillspector.python_tokens import (
    _PYTHON_OWNERSHIP_PARSE_LOCK,
    python_literal_spans,
    python_string_value_bounds_allowed,
)

_MAX_IMPORT_FILES = 64
_MAX_IMPORT_DEPTH = 16
_MAX_IMPORT_CHARS = 2_000_000
_MAX_IMPORT_NODES = 40_000
_INITIALIZATION = "<module-initialization>"


class PythonImportBounds:
    """A per-scan summary using only the scanner's already materialized files."""

    def __init__(self, components: Sequence[str], file_cache: Mapping[str, str]) -> None:
        self._paths = tuple(
            dict.fromkeys(path for path in components if path.lower().endswith((".py", ".pyw")))
        )
        self._contents = file_cache
        self._cache: dict[str, tuple[str, bool]] = {}

    def allows(self, path: str, content: str, check_runtime: Callable[[], None]) -> bool:
        """Return a completed import proof; interrupted work is never cached."""
        check_runtime()
        cached = self._cache.get(path)
        if cached is not None and cached[0] == content:
            return cached[1]
        active: set[tuple[str, str | None]] = set()
        complete: dict[tuple[str, str | None], bool] = {}
        files: set[str] = set()
        characters = 0
        nodes_seen = 0

        def resolve(module: str, caller: str, level: int) -> list[str] | None:
            check_runtime()
            suffix = module.replace(".", "/")
            if level:
                parent = PurePosixPath(caller.replace("\\", "/")).parent
                for _ in range(level - 1):
                    parent = parent.parent
                suffix = str(parent / suffix)
            matches = []
            for candidate in self._paths:
                check_runtime()
                normalized = candidate.replace("\\", "/")
                targets = (suffix + ".py", suffix + ".pyw", suffix + "/__init__.py")
                if any(
                    normalized == target or normalized.endswith("/" + target) for target in targets
                ):
                    matches.append(candidate)
            if matches:
                return matches
            # An absent relative module or a visible package with no resolvable
            # entry is missing bundled evidence, rather than an external import.
            if level or any(
                ("/" + suffix + "/") in ("/" + p.replace("\\", "/")) for p in self._paths
            ):
                return None
            return []

        def inspect(module_path: str, symbol: str | None, depth: int) -> bool:
            nonlocal characters, nodes_seen
            check_runtime()
            key = (module_path, symbol)
            if key in complete:
                return complete[key]
            if key in active or depth > _MAX_IMPORT_DEPTH:
                return False
            source = content if module_path == path else self._contents.get(module_path)
            if source is None:
                return False
            if module_path not in files:
                files.add(module_path)
                characters += len(source)
            if len(files) > _MAX_IMPORT_FILES or characters > _MAX_IMPORT_CHARS:
                return False
            if python_literal_spans(source, check_runtime) is None:
                return False
            # The ownership helper has already bounded and validated this parse.
            with _PYTHON_OWNERSHIP_PARSE_LOCK, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(source)
            check_runtime()
            active.add(key)
            # Importing pkg.helper also initializes each bundled ancestor
            # package. A namespace package with no initializer adds no code.
            for parent in PurePosixPath(module_path.replace("\\", "/")).parents:
                check_runtime()
                initializer = str(parent / "__init__.py")
                matches = [p for p in self._paths if p.replace("\\", "/") == initializer]
                if len(matches) > 1:
                    return False
                if matches and matches[0] != module_path:
                    if not inspect(matches[0], _INITIALIZATION, depth + 1):
                        return False
            if depth and symbol != _INITIALIZATION:
                if not inspect(module_path, _INITIALIZATION, depth):
                    return False
            declarations: dict[str, list[ast.stmt]] = {}
            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    declarations.setdefault(statement.name, []).append(statement)
                elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            declarations.setdefault(target.id, []).append(statement)

            def transparent_decorators(declaration: ast.stmt) -> bool:
                decorators = getattr(declaration, "decorator_list", [])
                if not decorators:
                    return True
                # The standard cache decorators preserve the wrapped callable's
                # body. Other decorators can replace it with an execution sink.
                if resolve("functools", module_path, 0) != []:
                    return False
                cache_names: set[str] = set()
                module_names: set[str] = set()
                for statement in tree.body:
                    if isinstance(statement, ast.ImportFrom) and statement.module == "functools":
                        cache_names.update(
                            alias.asname or alias.name
                            for alias in statement.names
                            if alias.name in {"cache", "lru_cache"}
                        )
                    elif isinstance(statement, ast.Import):
                        module_names.update(
                            alias.asname or alias.name
                            for alias in statement.names
                            if alias.name == "functools"
                        )
                for node in ast.walk(tree):
                    check_runtime()
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            bound = alias.asname or alias.name.split(".")[0]
                            if not (
                                isinstance(node, ast.ImportFrom)
                                and node.module == "functools"
                                and not node.level
                                and alias.name in {"cache", "lru_cache"}
                            ):
                                cache_names.discard(bound)
                            if not (isinstance(node, ast.Import) and alias.name == "functools"):
                                module_names.discard(bound)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        cache_names.discard(node.id)
                        module_names.discard(node.id)
                    elif isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.arg)
                    ):
                        name = node.arg if isinstance(node, ast.arg) else node.name
                        cache_names.discard(name)
                        module_names.discard(name)
                for decorator in decorators:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if not (
                        isinstance(target, ast.Name)
                        and target.id in cache_names
                        or isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in module_names
                        and target.attr in {"cache", "lru_cache"}
                    ):
                        return False
                    if isinstance(decorator, ast.Call) and (
                        any(not isinstance(arg, ast.Constant) for arg in decorator.args)
                        or any(
                            keyword.arg not in {"maxsize", "typed"}
                            or not isinstance(keyword.value, ast.Constant)
                            for keyword in decorator.keywords
                        )
                    ):
                        return False
                return True

            selected = list(tree.body)
            if symbol == _INITIALIZATION:
                selected = []
                for statement in tree.body:
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Defining a function evaluates defaults, annotations and
                        # decorators, but does not execute its body.
                        stub = ast.FunctionDef(
                            name=statement.name,
                            args=statement.args,
                            body=[ast.Pass()],
                            decorator_list=statement.decorator_list,
                            returns=statement.returns,
                            type_comment=statement.type_comment,
                            type_params=getattr(statement, "type_params", []),
                        )
                        selected.append(ast.fix_missing_locations(stub))
                    else:
                        selected.append(statement)
                # A top-level call or escaped callback can initialize an
                # executor through global mutation. The stub proof does not
                # model executing local definitions, so retain fallback.
                for statement in selected:
                    for node in ast.walk(statement):
                        check_runtime()
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id in declarations
                        ):
                            return False
                        if (
                            isinstance(node, ast.Name)
                            and isinstance(node.ctx, ast.Load)
                            and any(
                                isinstance(
                                    declaration,
                                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                                )
                                for declaration in declarations.get(node.id, [])
                            )
                        ):
                            return False
            elif symbol is not None:
                if len(declarations.get(symbol, [])) != 1:
                    return False
                selected = [
                    statement
                    for statement in tree.body
                    if not isinstance(
                        statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                ]
                pending = [symbol]
                included: set[str] = set()
                while pending:
                    check_runtime()
                    name = pending.pop()
                    if name in included:
                        continue
                    if len(declarations.get(name, [])) != 1:
                        return False
                    declaration = declarations[name][0]
                    if isinstance(declaration, ast.ClassDef) or not transparent_decorators(
                        declaration
                    ):
                        return False
                    if any(isinstance(node, ast.Lambda) for node in ast.walk(declaration)):
                        return False
                    included.add(name)
                    if declaration not in selected:
                        selected.append(declaration)
                    for node in ast.walk(declaration):
                        check_runtime()
                        nodes_seen += 1
                        if nodes_seen > _MAX_IMPORT_NODES:
                            return False
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                            if node.id in declarations:
                                pending.append(node.id)
                # Importing an unrelated function must not inherit every other
                # function in its module. Keep only imports used by the selected
                # scope, while preserving all non-import module initialization.
                used = {
                    node.id
                    for statement in selected
                    if not isinstance(statement, (ast.Import, ast.ImportFrom))
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
                trimmed: list[ast.stmt] = []
                for statement in selected:
                    if isinstance(statement, (ast.Import, ast.ImportFrom)):
                        aliases = [
                            alias
                            for alias in statement.names
                            if alias.name == "*"
                            or (alias.asname or alias.name.split(".")[0]) in used
                        ]
                        if aliases:
                            if isinstance(statement, ast.Import):
                                trimmed.append(ast.Import(names=aliases))
                            else:
                                trimmed.append(
                                    ast.ImportFrom(
                                        module=statement.module,
                                        names=aliases,
                                        level=statement.level,
                                    )
                                )
                    else:
                        trimmed.append(statement)
                selected = trimmed
            scope = ast.Module(body=selected, type_ignores=[])
            scope_nodes = []
            for node in ast.walk(scope):
                check_runtime()
                nodes_seen += 1
                if nodes_seen > _MAX_IMPORT_NODES:
                    return False
                scope_nodes.append(node)
            if depth and not python_string_value_bounds_allowed(
                ast.unparse(scope), check_runtime, allow_dynamic_subprocess=False
            ):
                return False
            for node in scope_nodes:
                check_runtime()
                if isinstance(node, ast.Import):
                    requests = [(alias.name, None, 0) for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    requests = [(node.module or "", alias.name, node.level) for alias in node.names]
                else:
                    continue
                for module, imported_symbol, level in requests:
                    matches = resolve(module, module_path, level)
                    if matches is None or len(matches) > 1:
                        return False
                    if matches and (
                        imported_symbol == "*"
                        or not inspect(
                            matches[0],
                            _INITIALIZATION if symbol == _INITIALIZATION else imported_symbol,
                            depth + 1,
                        )
                    ):
                        return False
            active.remove(key)
            complete[key] = True
            return True

        result = inspect(path, None, 0)
        self._cache[path] = (content, result)
        return result
