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

"""Bounded, conservative source proof for literal DB-API parameter binding.

This is an optional precision proof, not a Python evaluator. Ambiguous bindings,
opaque calls, source transformations and exhausted bounds retain lexical findings.
"""

from __future__ import annotations

import ast
from collections import defaultdict

from skillspector.python_ast import ParsedPythonFile

_MAX_PROOF_NODES = 20_000
_MAX_ORIGIN_DEPTH = 8
_DRIVERS = frozenset({"sqlite3", "psycopg", "psycopg2"})
_SQLITE_CONNECT_KEYWORDS = frozenset(
    {
        "database",
        "timeout",
        "detect_types",
        "isolation_level",
        "check_same_thread",
        "cached_statements",
        "uri",
        "autocommit",
    }
)
_Origin = tuple[str, str]


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _BindingProof:
    def __init__(self, parsed: ParsedPythonFile) -> None:
        self.parsed = parsed
        self.nodes: list[ast.AST] = []
        self.bindings: dict[str, list[tuple[ast.AST, ast.AST | _Origin | None]]] = defaultdict(list)
        self.invalid: set[str] = set()
        self.complete = False
        self.passive_calls: set[ast.Call] = set()
        self.valid_connections: dict[ast.Call, bool] = {}
        values: dict[ast.AST, ast.AST] = {}
        assert parsed.tree is not None
        stack: list[ast.AST] = [parsed.tree]
        while stack:
            node = stack.pop()
            self.nodes.append(node)
            if len(self.nodes) > _MAX_PROOF_NODES:
                return
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    values[targets[0]] = node.value
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                values[node.optional_vars] = node.context_expr
            stack.extend(ast.iter_child_nodes(node))
        for node in self.nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.bindings[node.id].append((node, values.get(node)))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    origin = (alias.name, "module") if alias.name in _DRIVERS else None
                    self.bindings[name].append((node, origin))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    origin = (
                        (node.module, "connector")
                        if node.level == 0 and node.module in _DRIVERS and alias.name == "connect"
                        else None
                    )
                    self.bindings[alias.asname or alias.name].append((node, origin))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bindings[node.name].append((node, None))
            elif isinstance(node, ast.arg):
                self.bindings[node.arg].append((node, None))
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                self.bindings[node.name].append((node, None))
            elif isinstance(node, ast.MatchMapping) and node.rest:
                self.bindings[node.rest].append((node, None))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self.bindings[node.name].append((node, None))
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                self.invalid.update(node.names)
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                root_binding_name = _root_name(node)
                if root_binding_name is not None:
                    self.invalid.add(root_binding_name)
        # A write through a simple alias also invalidates the original object.
        aliases: dict[str, set[str]] = defaultdict(set)
        for name, entries in self.bindings.items():
            for _owner, value in entries:
                if isinstance(value, ast.Name):
                    aliases[name].add(value.id)
                    aliases[value.id].add(name)
        pending = list(self.invalid)
        while pending:
            name = pending.pop()
            for alias_name in aliases.get(name, ()):
                if alias_name not in self.invalid:
                    self.invalid.add(alias_name)
                    pending.append(alias_name)
        self.complete = True

    @staticmethod
    def _position(node: ast.AST) -> tuple[int, int]:
        return getattr(node, "lineno", 0), getattr(node, "col_offset", 0)

    def origin(self, node: ast.AST, depth: int = 0) -> _Origin | None:
        if depth >= _MAX_ORIGIN_DEPTH:
            return None
        if isinstance(node, ast.Name):
            entries = self.bindings.get(node.id, ())
            if node.id in self.invalid or len(entries) != 1:
                return None
            binding, value = entries[0]
            if self._position(binding) >= self._position(node):
                return None
            if isinstance(value, tuple):
                return value
            return self.origin(value, depth + 1) if isinstance(value, ast.AST) else None
        if isinstance(node, ast.Attribute) and node.attr == "connect":
            owner = self.origin(node.value, depth + 1)
            if owner is not None and owner[1] == "module":
                return owner[0], "connector"
        if isinstance(node, ast.Call):
            connector = self.origin(node.func, depth + 1)
            if connector is not None and connector[1] == "connector":
                if node not in self.valid_connections:
                    self.valid_connections[node] = (
                        len(node.args) <= 1
                        and not any(isinstance(arg, ast.Starred) for arg in node.args)
                        and all(
                            keyword.arg is not None
                            and "factory" not in keyword.arg
                            and (
                                connector[0] != "sqlite3" or keyword.arg in _SQLITE_CONNECT_KEYWORDS
                            )
                            for keyword in node.keywords
                        )
                    )
                return (connector[0], "connection") if self.valid_connections[node] else None
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "cursor"
                and not node.args
                and not node.keywords
            ):
                owner = self.origin(node.func.value, depth + 1)
                if owner is not None and owner[1] == "connection":
                    return owner[0], "cursor"
        return None

    def _known_call(self, node: ast.Call) -> bool:
        if self.origin(node) is not None:
            return True
        if isinstance(node.func, ast.Attribute):
            owner = self.origin(node.func.value)
            return (
                owner is not None
                and owner[1] in {"connection", "cursor"}
                and node.func.attr
                in {
                    "execute",
                    "fetchone",
                    "fetchall",
                    "fetchmany",
                    "close",
                    "commit",
                    "rollback",
                }
            )
        return False

    def _passive_expression(self, node: ast.AST) -> bool:
        if isinstance(node, (ast.Constant, ast.Name)):
            return True
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self._passive_expression(element) for element in node.elts)
        if isinstance(node, ast.Dict):
            return all(isinstance(key, ast.Constant) for key in node.keys) and all(
                self._passive_expression(value) for value in node.values
            )
        if isinstance(node, ast.Call) and self._known_call(node):
            return all(self._passive_expression(arg) for arg in node.args) and all(
                keyword.arg is not None and self._passive_expression(keyword.value)
                for keyword in node.keywords
            )
        return isinstance(node, ast.Attribute) and self.origin(node) is not None

    def _passive_prefix(self, statements: list[ast.stmt]) -> bool:
        """Prove the module prefix once, stopping at implicit or opaque effects.

        Commit calls only after their entire expression/context is passive.
        Unsupported statements stop later proofs, including outside nested withs.
        """
        for statement in statements:
            if isinstance(statement, ast.Import):
                if any(alias.name not in _DRIVERS for alias in statement.names):
                    return False
                continue
            if isinstance(statement, ast.ImportFrom):
                if (
                    statement.level != 0
                    or statement.module not in _DRIVERS | {"__future__"}
                    or any(alias.name == "*" for alias in statement.names)
                ):
                    return False
                continue
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    return False
                expressions = [statement.value]
            elif isinstance(statement, ast.Expr):
                expressions = [statement.value]
            elif isinstance(statement, ast.With):
                expressions = [item.context_expr for item in statement.items]
                if any(
                    (origin := self.origin(expression)) is None
                    or origin[1] not in {"connection", "cursor"}
                    for expression in expressions
                ):
                    return False
            else:
                return False
            if not all(self._passive_expression(expression) for expression in expressions):
                return False
            self.passive_calls.update(
                node
                for expression in expressions
                for node in ast.walk(expression)
                if isinstance(node, ast.Call)
            )
            if isinstance(statement, ast.With) and not self._passive_prefix(statement.body):
                return False
        return True

    def _span(self, node: ast.AST) -> tuple[int, int] | None:
        start_line = getattr(node, "lineno", 0)
        end_line = getattr(node, "end_lineno", 0)
        start_column = self.parsed.character_column(start_line, getattr(node, "col_offset", None))
        end_column = self.parsed.character_column(end_line, getattr(node, "end_col_offset", None))
        if start_column is None or end_column is None or start_line < 1 or end_line < 1:
            return None
        return (
            self.parsed.line_character_starts[start_line - 1] + start_column,
            self.parsed.line_character_starts[end_line - 1] + end_column,
        )

    def safe_call_spans(self) -> tuple[tuple[int, int, int], ...]:
        calls = [node for node in self.nodes if isinstance(node, ast.Call)]
        # Reflective writes can happen in closures or deferred code, regardless
        # of their lexical order. Never infer a stable API in such a module.
        if any(
            isinstance(node.func, ast.Name)
            and node.func.id
            in {
                "setattr",
                "delattr",
                "exec",
                "eval",
                "globals",
                "locals",
                "vars",
            }
            for node in calls
        ):
            return ()
        assert self.parsed.tree is not None
        self._passive_prefix(self.parsed.tree.body)
        spans = []
        for call in calls:
            if call not in self.passive_calls:
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "execute":
                continue
            owner = self.origin(call.func.value)
            if owner is None or (
                owner[1] != "cursor"
                and owner not in {("sqlite3", "connection"), ("psycopg", "connection")}
            ):
                continue
            query: ast.expr | None
            params: ast.expr | None
            if len(call.args) == 2 and not call.keywords:
                query = call.args[0]
                params = call.args[1]
            elif not call.args and owner[0] in {"psycopg", "psycopg2"} and len(call.keywords) == 2:
                values = {keyword.arg: keyword.value for keyword in call.keywords}
                query = values.get("query")
                params = values.get("params")
            else:
                continue
            if (
                not isinstance(query, ast.Constant)
                or not isinstance(query.value, str)
                or params is None
                or isinstance(params, ast.Starred)
            ):
                continue
            # The old regex can consume an outer call through a nested unsafe
            # call. Keep that lexical finding rather than hide its only match.
            if any(isinstance(node, ast.Call) for node in ast.walk(params)):
                continue
            function_span, call_span = self._span(call.func), self._span(call)
            if function_span is not None and call_span is not None:
                spans.append((*function_span, call_span[1]))
        return tuple(sorted(spans))


def bound_sql_call_spans(
    parsed: ParsedPythonFile | None, content: str
) -> tuple[tuple[int, int, int], ...]:
    """Return exact callee/call bounds only when the unchanged source proves binding."""
    if parsed is None or parsed.tree is None or parsed.content != content:
        return ()
    proof = _BindingProof(parsed)
    return proof.safe_call_spans() if proof.complete else ()
