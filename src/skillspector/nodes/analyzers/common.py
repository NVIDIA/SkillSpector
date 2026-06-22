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

"""Shared helpers for analyzer nodes."""

from __future__ import annotations

import ast
from typing import Any

from skillspector.models import Finding


def make_dummy_finding(analyzer_id: str) -> Finding:
    """Create a deterministic dummy finding for a stub analyzer."""
    return Finding(
        rule_id=analyzer_id,
        message=f"Stub finding from {analyzer_id}",
        severity="LOW",
        confidence=0.5,
        file="SKILL.md",
        start_line=1,
    )


_CODE_EXAMPLE_INDICATORS: tuple[str, ...] = (
    "```",
    "example:",
    "for example",
    "e.g.",
    "such as",
    "documentation",
    "# warning:",
    "# note:",
    "**warning**",
    "**note**",
    # Code comments containing the match are almost always false positives
    "// ✅",
    "// ❌",
    "// good:",
    "// bad:",
    "// correct:",
    "// incorrect:",
    "// wrong:",
)


def is_code_example(context: str) -> bool:
    """Return True when the context appears to be a code example or documentation snippet."""
    ctx_lower = context.lower()
    return any(ind in ctx_lower for ind in _CODE_EXAMPLE_INDICATORS)


def get_line_number(content: str, offset: int) -> int:
    """Return the 1-based line number for a character offset in *content*."""
    return content[:offset].count("\n") + 1


def get_context(content: str, match_start: int, context_lines: int = 3) -> str:
    """Extract surrounding lines from *content* around the match at *match_start* (char offset)."""
    lines = content.splitlines()
    match_line = content[:match_start].count("\n")
    start_line = max(0, match_line - context_lines)
    end_line = min(len(lines), match_line + context_lines + 1)
    return "\n".join(lines[start_line:end_line])


def get_context_from_lines(lines: list[str], lineno: int, window: int = 3) -> str:
    """Extract surrounding lines given pre-split *lines* and a 1-based *lineno*."""
    start = max(0, lineno - 1 - window)
    end = min(len(lines), lineno + window)
    return "\n".join(lines[start:end])


def resolve_dotted_name(node: ast.expr) -> str | None:
    """Build a dotted name string from a Name or Attribute node.

    Examples: ``ast.Name(id='exec')`` → ``'exec'``,
    ``ast.Attribute(value=Name('os'), attr='system')`` → ``'os.system'``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = [node.attr]
        current: Any = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return None


def resolve_call_name(node: ast.Call, import_aliases: dict[str, str] | None = None) -> str | None:
    """Extract a dotted call name like ``'os.system'`` from a Call node, resolving aliases."""
    name = resolve_dotted_name(node.func)
    if name and import_aliases:
        root, _, rest = name.partition(".")
        resolved_root = import_aliases.get(root, root)
        return f"{resolved_root}.{rest}" if rest else resolved_root
    return name


def _build_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map locally imported names to their fully-qualified module paths.

    ``from pathlib import Path`` → ``{"Path": "pathlib.Path"}``
    ``import socket``           → ``{"socket": "socket"}``
    ``import pathlib``          → ``{"pathlib": "pathlib"}``
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{module}.{alias.name}" if module else alias.name
    return aliases


def build_type_map(tree: ast.Module) -> dict[str, str]:
    """Infer variable types from constructor calls.

    Scans assignments (``var = Type(...)``) and ``with`` statements
    (``with Type() as var``) and records ``{var: "fully.qualified.Type"}``.
    Import aliases are resolved so ``from pathlib import Path; p = Path(x)``
    maps ``p`` → ``"pathlib.Path"``.
    """
    import_aliases = _build_import_aliases(tree)
    type_map: dict[str, str] = {}

    def _resolve_ctor(call_node: ast.Call) -> str | None:
        raw = resolve_dotted_name(call_node.func)
        if raw is None:
            return None
        root, _, rest = raw.partition(".")
        resolved_root = import_aliases.get(root, root)
        return f"{resolved_root}.{rest}" if rest else resolved_root

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            ctor = _resolve_ctor(node.value)
            if ctor:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        type_map[target.id] = ctor
        elif isinstance(node, ast.With):
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and item.optional_vars is not None
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    ctor = _resolve_ctor(item.context_expr)
                    if ctor:
                        type_map[item.optional_vars.id] = ctor

    return type_map


def resolve_call_name_typed(node: ast.Call, type_map: dict[str, str] | None = None) -> str | None:
    """Like ``resolve_call_name`` but consults *type_map* for instance methods.

    For ``sock.recv(1024)`` where *type_map* maps ``sock`` → ``socket.socket``,
    this returns ``"socket.socket.recv"`` instead of ``"sock.recv"``.
    """
    plain = resolve_dotted_name(node.func)
    if plain is None or type_map is None or "." not in plain:
        return plain
    root, _, rest = plain.partition(".")
    inferred = type_map.get(root)
    if inferred is None:
        return plain
    return f"{inferred}.{rest}"


def get_source_segment(lines: list[str], lineno: int, end_lineno: int | None) -> str:
    """Extract the source text for a given line range, truncated to 200 chars."""
    start = max(0, lineno - 1)
    end = end_lineno or lineno
    return "\n".join(lines[start:end])[:200]
