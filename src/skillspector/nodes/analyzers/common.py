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


def _strip_builtins_prefix(name: str) -> str:
    """Collapse a ``builtins``-qualified name back to its bare builtin name.

    ``builtins.exec`` → ``exec`` (and ``builtins.eval``/``compile``/``__import__``…).
    The analyzers match dangerous builtins by their bare name (``call_name == "exec"``,
    ``name in _EXEC_SINKS``), but ``from builtins import exec`` / ``import builtins;
    builtins.exec(...)`` resolve, through the import-alias map, to the *qualified*
    spelling ``builtins.exec`` — which would otherwise slip past those checks. Since
    ``builtins.exec is exec`` at runtime, collapsing the prefix is semantically exact
    and re-enters the existing bare-name detection.

    Only the single-segment form ``builtins.<attr>`` is collapsed; deeper chains
    (``builtins.foo.bar``) are left untouched as they are not direct builtin calls.
    """
    root, sep, rest = name.partition(".")
    if root == "builtins" and sep and "." not in rest:
        return rest
    return name


def apply_import_aliases(name: str, aliases: dict[str, str]) -> str:
    """Rewrite a resolved call name to its fully-qualified form using import aliases.

    Bridges several evasion-prone spellings back to the canonical name that the
    analyzers match against:

    - ``from os import system`` → ``{"system": "os.system"}`` so a bare ``system``
      call resolves to ``"os.system"``.
    - ``import os as o`` → ``{"o": "os"}`` so ``o.system`` resolves to ``"os.system"``.
    - ``from builtins import exec`` / ``import builtins; builtins.exec(...)`` → the
      bare builtin ``exec`` (via :func:`_strip_builtins_prefix`), so dangerous
      builtins matched by bare name are not hidden behind a ``builtins.`` qualifier.

    Idempotent for already-canonical names (``os.system`` stays ``os.system``).
    """
    if name in aliases:
        return _strip_builtins_prefix(aliases[name])
    root, sep, rest = name.partition(".")
    if sep and root in aliases:
        return _strip_builtins_prefix(f"{aliases[root]}.{rest}")
    return _strip_builtins_prefix(name)


def resolve_call_name(node: ast.Call, aliases: dict[str, str] | None = None) -> str | None:
    """Extract a dotted call name like ``'os.system'`` from a Call node.

    When *aliases* (from :func:`build_import_aliases`) is supplied, locally aliased or
    ``from``-imported names are normalized to their fully-qualified form so that
    ``import os as o; o.system(...)`` and ``from os import system; system(...)`` both
    resolve to ``"os.system"``.
    """
    name = resolve_dotted_name(node.func)
    if name is not None and aliases:
        name = apply_import_aliases(name, aliases)
    return name


def _dynamic_import_target(node: ast.expr, aliases: dict[str, str] | None = None) -> str | None:
    """Return the imported module name for an ``importlib.import_module('mod')`` call.

    Recognizes both ``importlib.import_module('os')`` and the bare-imported
    ``from importlib import import_module; import_module('os')`` (resolved via the
    import-alias map), returning the string literal module name (``'os'``) when the
    first positional argument is a constant. Returns ``None`` for anything else
    (non-literal argument, unrelated call), so callers stay precise and avoid false
    positives on dynamic module names the analyzer cannot resolve statically.
    """
    if not isinstance(node, ast.Call):
        return None
    func_name = resolve_dotted_name(node.func)
    if func_name is not None and aliases:
        func_name = apply_import_aliases(func_name, aliases)
    if func_name not in ("importlib.import_module", "import_module"):
        return None
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return None


def resolve_dynamic_import_call(
    node: ast.Call, aliases: dict[str, str] | None = None
) -> str | None:
    """Resolve ``importlib.import_module('mod').attr(...)`` to the dotted sink ``'mod.attr'``.

    Bridges the dynamic-import evasion that mirrors ``__import__``: a skill writes
    ``importlib.import_module('os').system(cmd)`` (or imports ``import_module`` bare)
    so the dangerous module never appears as a static ``import``. When *node*'s callee
    is an attribute access on such a chain, this returns the canonical sink name
    (``'os.system'``, ``'subprocess.run'``) that the existing sink ladders already
    match. Returns ``None`` when the chain is not a literal dynamic import, keeping the
    resolution precise (no false positives on un-resolvable dynamic names).
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    module_name = _dynamic_import_target(func.value, aliases)
    if module_name is None:
        return None
    return f"{module_name}.{func.attr}"


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


def build_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map locally bound names to their fully-qualified import paths.

    Public entry point around the import scan already used by :func:`build_type_map`.
    Callers pass the result to :func:`resolve_call_name` /
    :func:`resolve_call_name_typed` to defeat import-alias evasion.
    """
    return _build_import_aliases(tree)


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


def resolve_call_name_typed(
    node: ast.Call,
    type_map: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Like ``resolve_call_name`` but consults *type_map* for instance methods.

    For ``sock.recv(1024)`` where *type_map* maps ``sock`` → ``socket.socket``,
    this returns ``"socket.socket.recv"`` instead of ``"sock.recv"``.

    When *aliases* (from :func:`build_import_aliases`) is supplied, import-aliased and
    ``from``-imported names are also normalized, so ``import subprocess as sp; sp.run``
    resolves to ``"subprocess.run"`` and ``from subprocess import run; run`` to the same.
    """
    plain = resolve_dotted_name(node.func)
    if plain is None:
        return None
    # Normalize the locally written spelling first. ``type_map`` values are already
    # canonical (``build_type_map`` resolves import aliases when recording them), so
    # aliasing must run before — not after — the type-map lookup to avoid re-expanding
    # an already-resolved name (e.g. ``from socket import socket`` would otherwise turn
    # ``socket.socket.recv`` into ``socket.socket.socket.recv``).
    if aliases:
        plain = apply_import_aliases(plain, aliases)
    if type_map is not None and "." in plain:
        root, _, rest = plain.partition(".")
        inferred = type_map.get(root)
        if inferred is not None:
            plain = f"{inferred}.{rest}"
    return plain


def get_source_segment(lines: list[str], lineno: int, end_lineno: int | None) -> str:
    """Extract the source text for a given line range, truncated to 200 chars."""
    start = max(0, lineno - 1)
    end = end_lineno or lineno
    return "\n".join(lines[start:end])[:200]
