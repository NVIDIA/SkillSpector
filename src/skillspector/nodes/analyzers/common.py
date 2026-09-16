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
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from skillspector.models import Finding, Location
from skillspector.python_ast import build_import_aliases

# Keep the analyzer and runner fence walkers lexically aligned without sharing
# their state machines, since they consume different coordinate systems.
MARKDOWN_FENCE_OPEN = re.compile(r"^[ ]{0,3}(`{3,}(?=[^`\r\n]*$)|~{3,})[^\r\n]*$")
MARKDOWN_FENCE_CLOSE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*$")
LOGICAL_LINE_BREAK = re.compile(r"\r\n|[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")
LINE_BREAK_CHARS = "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
MAX_FINDING_CONTEXT_CHARS = 1_000


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


def is_code_example(context: str, *, path: str = "") -> bool:
    """Return True when the context appears to be a code example or documentation snippet.

    SKILL.md is the primary attack surface and is never treated as a code example:
    an attacker can place a documentation-style phrase (``for example``, a fenced
    code block, ...) a few lines from an injected instruction to suppress the finding.
    """
    if path.replace("\\", "/").lower().endswith("skill.md"):
        return False
    ctx_lower = context.lower()
    return any(ind in ctx_lower for ind in _CODE_EXAMPLE_INDICATORS)


def get_line_number(content: str, offset: int) -> int:
    """Return the 1-based line number for a character offset in *content*."""
    return sum(1 for _ in LOGICAL_LINE_BREAK.finditer(content, 0, offset)) + 1


def logical_line_starts(content: str) -> tuple[int, ...]:
    """Return character offsets for every logical line start in *content*."""
    return (0, *(separator.end() for separator in LOGICAL_LINE_BREAK.finditer(content)))


@dataclass(frozen=True)
class SourceLocationIndex:
    """Map character offsets to public locations using one shared line index."""

    content: str
    file_path: str
    line_starts: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "line_starts",
            logical_line_starts(self.content),
        )

    def line_and_column(self, offset: int) -> tuple[int, int]:
        """Return a one-based line and zero-based character column."""
        bounded = min(max(offset, 0), len(self.content))
        line_index = max(0, bisect_right(self.line_starts, bounded) - 1)
        return line_index + 1, bounded - self.line_starts[line_index]

    def location(self, start_offset: int, end_offset: int) -> Location:
        """Build an exact location with zero-based, end-exclusive columns."""
        start_line, start_column = self.line_and_column(start_offset)
        end_line, end_column = self.line_and_column(end_offset)
        return Location(
            file=self.file_path,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
        )


def get_context(content: str, match_start: int, context_lines: int = 3) -> str:
    """Extract surrounding lines from *content* around the match at *match_start* (char offset)."""
    lines = content.splitlines()
    match_line = get_line_number(content, match_start) - 1
    start_line = max(0, match_line - context_lines)
    end_line = min(len(lines), match_line + context_lines + 1)
    selected_lines = lines[start_line:end_line]
    if not selected_lines:
        return ""
    relative_line = min(match_line - start_line, len(selected_lines) - 1)
    line_start = content.rfind("\n", 0, match_start) + 1
    column = min(max(0, match_start - line_start), len(selected_lines[relative_line]))
    anchor = sum(len(line) + 1 for line in selected_lines[:relative_line]) + column
    return _bounded_context("\n".join(selected_lines), anchor)


def get_context_from_lines(
    lines: list[str],
    lineno: int,
    window: int = 3,
    *,
    column: int = 0,
) -> str:
    """Extract bounded context around a 1-based line and character column."""
    start = max(0, lineno - 1 - window)
    end = min(len(lines), lineno + window)
    selected_lines = lines[start:end]
    if not selected_lines:
        return ""
    relative_line = min(max(0, lineno - 1 - start), len(selected_lines) - 1)
    bounded_column = min(max(0, column), len(selected_lines[relative_line]))
    anchor = sum(len(line) + 1 for line in selected_lines[:relative_line]) + bounded_column
    context_length = sum(len(line) for line in selected_lines) + len(selected_lines) - 1
    if context_length <= MAX_FINDING_CONTEXT_CHARS:
        return "\n".join(selected_lines)

    half_window = MAX_FINDING_CONTEXT_CHARS // 2
    slice_start = min(
        max(0, anchor - half_window),
        context_length - MAX_FINDING_CONTEXT_CHARS,
    )
    slice_end = slice_start + MAX_FINDING_CONTEXT_CHARS
    pieces: list[str] = []
    offset = 0
    for index, line in enumerate(selected_lines):
        line_end = offset + len(line)
        overlap_start = max(slice_start, offset)
        overlap_end = min(slice_end, line_end)
        if overlap_start < overlap_end:
            pieces.append(line[overlap_start - offset : overlap_end - offset])
        if index + 1 < len(selected_lines) and slice_start <= line_end < slice_end:
            pieces.append("\n")
        offset = line_end + 1
        if offset >= slice_end:
            break
    return "".join(pieces)


def _bounded_context(context: str, anchor: int) -> str:
    """Return a bounded context window that retains the finding anchor."""
    if len(context) <= MAX_FINDING_CONTEXT_CHARS:
        return context
    half_window = MAX_FINDING_CONTEXT_CHARS // 2
    start = min(
        max(0, anchor - half_window),
        len(context) - MAX_FINDING_CONTEXT_CHARS,
    )
    return context[start : start + MAX_FINDING_CONTEXT_CHARS]


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


def build_type_map(
    tree: ast.Module, import_aliases: dict[str, str] | None = None
) -> dict[str, str]:
    """Infer variable types from constructor calls.

    Scans assignments (``var = Type(...)``) and ``with`` statements
    (``with Type() as var``) and records ``{var: "fully.qualified.Type"}``.
    Import aliases are resolved so ``from pathlib import Path; p = Path(x)``
    maps ``p`` → ``"pathlib.Path"``.
    """
    import_aliases = build_import_aliases(tree) if import_aliases is None else import_aliases
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


def get_complete_source_segment(lines: list[str], lineno: int, end_lineno: int | None) -> str:
    """Extract the complete source text for a given line range."""
    start = max(0, lineno - 1)
    end = end_lineno or lineno
    return "\n".join(lines[start:end])


def get_source_segment(lines: list[str], lineno: int, end_lineno: int | None) -> str:
    """Extract a 200-character source preview for a given line range."""
    return get_complete_source_segment(lines, lineno, end_lineno)[:200]
