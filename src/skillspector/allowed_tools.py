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

"""Split and validate Agent Skills ``allowed-tools`` grants.

A string ``allowed-tools`` value lists grants separated by commas or by
whitespace. Each grant is a bare tool name (``Read``) or a scoped grant in the
``Tool(specifier)`` form (``Bash(git status:*)``, ``Read(./docs, notes.md)``).
As in Claude Code's permission rule syntax, parentheses inside the specifier
are literal characters, not nested structure: ``Bash(echo '(')`` is a complete
grant. A separator inside a specifier belongs to the specifier, so a scoped
grant is kept whole and its specifier text is never read as another grant.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# Tool names longer than this are not looked up.
MAX_ALLOWED_TOOL_NAME_CHARS = 64

_TOKENS = re.compile(r"[(),]|\s+")
_NON_WHITESPACE = re.compile(r"\S")
# Same tool-name alphabet as the settings permission-rule check in
# ``nodes/analyzers/bundled_execution_surface.py`` (``_permission_rule_is_valid``).
# Unlike that check, the specifier's parentheses do not have to balance.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_*.-]+")


def _ends_specifier(value: str, end: int) -> bool:
    """Return whether a ``)`` ending at ``end`` is followed by a separator or the end."""
    return end == len(value) or value[end] == "," or value[end].isspace()


def _separators(value: str) -> Iterator[re.Match[str]]:
    """Yield the commas and whitespace runs of ``value`` that lie outside a specifier.

    A ``(`` outside a specifier opens one. Inside it, ``(`` and ``)`` are
    literal, except that a ``)`` followed by a comma, whitespace or the end of
    the value closes it.
    """
    in_specifier = False
    for match in _TOKENS.finditer(value):
        token = match.group()
        if in_specifier:
            if token == ")" and _ends_specifier(value, match.end()):
                in_specifier = False
        elif token == "(":
            in_specifier = True
        elif token != ")":
            yield match


def iter_allowed_tools_entries(value: str) -> Iterator[str]:
    """Yield the stripped entries of a string ``allowed-tools`` value.

    A comma outside any specifier selects the comma-separated form; otherwise
    entries are separated by whitespace. Only separators outside a specifier
    split. A specifier runs from a ``(`` to the first ``)`` that is followed by
    a comma, whitespace or the end of the value, and any ``(`` or ``)`` in
    between is literal. So ``"Bash(git status:*) Read"`` and
    ``"Bash(echo '(') Read"`` each yield two entries, and
    ``"Read(./docs Bash(notes).md)"`` and ``"Read(./a, b)"`` each stay one
    entry. A specifier whose own text has a ``)`` followed by a separator
    cannot be written in a string value; the list form keeps it whole. An
    unclosed ``(`` keeps the rest of the value in its entry, which
    :func:`allowed_tool_grant_name` rejects.

    Entries are produced lazily. As with ``str.split``, the comma form yields
    an empty entry between adjacent commas and the whitespace form yields no
    empty entries; callers skip empty entries.
    """
    comma_form = any(match.group() == "," for match in _separators(value))
    start = 0
    for match in _separators(value):
        if comma_form and match.group() != ",":
            continue
        entry = value[start : match.start()].strip()
        if entry or comma_form:
            yield entry
        start = match.end()
    entry = value[start:].strip()
    if entry or comma_form:
        yield entry


def allowed_tool_grant_name(entry: str) -> str | None:
    """Return the tool name of a complete ``allowed-tools`` grant, else ``None``.

    A complete grant is either a bare tool name or ``Tool(specifier)``, where
    the specifier is everything between the first ``(`` and the final ``)``,
    which must be the last character, and is not blank. Parentheses inside the
    specifier are literal, so ``Bash(echo '(')`` and ``Bash(echo ')')`` are
    complete. Incomplete grants such as ``Bash(``, ``Bash(notes`` or
    ``Bash(echo '('`` and blank ones such as ``Bash()`` return ``None``, so
    they name no tool. Names longer than :data:`MAX_ALLOWED_TOOL_NAME_CHARS`
    also return ``None``.
    """
    entry = entry.strip()
    # Bound the search so a long specifier is not scanned for the name.
    paren = entry.find("(", 0, MAX_ALLOWED_TOOL_NAME_CHARS + 1)
    if paren == -1 and len(entry) > MAX_ALLOWED_TOOL_NAME_CHARS:
        return None
    name = entry if paren == -1 else entry[:paren]
    if _TOOL_NAME.fullmatch(name) is None:
        return None
    if paren == -1:
        return name
    if not entry.endswith(")") or _NON_WHITESPACE.search(entry, paren + 1, len(entry) - 1) is None:
        return None
    return name
