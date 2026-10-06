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
A separator inside a specifier belongs to the specifier, so a scoped grant is
kept whole and its specifier text is never read as another grant.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# Tool names longer than this are not looked up.
MAX_ALLOWED_TOOL_NAME_CHARS = 64

_COMMA_FORM_TOKENS = re.compile(r"[(),]")
_WHITESPACE_FORM_TOKENS = re.compile(r"[()]|\s+")
_PARENTHESES = re.compile(r"[()]")
_NON_WHITESPACE = re.compile(r"\S")
# Same tool-name alphabet as the settings permission-rule check in
# ``nodes/analyzers/bundled_execution_surface.py`` (``_permission_rule_is_valid``).
_TOOL_NAME = re.compile(r"[A-Za-z0-9_*.-]+")


def _has_top_level_comma(value: str) -> bool:
    depth = 0
    for match in _COMMA_FORM_TOKENS.finditer(value):
        token = match.group()
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            return True
    return False


def iter_allowed_tools_entries(value: str) -> Iterator[str]:
    """Yield the stripped entries of a string ``allowed-tools`` value.

    A comma outside any parentheses selects the comma-separated form;
    otherwise entries are separated by whitespace. Only separators outside
    parentheses split, so ``"Bash(git status:*) Read"`` yields
    ``Bash(git status:*)`` and ``Read``, and ``"Read(./a, b)"`` stays one
    entry. An unclosed ``(`` keeps the rest of the value in its entry, and a
    stray ``)`` closes nothing; :func:`allowed_tool_grant_name` rejects both.

    Entries are produced lazily. As with ``str.split``, the comma form yields
    an empty entry between adjacent commas and the whitespace form yields no
    empty entries; callers skip empty entries.
    """
    comma_form = _has_top_level_comma(value)
    pattern = _COMMA_FORM_TOKENS if comma_form else _WHITESPACE_FORM_TOKENS
    depth = 0
    start = 0
    for match in pattern.finditer(value):
        token = match.group()
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            entry = value[start : match.start()].strip()
            if entry or comma_form:
                yield entry
            start = match.end()
    entry = value[start:].strip()
    if entry or comma_form:
        yield entry


def allowed_tool_grant_name(entry: str) -> str | None:
    """Return the tool name of a well-formed ``allowed-tools`` grant, else ``None``.

    A well-formed grant is either a bare tool name or ``Tool(specifier)``
    where the specifier is not blank, its parentheses balance, and the ``)``
    that closes the first ``(`` is the last character. Incomplete or
    malformed grants such as ``Bash(``, ``Bash(notes``, ``Bash()`` or
    ``Bash(notes).md)`` return ``None``, so they name no tool. Names longer
    than :data:`MAX_ALLOWED_TOOL_NAME_CHARS` also return ``None``.
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
    depth = 0
    for match in _PARENTHESES.finditer(entry, paren + 1, len(entry) - 1):
        depth += 1 if match.group() == "(" else -1
        if depth < 0:
            return None
    return name if depth == 0 else None
