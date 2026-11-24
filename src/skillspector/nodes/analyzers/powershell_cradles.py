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

"""Locate PowerShell download-and-execute cradles with a small lexical pass.

A regular expression cannot tell a statement separator from a semicolon inside
a hashtable argument, a comment from code, or a quoted string from a command.
This module tokenizes PowerShell just far enough to answer those questions in
one linear pass:

* quoted strings, here-strings and comments are single tokens, so text inside
  them is never read as a command;
* ``( )``, ``{ }``, ``@( )``, ``@{ }`` and ``$( )`` are nested groups, so a
  ``;`` or newline inside one stays part of the argument that owns it;
* a backtick-newline, a trailing ``|`` and a leading ``|`` on the next line
  continue the pipeline, while an unquoted ``;``, ``||``, ``&&`` or newline at
  the owning level ends the statement.

A cradle is reported only when fetched content actually reaches an executor:
piped into ``iex`` or a PowerShell host that runs stdin, passed as the argument
of ``iex``, or turned into a script block that is invoked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_MAX_GROUP_DEPTH = 32
_MAX_NESTED_HOST_DEPTH = 2

_FETCH_COMMANDS = frozenset(
    {
        "iwr",
        "irm",
        "invoke-webrequest",
        "invoke-restmethod",
        "curl",
        "wget",
        "curl.exe",
        "wget.exe",
    }
)
_EXEC_COMMANDS = frozenset({"iex", "invoke-expression"})
_HOST_COMMANDS = frozenset({"pwsh", "powershell", "pwsh.exe", "powershell.exe"})
# Host switches that consume the next token as their value.
_HOST_VALUE_FLAGS = frozenset(
    {
        "executionpolicy",
        "ep",
        "windowstyle",
        "w",
        "inputformat",
        "if",
        "outputformat",
        "of",
        "version",
        "v",
        "workingdirectory",
        "wd",
        "configurationname",
        "config",
        "custompipename",
        "settingsfile",
        "psconsolefile",
    }
)
_COMMAND_FLAG_NAMES = ("c", "co", "com", "comm", "comma", "comman", "command")
_FILE_FLAG_NAMES = ("f", "fi", "fil", "file")
_ENCODED_FLAG_NAMES = ("e", "ec", "enc", "enco", "encod", "encode", "encoded", "encodedc")
_CREATE_SCRIPTBLOCK = re.compile(r"\[(?:[\w.]+\.)?scriptblock\]::create", re.IGNORECASE)
_EXECUTOR_WORD = re.compile(r"(?<![\w-])(?:iex|invoke-expression)(?![\w-])", re.IGNORECASE)
_URL_ATOM = re.compile(r"[\"']?https?://", re.IGNORECASE)
_ORDINARY_RUN = re.compile(r"[^\s`'\"|;&(){}#<@]+")
_PROSE_START = re.compile(
    r"(?<![\w-])(?:iwr|irm|curl|wget|iex|invoke-\w+|powershell|pwsh|\[scriptblock\]|\(\s*new-object)",
    re.IGNORECASE,
)
_MAX_PROSE_STARTS = 32
_HEREDOC_END = {"'": "\n'@", '"': '\n"@'}


@dataclass
class _Atom:
    kind: str  # "word", "str" or "group"
    start: int
    end: int
    text: str
    children: list[list[list[_Atom]]] = field(default_factory=list)
    inner: tuple[int, int] | None = None


_Element = list[_Atom]
_Statement = list[_Element]


class TextMatch:
    """The slice of ``re.Match`` that SC2 reporting needs."""

    def __init__(self, content: str, start: int, end: int, source: str) -> None:
        self._content = content
        self._start = start
        self._end = end
        self.source = source

    def start(self, group: int = 0) -> int:
        return self._start

    def end(self, group: int = 0) -> int:
        return self._end

    def span(self, group: int = 0) -> tuple[int, int]:
        return (self._start, self._end)

    def group(self, group: int = 0) -> str:
        return self._content[self._start : self._end]


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.i = 0
        self._sig_cache: tuple[int, int, int] | None = None

    # -- lexical helpers ---------------------------------------------------

    def _skip_comment(self, i: int) -> int:
        text = self.text
        if text.startswith("<#", i):
            end = text.find("#>", i + 2)
            return self.n if end < 0 else end + 2
        end = i
        while end < self.n and text[end] not in "\r\n":
            end += 1
        return end

    def _next_significant(self, j: int) -> int:
        """Return the first position at or after ``j`` that is not blank or comment."""
        cache = self._sig_cache
        if cache is not None and cache[0] <= j <= cache[1]:
            return cache[2]
        text, k = self.text, j
        while k < self.n:
            ch = text[k]
            if ch in " \t\r\n\f\v":
                k += 1
            elif ch == "#" or text.startswith("<#", k):
                k = self._skip_comment(k)
            else:
                break
        self._sig_cache = (j, k, k)
        return k

    def _continues_with_pipe(self, j: int) -> bool:
        k = self._next_significant(j)
        return k < self.n and self.text[k] == "|"

    def _single_quoted_end(self, i: int) -> int:
        text, n = self.text, self.n
        i += 1
        while i < n:
            if text[i] == "'":
                if i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                return i + 1
            i += 1
        return n

    def _double_quoted_end(self, i: int) -> int:
        text, n = self.text, self.n
        i += 1
        while i < n:
            ch = text[i]
            if ch == "`":
                i += 2
                continue
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    i += 2
                    continue
                return i + 1
            i += 1
        return n

    def _here_string_end(self, i: int) -> int | None:
        """End of ``@'...'@`` / ``@"..."@`` starting at ``i`` or ``None``."""
        text = self.text
        quote = text[i + 1] if i + 1 < self.n else ""
        if quote not in _HEREDOC_END:
            return None
        k = i + 2
        while k < self.n and text[k] in " \t":
            k += 1
        if k >= self.n or text[k] not in "\r\n":
            return None
        end = text.find(_HEREDOC_END[quote], k)
        return self.n if end < 0 else end + len(_HEREDOC_END[quote])

    # -- parser ------------------------------------------------------------

    def parse_block(self, closer: str | None, depth: int) -> list[_Statement]:
        text, n = self.text, self.n
        statements: list[_Statement] = []
        pipeline: _Statement = []
        atoms: _Element = []
        word_start: int | None = None
        after_pipe = False

        def end_word(at: int) -> None:
            nonlocal word_start
            if word_start is not None:
                atoms.append(_Atom("word", word_start, at, text[word_start:at]))
                word_start = None

        def end_element() -> None:
            nonlocal atoms
            if atoms:
                pipeline.append(atoms)
                atoms = []

        def end_statement() -> None:
            nonlocal pipeline
            end_element()
            if pipeline:
                statements.append(pipeline)
                pipeline = []

        while self.i < n:
            i = self.i
            c = text[i]
            if closer is not None and c == closer:
                end_word(i)
                self.i = i + 1
                end_statement()
                return statements
            if c in " \t\f\v":
                end_word(i)
                self.i = i + 1
            elif c == "`":
                nxt = text[i + 1 : i + 3]
                if nxt.startswith("\r\n") or nxt[:1] == "\n":
                    end_word(i)
                    self.i = i + (3 if nxt.startswith("\r\n") else 2)
                else:
                    if word_start is None:
                        word_start = i
                    after_pipe = False
                    self.i = min(n, i + 2)
            elif c in "\r\n":
                end_word(i)
                j = i + 2 if text.startswith("\r\n", i) else i + 1
                self.i = j
                if not (after_pipe or self._continues_with_pipe(j)):
                    end_statement()
            elif c == ";":
                end_word(i)
                end_statement()
                after_pipe = False
                self.i = i + 1
            elif c == "|":
                end_word(i)
                if text.startswith("||", i):
                    end_statement()
                    after_pipe = False
                    self.i = i + 2
                else:
                    end_element()
                    after_pipe = True
                    self.i = i + 1
            elif c == "&":
                if text.startswith("&&", i):
                    end_word(i)
                    end_statement()
                    after_pipe = False
                    self.i = i + 2
                elif word_start is None:
                    atoms.append(_Atom("word", i, i + 1, "&"))
                    after_pipe = False
                    self.i = i + 1
                else:
                    self.i = i + 1
            elif c == "#" and word_start is None:
                self.i = self._skip_comment(i)
            elif c == "<" and word_start is None and text.startswith("<#", i):
                self.i = self._skip_comment(i)
            elif c in "'\"" or (c == "@" and i + 1 < n and text[i + 1] in "'\""):
                here_end = self._here_string_end(i) if c == "@" else None
                if here_end is not None:
                    end, inner = here_end, None
                elif c == "@":
                    # ``@'`` that is not a here-string: ordinary word text.
                    if word_start is None:
                        word_start = i
                    after_pipe = False
                    self.i = i + 1
                    continue
                elif c == "'":
                    end = self._single_quoted_end(i)
                    inner = (i + 1, max(i + 1, end - 1))
                else:
                    end = self._double_quoted_end(i)
                    inner = (i + 1, max(i + 1, end - 1))
                after_pipe = False
                self.i = end
                if word_start is None:
                    atoms.append(_Atom("str", i, end, text[i:end], inner=inner))
            elif c in "({" and depth < _MAX_GROUP_DEPTH:
                end_word(i)
                self.i = i + 1
                children = self.parse_block(")" if c == "(" else "}", depth + 1)
                atoms.append(_Atom("group", i, self.i, text[i : self.i], children))
                after_pipe = False
            else:
                run = _ORDINARY_RUN.match(text, i)
                if word_start is None:
                    word_start = i
                after_pipe = False
                self.i = run.end() if run else i + 1
        end_word(n)
        end_statement()
        return statements


# -- semantic helpers --------------------------------------------------------


def _command_index(elem: _Element) -> int:
    """Index of the atom naming the command, past ``$x =`` and a call operator."""
    k = 0
    if (
        len(elem) >= 2
        and elem[0].kind == "word"
        and elem[0].text.startswith("$")
        and elem[1].kind == "word"
        and elem[1].text in ("=", "+=")
    ):
        k = 2
    if k < len(elem) and elem[k].kind == "word" and elem[k].text == "&":
        k += 1
    return k


def _command_name(elem: _Element) -> str:
    k = _command_index(elem)
    if k >= len(elem) or elem[k].kind != "word":
        return ""
    name = elem[k].text.lower()
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def _has_download_string(elem: _Element) -> bool:
    return any(
        atom.kind == "word"
        and atom.text.lower().endswith(".downloadstring")
        and idx + 1 < len(elem)
        and elem[idx + 1].kind == "group"
        and elem[idx + 1].text.startswith("(")
        for idx, atom in enumerate(elem)
    )


def _is_fetch_command(elem: _Element) -> bool:
    # A fetch cmdlet with no operand downloads nothing.
    return _command_name(elem) in _FETCH_COMMANDS and len(elem) > _command_index(elem) + 1


def _produces_fetch(elem: _Element) -> bool:
    return (
        _is_fetch_command(elem)
        or _has_download_string(elem)
        or any(atom.kind == "group" and _group_has_fetch(atom) for atom in elem)
    )


def _group_has_fetch(group: _Atom) -> bool:
    return any(_produces_fetch(elem) for stmt in group.children for elem in stmt)


def _args_have_fetch(atoms: list[_Atom]) -> bool:
    return _has_download_string(atoms) or any(
        atom.kind == "group" and _group_has_fetch(atom) for atom in atoms
    )


def _fetch_operand(elem: _Element) -> str:
    """The first URL-looking operand of the fetch inside ``elem``, or ``""``."""
    for atom in elem:
        if atom.kind in ("word", "str") and _URL_ATOM.match(atom.text):
            return atom.text
        if atom.kind == "group":
            for stmt in atom.children:
                for inner in stmt:
                    found = _fetch_operand(inner)
                    if found:
                        return found
    return ""


def _host_program(args: list[_Atom]) -> tuple[str, _Atom | None]:
    """Classify what a PowerShell host will run: stdin, a command, a file or encoded."""
    i = 0
    while i < len(args):
        atom = args[i]
        if atom.kind == "word" and atom.text.startswith("-"):
            flag = atom.text.lstrip("-").split(":", 1)[0].lower()
            nxt = args[i + 1] if i + 1 < len(args) else None
            if atom.text == "-":
                return ("stdin", None)
            if flag in _HOST_VALUE_FLAGS:
                i += 2
                continue
            if flag in _ENCODED_FLAG_NAMES:
                return ("encoded", None)
            if flag in _COMMAND_FLAG_NAMES or flag in _FILE_FLAG_NAMES:
                kind = "command" if flag in _COMMAND_FLAG_NAMES else "file"
                if nxt is None or (nxt.kind == "word" and nxt.text == "-"):
                    return ("stdin", None)
                return (kind, nxt)
            i += 1
            continue
        # A bare first argument is the command to run.
        return ("command", atom)
    return ("stdin", None)


def _host_executes_stdin(elem: _Element) -> bool:
    k = _command_index(elem)
    kind, atom = _host_program(elem[k + 1 :])
    if kind == "stdin":
        return True
    # An explicit program that itself runs its input (``$input | iex``).
    return kind == "command" and atom is not None and bool(_EXECUTOR_WORD.search(atom.text))


def _is_executor(elem: _Element) -> bool:
    name = _command_name(elem)
    return name in _EXEC_COMMANDS or (name in _HOST_COMMANDS and _host_executes_stdin(elem))


def _creates_fetched_scriptblock(atoms: list[_Atom]) -> bool:
    for idx, atom in enumerate(atoms):
        if (
            atom.kind == "word"
            and _CREATE_SCRIPTBLOCK.fullmatch(atom.text)
            and idx + 1 < len(atoms)
            and atoms[idx + 1].kind == "group"
            and _group_has_fetch(atoms[idx + 1])
        ):
            return True
        if atom.kind == "group" and any(
            _creates_fetched_scriptblock(inner) for stmt in atom.children for inner in stmt
        ):
            return True
    return False


class _Scan:
    def __init__(self, text: str, base: int, depth: int) -> None:
        self.text = text
        self.base = base
        self.depth = depth
        self.found: dict[tuple[int, int], str] = {}
        self.scriptblock_vars: dict[str, tuple[int, int, str]] = {}

    def add(self, start: int, end: int, source: str) -> None:
        self.found.setdefault((self.base + start, self.base + end), source)

    def walk(self, statements: list[_Statement]) -> None:
        for stmt in statements:
            self.check_pipeline(stmt)
            for elem in stmt:
                self.check_element(elem)
                for atom in elem:
                    if atom.kind == "group":
                        self.walk(atom.children)

    def check_pipeline(self, stmt: _Statement) -> None:
        for left, right in zip(stmt, stmt[1:], strict=False):
            if _produces_fetch(left) and _is_executor(right):
                end = (
                    right[_command_index(right)].end
                    if _command_name(right) in _EXEC_COMMANDS
                    else right[-1].end
                )
                self.add(left[0].start, end, _fetch_operand(left))

    def check_element(self, elem: _Element) -> None:
        k = _command_index(elem)
        name = _command_name(elem)
        if name in _EXEC_COMMANDS and _args_have_fetch(elem[k + 1 :]):
            self.add(elem[0].start, elem[-1].end, _fetch_operand(elem))
        if name in _HOST_COMMANDS:
            self.check_host_command(elem, k)
        self.check_scriptblock(elem)

    def check_host_command(self, elem: _Element, k: int) -> None:
        kind, atom = _host_program(elem[k + 1 :])
        if kind != "command" or atom is None or atom.kind != "str" or atom.inner is None:
            return
        if self.depth >= _MAX_NESTED_HOST_DEPTH:
            return
        lo, hi = atom.inner
        for (start, end), source in _scan(self.text[lo:hi], self.base + lo, self.depth + 1).items():
            self.found.setdefault((start, end), source)

    def check_scriptblock(self, elem: _Element) -> None:
        call = elem[0].kind == "word" and elem[0].text in ("&", ".")
        if call and len(elem) > 1:
            target = elem[1]
            if target.kind == "group" and _creates_fetched_scriptblock([target]):
                self.add(elem[0].start, target.end, _fetch_operand([target]))
            elif target.kind == "word" and target.text.lower() in self.scriptblock_vars:
                self.report_variable(target.text.lower())
        for idx, atom in enumerate(elem):
            if atom.kind != "word":
                continue
            lowered = atom.text.lower()
            if lowered.startswith(".invoke") and idx > 0:
                prev = elem[idx - 1]
                if prev.kind == "group" and _creates_fetched_scriptblock([prev]):
                    self.add(prev.start, atom.end, _fetch_operand([prev]))
                elif (
                    idx >= 2
                    and prev.kind == "group"
                    and elem[idx - 2].kind == "word"
                    and _CREATE_SCRIPTBLOCK.fullmatch(elem[idx - 2].text)
                    and _group_has_fetch(prev)
                ):
                    self.add(elem[idx - 2].start, atom.end, _fetch_operand([prev]))
            elif not lowered.startswith(".invoke"):
                for name in self.scriptblock_vars:
                    if lowered.startswith(f"{name}.invoke"):
                        self.report_variable(name)
        if (
            len(elem) > 2
            and elem[0].kind == "word"
            and elem[0].text.startswith("$")
            and elem[1].kind == "word"
            and elem[1].text == "="
            and _creates_fetched_scriptblock(elem[2:])
        ):
            self.scriptblock_vars[elem[0].text.lower()] = (
                elem[0].start,
                elem[-1].end,
                _fetch_operand(elem[2:]),
            )

    def report_variable(self, name: str) -> None:
        start, end, source = self.scriptblock_vars[name]
        self.add(start, end, source)


def _scan(text: str, base: int, depth: int) -> dict[tuple[int, int], str]:
    parser = _Parser(text)
    scan = _Scan(text, base, depth)
    scan.walk(parser.parse_block(None, 0))
    return scan.found


# -- unit selection -----------------------------------------------------------


def _units(content: str, file_type: str) -> list[tuple[int, str, bool]]:
    """Split ``content`` into independently tokenized (offset, text) units.

    PowerShell files are one unit.  In prose an apostrophe or backtick is not
    PowerShell syntax, so markdown prose and other file types are tokenized one
    line at a time, with fenced markdown blocks kept whole.
    """
    if file_type == "powershell":
        return [(0, content, False)]
    units: list[tuple[int, str, bool]] = []
    offset = 0
    fence: str | None = None
    fence_start = 0
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        marker = stripped[:3]
        is_fence = file_type == "markdown" and marker in ("```", "~~~")
        if fence is not None:
            if is_fence and marker == fence:
                units.append((fence_start, content[fence_start:offset], False))
                fence = None
        elif is_fence:
            fence = marker
            fence_start = offset + len(line)
        else:
            body = line.replace("`", " ") if file_type == "markdown" else line
            units.append((offset, body, True))
        offset += len(line)
    if fence is not None:
        units.append((fence_start, content[fence_start:], False))
    return units


def find_cradles(content: str, file_type: str) -> list[TextMatch]:
    """Return every PowerShell download-and-execute cradle in ``content``."""
    found: dict[tuple[int, int], str] = {}
    for offset, text, prose in _units(content, file_type):
        if not text.strip():
            continue
        if not prose:
            found.update(_scan(text, offset, 0))
            continue
        # A prose line ("Run: irm ... | iex") does not start with the command,
        # so tokenize from each place a cradle could begin.
        for keyword in list(_PROSE_START.finditer(text))[:_MAX_PROSE_STARTS]:
            found.update(_scan(text[keyword.start() :], offset + keyword.start(), 0))
    return [
        TextMatch(content, start, end, source) for (start, end), source in sorted(found.items())
    ]


def trusted_source_text(match: Any) -> str | None:
    """The fetch operand for a cradle match, or ``None`` for ordinary regex matches."""
    return getattr(match, "source", None)
