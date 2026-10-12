# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer-verified string and comment ownership for complete Python modules.

Static analyzers read Python source as text. In a module that the Python parser
accepts, every quote character is a string delimiter, string content, or
comment text, and the tokenizer says which. Lexical scanners can borrow that
ownership instead of guessing quote pairs. Fragments, malformed source, and
polyglots prove nothing and keep each caller's conservative path.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import warnings
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock

from skillspector.python_ast import MAX_PYTHON_AST_SOURCE_CHARS

# The tokenizer and the compiler disagree about a carriage return that does
# not start CRLF, so such source never acquires token ownership.
_LONE_CARRIAGE_RETURN_RE = re.compile(r"\r(?!\n)")
# A shebang that names another interpreter (for example a shell polyglot)
# makes the Python host ambiguous. Without a shebang, ``.py`` source is Python.
_PYTHON_SHEBANG_RE = re.compile(
    r"#![ \t]*+(?:\S*/)?(?:env[ \t]++(?:-\S*[ \t]++)*(?:\S*/)?)?"
    r"(?:python[0-9.]*|pypy[0-9.]*|uv)(?=[ \t\r\n]|$)"
)
# ``warnings.catch_warnings`` swaps the process-global filter list, and analyzer
# nodes run on concurrent graph worker threads. Two such blocks that exit out of
# order can leave their ``ignore`` filter installed for the whole process, so
# the ownership parse serializes its block.
_PYTHON_OWNERSHIP_PARSE_LOCK = Lock()
_PYTHON_TEMPLATE_STRING_STARTS = frozenset(
    {tokenize.FSTRING_START, getattr(tokenize, "TSTRING_START", tokenize.FSTRING_START)}
)
_PYTHON_TEMPLATE_STRING_ENDS = frozenset(
    {tokenize.FSTRING_END, getattr(tokenize, "TSTRING_END", tokenize.FSTRING_END)}
)

# Start offsets and matching end offsets of a module's outermost string and
# comment tokens, in source order. Results are shared, so they are immutable.
PythonLiteralSpans = tuple[tuple[int, ...], tuple[int, ...]]

# Three consumers ask for a module's spans during one scan: the declared-marker
# pass proves string closers, the tool-misuse shell parser proves token
# ownership, and AR2 proves comment ownership. Recent results are remembered so
# they share one parse. Entries are keyed by length and hash and confirmed by
# string equality, which is an identity check when the callers hold the same
# file-cache string.
#
# The consumers run in different analyzer nodes (the marker pass in every
# static-pattern node). The graph starts all analyzer nodes together on worker
# threads; the CLI and MCP server set no ``max_concurrency``, so the pool has
# Python's default size, min(32, CPUs + 4). Every node walks the components in
# the same order but at its own pace, so requests for other modules arrive
# between a module's first and last consumer. An entry holds one module however
# many consumers ask, so eight entries keep a result while up to seven other
# modules are requested in that gap. A wider gap only repeats the parse.
# Oversized source is never stored, so eight entries hold at most eight times
# MAX_PYTHON_AST_SOURCE_CHARS, which equals one scan's AST cache budget
# (MAX_PYTHON_AST_CACHE_SOURCE_CHARS), and eviction bounds how long that text
# outlives its scan. This lock guards only the entries, so a lookup never waits
# for another thread's parse.
_PYTHON_LITERAL_SPANS_CACHE_SIZE = 8
_PYTHON_LITERAL_SPANS_CACHE_LOCK = Lock()
_PYTHON_LITERAL_SPANS_CACHE: OrderedDict[tuple[int, int], tuple[str, PythonLiteralSpans | None]] = (
    OrderedDict()
)


def python_literal_spans(
    content: str,
    check_runtime: Callable[[], None],
) -> PythonLiteralSpans | None:
    """Return outermost string and comment token spans of valid Python source.

    Return ``None`` unless the whole text is accepted by the Python parser,
    so a fragment, malformed file, or shell-shebang polyglot cannot borrow
    host ownership. Every span is checked against the exact source text.
    Results, including ``None``, are memoized for the most recent sources.
    """
    if len(content) > MAX_PYTHON_AST_SOURCE_CHARS:
        return None
    key = (len(content), hash(content))
    with _PYTHON_LITERAL_SPANS_CACHE_LOCK:
        entry = _PYTHON_LITERAL_SPANS_CACHE.get(key)
        if entry is not None and entry[0] == content:
            _PYTHON_LITERAL_SPANS_CACHE.move_to_end(key)
        else:
            entry = None
    if entry is not None:
        # A shared result skips the parse, not the caller's runtime bound.
        check_runtime()
        return entry[1]
    # A runtime-limit exception propagates from here, so a proof that the
    # bound interrupted is never stored.
    spans = _python_literal_spans_uncached(content, check_runtime)
    with _PYTHON_LITERAL_SPANS_CACHE_LOCK:
        _PYTHON_LITERAL_SPANS_CACHE[key] = (content, spans)
        _PYTHON_LITERAL_SPANS_CACHE.move_to_end(key)
        while len(_PYTHON_LITERAL_SPANS_CACHE) > _PYTHON_LITERAL_SPANS_CACHE_SIZE:
            _PYTHON_LITERAL_SPANS_CACHE.popitem(last=False)
    return spans


def _python_literal_spans_uncached(
    content: str,
    check_runtime: Callable[[], None],
) -> PythonLiteralSpans | None:
    """Parse and tokenize ``content`` for :func:`python_literal_spans`."""
    # A leading U+FEFF byte-order mark fails the module parse below and keeps
    # every conservative bound. The file cache decodes with ``utf-8``, not
    # ``utf-8-sig``, so the AST analyzers already report such a file as a
    # syntax error; stripping the mark belongs with that decoding fix.
    if _LONE_CARRIAGE_RETURN_RE.search(content) is not None or (
        content.startswith("#!") and _PYTHON_SHEBANG_RE.match(content) is None
    ):
        return None
    check_runtime()
    try:
        # The lenient tokenizer accepts bytes such as ``$`` and a backtick as
        # operators. Requiring a module parse proves that every byte outside
        # the spans below is Python syntax, never a shell quote or expansion.
        # The scanned module's own parse already reports its compiler warnings
        # (for example an invalid ``"\d"`` escape). Repeating them here would
        # duplicate that output, and ``-W error`` would turn them into a
        # ``SyntaxError`` that silently drops ownership.
        with _PYTHON_OWNERSHIP_PARSE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return None
    check_runtime()
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer("\n", content))
    starts: list[int] = []
    ends: list[int] = []
    template_depth = 0
    template_start = 0

    def offset(position: tuple[int, int]) -> int:
        return line_starts[position[0] - 1] + position[1]

    try:
        for index, token in enumerate(tokenize.generate_tokens(io.StringIO(content).readline)):
            if index % 256 == 0:
                check_runtime()
            if token.type in _PYTHON_TEMPLATE_STRING_STARTS:
                if template_depth == 0:
                    template_start = offset(token.start)
                    if not content.startswith(token.string, template_start):
                        return None
                template_depth += 1
            elif token.type in _PYTHON_TEMPLATE_STRING_ENDS:
                template_depth -= 1
                end = offset(token.end)
                if template_depth < 0 or not content.endswith(token.string, 0, end):
                    return None
                if template_depth == 0:
                    starts.append(template_start)
                    ends.append(end)
            elif not template_depth and token.type in (tokenize.STRING, tokenize.COMMENT):
                # Inside a replacement field, nested strings and comments
                # belong to the enclosing f-string or t-string span.
                start, end = offset(token.start), offset(token.end)
                if content[start:end] != token.string:
                    return None
                starts.append(start)
                ends.append(end)
    except (tokenize.TokenError, SyntaxError, ValueError, IndexError):
        return None
    if template_depth or any(ends[index] > starts[index + 1] for index in range(len(starts) - 1)):
        return None
    return tuple(starts), tuple(ends)


class PythonStringClosers:
    """Lazily proven closing delimiters of one complete module's string literals.

    :meth:`closes_string` reports whether a source offset lies in the closing
    delimiter of an outermost string, f-string, or t-string token. Such a quote
    is Python syntax that ends a literal; it never opens text. The module is
    parsed and tokenized at most once, on the first query, so a caller that
    never asks pays nothing. Source that :func:`python_literal_spans` cannot
    prove answers ``False`` everywhere, which keeps the caller's lexical path.
    """

    def __init__(self, content: str, check_runtime: Callable[[], None]) -> None:
        self._content = content
        self._check_runtime = check_runtime
        self._spans: PythonLiteralSpans | None = None
        self._spans_computed = False

    def closes_string(self, offset: int) -> bool:
        """Return whether ``offset`` is in a proven string literal's closing delimiter."""
        if not self._spans_computed:
            self._spans = python_literal_spans(self._content, self._check_runtime)
            self._spans_computed = True
        if self._spans is None:
            return False
        starts, ends = self._spans
        index = bisect_right(starts, offset) - 1
        if index < 0 or offset >= ends[index]:
            return False
        content = self._content
        start, end = starts[index], ends[index]
        if content[start] == "#":
            return False
        # Skip a string prefix such as ``rb``, ``f`` or ``t`` to the opening
        # quote. The closing delimiter repeats it, three times when tripled.
        opener = start
        while content[opener] not in "'\"":
            opener += 1
        width = 3 if content.startswith(content[opener] * 3, opener, end) else 1
        return offset >= end - width
