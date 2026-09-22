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

"""Shared, per-scan Python AST parsing and import-alias metadata.

The graph prewarms this module's cache before its analyzer branches fan out.
Consumers must treat returned ASTs as read-only; keeping parsing, syntax-error
handling, and import aliases together lets later scope-aware resolution extend
one stable interface.
"""

from __future__ import annotations

import ast
import time
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

# Keep this in sync with the existing static-analyzer size gate.  It lives here
# so prewarming does not parse files that AST consumers will skip anyway.
MAX_PYTHON_AST_SOURCE_CHARS = 1_000_000
# AST nodes can be substantially larger than their source.  Limit the total
# source retained as parsed trees for any one scan; files beyond this budget
# use the existing on-demand behavior rather than retaining unbounded memory.
MAX_PYTHON_AST_CACHE_SOURCE_CHARS = 8_000_000


@dataclass(frozen=True, slots=True)
class ParsedPythonFile:
    """One Python source file's shared parse result and import aliases.

    ``tree`` is ``None`` when parsing failed.  The failed result is cached just
    like a successful one so every consumer can apply its own fallback policy
    without reparsing the same malformed source.
    """

    tree: ast.Module | None
    import_aliases: dict[str, str]
    lines: list[str]
    content: str
    parse_error: str | None = None
    # CPython AST columns are UTF-8 byte offsets. Retaining the encoded source
    # and per-line byte starts makes every node slice O(match size), rather than
    # repeatedly splitting and re-encoding the whole file for each finding.
    source_bytes: bytes = field(init=False, repr=False)
    line_byte_starts: tuple[int, ...] = field(init=False, repr=False)
    line_character_starts: tuple[int, ...] = field(init=False, repr=False)
    non_ascii_byte_ends: tuple[int, ...] = field(init=False, repr=False)
    non_ascii_extra_bytes: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source_bytes = self.content.encode("utf-8")
        line_byte_starts = [0]
        line_character_starts = [0]
        byte_offset = 0
        character_offset = 0
        for line in self.content.splitlines(keepends=True):
            byte_offset += len(line.encode("utf-8"))
            character_offset += len(line)
            line_byte_starts.append(byte_offset)
            line_character_starts.append(character_offset)
        non_ascii_byte_ends: list[int] = []
        non_ascii_extra_bytes = [0]
        if not source_bytes.isascii():
            byte_offset = 0
            extra_bytes = 0
            for character in self.content:
                width = 1 if character.isascii() else len(character.encode("utf-8"))
                byte_offset += width
                if width > 1:
                    extra_bytes += width - 1
                    non_ascii_byte_ends.append(byte_offset)
                    non_ascii_extra_bytes.append(extra_bytes)
        object.__setattr__(self, "source_bytes", source_bytes)
        object.__setattr__(self, "line_byte_starts", tuple(line_byte_starts))
        object.__setattr__(self, "line_character_starts", tuple(line_character_starts))
        object.__setattr__(self, "non_ascii_byte_ends", tuple(non_ascii_byte_ends))
        object.__setattr__(self, "non_ascii_extra_bytes", tuple(non_ascii_extra_bytes))

    @property
    def is_parseable(self) -> bool:
        """Return whether this result contains a usable Python AST."""
        return self.tree is not None

    def source_segment(self, node: ast.AST) -> str | None:
        """Return a node's exact source using precomputed UTF-8 byte offsets."""
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        col_offset = getattr(node, "col_offset", None)
        end_col_offset = getattr(node, "end_col_offset", None)
        if not all(
            isinstance(value, int) for value in (lineno, end_lineno, col_offset, end_col_offset)
        ):
            return None
        assert isinstance(lineno, int)
        assert isinstance(end_lineno, int)
        assert isinstance(col_offset, int)
        assert isinstance(end_col_offset, int)
        start_index = lineno - 1
        end_index = end_lineno - 1
        if start_index < 0 or end_index < start_index or end_index >= len(self.line_byte_starts):
            return None
        start = self.line_byte_starts[start_index] + col_offset
        end = self.line_byte_starts[end_index] + end_col_offset
        if start < 0 or end < start or end > len(self.source_bytes):
            return None
        try:
            return self.source_bytes[start:end].decode("utf-8")
        except UnicodeDecodeError:
            return None

    def character_column(self, lineno: int, byte_column: int) -> int | None:
        """Convert one CPython UTF-8 byte column to a public character column."""
        line_index = lineno - 1
        if line_index < 0 or line_index >= len(self.line_byte_starts):
            return None
        absolute_byte = self.line_byte_starts[line_index] + byte_column
        if absolute_byte < 0 or absolute_byte > len(self.source_bytes):
            return None
        non_ascii_count = bisect_right(self.non_ascii_byte_ends, absolute_byte)
        absolute_character = absolute_byte - self.non_ascii_extra_bytes[non_ascii_count]
        return absolute_character - self.line_character_starts[line_index]


PythonAstCache = dict[str, ParsedPythonFile]


@dataclass(slots=True)
class _RuntimePythonAstCache:
    """Per-scan LRU of parsed files with an aggregate source-size budget."""

    entries: OrderedDict[str, ParsedPythonFile]
    source_characters: int = 0


# AST nodes are intentionally kept outside LangGraph state: ``ast.Module`` is
# not checkpoint-serializable.  State carries a UUID cache key, while this
# process-local registry keeps one scan's parsed trees available to all of its
# parallel analyzer branches.  Completed scans release their entry in report.
_MAX_RUNTIME_AST_CACHES = 32
_runtime_ast_caches: OrderedDict[str, _RuntimePythonAstCache] = OrderedDict()
_runtime_ast_cache_lock = RLock()


def _remember_runtime_ast_cache(cache_key: str, cache: _RuntimePythonAstCache) -> None:
    """Store a cache under the lock and bound abandoned scan entries."""
    _runtime_ast_caches[cache_key] = cache
    _runtime_ast_caches.move_to_end(cache_key)
    while len(_runtime_ast_caches) > _MAX_RUNTIME_AST_CACHES:
        _runtime_ast_caches.popitem(last=False)


def _cache_runtime_entry(
    cache: _RuntimePythonAstCache, filename: str, parsed: ParsedPythonFile
) -> None:
    """Store one parsed source, evicting least-recent entries to stay bounded."""
    old = cache.entries.pop(filename, None)
    if old is not None:
        cache.source_characters -= len(old.content)

    source_characters = len(parsed.content)
    if source_characters > MAX_PYTHON_AST_CACHE_SOURCE_CHARS:
        return
    while (
        cache.entries
        and cache.source_characters + source_characters > MAX_PYTHON_AST_CACHE_SOURCE_CHARS
    ):
        _, evicted = cache.entries.popitem(last=False)
        cache.source_characters -= len(evicted.content)
    if cache.source_characters + source_characters <= MAX_PYTHON_AST_CACHE_SOURCE_CHARS:
        cache.entries[filename] = parsed
        cache.source_characters += source_characters


def build_import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map locally bound names to their fully-qualified import paths.

    ``from pathlib import Path`` becomes ``{"Path": "pathlib.Path"}``, while
    ``import pathlib as pl`` becomes ``{"pl": "pathlib"}``.
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


def parse_python_source(content: str, filename: str) -> ParsedPythonFile:
    """Parse *content* once and retain its aliases or a structured parse failure."""
    lines = content.splitlines()
    try:
        tree = ast.parse(content, filename=filename)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return ParsedPythonFile(
            tree=None,
            import_aliases={},
            lines=lines,
            content=content,
            parse_error=type(exc).__name__,
        )
    return ParsedPythonFile(
        tree=tree,
        import_aliases=build_import_aliases(tree),
        lines=lines,
        content=content,
    )


def build_python_ast_cache(
    components: Iterable[str],
    file_cache: Mapping[str, str],
    *,
    max_source_chars: int = MAX_PYTHON_AST_SOURCE_CHARS,
    max_cache_source_chars: int = MAX_PYTHON_AST_CACHE_SOURCE_CHARS,
    clock: Callable[[], float] = time.monotonic,
    started_at: float | None = None,
    deadline: float | None = None,
    runtime_limitations: list[tuple[str, float]] | None = None,
) -> PythonAstCache:
    """Preparse eligible Python files within one scan's aggregate cache budget."""
    cache: PythonAstCache = {}
    source_characters = 0
    effective_started_at = clock() if started_at is None else started_at

    def _expired(path: str) -> bool:
        if deadline is None:
            return False
        now = clock()
        if now < deadline:
            return False
        if runtime_limitations is not None and not runtime_limitations:
            runtime_limitations.append((path, max(0.0, now - effective_started_at)))
        return True

    for path in components:
        if not path.lower().endswith(".py"):
            continue
        content = file_cache.get(path)
        if (
            content is None
            or len(content) > max_source_chars
            or source_characters + len(content) > max_cache_source_chars
        ):
            continue
        if _expired(path):
            break
        cache[path] = parse_python_source(content, path)
        source_characters += len(content)
        if _expired(path):
            break
    return cache


def prewarm_python_ast_cache(
    components: Iterable[str],
    file_cache: Mapping[str, str],
    *,
    max_source_chars: int = MAX_PYTHON_AST_SOURCE_CHARS,
    max_cache_source_chars: int = MAX_PYTHON_AST_CACHE_SOURCE_CHARS,
    clock: Callable[[], float] = time.monotonic,
    started_at: float | None = None,
    deadline: float | None = None,
    runtime_limitations: list[tuple[str, float]] | None = None,
) -> str | None:
    """Preparse one scan's eligible Python files and return its runtime cache key."""
    cache = build_python_ast_cache(
        components,
        file_cache,
        max_source_chars=max_source_chars,
        max_cache_source_chars=max_cache_source_chars,
        clock=clock,
        started_at=started_at,
        deadline=deadline,
        runtime_limitations=runtime_limitations,
    )
    if not cache:
        return None

    cache_key = uuid4().hex
    with _runtime_ast_cache_lock:
        _remember_runtime_ast_cache(
            cache_key,
            _RuntimePythonAstCache(
                entries=OrderedDict(cache.items()),
                source_characters=sum(len(parsed.content) for parsed in cache.values()),
            ),
        )
    return cache_key


def get_python_ast(cache_key: str | None, content: str, filename: str) -> ParsedPythonFile:
    """Return a scan's prewarmed result, or parse for standalone analyzer use.

    If a checkpoint resumes in a new process, the cache key has no registry
    entry.  The lock recreates and fills it once per source before parallel
    analyzer branches can observe it.
    """
    if cache_key is None:
        return parse_python_source(content, filename)

    with _runtime_ast_cache_lock:
        cache = _runtime_ast_caches.get(cache_key)
        if cache is None:
            cache = _RuntimePythonAstCache(entries=OrderedDict())
            _remember_runtime_ast_cache(cache_key, cache)
        else:
            _runtime_ast_caches.move_to_end(cache_key)
        cached = cache.entries.get(filename)
        if cached is not None and cached.content == content:
            cache.entries.move_to_end(filename)
            return cached
        parsed = parse_python_source(content, filename)
        _cache_runtime_entry(cache, filename, parsed)
        return parsed


def peek_python_ast(cache_key: str | None, content: str, filename: str) -> ParsedPythonFile | None:
    """Return the scan's cached parse for *filename* without parsing anything.

    Only an exact content match hits.  Anything else (a windowed view fragment,
    an uncached file, an unknown cache key) returns ``None`` so the caller can
    apply its own fallback without disturbing the shared cache: unlike
    :func:`get_python_ast`, this never parses and never stores.
    """
    if cache_key is None:
        return None
    with _runtime_ast_cache_lock:
        cache = _runtime_ast_caches.get(cache_key)
        if cache is None:
            return None
        cached = cache.entries.get(filename)
        if cached is not None and cached.content == content:
            cache.entries.move_to_end(filename)
            return cached
        return None


def clear_python_ast_cache(cache_key: str | None) -> None:
    """Release one scan's process-local parsed trees after its analyzer phase."""
    if cache_key is None:
        return
    with _runtime_ast_cache_lock:
        _runtime_ast_caches.pop(cache_key, None)
