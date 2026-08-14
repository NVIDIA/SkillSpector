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
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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

    @property
    def is_parseable(self) -> bool:
        """Return whether this result contains a usable Python AST."""
        return self.tree is not None


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
) -> PythonAstCache:
    """Preparse eligible Python files within one scan's aggregate cache budget."""
    cache: PythonAstCache = {}
    source_characters = 0
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
        cache[path] = parse_python_source(content, path)
        source_characters += len(content)
    return cache


def prewarm_python_ast_cache(
    components: Iterable[str],
    file_cache: Mapping[str, str],
    *,
    max_source_chars: int = MAX_PYTHON_AST_SOURCE_CHARS,
    max_cache_source_chars: int = MAX_PYTHON_AST_CACHE_SOURCE_CHARS,
) -> str | None:
    """Preparse one scan's eligible Python files and return its runtime cache key."""
    cache = build_python_ast_cache(
        components,
        file_cache,
        max_source_chars=max_source_chars,
        max_cache_source_chars=max_cache_source_chars,
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


def clear_python_ast_cache(cache_key: str | None) -> None:
    """Release one scan's process-local parsed trees after its analyzer phase."""
    if cache_key is None:
        return
    with _runtime_ast_cache_lock:
        _runtime_ast_caches.pop(cache_key, None)
