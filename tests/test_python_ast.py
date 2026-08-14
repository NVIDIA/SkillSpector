# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared Python AST parsing utility."""

from __future__ import annotations

import skillspector.python_ast as python_ast
from skillspector.python_ast import (
    build_python_ast_cache,
    clear_python_ast_cache,
    get_python_ast,
    parse_python_source,
    prewarm_python_ast_cache,
)


def test_parse_python_source_exposes_import_aliases() -> None:
    parsed = parse_python_source(
        "import os as operating_system\nfrom subprocess import run\n", "script.py"
    )

    assert parsed.is_parseable
    assert parsed.tree is not None
    assert parsed.import_aliases == {
        "operating_system": "os",
        "run": "subprocess.run",
    }
    assert parsed.lines == ["import os as operating_system", "from subprocess import run"]
    assert parsed.content == "import os as operating_system\nfrom subprocess import run\n"


def test_parse_python_source_retains_syntax_error_result() -> None:
    parsed = parse_python_source("def broken(\n", "broken.py")

    assert not parsed.is_parseable
    assert parsed.tree is None
    assert parsed.import_aliases == {}
    assert parsed.parse_error == "SyntaxError"


def test_parse_python_source_retains_value_error_result(monkeypatch) -> None:
    def raise_value_error(*args, **kwargs):
        raise ValueError("invalid source")

    monkeypatch.setattr(python_ast.ast, "parse", raise_value_error)

    parsed = parse_python_source("x = 1\n", "broken.py")

    assert not parsed.is_parseable
    assert parsed.parse_error == "ValueError"


def test_parse_python_source_retains_recursion_error_result(monkeypatch) -> None:
    def raise_recursion_error(*args, **kwargs):
        raise RecursionError("expression is too deep")

    monkeypatch.setattr(python_ast.ast, "parse", raise_recursion_error)

    parsed = parse_python_source("x = 1\n", "deep.py")

    assert not parsed.is_parseable
    assert parsed.parse_error == "RecursionError"


def test_build_python_ast_cache_caches_failures_and_skips_oversized_files() -> None:
    cache = build_python_ast_cache(
        ["valid.py", "uppercase.PY", "broken.py", "oversized.py", "readme.md"],
        {
            "valid.py": "x = 1\n",
            "uppercase.PY": "x = 2\n",
            "broken.py": "def bad(\n",
            "oversized.py": "x" * 11,
            "readme.md": "not Python",
        },
        max_source_chars=10,
    )

    assert set(cache) == {"valid.py", "uppercase.PY", "broken.py"}
    assert cache["valid.py"].is_parseable
    assert cache["uppercase.PY"].is_parseable
    assert not cache["broken.py"].is_parseable


def test_build_python_ast_cache_respects_aggregate_source_budget() -> None:
    cache = build_python_ast_cache(
        ["first.py", "second.py"],
        {
            "first.py": "x = 1\n",
            "second.py": "x = 2\n",
        },
        max_cache_source_chars=8,
    )

    assert set(cache) == {"first.py"}


def test_get_python_ast_reparses_when_cached_source_changes() -> None:
    cache_key = prewarm_python_ast_cache(["script.py"], {"script.py": "import os as old_name\n"})
    assert cache_key is not None

    parsed = get_python_ast(cache_key, "import os as new_name\n", "script.py")

    assert parsed.import_aliases == {"new_name": "os"}
    clear_python_ast_cache(cache_key)


def test_runtime_ast_cache_registry_is_bounded_for_checkpoint_cache_misses() -> None:
    cache_keys = [
        f"resumed-scan-{index}" for index in range(python_ast._MAX_RUNTIME_AST_CACHES + 5)
    ]

    for cache_key in cache_keys:
        get_python_ast(cache_key, "x = 1\n", "script.py")

    assert len(python_ast._runtime_ast_caches) <= python_ast._MAX_RUNTIME_AST_CACHES

    for cache_key in cache_keys:
        clear_python_ast_cache(cache_key)
