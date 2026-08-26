# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned dependency and runtime-boundary contracts for Bash CST parsing."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import os
import subprocess
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

TREE_SITTER_VERSION = "0.25.2"
TREE_SITTER_BASH_VERSION = "0.25.1"
BASH_ABI_VERSION = 15
BASH_SEMANTIC_VERSION = (0, 25, 1)


def _frontend() -> Any:
    return importlib.import_module("skillspector.shell_frontend")


def _language() -> Any:
    tree_sitter = importlib.import_module("tree_sitter")
    tree_sitter_bash = importlib.import_module("tree_sitter_bash")
    return tree_sitter.Language(tree_sitter_bash.language())


def _parse(source: bytes) -> Any:
    tree_sitter = importlib.import_module("tree_sitter")
    tree = tree_sitter.Parser(_language()).parse(source)
    assert tree is not None
    return tree


def _descendants(node: Any) -> list[Any]:
    pending = [node]
    result = []
    while pending:
        current = pending.pop()
        result.append(current)
        pending.extend(reversed(current.children))
    return result


def test_installed_parser_distribution_versions_are_exact() -> None:
    assert importlib.metadata.version("tree-sitter") == TREE_SITTER_VERSION
    assert importlib.metadata.version("tree-sitter-bash") == TREE_SITTER_BASH_VERSION


def test_sc10_shell_path_has_no_raw_fallback_parser_surface() -> None:
    frontend = _frontend()
    adapters = importlib.import_module("skillspector.dependency_command_adapters")
    executable_sources = {
        "shell_frontend": inspect.getsource(frontend),
        "dependency_command_adapters": inspect.getsource(adapters),
    }

    for module_name, source in executable_sources.items():
        tree = ast.parse(source)
        imports = {
            alias.name.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        attribute_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert "shlex" not in imports, module_name
        assert "re" not in imports, f"{module_name} must not create regex shell evidence"
        assert "splitlines" not in attribute_calls, module_name
        assert not any("tokenizer" in name or "lexer" in name for name in function_names)
        assert not any(
            "raw_command" in name or "fallback_command" in name for name in function_names
        )

    recovery_names = {
        name
        for name, _value in inspect.getmembers(frontend._ShellLowerer)
        if "recover" in name or "recovery" in name or name.startswith("_fallback_")
    }
    assert recovery_names == {
        "_recovery_step",
        "_recover_cst_anchored_body_argument",
        "_recover_cst_anchored_heredoc_facts",
    }

    parse_tree = ast.parse(inspect.getsource(frontend.parse_bash_source))
    parser_calls = [
        node
        for node in ast.walk(parse_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse"
    ]
    assert len(parser_calls) == 1
    assert [getattr(argument, "id", None) for argument in parser_calls[0].args] == ["reader"]
    assert parser_calls[0].keywords == []


def test_bash_language_versions_and_minimal_public_api_parse() -> None:
    language = _language()
    assert language.abi_version == BASH_ABI_VERSION
    assert language.semantic_version == BASH_SEMANTIC_VERSION

    tree_sitter = importlib.import_module("tree_sitter")
    tree = tree_sitter.Parser(language).parse(b"printf ok\n")

    assert tree is not None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_bash_cst_command_and_file_redirect_fields_are_stable() -> None:
    tree = _parse(b"1>out printf ok\n")
    command = next(node for node in _descendants(tree.root_node) if node.type == "command")

    assert command.child_by_field_name("name").type == "command_name"
    assert command.child_by_field_name("argument").type == "word"
    redirect = command.child_by_field_name("redirect")
    assert redirect.type == "file_redirect"
    assert redirect.child_by_field_name("descriptor").type == "file_descriptor"
    assert redirect.child_by_field_name("destination").type == "word"


def test_bash_cst_heredoc_redirect_and_body_are_stable() -> None:
    tree = _parse(b"cat <<'EOF'\nvalue\nEOF\n")
    node_types = [node.type for node in _descendants(tree.root_node)]

    assert "heredoc_redirect" in node_types
    assert "heredoc_body" in node_types


def test_production_language_is_cached_but_each_parser_is_fresh() -> None:
    frontend = _frontend()

    assert frontend.load_bash_language() is frontend.load_bash_language()
    assert frontend.create_bash_parser() is not frontend.create_bash_parser()


@pytest.mark.parametrize(
    ("failure_mode", "abi_version", "semantic_version"),
    [
        ("import", BASH_ABI_VERSION, BASH_SEMANTIC_VERSION),
        ("abi", BASH_ABI_VERSION + 1, BASH_SEMANTIC_VERSION),
        ("semantic", BASH_ABI_VERSION, (0, 25, 0)),
    ],
)
def test_language_loader_failures_are_classified_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    abi_version: int,
    semantic_version: tuple[int, int, int],
) -> None:
    frontend = _frontend()
    language = SimpleNamespace(
        abi_version=abi_version,
        semantic_version=semantic_version,
    )
    modules = {
        "tree_sitter": SimpleNamespace(Language=lambda _capsule: language),
        "tree_sitter_bash": SimpleNamespace(language=lambda: object()),
    }

    def import_dependency(name: str) -> object:
        if failure_mode == "import":
            raise ModuleNotFoundError("private dependency detail")
        return modules[name]

    frontend.load_bash_language.cache_clear()
    monkeypatch.setattr(frontend, "import_module", import_dependency)
    try:
        with pytest.raises(frontend.ShellParserError) as caught:
            frontend.load_bash_language()
    finally:
        frontend.load_bash_language.cache_clear()

    assert caught.value.outcome is frontend.ShellParserOutcome.FAILED
    assert caught.value.reason is frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE
    assert caught.value.deadline_tripped is False
    assert "private dependency detail" not in str(caught.value)


def test_production_parser_initialization_failure_is_classified_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _frontend()

    class BrokenParser:
        def __init__(self, _language: object) -> None:
            raise RuntimeError("private parser detail")

    monkeypatch.setattr(frontend, "load_bash_language", lambda: object())
    monkeypatch.setattr(
        frontend,
        "import_module",
        lambda _name: SimpleNamespace(Parser=BrokenParser),
    )

    with pytest.raises(frontend.ShellParserError) as caught:
        frontend.create_bash_parser()

    assert caught.value.outcome is frontend.ShellParserOutcome.FAILED
    assert caught.value.reason is frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE
    assert caught.value.deadline_tripped is False
    assert "private parser detail" not in str(caught.value)


def test_production_parse_uses_only_a_bounded_callable_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _frontend()
    source = b"printf ok\n" * 1000
    sentinel_tree = object()
    observed_chunks: list[bytes] = []
    observed_deadlines: list[float | None] = []
    observed_parse_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class RecordingParser:
        def parse(self, *args: object, **kwargs: object) -> object:
            observed_parse_calls.append((args, kwargs))
            assert kwargs == {}, "production must not pass progress_callback"
            assert len(args) == 1
            reader = args[0]
            assert callable(reader)
            assert not isinstance(reader, bytes), "bytestring parser overload is prohibited"
            first = reader(0, (0, 0))
            observed_chunks.append(first)
            assert reader(len(source), (1000, 0)) == b""
            return sentinel_tree

    def create_parser(*, deadline_monotonic: float | None) -> RecordingParser:
        observed_deadlines.append(deadline_monotonic)
        return RecordingParser()

    monkeypatch.setattr(frontend, "create_bash_parser", create_parser)

    result = frontend.parse_bash_source(source)

    assert result is sentinel_tree
    assert observed_deadlines == [None]
    assert len(observed_parse_calls) == 1
    assert observed_chunks
    assert len(observed_chunks[0]) == frontend.MAX_TREE_SITTER_READ_BYTES
    assert observed_chunks[0] == source[: frontend.MAX_TREE_SITTER_READ_BYTES]


def test_native_timeout_is_derived_after_language_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _frontend()
    observed_timeouts: list[int] = []
    language_loaded = False
    sentinel_language = object()

    class Parser:
        def __init__(self, language: object, *, timeout_micros: int) -> None:
            assert language is sentinel_language
            observed_timeouts.append(timeout_micros)

    class TreeSitterModule:
        pass

    TreeSitterModule.Parser = Parser

    def load_language() -> object:
        nonlocal language_loaded
        language_loaded = True
        return sentinel_language

    def current_time() -> float:
        assert language_loaded
        return 100.0

    monkeypatch.setattr(frontend, "load_bash_language", load_language)
    monkeypatch.setattr(frontend, "import_module", lambda _name: TreeSitterModule)
    monkeypatch.setattr(frontend, "monotonic", current_time)

    parser = frontend.create_bash_parser(deadline_monotonic=101.0)

    assert isinstance(parser, Parser)
    assert len(observed_timeouts) == 1
    assert observed_timeouts[0] == 1_000_000


@pytest.mark.parametrize("meaningful_work", [False, True], ids=["first-unit", "later-unit"])
def test_reader_deadline_is_classified_as_local_runtime_partial(
    monkeypatch: pytest.MonkeyPatch,
    meaningful_work: bool,
) -> None:
    frontend = _frontend()

    class CancelledParser:
        def parse(
            self,
            reader: Callable[[int, tuple[int, int]], bytes],
        ) -> None:
            assert callable(reader)
            assert reader(0, (0, 0)) == b""
            return None

    monkeypatch.setattr(
        frontend,
        "create_bash_parser",
        lambda *, deadline_monotonic: CancelledParser(),
    )

    with pytest.raises(frontend.ShellParserError) as caught:
        frontend.parse_bash_source(
            b"printf ok\n",
            deadline_monotonic=time.monotonic() - 1.0,
            meaningful_work=meaningful_work,
        )

    assert caught.value.outcome is frontend.ShellParserOutcome.PARTIAL
    assert caught.value.reason is frontend.ShellParserFailureReason.RUNTIME_LIMIT
    assert caught.value.deadline_tripped is True


@pytest.mark.timeout(10)
def test_public_shell_analysis_propagates_the_production_deadline() -> None:
    frontend = _frontend()
    contracts = importlib.import_module("skillspector.dependency_source_types")
    raw = b"printf ok\n"
    unit = contracts.ShellUnit(
        dialect=contracts.ShellDialect.BASH,
        kind=contracts.ShellUnitKind.STANDALONE,
        provenance=contracts.SiteProvenance.FILE_SUFFIX,
        raw_bytes=raw,
        origin_span=contracts.SourceSpan(
            "scripts/setup.sh",
            0,
            len(raw),
            1,
            1,
            start_column=0,
            end_column=len(raw),
        ),
    )

    result = frontend.analyze_shell_unit(
        unit,
        budget=contracts.DependencyWorkBudget(),
        deadline_monotonic=time.monotonic() - 1.0,
    )

    assert [issue.reason for issue in result.issues] == [contracts.ShellIssueReason.RUNTIME_LIMIT]
    assert [item.outcome for item in result.work_items] == [contracts.ShellWorkOutcome.PARTIAL]


def test_native_parser_timeout_is_classified_only_after_deadline_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _frontend()

    class TimedOutParser:
        def parse(self, reader: Callable[[int, tuple[int, int]], bytes]) -> None:
            assert callable(reader)
            raise ValueError("Parsing failed")

    monkeypatch.setattr(
        frontend,
        "create_bash_parser",
        lambda *, deadline_monotonic: TimedOutParser(),
    )
    monkeypatch.setattr(frontend, "monotonic", lambda: 101.0)

    with pytest.raises(frontend.ShellParserError) as caught:
        frontend.parse_bash_source(b"printf ok\n", deadline_monotonic=100.0)

    assert caught.value.outcome is frontend.ShellParserOutcome.PARTIAL
    assert caught.value.reason is frontend.ShellParserFailureReason.RUNTIME_LIMIT
    assert caught.value.deadline_tripped is True


def test_parser_initialization_failure_before_work_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _frontend()

    def fail_initialization(*, deadline_monotonic: float | None) -> Any:
        raise RuntimeError("private parser detail")

    monkeypatch.setattr(frontend, "create_bash_parser", fail_initialization)

    with pytest.raises(frontend.ShellParserError) as caught:
        frontend.parse_bash_source(b"printf ok\n")

    assert caught.value.outcome is frontend.ShellParserOutcome.FAILED
    assert caught.value.reason is frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE
    assert caught.value.deadline_tripped is False
    assert "private parser detail" not in str(caught.value)


@pytest.mark.parametrize(
    ("meaningful_work", "expected_outcome"),
    [(False, "FAILED"), (True, "PARTIAL")],
    ids=["before-meaningful-work", "after-meaningful-work"],
)
def test_internal_parser_failure_classification_depends_on_meaningful_work(
    monkeypatch: pytest.MonkeyPatch,
    meaningful_work: bool,
    expected_outcome: str,
) -> None:
    frontend = _frontend()

    class BrokenParser:
        def parse(
            self,
            reader: Callable[[int, tuple[int, int]], bytes],
        ) -> None:
            assert callable(reader)
            raise RuntimeError("private parser detail")

    monkeypatch.setattr(
        frontend,
        "create_bash_parser",
        lambda *, deadline_monotonic: BrokenParser(),
    )

    with pytest.raises(frontend.ShellParserError) as caught:
        frontend.parse_bash_source(b"printf ok\n", meaningful_work=meaningful_work)

    assert caught.value.outcome is getattr(frontend.ShellParserOutcome, expected_outcome)
    assert caught.value.reason is frontend.ShellParserFailureReason.SHELL_PARSER_UNAVAILABLE
    assert caught.value.deadline_tripped is False
    assert "private parser detail" not in str(caught.value)


def test_production_never_passes_the_unsafe_parser_progress_callback() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from skillspector.shell_frontend import parse_bash_source; "
                "tree = parse_bash_source(b'printf ok\\n' * 90000); "
                "assert tree.root_node.type == 'program'"
            ),
        ],
        check=False,
        capture_output=True,
        env=os.environ.copy(),
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def test_parser_failure_is_not_retried_or_delegated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _frontend()
    constructor_calls = 0
    parser_pids: list[int] = []

    class BrokenParser:
        def parse(self, reader: Callable[[int, tuple[int, int]], bytes]) -> None:
            assert callable(reader)
            parser_pids.append(os.getpid())
            raise RuntimeError("private parser detail")

    def create_parser(*, deadline_monotonic: float | None) -> BrokenParser:
        nonlocal constructor_calls
        constructor_calls += 1
        return BrokenParser()

    monkeypatch.setattr(frontend, "create_bash_parser", create_parser)

    with pytest.raises(frontend.ShellParserError):
        frontend.parse_bash_source(b"printf ok\n")

    assert constructor_calls == 1
    assert parser_pids == [os.getpid()]
