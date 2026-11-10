# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown rule files use Markdown inspection without weakening executable checks."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.llm_analyzer_base import Batch, BatchExecutionResult
from skillspector.nested_artifacts import _virtual_type
from skillspector.nodes.analyzers import mcp_tool_poisoning, static_runner
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm
from skillspector.nodes.build_context import _infer_file_type as infer_context_type
from skillspector.python_ast import PythonSourceClassification, classify_python_source

_PATHS = ["rules/example.mdc", "rules/example.MDC"]
_BENIGN_PROSE = (
    "Use `research-cli` against the reader's reference.\n\n"
    + "The guide describes ordinary inputs and their displayed labels.\n" * 200
)


def _infer_nested_type(path: str) -> str:
    return _virtual_type(path, b"# Example rule\n", None, PythonSourceClassification.NON_PYTHON)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize(
    "infer",
    [infer_context_type, static_runner._infer_file_type, _infer_nested_type],
    ids=["build-context", "static-runner", "nested-artifact"],
)
def test_mdc_extension_is_markdown(path: str, infer: Callable[[str], str]) -> None:
    assert infer(path) == "markdown"


@pytest.mark.parametrize("path", ["rules/example.unknown", "rules/example"])
def test_unknown_extension_behavior_is_unchanged(path: str) -> None:
    assert infer_context_type(path) == "other"
    assert static_runner._infer_file_type(path) == "other"
    assert _infer_nested_type(path) == "text"
    assert (
        _virtual_type(path, b"\x00binary", None, PythonSourceClassification.NON_PYTHON) == "binary"
    )


@pytest.mark.parametrize("path", _PATHS)
def test_python_source_classification_precedes_mdc_extension(path: str) -> None:
    content = "#!/usr/bin/env python3\nprint('example')\n"
    classification = classify_python_source(path, content)
    assert classification is PythonSourceClassification.PYTHON
    assert infer_context_type(path, content) == "python"
    assert infer_context_type(path, source_classification=classification) == "python"
    assert _virtual_type(path, content.encode(), None, classification) == "python"
    assert static_runner._effective_module_file_type(path, tm, python_source=True) == "python"


def test_nested_container_classification_still_precedes_extension() -> None:
    assert (
        _virtual_type("container.mdc", b"example", "zip", PythonSourceClassification.PYTHON)
        == "zip"
    )


@pytest.mark.parametrize("path", _PATHS)
def test_mdc_inline_command_and_apostrophe_prose_completes(path: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: _BENIGN_PROSE}}, [tm]
    )
    assert result["findings"] == []
    assert len(result["inspection_ledger"]) == 1
    event = result["inspection_ledger"][0]
    assert event["path"] == path
    assert event["outcome"] is LedgerOutcome.COMPLETED
    assert "reason_code" not in event


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("content", ["rm -rf /\n", "```bash\nrm -rf /\n```\n"])
def test_mdc_destructive_command_remains_detected(path: str, content: str) -> None:
    # These strings are scanner input only; no command is executed.
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm]
    )
    assert any(
        finding.rule_id == "TM1" and finding.severity == "HIGH" for finding in result["findings"]
    )


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("content", ["$CMD -rf /\n", "```bash\n$CMD -rf /\n```\n"])
def test_mdc_unresolved_destructive_command_remains_partial(path: str, content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: content}}, [tm]
    )
    event = result["inspection_ledger"][0]
    assert event["path"] == path
    assert event["outcome"] is LedgerOutcome.PARTIAL
    assert event["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("path", _PATHS)
def test_mdc_inferred_metadata_selects_tp4_fence(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "# Example rule\n\n```python\nprint('example')\n```\n"
    batches_seen: list[Batch] = []

    async def inspect_batches(batches: list[Batch]) -> BatchExecutionResult:
        batches_seen.extend(batches)
        return BatchExecutionResult(
            successful=[
                (batch, [mcp_tool_poisoning._TP4AnalysisResult(is_mismatch=False)])
                for batch in batches
            ]
        )

    analyzer = SimpleNamespace(
        inference_usage=[], arun_batches_detailed=AsyncMock(side_effect=inspect_batches)
    )
    monkeypatch.setattr(mcp_tool_poisoning, "_TP4Analyzer", lambda *_args, **_kwargs: analyzer)
    monkeypatch.setattr(mcp_tool_poisoning, "get_max_input_tokens", lambda _model: 8192)
    monkeypatch.setattr(mcp_tool_poisoning, "estimate_tokens", lambda text: max(1, len(text) // 4))
    result = mcp_tool_poisoning._check_tp4(
        {
            "manifest": {"description": "Print a local example."},
            "components": [path],
            "file_cache": {path: content},
            "component_metadata": [{"path": path, "type": infer_context_type(path, content)}],
            "model_config": {"default": "test-model"},
        }
    )

    analyzer.arun_batches_detailed.assert_awaited_once()
    assert len(batches_seen) == 1
    batch = batches_seen[0]
    assert (batch.file_path, batch.start_line, batch.end_line) == (path, 4, 4)
    assert f"### {path} (python)" in batch.content
    assert "print('example')" in batch.content
    assert result.record is not None and result.record["ok"] is True
    assert result.findings == []
    assert any(event["outcome"] is LedgerOutcome.COMPLETED for event in result.ledger)
