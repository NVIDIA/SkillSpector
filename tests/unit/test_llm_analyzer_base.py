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

"""Tests for LLMAnalyzerBase progress output."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from skillspector.llm_analyzer_base import Batch, LLMAnalysisResult, LLMAnalyzerBase


def _make_analyzer(analyzer_id: str = "test-analyzer") -> LLMAnalyzerBase:
    """Create an LLMAnalyzerBase with mocked LLM dependencies."""
    with patch("skillspector.llm_analyzer_base.get_chat_model") as mock_get:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_get.return_value = mock_llm
        with patch("skillspector.llm_analyzer_base.get_max_input_tokens", return_value=100_000):
            return LLMAnalyzerBase(
                base_prompt="analyze this", model="test-model", analyzer_id=analyzer_id
            )


def test_analyzer_id_stored() -> None:
    """LLMAnalyzerBase stores the analyzer_id passed to __init__."""
    analyzer = _make_analyzer("my-id")
    assert analyzer.analyzer_id == "my-id"


def test_analyzer_id_default_empty() -> None:
    """analyzer_id defaults to empty string when not supplied."""
    analyzer = _make_analyzer("")
    assert analyzer.analyzer_id == ""


def test_progress_emitted_to_stderr(capsys: pytest.CaptureFixture) -> None:
    """run_batches must emit [LLM] progress lines to stderr."""
    analyzer = _make_analyzer("ssd-1")
    batch = Batch(file_path="SKILL.md", content="# test", findings=[])

    mock_response = LLMAnalysisResult(findings=[])
    analyzer._structured_llm.invoke.return_value = mock_response

    analyzer.run_batches([batch])
    captured = capsys.readouterr()
    assert "[LLM] ssd-1" in captured.err
    assert "requesting" in captured.err
    assert "done" in captured.err


def test_no_progress_when_no_analyzer_id(capsys: pytest.CaptureFixture) -> None:
    """When analyzer_id is empty, no progress line should be printed."""
    analyzer = _make_analyzer("")
    batch = Batch(file_path="SKILL.md", content="# test", findings=[])

    mock_response = LLMAnalysisResult(findings=[])
    analyzer._structured_llm.invoke.return_value = mock_response

    analyzer.run_batches([batch])
    captured = capsys.readouterr()
    assert "[LLM]" not in captured.err


def test_progress_includes_file_label(capsys: pytest.CaptureFixture) -> None:
    """Progress lines should include the file label from the batch."""
    analyzer = _make_analyzer("meta_analyzer")
    batch = Batch(file_path="path/to/SKILL.md", content="# test", findings=[])

    mock_response = LLMAnalysisResult(findings=[])
    analyzer._structured_llm.invoke.return_value = mock_response

    analyzer.run_batches([batch])
    captured = capsys.readouterr()
    assert "SKILL.md" in captured.err


def test_progress_shows_finding_count(capsys: pytest.CaptureFixture) -> None:
    """The 'done' progress line should include the number of findings."""
    analyzer = _make_analyzer("ssd-1")
    batch = Batch(file_path="SKILL.md", content="# test", findings=[])

    mock_response = LLMAnalysisResult(findings=[])
    analyzer._structured_llm.invoke.return_value = mock_response

    analyzer.run_batches([batch])
    captured = capsys.readouterr()
    assert "0 findings" in captured.err


def test_arun_batches_emits_progress(capsys: pytest.CaptureFixture) -> None:
    """arun_batches must also emit [LLM] progress lines to stderr."""
    analyzer = _make_analyzer("async-analyzer")
    batch = Batch(file_path="SKILL.md", content="# test", findings=[])

    mock_response = LLMAnalysisResult(findings=[])

    async def _fake_ainvoke(*args: object, **kwargs: object) -> LLMAnalysisResult:
        return mock_response

    analyzer._structured_llm.ainvoke = _fake_ainvoke

    asyncio.run(analyzer.arun_batches([batch]))
    captured = capsys.readouterr()
    assert "[LLM] async-analyzer" in captured.err
    assert "requesting" in captured.err
    assert "done" in captured.err


def test_arun_batches_no_progress_empty_id(capsys: pytest.CaptureFixture) -> None:
    """arun_batches with empty analyzer_id should not emit any progress."""
    analyzer = _make_analyzer("")
    batch = Batch(file_path="SKILL.md", content="# test", findings=[])

    mock_response = LLMAnalysisResult(findings=[])

    async def _fake_ainvoke(*args: object, **kwargs: object) -> LLMAnalysisResult:
        return mock_response

    analyzer._structured_llm.ainvoke = _fake_ainvoke

    asyncio.run(analyzer.arun_batches([batch]))
    captured = capsys.readouterr()
    assert "[LLM]" not in captured.err


def test_emit_progress_direct(capsys: pytest.CaptureFixture) -> None:
    """_emit_progress() with a set analyzer_id prints correctly to stderr."""
    analyzer = _make_analyzer("direct-test")
    analyzer._emit_progress("myfile.md", "requesting...")
    captured = capsys.readouterr()
    assert "[LLM] direct-test: myfile.md (requesting...)" in captured.err


def test_emit_progress_with_detail(capsys: pytest.CaptureFixture) -> None:
    """_emit_progress() with detail appends the detail in parentheses."""
    analyzer = _make_analyzer("direct-test")
    analyzer._emit_progress("myfile.md", "done", "3 findings")
    captured = capsys.readouterr()
    assert "(done) (3 findings)" in captured.err


def test_emit_progress_silent_empty_id(capsys: pytest.CaptureFixture) -> None:
    """_emit_progress() with empty analyzer_id prints nothing."""
    analyzer = _make_analyzer("")
    analyzer._emit_progress("myfile.md", "requesting...")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_multiple_batches_emit_per_batch(capsys: pytest.CaptureFixture) -> None:
    """Each batch should produce its own pair of progress lines."""
    analyzer = _make_analyzer("multi")
    batches = [
        Batch(file_path="a.md", content="a", findings=[]),
        Batch(file_path="b.md", content="b", findings=[]),
    ]

    mock_response = LLMAnalysisResult(findings=[])
    analyzer._structured_llm.invoke.return_value = mock_response

    analyzer.run_batches(batches)
    captured = capsys.readouterr()
    # Should see progress for both files
    assert "a.md" in captured.err
    assert "b.md" in captured.err
    # Two 'requesting' and two 'done' lines
    assert captured.err.count("requesting") == 2
    assert captured.err.count("done") == 2
