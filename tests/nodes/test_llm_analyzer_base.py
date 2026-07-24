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

"""Tests for llm_analyzer_base and meta_analyzer: batching, chunking, prompt building, filter/merge."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from skillspector.llm_analyzer_base import (
    Batch,
    LLMAnalysisResult,
    LLMAnalyzerBase,
    LLMFinding,
    chunk_file_by_lines,
    estimate_tokens,
    findings_in_range,
    number_lines,
)
from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import (
    LLMMetaAnalyzer,
    MetaAnalyzerFinding,
    MetaAnalyzerResult,
    _format_findings_for_prompt,
)

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_approximation(self) -> None:
        assert estimate_tokens("a" * 400) == 100

    def test_short_string(self) -> None:
        assert estimate_tokens("hi") == 0  # 2 // 4 == 0


# ---------------------------------------------------------------------------
# chunk_file_by_lines
# ---------------------------------------------------------------------------


class TestChunkFileByLines:
    def test_empty_content(self) -> None:
        chunks = chunk_file_by_lines("", max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == ("", 1, 1)

    def test_small_file_single_chunk(self) -> None:
        content = "line1\nline2\nline3\n"
        chunks = chunk_file_by_lines(content, max_tokens=10000)
        assert len(chunks) == 1
        text, start, end = chunks[0]
        assert text == content
        assert start == 1
        assert end == 3

    def test_large_file_splits_into_multiple_chunks(self) -> None:
        lines = [f"line {i}: {'x' * 40}\n" for i in range(100)]
        content = "".join(lines)
        tokens_per_line = estimate_tokens(lines[0])
        max_tokens = tokens_per_line * 20

        chunks = chunk_file_by_lines(content, max_tokens=max_tokens, overlap_lines=5)
        assert len(chunks) > 1
        assert chunks[0][1] == 1
        assert chunks[-1][2] == 100

    def test_overlap_between_chunks(self) -> None:
        lines = [f"line {i}: {'x' * 80}\n" for i in range(50)]
        content = "".join(lines)
        tokens_per_line = estimate_tokens(lines[0])
        max_tokens = tokens_per_line * 10

        chunks = chunk_file_by_lines(content, max_tokens=max_tokens, overlap_lines=3)
        assert len(chunks) >= 2
        _, _, end1 = chunks[0]
        _, start2, _ = chunks[1]
        assert start2 <= end1  # overlap exists

    def test_single_very_long_line(self) -> None:
        content = "x" * 100_000 + "\n"
        chunks = chunk_file_by_lines(content, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0][1] == 1


# ---------------------------------------------------------------------------
# findings_in_range
# ---------------------------------------------------------------------------


class TestFindingsInRange:
    def _make_finding(self, line: int, file: str = "test.py") -> Finding:
        return Finding(rule_id="R1", message="test", file=file, start_line=line)

    def test_all_in_range(self) -> None:
        fs = [self._make_finding(5), self._make_finding(10)]
        assert len(findings_in_range(fs, 1, 20)) == 2

    def test_partial_match(self) -> None:
        fs = [self._make_finding(5), self._make_finding(25)]
        result = findings_in_range(fs, 1, 10)
        assert len(result) == 1
        assert result[0].start_line == 5

    def test_empty_findings(self) -> None:
        assert findings_in_range([], 1, 100) == []

    def test_none_in_range(self) -> None:
        fs = [self._make_finding(50)]
        assert findings_in_range(fs, 1, 10) == []


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


class TestBatch:
    def test_full_file_label(self) -> None:
        b = Batch(file_path="SKILL.md", content="hi")
        assert b.file_label == "File: SKILL.md"
        assert not b.is_chunk

    def test_chunk_label(self) -> None:
        b = Batch(file_path="big.py", content="chunk", start_line=50, end_line=100)
        assert b.is_chunk
        assert "50" in b.file_label and "100" in b.file_label


# ---------------------------------------------------------------------------
# Shared mocks (used by multiple test classes below)
# ---------------------------------------------------------------------------


def _mock_get_chat_model(*_args, **_kwargs):
    """Return a mock ChatOpenAI that supports with_structured_output."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = MagicMock()
    return mock_llm


MOCK_PATCH_TARGET = "skillspector.llm_analyzer_base.get_chat_model"


class _RawTextAnalyzer(LLMAnalyzerBase):
    """Test analyzer for raw-string mode."""

    response_schema = None

    def parse_response(self, response: object, batch: Batch) -> list[str]:
        assert isinstance(response, str)
        return [response]


# ---------------------------------------------------------------------------
# number_lines
# ---------------------------------------------------------------------------


class TestNumberLines:
    def test_basic_numbering(self) -> None:
        result = number_lines("alpha\nbeta\ngamma")
        assert result == "L1: alpha\nL2: beta\nL3: gamma"

    def test_chunk_offset(self) -> None:
        result = number_lines("x\ny", start_line=100)
        assert result == "L100: x\nL101: y"

    def test_empty_content(self) -> None:
        assert number_lines("") == ""

    def test_single_line(self) -> None:
        assert number_lines("only") == "L1: only"

    def test_zero_padding(self) -> None:
        lines = "\n".join(f"line{i}" for i in range(11))
        result = number_lines(lines)
        assert result.startswith("L01: line0")
        assert "L11: line10" in result


# ---------------------------------------------------------------------------
# LLMAnalyzerBase.build_prompt (default implementation)
# ---------------------------------------------------------------------------


class TestDefaultBuildPrompt:
    MODEL = "nvidia/openai/gpt-oss-120b"
    ANALYZER_PROMPT = "Look for hardcoded credentials and secret leaks."

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_contains_analyzer_prompt(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt=self.ANALYZER_PROMPT, model=self.MODEL)
        batch = Batch(file_path="config.py", content="API_KEY = 'abc123'")
        prompt = analyzer.build_prompt(batch)
        assert self.ANALYZER_PROMPT in prompt

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_contains_file_label(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt=self.ANALYZER_PROMPT, model=self.MODEL)
        batch = Batch(file_path="config.py", content="x = 1")
        prompt = analyzer.build_prompt(batch)
        assert "File: config.py" in prompt

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_content_is_line_numbered(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt=self.ANALYZER_PROMPT, model=self.MODEL)
        batch = Batch(file_path="a.py", content="import os\nos.getenv('SECRET')")
        prompt = analyzer.build_prompt(batch)
        assert "L1: import os" in prompt
        assert "L2: os.getenv('SECRET')" in prompt

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_chunk_offset_preserved(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt=self.ANALYZER_PROMPT, model=self.MODEL)
        batch = Batch(
            file_path="big.py",
            content="dangerous()\nsafe()",
            start_line=50,
            end_line=51,
        )
        prompt = analyzer.build_prompt(batch)
        assert "L50: dangerous()" in prompt
        assert "L51: safe()" in prompt
        assert "lines 50" in prompt


# ---------------------------------------------------------------------------
# LLMAnalyzerBase.parse_response (default — returns Finding objects)
# ---------------------------------------------------------------------------


class TestBaseParseResponse:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_returns_finding_objects(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        batch = Batch(file_path="app.py", content="code")
        llm_result = LLMAnalysisResult(
            findings=[
                LLMFinding(
                    rule_id="SEC-001",
                    message="Hardcoded secret",
                    severity="HIGH",
                    start_line=5,
                    end_line=7,
                    confidence=0.9,
                    explanation="Contains API key",
                    remediation="Use env vars",
                ),
            ]
        )
        findings = analyzer.parse_response(llm_result, batch)
        assert len(findings) == 1
        assert isinstance(findings[0], Finding)
        assert findings[0].rule_id == "SEC-001"
        assert findings[0].file == "app.py"
        assert findings[0].start_line == 5
        assert findings[0].end_line == 7

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_empty_result(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        batch = Batch(file_path="a.py", content="code")
        findings = analyzer.parse_response(LLMAnalysisResult(findings=[]), batch)
        assert findings == []

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_raises_for_unknown_response(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        batch = Batch(file_path="a.py", content="code")
        with pytest.raises(NotImplementedError):
            analyzer.parse_response("raw string", batch)


# ---------------------------------------------------------------------------
# MetaAnalyzerResult — tolerate LLMs that stringify the `findings` array
# ---------------------------------------------------------------------------


class TestMetaAnalyzerResultFindingsValidator:
    _FINDING = {
        "pattern_id": "E2",
        "start_line": 12,
        "is_vulnerability": True,
        "confidence": 0.9,
        "intent": "malicious",
        "impact": "high",
    }

    def test_findings_as_json_string(self) -> None:
        """Some LLMs return the findings array as a JSON string, not a list."""
        result = MetaAnalyzerResult.model_validate({"findings": json.dumps([self._FINDING])})
        assert len(result.findings) == 1
        assert result.findings[0].pattern_id == "E2"

    def test_findings_as_native_list(self) -> None:
        result = MetaAnalyzerResult.model_validate({"findings": [self._FINDING]})
        assert len(result.findings) == 1

    def test_findings_invalid_string_yields_empty(self) -> None:
        result = MetaAnalyzerResult.model_validate({"findings": "not json"})
        assert result.findings == []

    def test_findings_non_list_json_yields_empty(self) -> None:
        result = MetaAnalyzerResult.model_validate({"findings": json.dumps({"a": 1})})
        assert result.findings == []


# ---------------------------------------------------------------------------
# LLMAnalyzerBase.collect_findings
# ---------------------------------------------------------------------------


class TestCollectFindings:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_flattens_batches(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        f1 = Finding(rule_id="A", message="a", file="x.py", start_line=1)
        f2 = Finding(rule_id="B", message="b", file="y.py", start_line=2)
        batch_a = Batch(file_path="x.py", content="x")
        batch_b = Batch(file_path="y.py", content="y")
        results = [(batch_a, [f1]), (batch_b, [f2])]
        findings = analyzer.collect_findings(results)
        assert len(findings) == 2
        assert findings[0].rule_id == "A"
        assert findings[1].rule_id == "B"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_empty_results(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        assert analyzer.collect_findings([]) == []


# ---------------------------------------------------------------------------
# LLMAnalyzerBase raw-string mode
# ---------------------------------------------------------------------------


class TestRawStringMode:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_run_batches_uses_message_text_for_content_blocks(self) -> None:
        analyzer = _RawTextAnalyzer(base_prompt="test", model=self.MODEL)
        analyzer._llm.invoke.return_value = AIMessage(content=[{"type": "text", "text": "chunk"}])

        results = analyzer.run_batches([Batch(file_path="a.py", content="code")])

        assert results[0][1] == ["chunk"]

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_arun_batches_uses_message_text_for_content_blocks(self) -> None:
        analyzer = _RawTextAnalyzer(base_prompt="test", model=self.MODEL)
        analyzer._llm.ainvoke = AsyncMock(
            return_value=AIMessage(content=[{"type": "text", "text": "async chunk"}])
        )

        results = await analyzer.arun_batches([Batch(file_path="a.py", content="code")])

        assert results[0][1] == ["async chunk"]


# ---------------------------------------------------------------------------
# LLMAnalyzerBase.arun_batches (async parallel execution)
# ---------------------------------------------------------------------------


class TestARunBatches:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_processes_all_batches(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = AsyncMock(
            return_value=LLMAnalysisResult(
                findings=[
                    LLMFinding(rule_id="T-1", message="hit", severity="LOW", start_line=1),
                ]
            )
        )
        batches = [
            Batch(file_path="a.py", content="code a"),
            Batch(file_path="b.py", content="code b"),
            Batch(file_path="c.py", content="code c"),
        ]
        results = await analyzer.arun_batches(batches)
        assert len(results) == 3
        assert analyzer._structured_llm.ainvoke.call_count == 3
        files = {batch.file_path for batch, _ in results}
        assert files == {"a.py", "b.py", "c.py"}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_returns_parsed_findings(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = AsyncMock(
            return_value=LLMAnalysisResult(
                findings=[
                    LLMFinding(
                        rule_id="SEC-001",
                        message="Bad",
                        severity="HIGH",
                        start_line=5,
                        confidence=0.9,
                    ),
                ]
            )
        )
        batches = [Batch(file_path="x.py", content="code")]
        results = await analyzer.arun_batches(batches)
        batch, findings = results[0]
        assert len(findings) == 1
        assert isinstance(findings[0], Finding)
        assert findings[0].rule_id == "SEC-001"
        assert findings[0].file == "x.py"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_empty_batches(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        results = await analyzer.arun_batches([])
        assert results == []

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_respects_max_concurrency(self) -> None:
        """Verify the semaphore limits concurrent LLM calls."""
        import asyncio

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        original_ainvoke = AsyncMock(return_value=LLMAnalysisResult(findings=[]))

        async def _tracking_ainvoke(prompt: str) -> LLMAnalysisResult:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            result = await original_ainvoke(prompt)
            async with lock:
                current_concurrent -= 1
            return result

        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = _tracking_ainvoke

        batches = [Batch(file_path=f"f{i}.py", content="code") for i in range(8)]
        await analyzer.arun_batches(batches, max_concurrency=3)
        assert max_concurrent <= 3

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_raw_string_mode(self) -> None:
        """When response_schema is None, arun_batches uses _llm.ainvoke."""
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm = None
        analyzer._llm.ainvoke = AsyncMock(return_value=AIMessage(content="raw text"))

        batch = Batch(file_path="a.py", content="code")
        with pytest.raises(NotImplementedError):
            await analyzer.arun_batches([batch])
        analyzer._llm.ainvoke.assert_called_once()

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_kwargs_passed_to_build_prompt(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = AsyncMock(return_value=LLMAnalysisResult(findings=[]))
        original_build = analyzer.build_prompt
        captured_kwargs: list[dict] = []

        def _spy_build(batch, **kwargs):
            captured_kwargs.append(kwargs)
            return original_build(batch, **kwargs)

        analyzer.build_prompt = _spy_build
        batches = [Batch(file_path="a.py", content="code")]
        await analyzer.arun_batches(batches, extra_key="extra_val")
        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["extra_key"] == "extra_val"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_no_race_conditions_under_high_concurrency(self) -> None:
        """Stress test: 100 batches with random delays to surface race conditions.

        Verifies that every batch result is present, correctly paired to its
        originating batch (no swaps), and that no results are lost or duplicated.
        """
        import asyncio
        import random

        num_batches = 100

        async def _delayed_ainvoke(prompt: str) -> LLMAnalysisResult:
            await asyncio.sleep(random.uniform(0.001, 0.02))
            for line in prompt.splitlines():
                if line.startswith("## File: "):
                    filename = line.removeprefix("## File: ").strip()
                    idx = int(filename.removesuffix(".py").removeprefix("file_"))
                    return LLMAnalysisResult(
                        findings=[
                            LLMFinding(
                                rule_id=f"R-{idx}",
                                message=f"finding for {filename}",
                                severity="LOW",
                                start_line=idx + 1,
                            ),
                        ]
                    )
            return LLMAnalysisResult(findings=[])

        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = _delayed_ainvoke

        batches = [Batch(file_path=f"file_{i}.py", content=f"code_{i}") for i in range(num_batches)]
        results = await analyzer.arun_batches(batches, max_concurrency=20)

        assert len(results) == num_batches

        seen_files: set[str] = set()
        for batch, findings in results:
            assert batch.file_path not in seen_files, f"duplicate: {batch.file_path}"
            seen_files.add(batch.file_path)

            assert len(findings) == 1
            idx = int(batch.file_path.removesuffix(".py").removeprefix("file_"))
            assert findings[0].rule_id == f"R-{idx}"
            assert findings[0].start_line == idx + 1
            assert findings[0].file == batch.file_path

        assert seen_files == {f"file_{i}.py" for i in range(num_batches)}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_failed_batch_does_not_abort_the_others(self) -> None:
        """A transient failure costs only its own batch, not the whole fan-out."""

        async def _flaky_ainvoke(prompt: str) -> LLMAnalysisResult:
            if "b.py" in prompt:
                raise RuntimeError("429 Too Many Requests")
            return LLMAnalysisResult(findings=[])

        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = _flaky_ainvoke

        batches = [
            Batch(file_path="a.py", content="code a"),
            Batch(file_path="b.py", content="code b"),
            Batch(file_path="c.py", content="code c"),
        ]
        results = await analyzer.arun_batches(batches)
        assert {batch.file_path for batch, _ in results} == {"a.py", "c.py"}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_all_batches_failed_returns_empty(self) -> None:
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        batches = [Batch(file_path="a.py", content="code")]
        assert await analyzer.arun_batches(batches) == []

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    async def test_value_error_still_propagates(self) -> None:
        """ValueError signals misconfiguration, not infra trouble — never swallowed."""
        analyzer = LLMAnalyzerBase(base_prompt="test", model=self.MODEL)
        analyzer._structured_llm.ainvoke = AsyncMock(side_effect=ValueError("no API key"))
        batches = [Batch(file_path="a.py", content="code")]
        with pytest.raises(ValueError, match="no API key"):
            await analyzer.arun_batches(batches)


# ---------------------------------------------------------------------------
# _format_findings_for_prompt (per-file, no truncation)
# ---------------------------------------------------------------------------


class TestFormatFindingsForPrompt:
    def test_empty_findings(self) -> None:
        text = _format_findings_for_prompt([])
        assert "No static analysis findings" in text

    def test_full_matched_text_preserved(self) -> None:
        long_match = "x" * 500
        f = Finding(rule_id="E1", message="msg", matched_text=long_match, file="a.py", start_line=1)
        text = _format_findings_for_prompt([f])
        assert long_match in text

    def test_full_context_preserved(self) -> None:
        long_ctx = "line\n" * 200
        f = Finding(rule_id="E1", message="msg", context=long_ctx, file="a.py", start_line=1)
        text = _format_findings_for_prompt([f])
        assert long_ctx.strip() in text.replace("   ", "")


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class TestLLMAnalysisResult:
    """Tests for the base discovery-mode schemas in llm_analyzer_base."""

    def test_valid_finding(self) -> None:
        f = LLMFinding(
            rule_id="SEC-001",
            message="Hardcoded credential",
            severity="HIGH",
            start_line=10,
            confidence=0.9,
        )
        result = LLMAnalysisResult(findings=[f])
        assert len(result.findings) == 1
        assert result.findings[0].confidence == 0.9

    def test_confidence_is_clamped(self) -> None:
        """Out-of-range confidence is clamped, not rejected, so a slightly off
        model value does not fail the whole structured-output parse."""
        hi = LLMFinding(rule_id="X", message="x", severity="LOW", start_line=1, confidence=1.5)
        lo = LLMFinding(rule_id="X", message="x", severity="LOW", start_line=1, confidence=-0.3)
        assert hi.confidence == 1.0
        assert lo.confidence == 0.0

    def test_confidence_100_scale_normalized(self) -> None:
        """Ollama and some models return confidence on 0-100 scale; must be normalized."""
        f = LLMFinding(rule_id="X", message="x", severity="LOW", start_line=1, confidence=100)
        assert f.confidence == pytest.approx(1.0)

    def test_confidence_85_scale_normalized(self) -> None:
        f = LLMFinding(rule_id="X", message="x", severity="LOW", start_line=1, confidence=85)
        assert f.confidence == pytest.approx(0.85)

    def test_confidence_negative_clamped_to_zero(self) -> None:
        f = LLMFinding(rule_id="X", message="x", severity="LOW", start_line=1, confidence=-10)
        assert f.confidence == pytest.approx(0.0)

    def test_confidence_overlarge_clamped_to_one(self) -> None:
        """Values > 100 (e.g. 150) are divided then clamped."""
        f = LLMFinding(rule_id="X", message="x", severity="LOW", start_line=1, confidence=150)
        assert f.confidence == pytest.approx(1.0)

    def test_confidence_validation(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            LLMFinding(
                rule_id="X",
                message="x",
                severity="LOW",
                start_line=1,
                confidence="not-a-number",
            )

    def test_severity_validation(self) -> None:
        with pytest.raises(ValueError):
            LLMFinding(
                rule_id="X",
                message="x",
                severity="UNKNOWN",
                start_line=1,
            )

    def test_empty_findings(self) -> None:
        result = LLMAnalysisResult(findings=[])
        assert result.findings == []

    def test_end_line_optional(self) -> None:
        f = LLMFinding(
            rule_id="X",
            message="x",
            severity="LOW",
            start_line=1,
        )
        assert f.end_line is None

        f2 = LLMFinding(
            rule_id="X",
            message="x",
            severity="LOW",
            start_line=1,
            end_line=5,
        )
        assert f2.end_line == 5

    def test_to_finding(self) -> None:
        f = LLMFinding(
            rule_id="SEC-001",
            message="Hardcoded secret",
            severity="HIGH",
            start_line=10,
            end_line=12,
            confidence=0.95,
            explanation="Contains API key",
            remediation="Use env vars",
        )
        finding = f.to_finding("config.py")
        assert isinstance(finding, Finding)
        assert finding.rule_id == "SEC-001"
        assert finding.file == "config.py"
        assert finding.start_line == 10
        assert finding.end_line == 12
        assert finding.confidence == 0.95
        assert finding.explanation == "Contains API key"
        assert finding.remediation == "Use env vars"

    def test_model_dump(self) -> None:
        f = LLMFinding(
            rule_id="SEC-002",
            message="Open redirect",
            severity="MEDIUM",
            start_line=42,
            confidence=0.8,
        )
        d = f.model_dump()
        assert d["rule_id"] == "SEC-002"
        assert d["severity"] == "MEDIUM"
        assert d["explanation"] == ""
        assert d["end_line"] is None


class TestMetaAnalyzerResult:
    """Tests for the meta-analyzer-specific schemas."""

    def test_valid_finding(self) -> None:
        f = MetaAnalyzerFinding(
            pattern_id="E1",
            is_vulnerability=True,
            confidence=0.9,
            intent="malicious",
            impact="high",
            explanation="Dangerous",
            remediation="Fix it",
        )
        result = MetaAnalyzerResult(findings=[f])
        assert len(result.findings) == 1
        assert result.findings[0].confidence == 0.9

    def test_confidence_is_clamped(self) -> None:
        """Out-of-range confidence is clamped, not rejected, so a slightly off
        model value does not fail the whole structured-output parse."""
        high = MetaAnalyzerFinding(
            pattern_id="E1",
            is_vulnerability=True,
            confidence=1.5,
            intent="malicious",
            impact="high",
        )
        low = MetaAnalyzerFinding(
            pattern_id="E1",
            is_vulnerability=True,
            confidence=-0.2,
            intent="malicious",
            impact="high",
        )
        assert high.confidence == 1.0
        assert low.confidence == 0.0

    def test_confidence_100_scale_normalized(self) -> None:
        """Ollama-style 0-100 scale must be normalized to 0-1."""
        f = MetaAnalyzerFinding(
            pattern_id="E1",
            is_vulnerability=True,
            confidence=100,
            intent="malicious",
            impact="high",
        )
        assert f.confidence == pytest.approx(1.0)

    def test_confidence_75_scale_normalized(self) -> None:
        f = MetaAnalyzerFinding(
            pattern_id="E1", is_vulnerability=True, confidence=75, intent="malicious", impact="high"
        )
        assert f.confidence == pytest.approx(0.75)

    def test_confidence_negative_clamped(self) -> None:
        f = MetaAnalyzerFinding(
            pattern_id="E1", is_vulnerability=True, confidence=-5, intent="malicious", impact="high"
        )
        assert f.confidence == pytest.approx(0.0)

    def test_confidence_validation(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            MetaAnalyzerFinding(
                pattern_id="E1",
                is_vulnerability=True,
                confidence="bad",
                intent="malicious",
                impact="high",
            )

    def test_intent_validation(self) -> None:
        with pytest.raises(ValueError):
            MetaAnalyzerFinding(
                pattern_id="E1",
                is_vulnerability=True,
                confidence=0.5,
                intent="unknown",
                impact="high",
            )

    def test_empty_findings(self) -> None:
        result = MetaAnalyzerResult(findings=[])
        assert result.findings == []

    def test_start_line_optional(self) -> None:
        f_no_line = MetaAnalyzerFinding(
            pattern_id="E1",
            is_vulnerability=True,
            confidence=0.9,
            intent="malicious",
            impact="high",
        )
        assert f_no_line.start_line is None

        f_with_line = MetaAnalyzerFinding(
            pattern_id="E1",
            start_line=42,
            is_vulnerability=True,
            confidence=0.9,
            intent="malicious",
            impact="high",
        )
        assert f_with_line.start_line == 42

    def test_model_dump(self) -> None:
        f = MetaAnalyzerFinding(
            pattern_id="E2",
            is_vulnerability=True,
            confidence=0.8,
            intent="negligent",
            impact="medium",
        )
        d = f.model_dump()
        assert d["pattern_id"] == "E2"
        assert d["confidence"] == 0.8
        assert d["explanation"] == ""
        assert d["start_line"] is None


class TestStructuredOutputSchema:
    """The response schemas must stay portable across structured-output backends.

    Pydantic ge/le bounds emit JSON-schema ``minimum`` / ``maximum``, which some
    OpenAI-compatible structured-output / tool-calling endpoints reject when they
    validate the response schema. The ranges are enforced by runtime validators
    instead, so these keywords must not appear in the emitted schema.
    """

    @staticmethod
    def _numeric_keywords(schema: dict) -> set[str]:
        found: set[str] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                found.update(k for k in ("minimum", "maximum") if k in node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        return found

    def test_llm_finding_schema_has_no_numeric_bounds(self) -> None:
        assert self._numeric_keywords(LLMFinding.model_json_schema()) == set()

    def test_meta_finding_schema_has_no_numeric_bounds(self) -> None:
        assert self._numeric_keywords(MetaAnalyzerFinding.model_json_schema()) == set()

    def test_llm_finding_clamps_confidence(self) -> None:
        hi = LLMFinding(rule_id="R", message="m", severity="LOW", start_line=1, confidence=1.5)
        lo = LLMFinding(rule_id="R", message="m", severity="LOW", start_line=1, confidence=-0.3)
        assert hi.confidence == 1.0
        assert lo.confidence == 0.0

    def test_llm_finding_clamps_start_line(self) -> None:
        assert LLMFinding(rule_id="R", message="m", severity="LOW", start_line=0).start_line == 1
        assert LLMFinding(rule_id="R", message="m", severity="LOW", start_line=42).start_line == 42

    def test_llm_finding_start_line_is_required(self) -> None:
        """start_line stays required: a finding with no location is rejected,
        not materialised at line 1."""
        with pytest.raises(ValueError):
            LLMFinding(rule_id="R", message="m", severity="LOW")


# ---------------------------------------------------------------------------
# LLMMetaAnalyzer.get_batches
# ---------------------------------------------------------------------------


class TestLLMMetaAnalyzerGetBatches:
    MODEL = "nvidia/openai/gpt-oss-120b"

    def _make_finding(self, file: str, line: int = 1, rule_id: str = "E1") -> Finding:
        return Finding(rule_id=rule_id, message="test", file=file, start_line=line)

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_single_file_single_batch(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py")]
        file_cache = {"a.py": "print('hello')"}
        batches = analyzer.get_batches(["a.py"], file_cache, findings)
        assert len(batches) == 1
        assert batches[0].file_path == "a.py"
        assert len(batches[0].findings) == 1

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_multiple_files_multiple_batches(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py"), self._make_finding("b.py")]
        file_cache = {"a.py": "code a", "b.py": "code b"}
        batches = analyzer.get_batches(["a.py", "b.py"], file_cache, findings)
        assert len(batches) == 2
        paths = {b.file_path for b in batches}
        assert paths == {"a.py", "b.py"}

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_findings_grouped_by_file(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [
            self._make_finding("a.py", rule_id="E1"),
            self._make_finding("a.py", rule_id="E2"),
            self._make_finding("b.py", rule_id="E3"),
        ]
        file_cache = {"a.py": "code a", "b.py": "code b"}
        batches = analyzer.get_batches(["a.py", "b.py"], file_cache, findings)
        a_batch = next(b for b in batches if b.file_path == "a.py")
        b_batch = next(b for b in batches if b.file_path == "b.py")
        assert len(a_batch.findings) == 2
        assert len(b_batch.findings) == 1

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_missing_file_gets_sentinel(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("missing.py")]
        batches = analyzer.get_batches(["missing.py"], {}, findings)
        assert len(batches) == 1
        assert "No content available" in batches[0].content

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_oversized_file_chunked(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        big_content = "\n".join(f"line {i}: {'x' * 200}" for i in range(5000))
        findings = [
            self._make_finding("big.py", line=10),
            self._make_finding("big.py", line=4000),
        ]
        file_cache = {"big.py": big_content}
        batches = analyzer.get_batches(["big.py"], file_cache, findings)
        assert all(b.file_path == "big.py" for b in batches)
        if len(batches) > 1:
            assert batches[0].is_chunk
            all_findings = [f for b in batches for f in b.findings]
            assert len(all_findings) >= 2

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_no_findings_still_creates_batch(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        file_cache = {"a.py": "code"}
        batches = analyzer.get_batches(["a.py"], file_cache, [])
        assert len(batches) == 1
        assert batches[0].findings == []


# ---------------------------------------------------------------------------
# LLMMetaAnalyzer.build_prompt
# ---------------------------------------------------------------------------


class TestLLMMetaAnalyzerBuildPrompt:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_prompt_contains_file_content(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        batch = Batch(file_path="test.py", content="import os\nos.environ['SECRET']")
        prompt = analyzer.build_prompt(batch, metadata_text="Name: test-skill")
        assert "import os" in prompt
        assert "Name: test-skill" in prompt

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_prompt_contains_findings(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        f = Finding(rule_id="E2", message="env leak", file="test.py", start_line=2)
        batch = Batch(file_path="test.py", content="code", findings=[f])
        prompt = analyzer.build_prompt(batch, metadata_text="")
        assert "E2" in prompt
        assert "env leak" in prompt

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_chunk_label_in_prompt(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        batch = Batch(file_path="big.py", content="chunk", start_line=100, end_line=200)
        prompt = analyzer.build_prompt(batch, metadata_text="")
        assert "100" in prompt and "200" in prompt

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_prompt_has_critical_instructions(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        batch = Batch(file_path="a.py", content="x")
        prompt = analyzer.build_prompt(batch, metadata_text="")
        assert "CRITICAL INSTRUCTIONS" in prompt


# ---------------------------------------------------------------------------
# LLMMetaAnalyzer.parse_response (structured output)
# ---------------------------------------------------------------------------


class TestLLMMetaAnalyzerParseResponse:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_converts_pydantic_to_dicts(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        batch = Batch(file_path="a.py", content="code")
        llm_result = MetaAnalyzerResult(
            findings=[
                MetaAnalyzerFinding(
                    pattern_id="E1",
                    is_vulnerability=True,
                    confidence=0.9,
                    intent="malicious",
                    impact="high",
                    explanation="Bad stuff",
                ),
            ]
        )
        items = analyzer.parse_response(llm_result, batch)
        assert len(items) == 1
        assert items[0]["pattern_id"] == "E1"
        assert items[0]["_file"] == "a.py"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_empty_findings(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        batch = Batch(file_path="a.py", content="code")
        items = analyzer.parse_response(MetaAnalyzerResult(findings=[]), batch)
        assert items == []


# ---------------------------------------------------------------------------
# LLMMetaAnalyzer.apply_filter (keyed by file + rule_id + start/end_line)
# ---------------------------------------------------------------------------


class TestLLMMetaAnalyzerApplyFilter:
    MODEL = "nvidia/openai/gpt-oss-120b"

    def _make_finding(
        self,
        file: str,
        rule_id: str,
        line: int = 1,
        end_line: int | None = None,
    ) -> Finding:
        return Finding(
            rule_id=rule_id,
            message="original",
            file=file,
            start_line=line,
            end_line=end_line or line,
        )

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_confirmed_finding_kept(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py", "E1")]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "E1",
                "is_vulnerability": True,
                "confidence": 0.9,
                "explanation": "Dangerous",
                "remediation": "Fix it",
                "_file": "a.py",
            }
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 1
        assert result[0].explanation == "Dangerous"
        assert result[0].confidence == 0.9

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_unconfirmed_finding_filtered_out(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py", "E1")]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "E1",
                "is_vulnerability": False,
                "confidence": 0.3,
            }
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 0

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_low_confidence_filtered_out(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py", "E1")]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "E1",
                "is_vulnerability": True,
                "confidence": 0.3,
            }
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 0

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_file_scoped_keying(self) -> None:
        """Same rule_id in different files should be independently filtered."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [
            self._make_finding("a.py", "E1"),
            self._make_finding("b.py", "E1"),
        ]
        batch_a = Batch(file_path="a.py", content="code a", findings=[findings[0]])
        batch_b = Batch(file_path="b.py", content="code b", findings=[findings[1]])
        llm_a = [
            {
                "pattern_id": "E1",
                "is_vulnerability": True,
                "confidence": 0.9,
                "explanation": "Bad in a.py",
                "_file": "a.py",
            }
        ]
        llm_b = [
            {"pattern_id": "E1", "is_vulnerability": False, "confidence": 0.2, "_file": "b.py"}
        ]
        result = analyzer.apply_filter(findings, [(batch_a, llm_a), (batch_b, llm_b)])
        assert len(result) == 1
        assert result[0].file == "a.py"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_multiple_findings_same_file(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [
            self._make_finding("a.py", "E1"),
            self._make_finding("a.py", "E2"),
        ]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "E1",
                "is_vulnerability": True,
                "confidence": 0.8,
                "explanation": "E1 bad",
                "_file": "a.py",
            },
            {
                "pattern_id": "E2",
                "is_vulnerability": True,
                "confidence": 0.7,
                "explanation": "E2 bad",
                "_file": "a.py",
            },
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 2

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_empty_batch_results(self) -> None:
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py", "E1")]
        result = analyzer.apply_filter(findings, [])
        assert len(result) == 0

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_granular_keying_filters_per_instance(self) -> None:
        """Two findings with the same rule_id in one file; LLM confirms only one."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [
            self._make_finding("a.py", "EA4", line=15),
            self._make_finding("a.py", "EA4", line=42),
        ]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "EA4",
                "start_line": 42,
                "is_vulnerability": True,
                "confidence": 0.85,
                "explanation": "Loops forever",
                "remediation": "Add a bound",
                "_file": "a.py",
            },
            {
                "pattern_id": "EA4",
                "start_line": 15,
                "is_vulnerability": False,
                "confidence": 0.3,
                "_file": "a.py",
            },
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 1
        assert result[0].start_line == 42
        assert result[0].explanation == "Loops forever"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_coarse_fallback_when_no_start_line(self) -> None:
        """LLM response without start_line falls back to coarse (file, rule_id) keying."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [self._make_finding("a.py", "E1", line=10)]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "E1",
                "is_vulnerability": True,
                "confidence": 0.9,
                "explanation": "Dangerous",
                "remediation": "Fix it",
                "_file": "a.py",
            }
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 1
        assert result[0].explanation == "Dangerous"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_granular_keying_both_confirmed(self) -> None:
        """Both instances confirmed independently via start_line."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        findings = [
            self._make_finding("a.py", "EA4", line=10),
            self._make_finding("a.py", "EA4", line=30),
        ]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "EA4",
                "start_line": 10,
                "is_vulnerability": True,
                "confidence": 0.8,
                "explanation": "No rate limit",
                "_file": "a.py",
            },
            {
                "pattern_id": "EA4",
                "start_line": 30,
                "is_vulnerability": True,
                "confidence": 0.9,
                "explanation": "Infinite loop",
                "_file": "a.py",
            },
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        assert len(result) == 2
        by_line = {f.start_line: f for f in result}
        assert by_line[10].explanation == "No rate limit"
        assert by_line[30].explanation == "Infinite loop"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_end_line_used_when_provided(self) -> None:
        """When LLM returns end_line, it is used for exact matching; a finding
        with a different end_line falls back to the start-only key."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        # Two findings at the same start_line but different end_lines
        f_short = self._make_finding("a.py", "E1", line=5, end_line=5)
        f_long = self._make_finding("a.py", "E1", line=5, end_line=10)
        findings = [f_short, f_long]
        batch = Batch(file_path="a.py", content="code", findings=findings)
        llm_items = [
            {
                "pattern_id": "E1",
                "start_line": 5,
                "end_line": 10,
                "is_vulnerability": True,
                "confidence": 0.9,
                "explanation": "Long block is dangerous",
                "remediation": "Refactor",
                "_file": "a.py",
            },
        ]
        result = analyzer.apply_filter(findings, [(batch, llm_items)])
        # exact match for f_long; f_short has no exact match, falls back to start_only (None end_line)
        # start_only key not in confirmed_granular, so f_short is not confirmed
        assert len(result) == 1
        assert result[0].end_line == 10
        assert result[0].explanation == "Long block is dangerous"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_static_finding_with_none_end_line_confirmed_by_start(self) -> None:
        """Issue #67: static finding with end_line=None must not be dropped when
        the LLM confirms the same start_line with an explicit end_line.

        Static analyzers typically emit end_line=None; the LLM always fills it
        in.  The confirmed_by_start fallback ensures the finding is kept.
        """
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        # Construct directly — _make_finding converts None to line via `or`.
        finding = Finding(
            rule_id="E2",
            message="env harvest",
            file="agent.py",
            start_line=42,
            end_line=None,
        )
        batch = Batch(file_path="agent.py", content="code", findings=[finding])
        llm_items = [
            {
                "pattern_id": "E2",
                "start_line": 42,
                "end_line": 42,
                "is_vulnerability": True,
                "confidence": 0.88,
                "explanation": "Harvests all env vars",
                "remediation": "Use specific env lookups",
                "_file": "agent.py",
            }
        ]
        result = analyzer.apply_filter([finding], [(batch, llm_items)])
        assert len(result) == 1, "Static finding with end_line=None must not be dropped"
        assert result[0].explanation == "Harvests all env vars"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_static_findings_at_different_lines_only_confirmed_kept(self) -> None:
        """Two static findings (end_line=None) at different start_lines; LLM
        confirms only one.  The unconfirmed finding must not survive the filter."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        f1 = Finding(
            rule_id="P1", message="override", file="skill.md", start_line=10, end_line=None
        )
        f2 = Finding(
            rule_id="P1", message="override", file="skill.md", start_line=30, end_line=None
        )
        batch = Batch(file_path="skill.md", content="code", findings=[f1, f2])
        llm_items = [
            {
                "pattern_id": "P1",
                "start_line": 10,
                "end_line": 10,
                "is_vulnerability": True,
                "confidence": 0.9,
                "explanation": "Instruction override at line 10",
                "_file": "skill.md",
            },
            {
                "pattern_id": "P1",
                "start_line": 30,
                "is_vulnerability": False,
                "confidence": 0.2,
                "_file": "skill.md",
            },
        ]
        result = analyzer.apply_filter([f1, f2], [(batch, llm_items)])
        assert len(result) == 1
        assert result[0].start_line == 10


# ---------------------------------------------------------------------------
# LLMMetaAnalyzer.apply_filter — severity-gated suppression floor
#
# Security invariant: CRITICAL and HIGH static findings must survive LLM
# filtering even if the LLM (operating on attacker-controlled skill content)
# omits or denies them.  MEDIUM/LOW findings are still filtered normally.
# ---------------------------------------------------------------------------


class TestApplyFilterSeverityFloor:
    """Tests for the severity-gated suppression floor in apply_filter.

    Verifies that CRITICAL/HIGH static findings are never silently dropped
    by the LLM filter (adversarial prompt-injection defence), while MEDIUM/LOW
    findings continue to be filtered as before.
    """

    MODEL = "nvidia/openai/gpt-oss-120b"

    def _make_finding(
        self,
        rule_id: str,
        severity: str,
        file: str = "skill.md",
        line: int = 1,
        tags: list[str] | None = None,
    ) -> Finding:
        return Finding(
            rule_id=rule_id,
            message=f"original message for {rule_id}",
            severity=severity,
            confidence=0.8,
            file=file,
            start_line=line,
            tags=tags or [],
        )

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_critical_unconfirmed_kept_with_llm_unconfirmed_tag(self) -> None:
        """A CRITICAL static finding NOT confirmed by the LLM must be kept.

        The finding must appear in the output with its original severity and
        message, and the tag 'llm-unconfirmed' must be present to let consumers
        know it was not LLM-validated (and may represent an adversarial suppression).
        """
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding("CRIT-001", "CRITICAL", line=10)
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        # LLM response: is_vulnerability=False — the LLM denies the finding
        llm_items = [
            {
                "pattern_id": "CRIT-001",
                "start_line": 10,
                "is_vulnerability": False,
                "confidence": 0.2,
                "_file": "skill.md",
            }
        ]
        result = analyzer.apply_filter([finding], [(batch, llm_items)])

        assert len(result) == 1, "CRITICAL finding must NOT be dropped by LLM filtering"
        kept = result[0]
        assert kept.severity == "CRITICAL"
        assert kept.rule_id == "CRIT-001"
        assert kept.message == "original message for CRIT-001"
        assert kept.confidence == 0.8  # original confidence preserved
        assert "llm-unconfirmed" in kept.tags

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_high_unconfirmed_kept_with_llm_unconfirmed_tag(self) -> None:
        """A HIGH static finding NOT confirmed by the LLM must be kept.

        Same invariant as CRITICAL: the original finding is preserved and
        tagged 'llm-unconfirmed'.
        """
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding("HIGH-001", "HIGH", line=5)
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        # LLM response: finding completely omitted (empty list) — simulates
        # a prompt-injection payload making the LLM silently drop the finding
        llm_items: list[dict] = []
        result = analyzer.apply_filter([finding], [(batch, llm_items)])

        assert len(result) == 1, "HIGH finding must NOT be dropped by LLM filtering"
        kept = result[0]
        assert kept.severity == "HIGH"
        assert kept.rule_id == "HIGH-001"
        assert kept.message == "original message for HIGH-001"
        assert kept.confidence == 0.8  # original confidence preserved
        assert "llm-unconfirmed" in kept.tags

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_medium_unconfirmed_still_dropped(self) -> None:
        """A MEDIUM static finding NOT confirmed by the LLM must still be dropped.

        The severity floor only applies to CRITICAL/HIGH.  MEDIUM and LOW
        findings remain subject to normal LLM filtering (false-positive reduction).
        """
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding("MED-001", "MEDIUM", line=3)
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        llm_items = [
            {
                "pattern_id": "MED-001",
                "start_line": 3,
                "is_vulnerability": False,
                "confidence": 0.1,
                "_file": "skill.md",
            }
        ]
        result = analyzer.apply_filter([finding], [(batch, llm_items)])

        assert len(result) == 0, "MEDIUM finding must be dropped when LLM does not confirm it"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_low_unconfirmed_still_dropped(self) -> None:
        """A LOW static finding NOT confirmed by the LLM must still be dropped."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding("LOW-001", "LOW", line=7)
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        llm_items: list[dict] = []  # LLM omits the finding entirely
        result = analyzer.apply_filter([finding], [(batch, llm_items)])

        assert len(result) == 0, "LOW finding must be dropped when LLM does not confirm it"

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_critical_confirmed_uses_llm_enrichment(self) -> None:
        """A CRITICAL finding confirmed by the LLM is still enriched as before.

        The floor does not interfere with the normal happy path: when the LLM
        confirms a CRITICAL/HIGH finding, the enriched version (with LLM
        explanation/remediation/confidence) is used and 'llm-unconfirmed' is
        NOT added.
        """
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding("CRIT-002", "CRITICAL", line=20)
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        llm_items = [
            {
                "pattern_id": "CRIT-002",
                "start_line": 20,
                "is_vulnerability": True,
                "confidence": 0.95,
                "explanation": "LLM-confirmed dangerous pattern",
                "remediation": "Remove immediately",
                "_file": "skill.md",
            }
        ]
        result = analyzer.apply_filter([finding], [(batch, llm_items)])

        assert len(result) == 1
        kept = result[0]
        assert kept.severity == "CRITICAL"
        assert kept.rule_id == "CRIT-002"
        assert kept.explanation == "LLM-confirmed dangerous pattern"
        assert kept.confidence == 0.95
        assert "llm-unconfirmed" not in kept.tags

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_llm_unconfirmed_tag_not_duplicated(self) -> None:
        """If the original finding already has 'llm-unconfirmed' in its tags,
        apply_filter must not append it again."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding(
            "CRIT-003", "CRITICAL", line=1, tags=["llm-unconfirmed", "existing-tag"]
        )
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        llm_items: list[dict] = []  # LLM omits finding
        result = analyzer.apply_filter([finding], [(batch, llm_items)])

        assert len(result) == 1
        assert result[0].tags.count("llm-unconfirmed") == 1

    @patch(MOCK_PATCH_TARGET, _mock_get_chat_model)
    def test_llm_unconfirmed_tag_surfaced_in_to_dict(self) -> None:
        """The 'llm-unconfirmed' marker must be visible in the JSON output
        (Finding.to_dict), so consumers can see a high-severity finding the LLM
        did not confirm."""
        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = self._make_finding("CRIT-004", "CRITICAL", line=1)
        batch = Batch(file_path="skill.md", content="code", findings=[finding])
        result = analyzer.apply_filter([finding], [(batch, [])])

        assert len(result) == 1
        assert "llm-unconfirmed" in result[0].to_dict()["tags"]


# ---------------------------------------------------------------------------


class _RawWithUsage:
    def __init__(self, usage_metadata: dict[str, int] | None = None) -> None:
        self.usage_metadata = usage_metadata


@patch("skillspector.llm_analyzer_base.get_chat_model")
def test_run_batches_records_token_usage_with_include_raw(mock_get_model: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_model.return_value = mock_llm
    response = LLMAnalysisResult(findings=[])
    mock_structured.invoke.return_value = {
        "raw": _RawWithUsage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        "parsed": response,
        "parsing_error": None,
    }
    analyzer = LLMAnalyzerBase(base_prompt="Analyze", model="test-model")
    results = analyzer.run_batches([Batch(file_path="a.py", content="code")])
    assert results == [(results[0][0], [])]
    mock_llm.with_structured_output.assert_called_once_with(LLMAnalysisResult, include_raw=True)
    assert analyzer.llm_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


@patch("skillspector.llm_analyzer_base.get_chat_model")
async def test_arun_batches_records_token_usage_with_include_raw(mock_get_model: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value={
            "raw": _RawWithUsage({"prompt_tokens": 7, "completion_tokens": 3}),
            "parsed": LLMAnalysisResult(findings=[]),
            "parsing_error": None,
        }
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_model.return_value = mock_llm
    analyzer = LLMAnalyzerBase(base_prompt="Analyze", model="test-model")
    await analyzer.arun_batches([Batch(file_path="a.py", content="code")])
    assert analyzer.llm_usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


@patch("skillspector.llm_analyzer_base.get_chat_model")
def test_run_batches_missing_usage_metadata_defaults_to_zero(mock_get_model: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = {
        "raw": object(),
        "parsed": LLMAnalysisResult(findings=[]),
        "parsing_error": None,
    }
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_model.return_value = mock_llm
    analyzer = LLMAnalyzerBase(base_prompt="Analyze", model="test-model")
    analyzer.run_batches([Batch(file_path="a.py", content="code")])
    assert analyzer.llm_usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# LLMMetaAnalyzer.run_batches (mocked LLM)
# ---------------------------------------------------------------------------


class TestLLMMetaAnalyzerRunBatches:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET)
    def test_run_batches_calls_structured_llm_per_batch(self, mock_get_model: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_get_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = MetaAnalyzerResult(
            findings=[
                MetaAnalyzerFinding(
                    pattern_id="E1",
                    is_vulnerability=True,
                    confidence=0.9,
                    intent="malicious",
                    impact="high",
                )
            ],
        )

        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        f1 = Finding(rule_id="E1", message="test", file="a.py", start_line=1)
        f2 = Finding(rule_id="E2", message="test", file="b.py", start_line=1)
        batches = [
            Batch(file_path="a.py", content="code a", findings=[f1]),
            Batch(file_path="b.py", content="code b", findings=[f2]),
        ]
        results = analyzer.run_batches(batches, metadata_text="Name: skill")
        assert mock_structured.invoke.call_count == 2
        assert len(results) == 2

    @patch(MOCK_PATCH_TARGET)
    def test_run_batches_propagates_value_error(self, mock_get_model: MagicMock) -> None:
        mock_get_model.side_effect = ValueError("No LLM API key configured.")
        with pytest.raises(ValueError, match="API key"):
            LLMMetaAnalyzer(model=self.MODEL)


# ---------------------------------------------------------------------------
# LLMMetaAnalyzer.arun_batches (async parallel execution)
# ---------------------------------------------------------------------------


class TestLLMMetaAnalyzerARunBatches:
    MODEL = "nvidia/openai/gpt-oss-120b"

    @patch(MOCK_PATCH_TARGET)
    async def test_arun_batches_calls_ainvoke_per_batch(self, mock_get_model: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_get_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.ainvoke = AsyncMock(
            return_value=MetaAnalyzerResult(
                findings=[
                    MetaAnalyzerFinding(
                        pattern_id="E1",
                        is_vulnerability=True,
                        confidence=0.9,
                        intent="malicious",
                        impact="high",
                    )
                ],
            )
        )

        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        f1 = Finding(rule_id="E1", message="test", file="a.py", start_line=1)
        f2 = Finding(rule_id="E2", message="test", file="b.py", start_line=1)
        batches = [
            Batch(file_path="a.py", content="code a", findings=[f1]),
            Batch(file_path="b.py", content="code b", findings=[f2]),
        ]
        results = await analyzer.arun_batches(batches, metadata_text="Name: skill")
        assert mock_structured.ainvoke.call_count == 2
        assert len(results) == 2

    @patch(MOCK_PATCH_TARGET)
    async def test_arun_batches_results_compatible_with_apply_filter(
        self,
        mock_get_model: MagicMock,
    ) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_get_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.ainvoke = AsyncMock(
            return_value=MetaAnalyzerResult(
                findings=[
                    MetaAnalyzerFinding(
                        pattern_id="E1",
                        is_vulnerability=True,
                        confidence=0.9,
                        intent="malicious",
                        impact="high",
                        explanation="Dangerous",
                        remediation="Fix it",
                    )
                ],
            )
        )

        analyzer = LLMMetaAnalyzer(model=self.MODEL)
        finding = Finding(rule_id="E1", message="test", file="a.py", start_line=1)
        batches = [Batch(file_path="a.py", content="code", findings=[finding])]
        batch_results = await analyzer.arun_batches(batches, metadata_text="")
        filtered = analyzer.apply_filter([finding], batch_results)
        assert len(filtered) == 1
        assert filtered[0].explanation == "Dangerous"


# ---------------------------------------------------------------------------
# constants.py: token budget functions
# ---------------------------------------------------------------------------


class TestTokenBudgetFunctions:
    def test_known_model(self) -> None:
        from skillspector.model_info import get_max_input_tokens, get_max_output_tokens

        inp = get_max_input_tokens("nvidia/openai/gpt-oss-120b")
        out = get_max_output_tokens("nvidia/openai/gpt-oss-120b")
        assert inp == int(131_072 * 0.75)
        assert out == int(131_072 * 0.25)

    def test_unknown_model_uses_default(self) -> None:
        """Unknown model uses the conftest-mocked context length (131_072)."""
        from skillspector.model_info import get_max_input_tokens, get_max_output_tokens

        mocked_ctx = 131_072
        inp = get_max_input_tokens("unknown/model")
        out = get_max_output_tokens("unknown/model")
        assert inp == int(mocked_ctx * 0.75)
        assert out == int(mocked_ctx * 0.25)
