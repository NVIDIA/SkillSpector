# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized prompt budgets must include every line and the real prompt wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import skillspector.llm_analyzer_base as lab
from skillspector.models import Finding


@pytest.fixture
def analyzer(monkeypatch: pytest.MonkeyPatch) -> lab.LLMAnalyzerBase:
    """Construct the real planner with an inert provider; no model credentials or calls."""
    monkeypatch.setattr(lab, "get_max_input_tokens", lambda model: 8192)
    monkeypatch.setattr(lab, "get_chat_model", lambda **kwargs: MagicMock())
    monkeypatch.setattr(lab, "_uses_native_connection_retries", lambda *args, **kwargs: False)
    monkeypatch.setattr(lab, "new_inference_usage_collector", lambda **kwargs: MagicMock())
    monkeypatch.setattr(lab.LLMAnalyzerBase, "_bind_structured_output", lambda *args: MagicMock())
    monkeypatch.setattr(lab, "_cli_max_input_bytes", lambda: None)
    return lab.LLMAnalyzerBase("Review every supplied line.", "fixture")


def test_short_line_rounding_does_not_exceed_chunk_budget() -> None:
    chunks = lab.chunk_file_by_lines("ab\nab\nab", 1)
    assert all(lab.estimate_tokens(text) <= 1 for text, _, _ in chunks)
    assert {line for _, start, end in chunks for line in range(start, end + 1)} == {1, 2, 3}


@pytest.mark.parametrize("line", ["ab\n", "ab\r\n", "漢字🙂\n"])
@pytest.mark.parametrize("compact", [False, True])
def test_real_serialized_prompt_fits_and_keeps_every_line(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch, line: str, compact: bool
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_COMPACT_PROMPTS", "1" if compact else "0")
    content = line * 5000
    batches = analyzer.get_batches(["source.js"], {"source.js": content})
    source_lines = content.splitlines(keepends=True)
    covered: set[int] = set()
    for batch in batches:
        prompt = analyzer.build_prompt(batch)
        if analyzer.response_schema is not None:
            prompt += lab.structured_output_instructions(analyzer.response_schema)
        assert len(prompt.encode("utf-8")) + lab.REQUEST_FRAMING_TOKENS <= analyzer._input_budget
        end = batch.end_line or len(source_lines)
        assert batch.content == "".join(source_lines[batch.start_line - 1 : end])
        covered.update(range(batch.start_line, end + 1))
    assert covered == set(range(1, len(source_lines) + 1))
    assert len(batches) < len(source_lines)


def test_real_custom_findings_and_metadata_overhead_is_measured(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = analyzer.build_prompt

    def custom_prompt(batch: lab.Batch, **kwargs: object) -> str:
        return (
            original(batch) + "\nMetadata: " + "m" * 1000 + "\n" + "detail " * len(batch.findings)
        )

    monkeypatch.setattr(analyzer, "build_prompt", custom_prompt)
    finding = Finding(
        rule_id="fixture", message="fixture", severity="LOW", file="source.js", start_line=1
    )
    batches = analyzer.get_batches(["source.js"], {"source.js": "line\n" * 4000}, [finding] * 100)
    for batch in batches:
        prompt = analyzer.build_prompt(batch) + lab.structured_output_instructions(
            analyzer.response_schema
        )
        assert len(prompt.encode("utf-8")) + lab.REQUEST_FRAMING_TOKENS <= analyzer._input_budget


def test_empty_file_stays_empty(analyzer: lab.LLMAnalyzerBase) -> None:
    assert lab.chunk_file_by_lines("", 1) == [("", 1, 1)]
    batches = analyzer.get_batches(["empty"], {"empty": ""})
    assert len(batches) == 1
    assert batches[0].content == ""


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unsplittable_line_is_an_explicit_failure_without_provider_call(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    provider = MagicMock(side_effect=AssertionError("provider must not be reached"))
    monkeypatch.setattr(lab, "_invoke_with_usage", provider)
    monkeypatch.setattr(lab, "_ainvoke_with_usage", provider)
    content = "x" * 10000
    batches = analyzer.get_batches(["long"], {"long": content})
    assert len(batches) == 1
    assert batches[0].content == content
    if asynchronous:
        outcome = await analyzer.arun_batches_detailed(batches, max_concurrency=1)
    else:
        outcome = analyzer.run_batches_detailed(batches)
    assert not outcome.successful
    assert len(outcome.failures) == 1
    assert outcome.failures[0].error_class == "LLMInputBudgetError"
    assert outcome.failures[0].reason is lab.LedgerReason.LLM_BATCH_FAILED
    provider.assert_not_called()


def test_unknown_openai_codec_uses_conservative_bytes() -> None:
    from langchain_openai import ChatOpenAI

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = None
    model.model_name = "unknown-future-model"
    counter, source = lab._input_token_counter(model)
    assert source == "conservative_utf8_bytes"
    assert counter("漢🙂") == len("漢🙂".encode())


def test_recognized_and_explicitly_configured_codec_is_respected() -> None:
    import tiktoken
    from langchain_openai import ChatOpenAI

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = "gpt-4o"
    model.model_name = "custom-provider-deployment"
    counter, source = lab._input_token_counter(model)
    text = "All source text stays data: <|endoftext|>"
    assert source == "recognized_tiktoken_codec"
    assert counter(text) == len(
        tiktoken.encoding_for_model("gpt-4o").encode(text, disallowed_special=())
    )


def test_explicit_custom_codec_is_respected_and_failure_is_conservative() -> None:
    from langchain_openai import ChatOpenAI

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = lambda text: [1, 2, 3]
    counter, source = lab._input_token_counter(model)
    assert source == "configured_chat_model_tokenizer"
    assert counter("source") == 3
    model.custom_get_token_ids = MagicMock(side_effect=ValueError("codec unavailable"))
    counter, _ = lab._input_token_counter(model)
    assert counter("漢🙂") == len("漢🙂".encode())


def test_actual_request_client_is_rechecked_when_its_codec_differs(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_openai import ChatOpenAI

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = None
    model.model_name = "unknown-runtime-model"
    monkeypatch.setattr(analyzer, "_model_for_call", lambda: (model, None))
    monkeypatch.setattr(analyzer, "_token_counter", lambda text: 1)
    provider = MagicMock(side_effect=AssertionError("provider must not be reached"))
    monkeypatch.setattr(lab, "_invoke_with_usage", provider)
    batch = lab.Batch(file_path="source", content="x" * 10000)
    with pytest.raises(lab.LLMInputBudgetError):
        analyzer._invoke_batch(batch, analyzer.build_prompt(batch))
    provider.assert_not_called()


def test_unfit_fixed_wrapper_retains_one_batch_without_per_line_refinement(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = analyzer.build_prompt
    rendered = MagicMock(side_effect=lambda batch, **kwargs: "metadata" * 10000 + original(batch))
    monkeypatch.setattr(analyzer, "build_prompt", rendered)
    content = "source line\n" * 10000
    batches = analyzer.get_batches(["source"], {"source": content})
    assert len(batches) == 1
    assert batches[0].content == content
    assert rendered.call_count <= 3
    with pytest.raises(lab.LLMInputBudgetError):
        analyzer._check_prompt_budget(analyzer.build_prompt(batches[0]))
