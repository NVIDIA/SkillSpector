# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized prompt budgets must include every line and the real prompt wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
async def test_unfit_fixed_prompt_is_an_explicit_failure_without_provider_call(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    provider = MagicMock(side_effect=AssertionError("provider must not be reached"))
    monkeypatch.setattr(lab, "_invoke_with_usage", provider)
    monkeypatch.setattr(lab, "_ainvoke_with_usage", provider)
    # Long source lines can be fragmented; fixed prompt overhead cannot.
    analyzer.base_prompt = "Irreducible prompt metadata. " * 5000
    content = "x" * 10000
    batches = analyzer.get_batches(["long"], {"long": content})
    assert len(batches) == 1
    assert batches[0].content == content
    assert batches[0].source_start_column is None
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


def test_recognized_and_explicitly_configured_codec_is_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_openai import ChatOpenAI
    from tiktoken import registry

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = "gpt-4o"
    model.model_name = "custom-provider-deployment"
    codec = MagicMock()
    codec.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(registry, "ENCODINGS", {"o200k_base": codec})
    counter, source = lab._input_token_counter(model)
    text = "All source text stays data: <|endoftext|>"
    assert source == "recognized_tiktoken_codec"
    assert counter(text) == 3
    codec.encode.assert_called_once_with(text, disallowed_special=())


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


@pytest.mark.parametrize("failure", [OSError, ValueError, RuntimeError])
def test_codec_name_lookup_failure_uses_conservative_bytes(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    import tiktoken
    from langchain_openai import ChatOpenAI

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = "gpt-4o"
    model.model_name = "recognized-model"
    loader = MagicMock(side_effect=failure("offline or invalid tokenizer vocabulary"))
    monkeypatch.setattr(tiktoken, "encoding_name_for_model", loader)
    counter, source = lab._input_token_counter(model)
    assert source == "conservative_utf8_bytes"
    assert counter("漢🙂") == len("漢🙂".encode())
    loader.assert_called_once_with("gpt-4o")


@pytest.mark.parametrize("cached", [False, True])
def test_codec_selection_never_loads_or_downloads_vocabulary(
    monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    import requests
    import tiktoken
    from langchain_openai import ChatOpenAI
    from tiktoken import registry

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = "gpt-4o"
    model.model_name = "recognized-model"
    codec = MagicMock()
    codec.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(registry, "ENCODINGS", {"o200k_base": codec} if cached else {})
    loader = MagicMock(side_effect=AssertionError("Vocabulary loading/network is forbidden"))
    for owner, name in [
        (tiktoken, "encoding_for_model"),
        (tiktoken, "get_encoding"),
        (requests, "get"),
    ]:
        monkeypatch.setattr(owner, name, loader)
    counter, source = lab._input_token_counter(model)
    text = "漢🙂 source"
    assert counter(text) == (3 if cached else len(text.encode()))
    assert source == ("recognized_tiktoken_codec" if cached else "conservative_utf8_bytes")
    loader.assert_not_called()


def test_cached_only_constructor_cannot_consume_shared_deadline(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tiktoken
    from langchain_openai import ChatOpenAI
    from tiktoken import registry

    model = MagicMock(spec=ChatOpenAI)
    model.custom_get_token_ids = None
    model.tiktoken_model_name = "gpt-4o"
    model.model_name = "recognized-model"
    monkeypatch.setattr(registry, "ENCODINGS", {})
    clock = [0.0]

    def unbounded_loader(_label):
        clock[0] += 1000.0
        return MagicMock()

    loader = MagicMock(side_effect=unbounded_loader)
    monkeypatch.setattr(tiktoken, "encoding_for_model", loader)
    monkeypatch.setattr(lab, "get_chat_model", lambda **kwargs: model)
    created = lab.LLMAnalyzerBase("Review all lines", "gpt-4o", timeout=lambda: 1.0 - clock[0])
    assert created._remaining_timeout() == 1.0
    assert created._token_counter_source == "conservative_utf8_bytes"
    loader.assert_not_called()


def test_selected_fallback_is_reused_for_equivalent_request_clients(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_openai import ChatOpenAI
    from tiktoken import registry

    def client():
        model = MagicMock(spec=ChatOpenAI)
        model.custom_get_token_ids = None
        model.tiktoken_model_name = "gpt-4o"
        model.model_name = "recognized-model"
        return model

    model = client()
    monkeypatch.setattr(registry, "ENCODINGS", {})
    before = analyzer._count_prompt_tokens("漢🙂 source", chat_model=model)
    codec = MagicMock()
    codec.encode.return_value = [1]
    registry.ENCODINGS["o200k_base"] = codec
    assert analyzer._count_prompt_tokens("漢🙂 source", chat_model=model) == before
    assert analyzer._count_prompt_tokens("漢🙂 source", chat_model=client()) == before
    codec.encode.assert_not_called()


def test_large_aggregate_findings_do_not_create_one_request_per_line(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_prompt = analyzer.build_prompt

    def prompt(batch: lab.Batch, **kwargs: object) -> str:
        return (
            original_prompt(batch) + "\nFindings:\n" + "\n".join(f.message for f in batch.findings)
        )

    monkeypatch.setattr(analyzer, "build_prompt", prompt)
    content = "original source line\n" * 5000
    findings = [
        Finding(
            rule_id="fixture",
            message="Retained finding detail " * 60,
            severity="HIGH",
            file="source",
            start_line=line,
        )
        for line in range(1, 5001, 80)
    ]
    assert sum(len(f.message) for f in findings) > 4 * analyzer._input_budget
    batches = analyzer.get_batches(["source"], {"source": content}, findings)
    assert len(batches) < 150, (
        "Range-specific findings fit; aggregate metadata must not force one-line requests"
    )
    covered = set()
    retained = set()
    lines = content.splitlines(keepends=True)
    for batch in batches:
        end = batch.end_line or len(lines)
        assert batch.content == "".join(lines[batch.start_line - 1 : end])
        covered.update(range(batch.start_line, end + 1))
        retained.update(f.finding_id for f in batch.findings)
        analyzer._check_prompt_budget(analyzer.build_prompt(batch))
    assert covered == set(range(1, len(lines) + 1))
    assert retained == {f.finding_id for f in findings}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit_source", ["serialized-overhead", "refreshed-client"])
async def test_field_recovery_keeps_known_findings_when_serialized_request_is_refused(
    analyzer: lab.LLMAnalyzerBase,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    limit_source: str,
) -> None:
    """A completed answer can be retained without submitting an oversized recovery."""
    source = '{"findings":[{"rule_id":"SSD-1","severity":"HIGH","start_line":1}]}'
    cause = lab.StructuredOutputParseError(
        "response lacks a required field",
        issue=lab.ResponseIssue.INVALID_SCHEMA,
        recovery_text=source,
    )
    invalid = lab._StructuredResponseValidationError(cause.issue)
    invalid.__cause__ = cause
    analyzer.recovery_rule_ids = ("SSD-1",)
    if limit_source == "serialized-overhead":
        # The initial prompt fits, while repeating it with recovery instructions
        # exceeds the real byte/token bound despite the old chars/4 estimate.
        prompt = "x" * (analyzer._input_budget - analyzer._count_prompt_tokens("") - 1)
        assert analyzer._prompt_fits_budget(prompt)
    else:
        from langchain_openai import ChatOpenAI

        prompt = "Assess the original source."
        monkeypatch.setattr(analyzer, "_token_counter", lambda text: 1)
        model = MagicMock(spec=ChatOpenAI)
        model.custom_get_token_ids = lambda text: range(analyzer._input_budget + 1)
        model.tiktoken_model_name = None
        model.model_name = "refreshed-recovery-client"
        monkeypatch.setattr(analyzer, "_model_for_call", lambda: (model, None))
    monkeypatch.setattr(analyzer, "build_prompt", lambda batch, **kwargs: prompt)
    rejected = AsyncMock(side_effect=invalid) if asynchronous else MagicMock(side_effect=invalid)
    monkeypatch.setattr(analyzer, "_ainvoke_batch" if asynchronous else "_invoke_batch", rejected)
    provider = AsyncMock() if asynchronous else MagicMock()
    monkeypatch.setattr(
        lab, "_ainvoke_with_usage" if asynchronous else "_invoke_with_usage", provider
    )
    batch = lab.Batch(file_path="source.py", content="original source")

    outcome = (
        await analyzer.arun_batches_detailed([batch], max_concurrency=1)
        if asynchronous
        else analyzer.run_batches_detailed([batch])
    )

    assert not outcome.successful and not outcome.warnings
    assert len(outcome.failures) == 1
    assert outcome.failures[0].error_class == "LLMInputBudgetError"
    assert outcome.failures[0].reason is lab.LedgerReason.LLM_BATCH_FAILED
    assert len(outcome.recovered) == 1
    retained_batch, findings = outcome.recovered[0]
    assert retained_batch is batch
    assert len(findings) == 1
    assert (findings[0].rule_id, findings[0].severity) == ("SSD-1", "HIGH")
    assert findings[0].evidence["assessment_complete"] is False
    assert "llm-format-recovery" in findings[0].tags
    provider.assert_not_called()
    rejected.assert_called_once()
    assert analyzer._request_ordinal == 0


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_in_budget_field_recovery_preserves_diagnostics_and_partial_warning(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    """Permitted raw recovery keeps the existing diagnostic and warning contracts."""
    from langchain_core.messages import AIMessage

    source = '{"findings":[{"rule_id":"SSD-1","severity":"HIGH","start_line":1}]}'
    cause = lab.StructuredOutputParseError(
        "response lacks a required field",
        issue=lab.ResponseIssue.INVALID_SCHEMA,
        recovery_text=source,
    )
    invalid = lab._StructuredResponseValidationError(cause.issue)
    invalid.__cause__ = cause
    analyzer.recovery_rule_ids = ("SSD-1",)
    monkeypatch.setattr(analyzer, "build_prompt", lambda batch, **kwargs: "Assess this source.")
    rejected = AsyncMock(side_effect=invalid) if asynchronous else MagicMock(side_effect=invalid)
    monkeypatch.setattr(analyzer, "_ainvoke_batch" if asynchronous else "_invoke_batch", rejected)
    response = AIMessage(content="UNKNOWN", response_metadata={"finish_reason": "stop"})
    provider = (
        AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    )
    monkeypatch.setattr(
        lab, "_ainvoke_with_usage" if asynchronous else "_invoke_with_usage", provider
    )
    name = "_ainvoke_with_failure_diagnostic" if asynchronous else "_invoke_with_failure_diagnostic"
    diagnostic = (
        AsyncMock(wraps=getattr(analyzer, name))
        if asynchronous
        else MagicMock(wraps=getattr(analyzer, name))
    )
    monkeypatch.setattr(analyzer, name, diagnostic)
    batch = lab.Batch(file_path="source.py", content="original source")

    outcome = (
        await analyzer.arun_batches_detailed([batch], max_concurrency=1)
        if asynchronous
        else analyzer.run_batches_detailed([batch])
    )

    assert not outcome.successful and not outcome.failures
    assert len(outcome.warnings) == 1
    assert outcome.warnings[0].reason is lab.LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    assert len(outcome.recovered) == 1
    assert outcome.recovered[0][0] is batch
    assert outcome.recovered[0][1][0].severity == "HIGH"
    assert outcome.recovered[0][1][0].evidence["assessment_complete"] is False
    assert provider.call_count > 0
    assert diagnostic.call_count == provider.call_count == analyzer._request_ordinal
