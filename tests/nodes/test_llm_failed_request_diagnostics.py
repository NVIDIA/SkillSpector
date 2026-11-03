# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed request retains actionable metadata without leaking its contents."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_openai.chat_models.base import OpenAIContextOverflowError

import skillspector.llm_analyzer_base as lab

CANARY = "PRIVATE_SOURCE_AUTH_BODY_CANARY_sk-not-a-credential-123456789"
REQUEST_ID = "req_0123456789abcdef0123456789abcdef"


@pytest.fixture
def analyzer(monkeypatch: pytest.MonkeyPatch) -> lab.LLMAnalyzerBase:
    model = MagicMock()
    model.max_tokens = 128000
    monkeypatch.setattr(lab, "get_max_input_tokens", lambda model: 787500)
    monkeypatch.setattr(lab, "get_chat_model", lambda **kwargs: model)
    monkeypatch.setattr(lab, "_uses_native_connection_retries", lambda *args, **kwargs: False)
    monkeypatch.setattr(lab, "new_inference_usage_collector", lambda **kwargs: MagicMock())
    monkeypatch.setattr(lab.LLMAnalyzerBase, "_bind_structured_output", lambda *args: MagicMock())
    return lab.LLMAnalyzerBase("Review the source.", "fixture")


def overflow() -> OpenAIContextOverflowError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        400, request=request, headers={"x-request-id": REQUEST_ID, "authorization": CANARY}
    )
    return OpenAIContextOverflowError(
        CANARY,
        response=response,
        body={"code": "context_length_exceeded", "message": CANARY, "prompt": CANARY},
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_invoke_seam_preserves_safe_metadata_and_original_exception(
    analyzer: lab.LLMAnalyzerBase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    asynchronous: bool,
) -> None:
    error = overflow()
    provider = AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error)
    monkeypatch.setattr(
        lab, "_ainvoke_with_usage" if asynchronous else "_invoke_with_usage", provider
    )
    batch = lab.Batch(file_path="fixture", content=CANARY)
    with caplog.at_level("WARNING"), pytest.raises(OpenAIContextOverflowError) as caught:
        if asynchronous:
            await analyzer._ainvoke_batch(batch, CANARY)
        else:
            analyzer._invoke_batch(batch, CANARY)
    assert caught.value is error
    entries = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM_REQUEST_FAILURE ")
    ]
    assert len(entries) == 1
    value = json.loads(entries[0].removeprefix("LLM_REQUEST_FAILURE "))
    assert value["error_class"] == "OpenAIContextOverflowError"
    assert value["status_code"] == 400
    assert value["provider_code"] == "context_length_exceeded"
    assert value["message_class"] == "context_overflow"
    assert value["request_id"] == REQUEST_ID
    assert value["estimated_input_budget"] == 787500
    assert value["configured_output_cap"] == 128000
    assert value["prompt_characters"] == len(CANARY)
    assert value["request_ordinal"] == 1
    assert CANARY not in caplog.text
    assert "authorization" not in caplog.text
    assert "api.openai.com" not in caplog.text
    analyzer._usage_collector.mark_response_received.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_broken_diagnostics_cannot_replace_provider_exception(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    error = overflow()
    provider = AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error)
    monkeypatch.setattr(
        lab, "_ainvoke_with_usage" if asynchronous else "_invoke_with_usage", provider
    )
    monkeypatch.setattr(
        lab, "_safe_failed_request_diagnostic", MagicMock(side_effect=ValueError(CANARY))
    )
    with pytest.raises(OpenAIContextOverflowError) as caught:
        if asynchronous:
            await analyzer._ainvoke_with_failure_diagnostic(object(), CANARY)
        else:
            analyzer._invoke_with_failure_diagnostic(object(), CANARY)
    assert caught.value is error


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_success_value_and_collector_are_unchanged(
    analyzer: lab.LLMAnalyzerBase, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    value, runnable = object(), object()
    provider = AsyncMock(return_value=value) if asynchronous else MagicMock(return_value=value)
    monkeypatch.setattr(
        lab, "_ainvoke_with_usage" if asynchronous else "_invoke_with_usage", provider
    )
    if asynchronous:
        actual = await analyzer._ainvoke_with_failure_diagnostic(runnable, CANARY)
    else:
        actual = analyzer._invoke_with_failure_diagnostic(runnable, CANARY)
    assert actual is value
    provider.assert_called_once_with(runnable, CANARY, analyzer._usage_collector)
    assert analyzer._request_ordinal == 1


@pytest.mark.parametrize(
    "bad", [CANARY, "req_" + CANARY, "https://private.invalid", "a" * 1000, None]
)
def test_untrusted_fields_are_not_copied(bad: object) -> None:
    error = overflow()
    error.request_id = bad
    error.response.headers.pop("x-request-id")
    error.code = bad
    error.body = {"error": {"code": bad, "message": CANARY}}
    value = lab._safe_failed_request_diagnostic(error, CANARY, True, CANARY, 1)
    assert value["request_id"] is None
    assert value["provider_code"] is None
    assert value["estimated_input_budget"] is None
    assert value["configured_output_cap"] is None
    assert CANARY not in json.dumps(value)


@pytest.mark.parametrize(
    "error_type", [TimeoutError, asyncio.TimeoutError], ids=["builtin", "asyncio_alias"]
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_builtin_timeout_keeps_class_category_and_original_exception(
    analyzer: lab.LLMAnalyzerBase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
    asynchronous: bool,
) -> None:
    error = error_type(CANARY)
    provider = AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error)
    monkeypatch.setattr(
        lab, "_ainvoke_with_usage" if asynchronous else "_invoke_with_usage", provider
    )
    batch = lab.Batch(file_path="fixture", content=CANARY)
    with caplog.at_level("WARNING"), pytest.raises(TimeoutError) as caught:
        if asynchronous:
            await analyzer._ainvoke_batch(batch, CANARY)
        else:
            analyzer._invoke_batch(batch, CANARY)
    assert caught.value is error
    entries = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM_REQUEST_FAILURE ")
    ]
    assert len(entries) == 1
    value = json.loads(entries[0].removeprefix("LLM_REQUEST_FAILURE "))
    assert value["error_class"] == "TimeoutError"
    assert value["message_class"] == "timeout"
    assert value["request_ordinal"] == 1
    assert CANARY not in caplog.text
    analyzer._usage_collector.mark_response_received.assert_not_called()
