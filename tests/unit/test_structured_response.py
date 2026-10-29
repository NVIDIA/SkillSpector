# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completed answers may be reformatted only without changing their contents."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from skillspector.inspection_ledger import LedgerReason
from skillspector.llm_analyzer_base import Batch, LLMAnalysisResult, LLMAnalyzerBase
from skillspector.llm_utils import bind_structured_output, register_chat_model_provider
from skillspector.nodes.analyzers.mcp_tool_poisoning import _TP4AnalysisResult, _TP4Analyzer
from skillspector.nodes.analyzers.semantic_security_discovery import node as security_discovery
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer, MetaAnalyzerResult
from skillspector.providers import reset_provider, use_provider
from skillspector.providers.openai import OpenAIProvider
from skillspector.structured_response import (
    MAX_RESPONSE_DEPTH,
    ResponseIssue,
    StructuredOutputParseError,
    formatting_tokens,
    validate_message,
    validate_response_text,
)


@pytest.mark.parametrize("schema", [LLMAnalysisResult, MetaAnalyzerResult])
def test_absent_findings_never_becomes_a_clean_result(schema):
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text("{}", schema)
    assert error.value.issue == ResponseIssue.MISSING_FIELDS
    assert validate_response_text('{"findings": []}', schema).findings == []


@pytest.mark.parametrize(
    "text",
    [
        '{"findings": [], "findings": []}',
        '{"findings": [{"severity":"HIGH", "severity":"LOW"}]}',
    ],
)
def test_duplicate_fields_are_never_silently_discarded(text):
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text(text, LLMAnalysisResult, completed=True)
    assert error.value.repair_text is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        '{"findings": [',
        '{"findings": [], "explanation": "unfinished',
        '{"findings": []} {"findings": []}',
        '{"findings": []} trailing prose',
        '{"findings": NaN}',
        '{"findings": Infinity}',
        '{"findings": 1e999}',
        "No problems found.",
        '{"findings": [}]}',
        '{"findings": ' + "[" * MAX_RESPONSE_DEPTH + "]" * MAX_RESPONSE_DEPTH + "}",
    ],
)
def test_incomplete_ambiguous_or_unbounded_text_is_not_repairable(text):
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text(text, LLMAnalysisResult, completed=True)
    assert error.value.repair_text is None


@pytest.mark.parametrize("completed", [False, True])
def test_only_complete_answers_with_all_values_intact_can_be_reformatted(completed):
    source = '{"findings": [],}'
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text(source, LLMAnalysisResult, completed=completed)
    assert error.value.repair_text == (source if completed else None)


@pytest.mark.parametrize(
    "changed",
    [
        '{"findings": []}',
        '{"findings": [{"severity": "LOW", "message": "secret", "line": 2}]}',
        '{"findings": [{"severity": "HIGH", "message": "secret", "line": 3}]}',
        '{"findings": [{"severity": "HIGH", "message": "different", "line": 2}]}',
    ],
)
def test_formatting_cannot_drop_or_change_a_finding(changed):
    source = '{"findings": [{"severity":"HIGH", "message":"secret", "line":2}],}'
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text(changed, LLMAnalysisResult, preserve_text=source)
    assert error.value.issue == ResponseIssue.FORMAT_CHANGED


def test_escaped_unicode_and_punctuation_inside_strings_are_preserved():
    assert formatting_tokens('{"a": "é,}\\"",}') == formatting_tokens('{"a":"\\u00e9,}\\""}')
    assert formatting_tokens('{"a": 1,}') != formatting_tokens('{"a": true}')


@pytest.mark.parametrize(
    "stop,extra,issue",
    [
        ("length", {}, ResponseIssue.TRUNCATED),
        ("max_tokens", {}, ResponseIssue.TRUNCATED),
        ("content_filter", {}, ResponseIssue.REFUSED),
        ("stop", {"refusal": "refused"}, ResponseIssue.REFUSED),
    ],
)
def test_stop_evidence_overrides_an_apparently_valid_clean_answer(stop, extra, issue):
    message = AIMessage(
        content='{"findings": []}',
        response_metadata={"finish_reason": stop},
        additional_kwargs=extra,
    )
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(message, LLMAnalysisResult, tool="LLMAnalysisResult")
    assert error.value.issue == issue
    assert not error.value.retryable


@pytest.fixture
def http_model(monkeypatch):
    """Actual SDK and LangChain parsing, with an in-memory HTTP transport."""
    clients = []
    async_clients = []
    provider_token = use_provider(OpenAIProvider())
    monkeypatch.delenv("SKILLSPECTOR_REASONING_EFFORT", raising=False)
    monkeypatch.setattr("skillspector.llm_analyzer_base.time.sleep", lambda _: None)

    def factory(
        answers,
        *,
        method="json_schema",
        model="test-model",
        after_response=None,
        reasoning_effort=None,
    ):
        if method is None:
            monkeypatch.delenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", raising=False)
        else:
            monkeypatch.setenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", method)
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            content, stop = answers[min(len(requests) - 1, len(answers) - 1)]
            message = {"role": "assistant", "content": content}
            if body.get("tools"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "assessment",
                            "type": "function",
                            "function": {
                                "name": body["tools"][0]["function"]["name"],
                                "arguments": content,
                            },
                        }
                    ],
                }
                if stop == "stop":
                    stop = "tool_calls"
            if after_response is not None:
                after_response(len(requests))
            return httpx.Response(
                200,
                json={
                    "id": "test-assessment",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "finish_reason": stop, "message": message}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                request=request,
            )

        client = httpx.Client(transport=httpx.MockTransport(respond))
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        async_clients.append(async_client)
        llm = ChatOpenAI(
            model=model,
            api_key="test",
            http_client=client,
            http_async_client=async_client,
            max_retries=0,
            reasoning_effort=reasoning_effort,
        )
        register_chat_model_provider(llm, "openai")
        monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", lambda **_: llm)
        return requests

    yield factory
    reset_provider(provider_token)
    for client in clients:
        client.close()
    for client in async_clients:
        asyncio.run(client.aclose())


def make_analyzer(name, *, timeout=None, model="test-model"):
    if name == "tp4":
        return _TP4Analyzer(model, timeout=timeout)
    if name == "review":
        return LLMMetaAnalyzer(model=model, timeout=timeout)
    return LLMAnalyzerBase(base_prompt="Assess the code", node=name, model=model, timeout=timeout)


@pytest.mark.parametrize(
    "name",
    [
        "semantic_security_discovery",
        "semantic_quality_policy",
        "semantic_developer_intent",
        "tp4",
        "review",
    ],
)
@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_five_checks_can_use_one_lossless_formatting_call(
    http_model, name, method, asynchronous
):
    payload = {"is_mismatch": False} if name == "tp4" else {"findings": []}
    answer = json.dumps(payload)
    requests = http_model([(answer[:-1] + ",}", "stop"), (answer, "stop")], method=method)
    analyzer = make_analyzer(name)
    # Keep the real shared invocation/validation; this test isolates response handling from prompt construction.
    analyzer.build_prompt = lambda *_args, **_kwargs: "Assess the code"
    batches = [Batch(file_path="code.py", content="print('hello')")]
    outcome = (
        asyncio.run(analyzer.arun_batches_detailed(batches))
        if asynchronous
        else analyzer.run_batches_detailed(batches)
    )
    assert not outcome.failures
    assert len(outcome.successful) == 1
    assert len(requests) == 2
    assert "Repair JSON punctuation" in requests[1]["messages"][0]["content"]
    assert sum(record.get("total_tokens", 0) or 0 for record in analyzer.inference_usage) == 30


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_already_valid_results_do_not_cost_a_second_call(http_model, method):
    requests = http_model([('{"findings": []}', "stop")], method=method)
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert outcome.successful and not outcome.failures
    assert len(requests) == 1


@pytest.mark.parametrize("stop", ["length", "content_filter"])
def test_sdk_termination_never_triggers_formatting_or_retries(http_model, stop):
    requests = http_model([('{"findings": []}', stop)])
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert not outcome.successful
    assert outcome.failures[0].reason == (
        LedgerReason.LLM_RESPONSE_TRUNCATED
        if stop == "length"
        else LedgerReason.LLM_RESPONSE_REFUSED
    )
    assert len(requests) == 1


def test_formatting_cannot_turn_a_finding_into_a_clean_result(http_model):
    source = '{"findings":[{"rule_id":"SSD-1","message":"leaks data","severity":"HIGH","start_line":1}],}'
    requests = http_model([(source, "stop"), ('{"findings": []}', "stop")])
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert not outcome.successful
    assert outcome.failures[0].reason == LedgerReason.LLM_FORMAT_CHANGED
    assert len(requests) == 2


def test_formatting_uses_the_original_deadline(http_model):
    remaining = [10.0]
    requests = http_model(
        [('{"findings": [],}', "stop")], after_response=lambda _: remaining.__setitem__(0, 0.0)
    )
    outcome = make_analyzer("discovery", timeout=lambda: remaining[0]).run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert not outcome.successful
    assert outcome.failures[0].reason == LedgerReason.RUNTIME_LIMIT
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_timeout_during_formatting_reports_the_exhausted_deadline(http_model, asynchronous):
    remaining = [10.0]

    def expire_during_formatting(count):
        if count == 2:
            remaining[0] = 0.0
            raise httpx.ReadTimeout("formatting exhausted the remaining time")

    requests = http_model([('{"findings": [],}', "stop")], after_response=expire_during_formatting)
    analyzer = make_analyzer("discovery", timeout=lambda: remaining[0])
    batches = [Batch(file_path="a.py", content="x")]
    outcome = (
        asyncio.run(analyzer.arun_batches_detailed(batches))
        if asynchronous
        else analyzer.run_batches_detailed(batches)
    )
    assert not outcome.successful
    assert outcome.failures[0].reason == LedgerReason.RUNTIME_LIMIT
    assert len(requests) == 2


def test_known_gateway_wrapper_is_unwrapped_without_forcing_tools_or_changing_reasoning(http_model):
    requests = http_model(
        [('{"json": {"findings": []}}', "stop")],
        model="azure/anthropic/claude-opus-5",
        reasoning_effort="high",
    )
    analyzer = make_analyzer("discovery", model="azure/anthropic/claude-opus-5")
    outcome = analyzer.run_batches_detailed([Batch(file_path="a.py", content="x")])
    assert outcome.successful and not outcome.failures
    assert len(requests) == 1
    assert "tool_choice" not in requests[0]
    assert requests[0]["reasoning_effort"] == "high"


def test_security_discovery_really_runs_independent_pieces_together(monkeypatch):
    monkeypatch.setenv("SKILLSPECTOR_MAX_LLM_CONCURRENCY", "2")
    active = 0
    peak = 0
    started = 0

    async def assess(*_args, **_kwargs):
        nonlocal active, peak, started
        active += 1
        started += 1
        peak = max(peak, active)
        try:
            # Sequential execution cannot satisfy this barrier.
            async with asyncio.timeout(1):
                while started < 2:
                    await asyncio.sleep(0)
            await asyncio.sleep(0)
            return LLMAnalysisResult(findings=[])
        finally:
            active -= 1

    model = MagicMock()
    model.with_structured_output.return_value = MagicMock(ainvoke=AsyncMock(side_effect=assess))
    monkeypatch.setattr("skillspector.llm_analyzer_base.get_chat_model", lambda **_: model)
    state = {
        "components": ["a.py", "b.py", "c.py"],
        "file_cache": dict.fromkeys(["a.py", "b.py", "c.py"], "x=1"),
    }
    result = security_discovery(state)
    assert peak == 2
    assert started == 3
    assert result["llm_call_log"][0]["ok"]


@pytest.mark.parametrize(
    "metadata,issue",
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            ResponseIssue.TRUNCATED,
        ),
        (
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
            ResponseIssue.REFUSED,
        ),
        ({"status": "in_progress"}, ResponseIssue.INCOMPLETE),
        ({"stop_reason": "pause_turn"}, ResponseIssue.INCOMPLETE),
    ],
)
def test_response_status_cannot_hide_an_incomplete_assessment(metadata, issue):
    message = AIMessage(content='{"findings": []}', response_metadata=metadata)
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(message, LLMAnalysisResult, tool="LLMAnalysisResult")
    assert error.value.issue == issue
    assert error.value.repair_text is None


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_formatting_preserves_real_findings_and_source_locations(http_model, method, asynchronous):
    payload = {
        "findings": [
            {
                "rule_id": "SSD-1",
                "message": "Leaks private data",
                "severity": "HIGH",
                "start_line": 42,
                "end_line": 44,
                "confidence": 0.95,
                "explanation": "User data is sent to an undeclared destination.",
                "remediation": "Remove the undeclared upload.",
            }
        ]
    }
    answer = json.dumps(payload)
    requests = http_model([(answer[:-1] + ",}", "stop"), (answer, "stop")], method=method)
    analyzer = make_analyzer("discovery")
    batches = [Batch(file_path="a.py", content="x", start_line=40, end_line=45)]
    outcome = (
        asyncio.run(analyzer.arun_batches_detailed(batches))
        if asynchronous
        else analyzer.run_batches_detailed(batches)
    )
    assert not outcome.failures
    finding = outcome.successful[0][1][0]
    assert (finding.file, finding.start_line, finding.end_line) == ("a.py", 42, 44)
    assert finding.severity == "HIGH"
    assert finding.confidence == 0.95
    assert finding.explanation == payload["findings"][0]["explanation"]
    assert len(requests) == 2


@pytest.mark.parametrize(
    "first,reason",
    [
        ("{}", LedgerReason.LLM_RESPONSE_MISSING_FIELDS),
        ('{"findings": "missing"}', LedgerReason.LLM_RESPONSE_INVALID_SCHEMA),
        (
            '{"findings": [], "hidden_findings": ["danger"]}',
            LedgerReason.LLM_RESPONSE_INVALID_SCHEMA,
        ),
        ('{"json": {"findings": []}}', LedgerReason.LLM_RESPONSE_MISSING_FIELDS),
    ],
)
def test_missing_or_extra_assessment_data_is_not_sent_for_formatting(http_model, first, reason):
    requests = http_model([(first, "stop")])
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert not outcome.successful
    assert outcome.failures[0].reason == reason
    assert len(requests) == 4
    assert all("Repair JSON punctuation" not in r["messages"][0]["content"] for r in requests)


def test_formatting_rechecks_the_input_allowance(http_model):
    requests = http_model([('{"findings": [],}', "stop")])
    analyzer = make_analyzer("discovery")
    analyzer._input_budget = 1
    outcome = analyzer.run_batches_detailed([Batch(file_path="a.py", content="x")])
    assert outcome.failures[0].reason == LedgerReason.LLM_RESPONSE_LIMIT
    assert len(requests) == 1


@pytest.mark.parametrize(
    "second,stop", [('{"findings": [],}', "stop"), ('{"findings": []}', "length")]
)
def test_a_failed_formatting_call_is_not_repeated(http_model, second, stop):
    requests = http_model([('{"findings": [],}', "stop"), (second, stop)])
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert not outcome.successful
    assert len(outcome.failures) == 1
    assert len(requests) == 2


def test_unknown_completion_state_does_not_authorize_formatting(http_model):
    requests = http_model([('{"findings": [],}', None), ('{"findings": []}', "stop")])
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert outcome.successful and not outcome.failures
    assert len(requests) == 2
    assert "Repair JSON punctuation" not in requests[1]["messages"][0]["content"]


def test_queued_pieces_cannot_restart_the_scan_clock(http_model, monkeypatch):
    monkeypatch.setenv("SKILLSPECTOR_MAX_LLM_CONCURRENCY", "1")
    remaining = [10.0]
    requests = http_model(
        [('{"findings": []}', "stop")], after_response=lambda _: remaining.__setitem__(0, 0.0)
    )
    analyzer = make_analyzer("discovery", timeout=lambda: remaining[0])
    batches = [Batch(file_path=f"{index}.py", content="x") for index in range(3)]
    outcome = asyncio.run(analyzer.arun_batches_detailed(batches))
    assert len(requests) == 1
    assert len(outcome.successful) == 1
    assert len(outcome.failures) == 2
    assert all(f.reason == LedgerReason.RUNTIME_LIMIT for f in outcome.failures)
    assert {b.file_path for b, _ in outcome.successful} | {
        f.batch.file_path for f in outcome.failures
    } == {"0.py", "1.py", "2.py"}


@pytest.mark.parametrize("schema", [LLMAnalysisResult, MetaAnalyzerResult, _TP4AnalysisResult])
def test_preserving_raw_answers_does_not_change_the_wire_schema(schema, monkeypatch):
    monkeypatch.setenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", "json_schema")
    requests = []
    payload = {"is_mismatch": False} if schema is _TP4AnalysisResult else {"findings": []}

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(payload)},
                    }
                ],
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = ChatOpenAI(model="test-model", api_key="test", http_client=client)
        llm.with_structured_output(schema).invoke("Assess")
        bind_structured_output(llm, schema, "test-model").invoke("Assess")
    assert requests[0]["response_format"] == requests[1]["response_format"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_responses_api_keeps_original_tool_arguments_for_validation(monkeypatch, duplicate):
    monkeypatch.setenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", "function_calling")
    arguments = '{"findings": [], "findings": []}' if duplicate else '{"findings": []}'

    def respond(request):
        assert request.url.path == "/v1/responses"
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "status": "completed",
                "error": None,
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_test",
                        "call_id": "call_test",
                        "name": "LLMAnalysisResult",
                        "arguments": arguments,
                        "status": "completed",
                    }
                ],
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = ChatOpenAI(
            model="test-model", api_key="test", http_client=client, use_responses_api=True
        )
        runnable = bind_structured_output(llm, LLMAnalysisResult, "test-model")
        if duplicate:
            with pytest.raises(StructuredOutputParseError) as error:
                runnable.invoke("Assess")
            assert error.value.issue == ResponseIssue.INVALID_SCHEMA
        else:
            assert runnable.invoke("Assess").findings == []
