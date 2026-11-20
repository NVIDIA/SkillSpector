# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completed answers may be reformatted only without changing their contents."""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Annotated, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage

from skillspector.inspection_ledger import LedgerReason
from skillspector.llm_analyzer_base import (
    FIELD_RECOVERY_MAX_CALLS,
    Batch,
    LLMAnalysisResult,
    LLMAnalyzerBase,
)
from skillspector.llm_utils import bind_structured_output, register_chat_model_provider
from skillspector.models import Finding
from skillspector.nodes.analyzers.mcp_tool_poisoning import _TP4AnalysisResult, _TP4Analyzer
from skillspector.nodes.analyzers.semantic_security_discovery import node as security_discovery
from skillspector.nodes.meta_analyzer import LLMMetaAnalyzer, MetaAnalyzerResult
from skillspector.providers import reset_provider, use_provider
from skillspector.providers.chat_models import AzureChatOpenAI, ChatOpenAI
from skillspector.providers.openai import OpenAIProvider
from skillspector.response_recovery import (
    parse_recovery_value,
    recover_payload,
    recovery_field_choices,
)
from skillspector.structured_response import (
    MAX_RESPONSE_CHARS,
    MAX_RESPONSE_DEPTH,
    ResponseIssue,
    StructuredOutputParseError,
    completed_response_text,
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
    assert error.value.recovery_text == text


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


@pytest.mark.parametrize("text", ["No security issues found.", '{"message": "No issues."}'])
@pytest.mark.parametrize("completed", [False, True])
def test_completed_prose_and_missing_schema_fields_are_separate_from_lossless_repair(
    text, completed
):
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text(text, LLMAnalysisResult, completed=completed)
    assert error.value.repair_text is None
    assert error.value.recovery_text == (text if completed else None)


@pytest.mark.parametrize(
    "text",
    [
        "",
        '{"findings": [',
        '{"findings": [], "findings": []}',
        '{"findings": []} {"findings": []}',
        '{"findings": NaN}',
        '{"findings": 1e999}',
        '```json\n{"findings": []',
        '[{"severity": "HIGH"}]',
    ],
)
@pytest.mark.parametrize("require_tool", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_completed_malformed_text_is_available_only_for_uncertain_recovery(
    text, require_tool, completed
):
    message = AIMessage(
        content=text, response_metadata={"finish_reason": "stop"} if completed else {}
    )
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(
            message, LLMAnalysisResult, tool="LLMAnalysisResult", require_tool=require_tool
        )
    assert error.value.repair_text is None
    assert error.value.recovery_text == (text if completed else None)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("x" * (MAX_RESPONSE_CHARS + 1), id="size"),
        pytest.param('{"findings": ' + "[" * MAX_RESPONSE_DEPTH, id="unbalanced-depth"),
        pytest.param("[" * (MAX_RESPONSE_DEPTH + 1), id="array-depth"),
        pytest.param(
            '{"findings": ' + "[" * MAX_RESPONSE_DEPTH + "]" * MAX_RESPONSE_DEPTH + "}",
            id="balanced-depth",
        ),
    ],
)
@pytest.mark.parametrize("require_tool", [False, True])
def test_recovery_does_not_waive_size_or_nesting_limits(text, require_tool):
    message = AIMessage(content=text, response_metadata={"finish_reason": "stop"})
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(
            message, LLMAnalysisResult, tool="LLMAnalysisResult", require_tool=require_tool
        )
    assert error.value.issue == ResponseIssue.LIMIT
    assert error.value.recovery_text is None


def test_malformed_text_nesting_guard_ignores_containers_inside_quoted_strings():
    text = '{"message": ' + json.dumps("[" * (MAX_RESPONSE_DEPTH + 1))
    with pytest.raises(StructuredOutputParseError) as error:
        validate_response_text(text, LLMAnalysisResult, completed=True)
    assert error.value.issue == ResponseIssue.INVALID_JSON
    assert error.value.recovery_text == text


def test_missing_tool_call_keeps_completed_prose_for_recovery():
    message = AIMessage(content="No issues found.", response_metadata={"finish_reason": "stop"})
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(message, LLMAnalysisResult, tool="LLMAnalysisResult", require_tool=True)
    assert error.value.issue == ResponseIssue.INVALID_TOOL
    assert error.value.recovery_text == message.content


def test_missing_tool_call_keeps_complete_json_for_recovery():
    message = AIMessage(content='{"findings": []}', response_metadata={"finish_reason": "stop"})
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(message, LLMAnalysisResult, tool="LLMAnalysisResult", require_tool=True)
    assert error.value.issue == ResponseIssue.INVALID_TOOL
    assert error.value.recovery_text == message.content


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "stop"},
        {"finish_reason": "STOP"},
        {"stop_reason": "end_turn"},
        {"stop_reason": "stop_sequence"},
        {"stopReason": "end_turn"},
        {"stopReason": "stop_sequence"},
        {"status": "completed"},
    ],
)
def test_completed_plain_fields_are_provider_independent(metadata):
    assert (
        completed_response_text(AIMessage(content="  Severity: HIGH  ", response_metadata=metadata))
        == "Severity: HIGH"
    )


@pytest.mark.parametrize(
    "metadata,additional,issue",
    [
        ({}, {}, ResponseIssue.INCOMPLETE),
        ({"finish_reason": "length"}, {}, ResponseIssue.TRUNCATED),
        ({"stop_reason": "max_tokens"}, {}, ResponseIssue.TRUNCATED),
        ({"status": "incomplete"}, {}, ResponseIssue.INCOMPLETE),
        ({"stop_reason": "pause_turn"}, {}, ResponseIssue.INCOMPLETE),
        ({"finish_reason": "content_filter"}, {}, ResponseIssue.REFUSED),
        ({"finish_reason": "stop"}, {"refusal": "Refused"}, ResponseIssue.REFUSED),
    ],
)
def test_plain_fields_require_the_same_completion_and_refusal_guards(metadata, additional, issue):
    message = AIMessage(content="HIGH", response_metadata=metadata, additional_kwargs=additional)
    with pytest.raises(StructuredOutputParseError) as error:
        completed_response_text(message)
    assert error.value.issue == issue


def test_plain_string_without_completion_metadata_is_not_a_completed_answer():
    with pytest.raises(StructuredOutputParseError) as error:
        completed_response_text("HIGH")
    assert error.value.issue == ResponseIssue.INCOMPLETE


def test_plain_field_refusal_blocks_are_rejected_even_with_completion():
    message = AIMessage(
        content=[{"type": "refusal", "refusal": "Refused"}, {"type": "text", "text": "LOW"}],
        response_metadata={"finish_reason": "stop"},
    )
    with pytest.raises(StructuredOutputParseError) as error:
        completed_response_text(message)
    assert error.value.issue == ResponseIssue.REFUSED


def test_plain_fields_reject_unparsed_tool_blocks():
    message = AIMessage(
        content=[
            {"type": "function_call", "name": "unexpected", "arguments": "{}"},
            {"type": "text", "text": "LOW"},
        ],
        response_metadata={"status": "completed"},
    )
    with pytest.raises(StructuredOutputParseError) as error:
        completed_response_text(message)
    assert error.value.issue == ResponseIssue.INVALID_TOOL


@pytest.mark.parametrize("names", [["other"], ["LLMAnalysisResult", "LLMAnalysisResult"]])
def test_wrong_or_multiple_tools_never_become_plain_recovery(names):
    message = AIMessage(
        content="No issues found.",
        response_metadata={"finish_reason": "tool_calls"},
        tool_calls=[{"name": name, "args": {}, "id": str(i)} for i, name in enumerate(names)],
    )
    with pytest.raises(StructuredOutputParseError) as error:
        validate_message(message, LLMAnalysisResult, tool="LLMAnalysisResult")
    assert error.value.issue == ResponseIssue.INVALID_TOOL
    assert error.value.recovery_text is None
    with pytest.raises(StructuredOutputParseError):
        completed_response_text(message)


@pytest.mark.parametrize(
    "text,issue",
    [(" ", ResponseIssue.EMPTY), ("x" * (MAX_RESPONSE_CHARS + 1), ResponseIssue.LIMIT)],
)
def test_plain_fields_remain_nonempty_and_bounded(text, issue):
    with pytest.raises(StructuredOutputParseError) as error:
        completed_response_text(
            AIMessage(content=text, response_metadata={"finish_reason": "stop"})
        )
    assert error.value.issue == issue


class _RecoverySeverity(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@pytest.mark.parametrize(
    "text,annotation,field_name,expected",
    [
        ("Severity: high", Literal["LOW", "HIGH", "CRITICAL"], "severity", "HIGH"),
        ('"HIGH"', _RecoverySeverity, "severity", _RecoverySeverity.HIGH),
        ("Rule ID: TM-1", Literal["TM-1", "TM-2"], "rule_id", "TM-1"),
        ("Findings count: 2", int, "findings_count", 2),
        ("Start_line: 1", int | None, "start_line", 1),
        ("Confidence: 0.75", float, "confidence", 0.75),
        ("1", float, None, 1.0),
        ("YES", bool, None, True),
        ("false", bool, None, False),
        ("No", Annotated[bool, "a field"], None, False),
        (
            "  Keep the quoted 'false' value.\nSecond line.  ",
            str,
            "message",
            "Keep the quoted 'false' value.\nSecond line.",
        ),
    ],
)
def test_recovery_field_parser_accepts_only_unambiguous_values(
    text, annotation, field_name, expected
):
    assert parse_recovery_value(text, annotation, field_name=field_name) == expected


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (Literal["low", "high"], ("low", "high")),
        (Annotated[Literal["low", "high"] | None, "impact"], ("low", "high")),
        (Annotated[_RecoverySeverity, "severity"] | None, ("HIGH", "LOW")),
        (str, ()),
        (int | str, ()),
    ],
)
def test_recovery_prompt_choices_match_scalar_validation(annotation, expected):
    assert recovery_field_choices(annotation) == expected
    for choice in expected:
        value = parse_recovery_value(json.dumps(choice), annotation)
        assert (value.value if isinstance(value, Enum) else value) == choice


@pytest.mark.parametrize(
    "text,annotation,field_name",
    [
        ("not high, maybe critical", Literal["LOW", "HIGH", "CRITICAL"], "severity"),
        ("Severity: HIGH or LOW", _RecoverySeverity, "severity"),
        ("Message: HIGH", _RecoverySeverity, "severity"),
        ("No, but yes", bool, None),
        ("1", bool, None),
        ("true", int, None),
        ("1.0", int, None),
        ("1e999", float, None),
        ("NaN", float, None),
        ("Infinity", float, None),
        ("2 findings", int, None),
        ("[2]", int, None),
        ("null", int | None, None),
        ("", str, None),
        ("x" * (MAX_RESPONSE_CHARS + 1), str, None),
    ],
)
def test_recovery_field_parser_does_not_guess(text, annotation, field_name):
    with pytest.raises(StructuredOutputParseError):
        parse_recovery_value(text, annotation, field_name=field_name)


@pytest.mark.parametrize(
    "source",
    [
        '{"findings":[{"rule_id":"SSD-1","severity":"HIGH","start_line":1}],}',
        '{"findings" [{"rule_id" "SSD-1" "severity" "HIGH" "start_line" 1}]}',
        '```json\n{"findings":[{"rule_id":"SSD-1","severity":"HIGH","start_line":1}],}\n```',
    ],
)
def test_recover_payload_preserves_original_finding_facts(source):
    assert recover_payload(source) == {
        "findings": [{"rule_id": "SSD-1", "severity": "HIGH", "start_line": 1}]
    }


def test_recover_payload_preserves_nested_order_and_scalar_types():
    source = '{"b" [true false null 1 -2 0.5 "a\\nb"] "a" {"escaped\\u006bey" "value"}}'
    payload = recover_payload(source)
    assert payload == {"b": [True, False, None, 1, -2, 0.5, "a\nb"], "a": {"escapedkey": "value"}}
    assert list(payload) == ["b", "a"]


def test_recover_payload_treats_json_like_prose_inside_a_string_as_one_value():
    text = 'HIGH, } ], "findings": [] is only quoted text.'
    assert recover_payload(json.dumps({"message": text})) == {"message": text}


def test_recover_payload_accepts_the_existing_depth_boundary():
    source = (
        '{"value":' + "[" * (MAX_RESPONSE_DEPTH - 1) + "0" + "]" * (MAX_RESPONSE_DEPTH - 1) + "}"
    )
    payload = recover_payload(source)
    assert payload is not None
    value = payload["value"]
    for _ in range(MAX_RESPONSE_DEPTH - 1):
        assert isinstance(value, list) and len(value) == 1
        value = value[0]
    assert value == 0


@pytest.mark.parametrize(
    "source",
    [
        "The severity is HIGH.",
        "",
        "[]",
        '{"findings": [',
        '{"findings": [], "findings": []}',
        '{"a": {"x": 1, "\\u0078": 2}}',
        '{"a" 1 "a" 2}',
        '{"findings":}',
        '{"a": 1, "b":}',
        '{"findings": ["HIGH", }',
        '{"findings": []} {"findings": []}',
        '{"findings": []} trailing prose',
        '{"findings": NaN}',
        '{"findings": 1e999}',
        '{"findings": "incomplete}',
        '{"findings": ' + "[" * MAX_RESPONSE_DEPTH + "]" * MAX_RESPONSE_DEPTH + "}",
        "x" * (MAX_RESPONSE_CHARS + 1),
    ],
)
def test_recover_payload_rejects_ambiguous_or_unusable_originals(source):
    assert recover_payload(source) is None


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
        failures=None,
        failure_headers=None,
        azure=False,
    ):
        if method is None:
            monkeypatch.delenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", raising=False)
        else:
            monkeypatch.setenv("SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD", method)
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            failure = (failures or {}).get(len(requests))
            if failure is not None:
                if isinstance(failure, Exception):
                    raise failure
                return httpx.Response(
                    failure,
                    json={"error": {"message": "temporarily unavailable"}},
                    headers=failure_headers,
                    request=request,
                )
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
        model_class = AzureChatOpenAI if azure else ChatOpenAI
        route_options = (
            {"azure_endpoint": "https://azure.example.test", "api_version": "2024-06-01"}
            if azure
            else {}
        )
        llm = model_class(
            model=model,
            api_key="test",
            http_client=client,
            http_async_client=async_client,
            max_retries=0,
            reasoning_effort=reasoning_effort,
            **route_options,
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


@pytest.mark.parametrize("azure", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "source,issue",
    [
        ("", ResponseIssue.EMPTY),
        (
            '{"findings": [{"severity": "HIGH"}], "findings": []}',
            ResponseIssue.INVALID_SCHEMA,
        ),
    ],
)
def test_openai_and_azure_keep_original_arguments_before_sdk_normalization(
    http_model, azure, asynchronous, source, issue
):
    requests = http_model([(source, "stop")], method="function_calling", azure=azure)
    analyzer = make_analyzer("discovery")
    runnable = bind_structured_output(analyzer._llm, LLMAnalysisResult, "test-model")
    with pytest.raises(StructuredOutputParseError) as error:
        if asynchronous:
            asyncio.run(runnable.ainvoke("Assess"))
        else:
            runnable.invoke("Assess")
    assert error.value.issue == issue
    assert error.value.recovery_text == source
    assert len(requests) == 1
    assert requests[0]["tools"][0]["function"]["name"] == "LLMAnalysisResult"


@pytest.mark.parametrize("model_class", [ChatOpenAI, AzureChatOpenAI])
def test_wire_tool_arguments_preserve_every_choice_and_generation_metadata(model_class):
    route_options = (
        {"azure_endpoint": "https://azure.example.test", "api_version": "2024-06-01"}
        if model_class is AzureChatOpenAI
        else {}
    )
    model = model_class(model="test-model", api_key="test", **route_options)
    choices = [
        {
            "index": index,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": f"response-{index}", "arguments": arguments},
                    }
                ],
            },
        }
        for index, arguments in enumerate(["", '{"severity":"HIGH","severity":"LOW"}'])
    ]
    result = model._create_chat_result(
        {
            "choices": choices,
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    assert len(result.generations) == 2
    assert result.llm_output["token_usage"]["total_tokens"] == 15
    for generation, choice in zip(result.generations, choices, strict=True):
        assert generation.message.additional_kwargs["tool_calls"] == choice["message"]["tool_calls"]
        assert generation.generation_info["finish_reason"] == "tool_calls"
        assert generation.message.usage_metadata["total_tokens"] == 15


def test_azure_raw_argument_preservation_keeps_native_content_filter_rejection():
    model = AzureChatOpenAI(
        model="test-model",
        api_key="test",
        azure_endpoint="https://azure.example.test",
        api_version="2024-06-01",
    )
    with pytest.raises(ValueError, match="content filter"):
        model._create_chat_result(
            {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"role": "assistant", "content": ""},
                    }
                ]
            }
        )


@pytest.mark.parametrize("model_class", [ChatOpenAI, AzureChatOpenAI])
def test_raw_argument_preservation_leaves_normal_text_messages_unchanged(model_class):
    route_options = (
        {"azure_endpoint": "https://azure.example.test", "api_version": "2024-06-01"}
        if model_class is AzureChatOpenAI
        else {}
    )
    model = model_class(model="test-model", api_key="test", **route_options)
    result = model._create_chat_result(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "HIGH", "tool_calls": None},
                }
            ]
        }
    )
    message = result.generations[0].message
    assert message.content == "HIGH"
    assert "tool_calls" not in message.additional_kwargs
    assert not message.tool_calls and not message.invalid_tool_calls


def test_meta_recovery_prompt_names_the_allowed_categorical_values(http_model):
    http_model([])
    analyzer = make_analyzer("review")
    batch = Batch(file_path="a.py", content="x")
    batch.findings = [
        Finding(rule_id="E2", message="Untrusted input", severity="HIGH", file="a.py")
    ]
    plan = analyzer._field_recovery_plan(batch, "Assess", "Completed prose assessment", [])
    prompts = {question.name: question.prompt for question in plan}
    assert "Allowed values: malicious, negligent, benign." in prompts["intent"]
    assert "Allowed values: critical, high, medium, low." in prompts["impact"]


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
    assert not outcome.failures and not outcome.warnings
    (finding,) = outcome.successful[0][1]
    assert (finding.rule_id, finding.severity) == ("SSD-1", "HIGH")
    assert finding.message == "leaks data"
    assert len(requests) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("object_container", [False, True])
@pytest.mark.parametrize("severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
@pytest.mark.parametrize(
    "original_fields,expected_confidence",
    [
        ({}, 0.5),
        ({"confidence": 99}, 0.99),
        ({"start_line": 0}, 0.5),
        ({"confidence": "not a number"}, 0.5),
        ({"start_line": "unknown"}, 0.5),
    ],
)
def test_field_recovery_cannot_erase_original_finding_when_details_remain_unknown(
    http_model, asynchronous, object_container, severity, original_fields, expected_confidence
):
    original = {"rule_id": "SSD-1", "severity": severity, "start_line": 1, **original_fields}
    source = json.dumps({"findings": original if object_container else [original]})
    requests = http_model([(source, "stop"), ("0", "stop"), ("UNKNOWN", "stop")])
    analyzer = make_analyzer("discovery")
    analyzer.recovery_rule_ids = ("SSD-1",)
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.successful and not outcome.failures
    (finding,) = outcome.recovered[0][1]
    assert (finding.rule_id, finding.severity) == ("SSD-1", severity)
    assert finding.confidence == expected_confidence
    assert finding.start_line == 1
    assert finding.evidence["assessment_complete"] is False
    assert outcome.warnings[0].reason == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    assert 1 < len(requests) <= 1 + FIELD_RECOVERY_MAX_CALLS
    assert "Requested field: finding_count." in requests[1]["messages"][0]["content"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("object_container", [False, True])
def test_known_high_switches_to_field_recovery_before_a_replacement_assessment(
    http_model, asynchronous, object_container
):
    original = {"rule_id": "SSD-1", "severity": "HIGH", "start_line": 1}
    if object_container:
        original["message"] = "Leaks data"
    source = json.dumps({"findings": original if object_container else [original]})
    requests = http_model([(source, "stop"), ("0", "stop")])
    analyzer = make_analyzer("discovery")
    analyzer.recovery_rule_ids = ("SSD-1",)
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)

    assert not outcome.successful and not outcome.failures
    (finding,) = outcome.recovered[0][1]
    assert (finding.rule_id, finding.severity) == ("SSD-1", "HIGH")
    assert outcome.warnings[0].reason == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    assert "Requested field: finding_count." in requests[1]["messages"][0]["content"]
    assert all("Requested field:" in r["messages"][0]["content"] for r in requests[1:])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_original_high_beyond_new_finding_limit_remains_visible(http_model, asynchronous):
    source = json.dumps(
        {
            "findings": [
                {
                    "rule_id": "SSD-1",
                    "severity": "HIGH" if index == 17 else "LOW",
                    "start_line": index,
                }
                for index in range(1, 18)
            ]
        }
    )
    requests = http_model([(source, "stop"), ("0", "stop"), ("UNKNOWN", "stop")])
    analyzer = make_analyzer("discovery")
    analyzer.recovery_rule_ids = ("SSD-1",)
    outcome = run_outcome(
        analyzer, [Batch(file_path="a.py", content="x\n" * 17, end_line=17)], asynchronous
    )

    assert not outcome.successful and not outcome.failures
    findings = outcome.recovered[0][1]
    assert len(findings) == 17
    assert [finding.severity for finding in findings] == ["LOW"] * 16 + ["HIGH"]
    assert findings[-1].start_line == 17
    assert all(finding.evidence["assessment_complete"] is False for finding in findings)
    assert outcome.warnings[0].reason == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    assert len(requests) <= 1 + FIELD_RECOVERY_MAX_CALLS


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tp4_invalid_original_confidence_can_be_recovered_without_losing_mismatch(
    http_model, asynchronous
):
    source = '{"is_mismatch": true, "confidence": 99}'
    requests = http_model([(source, "stop"), ("0.9", "stop"), ("UNKNOWN", "stop")])
    analyzer = make_analyzer("tp4")
    analyzer.build_prompt = lambda *_args, **_kwargs: "Assess the code"
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.failures and not outcome.warnings
    (assessment,) = outcome.successful[0][1]
    assert assessment.is_mismatch is True
    assert assessment.confidence == 0.9
    assert 1 < len(requests) <= 1 + FIELD_RECOVERY_MAX_CALLS
    assert "Requested field: confidence." in requests[1]["messages"][0]["content"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "stop,reason",
    [
        ("length", LedgerReason.LLM_RESPONSE_TRUNCATED),
        ("content_filter", LedgerReason.LLM_RESPONSE_REFUSED),
    ],
)
def test_failed_formatting_retains_original_high_and_terminal_failure(
    http_model, asynchronous, stop, reason
):
    source = '{"findings":[{"rule_id":"SSD-1","message":"leaks data","severity":"HIGH","start_line":1}],}'
    requests = http_model([(source, "stop"), ('{"findings": []}', stop)])
    analyzer = make_analyzer("discovery")
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.successful and not outcome.warnings
    assert outcome.failures[0].reason == reason
    (finding,) = outcome.recovered[0][1]
    assert (finding.rule_id, finding.severity) == ("SSD-1", "HIGH")
    assert len(requests) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("object_container", [False, True])
@pytest.mark.parametrize("failure_stage,expected_calls", [("formatting", 2), ("initial_count", 2)])
def test_provider_failure_preserves_original_high_before_recovery_fields_finish(
    http_model, asynchronous, object_container, failure_stage, expected_calls
):
    original = {"rule_id": "SSD-1", "severity": "HIGH", "start_line": 1}
    if failure_stage == "formatting":
        original["message"] = "Leaks data"
    source = json.dumps({"findings": original if object_container else [original]})
    if failure_stage == "formatting":
        source = source[:-1] + ",}"
    requests = http_model([(source, "stop")], failures={expected_calls: 401})
    analyzer = make_analyzer("discovery")
    analyzer.recovery_rule_ids = ("SSD-1",)
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)

    assert not outcome.successful and not outcome.warnings
    assert outcome.failures[0].reason == LedgerReason.LLM_BATCH_FAILED
    (finding,) = outcome.recovered[0][1]
    assert (finding.rule_id, finding.severity) == ("SSD-1", "HIGH")
    assert finding.evidence["assessment_complete"] is False
    assert len(requests) == expected_calls
    if failure_stage == "initial_count":
        assert "Requested field: finding_count." in requests[-1]["messages"][0]["content"]


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


@pytest.mark.parametrize("wrapper", ["json", "input", "parameters"])
def test_known_gateway_wrapper_is_unwrapped_without_forcing_tools_or_changing_reasoning(
    http_model, wrapper
):
    requests = http_model(
        [(json.dumps({wrapper: {"findings": []}}), "stop")],
        model="azure/anthropic/claude-opus-5",
        reasoning_effort="high",
    )
    analyzer = make_analyzer("discovery", model="azure/anthropic/claude-opus-5")
    outcome = analyzer.run_batches_detailed([Batch(file_path="a.py", content="x")])
    assert outcome.successful and not outcome.failures
    assert len(requests) == 1
    assert "tool_choice" not in requests[0]
    assert requests[0]["reasoning_effort"] == "high"


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
def test_verified_gateway_method_is_shared_by_all_five_checks(http_model, name):
    answer = {"is_mismatch": False} if name == "tp4" else {"findings": []}
    requests = http_model(
        [(json.dumps(answer), "stop")], model="azure/anthropic/claude-opus-5", method=None
    )
    analyzer = make_analyzer(name, model="azure/anthropic/claude-opus-5")
    analyzer.build_prompt = lambda *_args, **_kwargs: "Assess the code"
    outcome = analyzer.run_batches_detailed([Batch(file_path="a.py", content="x")])
    assert outcome.successful and not outcome.failures
    assert len(requests) == 1
    assert requests[0]["tools"]
    assert "response_format" not in requests[0]


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
    "first",
    [
        "{}",
        '{"findings": "missing"}',
        '{"findings": [], "hidden_findings": ["danger"]}',
        '{"json": {"findings": []}}',
    ],
)
def test_missing_or_extra_assessment_data_uses_bounded_recovery_without_formatting(
    http_model, first
):
    requests = http_model([(first, "stop")])
    outcome = make_analyzer("discovery").run_batches_detailed(
        [Batch(file_path="a.py", content="x")]
    )
    assert not outcome.successful and not outcome.failures
    assert outcome.warnings[0].reason == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    assert 4 <= len(requests) <= 4 + FIELD_RECOVERY_MAX_CALLS
    assert all("Requested field:" in r["messages"][0]["content"] for r in requests[4:])
    assert all("Repair JSON punctuation" not in r["messages"][0]["content"] for r in requests)


@pytest.mark.parametrize(
    "source",
    [
        "",
        '{"findings": [',
        '{"findings": [], "findings": [{"severity": "HIGH"}]}',
        '{"findings": NaN}',
        '{"findings": 1e999}',
        '{"findings": null}',
        '{"findings": "unknown"}',
        "{}",
        '{"findings": []} {"findings": [{"severity": "HIGH"}]}',
    ],
)
@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_completed_ambiguous_response_remains_partial_even_when_recovery_says_zero(
    http_model, source, method, asynchronous
):
    requests = http_model([(source, "stop")] * 4 + [("0", "stop")], method=method)
    analyzer = make_analyzer("discovery")
    batches = [Batch(file_path="a.py", content="x")]
    outcome = (
        asyncio.run(analyzer.arun_batches_detailed(batches))
        if asynchronous
        else analyzer.run_batches_detailed(batches)
    )
    assert not outcome.successful and not outcome.failures
    assert outcome.warnings[0].reason == LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED
    assert 4 < len(requests) <= 4 + FIELD_RECOVERY_MAX_CALLS
    assert "Requested field: finding_count" in requests[-1]["messages"][0]["content"]
    assert all("Repair JSON punctuation" not in r["messages"][0]["content"] for r in requests)


def test_formatting_rechecks_the_input_allowance(http_model):
    def reduce_input_allowance(_request_number):
        analyzer._input_budget = 1

    requests = http_model([('{"findings": [],}', "stop")], after_response=reduce_input_allowance)
    analyzer = make_analyzer("discovery")
    outcome = analyzer.run_batches_detailed([Batch(file_path="a.py", content="x")])
    assert not outcome.successful
    assert len(outcome.failures) == 1
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
    if stop == "length":
        assert not outcome.successful
        assert outcome.failures[0].reason == LedgerReason.LLM_RESPONSE_TRUNCATED
    else:
        assert outcome.successful[0][1] == []
        assert not outcome.failures and not outcome.warnings
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


def run_outcome(analyzer, batches, asynchronous):
    return (
        asyncio.run(analyzer.arun_batches_detailed(batches, max_concurrency=1))
        if asynchronous
        else analyzer.run_batches_detailed(batches)
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("connection lost"), httpx.ReadTimeout("timeout"), 425, 429, 503],
)
def test_formatting_retries_transient_failures_without_reassessing(
    http_model, asynchronous, method, failure
):
    answer = '{"findings":[{"rule_id":"SSD-1","message":"leaks data","severity":"HIGH","start_line":3,"end_line":4}]}'
    requests = http_model(
        [(answer[:-1] + ",}", "stop"), (answer, "stop")],
        method=method,
        failures={2: failure},
    )
    analyzer = make_analyzer("discovery", timeout=lambda: 120.0)
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.failures
    finding = outcome.successful[0][1][0]
    assert (finding.file, finding.start_line, finding.end_line, finding.severity) == (
        "a.py",
        3,
        4,
        "HIGH",
    )
    assert len(requests) == 3
    assert "Repair JSON punctuation" not in requests[0]["messages"][0]["content"]
    assert "Repair JSON punctuation" in requests[1]["messages"][0]["content"]
    assert requests[1]["messages"] == requests[2]["messages"]
    assert sum(record.get("total_tokens", 0) or 0 for record in analyzer.inference_usage) == 30


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("earlier_failures", range(4))
def test_formatting_shares_the_existing_provider_retry_budget(
    http_model, asynchronous, earlier_failures
):
    failures = dict.fromkeys(range(1, earlier_failures + 1), 503)
    failures.update(dict.fromkeys(range(earlier_failures + 2, 6), 503))
    requests = http_model([('{"findings": [],}', "stop")], failures=failures)
    analyzer = make_analyzer("discovery", timeout=lambda: 120.0)
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.successful
    assert outcome.failures[0].reason == LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
    assert len(requests) == 5
    assert all(
        "Repair JSON punctuation" in request["messages"][0]["content"]
        for request in requests[earlier_failures + 1 :]
    )
    sleep = analyzer._asleep_before_retry if asynchronous else analyzer._sleep_before_retry
    assert sleep.call_count == 3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("earlier_failures", [2, 3])
def test_formatting_never_extends_the_total_invocation_budget(
    http_model, asynchronous, earlier_failures
):
    failures = dict.fromkeys(range(4, 4 + earlier_failures), 503)
    if earlier_failures == 2:
        failures[7] = 503
    requests = http_model(
        [("{}", "stop")] * 3 + [('{"findings": [],}', "stop")] * 4,
        failures=failures,
    )
    analyzer = make_analyzer("discovery", timeout=lambda: 120.0)
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert len(requests) == 7
    if earlier_failures == 2:
        assert not outcome.successful
        assert outcome.failures[0].reason == LedgerReason.LLM_BATCH_FAILED
    else:
        assert outcome.successful[0][1] == []
        assert not outcome.failures and not outcome.warnings
    assert sum(
        "Repair JSON punctuation" in request["messages"][0]["content"] for request in requests
    ) == (1 if earlier_failures == 2 else 0)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "first,answer,stop,reason",
    [
        (
            '{"findings": [],}',
            '{"findings": [],}',
            "stop",
            None,
        ),
        (
            '{"findings": [],}',
            '{"findings": []}',
            "length",
            LedgerReason.LLM_RESPONSE_TRUNCATED,
        ),
        (
            '{"findings": [],}',
            '{"findings": []}',
            "content_filter",
            LedgerReason.LLM_RESPONSE_REFUSED,
        ),
        (
            '{"findings": [],}',
            '{"findings": [], "extra": 1}',
            "stop",
            None,
        ),
        ('{"findings": [],}', '{"findings": []}', None, None),
        (
            '{"findings": "invalid",}',
            '{"findings": "invalid"}',
            "stop",
            LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED,
        ),
    ],
)
def test_formatting_recovery_after_transport_retry_preserves_original_or_terminal_cause(
    http_model, asynchronous, first, answer, stop, reason
):
    requests = http_model([(first, "stop"), (answer, stop)], failures={2: 503})
    analyzer = make_analyzer("discovery", timeout=lambda: 120.0)
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    if reason is None:
        assert outcome.successful[0][1] == []
        assert not outcome.failures and not outcome.warnings
        assert len(requests) == 3
    elif reason is LedgerReason.LLM_FORMAT_RECOVERY_EXHAUSTED:
        assert not outcome.successful and not outcome.failures
        assert outcome.warnings[0].reason == reason
        assert 3 < len(requests) <= 3 + FIELD_RECOVERY_MAX_CALLS
    else:
        assert not outcome.successful
        assert outcome.failures[0].reason == reason
        assert len(requests) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "headers,reason",
    [
        ({"retry-after": "20"}, LedgerReason.RUNTIME_LIMIT),
        ({"x-should-retry": "false"}, LedgerReason.LLM_BATCH_FAILED),
    ],
)
def test_formatting_preserves_deadline_and_explicit_no_retry_stops(
    http_model, asynchronous, headers, reason
):
    requests = http_model(
        [('{"findings": [],}', "stop")], failures={2: 429}, failure_headers=headers
    )
    analyzer = make_analyzer("discovery", timeout=lambda: 10.0)
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.successful
    assert outcome.failures[0].reason == reason
    assert len(requests) == 2
    analyzer._sleep_before_retry.assert_not_called()
    analyzer._asleep_before_retry.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_formatting_keeps_native_provider_retry_ownership(http_model, asynchronous, exhausted):
    requests = http_model(
        [('{"findings": [],}', "stop"), ('{"findings": []}', "stop")],
        failures=dict.fromkeys(range(2, 6 if exhausted else 3), 503),
    )
    analyzer = make_analyzer("discovery")
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert len(requests) == (5 if exhausted else 3)
    if exhausted:
        assert not outcome.successful
        assert outcome.failures[0].reason == LedgerReason.LLM_BATCH_FAILED
    else:
        assert not outcome.failures
    analyzer._sleep_before_retry.assert_not_called()
    analyzer._asleep_before_retry.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_formatting_backoff_cannot_restart_an_expired_deadline(http_model, asynchronous):
    remaining = [120.0]
    requests = http_model([('{"findings": [],}', "stop")], failures={2: 503})
    analyzer = make_analyzer("discovery", timeout=lambda: remaining[0])

    def expire(_):
        remaining[0] = 0.0

    analyzer._sleep_before_retry = MagicMock(side_effect=expire)
    analyzer._asleep_before_retry = AsyncMock(side_effect=expire)
    outcome = run_outcome(analyzer, [Batch(file_path="a.py", content="x")], asynchronous)
    assert not outcome.successful
    assert outcome.failures[0].reason == LedgerReason.RUNTIME_LIMIT
    assert len(requests) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exhausted_formatting_retries_preserve_a_sibling_finding(http_model, asynchronous):
    answer = (
        '{"findings":[{"rule_id":"SSD-1","message":"leaks data","severity":"HIGH","start_line":1}]}'
    )
    requests = http_model(
        [('{"findings": [],}', "stop")] * 5 + [(answer, "stop")],
        failures=dict.fromkeys(range(2, 6), 503),
    )
    analyzer = make_analyzer("discovery", timeout=lambda: 120.0)
    analyzer._sleep_before_retry = MagicMock()
    analyzer._asleep_before_retry = AsyncMock()
    batches = [Batch(file_path=path, content="x") for path in ("a.py", "b.py")]
    outcome = run_outcome(analyzer, batches, asynchronous)
    assert len(outcome.failures) == 1
    assert outcome.failures[0].batch.file_path == "a.py"
    assert outcome.failures[0].reason == LedgerReason.LLM_CONNECTION_RETRIES_EXHAUSTED
    assert len(outcome.successful) == 1
    finding = outcome.successful[0][1][0]
    assert (finding.file, finding.severity, finding.message) == ("b.py", "HIGH", "leaks data")
    assert len(requests) == 6
    assert "Repair JSON punctuation" not in requests[-1]["messages"][0]["content"]
