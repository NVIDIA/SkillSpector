# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-response inference usage normalization tests."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from skillspector.inference_usage import (
    InferenceUsageCollector,
    _usage_record,
    sanitize_inference_usage,
)


def test_collector_captures_standardized_langchain_usage_without_double_counting_cache() -> None:
    """LangChain input_tokens is already inclusive of its cache partitions."""
    message = AIMessage(
        content="ok",
        response_metadata={"model_name": "claude-opus-4-8-20260801"},
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 60, "cache_creation": 10},
            "output_token_details": {"reasoning": 5},
        },
    )
    collector = InferenceUsageCollector(
        node="semantic_security_discovery",
        request_kind="structured_output",
        provider="anthropic",
        requested_model="claude-opus-4-8",
    )

    collector.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={}))

    assert collector.snapshot() == [
        {
            "node": "semantic_security_discovery",
            "request_kind": "structured_output",
            "provider": "anthropic",
            "model": "claude-opus-4-8-20260801",
            "model_source": "provider_response",
            "usage_source": "provider_response",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cached_tokens": 60,
            "cache_write_tokens": 10,
            "reasoning_tokens": 5,
            "total_tokens": 120,
        }
    ]


def test_collector_marks_response_received_without_usage_counters() -> None:
    collector = InferenceUsageCollector(
        node="semantic_quality_policy",
        request_kind="structured_output",
        provider="codex_cli",
        requested_model="gpt-5.6-sol",
    )
    message = AIMessage(content="provider returned without usage metadata")

    collector.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={}))

    assert collector.response_received is True
    assert collector.snapshot() == []


def test_raw_anthropic_usage_adds_external_cache_counters_to_prompt_total() -> None:
    """Anthropic raw input_tokens excludes cache reads and cache creation."""
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 7,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 20,
            },
        },
    )

    record = _usage_record(
        message,
        {},
        node="meta_analyzer",
        request_kind="structured_output",
        provider="anthropic",
        requested_model="claude-sonnet-4-6",
    )

    assert record is not None
    assert record["prompt_tokens"] == 100
    assert record["completion_tokens"] == 7
    assert record["cached_tokens"] == 50
    assert record["cache_write_tokens"] == 20
    assert record["total_tokens"] == 107


def test_raw_anthropic_ttl_cache_writes_are_included_in_prompt_total() -> None:
    """Raw TTL partitions are direct cache writes even without a generic total."""
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 85,
                "output_tokens": 7,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 10,
                    "ephemeral_1h_input_tokens": 5,
                },
            },
        },
    )

    record = _usage_record(
        message,
        {},
        node="meta_analyzer",
        request_kind="structured_output",
        provider="anthropic",
        requested_model="claude-sonnet-4-6",
    )

    assert record is not None
    assert record["prompt_tokens"] == 100
    assert record["completion_tokens"] == 7
    assert record["cache_write_tokens"] == 15
    assert record["total_tokens"] == 107


def test_standardized_prompt_wins_when_raw_anthropic_cache_usage_is_also_present() -> None:
    """A LangChain AIMessage can carry both normalized and raw usage views."""
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 7,
            "total_tokens": 107,
        },
        response_metadata={
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 7,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 20,
            },
        },
    )

    record = _usage_record(
        message,
        {},
        node="meta_analyzer",
        request_kind="structured_output",
        provider="anthropic",
        requested_model="claude-sonnet-4-6",
    )

    assert record is not None
    assert record["prompt_tokens"] == 100
    assert record["cached_tokens"] == 50
    assert record["cache_write_tokens"] == 20
    assert record["total_tokens"] == 107


def test_anthropic_ttl_cache_creation_partitions_override_zero_generic_counter() -> None:
    """LangChain exposes 5m/1h writes separately and zeros the generic field."""
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 7,
            "total_tokens": 107,
            "input_token_details": {
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 10,
                "ephemeral_1h_input_tokens": 5,
            },
        },
        response_metadata={
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 85,
                "output_tokens": 7,
                "cache_creation_input_tokens": 15,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 10,
                    "ephemeral_1h_input_tokens": 5,
                },
            },
        },
    )

    record = _usage_record(
        message,
        {},
        node="meta_analyzer",
        request_kind="structured_output",
        provider="anthropic",
        requested_model="claude-sonnet-4-6",
    )

    assert record is not None
    assert record["prompt_tokens"] == 100
    assert record["cache_write_tokens"] == 15
    assert record["total_tokens"] == 107


def test_openai_nested_cached_and_reasoning_counters_are_subsets() -> None:
    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "model_name": "gpt-5.6-sol",
            "token_usage": {
                "prompt_tokens": 90,
                "completion_tokens": 12,
                "total_tokens": 102,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 8},
            },
        },
    )

    record = _usage_record(
        message,
        {},
        node="semantic_quality_policy",
        request_kind="structured_output",
        provider="openai",
        requested_model="gpt-5.6-sol",
    )

    assert record is not None
    assert record["prompt_tokens"] == 90
    assert record["cached_tokens"] == 40
    assert record["reasoning_tokens"] == 8
    assert record["total_tokens"] == 102


def test_standardized_bedrock_total_is_recomputed_from_normalized_partitions() -> None:
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 7,
            "total_tokens": 92,
            "input_token_details": {"cache_read": 15},
        },
        response_metadata={
            "model": "us.anthropic.claude-sonnet-4-6-20250915-v1:0",
        },
    )

    record = _usage_record(
        message,
        {},
        node="meta_analyzer",
        request_kind="structured_output",
        provider="bedrock",
        requested_model="us.anthropic.claude-sonnet-4-6-20250915-v1:0",
    )

    assert record is not None
    assert record["prompt_tokens"] == 100
    assert record["completion_tokens"] == 7
    assert record["cached_tokens"] == 15
    assert record["total_tokens"] == 107


def test_no_provider_counters_produces_no_record() -> None:
    message = SimpleNamespace(usage_metadata=None, response_metadata={"model": "some-model"})
    assert (
        _usage_record(
            message,
            {},
            node="meta_analyzer",
            request_kind="structured_output",
            provider="nv_inference",
            requested_model="some-model",
        )
        is None
    )


def test_requested_model_fallback_is_explicit_when_response_omits_model() -> None:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        response_metadata={},
    )

    record = _usage_record(
        message,
        {},
        node="semantic_quality_policy",
        request_kind="structured_output",
        provider="nv_inference",
        requested_model="azure/anthropic/claude-opus-4-6",
    )

    assert record is not None
    assert record["model"] == "azure/anthropic/claude-opus-4-6"
    assert record["model_source"] == "requested_model"


def test_configured_model_echo_is_conservatively_marked_as_requested() -> None:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        response_metadata={"model_name": "gpt-5.4"},
    )

    record = _usage_record(
        message,
        {"model_name": "gpt-5.4"},
        node="semantic_quality_policy",
        request_kind="structured_output",
        provider="openai",
        requested_model="gpt-5.4",
    )

    assert record is not None
    assert record["model"] == "gpt-5.4"
    assert record["model_source"] == "requested_model"


def test_provider_model_url_with_userinfo_falls_back_to_requested_model() -> None:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        response_metadata={"model": "https://key@private-host/v1"},
    )

    record = _usage_record(
        message,
        {},
        node="semantic_quality_policy",
        request_kind="structured_output",
        provider="nv_inference",
        requested_model="azure/anthropic/claude-opus-4-6",
    )

    assert record is not None
    assert record["model"] == "azure/anthropic/claude-opus-4-6"
    assert record["model_source"] == "requested_model"


def test_report_sanitizer_rejects_url_and_userinfo_model_labels() -> None:
    common = {
        "node": "meta_analyzer",
        "request_kind": "structured_output",
        "provider": "anthropic",
        "model_source": "provider_response",
        "usage_source": "provider_response",
        "prompt_tokens": 11,
    }

    assert (
        sanitize_inference_usage(
            [
                {**common, "model": "https://key@private-host/v1"},
                {**common, "model": "key@private-host"},
            ]
        )
        == []
    )


def test_report_sanitizer_whitelists_fields_and_rejects_invalid_values() -> None:
    assert sanitize_inference_usage(
        [
            "not-a-record",
            {
                "node": "meta_analyzer",
                "request_kind": "structured_output",
                "provider": "anthropic",
                "model": "bad\nmodel",
                "model_source": "requested_model",
                "prompt_tokens": 11,
                "completion_tokens": -1,
                "api_key": "must-not-leak",
                "usage_source": "untrusted",
            },
            {
                "node": "meta_analyzer",
                "request_kind": "structured_output",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "model_source": "provider_response",
                "prompt_tokens": 11,
                "completion_tokens": -1,
                "api_key": "must-not-leak",
                "usage_source": "provider_response",
            },
            {
                "node": "meta_analyzer",
                "request_kind": "structured_output",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "model_source": "provider_response",
                "prompt_tokens": 1 << 63,
                "usage_source": "provider_response",
            },
        ]
    ) == [
        {
            "node": "meta_analyzer",
            "request_kind": "structured_output",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "model_source": "provider_response",
            "usage_source": "provider_response",
            "prompt_tokens": 11,
        }
    ]
