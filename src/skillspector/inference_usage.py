# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized provider-reported inference usage for scan reports.

The collector is attached as a LangChain callback at invocation time.  This is
important for structured output: the parser returns a Pydantic object and would
otherwise discard the provider message that carries token counters.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}")
_COUNTER_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_tokens",
)
_MAX_TOKEN_COUNT = (1 << 63) - 1


class InferenceUsageRecord(TypedDict):
    """One provider-reported inference request, safe to serialize."""

    node: str
    request_kind: str
    provider: str
    model: str
    model_source: str
    usage_source: str
    prompt_tokens: NotRequired[int]
    completion_tokens: NotRequired[int]
    cached_tokens: NotRequired[int]
    cache_write_tokens: NotRequired[int]
    reasoning_tokens: NotRequired[int]
    total_tokens: NotRequired[int]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _counter(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= _MAX_TOKEN_COUNT:
        return value
    if isinstance(value, float) and 0 <= value <= _MAX_TOKEN_COUNT and value.is_integer():
        return int(value)
    return None


def _first_counter(*values: object) -> int | None:
    for value in values:
        parsed = _counter(value)
        if parsed is not None:
            return parsed
    return None


def _positive_counter_sum(*values: object) -> int | None:
    """Return a positive sum when provider-specific partitions are present."""
    counters = [parsed for value in values if (parsed := _counter(value)) is not None]
    total = sum(counters)
    return total if total > 0 else None


def _label(value: object, fallback: str = "unknown") -> str:
    candidate = str(value or "").strip()
    if _LABEL_RE.fullmatch(candidate):
        return candidate
    clean_fallback = str(fallback or "").strip()
    return clean_fallback if _LABEL_RE.fullmatch(clean_fallback) else "unknown"


def _strict_label(value: object) -> str | None:
    candidate = str(value or "").strip()
    return candidate if _LABEL_RE.fullmatch(candidate) else None


def _strict_model_label(value: object) -> str | None:
    """Return a model label only when it cannot encode a URL or userinfo."""
    candidate = _strict_label(value)
    if candidate is None or "://" in candidate or "@" in candidate:
        return None
    return candidate


def _model_label(value: object, fallback: str = "unknown") -> str:
    return _strict_model_label(value) or _strict_model_label(fallback) or "unknown"


def provider_name(provider: object) -> str:
    """Return a stable provider label without endpoint or credential data."""
    names = {
        "AnthropicProvider": "anthropic",
        "AnthropicProxyProvider": "anthropic_proxy",
        "BedrockProvider": "bedrock",
        "ClaudeCLIProvider": "claude_cli",
        "CodexCLIProvider": "codex_cli",
        "GeminiCLIProvider": "gemini_cli",
        "NvBuildProvider": "nv_build",
        "NvInferenceProvider": "nv_inference",
        "OpenAIProvider": "openai",
    }
    return names.get(type(provider).__name__, _label(type(provider).__name__.lower()))


def _usage_record(
    message: object,
    llm_output: Mapping[str, object],
    *,
    node: str,
    request_kind: str,
    provider: str,
    requested_model: str,
) -> InferenceUsageRecord | None:
    usage_metadata = _mapping(_field(message, "usage_metadata"))
    response_metadata = _mapping(_field(message, "response_metadata"))
    response_usage = _mapping(response_metadata.get("usage"))
    token_usage = _mapping(response_metadata.get("token_usage"))
    if not token_usage:
        token_usage = _mapping(llm_output.get("token_usage"))

    input_details = _mapping(
        usage_metadata.get("input_token_details")
        or usage_metadata.get("input_tokens_details")
        or token_usage.get("prompt_tokens_details")
        or token_usage.get("input_tokens_details")
    )
    output_details = _mapping(
        usage_metadata.get("output_token_details")
        or usage_metadata.get("output_tokens_details")
        or token_usage.get("completion_tokens_details")
        or token_usage.get("output_tokens_details")
    )

    standardized_prompt = _first_counter(
        usage_metadata.get("input_tokens"),
        usage_metadata.get("prompt_tokens"),
    )
    # LangChain usage_metadata follows an inclusive input-token contract and
    # carries cache partitions in input_token_details. Raw Anthropic usage is
    # different: input_tokens excludes its separately reported cache fields.
    # Use the raw-direct mode only when a standardized prompt total is absent.
    # Some integrations populate unrelated usage metadata while leaving prompt
    # accounting solely in the raw response.
    direct_cache_read = (
        _first_counter(
            response_usage.get("cache_read_input_tokens"),
            token_usage.get("cache_read_input_tokens"),
        )
        if standardized_prompt is None
        else None
    )
    raw_cache_creation = _mapping(
        response_usage.get("cache_creation") or token_usage.get("cache_creation")
    )
    raw_ttl_cache_write_tokens = _positive_counter_sum(
        raw_cache_creation.get("ephemeral_5m_input_tokens"),
        raw_cache_creation.get("ephemeral_1h_input_tokens"),
    )
    direct_cache_write = (
        _first_counter(
            raw_ttl_cache_write_tokens,
            response_usage.get("cache_creation_input_tokens"),
            token_usage.get("cache_creation_input_tokens"),
            token_usage.get("cache_write_tokens"),
        )
        if standardized_prompt is None
        else None
    )
    cached_tokens = _first_counter(
        direct_cache_read,
        input_details.get("cache_read"),
        input_details.get("cached_tokens"),
        usage_metadata.get("cache_read_input_tokens"),
        response_usage.get("cache_read_input_tokens"),
        token_usage.get("cache_read_input_tokens"),
    )
    detail_ttl_cache_write_tokens = _positive_counter_sum(
        input_details.get("ephemeral_5m_input_tokens"),
        input_details.get("ephemeral_1h_input_tokens"),
    )
    ttl_cache_write_tokens = detail_ttl_cache_write_tokens or raw_ttl_cache_write_tokens
    cache_write_tokens = _first_counter(
        ttl_cache_write_tokens,
        direct_cache_write,
        input_details.get("cache_creation"),
        input_details.get("cache_write"),
        input_details.get("cache_write_tokens"),
        usage_metadata.get("cache_creation_input_tokens"),
        response_usage.get("cache_creation_input_tokens"),
        token_usage.get("cache_creation_input_tokens"),
        token_usage.get("cache_write_tokens"),
    )
    prompt_tokens = _first_counter(
        standardized_prompt,
        response_usage.get("input_tokens"),
        response_usage.get("prompt_tokens"),
        token_usage.get("prompt_tokens"),
        token_usage.get("input_tokens"),
    )
    completion_tokens = _first_counter(
        usage_metadata.get("output_tokens"),
        usage_metadata.get("completion_tokens"),
        response_usage.get("output_tokens"),
        response_usage.get("completion_tokens"),
        token_usage.get("completion_tokens"),
        token_usage.get("output_tokens"),
    )

    # Anthropic's raw response reports cache reads and writes outside
    # ``input_tokens``.  OpenAI-compatible nested cache counters are already a
    # subset of prompt_tokens and therefore must not be added again.
    if direct_cache_read is not None or direct_cache_write is not None:
        prompt_tokens = (prompt_tokens or 0) + (direct_cache_read or 0) + (direct_cache_write or 0)

    reasoning_tokens = _first_counter(
        output_details.get("reasoning"),
        output_details.get("reasoning_tokens"),
        usage_metadata.get("reasoning_tokens"),
        token_usage.get("reasoning_tokens"),
    )
    total_tokens = _first_counter(
        usage_metadata.get("total_tokens"),
        response_usage.get("total_tokens"),
        token_usage.get("total_tokens"),
    )
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    counters = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }
    if not any(value is not None for value in counters.values()):
        return None

    provider_model = (
        response_metadata.get("model_name")
        or response_metadata.get("model")
        or response_metadata.get("model_id")
        or llm_output.get("model_name")
        or llm_output.get("model")
    )
    requested_model_label = _model_label(requested_model)
    provider_model_label = _strict_model_label(provider_model)
    model = provider_model_label or requested_model_label
    record: InferenceUsageRecord = {
        "node": _label(node),
        "request_kind": _label(request_kind),
        "provider": _label(provider),
        "model": model,
        "model_source": (
            "provider_response"
            if provider_model_label is not None and provider_model_label != requested_model_label
            else "requested_model"
        ),
        "usage_source": "provider_response",
    }
    for key, value in counters.items():
        if value is not None:
            record[key] = value  # type: ignore[literal-required]
    return record


class InferenceUsageCollector(BaseCallbackHandler):
    """Collect one normalized record from each completed provider call."""

    def __init__(
        self,
        *,
        node: str,
        request_kind: str,
        provider: str,
        requested_model: str,
    ) -> None:
        self._node = node
        self._request_kind = request_kind
        self._provider = provider
        self._requested_model = requested_model
        self._records: list[InferenceUsageRecord] = []
        self._response_received = False
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        """Capture usage after a successful provider response."""
        message: object = None
        for generation_group in response.generations:
            for generation in generation_group:
                candidate = getattr(generation, "message", None)
                if candidate is not None:
                    message = candidate
                    break
            if message is not None:
                break
        record = _usage_record(
            message,
            _mapping(response.llm_output),
            node=self._node,
            request_kind=self._request_kind,
            provider=self._provider,
            requested_model=self._requested_model,
        )
        with self._lock:
            self._response_received = True
            if record is not None:
                self._records.append(record)

    def mark_response_received(self) -> None:
        """Record a completed response from a non-LangChain transport."""
        with self._lock:
            self._response_received = True

    def set_provider(self, provider: str) -> None:
        """Set the effective provider before the first response is observed."""
        label = _label(provider)
        with self._lock:
            if self._response_received and label != self._provider:
                raise RuntimeError("cannot change inference provider after a response")
            self._provider = label

    @property
    def response_received(self) -> bool:
        """Whether the provider returned, even when it reported no token usage."""
        with self._lock:
            return self._response_received

    def snapshot(self) -> list[InferenceUsageRecord]:
        """Return detached copies safe for graph-state serialization."""
        with self._lock:
            return [record.copy() for record in self._records]


def sanitize_inference_usage(
    records: Sequence[object] | None,
) -> list[InferenceUsageRecord]:
    """Whitelist report fields and discard malformed or counter-less records."""
    sanitized: list[InferenceUsageRecord] = []
    for source in records or []:
        if not isinstance(source, Mapping):
            continue
        if source.get("usage_source") != "provider_response":
            continue
        node = _strict_label(source.get("node"))
        request_kind = _strict_label(source.get("request_kind"))
        provider = _strict_label(source.get("provider"))
        model = _strict_model_label(source.get("model"))
        model_source = source.get("model_source")
        if (
            node is None
            or request_kind is None
            or provider is None
            or model is None
            or not isinstance(model_source, str)
            or model_source not in {"provider_response", "requested_model"}
        ):
            continue
        record: InferenceUsageRecord = {
            "node": node,
            "request_kind": request_kind,
            "provider": provider,
            "model": model,
            "model_source": model_source,
            "usage_source": "provider_response",
        }
        found = False
        for key in _COUNTER_KEYS:
            value = _counter(source.get(key))
            if value is not None:
                record[key] = value  # type: ignore[literal-required]
                found = True
        if found:
            sanitized.append(record)
    return sanitized
