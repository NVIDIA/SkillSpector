# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized provider-reported inference usage for scan reports.

The collector is attached as a LangChain callback at invocation time.  This is
important for structured output: the parser returns a Pydantic object and would
otherwise discard the provider message that carries token counters.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import weakref
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
_MIN_SAMPLING_SEED = -(1 << 63)
_MAX_SAMPLING_SEED = (1 << 63) - 1
_FORWARDED_CONTROL_NAMES = ("temperature", "seed", "reasoning_effort")
# Public effort telemetry is an enum, even when a provider accepts arbitrary text.
# Unrecognized provider-specific values must not become a channel for credentials.
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "auto"})
_CREDENTIAL_PREFIXES = (
    "sk-",
    "nvapi-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "bearer-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxr-",
    "hf_",
    "aiza",
    "akia",
    "asia",
    "aws-secret-",
)
_UNPREFIXED_CREDENTIAL = re.compile(r"(?:[0-9a-fA-F]{32,64}|[A-Za-z0-9_+/=]{40,88})\Z")
_AUTHORIZATION_CREDENTIAL = re.compile(r"(?:authorization\s*:\s*)?(?:bearer|basic)(?:\s|:)", re.I)
_COMPACT_CREDENTIAL = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*){2,4}\Z")

_CHAT_MODEL_CONTROLS: dict[
    int,
    tuple[
        weakref.ReferenceType[object],
        dict[str, float | int | str | None],
        dict[str, float | int | str | None],
    ],
] = {}
_CHAT_MODEL_CONTROLS_LOCK = threading.Lock()


def _is_compact_credential(value: str) -> bool:
    """Recognize JSON JWT/JWE headers without rejecting dotted model versions."""
    if not _COMPACT_CREDENTIAL.fullmatch(value):
        return False
    header = value.partition(".")[0]
    try:
        decoded = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
    except (ValueError, RecursionError):
        return False
    return isinstance(decoded, dict)


def looks_like_credential(value: object) -> bool:
    """Return whether a printable label resembles a common secret value."""
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    lowered = candidate.lower()
    return (
        lowered.startswith(_CREDENTIAL_PREFIXES)
        or bool(_AUTHORIZATION_CREDENTIAL.match(candidate))
        or _is_compact_credential(candidate)
        # A full hyphenated model name is not an opaque base64 credential.
        # Still reject long opaque components embedded in a namespaced label.
        or any(_UNPREFIXED_CREDENTIAL.fullmatch(part) for part in candidate.split("-"))
    )


def safe_reasoning_effort(value: object) -> str | None:
    """Return a recognized effort value safe for public configuration telemetry."""
    candidate = value.strip() if isinstance(value, str) else ""
    return candidate if candidate in _REASONING_EFFORTS else None


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
    # Internal-only construction evidence. ``sanitize_inference_usage`` never
    # includes this field in the public token-usage projection; the provenance
    # sanitizer consumes it separately after a provider response is observed.
    requested_controls: NotRequired[dict[str, float | int | str | None]]
    forwarded_controls: NotRequired[dict[str, float | int | str | None]]


def _forwarded_controls(value: Mapping[str, object] | None) -> dict[str, float | int | str | None]:
    """Return the fixed, non-secret sampling-control construction record."""
    source = value or {}
    controls: dict[str, float | int | str | None] = {}
    for name in _FORWARDED_CONTROL_NAMES:
        if name not in source:
            continue
        raw = source.get(name)
        if raw is None:
            controls[name] = None
        elif name == "temperature":
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and 0 <= float(raw) <= 1:
                controls[name] = float(raw)
        elif name == "seed":
            if (
                isinstance(raw, int)
                and not isinstance(raw, bool)
                and _MIN_SAMPLING_SEED <= raw <= _MAX_SAMPLING_SEED
            ):
                controls[name] = raw
        elif (setting := safe_reasoning_effort(raw)) is not None:
            controls[name] = setting
    return controls


def register_chat_model_controls(
    chat_model: object,
    forwarded_controls: Mapping[str, object],
    *,
    requested_controls: Mapping[str, object] | None = None,
) -> None:
    """Associate a model with requested and normalized request controls."""
    model_id = id(chat_model)
    sanitized_requested = _forwarded_controls(requested_controls)
    sanitized_forwarded = _forwarded_controls(forwarded_controls)

    def _discard(model_ref: weakref.ReferenceType[object]) -> None:
        with _CHAT_MODEL_CONTROLS_LOCK:
            current = _CHAT_MODEL_CONTROLS.get(model_id)
            if current is not None and current[0] is model_ref:
                _CHAT_MODEL_CONTROLS.pop(model_id, None)

    try:
        model_ref = weakref.ref(chat_model, _discard)
    except TypeError:
        return
    with _CHAT_MODEL_CONTROLS_LOCK:
        _CHAT_MODEL_CONTROLS[model_id] = (
            model_ref,
            sanitized_requested,
            sanitized_forwarded,
        )


def chat_model_controls(chat_model: object | None) -> dict[str, float | int | str | None]:
    """Return detached construction evidence for *chat_model*, when recorded."""
    if chat_model is None:
        return {}
    with _CHAT_MODEL_CONTROLS_LOCK:
        current = _CHAT_MODEL_CONTROLS.get(id(chat_model))
        if current is None or current[0]() is not chat_model:
            return {}
        return current[2].copy()


def chat_model_requested_controls(
    chat_model: object | None,
) -> dict[str, float | int | str | None]:
    """Return the controls resolved when *chat_model* was constructed."""
    if chat_model is None:
        return {}
    with _CHAT_MODEL_CONTROLS_LOCK:
        current = _CHAT_MODEL_CONTROLS.get(id(chat_model))
        if current is None or current[0]() is not chat_model:
            return {}
        return current[1].copy()


def retained_chat_model_controls(
    chat_model: object,
    names: Sequence[str],
) -> dict[str, float | int | str | None]:
    """Return controls retained by the normalized provider request payload.

    LangChain may accept a constructor option and then remove it for a
    provider/model combination. Provenance must describe the request that the
    adapter will send, not the raw constructor arguments supplied before that
    normalization.
    """
    selected = [name for name in names if name in _FORWARDED_CONTROL_NAMES]
    controls: dict[str, object] = dict.fromkeys(selected)
    payload: Mapping[str, object] | None = None
    payload_builder = getattr(chat_model, "_get_request_payload", None)
    if callable(payload_builder):
        try:
            candidate = payload_builder("SkillSpector provenance probe")
        except (TypeError, ValueError):
            candidate = None
        if isinstance(candidate, Mapping):
            payload = candidate

    if payload is not None:
        output_config = payload.get("output_config")
        output_config = output_config if isinstance(output_config, Mapping) else {}
        reasoning = payload.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, Mapping) else {}
        for name in selected:
            if name == "reasoning_effort":
                # Chat Completions, Responses, and Anthropic use different
                # payload fields for the same control. Preserve explicit nulls.
                controls[name] = payload.get(
                    name, reasoning.get("effort", output_config.get("effort"))
                )
            else:
                controls[name] = payload.get(name)
    else:
        for name in selected:
            attribute = "effort" if name == "reasoning_effort" else name
            controls[name] = getattr(chat_model, name, getattr(chat_model, attribute, None))
    return _forwarded_controls(controls)


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
    if (
        candidate is None
        or "://" in candidate
        or "@" in candidate
        or looks_like_credential(candidate)
    ):
        return None
    return candidate


def _model_label(value: object, fallback: str = "unknown") -> str:
    return _strict_model_label(value) or _strict_model_label(fallback) or "unknown"


def provider_name(provider: object) -> str:
    """Return a stable provider label without endpoint or credential data."""
    names = {
        "AntigravityCLIProvider": "antigravity_cli",
        "AnthropicProvider": "anthropic",
        "AnthropicProxyProvider": "anthropic_proxy",
        "AzureOpenAIProvider": "azure_openai",
        "BedrockProvider": "bedrock",
        "ClaudeCLIProvider": "claude_cli",
        "CodexCLIProvider": "codex_cli",
        "CopilotCLIProvider": "copilot_cli",
        "GeminiCLIProvider": "gemini_cli",
        "NvBuildProvider": "nv_build",
        "NvInferenceProvider": "nv_inference",
        "OllamaProvider": "ollama",
        "OpenAICompatibleProvider": "openai_compatible",
        "OpenAIProvider": "openai",
        "OpencodeCLIProvider": "opencode_cli",
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
        requested_controls: Mapping[str, object] | None = None,
        forwarded_controls: Mapping[str, object] | None = None,
    ) -> None:
        self._node = node
        self._request_kind = request_kind
        self._provider = provider
        self._requested_model = requested_model
        self._requested_controls = _forwarded_controls(requested_controls)
        self._forwarded_controls = _forwarded_controls(forwarded_controls)
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
                if self._requested_controls:
                    record["requested_controls"] = self._requested_controls.copy()
                if self._forwarded_controls:
                    record["forwarded_controls"] = self._forwarded_controls.copy()
                self._records.append(record)
            else:
                self._records.append(self._response_observation())

    def _response_observation(self) -> InferenceUsageRecord:
        """Build counter-less, internal-only evidence of a completed response."""
        return {
            "node": _label(self._node),
            "request_kind": _label(self._request_kind),
            "provider": _label(self._provider),
            "model": _model_label(self._requested_model),
            "model_source": "requested_model",
            "usage_source": "provider_response",
            "requested_controls": self._requested_controls.copy(),
            "forwarded_controls": self._forwarded_controls.copy(),
        }

    def mark_response_received(self) -> None:
        """Record a completed response from a non-LangChain transport."""
        with self._lock:
            self._response_received = True
            self._records.append(self._response_observation())

    def set_provider(self, provider: str) -> None:
        """Set the effective provider before the first response is observed."""
        label = _label(provider)
        with self._lock:
            if self._response_received and label != self._provider:
                raise RuntimeError("cannot change inference provider after a response")
            self._provider = label

    def set_controls(
        self,
        requested: Mapping[str, object] | None,
        forwarded: Mapping[str, object] | None,
    ) -> None:
        """Update constructor/request evidence before the next response."""
        with self._lock:
            self._requested_controls = _forwarded_controls(requested)
            self._forwarded_controls = _forwarded_controls(forwarded)

    @property
    def response_received(self) -> bool:
        """Whether the provider returned, even when it reported no token usage."""
        with self._lock:
            return self._response_received

    def snapshot(self) -> list[InferenceUsageRecord]:
        """Return usage plus counter-less response evidence for provenance."""
        with self._lock:
            snapshot: list[InferenceUsageRecord] = []
            for record in self._records:
                detached = record.copy()
                controls = record.get("forwarded_controls")
                if isinstance(controls, dict):
                    detached["forwarded_controls"] = controls.copy()
                requested = record.get("requested_controls")
                if isinstance(requested, dict):
                    detached["requested_controls"] = requested.copy()
                snapshot.append(detached)
            return snapshot


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
