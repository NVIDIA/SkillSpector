# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized, scan-level provenance for LLM-backed analysis configuration."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import version

from skillspector.inference_usage import provider_name
from skillspector.providers import get_active_provider, get_model_config_provider
from skillspector.providers.chat_models import (
    MAX_SAMPLING_SEED,
    MIN_SAMPLING_SEED,
    resolve_reasoning_effort,
    resolve_seed,
    resolve_temperature,
)

LLM_PROVENANCE_SCHEMA_VERSION = 1
LLM_ANALYZER_SLOTS = (
    "mcp_tool_poisoning",
    "semantic_developer_intent",
    "semantic_quality_policy",
    "semantic_security_discovery",
    "meta_analyzer",
)

_TEMPERATURE_ADAPTERS = frozenset(
    {
        "anthropic",
        "anthropic_proxy",
        "azure_openai",
        "bedrock",
        "nv_build",
        "ollama",
        "openai",
        "openai_compatible",
    }
)
_SEED_ADAPTERS = frozenset({"azure_openai", "nv_build", "ollama", "openai", "openai_compatible"})
_REASONING_EFFORT_ADAPTERS = frozenset(
    {
        "anthropic",
        "anthropic_proxy",
        "nv_build",
        "ollama",
        "openai",
        "openai_compatible",
    }
)
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}")
_SAFE_SETTING = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:/+\-]{0,255}")
_SECRET_PREFIXES = ("sk-", "nvapi-", "ghp_", "glpat-", "bearer-")
_MAX_SAFE_SEED = MAX_SAMPLING_SEED
_MIN_SAFE_SEED = MIN_SAMPLING_SEED
_CONTROL_SOURCES = frozenset(
    {"environment", "provider_default", "unset", "invalid_environment", "out_of_range", "unknown"}
)
_DEPLOYMENT_SOURCES = frozenset({"environment", "resolved_model", "not_applicable", "unknown"})
_API_VERSION_SOURCES = frozenset({"environment", "provider_default", "not_applicable", "unknown"})
_AZURE_OPENAI_DEFAULT_API_VERSION = "2024-06-01"


def _safe_label(value: object, fallback: str = "unknown") -> str:
    candidate = value if isinstance(value, str) else ""
    candidate = candidate.strip()
    lowered = candidate.lower()
    if (
        _SAFE_LABEL.fullmatch(candidate)
        and "://" not in candidate
        and "@" not in candidate
        and not lowered.startswith(_SECRET_PREFIXES)
    ):
        return candidate
    return fallback


def _safe_setting(value: object, fallback: str = "unknown") -> str:
    """Return a bounded printable setting while allowing provider-specific spaces."""
    candidate = value if isinstance(value, str) else ""
    candidate = candidate.strip()
    lowered = candidate.lower()
    if (
        _SAFE_SETTING.fullmatch(candidate)
        and "://" not in candidate
        and "@" not in candidate
        and not lowered.startswith(_SECRET_PREFIXES)
    ):
        return candidate
    return fallback


def _safe_optional_label(value: object) -> str | None:
    """Return a safe label or ``None`` without inventing a placeholder."""
    label = _safe_label(value, fallback="")
    return label or None


def _capture_provider_routing(resolved_adapter: str) -> dict[str, object]:
    """Capture non-secret provider routing inputs used by client construction."""
    if resolved_adapter != "azure_openai":
        return {
            "deployment_override": None,
            "deployment_source": "not_applicable",
            "api_version": None,
            "api_version_source": "not_applicable",
        }

    raw_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    deployment = _safe_optional_label(raw_deployment)
    raw_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    api_version = _safe_optional_label(raw_api_version or _AZURE_OPENAI_DEFAULT_API_VERSION)
    return {
        "deployment_override": deployment,
        "deployment_source": (
            "environment" if deployment else "unknown" if raw_deployment else "resolved_model"
        ),
        "api_version": api_version,
        "api_version_source": (
            "environment"
            if raw_api_version and api_version
            else "unknown"
            if raw_api_version
            else "provider_default"
        ),
    }


def _requested_temperature(raw: str) -> tuple[float | None, str]:
    if not raw:
        return None, "provider_default"
    try:
        value = resolve_temperature()
    except ValueError:
        return None, "invalid_environment"
    if value is None:
        return None, "provider_default"
    return value, "environment"


def _requested_seed(raw: str) -> tuple[int | None, str]:
    if not raw:
        return None, "unset"
    try:
        value = resolve_seed()
    except ValueError:
        return None, "out_of_range" if raw.lstrip("+-").isdigit() else "invalid_environment"
    if value is None:
        return None, "unset"
    return value, "environment"


def _requested_effort(raw: str) -> tuple[str | None, str]:
    if not raw:
        return None, "provider_default"
    value = resolve_reasoning_effort()
    value = _safe_setting(value, fallback="")
    return (value, "environment") if value else (None, "invalid_environment")


def _control(
    requested: float | int | str | None,
    source: str,
    *,
    adapter_support: bool,
) -> dict[str, object]:
    return {
        "requested": requested,
        "source": source,
        # Client forwarding is confirmed later from provider-response telemetry.
        # Configuration capture alone cannot prove that an analyzer made a call.
        "forwarded_to_client": None,
        "adapter_support": adapter_support,
        # SkillSpector can prove what it passed to the client constructor, not
        # whether a gateway/model honored the optional control.
        "provider_support": "unknown",
    }


def capture_llm_provenance(model_config: Mapping[str, object]) -> dict[str, object]:
    """Capture resolved LLM configuration once, before analyzer execution."""
    configured_adapter = provider_name(get_active_provider())
    # This provider resolves model defaults before any client is constructed.
    # Runtime response telemetry supplies the actual effective provider later.
    resolved_adapter = provider_name(get_model_config_provider())
    package_version = version("skillspector")

    temperature, temperature_source = _requested_temperature(
        os.environ.get("SKILLSPECTOR_TEMPERATURE", "").strip()
    )
    seed, seed_source = _requested_seed(os.environ.get("SKILLSPECTOR_SEED", "").strip())
    effort, effort_source = _requested_effort(
        os.environ.get("SKILLSPECTOR_REASONING_EFFORT", "").strip()
    )

    analyzers = [
        {
            "analyzer_id": slot,
            "model": _safe_label(model_config.get(slot), fallback="redacted"),
            "model_source": "resolved_configuration",
            "analyzer_revision": {
                "value": package_version,
                "source": "skillspector_package",
            },
        }
        for slot in LLM_ANALYZER_SLOTS
    ]
    sampling = {
        "temperature": _control(
            temperature,
            temperature_source,
            adapter_support=resolved_adapter in _TEMPERATURE_ADAPTERS,
        ),
        "seed": _control(
            seed,
            seed_source,
            adapter_support=resolved_adapter in _SEED_ADAPTERS,
        ),
        "reasoning_effort": _control(
            effort,
            effort_source,
            adapter_support=resolved_adapter in _REASONING_EFFORT_ADAPTERS,
        ),
    }
    return {
        "schema_version": LLM_PROVENANCE_SCHEMA_VERSION,
        "provider": {
            "configured_adapter": _safe_label(configured_adapter),
            "resolved_adapter": _safe_label(resolved_adapter),
            "routing": _capture_provider_routing(resolved_adapter),
            "service": "unknown",
        },
        "analyzers": analyzers,
        "sampling": sampling,
    }


def _sanitize_control(
    name: str,
    value: object,
    *,
    use_llm: bool,
    effective_adapters: Sequence[str],
) -> dict[str, object]:
    raw = value if isinstance(value, Mapping) else {}
    source = raw.get("source")
    source = source if isinstance(source, str) and source in _CONTROL_SOURCES else "unknown"
    adapter_support = raw.get("adapter_support")
    adapter_support = adapter_support if isinstance(adapter_support, bool) else False
    if effective_adapters:
        supported_adapters = (
            _TEMPERATURE_ADAPTERS
            if name == "temperature"
            else _SEED_ADAPTERS
            if name == "seed"
            else _REASONING_EFFORT_ADAPTERS
        )
        adapter_support = all(adapter in supported_adapters for adapter in effective_adapters)

    raw_requested = raw.get("requested")
    requested = raw_requested
    if name == "temperature":
        requested = (
            float(requested)
            if isinstance(requested, (int, float))
            and not isinstance(requested, bool)
            and math.isfinite(requested)
            and 0 <= requested <= 1
            else None
        )
    elif name == "seed":
        requested = (
            requested
            if isinstance(requested, int)
            and not isinstance(requested, bool)
            and _MIN_SAFE_SEED <= requested <= _MAX_SAFE_SEED
            else None
        )
    else:
        requested = _safe_setting(requested, fallback="") or None
    if source == "environment" and requested is None:
        source = (
            "out_of_range"
            if name == "seed"
            and isinstance(raw_requested, int)
            and not isinstance(raw_requested, bool)
            else "invalid_environment"
        )
    if source != "environment":
        requested = None

    # A configured client value is only reportable after provider-response
    # telemetry proves that an analyzer call reached an effective adapter.
    forwarded = requested if effective_adapters and adapter_support else None
    if name == "temperature":
        forwarded = (
            float(forwarded)
            if isinstance(forwarded, (int, float))
            and not isinstance(forwarded, bool)
            and math.isfinite(forwarded)
            and 0 <= forwarded <= 1
            else None
        )
    elif name == "seed":
        forwarded = (
            forwarded
            if isinstance(forwarded, int)
            and not isinstance(forwarded, bool)
            and _MIN_SAFE_SEED <= forwarded <= _MAX_SAFE_SEED
            else None
        )
    else:
        forwarded = _safe_setting(forwarded, fallback="") or None
    if not use_llm or not adapter_support or source != "environment" or forwarded != requested:
        forwarded = None

    return {
        "requested": requested,
        "source": source,
        "forwarded_to_client": forwarded,
        "adapter_support": adapter_support,
        "provider_support": "unknown",
    }


def _sanitize_provider_routing(value: object, *, resolved_adapter: str) -> dict[str, object]:
    """Return the fixed routing contract without endpoints or credentials."""
    if resolved_adapter == "unknown":
        return {
            "deployment_override": None,
            "deployment_source": "unknown",
            "api_version": None,
            "api_version_source": "unknown",
        }
    if resolved_adapter != "azure_openai":
        return {
            "deployment_override": None,
            "deployment_source": "not_applicable",
            "api_version": None,
            "api_version_source": "not_applicable",
        }

    raw = value if isinstance(value, Mapping) else {}
    deployment = _safe_optional_label(raw.get("deployment_override"))
    deployment_source = raw.get("deployment_source")
    if not isinstance(deployment_source, str) or deployment_source not in _DEPLOYMENT_SOURCES:
        deployment_source = "unknown"
    if deployment_source == "environment" and deployment is None:
        deployment_source = "unknown"
    elif deployment_source != "environment":
        deployment = None

    api_version = _safe_optional_label(raw.get("api_version"))
    api_version_source = raw.get("api_version_source")
    if not isinstance(api_version_source, str) or api_version_source not in _API_VERSION_SOURCES:
        api_version_source = "unknown"
    if api_version_source in {"environment", "provider_default"} and api_version is None:
        api_version_source = "unknown"
    elif api_version_source not in {"environment", "provider_default"}:
        api_version = None

    return {
        "deployment_override": deployment,
        "deployment_source": deployment_source,
        "api_version": api_version,
        "api_version_source": api_version_source,
    }


def _effective_adapters(inference_usage: object) -> list[str]:
    """Return providers proven by sanitized provider-response telemetry."""
    records = inference_usage if isinstance(inference_usage, Sequence) else []
    adapters = {
        adapter
        for record in records
        if isinstance(record, Mapping)
        and record.get("usage_source") == "provider_response"
        and (adapter := _safe_optional_label(record.get("provider"))) is not None
    }
    return sorted(adapters)


def sanitize_llm_provenance(
    value: object,
    *,
    use_llm: bool,
    inference_usage: object = None,
) -> dict[str, object]:
    """Return the fixed public provenance projection without arbitrary state."""
    raw = value if isinstance(value, Mapping) else {}
    raw_provider = raw.get("provider")
    provider = raw_provider if isinstance(raw_provider, Mapping) else {}
    configured_adapter = _safe_label(provider.get("configured_adapter"))
    resolved_adapter = _safe_label(provider.get("resolved_adapter"))
    effective_adapters = _effective_adapters(inference_usage) if use_llm else []
    effective_adapter = (
        "not_applicable"
        if not use_llm
        else effective_adapters[0]
        if len(effective_adapters) == 1
        else "mixed"
        if effective_adapters
        else "unknown"
    )

    raw_analyzers = raw.get("analyzers")
    by_id: dict[str, Mapping[object, object]] = {}
    if isinstance(raw_analyzers, list):
        for item in raw_analyzers:
            if not isinstance(item, Mapping):
                continue
            analyzer_id = item.get("analyzer_id")
            if isinstance(analyzer_id, str) and analyzer_id in LLM_ANALYZER_SLOTS:
                by_id[analyzer_id] = item
    analyzers: list[dict[str, object]] = []
    for slot in LLM_ANALYZER_SLOTS:
        item = by_id.get(slot)
        item = item if isinstance(item, Mapping) else {}
        revision = item.get("analyzer_revision")
        revision = revision if isinstance(revision, Mapping) else {}
        analyzers.append(
            {
                "analyzer_id": slot,
                "model": _safe_label(item.get("model"), fallback="redacted"),
                "model_source": "resolved_configuration",
                "analyzer_revision": {
                    "value": _safe_label(revision.get("value")),
                    "source": "skillspector_package",
                },
            }
        )

    raw_sampling = raw.get("sampling")
    sampling = raw_sampling if isinstance(raw_sampling, Mapping) else {}
    sanitized_sampling = {
        name: _sanitize_control(
            name,
            sampling.get(name),
            use_llm=use_llm,
            effective_adapters=effective_adapters,
        )
        for name in ("temperature", "seed", "reasoning_effort")
    }
    requested_controls = [
        control for control in sanitized_sampling.values() if control["source"] == "environment"
    ]
    if not use_llm:
        control_status = "not_applied"
    elif any(
        control["source"] in {"invalid_environment", "out_of_range"}
        for control in sanitized_sampling.values()
    ):
        control_status = "invalid_configuration"
    elif any(control["source"] == "unknown" for control in sanitized_sampling.values()):
        control_status = "configuration_unknown"
    elif not effective_adapters:
        control_status = "controls_not_observed"
    elif not requested_controls:
        control_status = "provider_defaults"
    elif all(control["forwarded_to_client"] is not None for control in requested_controls):
        control_status = "best_effort_controls_forwarded"
    else:
        control_status = "controls_partially_forwarded"

    return {
        "schema_version": LLM_PROVENANCE_SCHEMA_VERSION,
        "provider": {
            "configured_adapter": configured_adapter,
            "resolved_adapter": resolved_adapter,
            "effective_adapter": effective_adapter,
            "effective_adapters": effective_adapters,
            "service": "unknown",
            "routing": _sanitize_provider_routing(
                provider.get("routing"), resolved_adapter=resolved_adapter
            ),
        },
        "analyzers": analyzers,
        "sampling": sanitized_sampling,
        "determinism": {
            "classification": "nondeterministic" if use_llm else "not_applicable",
            "control_status": control_status,
            "provider_guarantee": False,
            "reason": (
                "Optional controls do not guarantee identical provider output."
                if use_llm
                else "LLM analysis was disabled for this scan."
            ),
        },
    }
