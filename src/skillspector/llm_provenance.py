# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanitized, scan-level provenance for LLM-backed analysis configuration."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import distribution, version

from skillspector.inference_usage import looks_like_credential, provider_name, safe_reasoning_effort
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
        "nv_inference",
        "ollama",
        "openai",
        "openai_compatible",
    }
)
_SEED_ADAPTERS = frozenset(
    {"azure_openai", "nv_build", "nv_inference", "ollama", "openai", "openai_compatible"}
)
_REASONING_EFFORT_ADAPTERS = frozenset(
    {
        "anthropic",
        "anthropic_proxy",
        "nv_build",
        "nv_inference",
        "ollama",
        "openai",
        "openai_compatible",
    }
)
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}")
_SAFE_DEPLOYMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]{0,255}")
_SAFE_API_VERSION = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:-preview)?")
_MAX_SAFE_SEED = MAX_SAMPLING_SEED
_MIN_SAFE_SEED = MIN_SAMPLING_SEED
_CONTROL_SOURCES = frozenset(
    {"environment", "provider_default", "unset", "invalid_environment", "out_of_range", "unknown"}
)
_DEPLOYMENT_SOURCES = frozenset({"environment", "resolved_model", "not_applicable", "unknown"})
_API_VERSION_SOURCES = frozenset({"environment", "provider_default", "not_applicable", "unknown"})
_AZURE_OPENAI_DEFAULT_API_VERSION = "2024-06-01"
_SOURCE_REVISION = re.compile(r"[0-9a-fA-F]{7,64}")
_SOURCE_REVISION_SOURCES = frozenset({"build_environment", "package_vcs_metadata", "unknown"})


def _safe_label(value: object, fallback: str = "unknown") -> str:
    candidate = value if isinstance(value, str) else ""
    candidate = candidate.strip()
    if (
        _SAFE_LABEL.fullmatch(candidate)
        and "://" not in candidate
        and "@" not in candidate
        and not looks_like_credential(candidate)
    ):
        return candidate
    return fallback


def _safe_optional_label(value: object) -> str | None:
    """Return a safe label or ``None`` without inventing a placeholder."""
    label = _safe_label(value, fallback="")
    return label or None


def _safe_deployment(value: object) -> str | None:
    """Deployment names have a narrower alphabet than namespaced model IDs."""
    candidate = _safe_optional_label(value)
    return candidate if candidate and _SAFE_DEPLOYMENT.fullmatch(candidate) else None


def _safe_api_version(value: object) -> str | None:
    """Azure API versions are dates with an optional preview suffix."""
    candidate = value.strip() if isinstance(value, str) else ""
    return candidate if _SAFE_API_VERSION.fullmatch(candidate) else None


def _capture_source_revision() -> tuple[str, str]:
    """Return an injected/packaged VCS identity without invoking Git."""
    injected = os.environ.get("SKILLSPECTOR_BUILD_REVISION", "").strip()
    if _SOURCE_REVISION.fullmatch(injected):
        return injected.lower(), "build_environment"

    try:
        direct_url = distribution("skillspector").read_text("direct_url.json")
        metadata = json.loads(direct_url) if direct_url else {}
    except (json.JSONDecodeError, OSError, TypeError):
        metadata = {}
    vcs_info = metadata.get("vcs_info") if isinstance(metadata, Mapping) else None
    vcs_info = vcs_info if isinstance(vcs_info, Mapping) else {}
    packaged = vcs_info.get("commit_id")
    if isinstance(packaged, str) and _SOURCE_REVISION.fullmatch(packaged.strip()):
        return packaged.strip().lower(), "package_vcs_metadata"
    return "unknown", "unknown"


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
    deployment = _safe_deployment(raw_deployment)
    raw_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    api_version = _safe_api_version(raw_api_version or _AZURE_OPENAI_DEFAULT_API_VERSION)
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
    value = safe_reasoning_effort(value)
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
    source_revision, source_revision_source = _capture_source_revision()

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
                "source_revision": {
                    "value": source_revision,
                    "source": source_revision_source,
                },
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


def _sanitize_observed_control_values(
    name: str,
    values: Sequence[object],
) -> tuple[list[float | int | str | None], bool]:
    observed: list[float | int | str | None] = []
    invalid = False
    for candidate in values:
        if candidate is None:
            observed.append(None)
        elif name == "temperature":
            if (
                isinstance(candidate, (int, float))
                and not isinstance(candidate, bool)
                and math.isfinite(candidate)
                and 0 <= candidate <= 1
            ):
                observed.append(float(candidate))
            else:
                invalid = True
        elif name == "seed":
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and _MIN_SAFE_SEED <= candidate <= _MAX_SAFE_SEED
            ):
                observed.append(candidate)
            else:
                invalid = True
        else:
            setting = safe_reasoning_effort(candidate)
            if setting:
                observed.append(setting)
            else:
                invalid = True
    return observed, invalid


def _sanitize_control(
    name: str,
    value: object,
    *,
    use_llm: bool,
    effective_adapters: Sequence[str],
    observed_requested_values: Sequence[object],
    observed_forwarded_values: Sequence[object],
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
        requested = safe_reasoning_effort(requested)
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

    # The constructor-time request supersedes the provisional preflight
    # capture. Forwarding is derived separately from the normalized provider
    # payload because a model adapter may accept and then omit a control.
    forwarded: float | int | str | None = None
    if use_llm and adapter_support and observed_requested_values:
        observed, invalid_observation = _sanitize_observed_control_values(
            name, observed_requested_values
        )
        distinct = []
        for candidate in observed:
            if candidate not in distinct:
                distinct.append(candidate)
        if (
            invalid_observation
            or len(observed) != len(observed_requested_values)
            or len(distinct) != 1
        ):
            requested = None
            source = "unknown"
        else:
            actual = distinct[0]
            if actual is None:
                requested = None
                source = "unset" if name == "seed" else "provider_default"
            else:
                requested = actual
                source = "environment"

    if use_llm and adapter_support and observed_forwarded_values:
        observed, invalid_observation = _sanitize_observed_control_values(
            name, observed_forwarded_values
        )
        distinct = []
        for candidate in observed:
            if candidate not in distinct:
                distinct.append(candidate)
        if (
            not invalid_observation
            and len(observed) == len(observed_forwarded_values)
            and len(distinct) == 1
            and distinct[0] is not None
        ):
            forwarded = distinct[0]

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
    deployment = _safe_deployment(raw.get("deployment_override"))
    deployment_source = raw.get("deployment_source")
    if not isinstance(deployment_source, str) or deployment_source not in _DEPLOYMENT_SOURCES:
        deployment_source = "unknown"
    if deployment_source == "environment" and deployment is None:
        deployment_source = "unknown"
    elif deployment_source != "environment":
        deployment = None

    api_version = _safe_api_version(raw.get("api_version"))
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


def _provider_response_records(inference_usage: object) -> list[Mapping[object, object]]:
    """Return internal provider-response evidence, including counter-less calls."""
    records = inference_usage if isinstance(inference_usage, Sequence) else []
    return [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("usage_source") == "provider_response"
    ]


def _effective_adapters(records: Sequence[Mapping[object, object]]) -> list[str]:
    """Return providers proven by provider-response telemetry."""
    adapters = {
        adapter
        for record in records
        if (adapter := _safe_optional_label(record.get("provider"))) is not None
    }
    return sorted(adapters)


def _observed_controls(
    records: Sequence[Mapping[object, object]],
    field: str,
) -> dict[str, list[object]]:
    """Collect fixed-field constructor controls from successful calls."""
    observed = {name: [] for name in ("temperature", "seed", "reasoning_effort")}
    for record in records:
        controls = record.get(field)
        if not isinstance(controls, Mapping):
            continue
        for name in observed:
            if name in controls:
                observed[name].append(controls.get(name))
    return observed


def _expected_observed_controls(effective_adapters: Sequence[str]) -> set[str]:
    """Return controls whose constructor state is observable for all adapters."""
    expected: set[str] = set()
    for name, supported in (
        ("temperature", _TEMPERATURE_ADAPTERS),
        ("seed", _SEED_ADAPTERS),
        ("reasoning_effort", _REASONING_EFFORT_ADAPTERS),
    ):
        if effective_adapters and all(adapter in supported for adapter in effective_adapters):
            expected.add(name)
    return expected


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
    response_records = _provider_response_records(inference_usage) if use_llm else []
    effective_adapters = _effective_adapters(response_records)
    observed_requested_controls = _observed_controls(response_records, "requested_controls")
    observed_forwarded_controls = _observed_controls(response_records, "forwarded_controls")
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
        raw_source_revision = revision.get("source_revision")
        raw_source_revision = (
            raw_source_revision if isinstance(raw_source_revision, Mapping) else {}
        )
        source_revision = raw_source_revision.get("value")
        source_revision = (
            source_revision.lower()
            if isinstance(source_revision, str) and _SOURCE_REVISION.fullmatch(source_revision)
            else "unknown"
        )
        source_revision_source = raw_source_revision.get("source")
        if (
            not isinstance(source_revision_source, str)
            or source_revision_source not in _SOURCE_REVISION_SOURCES
            or source_revision == "unknown"
        ):
            source_revision_source = "unknown"
        analyzers.append(
            {
                "analyzer_id": slot,
                "model": _safe_label(item.get("model"), fallback="redacted"),
                "model_source": "resolved_configuration",
                "analyzer_revision": {
                    "value": _safe_label(revision.get("value")),
                    "source": "skillspector_package",
                    "source_revision": {
                        "value": source_revision,
                        "source": source_revision_source,
                    },
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
            observed_requested_values=observed_requested_controls[name],
            observed_forwarded_values=observed_forwarded_controls[name],
        )
        for name in ("temperature", "seed", "reasoning_effort")
    }
    expected_observations = _expected_observed_controls(effective_adapters)
    controls_observed = all(
        all(
            isinstance(record.get("requested_controls"), Mapping)
            and name in record.get("requested_controls", {})
            and isinstance(record.get("forwarded_controls"), Mapping)
            and name in record.get("forwarded_controls", {})
            for record in response_records
        )
        for name in expected_observations
    )
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
    elif not effective_adapters or not controls_observed:
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
                else "LLM analysis was not executed for this scan."
            ),
        },
    }
