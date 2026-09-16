# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sanitized, scan-level LLM provenance."""

from __future__ import annotations

import json

import pytest

from skillspector.inference_usage import provider_name
from skillspector.llm_provenance import (
    LLM_ANALYZER_SLOTS,
    capture_llm_provenance,
    sanitize_llm_provenance,
)


class OpenAIProvider:
    pass


class AnthropicProvider:
    pass


class AzureOpenAIProvider:
    pass


def _models(value: str = "safe/model:1") -> dict[str, str]:
    return dict.fromkeys(LLM_ANALYZER_SLOTS, value)


def _usage(provider: str) -> list[dict[str, object]]:
    return [{"provider": provider, "usage_source": "provider_response"}]


def test_capture_records_resolved_adapters_models_and_forwarded_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_active_provider", lambda: AnthropicProvider()
    )
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "0")
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "low")

    captured = capture_llm_provenance(_models())
    result = sanitize_llm_provenance(captured, use_llm=True, inference_usage=_usage("openai"))

    assert result["provider"] == {
        "configured_adapter": "anthropic",
        "resolved_adapter": "openai",
        "effective_adapter": "openai",
        "effective_adapters": ["openai"],
        "service": "unknown",
        "routing": {
            "deployment_override": None,
            "deployment_source": "not_applicable",
            "api_version": None,
            "api_version_source": "not_applicable",
        },
    }
    assert [item["analyzer_id"] for item in result["analyzers"]] == list(LLM_ANALYZER_SLOTS)
    assert {item["model"] for item in result["analyzers"]} == {"safe/model:1"}
    assert {item["analyzer_revision"]["source"] for item in result["analyzers"]} == {
        "skillspector_package"
    }
    assert result["sampling"]["temperature"]["forwarded_to_client"] == 0.0
    assert result["sampling"]["seed"]["forwarded_to_client"] == 0
    assert result["sampling"]["reasoning_effort"]["forwarded_to_client"] == "low"
    assert result["determinism"] == {
        "classification": "nondeterministic",
        "control_status": "best_effort_controls_forwarded",
        "provider_guarantee": False,
        "reason": "Optional controls do not guarantee identical provider output.",
    }


def test_seed_is_requested_but_not_claimed_forwarded_for_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_active_provider", lambda: AnthropicProvider()
    )
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: AnthropicProvider()
    )
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.1")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "7")

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models()),
        use_llm=True,
        inference_usage=_usage("anthropic"),
    )

    assert result["sampling"]["temperature"]["forwarded_to_client"] == 0.1
    assert result["sampling"]["seed"] == {
        "requested": 7,
        "source": "environment",
        "forwarded_to_client": None,
        "adapter_support": False,
        "provider_support": "unknown",
    }
    assert result["determinism"]["control_status"] == "controls_partially_forwarded"


def test_capture_is_stable_if_environment_changes_before_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.2")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "11")
    captured = capture_llm_provenance(_models())

    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.9")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "99")
    result = sanitize_llm_provenance(
        captured,
        use_llm=True,
        inference_usage=_usage("openai"),
    )

    assert result["sampling"]["temperature"]["requested"] == 0.2
    assert result["sampling"]["seed"]["requested"] == 11


def test_configured_controls_are_not_claimed_forwarded_without_response_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.2")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "11")

    result = sanitize_llm_provenance(capture_llm_provenance(_models()), use_llm=True)

    assert result["provider"]["effective_adapter"] == "unknown"
    assert all(control["forwarded_to_client"] is None for control in result["sampling"].values())
    assert result["determinism"]["control_status"] == "controls_not_observed"


def test_sanitizer_drops_unverified_forwarding_without_response_evidence() -> None:
    result = sanitize_llm_provenance(
        {
            "sampling": {
                "temperature": {
                    "requested": 0.2,
                    "source": "environment",
                    "forwarded_to_client": 0.2,
                    "adapter_support": True,
                },
                "seed": {
                    "requested": None,
                    "source": "unset",
                    "forwarded_to_client": None,
                    "adapter_support": True,
                },
                "reasoning_effort": {
                    "requested": None,
                    "source": "provider_default",
                    "forwarded_to_client": None,
                    "adapter_support": True,
                },
            }
        },
        use_llm=True,
    )

    assert result["sampling"]["temperature"]["forwarded_to_client"] is None
    assert result["determinism"]["control_status"] == "controls_not_observed"


def test_static_scan_is_explicitly_not_applicable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )

    result = sanitize_llm_provenance(capture_llm_provenance(_models()), use_llm=False)

    assert result["determinism"]["classification"] == "not_applicable"
    assert result["determinism"]["control_status"] == "not_applied"
    assert result["provider"]["effective_adapter"] == "not_applicable"
    assert all(control["forwarded_to_client"] is None for control in result["sampling"].values())


def test_public_projection_drops_unknown_fields_and_redacts_unsafe_labels() -> None:
    malicious = {
        "provider": {
            "configured_adapter": "https://user:secret@provider.test",
            "effective_adapter": "sk-secret-value",
            "endpoint": "https://private.example.test",
            "routing": {
                "deployment_override": "sk-secret-deployment",
                "deployment_source": "environment",
                "api_version": "https://private.example.test/version",
                "api_version_source": "environment",
            },
        },
        "analyzers": [
            {
                "analyzer_id": LLM_ANALYZER_SLOTS[0],
                "model": "nvapi-secret-value",
                "analyzer_revision": {"value": "2.11.2", "prompt": "private prompt"},
                "credentials": "do-not-emit",
            },
            {"analyzer_id": "unknown", "model": "private/model"},
        ],
        "sampling": {
            "temperature": {
                "requested": float("nan"),
                "source": "environment",
                "forwarded_to_client": float("inf"),
                "adapter_support": True,
                "headers": {"authorization": "secret"},
            }
        },
        "raw_prompt": "do-not-emit",
    }

    result = sanitize_llm_provenance(malicious, use_llm=True)
    serialized = json.dumps(result)

    assert result["provider"]["configured_adapter"] == "unknown"
    assert result["provider"]["resolved_adapter"] == "unknown"
    assert result["provider"]["effective_adapter"] == "unknown"
    assert result["provider"]["routing"] == {
        "deployment_override": None,
        "deployment_source": "unknown",
        "api_version": None,
        "api_version_source": "unknown",
    }
    assert result["analyzers"][0]["model"] == "redacted"
    assert len(result["analyzers"]) == len(LLM_ANALYZER_SLOTS)
    assert "secret" not in serialized
    assert "private.example" not in serialized
    assert "do-not-emit" not in serialized
    assert "private prompt" not in serialized


def test_multiple_response_providers_are_reported_as_mixed() -> None:
    result = sanitize_llm_provenance(
        None,
        use_llm=True,
        inference_usage=[*_usage("openai"), *_usage("anthropic")],
    )

    assert result["provider"]["effective_adapter"] == "mixed"
    assert result["provider"]["effective_adapters"] == ["anthropic", "openai"]


def test_public_projection_repairs_inconsistent_control_shapes() -> None:
    result = sanitize_llm_provenance(
        {
            "sampling": {
                "temperature": {
                    "requested": 999,
                    "source": "environment",
                    "forwarded_to_client": 999,
                    "adapter_support": True,
                },
                "seed": {
                    "requested": 1.5,
                    "source": "environment",
                    "forwarded_to_client": 1.5,
                    "adapter_support": True,
                },
                "reasoning_effort": {
                    "requested": "low",
                    "source": "environment",
                    "forwarded_to_client": "high",
                    "adapter_support": True,
                },
            }
        },
        use_llm=True,
    )

    assert result["sampling"]["temperature"]["requested"] is None
    assert result["sampling"]["temperature"]["source"] == "invalid_environment"
    assert result["sampling"]["temperature"]["forwarded_to_client"] is None
    assert result["sampling"]["seed"]["requested"] is None
    assert result["sampling"]["seed"]["source"] == "invalid_environment"
    assert result["sampling"]["seed"]["forwarded_to_client"] is None
    assert result["sampling"]["reasoning_effort"]["requested"] == "low"
    assert result["sampling"]["reasoning_effort"]["forwarded_to_client"] is None
    assert result["determinism"]["control_status"] == "invalid_configuration"


def test_public_projection_ignores_unhashable_analyzer_ids() -> None:
    result = sanitize_llm_provenance(
        {"analyzers": [{"analyzer_id": {}, "model": "private/model"}]},
        use_llm=True,
    )

    assert len(result["analyzers"]) == len(LLM_ANALYZER_SLOTS)
    assert {item["model"] for item in result["analyzers"]} == {"redacted"}


def test_provider_specific_reasoning_effort_is_recorded_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "provider specific value")

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models()),
        use_llm=True,
        inference_usage=_usage("openai"),
    )

    assert result["sampling"]["reasoning_effort"]["requested"] == "provider specific value"
    assert (
        result["sampling"]["reasoning_effort"]["forwarded_to_client"] == "provider specific value"
    )


def test_invalid_and_unknown_controls_are_not_called_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "warm")

    invalid = sanitize_llm_provenance(capture_llm_provenance(_models()), use_llm=True)
    unknown = sanitize_llm_provenance(None, use_llm=True)

    assert invalid["determinism"]["control_status"] == "invalid_configuration"
    assert unknown["determinism"]["control_status"] == "configuration_unknown"


def test_effective_provider_comes_from_response_not_preflight_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Bedrock-to-OpenAI fallback must report the provider that answered."""
    bedrock_provider = type("BedrockProvider", (), {})()
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: bedrock_provider)
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: bedrock_provider
    )
    monkeypatch.setenv("SKILLSPECTOR_SEED", "17")

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models()),
        use_llm=True,
        inference_usage=_usage("openai"),
    )

    assert result["provider"]["resolved_adapter"] == "bedrock"
    assert result["provider"]["effective_adapter"] == "openai"
    assert result["provider"]["effective_adapters"] == ["openai"]
    assert result["sampling"]["seed"]["adapter_support"] is True
    assert result["sampling"]["seed"]["forwarded_to_client"] == 17


def test_azure_routing_records_deployment_and_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_active_provider", lambda: AzureOpenAIProvider()
    )
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: AzureOpenAIProvider()
    )
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "production-v2")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models("gpt-4o")),
        use_llm=True,
        inference_usage=_usage("azure_openai"),
    )

    assert result["analyzers"][0]["model"] == "gpt-4o"
    assert result["provider"]["routing"] == {
        "deployment_override": "production-v2",
        "deployment_source": "environment",
        "api_version": "2025-01-01",
        "api_version_source": "environment",
    }


def test_azure_routing_uses_model_and_default_api_version_without_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_active_provider", lambda: AzureOpenAIProvider()
    )
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: AzureOpenAIProvider()
    )

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models("gpt-4o")),
        use_llm=True,
        inference_usage=_usage("azure_openai"),
    )

    assert result["provider"]["routing"] == {
        "deployment_override": None,
        "deployment_source": "resolved_model",
        "api_version": "2024-06-01",
        "api_version_source": "provider_default",
    }


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("AzureOpenAIProvider", "azure_openai"),
        ("OllamaProvider", "ollama"),
        ("OpenAICompatibleProvider", "openai_compatible"),
        ("AntigravityCLIProvider", "antigravity_cli"),
    ],
)
def test_builtin_provider_names_are_canonical(class_name: str, expected: str) -> None:
    provider_type = type(class_name, (), {})

    assert provider_name(provider_type()) == expected
