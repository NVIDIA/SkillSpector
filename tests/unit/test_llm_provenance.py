# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sanitized, scan-level LLM provenance."""

from __future__ import annotations

import base64
import json

import pytest

from skillspector.inference_usage import provider_name
from skillspector.llm_provenance import (
    LLM_ANALYZER_SLOTS,
    capture_llm_provenance,
    sanitize_llm_provenance,
)
from skillspector.llm_utils import new_inference_usage_collector
from skillspector.providers.chat_models import create_openai_compatible_chat_model


class OpenAIProvider:
    pass


class AnthropicProvider:
    pass


class AzureOpenAIProvider:
    pass


class NvInferenceProvider:
    pass


def _models(value: str = "safe/model:1") -> dict[str, str]:
    return dict.fromkeys(LLM_ANALYZER_SLOTS, value)


def _usage(
    provider: str,
    controls: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    record: dict[str, object] = {
        "provider": provider,
        "usage_source": "provider_response",
    }
    if controls is not None:
        record["requested_controls"] = controls
        record["forwarded_controls"] = controls
    return [record]


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
    result = sanitize_llm_provenance(
        captured,
        use_llm=True,
        inference_usage=_usage(
            "openai",
            {"temperature": 0.0, "seed": 0, "reasoning_effort": "low"},
        ),
    )

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


def test_editable_source_build_reports_unknown_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EditableDistribution:
        @staticmethod
        def read_text(name: str) -> str | None:
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "file:///workspace/SkillSpector",
                    "dir_info": {"editable": True},
                }
            )

    monkeypatch.delenv("SKILLSPECTOR_BUILD_REVISION", raising=False)
    monkeypatch.setattr(
        "skillspector.llm_provenance.distribution", lambda _name: _EditableDistribution()
    )
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models()),
        use_llm=False,
    )

    assert result["analyzers"][0]["analyzer_revision"]["source_revision"] == {
        "value": "unknown",
        "source": "unknown",
    }


def test_injected_build_revision_is_reported_separately_from_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    monkeypatch.setenv("SKILLSPECTOR_BUILD_REVISION", revision)
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models()),
        use_llm=False,
    )
    analyzer_revision = result["analyzers"][0]["analyzer_revision"]

    assert analyzer_revision["source"] == "skillspector_package"
    assert analyzer_revision["source_revision"] == {
        "value": revision.lower(),
        "source": "build_environment",
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
        inference_usage=_usage(
            "anthropic",
            {"temperature": 0.1, "reasoning_effort": None},
        ),
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


def test_constructor_observation_supersedes_stale_configuration_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forwarded controls come from the client constructor, not preflight state."""
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setattr("skillspector.llm_utils.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.2")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "11")
    captured = capture_llm_provenance(_models("gpt-4o"))

    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.9")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "99")
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "high")

    chat_model = create_openai_compatible_chat_model(
        model="gpt-4o",
        credentials=("test-key", None),
        max_tokens=128,
    )
    collector = new_inference_usage_collector(
        node="semantic_developer_intent",
        request_kind="structured_output",
        model="gpt-4o",
        chat_model=chat_model,
    )
    collector.mark_response_received()

    result = sanitize_llm_provenance(
        captured,
        use_llm=True,
        inference_usage=collector.snapshot(),
    )

    assert result["sampling"]["temperature"]["requested"] == 0.9
    assert result["sampling"]["temperature"]["forwarded_to_client"] == 0.9
    assert result["sampling"]["seed"]["requested"] == 99
    assert result["sampling"]["seed"]["forwarded_to_client"] == 99
    assert result["sampling"]["reasoning_effort"]["requested"] == "high"
    assert result["sampling"]["reasoning_effort"]["forwarded_to_client"] == "high"


def test_gpt_5_4_reports_only_controls_retained_by_request_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setattr(
        "skillspector.llm_provenance.get_model_config_provider", lambda: OpenAIProvider()
    )
    monkeypatch.setattr("skillspector.llm_utils.get_active_provider", lambda: OpenAIProvider())
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.2")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "11")
    captured = capture_llm_provenance(_models("gpt-5.4"))

    chat_model = create_openai_compatible_chat_model(
        model="gpt-5.4",
        credentials=("test-key", None),
        max_tokens=128,
    )
    collector = new_inference_usage_collector(
        node="semantic_developer_intent",
        request_kind="structured_output",
        model="gpt-5.4",
        chat_model=chat_model,
    )
    collector.mark_response_received()

    result = sanitize_llm_provenance(
        captured,
        use_llm=True,
        inference_usage=collector.snapshot(),
    )

    assert result["sampling"]["temperature"] == {
        "requested": 0.2,
        "source": "environment",
        "forwarded_to_client": None,
        "adapter_support": True,
        "provider_support": "unknown",
    }
    assert result["sampling"]["seed"]["requested"] == 11
    assert result["sampling"]["seed"]["forwarded_to_client"] == 11
    assert result["determinism"]["control_status"] == "controls_partially_forwarded"


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


@pytest.mark.parametrize(
    "credential",
    [
        "AWS-secret-fake-value",
        "github_pat_fake-value",
        "xoxb-fake-value",
        "hf_fake-value",
        "AIza-fake-value",
        "Bearer synthetic-secret-token-123456",
        # Build a realistic Basic header from explicit dummy credentials.
        pytest.param(
            "Basic " + base64.b64encode(b"synthetic-user:synthetic-password").decode("ascii"),
            id="synthetic-basic-auth",
        ),
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.fake_signature",
        "ewogICJhbGciOiAibm9uZSIKfQ.eyJzdWIiOiJzeW50aGV0aWMifQ.",
        "0123456789abcdef" * 2,
        "AbCdEfGhIjKlMnOpQrSt" * 2,
    ],
)
def test_capture_and_projection_do_not_emit_credential_shaped_overrides(
    monkeypatch: pytest.MonkeyPatch,
    credential: str,
) -> None:
    provider = AzureOpenAIProvider()
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: provider)
    monkeypatch.setattr("skillspector.llm_provenance.get_model_config_provider", lambda: provider)
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", credential)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", credential)

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models(credential)),
        use_llm=False,
    )
    serialized = json.dumps(result)

    assert credential not in serialized
    assert {item["model"] for item in result["analyzers"]} == {"redacted"}
    assert result["provider"]["routing"]["deployment_override"] is None
    assert result["sampling"]["reasoning_effort"]["requested"] is None


def test_malformed_source_revision_is_not_emitted() -> None:
    secret = "github_pat_fake-value"
    result = sanitize_llm_provenance(
        {
            "analyzers": [
                {
                    "analyzer_id": LLM_ANALYZER_SLOTS[0],
                    "model": "safe/model:1",
                    "analyzer_revision": {
                        "value": "2.11.2",
                        "source_revision": {
                            "value": secret,
                            "source": "build_environment",
                        },
                    },
                }
            ]
        },
        use_llm=False,
    )

    assert secret not in json.dumps(result)
    assert result["analyzers"][0]["analyzer_revision"]["source_revision"] == {
        "value": "unknown",
        "source": "unknown",
    }


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


def test_unknown_provider_specific_reasoning_effort_is_not_published(
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
        inference_usage=_usage(
            "openai",
            {"temperature": None, "seed": None, "reasoning_effort": "provider specific value"},
        ),
    )

    assert result["sampling"]["reasoning_effort"]["requested"] is None
    assert result["sampling"]["reasoning_effort"]["forwarded_to_client"] is None
    assert "provider specific value" not in json.dumps(result)


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
        inference_usage=_usage(
            "openai",
            {"temperature": None, "seed": 17, "reasoning_effort": None},
        ),
    )

    assert result["provider"]["resolved_adapter"] == "bedrock"
    assert result["provider"]["effective_adapter"] == "openai"
    assert result["provider"]["effective_adapters"] == ["openai"]
    assert result["sampling"]["seed"]["adapter_support"] is True
    assert result["sampling"]["seed"]["forwarded_to_client"] == 17


def test_nv_inference_preserves_all_observed_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = NvInferenceProvider()
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: provider)
    monkeypatch.setattr("skillspector.llm_provenance.get_model_config_provider", lambda: provider)
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.25")
    monkeypatch.setenv("SKILLSPECTOR_SEED", "23")
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "high")

    result = sanitize_llm_provenance(
        capture_llm_provenance(_models()),
        use_llm=True,
        inference_usage=_usage(
            "nv_inference",
            {"temperature": 0.25, "seed": 23, "reasoning_effort": "high"},
        ),
    )

    assert result["provider"]["effective_adapter"] == "nv_inference"
    assert result["sampling"]["temperature"]["forwarded_to_client"] == 0.25
    assert result["sampling"]["seed"]["forwarded_to_client"] == 23
    assert result["sampling"]["reasoning_effort"]["forwarded_to_client"] == "high"


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
