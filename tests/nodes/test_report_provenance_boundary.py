# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential boundaries for captured and response-derived report provenance."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_openai import ChatOpenAI
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.llm_provenance import LLM_ANALYZER_SLOTS, capture_llm_provenance
from skillspector.llm_utils import new_inference_usage_collector
from skillspector.nodes.report import report
from skillspector.providers.chat_models import create_openai_compatible_chat_model
from skillspector.state import SkillspectorState

_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.synthetic_signature"
_AUTHORIZATION = "Bearer synthetic-secret-token-123456"
_LONG_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


def test_responses_api_effort_matches_emitted_request_in_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK's Responses request and final public provenance must agree."""
    provider = type("OpenAIProvider", (), {})()
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: provider)
    monkeypatch.setattr("skillspector.llm_provenance.get_model_config_provider", lambda: provider)
    monkeypatch.setattr("skillspector.llm_utils.get_active_provider", lambda: provider)
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "high")
    monkeypatch.delenv("SKILLSPECTOR_TEMPERATURE", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_SEED", raising=False)
    model = "gpt-5.4-pro"
    captured = capture_llm_provenance(dict.fromkeys(LLM_ANALYZER_SLOTS, model))
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": model,
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "skillspector.providers.chat_models.ChatOpenAI",
            lambda **kwargs: ChatOpenAI(http_client=client, **kwargs),
        )
        chat_model = create_openai_compatible_chat_model(
            model=model,
            credentials=("test-key", "https://provider.example/v1"),
            max_tokens=128,
        )
        assert chat_model is not None
        collector = new_inference_usage_collector(
            node="semantic_developer_intent",
            request_kind="chat_completion",
            model=model,
            chat_model=chat_model,
        )
        response = chat_model.invoke("Return ok", config={"callbacks": [collector]})

    assert response.text == "ok"
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in payload
    assert "temperature" not in payload
    assert "seed" not in payload
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": True,
        "llm_call_log": [],
        "inference_usage": collector.snapshot(),
        "llm_provenance": captured,
    }

    metadata = json.loads(report(state)["report_body"])["metadata"]
    provenance = metadata["llm_provenance"]

    assert metadata["inference_usage"][0]["total_tokens"] == 7
    assert provenance["provider"]["effective_adapter"] == "openai"
    assert provenance["sampling"]["reasoning_effort"]["requested"] == "high"
    assert (
        provenance["sampling"]["reasoning_effort"]["forwarded_to_client"]
        == (payload["reasoning"]["effort"])
    )
    assert provenance["determinism"]["control_status"] == "best_effort_controls_forwarded"


def test_no_llm_cli_report_rejects_credentials_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    safe_skill_dir: Path,
    tmp_path: Path,
) -> None:
    """Configuration capture must be safe even when no provider call is made."""
    provider = type("AzureOpenAIProvider", (), {})()
    monkeypatch.setattr("skillspector.llm_provenance.get_active_provider", lambda: provider)
    monkeypatch.setattr("skillspector.llm_provenance.get_model_config_provider", lambda: provider)
    monkeypatch.setenv("SKILLSPECTOR_MODEL", _JWT)
    monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", _AUTHORIZATION)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", _JWT)
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", _JWT)
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        ["scan", str(safe_skill_dir), "--no-llm", "--format", "json", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    serialized = output.read_text()
    provenance = json.loads(serialized)["metadata"]["llm_provenance"]
    assert _JWT not in serialized
    assert _AUTHORIZATION not in serialized
    assert {item["model"] for item in provenance["analyzers"]} == {"redacted"}
    assert provenance["sampling"]["reasoning_effort"]["requested"] is None
    assert provenance["sampling"]["reasoning_effort"]["source"] == "invalid_environment"
    assert provenance["provider"]["routing"]["deployment_override"] is None
    assert provenance["provider"]["routing"]["api_version"] is None


@pytest.mark.parametrize("use_llm", [False, True])
def test_report_rejects_credentials_in_raw_provenance_and_response_controls(
    monkeypatch: pytest.MonkeyPatch, use_llm: bool
) -> None:
    """Final serialization revalidates raw state instead of trusting capture."""
    monkeypatch.setattr("skillspector.nodes.report.is_llm_available", lambda: (True, None))
    state: SkillspectorState = {
        "filtered_findings": [],
        "component_metadata": [],
        "has_executable_scripts": False,
        "manifest": {},
        "output_format": "json",
        "use_llm": use_llm,
        "llm_call_log": [],
        "inference_usage": [
            {
                "node": "semantic_developer_intent",
                "request_kind": "structured_output",
                "provider": "openai",
                "model": model,
                "model_source": "provider_response",
                "usage_source": "provider_response",
                "total_tokens": 1,
                "requested_controls": {"reasoning_effort": _AUTHORIZATION},
                "forwarded_controls": {"reasoning_effort": _AUTHORIZATION},
            }
            for model in (_JWT, _LONG_MODEL)
        ],
        "llm_provenance": {
            "provider": {
                "configured_adapter": "azure_openai",
                "resolved_adapter": "azure_openai",
                "routing": {
                    "deployment_override": _JWT,
                    "deployment_source": "environment",
                    "api_version": _JWT,
                    "api_version_source": "environment",
                },
            },
            "analyzers": [{"analyzer_id": slot, "model": _JWT} for slot in LLM_ANALYZER_SLOTS],
            "sampling": {
                "reasoning_effort": {
                    "requested": _AUTHORIZATION,
                    "source": "environment",
                    "adapter_support": True,
                }
            },
        },
    }

    serialized = report(state)["report_body"]
    metadata = json.loads(serialized)["metadata"]
    provenance = metadata["llm_provenance"]

    assert _JWT not in serialized
    assert _AUTHORIZATION not in serialized
    assert {item["model"] for item in provenance["analyzers"]} == {"redacted"}
    assert provenance["sampling"]["reasoning_effort"]["requested"] is None
    assert provenance["sampling"]["reasoning_effort"]["forwarded_to_client"] is None
    assert provenance["provider"]["routing"]["deployment_override"] is None
    assert provenance["provider"]["routing"]["api_version"] is None
    assert [record["model"] for record in metadata["inference_usage"]] == [_LONG_MODEL]


def test_no_llm_cli_report_preserves_long_model_identifiers(
    monkeypatch: pytest.MonkeyPatch, safe_skill_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_MODEL", _LONG_MODEL)
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        ["scan", str(safe_skill_dir), "--no-llm", "--format", "json", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    provenance = json.loads(output.read_text())["metadata"]["llm_provenance"]
    assert {item["model"] for item in provenance["analyzers"]} == {_LONG_MODEL}


def test_no_llm_cli_report_preserves_sanitized_azure_routing_without_execution(
    monkeypatch: pytest.MonkeyPatch, safe_skill_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "production-v2")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        ["scan", str(safe_skill_dir), "--no-llm", "--format", "json", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    provider = json.loads(output.read_text())["metadata"]["llm_provenance"]["provider"]
    assert provider == {
        "configured_adapter": "azure_openai",
        "resolved_adapter": "unknown",
        "effective_adapter": "not_applicable",
        "effective_adapters": [],
        "service": "unknown",
        "routing": {
            "deployment_override": "production-v2",
            "deployment_source": "environment",
            "api_version": "2025-01-01",
            "api_version_source": "environment",
        },
    }
