# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential boundaries for captured and response-derived report provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.llm_provenance import LLM_ANALYZER_SLOTS
from skillspector.nodes.report import report
from skillspector.state import SkillspectorState

_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.synthetic_signature"
_AUTHORIZATION = "Bearer synthetic-secret-token-123456"
_LONG_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


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
