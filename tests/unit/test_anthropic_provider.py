# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Endpoint-specific temperature compatibility for native Anthropic requests."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from skillspector.inference_usage import chat_model_controls, chat_model_requested_controls
from skillspector.providers.anthropic import AnthropicProvider
from skillspector.providers.anthropic_proxy import AnthropicProxyProvider
from skillspector.providers.bedrock import BedrockProvider
from skillspector.providers.chat_models import (
    UnsupportedControlError,
    create_openai_compatible_chat_model,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_SCHEME",
        "ANTHROPIC_PROXY_API_KEY",
        "ANTHROPIC_PROXY_ENDPOINT_URL",
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_TEMPERATURE",
        "SKILLSPECTOR_SEED",
        "SKILLSPECTOR_REASONING_EFFORT",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
@pytest.mark.parametrize(
    "base_url", [None, "https://api.anthropic.com", "https://api.anthropic.com/"]
)
def test_native_claude_55_accepts_default_temperature_in_request_and_provenance(
    monkeypatch: pytest.MonkeyPatch, model: str, base_url: str | None
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "1.0")
    if base_url is not None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)

    llm = AnthropicProvider().create_chat_model(model, max_tokens=1_000)

    assert llm is not None
    assert llm._get_request_payload([HumanMessage(content="analyse this")])["temperature"] == 1.0
    assert chat_model_requested_controls(llm)["temperature"] == 1.0
    assert chat_model_controls(llm)["temperature"] == 1.0


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
@pytest.mark.parametrize("temperature", ["0", "0.2", "0.999999"])
def test_native_claude_55_rejects_nondefault_temperature(
    monkeypatch: pytest.MonkeyPatch, model: str, temperature: str
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", temperature)

    with pytest.raises(UnsupportedControlError, match="SKILLSPECTOR_TEMPERATURE"):
        AnthropicProvider().create_chat_model(model, max_tokens=1_000)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.example.com",
        "https://api.anthropic.com.example.com",
        "http://api.anthropic.com",
        "https://api.anthropic.com/proxy",
        "https://api.anthropic.com?route=proxy",
    ],
)
def test_anthropic_endpoint_override_does_not_inherit_native_temperature_compatibility(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "1.0")

    with pytest.raises(UnsupportedControlError, match="SKILLSPECTOR_TEMPERATURE"):
        AnthropicProvider().create_chat_model("claude-opus-5-5", max_tokens=1_000)


def test_native_endpoint_never_relaxes_gpt_6_1_sol_temperature_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "1.0")

    with pytest.raises(UnsupportedControlError, match="SKILLSPECTOR_TEMPERATURE"):
        AnthropicProvider().create_chat_model("gpt-6.1-sol", max_tokens=1_000)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_native_claude_55_omits_unrequested_temperature(model: str) -> None:
    llm = AnthropicProvider().create_chat_model(model, max_tokens=1_000)

    assert llm is not None
    assert "temperature" not in llm._get_request_payload([HumanMessage(content="analyse this")])
    assert chat_model_requested_controls(llm)["temperature"] is None


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_anthropic_proxy_still_rejects_default_temperature(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "proxy-test")
    monkeypatch.setenv("ANTHROPIC_PROXY_ENDPOINT_URL", "https://proxy.example.com/predict")
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "1.0")

    with pytest.raises(UnsupportedControlError, match="SKILLSPECTOR_TEMPERATURE"):
        AnthropicProxyProvider().create_chat_model(model, max_tokens=1_000)


@pytest.mark.parametrize("model", ["azure/anthropic/claude-opus-5-5", "claude-sonnet-5-5"])
def test_openai_compatible_gateway_still_rejects_default_temperature(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "1.0")

    with pytest.raises(UnsupportedControlError, match="SKILLSPECTOR_TEMPERATURE"):
        create_openai_compatible_chat_model(
            model=model,
            credentials=("gateway-test", "https://gateway.example.com/v1"),
            max_tokens=1_000,
        )


@pytest.mark.parametrize(
    "model", ["us.anthropic.claude-opus-5-5", "us.anthropic.claude-sonnet-5-5"]
)
def test_bedrock_still_rejects_default_temperature(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "1.0")

    with pytest.raises(UnsupportedControlError, match="SKILLSPECTOR_TEMPERATURE"):
        BedrockProvider().create_chat_model(model, max_tokens=1_000)
