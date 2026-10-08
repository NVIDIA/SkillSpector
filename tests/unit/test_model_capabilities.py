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

"""Per-model structured-output routing and request controls across the hosted providers."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from skillspector.llm_utils import bind_structured_output
from skillspector.providers import registry
from skillspector.providers.anthropic import AnthropicProvider
from skillspector.providers.anthropic_proxy import AnthropicProxyProvider
from skillspector.providers.structured_output import (
    claude_model_name,
    rejects_forced_tool_call,
    rejects_sampling_controls,
)


class Verdict(BaseModel):
    ok: bool


_CLAUDE_5_5 = ("claude-opus-5-5", "claude-sonnet-5-5")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "SKILLSPECTOR_PROVIDER",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_TEMPERATURE",
        "SKILLSPECTOR_SEED",
        "SKILLSPECTOR_REASONING_EFFORT",
        "SKILLSPECTOR_STRUCTURED_OUTPUT_METHOD",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_SCHEME",
        "ANTHROPIC_PROXY_API_KEY",
        "ANTHROPIC_PROXY_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


class TestClaudeModelRules:
    """The shared tables and the name normalizer every Claude-serving provider uses."""

    @pytest.mark.parametrize("model", [*_CLAUDE_5_5, "claude-opus-5-5-20260922"])
    def test_claude_5_5_rejects_forced_tool_calls_and_sampling(self, model: str) -> None:
        assert rejects_forced_tool_call(model)
        assert rejects_sampling_controls(model)

    @pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1"])
    def test_fable_and_mythos_reject_forced_tool_calls_only(self, model: str) -> None:
        assert rejects_forced_tool_call(model)
        assert not rejects_sampling_controls(model)

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-6", None],
    )
    def test_other_models_keep_both(self, model: str | None) -> None:
        assert not rejects_forced_tool_call(model)
        assert not rejects_sampling_controls(model)

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5-5", "claude-opus-5-5"),
            ("claude-opus-5-5-20260922", "claude-opus-5-5-20260922"),
            ("azure/anthropic/claude-opus-5-5", "claude-opus-5-5"),
            ("aws/anthropic/bedrock-claude-opus-5-5", "claude-opus-5-5"),
            ("anthropic/claude-sonnet-5-5", "claude-sonnet-5-5"),
            ("anthropic/claude-opus-5.5", "claude-opus-5-5"),
            ("openrouter/anthropic/claude-sonnet-5.5", "claude-sonnet-5-5"),
            ("vertex_ai/claude-opus-5-5@20260922", "claude-opus-5-5"),
            ("azure/anthropic/claude-opus-5-5:latest", "claude-opus-5-5"),
            ("Anthropic/Claude-Opus-5-5", "claude-opus-5-5"),
            ("us.anthropic.claude-opus-5-5", "claude-opus-5-5"),
            ("anthropic.claude-sonnet-5-5-v1:0", "claude-sonnet-5-5-v1"),
            (
                "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-5-5",
                "claude-sonnet-5-5",
            ),
            ("azure/anthropic/claude-opus-5", "claude-opus-5"),
            ("claude-opus-5-50", "claude-opus-5-50"),
            ("gpt-6.1-sol", None),
            ("openai/openai/gpt-6.1-sol", None),
            ("vendor/x/notclaude-opus-5-5", None),
        ],
    )
    def test_claude_model_name(self, model: str, expected: str | None) -> None:
        assert claude_model_name(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "azure/anthropic/claude-sonnet-5-5",
            "aws/anthropic/bedrock-claude-opus-5-5",
            "openrouter/anthropic/claude-opus-5.5",
            "vertex_ai/claude-opus-5-5@20260922",
            "anthropic.claude-opus-5-5-v1:0",
        ],
    )
    def test_normalized_ids_carry_the_claude_5_5_rules(self, model: str) -> None:
        assert rejects_forced_tool_call(claude_model_name(model))
        assert rejects_sampling_controls(claude_model_name(model))

    @pytest.mark.parametrize("model", ["azure/anthropic/claude-opus-5", "claude-opus-5-50"])
    def test_lookalike_ids_keep_both(self, model: str) -> None:
        assert not rejects_forced_tool_call(claude_model_name(model))
        assert not rejects_sampling_controls(claude_model_name(model))


def _anthropic_provider(
    monkeypatch: pytest.MonkeyPatch, provider_cls: type
) -> AnthropicProvider | AnthropicProxyProvider:
    if provider_cls is AnthropicProvider:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    else:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic_proxy")
        monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "proxy-token")
        monkeypatch.setenv("ANTHROPIC_PROXY_ENDPOINT_URL", "https://proxy.example.com/predict")
    return provider_cls()


def _stub_anthropic(monkeypatch: pytest.MonkeyPatch, answer: AIMessage) -> list[dict]:
    """Record each request payload ``ChatAnthropic`` would send and answer with *answer*."""
    requests: list[dict] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        requests.append(self._get_request_payload(messages, stop=stop, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=answer)])

    monkeypatch.setattr(ChatAnthropic, "_generate", _generate)
    return requests


_ANTHROPIC_PROVIDERS = pytest.mark.parametrize(
    "provider_cls", [AnthropicProvider, AnthropicProxyProvider]
)


class TestAnthropicClaude55:
    """Native Messages API and Vertex-style proxy requests for Claude Opus and Sonnet 5.5."""

    @_ANTHROPIC_PROVIDERS
    @pytest.mark.parametrize("model", _CLAUDE_5_5)
    def test_json_schema_request_without_tools_parses_past_thinking(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type, model: str
    ) -> None:
        provider = _anthropic_provider(monkeypatch, provider_cls)
        answer = AIMessage(
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": '{"ok": true}'},
            ]
        )
        requests = _stub_anthropic(monkeypatch, answer)
        llm = provider.create_chat_model(model, max_tokens=provider.get_max_output_tokens(model))

        chain = bind_structured_output(llm, Verdict, model, provider)

        assert chain.invoke("analyse this") == Verdict(ok=True)
        request = requests[0]
        assert request["output_config"]["format"]["type"] == "json_schema"
        assert "tools" not in request and "tool_choice" not in request
        assert request["max_tokens"] == 128_000

    @_ANTHROPIC_PROVIDERS
    def test_refusal_is_never_a_parsed_result(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type
    ) -> None:
        provider = _anthropic_provider(monkeypatch, provider_cls)
        _stub_anthropic(
            monkeypatch, AIMessage(content=[], response_metadata={"stop_reason": "refusal"})
        )
        llm = provider.create_chat_model("claude-opus-5-5", max_tokens=1_000)

        chain = bind_structured_output(llm, Verdict, "claude-opus-5-5", provider)

        # A ValueError after a response fails the analyzer instead of reading as clean.
        with pytest.raises(OutputParserException):
            chain.invoke("analyse this")

    @_ANTHROPIC_PROVIDERS
    @pytest.mark.parametrize("model", _CLAUDE_5_5)
    @pytest.mark.parametrize("temperature", ["0", "1.0"])
    def test_explicit_temperature_fails_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type, model: str, temperature: str
    ) -> None:
        provider = _anthropic_provider(monkeypatch, provider_cls)
        monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", temperature)

        with pytest.raises(
            ValueError, match=f"SKILLSPECTOR_TEMPERATURE is not supported by {model}"
        ):
            provider.create_chat_model(model, max_tokens=1_000)

    @_ANTHROPIC_PROVIDERS
    def test_missing_credentials_win_over_an_incompatible_temperature(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type
    ) -> None:
        # No credentials keep returning None, so the OpenAI fallback still runs.
        monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0")
        assert provider_cls().create_chat_model("claude-opus-5-5", max_tokens=1_000) is None

    @_ANTHROPIC_PROVIDERS
    def test_other_claude_models_keep_temperature(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type
    ) -> None:
        provider = _anthropic_provider(monkeypatch, provider_cls)
        monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.2")
        llm = provider.create_chat_model("claude-opus-4-6", max_tokens=1_000)
        assert llm.temperature == 0.2
