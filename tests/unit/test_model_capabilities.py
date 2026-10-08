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
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from skillspector.inference_usage import chat_model_controls, chat_model_requested_controls
from skillspector.llm_utils import StructuredOutputParseError, bind_structured_output
from skillspector.providers import registry
from skillspector.providers.anthropic import AnthropicProvider
from skillspector.providers.anthropic_proxy import AnthropicProxyProvider
from skillspector.providers.chat_models import (
    GPT_6_1_SOL_REASONING_EFFORTS,
    reject_unsupported_controls,
)
from skillspector.providers.openai import OpenAIProvider
from skillspector.providers.openai_compatible import OpenAICompatibleProvider
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
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_PROJECT_ID",
        "SKILLSPECTOR_COMPAT_API_KEY",
        "SKILLSPECTOR_COMPAT_BASE_URL",
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
        [
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "azure/anthropic/claude-opus-5",
            "claude-opus-5-50",
            "gpt-6.1-sol",
        ],
    )
    def test_other_models_keep_both(self, model: str) -> None:
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
        assert rejects_forced_tool_call(model)
        assert rejects_sampling_controls(model)


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

    @_ANTHROPIC_PROVIDERS
    @pytest.mark.parametrize(
        "model", ["aws/anthropic/bedrock-claude-opus-5-5", "azure/anthropic/claude-sonnet-5-5"]
    )
    def test_gateway_ids_use_json_schema(self, provider_cls: type, model: str) -> None:
        assert provider_cls().structured_output_method(model) == "json_schema"

    @_ANTHROPIC_PROVIDERS
    def test_gateway_ids_for_other_models_keep_the_default(self, provider_cls: type) -> None:
        assert provider_cls().structured_output_method("azure/anthropic/claude-opus-5") is None


def _stub_openai(monkeypatch: pytest.MonkeyPatch, answers: list[AIMessage]) -> list[dict]:
    """Record each request payload ``ChatOpenAI`` would send and answer with *answers* in order."""
    requests: list[dict] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        requests.append({**payload, "prompt": messages[-1].content})
        return ChatResult(generations=[ChatGeneration(message=answers.pop(0))])

    monkeypatch.setattr(ChatOpenAI, "_generate", _generate)
    return requests


class TestOpenAIStructuredOutput:
    """``openai`` keeps LangChain's JSON-schema default except for Claude IDs that reject a forced tool."""

    def test_gpt_6_1_sol_keeps_tool_free_json_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        answer = AIMessage(content='{"ok": true}', additional_kwargs={"parsed": Verdict(ok=True)})
        requests = _stub_openai(monkeypatch, [answer])
        provider = OpenAIProvider()
        model = "gpt-6.1-sol"
        llm = provider.create_chat_model(model, max_tokens=provider.get_max_output_tokens(model))

        chain = bind_structured_output(llm, Verdict, model, provider)

        assert chain.invoke("analyse this") == Verdict(ok=True)
        request = requests[0]
        assert request["response_format"] is Verdict
        assert "tools" not in request and "tool_choice" not in request
        assert request["max_completion_tokens"] == 128_000
        assert "temperature" not in request and "reasoning_effort" not in request

    @pytest.mark.parametrize(
        "model", ["azure/anthropic/claude-opus-5-5", "aws/anthropic/bedrock-claude-sonnet-5-5"]
    )
    def test_gateway_claude_5_5_binds_an_unforced_tool(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
        tool_call = {"name": "Verdict", "args": {"ok": True}, "id": "call_1"}
        requests = _stub_openai(
            monkeypatch,
            [AIMessage(content="Looks fine."), AIMessage(content="", tool_calls=[tool_call])],
        )
        provider = OpenAIProvider()
        llm = provider.create_chat_model(model, max_tokens=1_000)

        chain = bind_structured_output(llm, Verdict, model, provider)

        # A prose answer is a retryable parse failure; the next tool call parses.
        with pytest.raises(StructuredOutputParseError, match="Verdict"):
            chain.invoke("analyse this")
        assert chain.invoke("analyse this") == Verdict(ok=True)
        request = requests[0]
        assert "tool_choice" not in request and "response_format" not in request
        assert [tool["function"]["name"] for tool in request["tools"]] == ["Verdict"]
        assert request["parallel_tool_calls"] is False
        assert "calling the Verdict tool" in request["prompt"]

    @pytest.mark.parametrize(
        "model", ["gpt-6.1-sol", "openai/openai/gpt-6.1-sol", "azure/anthropic/claude-opus-5"]
    )
    def test_other_models_get_no_method_hint(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = OpenAIProvider()
        assert provider.forced_tool_choice_supported(model)
        assert provider.structured_output_method(model) is None
        assert provider.create_chat_model(model, max_tokens=1_000).disabled_params is None


class TestOpenAICompatibleClaude55:
    """``openai_compatible`` applies the Claude name rule unless the registry declares ``tool_choice``."""

    def test_gateway_claude_5_5_disables_forced_tool_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "sk-test")
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", "https://gateway.example.com/v1")
        provider = OpenAICompatibleProvider()
        model = "anthropic/claude-opus-5-5"
        llm = provider.create_chat_model(model, max_tokens=1_000)
        assert llm.disabled_params == {"tool_choice": None}
        assert provider.structured_output_method(model) == "function_calling"

    def test_registry_tool_choice_overrides_the_name_rule(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        override = tmp_path / "registry.yaml"
        override.write_text(
            'models:\n  "anthropic/claude-opus-5-5":\n    context_length: 1000000\n'
            "    tool_choice: required\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("SKILLSPECTOR_MODEL_REGISTRY", str(override))
        provider = OpenAICompatibleProvider()
        assert provider.forced_tool_choice_supported("anthropic/claude-opus-5-5")
        assert provider.structured_output_method("anthropic/claude-opus-5-5") is None


def _openai_protocol_provider(
    monkeypatch: pytest.MonkeyPatch, provider_cls: type
) -> OpenAIProvider | OpenAICompatibleProvider:
    if provider_cls is OpenAIProvider:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    else:
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "sk-test")
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", "https://gateway.example.com/v1")
    return provider_cls()


_OPENAI_PROTOCOL_PROVIDERS = pytest.mark.parametrize(
    "provider_cls", [OpenAIProvider, OpenAICompatibleProvider]
)


class TestOpenAIProtocolControls:
    """The shared OpenAI-protocol builder rejects controls GPT-6.1 Sol and Claude 5.5 refuse."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-6.1-sol",
            "openai/openai/gpt-6.1-sol",
            "azure/openai/gpt-6.1-sol-2026-09-15",
            "openai.gpt-6.1-sol",
            "GPT-6.1-Sol",
        ],
    )
    def test_gpt_6_1_sol_ids_reject_temperature_and_unsupported_effort(self, model: str) -> None:
        with pytest.raises(ValueError, match="SKILLSPECTOR_TEMPERATURE"):
            reject_unsupported_controls(model, {"temperature": 1.0})
        with pytest.raises(ValueError, match="SKILLSPECTOR_REASONING_EFFORT"):
            reject_unsupported_controls(model, {}, "minimal")

    @pytest.mark.parametrize(
        "model", ["gpt-5.4", "gpt-4.1", "gpt-6-sol", "gpt-6.1-solar", "vendor/x/notgpt-6.1-sol"]
    )
    def test_other_models_keep_both(self, model: str) -> None:
        reject_unsupported_controls(model, {"temperature": 0.0}, "minimal")

    @_OPENAI_PROTOCOL_PROVIDERS
    @pytest.mark.parametrize(
        "model", ["gpt-6.1-sol", "openai/openai/gpt-6.1-sol", "azure/anthropic/claude-opus-5-5"]
    )
    @pytest.mark.parametrize("temperature", ["0", "1.0"])
    def test_explicit_temperature_fails_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type, model: str, temperature: str
    ) -> None:
        provider = _openai_protocol_provider(monkeypatch, provider_cls)
        monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", temperature)

        with pytest.raises(
            ValueError, match=f"SKILLSPECTOR_TEMPERATURE is not supported by {model}; unset it"
        ):
            provider.create_chat_model(model, max_tokens=1_000)

    @_OPENAI_PROTOCOL_PROVIDERS
    @pytest.mark.parametrize("effort", ["none", "minimal", "High"])
    def test_unsupported_effort_fails_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch, provider_cls: type, effort: str
    ) -> None:
        provider = _openai_protocol_provider(monkeypatch, provider_cls)
        monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", effort)

        with pytest.raises(
            ValueError,
            match=(
                f"SKILLSPECTOR_REASONING_EFFORT='{effort}' is not supported by gpt-6.1-sol; "
                "use one of low, medium, high, xhigh, max or unset it"
            ),
        ):
            provider.create_chat_model("gpt-6.1-sol", max_tokens=1_000)

    @pytest.mark.parametrize("effort", GPT_6_1_SOL_REASONING_EFFORTS)
    def test_supported_effort_is_forwarded(
        self, monkeypatch: pytest.MonkeyPatch, effort: str
    ) -> None:
        provider = _openai_protocol_provider(monkeypatch, OpenAIProvider)
        monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", effort)
        llm = provider.create_chat_model("gpt-6.1-sol", max_tokens=1_000)
        assert chat_model_controls(llm)["reasoning_effort"] == effort

    def test_missing_credentials_win_over_incompatible_controls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No credentials keep returning None, so the no-credentials path is unchanged.
        monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0")
        monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "minimal")
        assert OpenAIProvider().create_chat_model("gpt-6.1-sol", max_tokens=1_000) is None

    @pytest.mark.parametrize("model", ["gpt-5.4", "gpt-4.1"])
    def test_other_models_pass_controls_through(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        provider = _openai_protocol_provider(monkeypatch, OpenAIProvider)
        monkeypatch.setenv("SKILLSPECTOR_TEMPERATURE", "0.2")
        monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "minimal")
        llm = provider.create_chat_model(model, max_tokens=1_000)
        assert chat_model_requested_controls(llm)["temperature"] == 0.2
        assert llm.reasoning_effort == "minimal"

    def test_seed_is_forwarded_to_gpt_6_1_sol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _openai_protocol_provider(monkeypatch, OpenAIProvider)
        monkeypatch.setenv("SKILLSPECTOR_SEED", "42")
        llm = provider.create_chat_model("gpt-6.1-sol", max_tokens=1_000)
        # Sent and recorded, though OpenAI treats the seed as best-effort only.
        assert chat_model_requested_controls(llm)["seed"] == 42
        assert chat_model_controls(llm)["seed"] == 42
