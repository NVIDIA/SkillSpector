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

"""Tests for the LLM credential resolution in llm_utils.

Order: active SkillSpector provider -> OPENAI_API_KEY / OPENAI_BASE_URL.
Provider-specific behavior (which env var resolves to which client) lives
in the active provider — see ``tests/unit/test_providers.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from skillspector import llm_utils
from skillspector.llm_utils import (
    AgentCLIChatModel,
    _extract_json_object,
    _resolve_llm_credentials,
    chat_completion,
    fetch_model_token_limits,
    get_chat_model,
    is_llm_available,
)
from skillspector.providers import NO_LLM_API_KEY_MESSAGE, resolve_provider_credentials
from skillspector.providers.nv_build import NvBuildProvider
from skillspector.providers.openai import OpenAIProvider

_LLM_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "NVIDIA_INFERENCE_KEY",
    "SKILLSPECTOR_MODEL",
    "SKILLSPECTOR_PROVIDER",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch):
    """Clear all LLM-related env vars for test isolation."""
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


class TestCredentialResolution:
    """Order: active provider first, then OPENAI_API_KEY / OPENAI_BASE_URL."""

    def test_provider_wins_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvidia-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        provider_creds = resolve_provider_credentials()
        assert provider_creds is not None  # active provider must answer
        key, base = _resolve_llm_credentials()
        assert key == "nvidia-key"
        assert base == provider_creds[1]

    def test_openai_used_when_provider_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        key, base = _resolve_llm_credentials()
        assert key == "openai-key"
        assert base is None

    def test_openai_base_url_used_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.example/v1")
        _, base = _resolve_llm_credentials()
        assert base == "http://openai.example/v1"

    def test_provider_base_url_not_overridden_by_openai_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OPENAI_BASE_URL is the OpenAI tier; it does not affect the provider tier."""
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvidia-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.example/v1")
        provider_creds = resolve_provider_credentials()
        assert provider_creds is not None
        _, base = _resolve_llm_credentials()
        assert base == provider_creds[1]

    def test_anthropic_provider_wins_with_native_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        key, base = _resolve_llm_credentials()
        assert key == "sk-ant-x"
        assert base is None

    def test_no_credentials_raises_with_helpful_message(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _resolve_llm_credentials()
        assert str(exc_info.value) == NO_LLM_API_KEY_MESSAGE

    def test_get_chat_model_returns_native_anthropic_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        llm = get_chat_model(model="claude-opus-4-6")
        assert isinstance(llm, ChatAnthropic)
        assert llm.model == "claude-opus-4-6"


class TestFetchModelTokenLimits:
    def test_returns_input_and_output_token_pair(self) -> None:
        max_input, max_output = fetch_model_token_limits("claude-opus-4-6")
        assert isinstance(max_input, int)
        assert isinstance(max_output, int)
        assert max_input > 0
        assert max_output > 0


class TestChatCompletion:
    """``chat_completion`` invokes the active chat model and normalizes content."""

    def test_returns_string_content_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeLLM:
            def invoke(self, prompt: str) -> AIMessage:
                assert prompt == "ping"
                return AIMessage(content="hello world")

        monkeypatch.setattr(llm_utils, "get_chat_model", lambda model=None: _FakeLLM())
        assert chat_completion("ping") == "hello world"

    def test_returns_text_from_langchain_content_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeLLM:
            def invoke(self, prompt: str) -> AIMessage:
                return AIMessage(content=[{"type": "text", "text": "chunk"}])

        captured: dict[str, str | None] = {}

        def _fake_get_chat_model(model: str | None = None) -> _FakeLLM:
            captured["model"] = model
            return _FakeLLM()

        monkeypatch.setattr(llm_utils, "get_chat_model", _fake_get_chat_model)
        result = chat_completion("prompt", model="some-model")
        assert result == "chunk"
        assert captured["model"] == "some-model"

    def test_returns_empty_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeLLM:
            def invoke(self, prompt: str) -> AIMessage:
                return AIMessage(content="")

        monkeypatch.setattr(llm_utils, "get_chat_model", lambda model=None: _FakeLLM())
        assert chat_completion("prompt") == ""


class TestIsLlmAvailable:
    def test_returns_true_when_credentials_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        ok, msg = is_llm_available()
        assert ok is True
        assert msg is None

    def test_returns_true_via_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "k")
        ok, msg = is_llm_available()
        assert ok is True
        assert msg is None

    def test_returns_false_with_message_when_no_credentials(self) -> None:
        ok, msg = is_llm_available()
        assert ok is False
        assert msg == NO_LLM_API_KEY_MESSAGE

    def test_cli_provider_delegates_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When SKILLSPECTOR_PROVIDER=claude_cli, is_llm_available asks the provider."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "claude_cli")
        # Mock the provider's is_available directly to simulate binary absence.
        with patch(
            "skillspector.providers.claude_cli.provider.ClaudeCLIProvider.is_available",
            return_value=(False, "binary not found on PATH"),
        ):
            ok, err = is_llm_available()
        assert ok is False
        assert "not found" in (err or "").lower()


class TestChatCompletionCLIDispatch:
    """chat_completion dispatches to provider.complete() for CLI providers."""

    def test_dispatches_to_cli_provider_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "claude_cli")

        fake_complete = MagicMock(return_value="mocked CLI response")
        with patch(
            "skillspector.providers.claude_cli.provider.ClaudeCLIProvider.complete",
            fake_complete,
        ):
            result = chat_completion("test prompt", model="claude-haiku-3-5")

        assert result == "mocked CLI response"
        fake_complete.assert_called_once()
        call_kwargs = fake_complete.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-3-5"

    def test_does_not_call_complete_for_http_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For HTTP providers, the native provider chat-model path is used."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        fake_instance = MagicMock()
        fake_instance.invoke.return_value = MagicMock(content="http response", text="http response")

        with patch("skillspector.llm_utils.get_chat_model", return_value=fake_instance):
            result = chat_completion("test prompt")

        assert result == "http response"
        # The CLI .complete() should never have been called
        fake_instance.complete.assert_not_called()


class TestGetChatModelCLIAdapter:
    """get_chat_model returns a CLI adapter for CLI providers; the adapter
    mimics the slice of the ChatOpenAI interface the analyzers use."""

    def test_returns_adapter_for_cli_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "claude_cli")
        model = get_chat_model(model="claude-sonnet-4-6")
        assert isinstance(model, AgentCLIChatModel)

    def test_returns_chatopenai_for_http_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        model = get_chat_model(model="claude-opus-4-6")
        assert not isinstance(model, AgentCLIChatModel)

    def test_adapter_invoke_returns_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "claude_cli")
        with patch(
            "skillspector.providers.claude_cli.provider.ClaudeCLIProvider.complete",
            MagicMock(return_value="hello"),
        ):
            msg = get_chat_model(model="claude-sonnet-4-6").invoke("hi")
        assert msg.content == "hello"

    def test_structured_output_parses_and_validates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "claude_cli")

        class _Schema(BaseModel):
            verdict: str
            score: int

        raw = '```json\n{"verdict": "unsafe", "score": 7}\n```'
        with patch(
            "skillspector.providers.claude_cli.provider.ClaudeCLIProvider.complete",
            MagicMock(return_value=raw),
        ):
            out = (
                get_chat_model(model="claude-sonnet-4-6")
                .with_structured_output(_Schema)
                .invoke("x")
            )
        assert isinstance(out, _Schema)
        assert out.verdict == "unsafe"
        assert out.score == 7

    def test_structured_output_fail_closed_on_garbage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "claude_cli")

        class _Schema(BaseModel):
            verdict: str

        with patch(
            "skillspector.providers.claude_cli.provider.ClaudeCLIProvider.complete",
            MagicMock(return_value="no json here at all"),
        ):
            with pytest.raises(ValueError, match="JSON"):
                get_chat_model(model="claude-sonnet-4-6").with_structured_output(_Schema).invoke(
                    "x"
                )


class TestExtractJsonObject:
    def test_plain_json(self) -> None:
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert _extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped_json(self) -> None:
        assert _extract_json_object('Here you go:\n{"a": 1}\nDone.') == {"a": 1}

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            _extract_json_object("not json")


class TestGetChatModel:
    def test_openai_fallback_uses_openai_default_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-only")

        llm = get_chat_model()

        assert _chat_model_name(llm) == OpenAIProvider.DEFAULT_MODEL

    def test_explicit_model_still_overrides_openai_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-only")

        llm = get_chat_model(model="custom/model")

        assert _chat_model_name(llm) == "custom/model"

    def test_provider_credentials_use_provider_default_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvapi-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

        llm = get_chat_model()

        assert _chat_model_name(llm) == NvBuildProvider.DEFAULT_MODEL


def _chat_model_name(llm: object) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", None))
