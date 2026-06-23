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

"""Tests for the AnthropicProxyProvider (Vertex-style raw-predict endpoints)."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_anthropic import ChatAnthropic

from skillspector.providers import create_chat_model, get_metadata_provider, registry
from skillspector.providers.anthropic_proxy import AnthropicProxyProvider
from skillspector.providers.anthropic_proxy.provider import (
    DEFAULT_API_VERSION,
    _ChatAnthropicProxy,
    _get_api_version,
    _ProxyTransport,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate provider-related env vars for each test."""
    monkeypatch.delenv("ANTHROPIC_PROXY_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROXY_API_VERSION", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_PROVIDER", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_SSL_VERIFY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_INFERENCE_KEY", raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


class TestAnthropicProxyProviderCredentials:
    """Credential resolution tests."""

    def test_returns_none_without_env_vars(self) -> None:
        assert AnthropicProxyProvider().resolve_credentials() is None

    def test_returns_none_with_only_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "token-123")
        assert AnthropicProxyProvider().resolve_credentials() is None

    def test_returns_none_with_only_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_PROXY_ENDPOINT_URL", "https://proxy.example.com/predict")
        assert AnthropicProxyProvider().resolve_credentials() is None

    def test_resolves_credentials_with_both_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "bearer-tok")
        monkeypatch.setenv("ANTHROPIC_PROXY_ENDPOINT_URL", "https://proxy.example.com/predict")
        creds = AnthropicProxyProvider().resolve_credentials()
        assert creds == ("bearer-tok", "https://proxy.example.com/predict")


class TestAnthropicProxyProviderChatModel:
    """Chat model creation tests."""

    def test_create_chat_model_returns_none_without_creds(self) -> None:
        result = AnthropicProxyProvider().create_chat_model("claude-sonnet-4-6", max_tokens=1024)
        assert result is None

    def test_creates_chat_anthropic_subclass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "bearer-tok")
        monkeypatch.setenv(
            "ANTHROPIC_PROXY_ENDPOINT_URL",
            "https://proxy.example.com/models/claude-sonnet-4-6:streamRawPredict",
        )
        llm = AnthropicProxyProvider().create_chat_model("claude-sonnet-4-6", max_tokens=4096)
        assert isinstance(llm, ChatAnthropic)
        assert isinstance(llm, _ChatAnthropicProxy)
        assert llm.model == "claude-sonnet-4-6"
        assert llm.max_tokens == 4096


class TestAnthropicProxyProviderMetadata:
    """Token-budget metadata and model resolution tests."""

    def test_metadata_known_model(self) -> None:
        provider = AnthropicProxyProvider()
        assert provider.get_context_length("claude-sonnet-4-6") == 1_000_000
        assert provider.get_max_output_tokens("claude-sonnet-4-6") == 128_000

    def test_metadata_unknown_model_returns_none(self) -> None:
        provider = AnthropicProxyProvider()
        assert provider.get_context_length("unknown-model") is None
        assert provider.get_max_output_tokens("unknown-model") is None

    def test_default_model(self) -> None:
        assert AnthropicProxyProvider().resolve_model() == "claude-sonnet-4-6"

    def test_resolve_model_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "claude-opus-4-6")
        assert AnthropicProxyProvider().resolve_model() == "claude-opus-4-6"
        assert AnthropicProxyProvider().resolve_model("meta_analyzer") == "claude-opus-4-6"


class TestProxyTransport:
    """Transport request-rewriting tests."""

    def _make_request(
        self, body: dict, *, monkeypatch: pytest.MonkeyPatch | None = None
    ) -> tuple[httpx.Request, dict]:
        """Create a transport and capture the rewritten request.

        Returns the rewritten httpx.Request and the decoded body.
        """
        transport = _ProxyTransport(
            endpoint_url="https://proxy.example.com/models/claude:predict",
            bearer_token="my-token",
            verify=False,
        )

        original_request = httpx.Request(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": "sk-ant-secret",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            content=json.dumps(body).encode("utf-8"),
        )

        captured: list[httpx.Request] = []

        class _CapturingTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(request)
                return httpx.Response(
                    status_code=200,
                    json={
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}],
                        "model": "claude-sonnet-4-6",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )

        transport._inner = _CapturingTransport()
        transport.handle_request(original_request)

        assert len(captured) == 1
        rewritten = captured[0]
        rewritten_body = json.loads(rewritten.content)
        return rewritten, rewritten_body

    def test_rewrites_url(self) -> None:
        rewritten, _ = self._make_request(
            {"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 100}
        )
        assert str(rewritten.url) == "https://proxy.example.com/models/claude:predict"

    def test_removes_model_from_body(self) -> None:
        _, body = self._make_request(
            {"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 100}
        )
        assert "model" not in body

    def test_injects_anthropic_version(self) -> None:
        _, body = self._make_request(
            {"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 100}
        )
        assert body["anthropic_version"] == DEFAULT_API_VERSION

    def test_replaces_auth_header(self) -> None:
        rewritten, _ = self._make_request(
            {"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 100}
        )
        assert rewritten.headers["authorization"] == "Bearer my-token"
        assert "x-api-key" not in rewritten.headers
        assert "anthropic-version" not in rewritten.headers

    def test_preserves_content_type(self) -> None:
        rewritten, _ = self._make_request(
            {"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 100}
        )
        assert rewritten.headers["content-type"] == "application/json"

    def test_preserves_other_body_fields(self) -> None:
        _, body = self._make_request(
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 200,
                "temperature": 0.5,
            }
        )
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["max_tokens"] == 200
        assert body["temperature"] == 0.5


class TestApiVersionConfiguration:
    """Tests for ANTHROPIC_PROXY_API_VERSION env var."""

    def test_default_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_PROXY_API_VERSION", raising=False)
        assert _get_api_version() == "vertex-2023-10-16"

    def test_custom_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_PROXY_API_VERSION", "bedrock-2023-05-31")
        assert _get_api_version() == "bedrock-2023-05-31"

    def test_empty_string_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_PROXY_API_VERSION", "  ")
        assert _get_api_version() == DEFAULT_API_VERSION


class TestProviderSelection:
    """Integration with the provider selector."""

    def test_select_anthropic_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic_proxy")
        assert isinstance(get_metadata_provider(), AnthropicProxyProvider)

    def test_create_chat_model_via_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic_proxy")
        monkeypatch.setenv("ANTHROPIC_PROXY_API_KEY", "tok")
        monkeypatch.setenv("ANTHROPIC_PROXY_ENDPOINT_URL", "https://proxy.example.com/predict")
        llm = create_chat_model("claude-sonnet-4-6", max_tokens=1024)
        assert isinstance(llm, _ChatAnthropicProxy)

    def test_falls_back_to_openai_when_proxy_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When anthropic_proxy is selected but not configured, falls back to OpenAI."""
        from langchain_openai import ChatOpenAI

        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic_proxy")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
        llm = create_chat_model("gpt-5.4", max_tokens=1024)
        assert isinstance(llm, ChatOpenAI)
