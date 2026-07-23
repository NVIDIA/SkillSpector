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

"""Tests for Ollama, Azure OpenAI, and generic OpenAI-compatible providers."""

from __future__ import annotations

import pytest
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from skillspector.providers import (
    create_chat_model,
    get_metadata_provider,
    registry,
    resolve_provider_credentials,
)
from skillspector.providers.azure_openai import AzureOpenAIProvider
from skillspector.providers.ollama import OLLAMA_DEFAULT_BASE_URL, OllamaProvider
from skillspector.providers.openai_compatible import OpenAICompatibleProvider


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate provider-related env vars and the YAML cache for each test."""
    for key in (
        "NVIDIA_INFERENCE_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_PROVIDER",
        "OLLAMA_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "SKILLSPECTOR_COMPAT_API_KEY",
        "SKILLSPECTOR_COMPAT_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


# ── Ollama ──────────────────────────────────────────────────────────────────


class TestOllamaProvider:
    """Ollama provider — local/self-hosted LLM endpoint."""

    def test_always_returns_credentials(self) -> None:
        creds = OllamaProvider().resolve_credentials()
        assert creds is not None
        api_key, base_url = creds
        assert api_key == "ollama"
        assert base_url == OLLAMA_DEFAULT_BASE_URL

    def test_custom_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-server:11434/v1")
        creds = OllamaProvider().resolve_credentials()
        assert creds == ("ollama", "http://gpu-server:11434/v1")

    def test_creates_chat_openai(self) -> None:
        llm = OllamaProvider().create_chat_model("llama3.1:8b", max_tokens=512)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "llama3.1:8b"
        assert llm.max_tokens == 512
        assert str(llm.openai_api_base).rstrip("/") == OLLAMA_DEFAULT_BASE_URL.rstrip("/")

    def test_default_model(self) -> None:
        assert OllamaProvider().resolve_model() == "llama3.1:8b"

    def test_metadata_known_model(self) -> None:
        provider = OllamaProvider()
        assert provider.get_context_length("llama3.1:8b") == 131072
        assert provider.get_max_output_tokens("llama3.1:8b") == 4096

    def test_metadata_unknown_model_returns_none(self) -> None:
        provider = OllamaProvider()
        assert provider.get_context_length("unknown-model") is None

    def test_env_model_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "mistral:7b")
        assert OllamaProvider().resolve_model() == "mistral:7b"


class TestOllamaProviderSelection:
    """SKILLSPECTOR_PROVIDER=ollama selects the Ollama provider."""

    def test_select_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "ollama")
        assert isinstance(get_metadata_provider(), OllamaProvider)

    def test_ollama_credentials_via_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "ollama")
        creds = resolve_provider_credentials()
        assert creds is not None
        assert creds[0] == "ollama"

    def test_create_chat_model_with_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "ollama")
        llm = create_chat_model("llama3.1:8b", max_tokens=512)
        assert isinstance(llm, ChatOpenAI)


# ── Azure OpenAI ────────────────────────────────────────────────────────────


class TestAzureOpenAIProvider:
    """Azure OpenAI provider — enterprise Azure deployments."""

    def test_returns_none_without_env_vars(self) -> None:
        assert AzureOpenAIProvider().resolve_credentials() is None

    def test_returns_none_with_key_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        assert AzureOpenAIProvider().resolve_credentials() is None

    def test_returns_none_with_endpoint_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        assert AzureOpenAIProvider().resolve_credentials() is None

    def test_resolves_with_both_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        creds = AzureOpenAIProvider().resolve_credentials()
        assert creds == ("azure-key", "https://myorg.openai.azure.com/")

    def test_creates_azure_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-gpt4o")
        llm = AzureOpenAIProvider().create_chat_model("gpt-4o", max_tokens=1024)
        assert isinstance(llm, AzureChatOpenAI)
        assert llm.deployment_name == "my-gpt4o"

    def test_deployment_defaults_to_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        llm = AzureOpenAIProvider().create_chat_model("gpt-4o", max_tokens=1024)
        assert isinstance(llm, AzureChatOpenAI)
        assert llm.deployment_name == "gpt-4o"

    def test_api_version_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        llm = AzureOpenAIProvider().create_chat_model("gpt-4o", max_tokens=1024)
        assert isinstance(llm, AzureChatOpenAI)
        assert llm.openai_api_version == "2024-06-01"

    def test_custom_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
        llm = AzureOpenAIProvider().create_chat_model("gpt-4o", max_tokens=1024)
        assert isinstance(llm, AzureChatOpenAI)
        assert llm.openai_api_version == "2025-01-01"

    def test_default_model(self) -> None:
        assert AzureOpenAIProvider().resolve_model() == "gpt-4o"

    def test_metadata_known_model(self) -> None:
        provider = AzureOpenAIProvider()
        assert provider.get_context_length("gpt-4o") == 128000
        assert provider.get_max_output_tokens("gpt-4o") == 16384

    def test_create_returns_none_without_credentials(self) -> None:
        assert AzureOpenAIProvider().create_chat_model("gpt-4o", max_tokens=1024) is None


class TestAzureOpenAIProviderSelection:
    """SKILLSPECTOR_PROVIDER=azure_openai selects the Azure provider."""

    def test_select_azure_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "azure_openai")
        assert isinstance(get_metadata_provider(), AzureOpenAIProvider)

    def test_azure_credentials_via_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "azure_openai")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myorg.openai.azure.com/")
        creds = resolve_provider_credentials()
        assert creds == ("azure-key", "https://myorg.openai.azure.com/")


# ── Generic OpenAI-Compatible ───────────────────────────────────────────────


class TestOpenAICompatibleProvider:
    """Generic OpenAI-compatible provider — Groq, Together AI, Mistral, etc."""

    def test_returns_none_without_env_vars(self) -> None:
        assert OpenAICompatibleProvider().resolve_credentials() is None

    def test_returns_none_with_key_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "gsk_abc")
        assert OpenAICompatibleProvider().resolve_credentials() is None

    def test_returns_none_with_url_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", "https://api.groq.com/openai/v1")
        assert OpenAICompatibleProvider().resolve_credentials() is None

    def test_resolves_with_both_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "gsk_abc")
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", "https://api.groq.com/openai/v1")
        creds = OpenAICompatibleProvider().resolve_credentials()
        assert creds == ("gsk_abc", "https://api.groq.com/openai/v1")

    def test_creates_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "gsk_abc")
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", "https://api.groq.com/openai/v1")
        llm = OpenAICompatibleProvider().create_chat_model(
            "llama-3.1-70b-versatile", max_tokens=1024
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "llama-3.1-70b-versatile"
        assert str(llm.openai_api_base).rstrip("/") == "https://api.groq.com/openai/v1"

    def test_default_model(self) -> None:
        assert OpenAICompatibleProvider().resolve_model() == "llama-3.1-70b-versatile"

    def test_metadata_known_model(self) -> None:
        provider = OpenAICompatibleProvider()
        assert provider.get_context_length("llama-3.1-70b-versatile") == 131072
        assert provider.get_max_output_tokens("llama-3.1-70b-versatile") == 8192

    def test_metadata_unknown_model_returns_none(self) -> None:
        provider = OpenAICompatibleProvider()
        assert provider.get_context_length("some-random-model") is None

    def test_create_returns_none_without_credentials(self) -> None:
        assert (
            OpenAICompatibleProvider().create_chat_model(
                "llama-3.1-70b-versatile", max_tokens=1024
            )
            is None
        )


class TestOpenAICompatibleProviderSelection:
    """SKILLSPECTOR_PROVIDER=openai_compatible selects the generic provider."""

    def test_select_openai_compatible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai_compatible")
        assert isinstance(get_metadata_provider(), OpenAICompatibleProvider)

    def test_compat_credentials_via_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai_compatible")
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_API_KEY", "gsk_abc")
        monkeypatch.setenv("SKILLSPECTOR_COMPAT_BASE_URL", "https://api.groq.com/openai/v1")
        creds = resolve_provider_credentials()
        assert creds == ("gsk_abc", "https://api.groq.com/openai/v1")


class TestUnknownProviderError:
    """Verify the error message lists all providers including new ones."""

    def test_error_message_includes_new_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "nonexistent")
        with pytest.raises(ValueError, match="ollama") as exc_info:
            get_metadata_provider()
        error_msg = str(exc_info.value)
        assert "azure_openai" in error_msg
        assert "openai_compatible" in error_msg
