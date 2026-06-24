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

"""Tests for the NVIDIA provider chain (credentials + bundled YAML metadata).

Catalog-side metadata behavior (``NvInferenceProvider`` against the
NVIDIA catalog API) is covered by the layered tests in
``test_model_info.py``.
"""

from __future__ import annotations

import sys

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from skillspector.providers import (
    NO_LLM_API_KEY_MESSAGE,
    create_chat_model,
    get_metadata_provider,
    registry,
    resolve_chat_model_credentials,
    resolve_provider_credentials,
)
from skillspector.providers.anthropic import AnthropicProvider
from skillspector.providers.chat_models import create_openai_compatible_chat_model
from skillspector.providers.nv_build import BUILD_BASE_URL, NvBuildProvider
from skillspector.providers.openai import OpenAIProvider

try:
    from skillspector.providers.nv_inference import (
        INFERENCE_BASE_URL,
        NvInferenceProvider,
    )

    _NV_INFERENCE_AVAILABLE = True
except ImportError:
    _NV_INFERENCE_AVAILABLE = False

nv_inference_required = pytest.mark.skipif(
    not _NV_INFERENCE_AVAILABLE,
    reason="optional NVIDIA Inference Hub provider not present (public-OSS build)",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate provider-related env vars and the YAML cache for each test."""
    monkeypatch.delenv("NVIDIA_INFERENCE_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_INFERENCE_METADATA_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_MODEL_REGISTRY", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_PROVIDER", raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


class TestNvBuildProvider:
    """build.nvidia.com provider — credentials + bundled YAML metadata."""

    def test_returns_none_without_env_var(self) -> None:
        assert NvBuildProvider().resolve_credentials() is None

    def test_resolves_to_build_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvapi-x")
        creds = NvBuildProvider().resolve_credentials()
        assert creds == ("nvapi-x", BUILD_BASE_URL)

    def test_creates_openai_compatible_chat_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvapi-x")
        llm = NvBuildProvider().create_chat_model(
            "deepseek-ai/deepseek-v4-flash",
            max_tokens=123,
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "deepseek-ai/deepseek-v4-flash"
        assert llm.max_tokens == 123
        assert str(llm.openai_api_base).rstrip("/") == BUILD_BASE_URL.rstrip("/")

    def test_metadata_known_model_from_bundled_yaml(self) -> None:
        """deepseek-v4-flash ships in nv_build/model_registry.yaml."""
        provider = NvBuildProvider()
        assert provider.get_context_length("deepseek-ai/deepseek-v4-flash") == 1_000_000
        assert provider.get_max_output_tokens("deepseek-ai/deepseek-v4-flash") == 128_000

    def test_metadata_unknown_model_returns_none(self) -> None:
        provider = NvBuildProvider()
        assert provider.get_context_length("unknown/model-xyz") is None
        assert provider.get_max_output_tokens("unknown/model-xyz") is None

    def test_resolve_model_default_when_no_env(self) -> None:
        assert NvBuildProvider().resolve_model() == NvBuildProvider.DEFAULT_MODEL

    def test_resolve_model_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "user/override")
        assert NvBuildProvider().resolve_model() == "user/override"
        # Env override applies to every slot.
        assert NvBuildProvider().resolve_model("meta_analyzer") == "user/override"

    def test_resolve_model_meta_analyzer_uses_slot_override(self) -> None:
        # meta_analyzer is upgraded to deepseek-v4-pro on NvBuild.
        assert (
            NvBuildProvider().resolve_model("meta_analyzer")
            == NvBuildProvider.SLOT_DEFAULTS["meta_analyzer"]
        )

    def test_resolve_model_unknown_slot_falls_to_default(self) -> None:
        # Slots without an explicit override inherit DEFAULT_MODEL.
        assert (
            NvBuildProvider().resolve_model("mcp_least_privilege") == NvBuildProvider.DEFAULT_MODEL
        )


@nv_inference_required
class TestNvInferenceProvider:
    """Internal Inference Hub provider — credentials + bundled YAML metadata."""

    def test_returns_none_without_env_var(self) -> None:
        provider = NvInferenceProvider()
        assert provider.resolve_credentials() is None

    def test_resolves_to_inference_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "internal-key")
        creds = NvInferenceProvider().resolve_credentials()
        assert creds == ("internal-key", INFERENCE_BASE_URL)

    def test_creates_openai_compatible_chat_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "internal-key")
        llm = NvInferenceProvider().create_chat_model(
            "azure/anthropic/claude-sonnet-4-6",
            max_tokens=123,
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "azure/anthropic/claude-sonnet-4-6"
        assert llm.max_tokens == 123
        assert str(llm.openai_api_base).rstrip("/") == INFERENCE_BASE_URL.rstrip("/")

    def test_metadata_key_not_required_for_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The metadata env var is independent of the credentials env var."""
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "internal-key")
        creds = NvInferenceProvider().resolve_credentials()
        assert creds is not None

    def test_yaml_fallback_when_catalog_not_configured(self) -> None:
        """With NVIDIA_INFERENCE_METADATA_KEY unset, we fall back to bundled YAML."""
        provider = NvInferenceProvider()
        assert provider.get_context_length("azure/anthropic/claude-sonnet-4-6") == 1_000_000
        assert provider.get_max_output_tokens("azure/anthropic/claude-sonnet-4-6") == 128_000

    def test_metadata_unknown_model_returns_none(self) -> None:
        provider = NvInferenceProvider()
        assert provider.get_context_length("unknown/model-xyz") is None
        assert provider.get_max_output_tokens("unknown/model-xyz") is None

    def test_resolve_model_default(self) -> None:
        assert NvInferenceProvider().resolve_model() == NvInferenceProvider.DEFAULT_MODEL

    def test_resolve_model_meta_analyzer_uses_slot_override(self) -> None:
        # meta_analyzer is the only configured downgrade slot.
        assert (
            NvInferenceProvider().resolve_model("meta_analyzer")
            == NvInferenceProvider.SLOT_DEFAULTS["meta_analyzer"]
        )

    def test_resolve_model_env_overrides_slot_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "user/override")
        # Env wins over the meta_analyzer slot default.
        assert NvInferenceProvider().resolve_model("meta_analyzer") == "user/override"


class TestOpenAIProvider:
    """Stock OpenAI provider — credentials + bundled YAML metadata."""

    def test_returns_none_without_env_var(self) -> None:
        assert OpenAIProvider().resolve_credentials() is None

    def test_resolves_to_openai_with_default_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        creds = OpenAIProvider().resolve_credentials()
        assert creds == ("sk-x", None)  # None → ChatOpenAI uses api.openai.com

    def test_honors_openai_base_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        creds = OpenAIProvider().resolve_credentials()
        assert creds == ("sk-x", "http://localhost:11434/v1")

    def test_creates_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        llm = OpenAIProvider().create_chat_model("gpt-5.4", max_tokens=123)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-5.4"
        assert llm.max_tokens == 123

    def test_openai_project_id_sets_default_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_123")
        llm = OpenAIProvider().create_chat_model("gpt-5.4", max_tokens=123)
        assert isinstance(llm, ChatOpenAI)
        assert llm.default_headers == {"OpenAI-Project": "proj_123"}

    def test_default_model(self) -> None:
        assert OpenAIProvider().resolve_model() == "gpt-5.4"
        # All slots inherit DEFAULT_MODEL — gpt-5.4 everywhere.
        assert OpenAIProvider().resolve_model("meta_analyzer") == "gpt-5.4"

    def test_metadata_known_model(self) -> None:
        provider = OpenAIProvider()
        assert provider.get_context_length("gpt-5.4") == 1_000_000
        assert provider.get_max_output_tokens("gpt-5.4") == 128_000


class TestAnthropicProvider:
    """Anthropic provider — Claude credentials + bundled YAML metadata."""

    def test_returns_none_without_env_var(self) -> None:
        assert AnthropicProvider().resolve_credentials() is None

    def test_resolves_anthropic_api_key_without_openai_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        creds = AnthropicProvider().resolve_credentials()
        assert creds == ("sk-ant-x", None)

    def test_creates_native_chat_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        llm = AnthropicProvider().create_chat_model("claude-opus-4-6", max_tokens=123)
        assert isinstance(llm, ChatAnthropic)
        assert llm.model == "claude-opus-4-6"
        assert llm.max_tokens == 123

    def test_create_chat_model_returns_none_without_key(self) -> None:
        # No ANTHROPIC_API_KEY → no client, signalling the caller to fall back.
        assert AnthropicProvider().create_chat_model("claude-opus-4-6", max_tokens=123) is None

    def test_default_model_and_meta_downgrade(self) -> None:
        assert AnthropicProvider().resolve_model() == "claude-opus-4-6"
        assert AnthropicProvider().resolve_model("meta_analyzer") == "claude-sonnet-4-6"

    def test_metadata_known_models(self) -> None:
        provider = AnthropicProvider()
        assert provider.get_context_length("claude-opus-4-6") == 1_000_000
        assert provider.get_max_output_tokens("claude-opus-4-6") == 128_000
        assert provider.get_context_length("claude-sonnet-4-6") == 1_000_000


class TestOpenAICompatibleConstructor:
    """The shared OpenAI-compatible chat-model constructor."""

    def test_returns_none_when_credentials_missing(self) -> None:
        assert (
            create_openai_compatible_chat_model(
                model="gpt-5.4",
                credentials=None,
                max_tokens=123,
            )
            is None
        )

    def test_builds_chat_openai_from_credentials(self) -> None:
        llm = create_openai_compatible_chat_model(
            model="gpt-5.4",
            credentials=("sk-x", "http://localhost:1234/v1"),
            max_tokens=123,
        )
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-5.4"
        assert llm.max_tokens == 123
        assert str(llm.openai_api_base).rstrip("/") == "http://localhost:1234/v1"


class TestProviderSelection:
    """SKILLSPECTOR_PROVIDER selects which provider answers credentials."""

    def test_no_env_defaults_to_nvidia_path(self) -> None:
        # Without credentials, the default-path provider returns None.
        assert resolve_provider_credentials() is None

    def test_active_nvidia_provider_returns_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "active-key")
        creds = resolve_provider_credentials()
        assert creds is not None
        api_key, base_url = creds
        assert api_key == "active-key"
        expected_url = INFERENCE_BASE_URL if _NV_INFERENCE_AVAILABLE else BUILD_BASE_URL
        assert base_url == expected_url

    def test_select_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        # NVIDIA env set but ignored when SKILLSPECTOR_PROVIDER=openai.
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "should-be-ignored")
        creds = resolve_provider_credentials()
        assert creds == ("sk-x", None)
        assert isinstance(get_metadata_provider(), OpenAIProvider)

    def test_select_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        creds = resolve_provider_credentials()
        assert creds == ("sk-ant-x", None)
        assert isinstance(get_metadata_provider(), AnthropicProvider)

    def test_create_chat_model_uses_native_anthropic_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-should-not-win")
        llm = create_chat_model("claude-opus-4-6", max_tokens=123)
        assert isinstance(llm, ChatAnthropic)
        assert llm.model == "claude-opus-4-6"

    def test_chat_model_credentials_fall_back_to_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        creds = resolve_chat_model_credentials()
        assert creds == ("sk-x", None)

    def test_select_nv_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "nv_build")
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvapi-x")
        creds = resolve_provider_credentials()
        assert creds == ("nvapi-x", BUILD_BASE_URL)
        assert isinstance(get_metadata_provider(), NvBuildProvider)

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "vertex")
        with pytest.raises(ValueError, match="Unknown SKILLSPECTOR_PROVIDER"):
            get_metadata_provider()

    def test_falls_back_to_nv_build_when_nv_inference_unimportable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the optional nv_inference subpackage can't be imported,
        the default/``nv_inference`` selection degrades to ``NvBuildProvider``."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "nv_inference")
        # Setting the module entry to None forces ``import`` to raise ImportError.
        monkeypatch.setitem(sys.modules, "skillspector.providers.nv_inference", None)
        assert isinstance(get_metadata_provider(), NvBuildProvider)

    def test_create_chat_model_falls_back_to_openai_when_provider_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Active provider is anthropic but ANTHROPIC_API_KEY is unset, so it
        # yields no client; OPENAI_API_KEY then satisfies the fallback.
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        llm = create_chat_model("gpt-5.4", max_tokens=123)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-5.4"

    def test_create_chat_model_raises_when_no_credentials_anywhere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Anthropic active, but neither ANTHROPIC_API_KEY nor OPENAI_API_KEY set.
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        with pytest.raises(ValueError) as exc_info:
            create_chat_model("claude-opus-4-6", max_tokens=123)
        assert str(exc_info.value) == NO_LLM_API_KEY_MESSAGE

    def test_create_chat_model_raises_for_openai_provider_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the active provider is already OpenAI, there is no second
        # fallback attempt — it raises directly.
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "openai")
        with pytest.raises(ValueError) as exc_info:
            create_chat_model("gpt-5.4", max_tokens=123)
        assert str(exc_info.value) == NO_LLM_API_KEY_MESSAGE
