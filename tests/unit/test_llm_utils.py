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

Order: active NVIDIA provider (NVIDIA_INFERENCE_KEY) -> OPENAI_API_KEY /
OPENAI_BASE_URL.  NVIDIA-specific behavior (which env var resolves to
which endpoint) lives in the active provider — see ``tests/unit/test_providers.py``.
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from skillspector.llm_utils import (
    _resolve_llm_credentials,
    get_chat_model,
    is_llm_available,
)
from skillspector.providers import resolve_provider_credentials
from skillspector.providers.nv_build import NvBuildProvider
from skillspector.providers.openai import OpenAIProvider

_LLM_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "NVIDIA_INFERENCE_KEY",
    "NVIDIA_INFERENCE_METADATA_KEY",
    "ANTHROPIC_API_KEY",
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
    """Order: active NVIDIA provider first, then OPENAI_API_KEY / OPENAI_BASE_URL."""

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

    def test_no_credentials_raises_with_helpful_message(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            _resolve_llm_credentials()


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
        assert msg is not None
        assert "API key" in msg


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
        monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "nvapi-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

        llm = get_chat_model()

        assert _chat_model_name(llm) == NvBuildProvider.DEFAULT_MODEL


def _chat_model_name(llm: ChatOpenAI) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", None))
