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

"""Tests for the MiniMax provider."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

import skillspector.providers as providers_module
from skillspector.providers import get_metadata_provider, registry, resolve_provider_credentials
from skillspector.providers._agent_cli import _scrub_env
from skillspector.providers.minimax import (
    MINIMAX_CN_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    MiniMaxProvider,
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "MINIMAX_API_KEY",
        "MINIMAX_BASE_URL",
        "MINIMAX_REGION",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    providers_module._INJECTED_PROVIDER.set(None)
    registry._load.cache_clear()
    yield
    providers_module._INJECTED_PROVIDER.set(None)
    registry._load.cache_clear()


class TestMiniMaxProvider:
    def test_returns_none_without_api_key(self) -> None:
        assert MiniMaxProvider().resolve_credentials() is None

    def test_uses_global_endpoint_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        assert MiniMaxProvider().resolve_credentials() == (
            "test-key",
            MINIMAX_GLOBAL_BASE_URL,
        )

    def test_selects_china_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
        assert MiniMaxProvider().resolve_credentials() == (
            "test-key",
            MINIMAX_CN_BASE_URL,
        )

    def test_honors_base_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://minimax.example.com/v1")
        assert MiniMaxProvider().resolve_credentials() == (
            "test-key",
            "https://minimax.example.com/v1",
        )

    def test_rejects_unknown_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("MINIMAX_REGION", "unknown")
        with pytest.raises(ValueError, match="global_en.*cn_zh"):
            MiniMaxProvider().resolve_credentials()

    def test_creates_chat_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        llm = MiniMaxProvider().create_chat_model("MiniMax-M2.7", max_tokens=123)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "MiniMax-M2.7"
        assert llm.max_tokens == 123
        assert str(llm.openai_api_base).rstrip("/") == MINIMAX_GLOBAL_BASE_URL

    def test_bundled_models_and_context_windows(self) -> None:
        provider = MiniMaxProvider()
        assert provider.resolve_model() == "MiniMax-M2.7"
        assert provider.get_context_length("MiniMax-M3") == 1_000_000
        assert provider.get_context_length("MiniMax-M2.7") == 204_800
        assert provider.get_max_output_tokens("MiniMax-M3") is None

    def test_selector_uses_minimax_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "minimax")
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        assert resolve_provider_credentials() == ("test-key", MINIMAX_GLOBAL_BASE_URL)
        assert isinstance(get_metadata_provider(), MiniMaxProvider)

    def test_api_key_is_removed_from_cli_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        assert "MINIMAX_API_KEY" not in _scrub_env()
