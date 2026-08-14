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

"""Anthropic provider — Claude models via api.anthropic.com.

Reads ``ANTHROPIC_API_KEY`` for credentials and honors ``ANTHROPIC_BASE_URL``
as an explicit endpoint override (e.g. a local proxy); when unset, requests
go to api.anthropic.com. Constructs ``langchain_anthropic.ChatAnthropic``
directly. It defaults to Opus 4.6 for analyzers and Sonnet 4.6 for
``meta_analyzer`` (cheaper for the high-volume filter pass).
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from skillspector.providers import registry
from skillspector.providers.chat_models import resolve_reasoning_effort

# Default endpoint; overridden by ``ANTHROPIC_BASE_URL`` when set.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class AnthropicProvider:
    """Anthropic credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL = "claude-opus-4-6"
    SLOT_DEFAULTS: dict[str, str] = {
        "meta_analyzer": "claude-sonnet-4-6",
    }

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, base_url)`` from ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL``."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip() or None
        return api_key, base_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatAnthropic`` using native Anthropic credentials."""
        creds = self.resolve_credentials()
        if creds is None:
            return None

        api_key, base_url = creds
        kwargs = {
            "model_name": model,
            "api_key": SecretStr(api_key),
            "base_url": base_url or ANTHROPIC_BASE_URL,
            "max_tokens_to_sample": max_tokens,
            "timeout": timeout,
            "stop": None,
        }
        effort = resolve_reasoning_effort()
        if effort is not None:
            kwargs["effort"] = effort
        return ChatAnthropic(**kwargs)

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
