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

"""Azure OpenAI provider — enterprise Azure-hosted OpenAI deployments.

Uses ``AzureChatOpenAI`` from ``langchain_openai`` which handles Azure's
deployment-based routing and ``api-version`` query parameter natively.

Required env vars:
    AZURE_OPENAI_ENDPOINT    — Azure resource endpoint
    AZURE_OPENAI_API_KEY     — Azure API key

Optional env vars:
    AZURE_OPENAI_DEPLOYMENT  — deployment name (defaults to model label)
    AZURE_OPENAI_API_VERSION — API version (defaults to ``2024-06-01``)
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from skillspector.providers import registry

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class AzureOpenAIProvider:
    """Azure OpenAI credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL = "gpt-4o"
    SLOT_DEFAULTS: dict[str, str] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, endpoint)`` from Azure OpenAI env vars."""
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        if not api_key or not endpoint:
            return None
        return api_key, endpoint

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``AzureChatOpenAI`` using Azure-specific credentials."""
        creds = self.resolve_credentials()
        if creds is None:
            return None

        api_key, endpoint = creds
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip() or model
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip() or "2024-06-01"

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=SecretStr(api_key),
            api_version=api_version,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
