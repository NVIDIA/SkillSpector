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

"""OpenAI provider — stock api.openai.com (or any OpenAI-compatible endpoint).

Reads ``OPENAI_API_KEY`` for credentials and honors ``OPENAI_BASE_URL`` as
an explicit endpoint override.  When the base URL is unset, returns
``None`` so ``langchain_openai.ChatOpenAI`` defaults to api.openai.com.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry
from skillspector.providers.chat_models import create_openai_compatible_chat_model

# Documented for completeness — ChatOpenAI defaults here when base_url=None.
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


def _resolve_openai_project_headers() -> dict[str, str] | None:
    project_id = os.environ.get("OPENAI_PROJECT_ID", "").strip()
    if not project_id:
        return None
    return {"OpenAI-Project": project_id}


class OpenAIProvider:
    """Stock OpenAI credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL = "gpt-5.4"
    SLOT_DEFAULTS: dict[str, str] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, base_url)`` from ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``."""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        return api_key, base_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatOpenAI`` using standard OpenAI environment variables."""
        return create_openai_compatible_chat_model(
            model=model,
            credentials=self.resolve_credentials(),
            max_tokens=max_tokens,
            timeout=timeout,
            default_headers=_resolve_openai_project_headers(),
        )

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
