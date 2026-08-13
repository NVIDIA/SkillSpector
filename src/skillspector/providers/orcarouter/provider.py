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

"""OrcaRouter provider — model-routing gateway via api.orcarouter.ai.

Reads ``ORCAROUTER_API_KEY`` for credentials and serves the OpenAI-compatible
``https://api.orcarouter.ai/v1`` endpoint (optionally overridden with
``ORCAROUTER_BASE_URL``). The ``orcarouter/auto`` model routes each request
to the best available frontier model for the task, so no per-model API key
or model id is required.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry
from skillspector.providers.chat_models import create_openai_compatible_chat_model

# Default endpoint; overridden by ``ORCAROUTER_BASE_URL`` when set.
ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class OrcaRouterProvider:
    """OrcaRouter credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL = "orcarouter/auto"
    SLOT_DEFAULTS: dict[str, str] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, base_url)`` from ``ORCAROUTER_API_KEY`` / ``ORCAROUTER_BASE_URL``."""
        api_key = os.environ.get("ORCAROUTER_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("ORCAROUTER_BASE_URL", "").strip() or ORCAROUTER_BASE_URL
        return api_key, base_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatOpenAI`` for the OrcaRouter OpenAI-compatible endpoint."""
        return create_openai_compatible_chat_model(
            model=model,
            credentials=self.resolve_credentials(),
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
