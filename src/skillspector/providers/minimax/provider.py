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

"""MiniMax provider with global and China regional endpoint selection.

``MINIMAX_REGION`` accepts ``global_en`` (the default) or ``cn_zh``.
``MINIMAX_BASE_URL`` can override the selected regional endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry
from skillspector.providers.chat_models import create_openai_compatible_chat_model

MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_BASE_URLS = {
    "global_en": MINIMAX_GLOBAL_BASE_URL,
    "cn_zh": MINIMAX_CN_BASE_URL,
}

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


def _resolve_base_url() -> str:
    override = os.environ.get("MINIMAX_BASE_URL", "").strip()
    if override:
        return override

    region = os.environ.get("MINIMAX_REGION", "").strip().lower() or "global_en"
    try:
        return MINIMAX_BASE_URLS[region]
    except KeyError as exc:
        raise ValueError("MINIMAX_REGION must be 'global_en' or 'cn_zh'") from exc


class MiniMaxProvider:
    """MiniMax credentials, regional routing, and bundled model metadata."""

    DEFAULT_MODEL = "MiniMax-M3"
    SLOT_DEFAULTS: dict[str, str] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return the MiniMax API key and selected regional base URL."""
        api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if not api_key:
            return None
        return api_key, _resolve_base_url()

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create a chat model for the selected MiniMax endpoint."""
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
        """Resolve model from an environment override or the bundled default."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
