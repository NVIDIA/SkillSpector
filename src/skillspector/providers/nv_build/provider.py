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

"""build.nvidia.com provider — OSS-friendly NVIDIA path.

Resolves ``NVIDIA_INFERENCE_KEY`` to the OpenAI-compatible endpoint
served at https://integrate.api.nvidia.com/v1.  Token-budget metadata
comes from the bundled ``model_registry.yaml`` next to this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry
from skillspector.providers.chat_models import create_openai_compatible_chat_model

BUILD_BASE_URL = "https://integrate.api.nvidia.com/v1"

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class NvBuildProvider:
    """build.nvidia.com credentials + bundled-YAML metadata provider."""

    # General default — DeepSeek v4 flash for the high-volume per-file
    # analyzer calls.  meta_analyzer is upgraded to v4 pro for the
    # aggregation/filter pass where precision matters more.
    DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"
    SLOT_DEFAULTS: dict[str, str] = {
        "meta_analyzer": "deepseek-ai/deepseek-v4-pro",
    }

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, base_url)`` from ``NVIDIA_INFERENCE_KEY``."""
        api_key = os.environ.get("NVIDIA_INFERENCE_KEY", "").strip()
        if not api_key:
            return None
        return api_key, BUILD_BASE_URL

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatOpenAI`` for the build.nvidia.com endpoint."""
        return create_openai_compatible_chat_model(
            model=model,
            credentials=self.resolve_credentials(),
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def get_context_length(self, model: str) -> int | None:
        """Look up *model*'s context window in the bundled ``model_registry.yaml``."""
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        """Look up *model*'s max-output cap in the bundled ``model_registry.yaml``."""
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
