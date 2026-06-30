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

"""VertexAI provider — Gemini models via VertexAI OpenAI-compatible endpoint.

Reads ``GOOGLE_APPLICATION_CREDENTIALS``, ``GOOGLE_CLOUD_PROJECT``, and
``GOOGLE_CLOUD_LOCATION`` for credentials and constructs the VertexAI
OpenAI-compatible endpoint URL.  Uses Google Cloud Application Default
Credentials (ADC) to generate access tokens.  Defaults to Gemini 2.5 Flash.
"""

from __future__ import annotations

import os
from pathlib import Path

import google.auth
import google.auth.transport.requests
from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry
from skillspector.providers.chat_models import create_openai_compatible_chat_model

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class VertexAIProvider:
    """Stock VertexAI credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    SLOT_DEFAULTS: dict[str, str] = {}

    WIRE_MODEL_PREFIX = "google/"

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(access_token, base_url)`` from Google Cloud credentials.

        Uses Application Default Credentials (ADC) via ``google.auth.default()``.
        The access token is refreshed from the credentials object and returned
        as the API key for the OpenAI-compatible client.

        Returns ``None`` when required environment variables are not set.

        Raises:
            google.auth.exceptions.DefaultCredentialsError: When credentials
                are configured but invalid or malformed.
            ValueError: When token refresh fails.
        """

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()

        if not project_id or not location:
            return None

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        credentials.refresh(google.auth.transport.requests.Request())

        access_token = credentials.token
        if not access_token:
            raise ValueError(
                "Failed to obtain access token from Google Cloud credentials. "
                "Ensure GOOGLE_APPLICATION_CREDENTIALS points to a valid "
                "service account key file."
            )

        base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi"
        )

        return access_token, base_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatOpenAI`` for the VertexAI OpenAI-compatible endpoint.

        The endpoint requires model names prefixed with ``google/``
        (e.g. ``google/gemini-2.5-flash``).  The prefix is applied here
        at the wire boundary so that registry lookups and token-budget
        calculations continue to use bare model labels.
        """
        wire_model = model if model.startswith(self.WIRE_MODEL_PREFIX) else f"{self.WIRE_MODEL_PREFIX}{model}"
        return create_openai_compatible_chat_model(
            model=wire_model,
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
