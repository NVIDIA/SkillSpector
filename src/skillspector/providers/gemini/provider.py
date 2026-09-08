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

"""Gemini on Google Cloud provider using Application Default Credentials (ADC).

Serves Gemini chat models through Google Cloud's OpenAI-compatible endpoint
using Google Cloud Application Default Credentials (ADC) or GKE Workload
Identity.

Required env vars:
    GOOGLE_CLOUD_PROJECT    — Google Cloud project ID (required)

Optional env vars:
    GOOGLE_CLOUD_LOCATION   — Google Cloud location (default: global)
    GOOGLE_APPLICATION_CREDENTIALS — Optional path to credential/config JSON file
    SKILLSPECTOR_MODEL      — Model override
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import ClassVar

import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request
from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry
from skillspector.providers.chat_models import create_openai_compatible_chat_model

logger = logging.getLogger(__name__)

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))

_AUTH_LOCK = threading.Lock()
_CACHED_CREDENTIALS: google.auth.credentials.Credentials | None = None

_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_RESTRICTED_PROJECT_SUBSTRINGS = ("google", "ssl")
_RESTRICTED_EXACT_PROJECTS = ("undefined", "null")
_LOCATION_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,30}[a-z0-9]$")


def validate_project_id(project: str) -> None:
    """Validate that *project* complies with Google Cloud project ID rules."""
    if not project or not _PROJECT_ID_PATTERN.match(project):
        raise ValueError(
            f"Invalid GOOGLE_CLOUD_PROJECT: {project!r}. Must be 6-30 characters of lowercase "
            "letters, numbers, and hyphens, start with a letter, and not end with a hyphen."
        )
    if project in _RESTRICTED_EXACT_PROJECTS or project.endswith("-none"):
        raise ValueError(
            f"Invalid GOOGLE_CLOUD_PROJECT: {project!r} uses a restricted project identifier."
        )
    for restricted in _RESTRICTED_PROJECT_SUBSTRINGS:
        if restricted in project:
            raise ValueError(
                f"Invalid GOOGLE_CLOUD_PROJECT: {project!r} contains restricted string {restricted!r}."
            )


def validate_location(location: str) -> None:
    """Validate that *location* is a valid GCP region or multi-region identifier."""
    if not location or not _LOCATION_PATTERN.match(location):
        raise ValueError(
            f"Invalid GOOGLE_CLOUD_LOCATION: {location!r}. Must be a valid region identifier "
            "(lowercase alphanumeric and hyphens, e.g. 'global', 'us', 'eu', 'us-central1')."
        )


def get_base_url(project: str, location: str) -> str:
    """Return the Google Cloud OpenAI-compatible Gemini base URL."""
    if location == "global":
        return f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"
    if location == "us":
        return f"https://aiplatform.us.rep.googleapis.com/v1/projects/{project}/locations/us/endpoints/openapi"
    if location == "eu":
        return f"https://aiplatform.eu.rep.googleapis.com/v1/projects/{project}/locations/eu/endpoints/openapi"
    return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"


def _reset_cached_credentials() -> None:
    """Reset cached ADC credentials. Used primarily for test isolation."""
    global _CACHED_CREDENTIALS
    with _AUTH_LOCK:
        _CACHED_CREDENTIALS = None


def _get_credentials() -> str:
    # ponytail: cache ADC credentials and refresh under a lock only when invalid/expired
    global _CACHED_CREDENTIALS
    creds = _CACHED_CREDENTIALS
    if creds is not None and getattr(creds, "valid", False):
        token = getattr(creds, "token", None)
        if token:
            return str(token)

    with _AUTH_LOCK:
        if _CACHED_CREDENTIALS is None:
            try:
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                _CACHED_CREDENTIALS = creds
            except (DefaultCredentialsError, RefreshError) as exc:
                raise ValueError(
                    f"Failed to obtain Google Cloud Application Default Credentials: {exc}"
                ) from exc
            except Exception as exc:
                logger.debug("Unexpected error obtaining Google Cloud credentials", exc_info=True)
                raise ValueError(
                    f"Unexpected error obtaining Google Cloud credentials: {exc}"
                ) from exc

        if not _CACHED_CREDENTIALS.valid:
            try:
                _CACHED_CREDENTIALS.refresh(Request())
            except (DefaultCredentialsError, RefreshError) as exc:
                raise ValueError(
                    f"Failed to refresh Google Cloud Application Default Credentials: {exc}"
                ) from exc
            except Exception as exc:
                logger.debug("Unexpected error refreshing Google Cloud credentials", exc_info=True)
                raise ValueError(
                    f"Unexpected error refreshing Google Cloud credentials: {exc}"
                ) from exc

        token = getattr(_CACHED_CREDENTIALS, "token", None)
        if not token:
            raise ValueError(
                "Google Cloud Application Default Credentials refresh yielded no access token."
            )
        return str(token)


class GeminiProvider:
    """Gemini on Google Cloud credentials + bundled-YAML metadata provider."""

    DEFAULT_MODEL: ClassVar[str] = "gemini-3.8-flash"
    SLOT_DEFAULTS: ClassVar[dict[str, str]] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(access_token, base_url)`` using Google ADC.

        Deliberately raises ``ValueError`` when ``GOOGLE_CLOUD_PROJECT`` is missing or
        invalid, rather than returning ``None``. This enforces fail-closed behavior so
        the scan never silently falls back to OpenAI when Gemini was explicitly chosen.
        """
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for the Gemini provider. "
                "Set it to your Google Cloud project ID."
            )
        validate_project_id(project)
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or "global"
        validate_location(location)
        base_url = get_base_url(project, location)
        token = _get_credentials()
        return token, base_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatOpenAI`` for Google Cloud's OpenAI-compatible Gemini endpoint."""
        wire_model = model if model.startswith("google/") else f"google/{model}"
        credentials = self.resolve_credentials()
        default_headers: dict[str, str] = {}
        quota_project = getattr(_CACHED_CREDENTIALS, "quota_project_id", None)
        if quota_project:
            default_headers["x-goog-user-project"] = quota_project

        return create_openai_compatible_chat_model(
            model=wire_model,
            credentials=credentials,
            max_tokens=max_tokens,
            timeout=timeout,
            default_headers=default_headers,
        )

    def get_context_length(self, model: str) -> int | None:
        bare_model = model.removeprefix("google/")
        return registry.lookup_context_length(REGISTRY_PATH, bare_model)

    def get_max_output_tokens(self, model: str) -> int | None:
        bare_model = model.removeprefix("google/")
        return registry.lookup_max_output_tokens(REGISTRY_PATH, bare_model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        model = (
            user_input.removeprefix("google/")
            if user_input
            else (self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL)
        )
        return model
