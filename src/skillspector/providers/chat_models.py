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

"""Shared constructors for provider-backed LangChain chat models."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)


def resolve_reasoning_effort() -> str | None:
    """Resolve the optional provider- and model-dependent reasoning effort."""
    reasoning_effort = os.environ.get("SKILLSPECTOR_REASONING_EFFORT", "").strip()
    return reasoning_effort or None


def validate_base_url(url: str | None) -> None:
    """Warn if *url* is not a well-formed http(s) URL.

    Raises nothing — misconfigured URLs will still fail at the HTTP
    layer, but an early warning helps operators catch typos.
    """
    if url is None:
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning(
            "Provider base_url %r has scheme %r — expected http or https. "
            "Requests will likely fail.",
            url,
            parsed.scheme or "(empty)",
        )
    if not parsed.netloc:
        logger.warning(
            "Provider base_url %r has no host component. Requests will likely fail.",
            url,
        )


def create_openai_compatible_chat_model(
    *,
    model: str,
    credentials: tuple[str, str | None] | None,
    max_tokens: int,
    timeout: float | None = 120,
    default_headers: dict[str, str] | None = None,
) -> BaseChatModel | None:
    """Create ``ChatOpenAI`` for providers serving OpenAI-compatible endpoints."""
    if credentials is None:
        return None

    api_key, base_url = credentials
    validate_base_url(base_url)
    kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": SecretStr(api_key),
        "max_completion_tokens": max_tokens,
        "timeout": timeout,
        "default_headers": default_headers,
    }
    reasoning_effort = resolve_reasoning_effort()
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)
