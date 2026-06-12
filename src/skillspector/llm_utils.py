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

"""Shared LLM utilities (OpenAI-compatible chat models).

Credentials are resolved in this order:
    1. The active NVIDIA provider (see :mod:`skillspector.providers`) —
       reads ``NVIDIA_INFERENCE_KEY`` and supplies the matching endpoint.
    2. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` (the langchain-openai
       defaults).

There is no SkillSpector-specific credential env var: setting
``NVIDIA_INFERENCE_KEY`` configures whichever NVIDIA endpoint the
deployment ships with, and any other OpenAI-compatible endpoint is
configured via the standard ``OPENAI_*`` envs.
"""

from __future__ import annotations

import asyncio
import os
import time

from langchain_openai import ChatOpenAI

from skillspector.constants import MODEL_CONFIG
from skillspector.model_info import get_max_input_tokens, get_max_output_tokens
from skillspector.providers import resolve_provider_credentials


def _resolve_llm_credentials() -> tuple[str, str | None]:
    """Return ``(api_key, base_url)`` resolved from the environment.

    Tries the active NVIDIA provider first; falls back to ``OPENAI_API_KEY``
    / ``OPENAI_BASE_URL`` when the provider is not configured.

    Raises:
        ValueError: when no API key can be resolved from any source.
    """
    creds = resolve_provider_credentials()
    if creds is not None:
        return creds

    resolved_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not resolved_key:
        raise ValueError(
            "No LLM API key configured. Set the credential env var for the "
            "active provider, or set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL) to use a standard OpenAI-compatible endpoint. "
            "Use --no-llm to skip LLM analysis and run static checks only."
        )

    resolved_base = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return resolved_key, resolved_base


def is_llm_available() -> tuple[bool, str | None]:
    """Return ``(available, error_message)`` describing LLM credential status."""
    try:
        _resolve_llm_credentials()
    except ValueError as exc:
        return False, str(exc)
    return True, None


def fetch_model_token_limits(model_label: str) -> tuple[int, int]:
    """Return ``(max_input_tokens, max_output_tokens)`` for *model_label*."""
    return get_max_input_tokens(model_label), get_max_output_tokens(model_label)


def get_chat_model(model: str | None = None) -> ChatOpenAI:
    """Return a :class:`ChatOpenAI` configured against the resolved endpoint.

    Raises:
        ValueError: when no API key is configured (see ``is_llm_available``).
    """
    resolved_key, resolved_base = _resolve_llm_credentials()
    model = model or MODEL_CONFIG["default"]

    return ChatOpenAI(
        model=model,
        base_url=resolved_base,
        api_key=resolved_key,
        max_tokens=get_max_output_tokens(model),
        timeout=120,
    )


def chat_completion(prompt: str, *, model: str | None = None) -> str:
    """Request a single chat completion and return the assistant content."""
    llm = get_chat_model(model=model)
    response = llm.invoke(prompt)
    return response.content or ""


def retry_llm_call_sync(call_func, max_attempts=4):
    """Retry transient LLM errors (429, timeout) with exponential backoff (sync)."""
    for attempt in range(max_attempts):
        try:
            return call_func()  # Call it each retry
        except Exception as e:
            error_str = str(e).lower()
            error_name = type(e).__name__.lower()

            is_retryable = (
                "429" in error_str
                or "ratelimit" in error_name
                or "timeout" in error_name
                or "timeout" in error_str
            )

            if not is_retryable:
                raise
            if attempt == max_attempts - 1:
                raise

            wait = 2**attempt
            time.sleep(wait)


async def retry_llm_call(coro_func, max_attempts=4):
    """Retry transient LLM errors (429, timeout) with exponential backoff (async)."""
    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except Exception as e:
            error_str = str(e).lower()
            error_name = type(e).__name__.lower()

            is_retryable = (
                "429" in error_str
                or "ratelimit" in error_name
                or "timeout" in error_name
                or "timeout" in error_str
            )

            if not is_retryable:
                raise
            if attempt == max_attempts - 1:
                raise

            wait = 2**attempt
            await asyncio.sleep(wait)
