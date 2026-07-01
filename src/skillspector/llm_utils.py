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

"""Shared LLM utilities.

Credentials are resolved in this order:
    1. The active SkillSpector provider (see :mod:`skillspector.providers`) —
       reads its own credential env var and supplies the matching client.
    2. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` (the langchain-openai
       defaults).

There is no SkillSpector-specific credential env var: setting
``NVIDIA_INFERENCE_KEY`` configures whichever NVIDIA endpoint the
deployment ships with, Anthropic reads ``ANTHROPIC_API_KEY``, and any
other OpenAI-compatible endpoint is configured via the standard
``OPENAI_*`` envs.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from skillspector.model_info import get_max_input_tokens, get_max_output_tokens
from skillspector.providers import (
    create_chat_model,
    get_metadata_provider,
    raise_no_llm_api_key_configured,
    resolve_chat_model_credentials,
    resolve_provider_credentials,
)
from skillspector.providers.openai import OpenAIProvider


def _resolve_llm_credentials() -> tuple[str, str | None]:
    """Return ``(api_key, base_url)`` resolved from the environment.

    Tries the active NVIDIA provider first; falls back to ``OPENAI_API_KEY``
    / ``OPENAI_BASE_URL`` when the provider is not configured.

    Raises:
        ValueError: when no API key can be resolved from any source.
    """
    creds = resolve_chat_model_credentials()
    if creds is None:
        raise_no_llm_api_key_configured()
    return creds


def _resolve_default_chat_model() -> str:
    """Return the default chat model for the endpoint that will be used."""
    if resolve_provider_credentials() is not None:
        return get_metadata_provider().resolve_model()

    openai_provider = OpenAIProvider()
    if openai_provider.resolve_credentials() is not None:
        return openai_provider.resolve_model()

    raise_no_llm_api_key_configured()


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


def get_chat_model(model: str | None = None) -> BaseChatModel:
    """Return the active provider's native LangChain chat model.

    Raises:
        ValueError: when no API key is configured (see ``is_llm_available``).
    """
    model = model or _resolve_default_chat_model()
    return create_chat_model(
        model=model,
        max_tokens=get_max_output_tokens(model),
        timeout=120,
    )


def chat_completion(prompt: str, *, model: str | None = None) -> str:
    """Request a single chat completion and return the assistant content.

    Routes through :func:`get_chat_model`, which dispatches to the CLI adapter
    for CLI providers and to the provider's native chat model for HTTP providers.

    Uses ``.text`` when available (real LangChain ``BaseMessage`` objects,
    which normalise content blocks to a single string) and falls back to
    ``.content`` for the CLI adapter's ``_AgentCLIMessage``.
    """
    response = get_chat_model(model=model).invoke(prompt)
    if hasattr(response, "text"):
        return response.text  # type: ignore[union-attr]
    return response.content or ""  # type: ignore[union-attr]


def run_async(coroutine: Coroutine) -> Any:
    """
    Run an async coroutine in a synchronous context, even if there's already a running event loop.

    This function safely handles nested event loop scenarios (e.g. Jupyter Notebooks, FastAPI,
    LangGraph Studio) by offloading the coroutine execution to a separate thread with its own
    event loop when a running loop is detected.

    Args:
        coroutine: The async coroutine to run

    Returns:
        The result of the coroutine execution

    Raises:
        Any exception raised by the coroutine is re-raised as-is
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()
