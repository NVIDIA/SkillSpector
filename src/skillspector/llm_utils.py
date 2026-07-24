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

"""Shared LLM utilities (OpenAI-compatible chat models + agent CLI transports).

Credentials are resolved in this order:
    1. The active provider (see :mod:`skillspector.providers`):
       - CLI providers (``claude_cli``, ``codex_cli``, ``gemini_cli``): use
         ``is_available()`` and ``complete()`` — no API key needed.
       - HTTP providers (``anthropic``, ``openai``, ``nv_build``): read their
         respective credential env vars and supply a base URL.
    2. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` (the langchain-openai
       defaults) — only consulted for HTTP providers when the provider's
       own credential env var is unset.

There is no SkillSpector-specific credential env var: setting
``NVIDIA_INFERENCE_KEY`` configures whichever NVIDIA endpoint the
deployment ships with, Anthropic reads ``ANTHROPIC_API_KEY``, and any
other OpenAI-compatible endpoint is configured via the standard
``OPENAI_*`` envs.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import Coroutine
from typing import Any, NoReturn, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from skillspector.model_info import get_max_input_tokens, get_max_output_tokens
from skillspector.providers import (
    create_chat_model,
    get_active_provider,
    get_metadata_provider,
    has_cli_capability,
    has_provider_binding,
    raise_no_llm_api_key_configured,
    resolve_chat_model_credentials,
    resolve_provider_credentials,
)
from skillspector.providers.openai import OpenAIProvider


class LLMTokenUsage(TypedDict):
    """Provider-normalized token usage for LLM calls."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


def empty_token_usage() -> LLMTokenUsage:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_token_usage(raw: object) -> LLMTokenUsage:
    usage = getattr(raw, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return empty_token_usage()
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _resolve_llm_credentials() -> tuple[str, str | None]:
    """Return ``(api_key, base_url)`` resolved from the environment.

    Tries the active SkillSpector provider first; falls back to
    ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` when the provider is not
    configured.

    Raises:
        ValueError: when no API key can be resolved from any source.
    """
    creds = resolve_chat_model_credentials()
    if creds is None:
        raise_no_llm_api_key_configured()
    return creds


def _resolve_default_chat_model() -> str:
    """Return the default chat model for the endpoint that will be used."""
    if has_provider_binding():
        return get_metadata_provider().resolve_model()

    if resolve_provider_credentials() is not None:
        return get_metadata_provider().resolve_model()

    openai_provider = OpenAIProvider()
    if openai_provider.resolve_credentials() is not None:
        return openai_provider.resolve_model()

    raise_no_llm_api_key_configured()


def is_llm_available() -> tuple[bool, str | None]:
    """Return ``(available, error_message)`` describing LLM availability.

    CLI providers (``claude_cli``, ``codex_cli``, ``gemini_cli``) are checked
    through their ``is_available()`` method first.  Other providers probe the
    same native chat-model path used by :func:`get_chat_model`; unbound HTTP
    providers keep the credential-resolution and OpenAI fallback path.
    """
    provider = get_active_provider()
    if has_cli_capability(provider):
        return provider.is_available()  # type: ignore[attr-defined]

    if has_provider_binding():
        try:
            model = provider.resolve_model()
            create_chat_model(
                model=model,
                max_tokens=get_max_output_tokens(model),
                timeout=120,
            )
        except ValueError as exc:
            return False, str(exc)
        return True, None
    try:
        _resolve_llm_credentials()
    except ValueError as exc:
        return False, str(exc)
    return True, None


def fetch_model_token_limits(model_label: str) -> tuple[int, int]:
    """Return ``(max_input_tokens, max_output_tokens)`` for *model_label*."""
    return get_max_input_tokens(model_label), get_max_output_tokens(model_label)


# ---------------------------------------------------------------------------
# Agent CLI chat-model adapter
# ---------------------------------------------------------------------------
#
# The LLM analyzers (meta_analyzer, semantic_*) obtain a model from
# ``get_chat_model()`` and call ``.invoke()`` / ``.with_structured_output(
# schema).invoke()`` on it (see ``llm_analyzer_base``) — they never go through
# ``chat_completion``. To support CLI providers there, ``get_chat_model``
# returns this minimal adapter, which mimics the slice of the ``ChatOpenAI``
# interface the analyzers rely on, backed by the provider's ``complete()``
# subprocess transport.


class _AgentCLIMessage:
    """Minimal stand-in for a LangChain message: exposes ``.content``."""

    def __init__(self, content: str) -> None:
        self.content = content


def _extract_json_object(raw: str) -> dict:
    """Extract a single JSON object from a CLI model's text response.

    Tolerates markdown code fences and surrounding prose. Raises ``ValueError``
    (fail-closed) when no JSON object can be parsed.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and any closing fence.
        text = text.split("\n", 1)[1] if "\n" in text else ""
        fence = text.rfind("```")
        if fence != -1:
            text = text[:fence]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError(f"could not extract a JSON object from CLI response: {raw[:200]!r}")


class _StructuredAgentCLIModel:
    """Mimics ``ChatOpenAI.with_structured_output(schema)`` for a CLI provider.

    ``invoke`` augments the prompt with the schema, calls the provider's
    ``complete()``, then parses and validates the response into *schema*.
    """

    def __init__(
        self,
        provider: object,
        model: str,
        max_output_tokens: int,
        schema: type[BaseModel],
        include_raw: bool = False,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._schema = schema
        self._include_raw = include_raw

    def _augment(self, prompt: str) -> str:
        schema_json = json.dumps(self._schema.model_json_schema(), indent=2)
        return (
            f"{prompt}\n\n"
            "Respond with ONLY a single JSON object conforming to the JSON Schema "
            "below. Do not wrap it in markdown code fences and do not add any prose "
            f"before or after the JSON.\n\nJSON Schema:\n{schema_json}"
        )

    def invoke(self, prompt: str) -> object:
        raw = self._provider.complete(  # type: ignore[attr-defined]
            self._augment(prompt),
            model=self._model,
            max_output_tokens=self._max_output_tokens,
        )
        try:
            parsed = self._schema.model_validate(_extract_json_object(raw))
            parsing_error = None
        except Exception as exc:
            parsed = None
            parsing_error = exc
            if not self._include_raw:
                raise
        if self._include_raw:
            return {"raw": _AgentCLIMessage(raw), "parsed": parsed, "parsing_error": parsing_error}
        return parsed

    async def ainvoke(self, prompt: str) -> object:
        return await asyncio.to_thread(self.invoke, prompt)


class AgentCLIChatModel:
    """Minimal ``ChatOpenAI``-compatible adapter backed by a CLI provider.

    Implements only the surface the analyzers use: ``invoke`` (returns an
    object with ``.content``), ``ainvoke``, and ``with_structured_output``.
    The rest of the ``BaseChatModel`` surface (``batch``, ``stream``,
    callbacks) is intentionally unsupported; the stubs below make that boundary
    explicit so a future analyzer reaching for it fails loudly with a clear
    message rather than a confusing ``AttributeError``.
    """

    def __init__(self, provider: object, model: str, max_output_tokens: int) -> None:
        self._provider = provider
        self._model = model
        self._max_output_tokens = max_output_tokens

    def batch(self, *args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError(
            "AgentCLIChatModel supports only invoke/ainvoke/with_structured_output; "
            "batch() is not available for CLI providers."
        )

    def stream(self, *args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError(
            "AgentCLIChatModel supports only invoke/ainvoke/with_structured_output; "
            "stream() is not available for CLI providers."
        )

    def invoke(self, prompt: str) -> _AgentCLIMessage:
        text = self._provider.complete(  # type: ignore[attr-defined]
            prompt,
            model=self._model,
            max_output_tokens=self._max_output_tokens,
        )
        return _AgentCLIMessage(text)

    async def ainvoke(self, prompt: str) -> _AgentCLIMessage:
        return await asyncio.to_thread(self.invoke, prompt)

    def with_structured_output(
        self, schema: type[BaseModel], *, include_raw: bool = False
    ) -> _StructuredAgentCLIModel:
        return _StructuredAgentCLIModel(
            self._provider, self._model, self._max_output_tokens, schema, include_raw=include_raw
        )


def get_chat_model(model: str | None = None) -> BaseChatModel | AgentCLIChatModel:
    """Return a chat model for the active provider.

    For CLI providers (``claude_cli``, ``codex_cli``, ``gemini_cli``) this
    returns an :class:`AgentCLIChatModel` adapter backed by the provider's
    ``complete()`` subprocess transport — so the LLM analyzers (which use
    ``.invoke()`` and ``.with_structured_output()``) work with no API key.

    For HTTP providers it delegates to
    :func:`skillspector.providers.create_chat_model`, which uses the
    provider's own native client (e.g. ``ChatAnthropic`` for Anthropic) with
    an ``OPENAI_API_KEY`` / ``ChatOpenAI`` fallback.

    Raises:
        ValueError: when an HTTP provider has no API key configured.
    """
    provider = get_active_provider()
    if has_cli_capability(provider):
        resolved_model = model or provider.resolve_model()
        return AgentCLIChatModel(provider, resolved_model, get_max_output_tokens(resolved_model))

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
    content, _usage = chat_completion_with_usage(prompt, model=model)
    return content


def chat_completion_with_usage(
    prompt: str, *, model: str | None = None
) -> tuple[str, LLMTokenUsage]:
    """Request a chat completion and return its content and token usage."""
    response = get_chat_model(model=model).invoke(prompt)
    if hasattr(response, "text"):
        content = response.text  # type: ignore[union-attr]
    else:
        content = response.content or ""  # type: ignore[union-attr]
    return content, extract_token_usage(response)


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
