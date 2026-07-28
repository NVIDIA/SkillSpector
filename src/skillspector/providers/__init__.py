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

"""Pluggable LLM provider package.

The active provider supplies credentials, token-budget metadata, per-slot
default model labels, and a native LangChain chat model. Each provider
is its own subpackage with a ``provider.py`` and a bundled
``model_registry.yaml``.

Selection happens via the ``SKILLSPECTOR_PROVIDER`` env var:

    openai          → OpenAIProvider          (api.openai.com)
    anthropic       → AnthropicProvider       (api.anthropic.com)
    anthropic_proxy → AnthropicProxyProvider  (Vertex-style raw-predict proxy)
    bedrock         → BedrockProvider         (AWS Bedrock Runtime, SigV4)
    nv_build        → NvBuildProvider          (build.nvidia.com)
    claude_cli      → ClaudeCLIProvider       (local ``claude`` binary, no API key)
    codex_cli       → CodexCLIProvider        (local ``codex`` binary, no API key)
    gemini_cli      → GeminiCLIProvider       (local ``gemini`` binary, no API key)
    antigravity_cli → AntigravityCLIProvider  (local ``agy`` binary; registered
                                               but disabled — agy is TTY-only and
                                               can't be captured; use gemini_cli)

When unset, the selector defaults to ``nv_build``.

CLI providers (``claude_cli``, ``codex_cli``, ``gemini_cli``) implement the
optional :class:`~skillspector.providers.base.AgentCLICapable` interface — they
expose ``is_available()`` and ``complete()`` so that
:func:`skillspector.llm_utils.get_chat_model` uses the local CLI subprocess
instead of the ``ChatOpenAI`` HTTP transport.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import NoReturn

from langchain_core.language_models.chat_models import BaseChatModel

from .base import (
    AgentCLICapable,
    ChatModelProvider,
    CredentialsProvider,
    LLMProvider,
    ModelMetadataProvider,
    has_cli_capability,
)
from .nv_build import NvBuildProvider

NO_LLM_API_KEY_MESSAGE = (
    "No LLM API key configured. Set the credential env var for the "
    "active provider, or set OPENAI_API_KEY (and optionally "
    "OPENAI_BASE_URL) to use a standard OpenAI-compatible endpoint. "
    "Use --no-llm to skip LLM analysis and run static checks only."
)

_INJECTED_PROVIDER: ContextVar[LLMProvider | None] = ContextVar(
    "skillspector_injected_provider",
    default=None,
)


def raise_no_llm_api_key_configured() -> NoReturn:
    """Raise the shared no-LLM-credentials error."""
    raise ValueError(NO_LLM_API_KEY_MESSAGE)


def use_provider(provider: LLMProvider) -> Token[LLMProvider | None]:
    """Bind *provider* for the current context."""
    return _INJECTED_PROVIDER.set(provider)


def reset_provider(token: Token[LLMProvider | None]) -> None:
    """Restore the provider binding represented by *token*."""
    _INJECTED_PROVIDER.reset(token)


def has_provider_binding() -> bool:
    """Return whether the current context has an injected provider."""
    return _INJECTED_PROVIDER.get() is not None


def _select_active_provider() -> LLMProvider:
    """Construct the active provider based on ``SKILLSPECTOR_PROVIDER``."""
    injected_provider = _INJECTED_PROVIDER.get()
    if injected_provider is not None:
        return injected_provider

    name = os.environ.get("SKILLSPECTOR_PROVIDER", "").strip().lower()

    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider()
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "anthropic_proxy":
        from .anthropic_proxy import AnthropicProxyProvider

        return AnthropicProxyProvider()
    if name == "bedrock":
        from .bedrock import BedrockProvider

        return BedrockProvider()
    if name == "nv_build":
        return NvBuildProvider()
    if name == "claude_cli":
        from .claude_cli import ClaudeCLIProvider

        return ClaudeCLIProvider()
    if name == "codex_cli":
        from .codex_cli import CodexCLIProvider

        return CodexCLIProvider()
    if name == "gemini_cli":
        from .gemini_cli import GeminiCLIProvider

        return GeminiCLIProvider()
    if name == "antigravity_cli":
        from .antigravity_cli import AntigravityCLIProvider

        return AntigravityCLIProvider()
    if name in ("nv_inference", ""):
        # Try the optional nv_inference subpackage if it's bundled with
        # this installation; otherwise fall through to nv_build.
        try:
            from .nv_inference import NvInferenceProvider

            return NvInferenceProvider()
        except ImportError:
            return NvBuildProvider()

    raise ValueError(
        f"Unknown SKILLSPECTOR_PROVIDER: {name!r}. "
        "Expected one of: openai, anthropic, anthropic_proxy, bedrock, nv_build, "
        "claude_cli, codex_cli, gemini_cli, antigravity_cli (or unset)."
    )


def get_metadata_provider() -> ModelMetadataProvider:
    """Return the active provider for token-budget + default-model lookups."""
    return _select_active_provider()


def get_active_provider() -> ModelMetadataProvider:
    """Return the active provider (alias for :func:`get_metadata_provider`).

    Preferred over :func:`get_metadata_provider` when callers also need to
    check for optional capabilities (e.g. :func:`has_cli_capability`).
    """
    return _select_active_provider()


def resolve_provider_credentials() -> tuple[str, str | None] | None:
    """Return ``(api_key, base_url)`` from the active provider.

    Returns ``None`` when the provider's credential env var is unset, so
    callers can fall through to other credential sources.  CLI providers
    always return ``None`` from this method; availability is checked via
    ``is_available()`` instead.
    """
    return _select_active_provider().resolve_credentials()


def _openai_fallback_provider() -> LLMProvider:
    """Return the standard OpenAI fallback provider."""
    from .openai import OpenAIProvider

    return OpenAIProvider()


def resolve_chat_model_credentials() -> tuple[str, str | None] | None:
    """Return credentials used for chat model construction, including fallback."""
    creds = resolve_provider_credentials()
    if creds is not None:
        return creds

    if has_provider_binding():
        return None

    return _openai_fallback_provider().resolve_credentials()


def create_chat_model(
    model: str,
    *,
    max_tokens: int,
    timeout: float | None = 120,
) -> BaseChatModel:
    """Create the active provider's native LangChain chat model.

    CLI providers (``claude_cli``, ``codex_cli``, ``gemini_cli``) do not have
    a native LangChain chat model — callers that need CLI transport should use
    :func:`skillspector.llm_utils.get_chat_model` instead (which returns an
    :class:`~skillspector.llm_utils.AgentCLIChatModel` adapter).

    If the active provider is not configured, fall back to standard OpenAI
    environment variables. This preserves the historical ``OPENAI_API_KEY``
    escape hatch while letting configured providers choose their own client.
    """
    provider = _select_active_provider()

    # CLI providers don't participate in the create_chat_model path.
    if not has_cli_capability(provider):
        llm = provider.create_chat_model(model, max_tokens=max_tokens, timeout=timeout)
        if llm is not None:
            return llm

        if has_provider_binding():
            raise_no_llm_api_key_configured()

        from .openai import OpenAIProvider

        if not isinstance(provider, OpenAIProvider):
            llm = _openai_fallback_provider().create_chat_model(
                model,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if llm is not None:
                return llm

    raise_no_llm_api_key_configured()


__all__ = [
    "AgentCLICapable",
    "ChatModelProvider",
    "CredentialsProvider",
    "LLMProvider",
    "ModelMetadataProvider",
    "NO_LLM_API_KEY_MESSAGE",
    "create_chat_model",
    "get_active_provider",
    "get_metadata_provider",
    "has_cli_capability",
    "has_provider_binding",
    "reset_provider",
    "raise_no_llm_api_key_configured",
    "resolve_chat_model_credentials",
    "resolve_provider_credentials",
    "use_provider",
]
