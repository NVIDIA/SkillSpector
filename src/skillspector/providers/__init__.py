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

    openai           → OpenAIProvider          (api.openai.com)
    anthropic        → AnthropicProvider       (api.anthropic.com)
    anthropic_proxy  → AnthropicProxyProvider  (Vertex-style raw-predict proxy)
    bedrock          → BedrockProvider         (AWS Bedrock Runtime, SigV4)
    nv_build         → NvBuildProvider         (build.nvidia.com)

When unset, the selector defaults to ``nv_build``.
"""

from __future__ import annotations

import os
from typing import NoReturn

from langchain_core.language_models.chat_models import BaseChatModel

from .base import ChatModelProvider, CredentialsProvider, LLMProvider, ModelMetadataProvider
from .nv_build import NvBuildProvider

NO_LLM_API_KEY_MESSAGE = (
    "No LLM API key configured. Set the credential env var for the "
    "active provider, or set OPENAI_API_KEY (and optionally "
    "OPENAI_BASE_URL) to use a standard OpenAI-compatible endpoint. "
    "Use --no-llm to skip LLM analysis and run static checks only."
)


def raise_no_llm_api_key_configured() -> NoReturn:
    """Raise the shared no-LLM-credentials error."""
    raise ValueError(NO_LLM_API_KEY_MESSAGE)


def _select_active_provider() -> LLMProvider:
    """Construct the active provider based on ``SKILLSPECTOR_PROVIDER``."""
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
        "Expected one of: openai, anthropic, anthropic_proxy, bedrock, nv_build (or unset)."
    )


def get_metadata_provider() -> ModelMetadataProvider:
    """Return the active provider for token-budget + default-model lookups."""
    return _select_active_provider()


def resolve_provider_credentials() -> tuple[str, str | None] | None:
    """Return ``(api_key, base_url)`` from the active provider.

    Returns ``None`` when the provider's credential env var is unset, so
    callers can fall through to other credential sources.
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

    return _openai_fallback_provider().resolve_credentials()


def create_chat_model(
    model: str,
    *,
    max_tokens: int,
    timeout: float | None = 120,
) -> BaseChatModel:
    """Create the active provider's native LangChain chat model.

    If the active provider is not configured, fall back to standard OpenAI
    environment variables. This preserves the historical ``OPENAI_API_KEY``
    escape hatch while letting configured providers choose their own client.
    """
    provider = _select_active_provider()
    llm = provider.create_chat_model(model, max_tokens=max_tokens, timeout=timeout)
    if llm is not None:
        return llm

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
    "ChatModelProvider",
    "CredentialsProvider",
    "LLMProvider",
    "ModelMetadataProvider",
    "NO_LLM_API_KEY_MESSAGE",
    "create_chat_model",
    "get_metadata_provider",
    "raise_no_llm_api_key_configured",
    "resolve_chat_model_credentials",
    "resolve_provider_credentials",
]
