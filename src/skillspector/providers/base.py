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

"""Protocols for pluggable LLM providers.

Two optional capability protocols are also defined here for providers that
bypass the HTTP API entirely (e.g. CLI-based providers):

- :class:`AgentCLICapable` — providers that implement ``is_available()``
  and ``complete()`` use these instead of the ``ChatOpenAI`` path.

Callers use :func:`has_cli_capability` to detect these providers at
runtime without requiring a formal ``isinstance`` check against the
protocols (which Python structural subtyping does not enforce at runtime
without an explicit ``runtime_checkable`` decorator).
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from langchain_core.language_models.chat_models import BaseChatModel


class ModelMetadataProvider(Protocol):
    """Provider-side knowledge about models — token budgets and defaults.

    ``get_context_length`` / ``get_max_output_tokens`` return ``None`` to
    signal "I don't know" so callers fall back to defaults.

    ``resolve_model`` runs the per-provider waterfall:
    ``SKILLSPECTOR_MODEL`` env var → provider's slot-specific default →
    provider's general default.  Always returns a non-empty string.

    ``DEFAULT_MODEL`` is the provider's general default model label.
    ``SLOT_DEFAULTS`` maps specific slots to their preferred models.
    """

    DEFAULT_MODEL: ClassVar[str]
    SLOT_DEFAULTS: ClassVar[dict[str, str]]

    def get_context_length(self, model: str) -> int | None: ...

    def get_max_output_tokens(self, model: str) -> int | None: ...

    def resolve_model(self, slot: str = "default") -> str: ...


class CredentialsProvider(Protocol):
    """Anything that can supply ``(api_key, base_url)`` for the LLM client.

    Implementations return ``None`` when the relevant environment is not
    configured, so the caller can fall back to other credential sources.
    ``base_url`` may be ``None`` to defer to the LLM client's own default.
    """

    def resolve_credentials(self) -> tuple[str, str | None] | None: ...


class ChatModelProvider(Protocol):
    """Anything that can construct its native LangChain chat model."""

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None: ...


class AgentCLICapable(Protocol):
    """Optional extension for providers that drive a local agent CLI.

    Providers that implement these two methods opt in to the CLI transport
    path in :func:`skillspector.llm_utils.chat_completion`.  Existing
    HTTP-based providers are not required to implement them.

    ``is_available()``
        Return ``(True, None)`` when the underlying binary is on PATH and
        the CLI appears to be authenticated.  Return ``(False, reason)``
        otherwise.  This replaces the credential-based availability check
        in :func:`skillspector.llm_utils.is_llm_available` for CLI providers.

    ``complete(prompt, *, model, max_output_tokens)``
        Execute the CLI, pass the prompt via stdin, and return the
        assistant's text response.  Raises on any failure (fail-closed).
    """

    def is_available(self) -> tuple[bool, str | None]: ...

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_output_tokens: int,
    ) -> str: ...


class LLMProvider(ModelMetadataProvider, CredentialsProvider, ChatModelProvider, Protocol):
    """Complete provider surface used by SkillSpector's LLM stack."""


def has_cli_capability(provider: object) -> bool:
    """Return ``True`` when *provider* implements the :class:`AgentCLICapable` interface.

    Uses duck-typing rather than ``isinstance`` so that providers added
    externally (outside this package) also qualify.
    """
    return callable(getattr(provider, "is_available", None)) and callable(
        getattr(provider, "complete", None)
    )
