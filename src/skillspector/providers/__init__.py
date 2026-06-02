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

The active provider supplies credentials, an OpenAI-compatible base URL,
token-budget metadata, and per-slot default model labels.  Each provider
is its own subpackage with a ``provider.py`` and a bundled
``model_registry.yaml``.

Selection happens via the ``SKILLSPECTOR_PROVIDER`` env var:

    openai        → OpenAIProvider          (api.openai.com)
    anthropic     → AnthropicProvider       (api.anthropic.com)
    nv_build      → NvBuildProvider         (build.nvidia.com)

When unset, the selector defaults to ``nv_build``.
"""

from __future__ import annotations

import os

from .base import CredentialsProvider, ModelMetadataProvider
from .nv_build import NvBuildProvider


def _select_active_provider() -> ModelMetadataProvider:
    """Construct the active provider based on ``SKILLSPECTOR_PROVIDER``."""
    name = os.environ.get("SKILLSPECTOR_PROVIDER", "").strip().lower()

    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider()
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "vertexai":
        from .vertexai import VertexAIProvider

        return VertexAIProvider()
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
        "Expected one of: openai, anthropic, vertexai, nv_build (or unset)."
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


__all__ = [
    "CredentialsProvider",
    "ModelMetadataProvider",
    "get_metadata_provider",
    "resolve_provider_credentials",
]
