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

"""Live OSS provider endpoint tests."""

from __future__ import annotations

import os
import warnings

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

pytestmark = [
    pytest.mark.provider,
    pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning"),
]


class ProviderResult(BaseModel):
    """Tiny schema used to validate provider structured-output wiring."""

    ok: bool = Field(description="Whether the provider request succeeded.")


def _skip_without_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        message = f"{name} is not set; skipping this live provider test"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        pytest.skip(message)
    return value


def _model_from_env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def test_openai_provider_makes_live_structured_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI provider reaches its default endpoint and returns structured output."""
    from skillspector.providers.openai import OpenAIProvider

    _skip_without_env("OPENAI_API_KEY")
    # This live provider check must hit OpenAI's default base URL, not a proxy.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    model = _model_from_env("SKILLSPECTOR_OPENAI_TEST_MODEL", OpenAIProvider.DEFAULT_MODEL)
    llm = OpenAIProvider().create_chat_model(model, max_tokens=32, timeout=60)
    assert llm is not None
    assert llm.openai_api_base is None

    result = llm.with_structured_output(ProviderResult).invoke(
        [HumanMessage(content="Return only the requested structured output with ok=true.")]
    )

    assert result == ProviderResult(ok=True)


def test_anthropic_provider_makes_live_structured_request() -> None:
    """Anthropic provider reaches its default endpoint and returns structured output."""
    from skillspector.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicProvider

    _skip_without_env("ANTHROPIC_API_KEY")

    model = _model_from_env("SKILLSPECTOR_ANTHROPIC_TEST_MODEL", AnthropicProvider.DEFAULT_MODEL)
    llm = AnthropicProvider().create_chat_model(model, max_tokens=32, timeout=60)
    assert llm is not None
    assert str(llm.anthropic_api_url).rstrip("/") == ANTHROPIC_BASE_URL.rstrip("/")

    result = llm.with_structured_output(ProviderResult).invoke(
        [HumanMessage(content="Return only the requested structured output with ok=true.")]
    )

    assert result == ProviderResult(ok=True)


def test_nv_build_provider_makes_live_structured_request() -> None:
    """NVIDIA Build provider reaches its default endpoint and returns structured output."""
    from skillspector.providers.nv_build import BUILD_BASE_URL, NvBuildProvider

    _skip_without_env("NVIDIA_INFERENCE_KEY")

    model = _model_from_env("SKILLSPECTOR_NV_BUILD_TEST_MODEL", NvBuildProvider.DEFAULT_MODEL)
    llm = NvBuildProvider().create_chat_model(model, max_tokens=32, timeout=60)
    assert llm is not None
    assert str(llm.openai_api_base).rstrip("/") == BUILD_BASE_URL.rstrip("/")

    result = llm.with_structured_output(ProviderResult).invoke(
        [HumanMessage(content="Return only the requested structured output with ok=true.")]
    )

    assert result == ProviderResult(ok=True)
