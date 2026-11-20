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

"""Malformed registry entries do not declare model controls or block clients."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from langchain_openai import ChatOpenAI

from skillspector.model_info import get_max_output_tokens
from skillspector.providers import registry, reset_provider, use_provider
from skillspector.providers.openai import OpenAIProvider


@pytest.fixture(autouse=True)
def mock_resolve_context_length(monkeypatch: pytest.MonkeyPatch):
    """Keep registry resolution real with the OpenAI provider, which does no metadata IO."""
    for name in (
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_TEMPERATURE",
        "SKILLSPECTOR_SEED",
        "SKILLSPECTOR_REASONING_EFFORT",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_PROJECT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    token = use_provider(OpenAIProvider())
    registry._load.cache_clear()
    try:
        yield
    finally:
        reset_provider(token)
        registry._load.cache_clear()


@pytest.mark.parametrize("model", ["gpt-5.4", "openai/gpt-5.4"])
@pytest.mark.parametrize(
    "entry",
    [128_000, "auto", True, ["auto"], None],
    ids=["integer", "string", "boolean", "list", "null"],
)
def test_non_mapping_entries_do_not_declare_settings(
    model: str, entry: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "registry.yaml"
    override.write_text(yaml.safe_dump({"models": {model: entry}}), encoding="utf-8")
    monkeypatch.setenv("SKILLSPECTOR_MODEL_REGISTRY", str(override))

    for key in ("tool_choice", "sampling", "structured_output"):
        assert registry.lookup_setting("unused-default.yaml", model, key) is None
    assert registry.lookup_structured_output_method("unused-default.yaml", model) is None


@pytest.mark.parametrize("model", ["gpt-5.4", "openai/gpt-5.4"])
def test_mapping_entries_keep_declared_settings(
    model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "registry.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "models": {
                    model: {
                        "tool_choice": "auto",
                        "sampling": "rejected",
                        "structured_output": "json_schema",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILLSPECTOR_MODEL_REGISTRY", str(override))

    assert registry.lookup_setting("unused-default.yaml", model, "tool_choice") == "auto"
    assert registry.lookup_setting("unused-default.yaml", model, "sampling") == "rejected"
    assert registry.lookup_structured_output_method("unused-default.yaml", model) == "json_schema"


@pytest.mark.parametrize("model", ["gpt-5.4", "openai/gpt-5.4"])
@pytest.mark.parametrize("entry", [128_000, [128_000], None], ids=["integer", "list", "null"])
def test_non_mapping_entries_fall_back_to_default_budget_and_construct_openai_client(
    model: str,
    entry: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    override = tmp_path / "registry.yaml"
    override.write_text(yaml.safe_dump({"models": {model: entry}}), encoding="utf-8")
    monkeypatch.setenv("SKILLSPECTOR_MODEL_REGISTRY", str(override))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider()

    budget = get_max_output_tokens(model)
    client = provider.create_chat_model(model, max_tokens=budget)

    assert budget == 32_000
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == model
    assert client.max_tokens == 32_000
    assert client.disabled_params is None
    assert any("using 128000-token default" in record.getMessage() for record in caplog.records)
