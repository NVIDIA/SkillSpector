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

"""Tests for per-slot model env overrides and model validation."""

from __future__ import annotations

import importlib
import logging

import pytest

from skillspector.providers import registry


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate provider and model env vars for each test."""
    for key in (
        "SKILLSPECTOR_PROVIDER",
        "SKILLSPECTOR_MODEL",
        "SKILLSPECTOR_MODEL_REGISTRY",
        "SKILLSPECTOR_STRICT_MODEL_VALIDATION",
        "NVIDIA_INFERENCE_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "SKILLSPECTOR_REASONING_EFFORT",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    # Clear per-slot env vars that tests may set.
    for slot in (
        "DEFAULT",
        "META_ANALYZER",
        "SEMANTIC_DEVELOPER_INTENT",
        "MCP_LEAST_PRIVILEGE",
    ):
        monkeypatch.delenv(f"SKILLSPECTOR_MODEL_{slot}", raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


def _reload_constants():
    """Re-import constants to re-run module-level config resolution."""
    import skillspector.constants as mod

    return importlib.reload(mod)


class TestPerSlotModelOverrides:
    """SKILLSPECTOR_MODEL_{SLOT} env vars override per-slot model selection."""

    def test_slot_env_overrides_provider_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL_META_ANALYZER", "gpt-4o-mini")
        mod = _reload_constants()
        assert mod.MODEL_CONFIG["meta_analyzer"] == "gpt-4o-mini"

    def test_slot_env_overrides_global_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "global-model")
        monkeypatch.setenv("SKILLSPECTOR_MODEL_META_ANALYZER", "slot-specific")
        mod = _reload_constants()
        assert mod.MODEL_CONFIG["meta_analyzer"] == "slot-specific"
        # Other slots should still use the global model.
        assert mod.MODEL_CONFIG["default"] == "global-model"

    def test_unset_slot_env_falls_through_to_provider(self) -> None:
        mod = _reload_constants()
        # Without any slot override, the provider's resolve_model() runs.
        assert mod.MODEL_CONFIG["default"] == mod._SKILLSPECTOR_DEFAULT_MODEL

    def test_multiple_slots_independently_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL_META_ANALYZER", "model-a")
        monkeypatch.setenv("SKILLSPECTOR_MODEL_SEMANTIC_DEVELOPER_INTENT", "model-b")
        mod = _reload_constants()
        assert mod.MODEL_CONFIG["meta_analyzer"] == "model-a"
        assert mod.MODEL_CONFIG["semantic_developer_intent"] == "model-b"
        # Unset slots use provider default.
        assert mod.MODEL_CONFIG["mcp_rug_pull"] == mod._SKILLSPECTOR_DEFAULT_MODEL

    def test_whitespace_only_slot_env_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL_META_ANALYZER", "   ")
        mod = _reload_constants()
        # Whitespace-only treated as unset — falls through to provider.
        assert mod.MODEL_CONFIG["meta_analyzer"] != "   "


class TestModelValidation:
    """_validate_model_config warns or raises on unknown model IDs."""

    def test_unknown_model_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "totally-unknown-model-xyz")
        with caplog.at_level(logging.WARNING, logger="skillspector.constants"):
            _reload_constants()
        assert any("totally-unknown-model-xyz" in r.message for r in caplog.records)
        assert any("not found in model_registry.yaml" in r.message for r in caplog.records)

    def test_known_model_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # nv_build's default model is in its registry — no warnings expected
        # for that model.
        with caplog.at_level(logging.WARNING, logger="skillspector.constants"):
            mod = _reload_constants()
        default_model = mod._SKILLSPECTOR_DEFAULT_MODEL
        warnings_for_default = [r for r in caplog.records if default_model in r.message]
        assert len(warnings_for_default) == 0

    def test_strict_validation_raises_on_unknown_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "nonexistent-model")
        monkeypatch.setenv("SKILLSPECTOR_STRICT_MODEL_VALIDATION", "true")
        with pytest.raises(ValueError, match="Strict model validation enabled"):
            _reload_constants()

    def test_strict_validation_passes_with_known_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_STRICT_MODEL_VALIDATION", "true")
        # Default provider model is in the registry — should not raise.
        mod = _reload_constants()
        assert mod.MODEL_CONFIG["default"] == mod._SKILLSPECTOR_DEFAULT_MODEL

    def test_strict_validation_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "nonexistent-model")
        # No SKILLSPECTOR_STRICT_MODEL_VALIDATION set — should warn, not raise.
        mod = _reload_constants()
        assert mod.MODEL_CONFIG["default"] == "nonexistent-model"

    def test_strict_validation_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "nonexistent-model")
        monkeypatch.setenv("SKILLSPECTOR_STRICT_MODEL_VALIDATION", "True")
        with pytest.raises(ValueError, match="Strict model validation enabled"):
            _reload_constants()
