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

"""Tests for the layered model-info resolution in model_info.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from skillspector.constants import DEFAULT_CONTEXT_LENGTH, MAX_INPUT_TOKENS_PCT
from skillspector.providers import reset_provider, use_provider

MODULE = "skillspector.model_info"
NV_PROVIDER_MODULE = "skillspector.providers.nv_inference.provider"

try:
    import skillspector.providers.nv_inference.provider  # noqa: F401

    _NV_PROVIDER_AVAILABLE = True
except ImportError:
    _NV_PROVIDER_AVAILABLE = False

nv_provider_required = pytest.mark.skipif(
    not _NV_PROVIDER_AVAILABLE,
    reason="optional NVIDIA metadata provider not present (public-OSS build)",
)


def _clear_caches() -> None:
    """Clear the provider registry cache used by model-info lookups."""
    from skillspector.providers import registry

    registry._load_registry.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_caches(mock_resolve_context_length):
    """Clear caches before and after every test.

    Depends on the conftest autouse ``mock_resolve_context_length`` fixture
    so it is active for the rest of the suite.  Tests in this module bypass
    the mock by reloading the real module.
    """
    _clear_caches()
    yield
    _clear_caches()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_registry(path: Path, models: dict) -> None:
    path.write_text(yaml.dump({"models": models}), encoding="utf-8")


def _get_real_functions():
    """Import the real (unpatched) module-level functions."""
    import importlib

    import skillspector.model_info as mod
    from skillspector.providers import registry

    importlib.reload(mod)
    registry._load_registry.cache_clear()
    return mod


# ---------------------------------------------------------------------------
# Layer 1 — NVIDIA metadata API
# ---------------------------------------------------------------------------


@nv_provider_required
class TestLayer1NvidiaApi:
    """Layer 1: NVIDIA catalog API resolution."""

    def test_layer1_success(self) -> None:
        """When NVIDIA_INFERENCE_METADATA_KEY is set and API succeeds, use that value."""
        mod = _get_real_functions()

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"context_length_tokens": 500_000}]
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"NVIDIA_INFERENCE_METADATA_KEY": "test-key"}, clear=False),
            patch(f"{NV_PROVIDER_MODULE}.requests.get", return_value=mock_resp),
        ):
            result = mod._resolve_context_length("some/model")
            assert result == 500_000

    def test_layer1_failure_falls_to_layer2(self, tmp_path: Path) -> None:
        """When API fails, fall through to Layer 2 registry."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "some/model": {"context_length": 256_000},
            },
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "NVIDIA_INFERENCE_METADATA_KEY": "test-key",
                    "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
                },
                clear=False,
            ),
            patch(f"{NV_PROVIDER_MODULE}.requests.get", side_effect=Exception("network down")),
        ):
            result = mod._resolve_context_length("some/model")
            assert result == 256_000

    def test_layer1_skipped_when_key_absent(self, tmp_path: Path) -> None:
        """When NVIDIA_INFERENCE_METADATA_KEY is not set, skip straight to Layer 2."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "some/model": {"context_length": 300_000},
            },
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "NVIDIA_INFERENCE_METADATA_KEY": "",
                    "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
                },
                clear=False,
            ),
            patch(f"{NV_PROVIDER_MODULE}.requests.get") as mock_get,
        ):
            result = mod._resolve_context_length("some/model")
            assert result == 300_000
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2 — YAML registry (via SKILLSPECTOR_MODEL_REGISTRY env var)
# ---------------------------------------------------------------------------


class TestLayer2Registry:
    """Layer 2: YAML registry resolution via env var."""

    def test_registry_lookup(self, tmp_path: Path) -> None:
        """Model found in registry file pointed to by env var."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "my-provider/my-model": {"context_length": 1_000_000},
            },
        )

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
            },
            clear=False,
        ):
            result = mod._resolve_context_length("my-provider/my-model")
            assert result == 1_000_000

    def test_registry_adds_model(self, tmp_path: Path) -> None:
        """Registry can provide limits for any model label."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "my-org/custom-model": {"context_length": 64_000},
            },
        )

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
            },
            clear=False,
        ):
            result = mod._resolve_context_length("my-org/custom-model")
            assert result == 64_000

    def test_no_registry_env_var_returns_empty(self) -> None:
        """When SKILLSPECTOR_MODEL_REGISTRY is unset, registry is empty."""
        mod = _get_real_functions()

        with patch.dict(
            "os.environ",
            {"NVIDIA_INFERENCE_METADATA_KEY": "", "SKILLSPECTOR_MODEL_REGISTRY": ""},
            clear=False,
        ):
            result = mod._resolve_context_length("any/model")
            assert result == DEFAULT_CONTEXT_LENGTH

    def test_bad_registry_path_returns_empty(self) -> None:
        """When registry path doesn't exist, falls through to default."""
        mod = _get_real_functions()

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": "/nonexistent/path.yaml",
            },
            clear=False,
        ):
            result = mod._resolve_context_length("any/model")
            assert result == DEFAULT_CONTEXT_LENGTH


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


class TestFallback:
    """Fallback when neither layer resolves."""

    def test_unknown_model_returns_default(self, tmp_path: Path) -> None:
        """Unknown model falls back to DEFAULT_CONTEXT_LENGTH with a warning."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "known/model": {"context_length": 200_000},
            },
        )

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
            },
            clear=False,
        ):
            result = mod._resolve_context_length("nonexistent/model-xyz")
            assert result == DEFAULT_CONTEXT_LENGTH


# ---------------------------------------------------------------------------
# Public API — get_max_input_tokens / get_max_output_tokens
# ---------------------------------------------------------------------------


class TestPublicApi:
    """get_max_input_tokens and get_max_output_tokens."""

    def test_max_input_tokens(self, tmp_path: Path) -> None:
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "test/model": {"context_length": 1_000_000},
            },
        )

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
            },
            clear=False,
        ):
            result = mod.get_max_input_tokens("test/model")
            assert result == int(1_000_000 * MAX_INPUT_TOKENS_PCT)

    def test_max_output_tokens_with_explicit_cap(self, tmp_path: Path) -> None:
        """Registry entry with max_output_tokens caps the percentage budget."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "test/model": {"context_length": 128_000, "max_output_tokens": 16_384},
            },
        )

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
            },
            clear=False,
        ):
            result = mod.get_max_output_tokens("test/model")
            pct_budget = int(128_000 * (1 - MAX_INPUT_TOKENS_PCT))
            assert result == min(pct_budget, 16_384)
            assert result == 16_384

    def test_max_output_tokens_without_explicit_cap(self, tmp_path: Path) -> None:
        """When no max_output_tokens in registry, use percentage-based budget."""
        mod = _get_real_functions()

        registry_file = tmp_path / "registry.yaml"
        _write_registry(
            registry_file,
            {
                "bare/model": {"context_length": 200_000},
            },
        )

        with patch.dict(
            "os.environ",
            {
                "NVIDIA_INFERENCE_METADATA_KEY": "",
                "SKILLSPECTOR_MODEL_REGISTRY": str(registry_file),
            },
            clear=False,
        ):
            result = mod.get_max_output_tokens("bare/model")
            expected = int(200_000 * (1 - MAX_INPUT_TOKENS_PCT))
            assert result == expected

    def test_token_limits_follow_current_bound_provider_for_same_model_label(self) -> None:
        """Same labels must resolve against the provider bound in this context."""

        class _BoundProvider:
            DEFAULT_MODEL = "shared/model"
            SLOT_DEFAULTS = {"meta_analyzer": "shared/model"}

            def __init__(self, context_length: int, max_output_tokens: int) -> None:
                self._context_length = context_length
                self._max_output_tokens = max_output_tokens

            def get_context_length(self, model: str) -> int | None:
                return self._context_length if model == "shared/model" else None

            def get_max_output_tokens(self, model: str) -> int | None:
                return self._max_output_tokens if model == "shared/model" else None

            def resolve_model(self, slot: str = "default") -> str:
                return "shared/model"

            def resolve_credentials(self) -> tuple[str, str | None] | None:
                return None

        mod = _get_real_functions()
        first = _BoundProvider(context_length=100, max_output_tokens=10)
        second = _BoundProvider(context_length=200, max_output_tokens=20)

        first_token = use_provider(first)
        try:
            assert mod.get_max_input_tokens("shared/model") == 75
            assert mod.get_max_output_tokens("shared/model") == 10
        finally:
            reset_provider(first_token)

        second_token = use_provider(second)
        try:
            assert mod.get_max_input_tokens("shared/model") == 150
            assert mod.get_max_output_tokens("shared/model") == 20
        finally:
            reset_provider(second_token)
