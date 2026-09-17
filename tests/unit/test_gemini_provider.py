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

"""Unit tests for the Gemini on Google Cloud provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import google.auth
import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from langchain_openai import ChatOpenAI
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.constants import build_model_config
from skillspector.inference_usage import provider_name
from skillspector.providers import (
    create_chat_model,
    get_metadata_provider,
    get_model_config_provider,
    registry,
)
from skillspector.providers.gemini import GeminiProvider
from skillspector.providers.gemini.provider import (
    _reset_cached_credentials,
    get_base_url,
    validate_location,
    validate_project_id,
)


class MockCredentials:
    """Mock Google Auth credentials with configurable token, validity, and refresh counter."""

    def __init__(
        self,
        token: str | None = "mock-token",
        valid: bool = False,
        raise_on_refresh: Exception | None = None,
        quota_project_id: str | None = None,
    ) -> None:
        self.token = token
        self._valid = valid
        self.raise_on_refresh = raise_on_refresh
        self.quota_project_id = quota_project_id
        self.refresh_count = 0

    @property
    def valid(self) -> bool:
        return self._valid

    def refresh(self, request: object) -> None:
        self.refresh_count += 1
        if self.raise_on_refresh:
            raise self.raise_on_refresh
        self._valid = True


@pytest.fixture(autouse=True)
def _clean_gemini_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure clean provider environment and reset cached ADC credentials."""
    for key in (
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SKILLSPECTOR_PROVIDER",
        "SKILLSPECTOR_MODEL",
        "OPENAI_API_KEY",
        "SKILLSPECTOR_SEED",
        "SKILLSPECTOR_TEMPERATURE",
        "SKILLSPECTOR_REASONING_EFFORT",
    ):
        monkeypatch.delenv(key, raising=False)
    _reset_cached_credentials()
    registry._load.cache_clear()
    yield
    _reset_cached_credentials()
    registry._load.cache_clear()


class TestGeminiProvider:
    """Unit test suite covering Gemini on Google Cloud provider specifications."""

    def test_missing_project_fails_closed_without_openai_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """1. Explicit SKILLSPECTOR_PROVIDER=gemini fails closed when project is missing, even with OPENAI_API_KEY."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "gemini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-openai-key")

        provider = GeminiProvider()
        with pytest.raises(
            ValueError, match="GOOGLE_CLOUD_PROJECT environment variable is required"
        ):
            provider.resolve_credentials()

        with pytest.raises(
            ValueError, match="GOOGLE_CLOUD_PROJECT environment variable is required"
        ):
            create_chat_model("gemini-3.5-flash", max_tokens=100)

    def test_default_location_and_adc_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """2. With project only, location defaults to global and ADC is refreshed."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token="refreshed-token", valid=False)
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        resolved = provider.resolve_credentials()

        assert resolved is not None
        token, base_url = resolved
        assert token == "refreshed-token"
        assert base_url == (
            "https://aiplatform.googleapis.com/v1/projects/test-project-123/locations/global/endpoints/openapi"
        )
        assert creds.refresh_count == 1

    def test_regional_location_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """3. With regional location, base URL uses {location}-aiplatform.googleapis.com."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        creds = MockCredentials(token="token", valid=True)
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        resolved = provider.resolve_credentials()

        assert resolved is not None
        _, base_url = resolved
        assert base_url == (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project-123/locations/us-central1/endpoints/openapi"
        )

    def test_missing_token_after_refresh_raises_sanitized_refresh_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4. A refresh that yields no token never leaks credential material."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token=None, valid=False)
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        with pytest.raises(RefreshError) as exc_info:
            provider.resolve_credentials()
        assert str(exc_info.value) == "Google Cloud credential refresh failed."

    @pytest.mark.parametrize(
        "invalid_loc",
        [
            "us/central1",
            "us:central1",
            "us.central1",
            "us@central1",
            "us central1",
            "US-CENTRAL1",
            "us-central1-",
            "-us-central1",
            "u",
            "a" * 35,
        ],
    )
    def test_invalid_locations_fail(self, invalid_loc: str) -> None:
        """5. Invalid locations fail before hostname construction."""
        with pytest.raises(ValueError, match="Invalid GOOGLE_CLOUD_LOCATION"):
            validate_location(invalid_loc)

    def test_us_and_eu_multi_regions(self) -> None:
        """6. us and eu locations use rep endpoints."""
        assert get_base_url("my-project", "us") == (
            "https://aiplatform.us.rep.googleapis.com/v1/projects/my-project/locations/us/endpoints/openapi"
        )
        assert get_base_url("my-project", "eu") == (
            "https://aiplatform.eu.rep.googleapis.com/v1/projects/my-project/locations/eu/endpoints/openapi"
        )

    @pytest.mark.parametrize(
        "invalid_proj",
        [
            "short",  # <6 chars
            "a" * 31,  # >30 chars
            "Project123",  # uppercase
            "1project",  # starts with digit
            "-project",  # starts with hyphen
            "project-",  # ends with hyphen
            "proj/123",  # slash
            "proj?123",  # question mark
            "proj#123",  # hash
            "proj 123",  # whitespace
            "proj-google-1",  # restricted string 'google'
            "my-ssl-proj",  # restricted string 'ssl'
            "undefined",  # exact restricted string 'undefined'
            "project-none",  # restricted suffix '-none'
        ],
    )
    def test_invalid_project_ids_fail(self, invalid_proj: str) -> None:
        """7. Invalid project IDs fail before URL construction."""
        with pytest.raises(ValueError, match="Invalid GOOGLE_CLOUD_PROJECT"):
            validate_project_id(invalid_proj)

    def test_valid_project_ids_containing_null(self) -> None:
        """Legitimate project IDs containing 'null' are accepted."""
        validate_project_id("null-data-123")
        validate_project_id("billing-null-prod")

    def test_default_credentials_error_and_refresh_error_are_classified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """8. ADC setup remains config-facing; refresh errors are sanitized."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")

        def mock_default_err(*, scopes: list[str], request: object):
            raise DefaultCredentialsError("ADC not found")

        monkeypatch.setattr(google.auth, "default", mock_default_err)
        with pytest.raises(
            ValueError, match="Failed to obtain Google Cloud Application Default Credentials"
        ):
            GeminiProvider().resolve_credentials()

        _reset_cached_credentials()
        creds = MockCredentials(
            token=None,
            valid=False,
            raise_on_refresh=RefreshError("Token refresh failed: sentinel-secret"),
        )
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))
        with pytest.raises(RefreshError) as exc_info:
            GeminiProvider().resolve_credentials()
        assert str(exc_info.value) == "Google Cloud credential refresh failed."
        assert "sentinel-secret" not in str(exc_info.value)

    def test_bounded_request_caps_none_timeout_to_remaining_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller's None timeout still respects an active auth deadline."""
        import skillspector.providers.gemini.provider as gemini_provider

        request = MagicMock()

        monkeypatch.setattr(gemini_provider, "Request", lambda: request)
        monkeypatch.setattr(gemini_provider, "monotonic", lambda: 10.0)

        gemini_provider._bounded_request(15.0)("https://example.invalid", timeout=None)

        assert request.call_args.kwargs["timeout"] == 5.0

    def test_credentials_skip_adc_when_lock_acquire_exhausts_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A just-in-time lock acquisition cannot start ADC work after expiry."""
        import skillspector.providers.gemini.provider as gemini_provider

        state = {"now": 0.0, "default_called": False}
        lock = MagicMock()

        def acquire(timeout: float) -> bool:
            assert timeout == 10.0
            state["now"] = 10.0
            return True

        lock.acquire.side_effect = acquire
        monkeypatch.setattr(gemini_provider, "_AUTH_LOCK", lock)
        monkeypatch.setattr(gemini_provider, "monotonic", lambda: state["now"])
        monkeypatch.setattr(
            google.auth,
            "default",
            lambda **_kwargs: state.__setitem__("default_called", True),
        )

        with pytest.raises(TimeoutError, match="Gemini authentication timed out"):
            gemini_provider._get_credentials(10.0)

        assert state == {"now": 10.0, "default_called": False}
        lock.release.assert_called_once()

    def test_create_chat_model_spends_one_budget_across_auth_and_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auth lock, ADC lookup, refresh request, and model share one timeout."""
        import skillspector.providers.gemini.provider as gemini_provider

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        observed: dict[str, object] = {}

        class _Lock:
            def acquire(self, timeout: float | None = None) -> bool:
                observed["lock_timeout"] = timeout
                return True

            def release(self) -> None:
                return None

            def __enter__(self) -> _Lock:
                self.acquire()
                return self

            def __exit__(self, *_args: object) -> None:
                self.release()

        class _Request:
            def __call__(self, *_args: object, timeout: float = 120, **_kwargs: object) -> object:
                observed["request_timeout"] = timeout
                return object()

        class _Credentials(MockCredentials):
            def refresh(self, request: object) -> None:
                request("https://token.invalid", timeout=120)  # type: ignore[operator]
                super().refresh(request)

        creds = _Credentials(token="token", valid=False)

        def _default(*, scopes: list[str], request: object) -> tuple[MockCredentials, str]:
            observed["default_request"] = request
            return creds, "test-project-123"

        model_factory = MagicMock(return_value=object())
        clock = iter(float(i) for i in range(20))
        monkeypatch.setattr(gemini_provider, "_AUTH_LOCK", _Lock())
        monkeypatch.setattr(gemini_provider, "Request", _Request)
        monkeypatch.setattr(gemini_provider, "monotonic", lambda: next(clock), raising=False)
        monkeypatch.setattr(google.auth, "default", _default)
        monkeypatch.setattr(gemini_provider, "create_openai_compatible_chat_model", model_factory)

        GeminiProvider().create_chat_model("gemini-3.5-flash", max_tokens=100, timeout=20)

        assert observed["default_request"] is not None
        assert (
            20
            > float(observed["lock_timeout"])
            > float(observed["request_timeout"])
            > float(model_factory.call_args.kwargs["timeout"])
            > 0
        )

    def test_cached_credentials_reused_and_refresh_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """9. Cached ADC credentials are reused and refreshed only when invalid/expired."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token="initial-token", valid=False)
        auth_default_calls = 0

        def mock_default(*, scopes: list[str], request: object):
            nonlocal auth_default_calls
            auth_default_calls += 1
            return creds, "test-project-123"

        monkeypatch.setattr(google.auth, "default", mock_default)
        provider = GeminiProvider()

        # First call: calls default and refreshes
        res1 = provider.resolve_credentials()
        assert res1 == ("initial-token", get_base_url("test-project-123", "global"))
        assert auth_default_calls == 1
        assert creds.refresh_count == 1

        # Second call: credentials are now valid, no default or refresh call
        res2 = provider.resolve_credentials()
        assert res2 == res1
        assert auth_default_calls == 1
        assert creds.refresh_count == 1

    def test_reset_cached_credentials_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """10. Reset cached credentials clears internal state across tests."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token="token", valid=True)
        calls = 0

        def mock_default(*, scopes: list[str], request: object):
            nonlocal calls
            calls += 1
            return creds, "test-project-123"

        monkeypatch.setattr(google.auth, "default", mock_default)
        GeminiProvider().resolve_credentials()
        assert calls == 1

        _reset_cached_credentials()
        GeminiProvider().resolve_credentials()
        assert calls == 2

    def test_create_chat_model_prefixes_google(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """11. create_chat_model('gemini-3.5-flash') sends google/gemini-3.5-flash to OpenAI-compatible helper."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token="token", valid=True)
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        llm = provider.create_chat_model("gemini-3.5-flash", max_tokens=100)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "google/gemini-3.5-flash"
        # x-goog-user-project is omitted when quota_project_id is absent on credentials
        assert getattr(llm, "default_headers", {}).get("x-goog-user-project") is None

    def test_create_chat_model_includes_quota_project_when_explicit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit quota_project_id from ADC credentials is forwarded in x-goog-user-project."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token="token", valid=True, quota_project_id="my-quota-project")
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        llm = provider.create_chat_model("gemini-3.5-flash", max_tokens=100)
        assert isinstance(llm, ChatOpenAI)
        assert getattr(llm, "default_headers", {}).get("x-goog-user-project") == "my-quota-project"

    def test_create_chat_model_avoids_double_prefixing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """12. Already-prefixed google/gemini-3.5-flash is not double-prefixed."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        creds = MockCredentials(token="token", valid=True)
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        llm = provider.create_chat_model("google/gemini-3.5-flash", max_tokens=100)
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "google/gemini-3.5-flash"

    def test_bare_and_prefixed_models_resolve_registry_metadata(self) -> None:
        """13. Bare and prefixed model names both resolve registry metadata."""
        provider = GeminiProvider()
        assert provider.get_context_length("gemini-3.5-flash") == 1_000_000
        assert provider.get_context_length("google/gemini-3.5-flash") == 1_000_000
        assert provider.get_max_output_tokens("gemini-3.5-flash") == 64_000
        assert provider.get_max_output_tokens("google/gemini-3.5-flash") == 64_000
        assert provider.get_context_length("unknown-model") is None

    def test_default_model_and_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """14. Default model and SKILLSPECTOR_MODEL override work."""
        provider = GeminiProvider()
        assert provider.resolve_model() == "gemini-3.8-flash"

        monkeypatch.setenv("SKILLSPECTOR_MODEL", "gemini-3.7-flash")
        assert provider.resolve_model() == "gemini-3.7-flash"

        monkeypatch.setenv("SKILLSPECTOR_MODEL", "google/gemini-3.7-flash")
        assert provider.resolve_model() == "gemini-3.7-flash"

    def test_registry_contains_default_model(self) -> None:
        """15. Registry returns metadata for default model."""
        provider = GeminiProvider()
        assert provider.get_context_length(provider.DEFAULT_MODEL) == 1_000_000
        assert provider.get_max_output_tokens(provider.DEFAULT_MODEL) == 64_000

    def test_provider_selection_selects_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """16. SKILLSPECTOR_PROVIDER=gemini selects GeminiProvider."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "gemini")
        assert isinstance(get_metadata_provider(), GeminiProvider)

    def test_provider_name_telemetry(self) -> None:
        """17. provider_name(GeminiProvider()) == 'gemini'."""
        assert provider_name(GeminiProvider()) == "gemini"

    def test_seed_and_reasoning_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """18. SKILLSPECTOR_SEED and SKILLSPECTOR_REASONING_EFFORT are forwarded."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-123")
        monkeypatch.setenv("SKILLSPECTOR_SEED", "42")
        monkeypatch.setenv("SKILLSPECTOR_REASONING_EFFORT", "medium")
        creds = MockCredentials(token="token", valid=True)
        monkeypatch.setattr(google.auth, "default", lambda **_: (creds, "test-project-123"))

        provider = GeminiProvider()
        llm = provider.create_chat_model("gemini-3.5-flash", max_tokens=100)
        assert isinstance(llm, ChatOpenAI)
        assert getattr(llm, "seed", None) == 42
        assert getattr(llm, "reasoning_effort", None) == "medium"

    def test_unconfigured_gemini_get_model_config_provider_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """19. get_model_config_provider returns GeminiProvider without raising when unconfigured."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "gemini")
        # GOOGLE_CLOUD_PROJECT is explicitly unset
        provider = get_model_config_provider()
        assert isinstance(provider, GeminiProvider)
        assert provider.resolve_model() == "gemini-3.8-flash"

    def test_unconfigured_gemini_build_model_config_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """20. build_model_config succeeds with unconfigured gemini provider."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "gemini")
        config = build_model_config()
        assert isinstance(config, dict)
        assert config.get("meta_analyzer") == "gemini-3.8-flash"

    def test_unconfigured_gemini_static_scan_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """21. Static scan (--no-llm) succeeds with unconfigured SKILLSPECTOR_PROVIDER=gemini."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "gemini")
        test_skill = tmp_path / "SKILL.md"
        test_skill.write_text("---\nname: safe-test\n---\n# Safe\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["scan", str(tmp_path), "--no-llm", "--format", "json"])
        assert result.exit_code == 0
        assert "findings" in result.output
