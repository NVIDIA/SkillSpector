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

"""Tests for the AWS Bedrock provider.

Bedrock authenticates via SigV4, not API keys, so ``resolve_credentials``
returns ``None`` and the provider implements ``ChatModelProvider.create_chat_model``
to construct ``ChatBedrockConverse`` directly.  These tests stub
``boto3.Session`` and ``ChatBedrockConverse`` so no AWS calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skillspector.providers import (
    get_metadata_provider,
    registry,
    resolve_provider_credentials,
)
from skillspector.providers.bedrock import (
    BEDROCK_DEFAULT_MODEL,
    BEDROCK_DEFAULT_REGION,
    BedrockProvider,
)

# A real application-inference-profile ARN shape for testing ARN-specific
# behavior.  Account ID and profile ID are placeholders — no live resource.
_TEST_ARN = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123def456"


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate provider-related env vars and the YAML cache for each test."""
    monkeypatch.delenv("SKILLSPECTOR_PROVIDER", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_MODEL", raising=False)
    monkeypatch.delenv("SKILLSPECTOR_MODEL_REGISTRY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    registry._load.cache_clear()
    yield
    registry._load.cache_clear()


class TestBedrockProviderCredentials:
    """Bedrock has no API key — resolve_credentials always returns None."""

    def test_resolve_credentials_returns_none(self) -> None:
        assert BedrockProvider().resolve_credentials() is None

    def test_resolve_credentials_returns_none_even_with_aws_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_PROFILE", "some-profile")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert BedrockProvider().resolve_credentials() is None


class TestBedrockProviderMetadata:
    """Token-budget metadata is read from the bundled model_registry.yaml."""

    def test_metadata_known_default_model(self) -> None:
        provider = BedrockProvider()
        assert provider.get_context_length(BEDROCK_DEFAULT_MODEL) == 1_000_000
        assert provider.get_max_output_tokens(BEDROCK_DEFAULT_MODEL) == 128_000

    def test_metadata_known_inference_profile_id(self) -> None:
        provider = BedrockProvider()
        model = "us.anthropic.claude-opus-4-6-20250915-v1:0"
        assert provider.get_context_length(model) == 1_000_000
        assert provider.get_max_output_tokens(model) == 128_000

    def test_metadata_unknown_model_returns_none(self) -> None:
        provider = BedrockProvider()
        assert provider.get_context_length("unknown.model") is None
        assert provider.get_max_output_tokens("unknown.model") is None


class TestBedrockProviderResolveModel:
    """resolve_model: SKILLSPECTOR_MODEL env > slot > DEFAULT_MODEL."""

    def test_default_model_is_public_cross_region_inference_profile(self) -> None:
        # The default must be a public Bedrock model ID, not a private ARN —
        # this is checked in the OSS PR review and is load-bearing.
        assert BEDROCK_DEFAULT_MODEL == "us.anthropic.claude-sonnet-4-6-20250915-v1:0"
        assert not BEDROCK_DEFAULT_MODEL.startswith("arn:")
        assert BedrockProvider().resolve_model() == BEDROCK_DEFAULT_MODEL

    def test_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "us.anthropic.claude-opus-4-6-20250915-v1:0")
        assert BedrockProvider().resolve_model() == "us.anthropic.claude-opus-4-6-20250915-v1:0"

    def test_env_applies_to_every_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL", "user/override")
        assert BedrockProvider().resolve_model("meta_analyzer") == "user/override"

    def test_unknown_slot_falls_back_to_default(self) -> None:
        assert BedrockProvider().resolve_model("mcp_least_privilege") == BEDROCK_DEFAULT_MODEL


class TestBedrockProviderCreateChatModel:
    """create_chat_model wires boto3 + ChatBedrockConverse with the right config."""

    @patch("skillspector.providers.bedrock.provider.ChatBedrockConverse")
    @patch("skillspector.providers.bedrock.provider.boto3.Session")
    def test_returns_none_when_no_aws_credentials(
        self, mock_session: MagicMock, mock_chat: MagicMock
    ) -> None:
        """No AWS credentials in any chain → return None so the orchestrator falls through."""
        mock_session.return_value.get_credentials.return_value = None

        result = BedrockProvider().create_chat_model(
            "us.anthropic.claude-sonnet-4-6-20250915-v1:0",
            max_tokens=1024,
            timeout=60,
        )

        assert result is None
        mock_chat.assert_not_called()

    @patch("skillspector.providers.bedrock.provider.ChatBedrockConverse")
    @patch("skillspector.providers.bedrock.provider.boto3.Session")
    def test_omits_profile_when_aws_profile_unset(
        self, mock_session: MagicMock, mock_chat: MagicMock
    ) -> None:
        """No AWS_PROFILE → boto3.Session called without profile_name.

        Defers to the standard boto3 credential chain (env vars, instance
        metadata, SSO).  This is the OSS-default behavior; hardcoding a
        named profile is a footgun for external users.
        """
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_session.return_value.client.return_value = MagicMock()

        BedrockProvider().create_chat_model(
            "us.anthropic.claude-sonnet-4-6-20250915-v1:0",
            max_tokens=1024,
            timeout=60,
        )

        session_kwargs = mock_session.call_args.kwargs
        assert "profile_name" not in session_kwargs
        assert session_kwargs["region_name"] == BEDROCK_DEFAULT_REGION

    @patch("skillspector.providers.bedrock.provider.ChatBedrockConverse")
    @patch("skillspector.providers.bedrock.provider.boto3.Session")
    def test_env_overrides_profile_and_region(
        self,
        mock_session: MagicMock,
        mock_chat: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AWS_PROFILE", "custom-profile")
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_session.return_value.client.return_value = MagicMock()

        BedrockProvider().create_chat_model(
            "us.anthropic.claude-sonnet-4-6-20250915-v1:0",
            max_tokens=1024,
            timeout=60,
        )

        mock_session.assert_called_once_with(
            profile_name="custom-profile",
            region_name="eu-central-1",
        )

    @patch("skillspector.providers.bedrock.provider.ChatBedrockConverse")
    @patch("skillspector.providers.bedrock.provider.boto3.Session")
    def test_timeout_applied_to_botocore_config(
        self, mock_session: MagicMock, mock_chat: MagicMock
    ) -> None:
        """The ``timeout`` argument flows through to the boto3 client.

        Without this, long Bedrock calls hang indefinitely instead of
        respecting the caller's timeout budget.
        """
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_session.return_value.client.return_value = MagicMock()

        BedrockProvider().create_chat_model(
            "us.anthropic.claude-sonnet-4-6-20250915-v1:0",
            max_tokens=1024,
            timeout=90,
        )

        client_call = mock_session.return_value.client.call_args
        config = client_call.kwargs["config"]
        # botocore.config.Config exposes timeouts as attributes.
        assert config.read_timeout == 90
        assert config.connect_timeout == 10

    @patch("skillspector.providers.bedrock.provider.ChatBedrockConverse")
    @patch("skillspector.providers.bedrock.provider.boto3.Session")
    def test_arn_pins_provider_to_anthropic(
        self, mock_session: MagicMock, mock_chat: MagicMock
    ) -> None:
        """ChatBedrockConverse requires explicit provider= when model is an ARN."""
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_session.return_value.client.return_value = MagicMock()

        BedrockProvider().create_chat_model(
            _TEST_ARN,
            max_tokens=2048,
            timeout=120,
        )

        kwargs = mock_chat.call_args.kwargs
        assert kwargs["model"] == _TEST_ARN
        assert kwargs["provider"] == "anthropic"
        assert kwargs["max_tokens"] == 2048
        assert kwargs["region_name"] == BEDROCK_DEFAULT_REGION

    @patch("skillspector.providers.bedrock.provider.ChatBedrockConverse")
    @patch("skillspector.providers.bedrock.provider.boto3.Session")
    def test_plain_model_id_does_not_pin_provider(
        self, mock_session: MagicMock, mock_chat: MagicMock
    ) -> None:
        """For non-ARN model IDs, provider is inferred from the prefix — omit it."""
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_session.return_value.client.return_value = MagicMock()

        BedrockProvider().create_chat_model(
            "us.anthropic.claude-sonnet-4-6-20250915-v1:0",
            max_tokens=1024,
            timeout=60,
        )

        assert "provider" not in mock_chat.call_args.kwargs


class TestBedrockProviderSelection:
    """SKILLSPECTOR_PROVIDER=bedrock activates BedrockProvider."""

    def test_select_bedrock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "bedrock")
        # Bedrock returns no OpenAI-style credentials.
        assert resolve_provider_credentials() is None
        assert isinstance(get_metadata_provider(), BedrockProvider)
