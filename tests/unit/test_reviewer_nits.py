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

"""Tests for non-blocking reviewer nits from PRs #178, #179, #157."""

from __future__ import annotations

import logging

import pytest

from skillspector.providers.base import ModelMetadataProvider
from skillspector.providers.chat_models import validate_base_url


class TestModelMetadataProtocolAttributes:
    """ModelMetadataProvider Protocol declares DEFAULT_MODEL and SLOT_DEFAULTS."""

    def test_protocol_declares_default_model(self) -> None:
        annotations = getattr(ModelMetadataProvider, "__annotations__", {})
        assert "DEFAULT_MODEL" in annotations

    def test_protocol_declares_slot_defaults(self) -> None:
        annotations = getattr(ModelMetadataProvider, "__annotations__", {})
        assert "SLOT_DEFAULTS" in annotations

    def test_existing_providers_satisfy_protocol(self) -> None:
        from skillspector.providers.anthropic import AnthropicProvider
        from skillspector.providers.nv_build import NvBuildProvider
        from skillspector.providers.openai import OpenAIProvider

        for provider_cls in (NvBuildProvider, OpenAIProvider, AnthropicProvider):
            assert hasattr(provider_cls, "DEFAULT_MODEL")
            assert isinstance(provider_cls.DEFAULT_MODEL, str)
            assert hasattr(provider_cls, "SLOT_DEFAULTS")
            assert isinstance(provider_cls.SLOT_DEFAULTS, dict)


class TestValidateBaseUrl:
    """validate_base_url warns on malformed URLs without raising."""

    def test_valid_https_url_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_base_url("https://api.openai.com/v1")
        assert len(caplog.records) == 0

    def test_valid_http_url_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_base_url("http://localhost:11434/v1")
        assert len(caplog.records) == 0

    def test_none_url_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_base_url(None)
        assert len(caplog.records) == 0

    def test_ftp_scheme_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_base_url("ftp://files.example.com/models")
        assert any("ftp" in r.message for r in caplog.records)

    def test_empty_scheme_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_base_url("just-a-hostname:8080/v1")
        assert any("(empty)" in r.message or "no host" in r.message for r in caplog.records)

    def test_no_host_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            validate_base_url("http://")
        assert any("no host" in r.message for r in caplog.records)

    def test_does_not_raise(self) -> None:
        validate_base_url("not-a-url-at-all")
        validate_base_url("")
        validate_base_url("ftp://bad")
