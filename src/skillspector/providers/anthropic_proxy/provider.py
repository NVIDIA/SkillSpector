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

"""Anthropic proxy provider — Claude models via Vertex-style raw-predict endpoints.

Supports corporate API gateways, GCP Vertex AI, and self-hosted proxies
that expose Anthropic models through a raw-predict interface rather than
the standard ``api.anthropic.com`` Messages API.

These endpoints differ from the standard Anthropic API in three ways:

1. **URL**: Full path includes model name + ``:streamRawPredict`` suffix
   rather than ``/v1/messages``.
2. **Auth**: ``Authorization: Bearer <token>`` instead of ``x-api-key``.
3. **Body**: Contains an ``anthropic_version`` field (no ``model`` field;
   model is encoded in the URL path).

This provider uses custom httpx transports to rewrite requests from the
Anthropic SDK into the proxy's expected format, preserving all LangChain
``ChatAnthropic`` features (tool calling, structured output, streaming).

Environment variables::

    ANTHROPIC_PROXY_ENDPOINT_URL   Full endpoint URL (including model path)
    ANTHROPIC_PROXY_API_KEY        Bearer token for the Authorization header
    ANTHROPIC_PROXY_API_VERSION    anthropic_version value (default: vertex-2023-10-16)
"""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from skillspector.providers import registry

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))

DEFAULT_API_VERSION = "vertex-2023-10-16"
_PROXY_STRIPPED_HEADERS = frozenset({"x-api-key", "anthropic-version", "host", "content-length"})


def _get_api_version() -> str:
    """Return the ``anthropic_version`` value from env or the default."""
    return os.environ.get("ANTHROPIC_PROXY_API_VERSION", "").strip() or DEFAULT_API_VERSION


def _rewrite_proxy_request(
    request: httpx.Request,
    *,
    endpoint_url: str,
    bearer_token: str,
) -> httpx.Request:
    """Build a proxy-compatible request from an Anthropic SDK request."""
    body = json.loads(request.content)
    body["anthropic_version"] = _get_api_version()
    body.pop("model", None)
    encoded_body = json.dumps(body).encode("utf-8")

    headers = httpx.Headers(
        {k: v for k, v in request.headers.items() if k.lower() not in _PROXY_STRIPPED_HEADERS}
    )
    headers["authorization"] = f"Bearer {bearer_token}"
    headers["content-length"] = str(len(encoded_body))

    return httpx.Request(
        method=request.method,
        url=endpoint_url,
        headers=headers,
        content=encoded_body,
    )


class _ProxyTransport(httpx.BaseTransport):
    """Sync httpx transport that rewrites Anthropic SDK requests for the proxy."""

    def __init__(
        self,
        endpoint_url: str,
        bearer_token: str,
        *,
        verify: bool | str = True,
    ):
        self._endpoint_url = endpoint_url
        self._bearer_token = bearer_token
        self._inner = httpx.HTTPTransport(verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        new_request = _rewrite_proxy_request(
            request,
            endpoint_url=self._endpoint_url,
            bearer_token=self._bearer_token,
        )
        return self._inner.handle_request(new_request)

    def close(self) -> None:
        self._inner.close()


class _ProxyAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that rewrites Anthropic SDK requests for the proxy."""

    def __init__(
        self,
        endpoint_url: str,
        bearer_token: str,
        *,
        verify: bool | str = True,
    ):
        self._endpoint_url = endpoint_url
        self._bearer_token = bearer_token
        self._inner = httpx.AsyncHTTPTransport(verify=verify)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        new_request = _rewrite_proxy_request(
            request,
            endpoint_url=self._endpoint_url,
            bearer_token=self._bearer_token,
        )
        return await self._inner.handle_async_request(new_request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _ssl_verify() -> bool | str:
    """Return the SSL verification setting from ``SKILLSPECTOR_SSL_VERIFY``."""
    val = os.environ.get("SKILLSPECTOR_SSL_VERIFY", "").strip().lower()
    if val == "false":
        return False
    if val and val != "true":
        return val
    return True


class _ChatAnthropicProxy(ChatAnthropic):
    """``ChatAnthropic`` subclass that routes requests through a custom proxy."""

    _proxy_endpoint_url: str = ""
    _proxy_bearer_token: str = ""
    _proxy_ssl_verify: bool | str = True

    def __init__(self, *, proxy_endpoint_url: str, proxy_bearer_token: str, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "_proxy_endpoint_url", proxy_endpoint_url)
        object.__setattr__(self, "_proxy_bearer_token", proxy_bearer_token)
        object.__setattr__(self, "_proxy_ssl_verify", _ssl_verify())

    @cached_property
    def _client(self) -> anthropic.Client:  # type: ignore[override]
        params = self._client_params
        transport = _ProxyTransport(
            self._proxy_endpoint_url,
            self._proxy_bearer_token,
            verify=self._proxy_ssl_verify,
        )
        http_client = httpx.Client(transport=transport)
        return anthropic.Client(
            api_key=params["api_key"],
            base_url=params.get("base_url"),
            max_retries=params.get("max_retries", 2),
            default_headers=params.get("default_headers"),
            timeout=params.get("timeout"),
            http_client=http_client,
        )

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:  # type: ignore[override]
        params = self._client_params
        transport = _ProxyAsyncTransport(
            self._proxy_endpoint_url,
            self._proxy_bearer_token,
            verify=self._proxy_ssl_verify,
        )
        http_client = httpx.AsyncClient(transport=transport)
        return anthropic.AsyncClient(
            api_key=params["api_key"],
            base_url=params.get("base_url"),
            max_retries=params.get("max_retries", 2),
            default_headers=params.get("default_headers"),
            timeout=params.get("timeout"),
            http_client=http_client,
        )


class AnthropicProxyProvider:
    """Anthropic proxy provider for Vertex-style raw-predict endpoints."""

    DEFAULT_MODEL = "claude-sonnet-4-6"
    SLOT_DEFAULTS: ClassVar[dict[str, str]] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Return ``(api_key, endpoint_url)`` from proxy env vars."""
        api_key = os.environ.get("ANTHROPIC_PROXY_API_KEY", "").strip()
        endpoint_url = os.environ.get("ANTHROPIC_PROXY_ENDPOINT_URL", "").strip()
        if not api_key or not endpoint_url:
            return None
        return api_key, endpoint_url

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create ``ChatAnthropic`` with custom transport for the proxy."""
        creds = self.resolve_credentials()
        if creds is None:
            return None

        bearer_token, endpoint_url = creds

        return _ChatAnthropicProxy(
            proxy_endpoint_url=endpoint_url,
            proxy_bearer_token=bearer_token,
            model_name=model,
            anthropic_api_key=SecretStr("anthropic-proxy-placeholder"),
            max_tokens=max_tokens,
            default_request_timeout=timeout,
            stop_sequences=None,
        )

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > DEFAULT_MODEL."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
