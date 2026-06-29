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

"""AWS Bedrock provider — Claude models via the Bedrock Runtime.

Bedrock authenticates with AWS SigV4, not API keys.  ``resolve_credentials``
returns ``None`` because the ``(api_key, base_url)`` shape doesn't fit;
``create_chat_model`` instead probes the boto3 credential chain and
constructs ``ChatBedrockConverse`` directly when AWS credentials resolve.

Environment variables:
    ``AWS_PROFILE``   — when set, used as the boto3 named profile.
                        When unset, the standard boto3 credential chain
                        (env vars, instance metadata, SSO, etc.) resolves.
    ``AWS_REGION``    — defaults to ``us-west-2``.
    ``SKILLSPECTOR_MODEL`` — overrides the default model (a Bedrock
    model ID, a cross-region inference-profile ID, or your own
    application-inference-profile ARN).
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.config import Config as BotocoreConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry

BEDROCK_DEFAULT_REGION = "us-west-2"
# Cross-region inference profile ID for Claude Sonnet 4.6. Public,
# available to any account with Anthropic-on-Bedrock model access.
# Users can override with SKILLSPECTOR_MODEL to point at a different
# model or their own application-inference-profile ARN.
BEDROCK_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6-20250915-v1:0"
# Connect timeout for the Bedrock Runtime client. The per-call
# ``timeout`` from ``create_chat_model`` is applied as the read timeout.
_BEDROCK_CONNECT_TIMEOUT = 10

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class BedrockProvider:
    """AWS Bedrock provider — SigV4 auth via boto3, bundled-YAML metadata."""

    DEFAULT_MODEL = BEDROCK_DEFAULT_MODEL
    SLOT_DEFAULTS: dict[str, str] = {}

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Bedrock uses SigV4, not ``(api_key, base_url)`` — always returns ``None``.

        ``providers.resolve_chat_model_credentials`` treats ``None`` as "this
        provider doesn't supply OpenAI-style credentials" and falls through
        to its OpenAI fallback (which is irrelevant for Bedrock — the
        chat-model construction path in ``create_chat_model`` is what
        matters).
        """
        return None

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Construct a ``ChatBedrockConverse`` bound to a Bedrock client.

        Returns ``None`` if no AWS credentials resolve in the boto3
        credential chain, so the orchestrator can fall through to its
        OpenAI fallback.  Otherwise returns a configured client.

        ``AWS_PROFILE`` selects a named profile when set; otherwise the
        standard boto3 credential chain (env vars, instance metadata,
        SSO, etc.) resolves.  ``AWS_REGION`` defaults to ``us-west-2``.

        ``timeout`` is applied as the boto3 client's read timeout via a
        ``botocore.config.Config`` so long Bedrock calls actually time
        out instead of hanging.

        ``ChatBedrockConverse`` requires an explicit ``provider``
        argument when the model is supplied as an ARN (inference
        profiles); we pin it to ``anthropic`` since this provider is
        Claude-only.  When a plain Bedrock model ID is supplied,
        ``provider`` is inferred from the prefix and we omit it.
        """
        profile = os.environ.get("AWS_PROFILE", "").strip() or None
        region = os.environ.get("AWS_REGION", "").strip() or BEDROCK_DEFAULT_REGION

        session_kwargs: dict[str, object] = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)

        # If nothing in the boto3 credential chain resolves, return None so
        # the orchestrator can fall through.
        if session.get_credentials() is None:
            return None

        client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=BotocoreConfig(
                read_timeout=timeout,
                connect_timeout=_BEDROCK_CONNECT_TIMEOUT,
            ),
        )

        kwargs: dict[str, object] = {
            "model": model,
            "client": client,
            "region_name": region,
            "max_tokens": max_tokens,
        }
        if model.startswith("arn:"):
            kwargs["provider"] = "anthropic"

        return ChatBedrockConverse(**kwargs)

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
