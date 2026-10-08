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

"""Structured-output strategy shared by the Claude-serving providers.

LangChain's ``with_structured_output`` defaults to ``function_calling``,
which forces a tool call.  Some Claude models reject a forced tool call
with HTTP 400 (``tool_choice: type "tool" and "any" are not supported for
this model``).  What replaces it depends on the platform:

- The direct-API providers request the native JSON-schema response format
  (``method="json_schema"``) through ``structured_output_method(model)``.
- Amazon Bedrock has no JSON-schema output for these models (the Converse
  ``outputConfig`` comes back as ``output_config.format: Extra inputs are
  not permitted``), so ``BedrockProvider`` leaves ``toolChoice`` at ``auto``
  and :func:`skillspector.llm_utils.bind_structured_output` asks for the
  tool call in the prompt and retries when the model answers in prose.
- OpenAI-compatible gateways serving Claude can turn a ``json_schema``
  response format into a forced tool call too, so the ``openai`` and
  ``openai_compatible`` providers bind the schema as a tool with
  ``tool_choice`` left at ``auto`` and ask for the call in the prompt, as on
  Bedrock.

Gateway model IDs namespace or prefix the Claude name
(``azure/anthropic/claude-opus-5-5``, ``aws/anthropic/bedrock-claude-opus-5-5``);
:func:`claude_model_name` reads it back so every route applies the same rules.
"""

from __future__ import annotations

import re

# Bare model names documented to answer a forced tool call with HTTP 400.
FORCED_TOOL_CALL_REJECTED_MODELS = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
)

# Bare model names documented to reject sampling controls (``temperature``) with HTTP 400.
SAMPLING_REJECTED_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")

_BEDROCK_VENDOR_PREFIX = "anthropic."

# A Claude model name ending an identifier, at its start or after a ``-``/``.``
# routing prefix (``bedrock-claude-...``, ``us.anthropic.claude-...``).
_CLAUDE_MODEL_NAME = re.compile(r"(?:^|[-.])(claude-[a-z0-9][a-z0-9.-]*)$")
_DOTTED_VERSION = re.compile(r"(?<=\d)\.(?=\d)")


def _names_model(model: str | None, names: tuple[str, ...]) -> bool:
    return model is not None and any(
        model == name or model.startswith(name + "-") for name in names
    )


def rejects_forced_tool_call(model: str | None) -> bool:
    """Return ``True`` when the bare *model* name (optionally version-suffixed) rejects forced tool calls."""
    return _names_model(model, FORCED_TOOL_CALL_REJECTED_MODELS)


def rejects_sampling_controls(model: str | None) -> bool:
    """Return ``True`` when the bare *model* name (optionally version-suffixed) rejects ``temperature``."""
    return _names_model(model, SAMPLING_REJECTED_MODELS)


def claude_model_name(model: str) -> str | None:
    """Return the bare Claude model name carried by *model*, or ``None``.

    Reads the identifier's last path segment, case-insensitively and up to any
    ``@`` or ``:`` suffix (``@20260922``, ``:latest``, Bedrock's ``:0``), so
    gateway IDs (``azure/anthropic/claude-opus-5-5``), routing prefixes
    (``bedrock-claude-opus-5-5``, ``us.anthropic.claude-opus-5-5``) and dotted
    versions (``claude-opus-5.5``) all resolve to ``claude-opus-5-5``.
    """
    name = re.split(r"[@:]", model.rpartition("/")[2].lower(), maxsplit=1)[0]
    match = _CLAUDE_MODEL_NAME.search(name)
    return _DOTTED_VERSION.sub("-", match.group(1)) if match else None


def claude_model_from_bedrock_id(model: str) -> str | None:
    """Return the bare Claude model name carried by a Bedrock *model* identifier.

    Handles plain model IDs (``anthropic.claude-fable-5-1``), geo and global
    inference-profile IDs (``us.``/``eu.``/``global.`` prefixes), and
    foundation-model / inference-profile ARNs whose last path segment is one
    of those.  Returns ``None`` for identifiers that do not name the model,
    such as application-inference-profile ARNs; declare those in the registry.
    """
    _, _, name = model.rpartition("/")
    if _BEDROCK_VENDOR_PREFIX not in name:
        return None
    return name.split(_BEDROCK_VENDOR_PREFIX, 1)[1] or None
