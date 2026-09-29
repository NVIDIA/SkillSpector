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
"""

from __future__ import annotations

# Bare model names documented to answer a forced tool call with HTTP 400.
FORCED_TOOL_CALL_REJECTED_MODELS = ("claude-fable-5-1", "claude-mythos-5-1")

_BEDROCK_VENDOR_PREFIX = "anthropic."


def rejects_forced_tool_call(model: str) -> bool:
    """Return ``True`` when the bare *model* name (optionally version-suffixed) rejects forced tool calls."""
    return any(
        model == name or model.startswith(name + "-") for name in FORCED_TOOL_CALL_REJECTED_MODELS
    )


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
