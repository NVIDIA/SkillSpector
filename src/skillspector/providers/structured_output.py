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

Gateway and Bedrock model IDs namespace or prefix the model name
(``azure/anthropic/claude-opus-5-5``, ``us.anthropic.claude-opus-5-5``);
:func:`model_name` reads it back so every route applies the same rules.
"""

from __future__ import annotations

import re

from skillspector.providers import registry

# Bare model names documented to answer a forced tool call with HTTP 400.
FORCED_TOOL_CALL_REJECTED_MODELS = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
)

# Bare model names documented to reject sampling controls (``temperature``) with HTTP 400.
SAMPLING_REJECTED_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")

# A Claude model name ending an identifier, at its start or after a ``-``/``.``
# routing prefix (``bedrock-claude-...``, ``us.anthropic.claude-...``).
_CLAUDE_MODEL_NAME = re.compile(r"(?:^|[-.])(claude-[a-z0-9][a-z0-9.-]*)$")
_DOTTED_VERSION = re.compile(r"(?<=\d)\.(?=\d)")
# GPT-6.1 Sol, bare or after a routing prefix (``openai.gpt-6.1-sol``),
# optionally with a snapshot suffix (``gpt-6.1-sol-2026-09-15``).
_GPT_6_1_SOL = re.compile(r"(?:^|[-.])gpt-6\.1-sol(?:-.*)?$")


def model_name(model: str) -> str:
    """Return *model*'s last path segment, lowercased and cut at any ``@`` or ``:``.

    ``openai/gpt-6.1-sol:nitro`` and ``azure/anthropic/claude-opus-5-5@20260922``
    read as ``gpt-6.1-sol`` and ``claude-opus-5-5``; Bedrock's ``:0`` version
    suffix and ARN prefixes drop the same way.
    """
    return re.split(r"[@:]", model.rpartition("/")[2].lower(), maxsplit=1)[0]


def is_gpt_6_1_sol(model: str) -> bool:
    """Return ``True`` when *model* names GPT-6.1 Sol, bare or behind a gateway ID."""
    return _GPT_6_1_SOL.search(model_name(model)) is not None


def _names_model(model: str, names: tuple[str, ...]) -> bool:
    name = claude_model_name(model)
    return name is not None and any(name == bare or name.startswith(bare + "-") for bare in names)


def rejects_forced_tool_call(model: str) -> bool:
    """Return ``True`` when *model* names a Claude model that rejects forced tool calls."""
    return _names_model(model, FORCED_TOOL_CALL_REJECTED_MODELS)


def rejects_sampling_controls(model: str, registry_path: str | None = None) -> bool:
    """Return ``True`` when *model* rejects ``temperature``.

    ``sampling: rejected`` in the registry at *registry_path* declares it for
    IDs that hide the model name (a Bedrock application-inference-profile ARN);
    otherwise the Claude model name read from *model* decides.
    """
    if (
        registry_path is not None
        and registry.lookup_setting(registry_path, model, "sampling") == "rejected"
    ):
        return True
    return _names_model(model, SAMPLING_REJECTED_MODELS)


def forced_tool_choice_supported(model: str, registry_path: str) -> bool:
    """``False`` when *model* answers a forced tool call with HTTP 400.

    A ``tool_choice`` entry in the registry at *registry_path* wins (``auto``
    means the model must not be forced); otherwise the Claude model name read
    from *model* decides.
    """
    declared = registry.lookup_setting(registry_path, model, "tool_choice")
    if declared:
        return declared != "auto"
    return not rejects_forced_tool_call(model)


def claude_model_name(model: str) -> str | None:
    """Return the bare Claude model name carried by *model*, or ``None``.

    Reads :func:`model_name` (the last path segment, case-insensitively and up
    to any ``@`` or ``:`` suffix such as ``@20260922``, ``:latest`` or
    Bedrock's ``:0``), so gateway IDs (``azure/anthropic/claude-opus-5-5``), routing prefixes
    (``bedrock-claude-opus-5-5``, ``us.anthropic.claude-opus-5-5``), Bedrock
    foundation-model and inference-profile ARNs, and dotted versions
    (``claude-opus-5.5``) all resolve to ``claude-opus-5-5``.  Returns ``None``
    for identifiers that do not name the model, such as Bedrock
    application-inference-profile ARNs; declare those in the registry.
    """
    match = _CLAUDE_MODEL_NAME.search(model_name(model))
    return _DOTTED_VERSION.sub("-", match.group(1)) if match else None
