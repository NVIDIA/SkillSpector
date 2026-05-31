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

"""Structured-output helpers with provider-aware tool-schema sanitization.

``LLMAnalyzerBase`` drives every LLM analyzer through LangChain's
``with_structured_output``, which turns a Pydantic ``response_schema`` into an
OpenAI tool/function schema and asks the model to fill it in.

Pydantic ``Field(ge=, le=, gt=, lt=)`` constraints emit the JSON-schema
keywords ``minimum`` / ``maximum`` / ``exclusiveMinimum`` / ``exclusiveMaximum``
on the generated tool parameters.  The OpenAI and NVIDIA endpoints tolerate
these keywords, but Anthropic's tool-schema validator rejects them with an
HTTP 400::

    For 'integer' type, property 'minimum' is not supported.
    For 'number' type, property 'maximum' is not supported.

When the Anthropic provider is active that 400 makes every structured LLM call
fail, and each analyzer silently falls back to static-only analysis.

:func:`build_structured_llm` keeps the constraints on every provider that
tolerates them and, **only for the Anthropic provider**, strips the unsupported
numeric-bound keywords from the tool schema before it is sent — while still
validating responses against the original Pydantic model (constraints included).
"""

from __future__ import annotations

import copy
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool

from skillspector.providers import is_anthropic_provider

# JSON-schema numeric-bound keywords that Anthropic's tool-schema validator
# rejects.  Stripping them loses input-side validation at the API boundary, but
# the Pydantic model still enforces every constraint when the response is parsed.
UNSUPPORTED_NUMERIC_KEYWORDS: frozenset[str] = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
)


def strip_unsupported_numeric_bounds(schema: Any) -> Any:
    """Recursively remove unsupported numeric-bound keywords from a JSON schema.

    Returns a deep copy with every occurrence of the keywords in
    :data:`UNSUPPORTED_NUMERIC_KEYWORDS` removed, at any nesting depth (object
    properties, array ``items``, ``$defs``, ``anyOf`` branches, etc.).  The
    input is not mutated.
    """
    if isinstance(schema, dict):
        return {
            key: strip_unsupported_numeric_bounds(value)
            for key, value in schema.items()
            if key not in UNSUPPORTED_NUMERIC_KEYWORDS
        }
    if isinstance(schema, list):
        return [strip_unsupported_numeric_bounds(item) for item in schema]
    return schema


def sanitized_openai_tool(schema: type) -> dict[str, Any]:
    """Convert *schema* to an OpenAI tool dict with numeric bounds stripped."""
    tool = convert_to_openai_tool(schema)
    return cast(dict[str, Any], strip_unsupported_numeric_bounds(copy.deepcopy(tool)))


def build_structured_llm(llm: BaseChatModel, schema: type) -> Runnable:
    """Return a runnable that yields validated *schema* instances.

    For every provider except Anthropic this is exactly
    ``llm.with_structured_output(schema)`` — the numeric-bound constraints are
    preserved in the tool schema sent to the API.

    For the Anthropic provider the tool schema is sanitized of the numeric-bound
    keywords Anthropic rejects (see :func:`strip_unsupported_numeric_bounds`),
    the sanitized tool is force-selected via tool-calling, and the response is
    still parsed and validated against the original Pydantic *schema* (so the
    constraints are enforced on output even though they were dropped from the
    request).
    """
    if not is_anthropic_provider():
        return llm.with_structured_output(schema)

    tool = sanitized_openai_tool(schema)
    tool_name = tool["function"]["name"]
    bound = llm.bind_tools([tool], tool_choice=tool_name)
    parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
    return bound | parser
