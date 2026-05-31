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

"""Tests for provider-aware structured-output tool-schema sanitization.

Covers the Anthropic 400 regression: Pydantic numeric-bound constraints
(``Field(ge=, le=)``) emit ``minimum`` / ``maximum`` JSON-schema keywords that
Anthropic's tool-schema validator rejects, silently dropping every LLM
analyzer to static-only analysis.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from skillspector.llm_analyzer_base import LLMAnalysisResult
from skillspector.nodes.meta_analyzer import MetaAnalyzerResult
from skillspector.structured_output import (
    UNSUPPORTED_NUMERIC_KEYWORDS,
    build_structured_llm,
    sanitized_openai_tool,
    strip_unsupported_numeric_bounds,
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SKILLSPECTOR_PROVIDER", raising=False)
    yield


def _collect_keys(obj: object) -> set[str]:
    """Recursively collect every dict key in a nested structure."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            keys |= _collect_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _collect_keys(item)
    return keys


# ---------------------------------------------------------------------------
# Regression guard: the unsanitized schema really does emit the bad keywords
# ---------------------------------------------------------------------------


class TestRegressionReproduction:
    """Without sanitization the generated tool schema carries the keywords
    Anthropic rejects.  These assertions document the bug the fix targets."""

    def test_llm_analysis_result_emits_minimum_and_maximum(self) -> None:
        tool = convert_to_openai_tool(LLMAnalysisResult)
        keys = _collect_keys(tool)
        # start_line=Field(ge=1) -> minimum; confidence=Field(ge=0.0, le=1.0) -> minimum/maximum
        assert "minimum" in keys
        assert "maximum" in keys

    def test_meta_analyzer_result_emits_minimum_and_maximum(self) -> None:
        tool = convert_to_openai_tool(MetaAnalyzerResult)
        keys = _collect_keys(tool)
        assert "minimum" in keys
        assert "maximum" in keys


# ---------------------------------------------------------------------------
# strip_unsupported_numeric_bounds
# ---------------------------------------------------------------------------


class TestStripUnsupportedNumericBounds:
    def test_strips_top_level_keywords(self) -> None:
        out = strip_unsupported_numeric_bounds(
            {"type": "integer", "minimum": 1, "maximum": 10}
        )
        assert out == {"type": "integer"}

    def test_strips_exclusive_variants(self) -> None:
        out = strip_unsupported_numeric_bounds(
            {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
        )
        assert out == {"type": "number"}

    def test_strips_nested_in_properties_and_items(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_line": {"type": "integer", "minimum": 1},
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                    },
                }
            },
        }
        out = strip_unsupported_numeric_bounds(schema)
        assert _collect_keys(out).isdisjoint(UNSUPPORTED_NUMERIC_KEYWORDS)
        # Non-bound keys survive.
        item_props = out["properties"]["findings"]["items"]["properties"]
        assert item_props["start_line"] == {"type": "integer"}
        assert item_props["confidence"] == {"type": "number"}

    def test_preserves_unrelated_keys(self) -> None:
        schema = {
            "type": "string",
            "description": "keep me",
            "enum": ["A", "B"],
            "default": "A",
        }
        assert strip_unsupported_numeric_bounds(schema) == schema

    def test_does_not_mutate_input(self) -> None:
        schema = {"type": "integer", "minimum": 1}
        strip_unsupported_numeric_bounds(schema)
        assert schema == {"type": "integer", "minimum": 1}

    def test_handles_anyof_branches(self) -> None:
        schema = {
            "anyOf": [
                {"type": "integer", "minimum": 1},
                {"type": "null"},
            ]
        }
        out = strip_unsupported_numeric_bounds(schema)
        assert out == {"anyOf": [{"type": "integer"}, {"type": "null"}]}


# ---------------------------------------------------------------------------
# sanitized_openai_tool — real schemas, fully cleaned
# ---------------------------------------------------------------------------


class TestSanitizedOpenAITool:
    @pytest.mark.parametrize("schema", [LLMAnalysisResult, MetaAnalyzerResult])
    def test_no_unsupported_keywords_remain(self, schema: type) -> None:
        tool = sanitized_openai_tool(schema)
        assert _collect_keys(tool).isdisjoint(UNSUPPORTED_NUMERIC_KEYWORDS)

    def test_tool_structure_and_descriptions_preserved(self) -> None:
        tool = sanitized_openai_tool(LLMAnalysisResult)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "LLMAnalysisResult"
        # Field types and the enum survive the strip.
        item = tool["function"]["parameters"]["properties"]["findings"]["items"]
        props = item["properties"]
        assert props["start_line"]["type"] == "integer"
        assert props["confidence"]["type"] == "number"
        assert props["severity"]["enum"] == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ---------------------------------------------------------------------------
# build_structured_llm — provider routing
# ---------------------------------------------------------------------------


class TestBuildStructuredLLM:
    def test_non_anthropic_uses_with_structured_output(self) -> None:
        """Other providers keep the constraints via plain with_structured_output."""
        llm = MagicMock()
        sentinel = object()
        llm.with_structured_output.return_value = sentinel
        result = build_structured_llm(llm, LLMAnalysisResult)
        llm.with_structured_output.assert_called_once_with(LLMAnalysisResult)
        llm.bind_tools.assert_not_called()
        assert result is sentinel

    def test_anthropic_binds_sanitized_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        llm = MagicMock()
        build_structured_llm(llm, LLMAnalysisResult)

        llm.with_structured_output.assert_not_called()
        llm.bind_tools.assert_called_once()
        (tools,), kwargs = llm.bind_tools.call_args
        assert kwargs["tool_choice"] == "LLMAnalysisResult"
        assert _collect_keys(tools).isdisjoint(UNSUPPORTED_NUMERIC_KEYWORDS)

    def test_anthropic_parser_validates_against_original_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanitizing the request schema must not relax output validation —
        the original Pydantic constraints still apply when the response parses."""
        monkeypatch.setenv("SKILLSPECTOR_PROVIDER", "anthropic")
        # The parser the Anthropic path attaches is keyed to the original schema.
        parser = PydanticToolsParser(tools=[LLMAnalysisResult], first_tool_only=True)

        # A valid tool-call round-trips into a validated Pydantic instance.
        valid = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "LLMAnalysisResult",
                    "args": {
                        "findings": [
                            {
                                "rule_id": "SEC-1",
                                "message": "x",
                                "severity": "HIGH",
                                "start_line": 5,
                                "confidence": 0.9,
                            }
                        ]
                    },
                    "id": "call_1",
                }
            ],
        )
        parsed = parser.invoke(valid)
        assert isinstance(parsed, LLMAnalysisResult)
        assert parsed.findings[0].confidence == 0.9

        # An out-of-bound confidence is still rejected by the Pydantic model.
        invalid = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "LLMAnalysisResult",
                    "args": {
                        "findings": [
                            {
                                "rule_id": "SEC-1",
                                "message": "x",
                                "severity": "HIGH",
                                "start_line": 5,
                                "confidence": 1.5,
                            }
                        ]
                    },
                    "id": "call_2",
                }
            ],
        )
        with pytest.raises(ValidationError):
            parser.invoke(invalid)
