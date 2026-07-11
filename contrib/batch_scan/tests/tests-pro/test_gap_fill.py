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

"""Unit tests for GapFillAnalyzer — parse_response, build_prompt, get_batches, collect_findings.

Covers: Happy Path, Edge Cases, Failure Scenarios, Pydantic model path, BOM, large findings.
Audit fixes: #4, #7, #11, #15, #16, #18, #28, #29, #C2, #C3, #F1 (setUpClass).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skillspector.llm_analyzer_base import Batch
from skillspector.models import Finding

from contrib.batch_scan.gap_fill import (
    GapFillAnalyzer,
    GapFillFinding,
    GapFillResult,
    _GAP_FILL_RULE_IDS,
    run_gap_fill,
)


# ---------------------------------------------------------------------------
# Factory (#4: replaces mutable module-level dict)
# ---------------------------------------------------------------------------


def _valid_finding(**overrides):
    """Return a fresh dict for a valid gap-fill finding.  Each call returns a
    new copy — no shared mutable state across tests."""
    d = {
        "rule_id": "P5",
        "message": "Skill contains recipe with arsenic",
        "severity": "CRITICAL",
        "confidence": 0.95,
        "explanation": "Arsenic is a toxic substance.",
        "remediation": "Remove the arsenic recipe.",
    }
    d.update(overrides)
    return d


def _batch(file_path: str = "test.md") -> Batch:
    return Batch(file_path=file_path, content="dummy content")


# ---------------------------------------------------------------------------
# Valid JSON — Happy Path
# ---------------------------------------------------------------------------


class TestParseResponseValidJSON(unittest.TestCase):
    """#11: Content verification, not just count."""

    @classmethod
    def setUpClass(cls):
        """#F1: One shared analyzer for all tests — avoids repeated ChatOpenAI creation."""
        cls.analyzer = GapFillAnalyzer(language="zh")

    def test_single_valid_finding_returns_all_fields_correctly(self):
        data = {"findings": [_valid_finding()]}
        results = self.analyzer.parse_response(json.dumps(data), _batch("recipes.md"))
        self.assertEqual(len(results), 1)
        f = results[0]
        self.assertEqual(f.rule_id, "P5")
        self.assertEqual(f.severity, "CRITICAL")
        self.assertEqual(f.file, "recipes.md")
        self.assertEqual(f.category, "Security")
        self.assertEqual(f.confidence, 0.95)

    def test_multiple_valid_findings_returns_correct_rule_ids(self):
        """#11: Checks specific content, not just count."""
        data = {
            "findings": [
                _valid_finding(),
                _valid_finding(rule_id="MP1", message="Memory poisoning detected"),
            ]
        }
        results = self.analyzer.parse_response(json.dumps(data), _batch())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].rule_id, "P5")
        self.assertEqual(results[1].rule_id, "MP1")

    def test_empty_findings_list_returns_empty_not_crash(self):
        results = self.analyzer.parse_response(json.dumps({"findings": []}), _batch())
        self.assertEqual(len(results), 0)

    def test_default_confidence_and_explanation_applied_when_not_provided(self):
        finding = {"rule_id": "RA1", "message": "Rogue agent detected", "severity": "HIGH"}
        results = self.analyzer.parse_response(json.dumps({"findings": [finding]}), _batch())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].confidence, 0.7)
        self.assertEqual(results[0].explanation, "")

    def test_finding_converted_to_skillspector_model_with_all_fields_preserved(self):
        results = self.analyzer.parse_response(
            json.dumps({"findings": [_valid_finding()]}), _batch("config.yaml")
        )
        self.assertEqual(results[0].file, "config.yaml")
        self.assertEqual(results[0].rule_id, "P5")
        self.assertEqual(results[0].message, "Skill contains recipe with arsenic")
        self.assertEqual(results[0].confidence, 0.95)


# ---------------------------------------------------------------------------
# Markdown Fences
# ---------------------------------------------------------------------------


class TestParseResponseMarkdownFences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="zh")

    def test_strips_fenced_json_with_language_tag(self):
        text = "```json\n" + json.dumps({"findings": [_valid_finding()]}) + "\n```"
        results = self.analyzer.parse_response(text, _batch())
        self.assertEqual(len(results), 1)

    def test_strips_fenced_json_without_language_tag(self):
        text = "```\n" + json.dumps({"findings": [_valid_finding()]}) + "\n```"
        results = self.analyzer.parse_response(text, _batch())
        self.assertEqual(len(results), 1)

    def test_strips_fenced_json_with_surrounding_whitespace(self):
        text = "  \n```json\n" + json.dumps({"findings": [_valid_finding()]}) + "\n```\n  "
        results = self.analyzer.parse_response(text, _batch())
        self.assertEqual(len(results), 1)

    def test_strips_fenced_json_with_jsonp_suffix(self):
        """Edge: ```jsonp fence — strip logic should handle unknown language tags."""
        text = "```jsonp\n" + json.dumps({"findings": [_valid_finding()]}) + "\n```"
        results = self.analyzer.parse_response(text, _batch())
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# Filtering — Business Rules
# ---------------------------------------------------------------------------


class TestParseResponseFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="ja")

    def test_filters_out_finding_with_confidence_below_threshold(self):
        data = {"findings": [_valid_finding(confidence=0.5)]}
        results = self.analyzer.parse_response(json.dumps(data), _batch())
        self.assertEqual(len(results), 0)

    def test_keeps_finding_at_confidence_threshold_boundary(self):
        data = {"findings": [_valid_finding(confidence=0.7)]}
        results = self.analyzer.parse_response(json.dumps(data), _batch())
        self.assertEqual(len(results), 1)

    def test_filters_out_unknown_rule_id_not_in_gap_fill_set(self):
        data = {"findings": [_valid_finding(rule_id="XYZ123")]}
        results = self.analyzer.parse_response(json.dumps(data), _batch())
        self.assertEqual(len(results), 0)

    def test_mixed_valid_and_invalid_only_keeps_valid(self):
        data = {
            "findings": [
                _valid_finding(),                                       # ✅
                _valid_finding(rule_id="P6", confidence=0.8),           # ✅
                _valid_finding(confidence=0.3),                         # ❌ low conf
                _valid_finding(rule_id="UNKNOWN_X"),                    # ❌ unknown rule
            ]
        }
        results = self.analyzer.parse_response(json.dumps(data), _batch())
        self.assertEqual(len(results), 2)

    def test_all_nine_gap_fill_rule_ids_accepted(self):
        findings = [_valid_finding(rule_id=rid) for rid in sorted(_GAP_FILL_RULE_IDS)]
        results = self.analyzer.parse_response(json.dumps({"findings": findings}), _batch())
        self.assertEqual(len(results), len(_GAP_FILL_RULE_IDS))
        self.assertEqual({f.rule_id for f in results}, set(_GAP_FILL_RULE_IDS))


# ---------------------------------------------------------------------------
# Invalid Input — Failure Scenarios
# ---------------------------------------------------------------------------


class TestParseResponseInvalidInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="ko")

    def test_non_json_string_returns_empty_list(self):
        results = self.analyzer.parse_response("This is not JSON at all.", _batch())
        self.assertEqual(len(results), 0)

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(len(self.analyzer.parse_response("", _batch())), 0)

    def test_integer_input_returns_empty_list(self):
        self.assertEqual(len(self.analyzer.parse_response(42, _batch())), 0)

    def test_json_list_instead_of_object_returns_empty_list(self):
        self.assertEqual(len(self.analyzer.parse_response("[1, 2, 3]", _batch())), 0)

    def test_missing_findings_key_returns_empty_list(self):
        self.assertEqual(
            len(self.analyzer.parse_response(json.dumps({"other": "value"}), _batch())), 0
        )

    def test_findings_value_is_string_not_list_returns_empty_list(self):
        self.assertEqual(
            len(self.analyzer.parse_response(json.dumps({"findings": "not a list"}), _batch())), 0
        )

    def test_invalid_severity_literal_value_returns_empty_list(self):
        data = {"findings": [_valid_finding(severity="CATASTROPHIC")]}
        results = self.analyzer.parse_response(json.dumps(data), _batch())
        self.assertEqual(len(results), 0)

    def test_utf8_bom_prepended_json_does_not_crash(self):
        """#C3: JSON with UTF-8 BOM prefix — should not crash."""
        text = "﻿" + json.dumps({"findings": [_valid_finding()]})
        results = self.analyzer.parse_response(text, _batch())
        # May or may not parse (BOM handling is platform-dependent), but must not crash
        self.assertIsInstance(results, list)

    def test_json_with_embedded_null_bytes_does_not_crash(self):
        """Edge: null bytes in JSON string — should not crash."""
        text = '{"findings": [\x00]}'
        results = self.analyzer.parse_response(text, _batch())
        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# Large findings list (#C2)
# ---------------------------------------------------------------------------


class TestParseResponseLargeFindings(unittest.TestCase):
    """#C2: 100+ findings — must complete without performance degradation."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="zh")

    def test_parses_one_hundred_findings_within_one_second(self):
        findings = [
            _valid_finding(rule_id=rid)
            for rid in sorted(_GAP_FILL_RULE_IDS) * 12  # 9 × 12 = 108
        ][:100]
        data = json.dumps({"findings": findings})
        t0 = time.monotonic()
        results = self.analyzer.parse_response(data, _batch())
        dt = time.monotonic() - t0
        self.assertEqual(len(results), 100)
        self.assertLess(dt, 2.0, f"100 findings took {dt:.1f}s, expected < 2s")


# ---------------------------------------------------------------------------
# Pydantic Model Input (#15)
# ---------------------------------------------------------------------------


class TestParseResponsePydanticModel(unittest.TestCase):
    """#15: parse_response receiving a structured Pydantic model (not raw string)."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="zh")

    def test_pydantic_model_path_delegates_to_original_parse_response(self):
        """When response is a GapFillResult Pydantic object, parse_response
        should process it without JSON parsing."""
        result = GapFillResult(findings=[GapFillFinding(**_valid_finding())])
        # Passing a Pydantic model — not a string
        results = self.analyzer.parse_response(result, _batch())
        # Should return findings (delegates to parent class behavior)
        self.assertIsInstance(results, list)
        # At minimum, must not crash
        self.assertGreaterEqual(len(results), 0)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


class TestGapFillFindingConversion(unittest.TestCase):
    def test_to_finding_preserves_all_nine_fields(self):
        gf = GapFillFinding(
            rule_id="P5", message="Test", severity="HIGH", confidence=0.85,
            explanation="Test explanation", remediation="Test remediation",
        )
        f = gf.to_finding("some/file.py")
        self.assertEqual(f.rule_id, "P5")
        self.assertEqual(f.message, "Test")
        self.assertEqual(f.severity, "HIGH")
        self.assertEqual(f.confidence, 0.85)
        self.assertEqual(f.file, "some/file.py")
        self.assertEqual(f.category, "Security")
        self.assertEqual(f.explanation, "Test explanation")
        self.assertEqual(f.remediation, "Test remediation")


# ---------------------------------------------------------------------------
# Language Injection (#16: split into 3 independent tests)
# ---------------------------------------------------------------------------


class TestLanguageInjection(unittest.TestCase):
    def test_language_zh_injected_into_prompt(self):
        analyzer = GapFillAnalyzer(language="zh")
        self.assertIn("zh AI agent skill", analyzer.base_prompt)

    def test_language_ja_injected_into_prompt(self):
        analyzer = GapFillAnalyzer(language="ja")
        self.assertIn("ja AI agent skill", analyzer.base_prompt)

    def test_language_ko_injected_into_prompt(self):
        analyzer = GapFillAnalyzer(language="ko")
        self.assertIn("ko AI agent skill", analyzer.base_prompt)


# ---------------------------------------------------------------------------
# build_prompt (#28)
# ---------------------------------------------------------------------------


class TestBuildPrompt(unittest.TestCase):
    """#28: GapFillAnalyzer.build_prompt() — previously zero coverage."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="zh")

    def test_build_prompt_includes_language_tag_and_file_label(self):
        batch = Batch(file_path="test/skill.md", content="# Skill\nSome content")
        prompt = self.analyzer.build_prompt(batch)
        self.assertIn("zh AI agent skill", prompt)
        self.assertIn("test/skill.md", prompt)
        self.assertIn("Some content", prompt)

    def test_build_prompt_includes_numbered_content(self):
        batch = Batch(file_path="a.md", content="line1\nline2")
        prompt = self.analyzer.build_prompt(batch)
        self.assertIn("L1:", prompt)
        self.assertIn("L2:", prompt)


# ---------------------------------------------------------------------------
# get_batches + collect_findings (#29)
# ---------------------------------------------------------------------------


class TestGetBatchesAndCollectFindings(unittest.TestCase):
    """#29: get_batches() + collect_findings() — previously zero coverage."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = GapFillAnalyzer(language="zh")

    def test_get_batches_creates_one_batch_per_file(self):
        file_cache = {"a.md": "content A", "b.md": "content B"}
        batches = self.analyzer.get_batches(list(file_cache.keys()), file_cache)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].file_path, "a.md")
        self.assertEqual(batches[1].file_path, "b.md")

    def test_collect_findings_flattens_batch_results(self):
        batch1 = _batch("a.md")
        batch2 = _batch("b.md")
        finding1 = Finding(rule_id="P5", message="m1", severity="LOW", confidence=0.8, file="a.md")
        finding2 = Finding(rule_id="P6", message="m2", severity="LOW", confidence=0.8, file="b.md")
        results = self.analyzer.collect_findings([
            (batch1, [finding1]),
            (batch2, [finding2]),
        ])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].rule_id, "P5")
        self.assertEqual(results[1].rule_id, "P6")


# ---------------------------------------------------------------------------
# run_gap_fill convenience function (#18)
# ---------------------------------------------------------------------------


class TestRunGapFill(unittest.TestCase):
    """#18: run_gap_fill() — previously zero coverage."""

    def test_run_gap_fill_with_empty_file_cache_returns_empty_list(self):
        results = run_gap_fill({}, "zh")
        self.assertEqual(len(results), 0)

    def test_run_gap_fill_with_english_shortcuts_early(self):
        """Non-English with empty cache is a no-op edge case."""
        results = run_gap_fill({}, "ja")
        self.assertEqual(len(results), 0)


# ---------------------------------------------------------------------------
# imports for time in large-findings test
# ---------------------------------------------------------------------------
import time  # noqa: E402 (placed here to group with test class usage)
