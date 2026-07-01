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

"""Unit tests for annotation.py — annotate_findings, is_language_compatible.

Covers: #27, #C5 (empty list), #C6 (missing fields).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skillspector.models import Finding

from contrib.multilingual.annotation import annotate_findings, is_language_compatible


def _make_finding(rule_id: str = "P1", file: str = "test.md") -> dict:
    """NB: annotate_findings reads the rule ID from the 'id' key, not 'rule_id'."""
    return {
        "id": rule_id,
        "message": "test message",
        "severity": "LOW",
        "confidence": 0.8,
        "file": file,
    }


class TestAnnotateFindings(unittest.TestCase):
    """#27: Coverage for the annotation layer Max praised."""

    def test_english_keyword_rule_marked_incompatible_for_chinese_skill(self):
        findings = [_make_finding(rule_id="P1"), _make_finding(rule_id="E1")]
        annotated = annotate_findings(findings, "zh")
        self.assertEqual(len(annotated), 2)
        for f in annotated:
            self.assertFalse(
                f.get("language_compatible", True),
                f"Rule {f.get('id', '?')} should be incompatible with zh",
            )

    def test_llm_rule_marked_compatible_for_chinese_skill(self):
        findings = [_make_finding(rule_id="SSD1"), _make_finding(rule_id="SDI1")]
        annotated = annotate_findings(findings, "zh")
        self.assertEqual(len(annotated), 2)
        for f in annotated:
            self.assertTrue(
                f.get("language_compatible", False),
                f"LLM rule {f.get('id', '?')} should be compatible with any language",
            )

    def test_code_rule_marked_compatible_for_chinese_skill(self):
        findings = [_make_finding(rule_id="AST1"), _make_finding(rule_id="TT1")]
        annotated = annotate_findings(findings, "ja")
        self.assertEqual(len(annotated), 2)
        for f in annotated:
            self.assertTrue(f.get("language_compatible", False))

    def test_all_rules_compatible_for_english_skill(self):
        findings = [_make_finding(rule_id="P1"), _make_finding(rule_id="SSD1")]
        annotated = annotate_findings(findings, "en")
        self.assertEqual(len(annotated), 2)
        for f in annotated:
            self.assertTrue(
                f.get("language_compatible", False),
                f"All rules should be compatible with en, but {f.get('id', '?')} is not",
            )

    def test_empty_findings_list_returns_empty(self):
        """#C5: Empty list edge case."""
        result = annotate_findings([], "zh")
        self.assertEqual(len(result), 0)

    def test_mixed_rules_partial_compatibility(self):
        """Mix of English-keyword and LLM rules."""
        findings = [
            _make_finding(rule_id="P1"),     # English keyword — incompatible with zh
            _make_finding(rule_id="SSD1"),   # LLM — compatible
            _make_finding(rule_id="E2"),     # English keyword — incompatible
            _make_finding(rule_id="AST1"),   # Code — compatible
        ]
        annotated = annotate_findings(findings, "zh")
        compatible = [f for f in annotated if f["language_compatible"]]
        incompatible = [f for f in annotated if not f["language_compatible"]]
        self.assertEqual(len(compatible), 2)
        self.assertEqual(len(incompatible), 2)

    def test_missing_rule_id_field_does_not_crash(self):
        """#C6: Finding with missing rule_id — must not crash."""
        findings = [{"message": "test", "severity": "LOW", "file": "x.md"}]
        annotated = annotate_findings(findings, "zh")
        self.assertEqual(len(annotated), 1)
        self.assertIn("language_compatible", annotated[0])

    def test_is_language_compatible_returns_true_for_english(self):
        self.assertTrue(is_language_compatible("P1", "en"))
        self.assertTrue(is_language_compatible("SSD1", "en"))

    def test_is_language_compatible_returns_false_for_english_keyword_rules_in_chinese(self):
        self.assertFalse(is_language_compatible("P1", "zh"))
        self.assertFalse(is_language_compatible("E1", "zh"))

    def test_is_language_compatible_returns_true_for_llm_rules_in_chinese(self):
        self.assertTrue(is_language_compatible("SSD1", "zh"))
        self.assertTrue(is_language_compatible("SDI1", "zh"))


if __name__ == "__main__":
    unittest.main()
