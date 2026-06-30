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

"""Tests for SARIF rules[] array generation and empty finding filtering."""

from __future__ import annotations

from skillspector.models import Finding
from skillspector.nodes.report import _build_sarif


def _make_finding(rule_id: str = "PE3", message: str = "Credential Access", **kwargs) -> Finding:
    defaults = {
        "severity": "HIGH",
        "confidence": 0.9,
        "file": "tool.py",
        "start_line": 1,
        "end_line": 1,
        "remediation": "Remove credential access",
        "tags": ["privilege_escalation"],
        "context": "context",
        "matched_text": "match",
        "category": "privilege_escalation",
        "pattern": "PE3",
        "finding": "snippet",
        "explanation": "explain",
        "code_snippet": "code",
        "intent": None,
    }
    defaults.update(kwargs)
    return Finding(rule_id=rule_id, message=message, **defaults)


class TestEmptyFindingsFiltered:
    """Findings with missing rule_id or message are excluded from SARIF output."""

    def test_empty_rule_id_filtered(self) -> None:
        findings = [
            _make_finding(rule_id="", message="Some message"),
            _make_finding(rule_id="PE3", message="Credential Access"),
        ]
        sarif = _build_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        assert results[0]["ruleId"] == "PE3"

    def test_none_rule_id_filtered(self) -> None:
        findings = [
            _make_finding(rule_id=None, message="Some message"),
            _make_finding(rule_id="TM1", message="Tool Misuse"),
        ]
        sarif = _build_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        assert results[0]["ruleId"] == "TM1"

    def test_empty_message_filtered(self) -> None:
        findings = [
            _make_finding(rule_id="PE3", message=""),
            _make_finding(rule_id="MP1", message="Memory Poisoning"),
        ]
        sarif = _build_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        assert results[0]["ruleId"] == "MP1"

    def test_none_message_filtered(self) -> None:
        findings = [
            _make_finding(rule_id="PE3", message=None),
            _make_finding(rule_id="MP1", message="Memory Poisoning"),
        ]
        sarif = _build_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1

    def test_all_empty_produces_zero_results(self) -> None:
        findings = [
            _make_finding(rule_id="", message=""),
            _make_finding(rule_id=None, message=None),
        ]
        sarif = _build_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 0

    def test_valid_findings_unchanged(self) -> None:
        findings = [
            _make_finding(rule_id="PE3", message="Credential Access"),
            _make_finding(rule_id="TM1", message="Tool Misuse"),
        ]
        sarif = _build_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 2


class TestSarifRulesArray:
    """SARIF output includes tool.driver.rules[] with rule descriptors."""

    def test_rules_present_in_output(self) -> None:
        findings = [_make_finding(rule_id="PE3", message="Credential Access")]
        sarif = _build_sarif(findings)
        driver = sarif["runs"][0]["tool"]["driver"]
        assert "rules" in driver
        assert len(driver["rules"]) == 1

    def test_rule_has_id_and_description(self) -> None:
        findings = [_make_finding(rule_id="PE3", message="Credential Access")]
        sarif = _build_sarif(findings)
        rule = sarif["runs"][0]["tool"]["driver"]["rules"][0]
        assert rule["id"] == "PE3"
        assert rule["shortDescription"]["text"] == "Credential Access"

    def test_multiple_rules_deduplicated(self) -> None:
        findings = [
            _make_finding(rule_id="PE3", message="Credential Access"),
            _make_finding(rule_id="PE3", message="Credential Access", file="other.py"),
            _make_finding(rule_id="TM1", message="Tool Misuse"),
        ]
        sarif = _build_sarif(findings)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules) == 2
        rule_ids = {r["id"] for r in rules}
        assert rule_ids == {"PE3", "TM1"}

    def test_rules_sorted_by_id(self) -> None:
        findings = [
            _make_finding(rule_id="TM1", message="Tool Misuse"),
            _make_finding(rule_id="MP1", message="Memory Poisoning"),
            _make_finding(rule_id="PE3", message="Credential Access"),
        ]
        sarif = _build_sarif(findings)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        ids = [r["id"] for r in rules]
        assert ids == ["MP1", "PE3", "TM1"]

    def test_empty_findings_no_rules(self) -> None:
        findings = [_make_finding(rule_id="", message="")]
        sarif = _build_sarif(findings)
        driver = sarif["runs"][0]["tool"]["driver"]
        assert "rules" not in driver or driver.get("rules") is None

    def test_sarif_schema_present(self) -> None:
        findings = [_make_finding()]
        sarif = _build_sarif(findings)
        assert "$schema" in sarif
        assert sarif["version"] == "2.1.0"
